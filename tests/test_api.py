"""Testes da API (modelo aleatório pequeno, ver conftest)."""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps

from api.app import MAX_BYTES, app
from tests.conftest import CLASSES_TESTE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def png(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def folha(w=320, h=240, seed=0):
    """Imagem sintética assimétrica (para o teste de rotação não ser trivial)."""
    a = (np.random.default_rng(seed).random((h, w, 3)) * 60 + 90).astype(np.uint8)
    a[: h // 2, : w // 3] = (30, 140, 40)
    a[h // 2:, w // 2:] = (150, 110, 30)
    return Image.fromarray(a)


def enviar(client, dados, nome="x.png", **params):
    return client.post("/predict", files={"file": (nome, dados)}, params=params)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["classes"] == len(CLASSES_TESTE)


def test_pagina_inicial(client):
    r = client.get("/")
    assert r.status_code == 200 and "Diagnóstico" in r.text


def test_estrutura_e_ordem(client):
    d = enviar(client, png(folha()), top_k=3).json()
    ps = [t["probabilidade"] for t in d["top"]]
    assert len(ps) == 3 and ps == sorted(ps, reverse=True) and sum(ps) <= 1.0001
    assert d["predicao"] == d["top"][0] and isinstance(d["confiavel"], bool)
    assert Image.open(io.BytesIO(base64.b64decode(d["gradcam_png_base64"]))).size == (320, 320)


def test_sem_gradcam(client):
    assert "gradcam_png_base64" not in enviar(client, png(folha()), gradcam="false").json()


def test_arquivo_invalido(client):
    assert enviar(client, b"isto nao e uma imagem", "x.txt").status_code == 400


def test_arquivo_grande_demais(client):
    assert enviar(client, b"\0" * (MAX_BYTES + 1)).status_code == 413


def test_megapixels_demais(client):
    dados = png(Image.new("L", (7000, 7000)))           # 49 MP, PNG minúsculo: passaria no limite de bytes
    assert len(dados) < MAX_BYTES
    assert enviar(client, dados).status_code == 413


def test_orientacao_exif(client):
    base = folha()
    exif = Image.Exif()
    exif[0x0112] = 6                                    # "gire 90° para exibir"
    b = io.BytesIO()
    base.save(b, "JPEG", quality=95, exif=exif)
    com_exif = b.getvalue()
    endireitada = ImageOps.exif_transpose(Image.open(io.BytesIO(com_exif)))
    sem_exif = io.BytesIO()
    base.save(sem_exif, "JPEG", quality=95)

    def probs(dados, nome):
        return {t["classe"]: t["probabilidade"] for t in enviar(client, dados, nome, top_k=10, gradcam="false").json()["top"]}

    a, certa, errada = probs(com_exif, "a.jpg"), probs(png(endireitada), "b.png"), probs(sem_exif.getvalue(), "c.jpg")
    assert all(abs(a[k] - certa[k]) < 1e-3 for k in a)        # com EXIF == imagem já endireitada
    assert any(abs(a[k] - errada[k]) > 1e-4 for k in a)       # e diferente de ignorar o EXIF (o teste não é trivial)
