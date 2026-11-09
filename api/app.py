"""API de diagnóstico de doenças foliares.

Executar:  uvicorn api.app:app --port 8000
O modelo (model.pt + model_config.json) é lido de `models/` ou da pasta indicada em MODEL_DIR.
"""
import base64
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from folhas import modelo
from folhas.gradcam import GradCAM, sobrepor
from folhas.rotulos import n_culturas, traduzir

RAIZ = Path(__file__).resolve().parent.parent
CONFIANCA_MIN = 0.60               # abaixo disso a resposta vem marcada como pouco confiável
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000            # um JPEG pequeno pode descomprimir para centenas de megapixels: rejeita-se antes de decodificar


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    pasta = Path(os.environ.get("MODEL_DIR", RAIZ / "models"))
    cfg = json.load(open(pasta / "model_config.json"))
    m = modelo.criar(len(cfg["classes"]), pretreinado=False)
    m.load_state_dict(torch.load(pasta / "model.pt", map_location=modelo.DEV))
    m.eval()
    torch.set_num_threads(4)
    app.state.modelo, app.state.cam = m, GradCAM(m, m.stages[-1])
    app.state.classes, app.state.tamanho = cfg["classes"], cfg["tamanho"]
    app.state.n_culturas = n_culturas(cfg["classes"])
    yield


app = FastAPI(title="Diagnóstico de doenças foliares", lifespan=ciclo_de_vida)


def _abrir(dados: bytes, tamanho: int) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(dados))                     # só lê o cabeçalho
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, f"Imagem com mais de {MAX_PIXELS // 1_000_000} megapixels.")
        img.draft("RGB", (tamanho * 2, tamanho * 2))            # JPEG: decodifica já reduzido
        img.load()
        return ImageOps.exif_transpose(img)                     # fotos de celular trazem a orientação só no EXIF
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise HTTPException(400, "Arquivo não é uma imagem válida.")


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), gradcam: bool = True, top_k: int = 5):
    st = request.app.state
    dados = await file.read()
    if len(dados) > MAX_BYTES:
        raise HTTPException(413, "Imagem maior que 10 MB.")
    img = _abrir(dados, st.tamanho)
    arr = np.array(img.convert("RGB").resize((st.tamanho, st.tamanho), Image.BICUBIC))   # mesmo pré-processamento do treino
    x = modelo.preparar_avaliacao(torch.from_numpy(arr).unsqueeze(0), st.tamanho)
    with torch.enable_grad():
        mapa, probs = st.cam(x)
    ordem = np.argsort(probs)[::-1][:max(1, min(top_k, 10))]
    top = []
    for i in ordem:
        cultura, condicao, saudavel = traduzir(st.classes[i])
        top.append({"classe": st.classes[i], "cultura": cultura, "condicao": condicao, "saudavel": saudavel, "probabilidade": float(probs[i])})
    confiavel = top[0]["probabilidade"] >= CONFIANCA_MIN
    saida = {"predicao": top[0], "top": top, "confiavel": bool(confiavel),
             "aviso": None if confiavel else f"Confiança baixa: a imagem pode não ser uma folha das {st.n_culturas} culturas suportadas, "
                                             "ou ter fundo/iluminação muito diferentes dos dados de treino."}
    if gradcam:
        buf = io.BytesIO()
        Image.fromarray(sobrepor(arr, mapa)).resize((320, 320)).save(buf, "PNG")
        saida["gradcam_png_base64"] = base64.b64encode(buf.getvalue()).decode()
    return saida


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "classes": len(request.app.state.classes), "culturas": request.app.state.n_culturas}


app.mount("/static", StaticFiles(directory=RAIZ / "api" / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(RAIZ / "api" / "static" / "index.html")
