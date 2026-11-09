"""Bases sintéticas minúsculas, com a mesma estrutura de pastas das reais (PlantVillage, PlantDoc, PlantWild)."""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from folhas.catalogo import CLASSES_PV, PD_MAP, PW_BR, PW_MAP


def _img(caminho: Path, cor: int, rng, tam=40, qualidade=90):
    """Tom da classe + padrão aleatório por imagem: imagens diferentes não são parecidas entre si."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    base = np.array([(cor * 53) % 200 + 30, (cor * 97) % 200 + 30, (cor * 29) % 200 + 30])
    padrao = Image.fromarray((rng.random((4, 4, 3)) * 255).astype(np.uint8)).resize((tam, tam), Image.BILINEAR)
    arr = np.clip(0.6 * np.asarray(padrao, float) + 0.4 * base + rng.normal(0, 6, (tam, tam, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(caminho, "JPEG", quality=qualidade)


def _copia_reencodada(origem: Path, destino: Path, qualidade=55):
    """Mesma imagem com outro JPEG: hash diferente, conteúdo praticamente idêntico (vazamento que o hash não pega)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    Image.open(origem).save(destino, "JPEG", quality=qualidade)


def criar_bases(raiz: Path, seed: int = 0) -> dict:
    """PlantVillage: 38 classes (26 treino, 3 teste). PlantDoc: pastas do mapeamento (8 treino, 3 teste).
    PlantWild: classes mapeadas + 2 sem mapeamento (12 por classe: 8 treino, 2 val, 2 teste), com uma cópia exata
    de uma imagem de teste do PlantDoc no treino. Devolve as pastas."""
    rng = np.random.default_rng(seed)
    pv, pd, pw = raiz / "pv", raiz / "pd", raiz / "pw"
    for i, c in enumerate(CLASSES_PV):
        for sp, n in (("train", 26), ("test", 3)):
            for k in range(n):
                _img(pv / sp / c / f"{k}.jpg", i, rng)
    for i, nome in enumerate(PD_MAP):
        for sp, n in (("train", 8), ("test", 3)):
            if sp == "test" and "spider mites" in nome:       # esta pasta só existe no treino, como na base real
                continue
            for k in range(n):
                _img(pd / sp / nome / f"{k}.jpg", i + 100, rng)
    nomes = list(PW_MAP) + list(PW_BR) + ["apple mosaic virus", "squash leaf"]
    (pw / "images").mkdir(parents=True, exist_ok=True)
    linhas = []
    for cid, nome in enumerate(nomes):
        for k in range(12):
            modo = 1 if k < 8 else (2 if k < 10 else 0)
            _img(pw / "images" / nome / f"{k}.jpg", cid + 200, rng)
            linhas.append(f"{nome}/{k}.jpg={cid}={modo}")
    (pw / "classes.txt").write_text("\n".join(f"{i} {n}" for i, n in enumerate(nomes)))
    primeiro = sorted((pd / "test").iterdir())[0]
    shutil.copy(sorted(primeiro.iterdir())[0], pw / "images" / nomes[0] / "copia_do_teste_do_plantdoc.jpg")
    linhas.append(f"{nomes[0]}/copia_do_teste_do_plantdoc.jpg=0=1")
    # vazamentos plantados que o pipeline precisa marcar: cópia re-codificada de uma imagem de treino dentro do teste
    for k in range(5):      # cinco origens: algumas podem cair na validação (4% do treino), que não conta como treino
        _copia_reencodada(pv / "train" / CLASSES_PV[0] / f"{k}.jpg", pv / "test" / CLASSES_PV[0] / f"vazada{k}.jpg")
    shutil.copy(sorted(primeiro.iterdir())[0], pd / "train" / primeiro.name / "vazada.jpg")           # PlantDoc: cópia exata do teste no treino
    _copia_reencodada(pw / "images" / nomes[1] / "0.jpg", pw / "images" / nomes[1] / "vazada.jpg")
    linhas.append(f"{nomes[1]}/vazada.jpg=1=0")
    (pw / "trainval.txt").write_text("\n".join(linhas))
    return {"pv": pv, "pd": pd, "pw": pw}
