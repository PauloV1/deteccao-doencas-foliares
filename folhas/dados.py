"""Download, cache em .npy e carga das bases: PlantVillage (treino/teste), PlantDoc e PlantWild (fotos de campo)."""
import hashlib
import io
import json
import os
import subprocess
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from . import vazamento
from .catalogo import CLASSES_PV, PD_MAP, classes_do_projeto, mapa_plantwild

# Fontes fixadas por tag/commit/revisão: a mesma configuração baixa sempre os mesmos arquivos.
URL_PV = "https://github.com/attaullah/downsampled-plant-disease-dataset/releases/download/v1/plant-full.zip"
URL_PD = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/5467f6012d78d1c446145d5f582da6096f852ae8.zip"
URL_PW = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/527a72eb8f00c95e41698bb09d982c7d4625dce3/plantwild.zip"
IMAGENS = {".jpg", ".jpeg", ".png"}


@dataclass
class Dados:
    classes_pv: list            # as 38 do PlantVillage (índices 0..37)
    classes: list               # 38 + classes novas
    saudavel: np.ndarray        # bool por classe
    # PlantVillage
    Xtr: np.ndarray; Ytr: np.ndarray; idx_tr: np.ndarray; idx_va: np.ndarray
    Xva: np.ndarray; Yva: np.ndarray; em_treino: np.ndarray; n_por_classe: np.ndarray; minoritarias: np.ndarray
    Xte: np.ndarray; Yte: np.ndarray
    # PlantDoc
    Xpd_tr: np.ndarray; Ypd_tr: np.ndarray; Xpd_te: np.ndarray; Ypd_te: np.ndarray
    # PlantWild (None se desligado)
    Xpw: np.ndarray | None; Ypw: np.ndarray | None
    pw_tr: np.ndarray | None; pw_va: np.ndarray | None; pw_te: np.ndarray | None
    Xpw_va: np.ndarray | None; Ypw_va: np.ndarray | None; Xpw_te: np.ndarray | None; Ypw_te: np.ndarray | None
    # máscaras de vazamento nos testes e fundos para a troca de fundo
    vaz_pv: np.ndarray; vaz_pd: np.ndarray; vaz_pw: np.ndarray
    fundos: torch.Tensor

    @property
    def n_pv(self) -> int:
        return len(self.classes_pv)

    @property
    def tem_pw(self) -> bool:
        return self.Xpw is not None


# ------------------------------------------------------------------ download
def _sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def _baixar_zip(url, destino: Path, trabalho: Path, nome: str, cfg):
    """Baixa, registra o SHA-256 em `saidas/fontes.json` e descompacta em `destino`."""
    arq_zip = trabalho / f"{nome}.zip"
    _sh(f"curl -L --retry 5 -s -o {arq_zip} {url}")
    h = hashlib.sha256()
    with open(arq_zip, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    cfg.saidas_dir.mkdir(parents=True, exist_ok=True)
    arq = cfg.saidas_dir / "fontes.json"
    fontes = json.load(open(arq)) if arq.exists() else {}
    fontes[nome] = {"url": url, "sha256": h.hexdigest(), "bytes": arq_zip.stat().st_size}
    json.dump(fontes, open(arq, "w"), indent=1)
    _sh(f"unzip -q {arq_zip} -d {destino} && rm {arq_zip}")


def baixar(cfg):
    """Devolve (pv, pd, pw): pastas das bases, baixando em `cfg.trabalho` o que não foi informado."""
    w = Path(cfg.trabalho)
    w.mkdir(parents=True, exist_ok=True)
    pv = Path(cfg.pv_dir) if cfg.pv_dir else w / "plantvillage"
    if not (pv / "train").exists():
        _baixar_zip(URL_PV, pv, w, "plantvillage", cfg)
    if cfg.pd_dir:
        pd = Path(cfg.pd_dir)
    else:
        if not list((w / "plantdoc").glob("*/train")):
            _baixar_zip(URL_PD, w / "plantdoc", w, "plantdoc", cfg)
        pd = next((w / "plantdoc").glob("*/train")).parent
    pw = None
    if cfg.usar_plantwild:
        pw = Path(cfg.pw_dir) if cfg.pw_dir else w / "plantwild" / "plantwild"
        if not (pw / "images").exists():
            _baixar_zip(URL_PW, w / "plantwild", w, "plantwild", cfg)
    return pv, pd, pw


# ------------------------------------------------------------------ cache
def _ler(p, cs):
    try:
        return np.asarray(Image.open(p).convert("RGB").resize((cs, cs), Image.BICUBIC), dtype=np.uint8)
    except Exception as e:
        print("falha ao ler", p, e)
        return np.zeros((cs, cs, 3), np.uint8)


def _ler_grande(p, cs):
    """Fotos da web/campo podem ter milhares de pixels: `draft` decodifica o JPEG já reduzido."""
    try:
        b = Path(p).read_bytes()
        im = Image.open(io.BytesIO(b))
        im.draft("RGB", (cs * 2, cs * 2))
        return np.asarray(im.convert("RGB").resize((cs, cs), Image.BICUBIC), dtype=np.uint8), hashlib.md5(b).hexdigest()
    except Exception as e:
        print("falha ao ler", p, e)
        return np.zeros((cs, cs, 3), np.uint8), "FAIL"


def md5(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _listar(raiz: Path, mapa):
    arquivos, rotulos = [], []
    for d in sorted(p for p in raiz.iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGENS:
                arquivos.append(f)
                rotulos.append(mapa(d.name))
    return arquivos, rotulos


def _cache(cache: Path, nome, arquivos, rotulos, cs):
    if (cache / f"{nome}_y.npy").exists():
        return
    X = np.lib.format.open_memmap(cache / f"{nome}_x.npy", mode="w+", dtype=np.uint8, shape=(len(arquivos), cs, cs, 3))
    with Pool(os.cpu_count()) as pool:
        for i, a in enumerate(pool.imap(partial(_ler, cs=cs), arquivos, chunksize=64)):
            X[i] = a
    X.flush()
    del X
    np.save(cache / f"{nome}_y.npy", np.array(rotulos, dtype=np.int16))   # salvo por último: marca o cache completo
    print("cache criado:", nome, len(arquivos))


def _cache_grande(cache: Path, nome, arquivos, rotulos, splits, cs):
    """Imagens + rótulo + split de origem (0 teste, 1 treino, 2 val) + md5 para deduplicar."""
    if (cache / f"{nome}_md5.npy").exists():
        return
    X = np.lib.format.open_memmap(cache / f"{nome}_x.npy", mode="w+", dtype=np.uint8, shape=(len(arquivos), cs, cs, 3))
    hashes = []
    with Pool(os.cpu_count()) as pool:
        for i, (a, h) in enumerate(pool.imap(partial(_ler_grande, cs=cs), arquivos, chunksize=16)):
            X[i] = a
            hashes.append(h)
    X.flush()
    del X
    np.save(cache / f"{nome}_y.npy", np.array(rotulos, dtype=np.int16))
    np.save(cache / f"{nome}_split.npy", np.array(splits, dtype=np.int8))
    np.save(cache / f"{nome}_md5.npy", np.array(hashes))               # salvo por último
    print("cache criado:", nome, len(arquivos))


def _coletar_pw(pw: Path, mapa, idx):
    nomes = {int(a): b for a, b in (l.split(" ", 1) for l in (pw / "classes.txt").read_text().splitlines() if l.strip())}
    ausentes = [k for k in mapa if k not in nomes.values()]
    if ausentes:
        raise ValueError(f"classes do mapeamento ausentes em classes.txt: {ausentes}")
    arquivos, rotulos, splits = [], [], []
    for linha in (pw / "trainval.txt").read_text().splitlines():     # <pasta/arquivo>=<id da classe>=<modo: 0 teste, 1 treino, 2 val>
        rel, cid, modo = linha.rsplit("=", 2)
        if nomes[int(cid)] in mapa:
            arquivos.append(pw / "images" / rel)
            rotulos.append(idx[mapa[nomes[int(cid)]]])
            splits.append(int(modo))
    return arquivos, rotulos, splits


def _preparar_cache(cfg, pv, pd, pw, classes_pv, classes, mapa_pw):
    cache = cfg.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    info = {"tamanho": cfg.tamanho, "classes": classes}
    arq = cache / "info.json"
    if arq.exists() and json.load(open(arq)) != info:       # tamanho ou lista de classes mudou: refaz
        for f in cache.iterdir():
            f.unlink()
    json.dump(info, open(arq, "w"))
    idx = {c: i for i, c in enumerate(classes)}
    for sp in ("train", "test"):
        _cache(cache, sp, *_listar(pv / sp, idx.__getitem__), cfg.tamanho)
        desconhecidas = {d.name for d in (pd / sp).iterdir() if d.is_dir()} - set(PD_MAP)
        if desconhecidas:
            raise ValueError(f"pastas do PlantDoc sem mapeamento: {desconhecidas}")
        _cache(cache, f"pd_{sp}", *_listar(pd / sp, lambda n: idx[PD_MAP[n]]), cfg.tamanho)
    # hashes do teste do PlantDoc: tiram do PlantWild qualquer cópia exata (as duas bases vêm de buscas na web)
    json.dump(sorted(md5(f) for f in _listar(pd / "test", lambda n: 0)[0]), open(cache / "pd_test_md5.json", "w"))
    if pw is not None:
        _cache_grande(cache, "pw", *_coletar_pw(pw, mapa_pw, idx), cfg.tamanho)


# ------------------------------------------------------------------ divisão e carga
def dividir(M, S, proibidos=None):
    """Índices de treino (sem duplicatas exatas), validação e teste (sem repetir imagens do treino).
    M: hashes; S: split de origem (0 teste, 1 treino, 2 val); `proibidos`: hashes a excluir."""
    ok = M != "FAIL"
    if proibidos is not None:
        ok &= ~np.isin(M, proibidos)
    cand = np.flatnonzero(ok & (S == 1))
    _, primeiro = np.unique(M[cand], return_index=True)
    tr = np.sort(cand[primeiro])
    repete = np.isin(M, M[tr])
    va = np.flatnonzero(ok & (S == 2) & ~repete)
    te = np.flatnonzero(ok & (S == 0) & ~repete)
    info = dict(falhas=int((M == "FAIL").sum()),
                proibidas=int((np.isin(M, proibidos) & (M != "FAIL")).sum()) if proibidos is not None else 0,
                dup_treino=len(cand) - len(tr), repetidas=int((S != 1)[ok].sum()) - len(va) - len(te))
    return tr, va, te, info


def _fundos(Xpd_tr, Xpw, pw_tr, n_pw=3000):
    """Recortes de fundo: só imagens de TREINO de campo (nunca de teste)."""
    partes = [Xpd_tr]
    if Xpw is not None:
        sel = np.sort(np.random.default_rng(0).choice(pw_tr, min(n_pw, len(pw_tr)), replace=False))
        partes.append(np.ascontiguousarray(Xpw[sel]))
    return torch.from_numpy(np.concatenate(partes))


def carregar(cfg) -> Dados:
    pv, pd, pw = baixar(cfg)
    classes_pv = sorted(p.name for p in (pv / "train").iterdir() if p.is_dir())
    if classes_pv != CLASSES_PV:
        raise ValueError(f"as pastas do PlantVillage não são as 38 classes esperadas; diferença: {set(classes_pv) ^ set(CLASSES_PV)}")
    mapa_pw = mapa_plantwild(cfg.usar_br, cfg.br_hortalicas) if cfg.usar_plantwild else {}
    classes = classes_do_projeto(classes_pv, mapa_pw)
    print(f"{len(classes_pv)} classes do PlantVillage + {len(classes) - len(classes_pv)} novas = {len(classes)}")
    _preparar_cache(cfg, pv, pd, pw, classes_pv, classes, mapa_pw)

    c = cfg.cache_dir
    Xtr, Ytr = np.load(c / "train_x.npy", mmap_mode="r"), np.load(c / "train_y.npy").astype(np.int64)
    Xte, Yte = np.load(c / "test_x.npy", mmap_mode="r"), np.load(c / "test_y.npy").astype(np.int64)
    Xpd_tr, Ypd_tr = np.load(c / "pd_train_x.npy"), np.load(c / "pd_train_y.npy").astype(np.int64)
    Xpd_te, Ypd_te = np.load(c / "pd_test_x.npy"), np.load(c / "pd_test_y.npy").astype(np.int64)
    idx_tr, idx_va = train_test_split(np.arange(len(Ytr)), test_size=0.04, stratify=Ytr, random_state=0)
    idx_va = np.sort(idx_va)
    em_treino = np.zeros(len(Ytr), bool)
    em_treino[idx_tr] = True
    n_por_classe = np.bincount(Ytr[em_treino], minlength=len(classes_pv))
    print(f"PlantVillage treino {len(idx_tr)} | val {len(idx_va)} | teste {len(Yte)} | PlantDoc treino {len(Ypd_tr)} / teste {len(Ypd_te)}")

    Xpw = Ypw = pw_tr = pw_va = pw_te = Xpw_va = Ypw_va = Xpw_te = Ypw_te = None
    if pw is not None:
        Xpw, Ypw = np.load(c / "pw_x.npy", mmap_mode="r"), np.load(c / "pw_y.npy").astype(np.int64)
        proibidos = np.array(json.load(open(c / "pd_test_md5.json")))
        pw_tr, pw_va, pw_te, info = dividir(np.load(c / "pw_md5.npy"), np.load(c / "pw_split.npy"), proibidos)
        Xpw_va, Ypw_va = np.ascontiguousarray(Xpw[pw_va]), Ypw[pw_va]
        Xpw_te, Ypw_te = np.ascontiguousarray(Xpw[pw_te]), Ypw[pw_te]
        print(f"PlantWild (classes mapeadas): {len(Ypw)} imagens | removidas: {info['falhas']} ilegíveis, {info['proibidas']} cópias exatas do teste do PlantDoc, "
              f"{info['dup_treino']} duplicatas no treino, {info['repetidas']} val/teste repetidos do treino")
        print(f"  -> treino {len(pw_tr)} | val {len(pw_va)} | teste {len(pw_te)} | {len(set(Ypw[pw_tr]))} classes")

    vaz_pv = vazamento.mascara_pv(Xtr, idx_tr, Xte, cfg.limiar_vazamento)
    vaz_pd, vaz_pw = vazamento.mascaras_campo(Xpd_tr, Xpd_te, Xpw, pw_tr, Xpw_te, cfg.limiar_vazamento)
    cfg.saidas_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.saidas_dir / "mascaras_vazamento.npz", pv=vaz_pv, pd=vaz_pd, pw=vaz_pw)
    print(f"imagens de teste com vizinho quase idêntico no treino (similaridade >= {cfg.limiar_vazamento}): PlantVillage {int(vaz_pv.sum())}/{len(vaz_pv)}, "
          f"PlantDoc {int(vaz_pd.sum())}/{len(vaz_pd)}" + (f", PlantWild {int(vaz_pw.sum())}/{len(vaz_pw)}" if pw is not None else "") + " (as métricas \"sem vazamento\" as excluem)")

    fundos = _fundos(Xpd_tr, Xpw, pw_tr)
    return Dados(classes_pv, classes, np.array(["healthy" in k for k in classes]), Xtr, Ytr, idx_tr, idx_va,
                 np.ascontiguousarray(Xtr[idx_va]), Ytr[idx_va], em_treino, n_por_classe, np.argsort(n_por_classe)[:10], Xte, Yte,
                 Xpd_tr, Ypd_tr, Xpd_te, Ypd_te, Xpw, Ypw, pw_tr, pw_va, pw_te, Xpw_va, Ypw_va, Xpw_te, Ypw_te,
                 vaz_pv, vaz_pd, vaz_pw, fundos)
