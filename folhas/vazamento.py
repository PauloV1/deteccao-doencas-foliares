"""Vazamento treino -> teste: imagens de teste quase idênticas a alguma imagem de treino.

Duas evidências: similaridade de cosseno em features de uma ConvNeXt-Tiny só com pesos ImageNet (128 px) e,
no PlantVillage, duplicatas por pixel invariantes a flips/rotações de 90°. Uma imagem de teste conta como
vazada quando alguma das evidências atinge o limiar.
"""
import numpy as np
import timm
import torch
import torch.nn.functional as F

from .catalogo import BACKBONE
from .modelo import AMP, DEV, MEAN, STD, para_float


def _blocos(X, idx, lote):
    idx = np.arange(len(X)) if idx is None else np.asarray(idx)
    for i in range(0, len(idx), lote):
        yield X[np.sort(idx[i:i + lote])] if len(idx) < len(X) else X[i:i + lote]


def _extrator():
    return timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(DEV).eval()


@torch.no_grad()
def _features(m, X, idx=None, lote=128):
    saida = []
    for blk in _blocos(X, idx, lote):
        x = F.interpolate(para_float(torch.from_numpy(np.array(blk))), size=(128, 128), mode="bilinear", antialias=True, align_corners=False)
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            saida.append(F.normalize(m((x - MEAN) / STD).float(), dim=1).cpu())
    return torch.cat(saida)


def _sim_max(fq, fr, lote=2048):
    fr = fr.to(DEV)
    return torch.cat([(fq[i:i + lote].to(DEV) @ fr.T).max(1).values.cpu() for i in range(0, len(fq), lote)]).numpy()


def _pixels(X, idx=None, lote=1024):
    saida = []
    for blk in _blocos(X, idx, lote):
        x = torch.from_numpy(np.array(blk)).permute(0, 3, 1, 2).float().mean(1, keepdim=True)
        d = F.adaptive_avg_pool2d(x, 16).flatten(1)
        saida.append(F.normalize(d - d.mean(1, keepdim=True), dim=1))
    return torch.cat(saida)


def _dihedrais(t):
    g = t.view(-1, 16, 16)
    v = [g, g.flip(2)]
    for k in (1, 2, 3):
        v += [torch.rot90(g, k, (1, 2)), torch.rot90(g.flip(2), k, (1, 2))]
    return [a.reshape(len(t), -1) for a in v]


def _corr_pixels(tr, te):
    tr = tr.to(DEV)
    melhor = torch.full((len(te),), -1.0, device=DEV)
    for v in _dihedrais(te.to(DEV)):
        for i in range(0, len(tr), 8192):
            melhor = torch.maximum(melhor, (v @ tr[i:i + 8192].T).max(1).values)
    return melhor.cpu().numpy()


def mascara_pv(Xtr, idx_tr, Xte, limiar):
    """Teste do PlantVillage contra as imagens de treino realmente usadas (`idx_tr`)."""
    m = _extrator()
    sim = _sim_max(_features(m, Xte), _features(m, Xtr, idx_tr))
    pix = _corr_pixels(_pixels(Xtr, idx_tr), _pixels(Xte))
    del m
    return (sim >= limiar) | (pix >= limiar)


def mascaras_campo(Xpd_tr, Xpd_te, Xpw, pw_tr, Xpw_te, limiar):
    """Testes de campo (PlantDoc e PlantWild) contra o treino de campo (PlantDoc + PlantWild)."""
    m = _extrator()
    partes = [_features(m, Xpd_tr)]
    if Xpw is not None:
        partes.append(_features(m, Xpw, pw_tr))
    ref = torch.cat(partes)
    vaz_pd = _sim_max(_features(m, Xpd_te), ref) >= limiar
    vaz_pw = _sim_max(_features(m, Xpw_te), ref) >= limiar if Xpw is not None else np.zeros(0, bool)
    del m
    return vaz_pd, vaz_pw
