"""Modelo (ConvNeXt-Tiny), pré-processamento e predição."""
import numpy as np
import timm
import torch
import torch.nn.functional as F

from .catalogo import BACKBONE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEV == "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def criar(nc: int, pretreinado: bool = True):
    """ConvNeXt-Tiny com `nc` saídas (pesos ImageNet-22k → 1k quando `pretreinado`)."""
    m = timm.create_model(BACKBONE, pretrained=pretreinado, num_classes=nc, drop_path_rate=0.1)
    return m.to(DEV).to(memory_format=torch.channels_last)


def carregar_pesos(modelo, caminho):
    """Carrega pesos; se a cabeça tem mais classes que o checkpoint, copia as linhas treinadas
    e mantém as novas com a inicialização padrão."""
    sd = torch.load(caminho, map_location=DEV)
    own = modelo.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape:
            novo = own[k].clone()
            novo[: v.shape[0]] = v
            sd[k] = novo
    modelo.load_state_dict(sd)


def congelar(modelo, n: int):
    """Congela o stem e os `n` primeiros estágios (n=0: tudo treinável)."""
    for p in modelo.stem.parameters():
        p.requires_grad = n == 0
    for i, s in enumerate(modelo.stages):
        for p in s.parameters():
            p.requires_grad = i >= n


def para_float(xb):
    """uint8 NHWC (CPU) -> float NCHW em [0, 1] (dispositivo)."""
    return xb.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)


def preparar_avaliacao(xb, tamanho: int):
    x = para_float(xb)
    if x.shape[-1] != tamanho:
        x = F.interpolate(x, size=(tamanho, tamanho), mode="bilinear", antialias=True, align_corners=False)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def prever(modelo, X, tamanho: int, lote: int = 128) -> np.ndarray:
    """Probabilidades (softmax) para um array uint8 NHWC."""
    modelo.eval()
    saida = []
    for i in range(0, len(X), lote):
        xb = torch.from_numpy(np.array(X[i:i + lote]))
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            saida.append(torch.softmax(modelo(preparar_avaliacao(xb, tamanho)).float(), 1).cpu())
    return torch.cat(saida).numpy()
