"""Aumento de dados em lote na GPU: troca de fundo, recorte/rotação/flip (uma transformação afim) e cor."""
import math

import torch
import torch.nn.functional as F

from .modelo import DEV, MEAN, STD, para_float

BG_DIST, BG_SAT = 0.12, 0.25   # "pixel de fundo": cor próxima ao fundo estimado E pouco saturado
AREA_MIN, AREA_MAX = 0.08, 0.85  # área plausível da folha; fora disso a segmentação é descartada


def mascara_fundo(x):
    """Fundo = pixels de cor de fundo CONECTADOS à borda (preenchimento por inundação), x: (B,3,H,W) em [0,1].
    O fundo é estimado interpolando as medianas das 4 bordas, o que acompanha gradientes de iluminação.
    Manchas claras dentro da folha não viram buraco."""
    B, _, H, W = x.shape
    topo, base = x[:, :, :4].flatten(2).median(2).values, x[:, :, -4:].flatten(2).median(2).values
    esq, dir_ = x[:, :, :, :4].flatten(2).median(2).values, x[:, :, :, -4:].flatten(2).median(2).values
    ty = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1)
    tx = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W)
    v = lambda t: t.view(B, 3, 1, 1)
    fundo = 0.5 * (v(topo) * (1 - ty) + v(base) * ty) + 0.5 * (v(esq) * (1 - tx) + v(dir_) * tx)
    mx, mn = x.max(1, keepdim=True).values, x.min(1, keepdim=True).values
    sat = (mx - mn) / (mx + 1e-3)
    parecido = (((x - fundo).norm(dim=1, keepdim=True) < BG_DIST) & (sat < BG_SAT)).float()
    semente = torch.zeros_like(parecido)
    semente[:, :, :2], semente[:, :, -2:], semente[..., :2], semente[..., -2:] = 1, 1, 1, 1
    s = semente * parecido
    for _ in range(H // 3):                      # kernel 5x5 avança 2 px por passo
        s = F.max_pool2d(s, 5, 1, 2) * parecido
    return s


def fundo_aleatorio(B, H, W, pool):
    """75% recortes (zoom/flip aleatórios) de fotos de campo; 25% manchas de cor suaves."""
    u = lambda a, b: torch.empty(B, device=DEV).uniform_(a, b)
    imgs = pool[torch.randint(len(pool), (B,))].to(DEV).permute(0, 3, 1, 2).float().div(255)
    sc = u(0.35, 1.0)
    fx = (torch.rand(B, device=DEV) < 0.5).float() * 2 - 1
    th = torch.zeros(B, 2, 3, device=DEV)
    th[:, 0, 0], th[:, 1, 1] = sc * fx, sc
    th[:, 0, 2], th[:, 1, 2] = u(-1, 1) * (1 - sc), u(-1, 1) * (1 - sc)
    real = F.grid_sample(imgs, F.affine_grid(th, (B, 3, H, W), align_corners=False), mode="bilinear", padding_mode="reflection", align_corners=False)
    sintetico = F.interpolate(torch.rand(B, 3, 3, 3, device=DEV), size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
    return torch.where((torch.rand(B, device=DEV) < 0.25).view(B, 1, 1, 1), sintetico, real)


def trocar_fundo(x, eh_pv, p, pool):
    """Troca o fundo das fotos do PlantVillage (`eh_pv`) com probabilidade `p`. Se a área da folha segmentada
    parecer implausível, mantém a imagem original."""
    sel = eh_pv & (torch.rand(x.shape[0], device=DEV) < p)
    if not bool(sel.any()):
        return x
    xs = x[sel]
    folha = 1 - F.max_pool2d(mascara_fundo(xs), 3, 1, 1)             # encolhe 1 px: tira o halo da borda
    area = folha.mean((1, 2, 3))
    alpha = F.avg_pool2d(folha, 3, 1, 1)                              # borda suavizada
    novo = alpha * xs + (1 - alpha) * fundo_aleatorio(xs.shape[0], x.shape[2], x.shape[3], pool)
    x = x.clone()
    x[sel] = torch.where(((area > AREA_MIN) & (area < AREA_MAX)).view(-1, 1, 1, 1), novo, xs)
    return x


def preparar_treino(xb, tamanho, fonte=None, fundo_p=0.0, pool=None):
    """Lote de treino: [troca de fundo] + recorte aleatório + rotação ±20° + flips + cor.
    `fonte` identifica a base de cada amostra (0 = PlantVillage); a troca de fundo só vale para ela."""
    x = para_float(xb)
    B = x.shape[0]
    if fundo_p > 0 and fonte is not None and pool is not None:
        x = trocar_fundo(x, fonte == 0, fundo_p, pool)
    u = lambda a, b: torch.empty(B, device=DEV).uniform_(a, b)
    area, razao = u(0.5, 1.0), torch.exp(u(math.log(0.8), math.log(1.25)))
    w, h = torch.sqrt(area * razao).clamp(max=1.0), torch.sqrt(area / razao).clamp(max=1.0)
    tx, ty = u(-1, 1) * (1 - w), u(-1, 1) * (1 - h)
    ang = u(-20, 20) * math.pi / 180
    fx = (torch.rand(B, device=DEV) < 0.5).float() * 2 - 1
    fy = (torch.rand(B, device=DEV) < 0.5).float() * 2 - 1
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=DEV)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = c * w * fx, -s * h * fy, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = s * w * fx, c * h * fy, ty
    grade = F.affine_grid(theta, (B, 3, tamanho, tamanho), align_corners=False)
    x = F.grid_sample(x, grade, mode="bilinear", padding_mode="reflection", align_corners=False)
    brilho, contraste, sat = (u(0.7, 1.3).view(B, 1, 1, 1) for _ in range(3))
    sat = 0.8 + (sat - 0.7) / 0.6 * 0.4                               # saturação em [0.8, 1.2]
    x = x * brilho
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * contraste + m
    g = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    x = ((x - g) * sat + g).clamp(0, 1)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
