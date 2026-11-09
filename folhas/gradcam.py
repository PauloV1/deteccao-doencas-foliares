"""Grad-CAM para a ConvNeXt-Tiny (camada: saída do último estágio)."""
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, modelo, camada):
        self.modelo, self.acts, self.grads = modelo, None, None
        camada.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        camada.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, alvo=None):
        """x: tensor (1,3,H,W) normalizado. Devolve (mapa [0,1] HxW, probabilidades)."""
        self.modelo.eval()
        x = x.clone().requires_grad_(True)               # garante grafo mesmo com o início congelado
        logits = self.modelo(x)
        t = int(logits.argmax(1)) if alvo is None else alvo
        self.modelo.zero_grad()
        logits[0, t].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=x.shape[2:], mode="bicubic", align_corners=False)[0, 0].clamp(min=0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.softmax(logits, 1)[0].detach().cpu().numpy()


def sobrepor(rgb_uint8, cam, alpha=0.45):
    """Mapa de calor sobre a imagem (uint8 HxWx3)."""
    if cam.shape != rgb_uint8.shape[:2]:
        cam = F.interpolate(torch.from_numpy(cam)[None, None], size=rgb_uint8.shape[:2], mode="bicubic", align_corners=False)[0, 0].clamp(0, 1).numpy()
    calor = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    return (alpha * calor + (1 - alpha) * rgb_uint8).astype(np.uint8)
