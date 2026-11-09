"""Registro do ambiente de execução e dos tempos de treino, para a proveniência dos resultados."""
import json
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import torch

PACOTES = ("torch", "torchvision", "timm", "numpy", "scikit-learn", "pandas", "pillow", "matplotlib")


def _versao(nome):
    try:
        return version(nome)
    except PackageNotFoundError:
        return None


def registrar(cfg) -> dict:
    """Grava `saidas/ambiente.json`: versões das bibliotecas, GPU, Python e data de início."""
    info = {
        "inicio": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "sistema": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda, "pacotes": {p: _versao(p) for p in PACOTES},
    }
    cfg.saidas_dir.mkdir(parents=True, exist_ok=True)
    json.dump(info, open(cfg.saidas_dir / "ambiente.json", "w"), indent=1)
    return info


def tempos(cfg) -> pd.DataFrame:
    """Minutos de treino de cada run (soma das épocas), salvos em `saidas/tempos.csv`."""
    linhas = []
    for f in sorted(cfg.saidas_dir.glob("*/done.json")):
        h = json.load(open(f))["history"]
        linhas.append({"run": f.parent.name, "epocas": len(h), "minutos": sum(e["sec"] for e in h) / 60})
    df = pd.DataFrame(linhas)
    if len(df):
        df.to_csv(cfg.saidas_dir / "tempos.csv", index=False)
    return df
