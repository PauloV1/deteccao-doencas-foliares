import json
import os
import sys
from pathlib import Path

import pytest
import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

CLASSES_TESTE = ["Apple___Apple_scab", "Apple___healthy", "Coffee___Leaf_rust", "Rice___Blast", "Tomato___healthy"]


@pytest.fixture(scope="session", autouse=True)
def modelo_de_teste(tmp_path_factory):
    """Modelo pequeno com pesos aleatórios: os testes não dependem de `models/model.pt`."""
    from folhas import modelo
    pasta = tmp_path_factory.mktemp("modelo")
    torch.save(modelo.criar(len(CLASSES_TESTE), pretreinado=False).state_dict(), pasta / "model.pt")
    json.dump({"tamanho": 64, "classes": CLASSES_TESTE}, open(pasta / "model_config.json", "w"))
    os.environ["MODEL_DIR"] = str(pasta)
    return pasta
