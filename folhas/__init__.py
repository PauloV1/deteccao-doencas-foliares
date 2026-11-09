"""Detecção de doenças em folhas: dados, treino, avaliação e inferência."""
import sys

import matplotlib

if "ipykernel" not in sys.modules:      # fora de notebook (testes, scripts) não há janela: gráficos só em arquivo
    matplotlib.use("Agg")
