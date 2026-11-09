"""Ponta a ponta em CPU com bases sintéticas minúsculas: sem rede, sem dados reais, sem GPU."""
import json
import re

import numpy as np
import pandas as pd
import pytest
import timm
import torch

from folhas import ambiente, dados, modelo, pipeline as p, vazamento
from folhas.catalogo import CLASSES_PV
from folhas.config import Config
from tests import sintetico


class _Extrator(torch.nn.Module):
    """Substitui a rede de features do vazamento: descritor de cores 4x4 centrado (imagens diferentes ficam pouco parecidas,
    cópias ficam quase idênticas). Uma rede aleatória daria similaridade alta para qualquer par."""

    def forward(self, x):
        f = torch.nn.functional.adaptive_avg_pool2d(x, 4).flatten(1)
        return f - f.mean(1, keepdim=True)


@pytest.fixture
def sem_rede(monkeypatch):
    """Sem downloads: pesos aleatórios no lugar dos pré-treinados e extrator simples para o vazamento."""
    original = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda *a, **k: original(*a, **{**k, "pretrained": False}))
    monkeypatch.setattr(vazamento, "_extrator", lambda: _Extrator().to(modelo.DEV).eval())


def _cfg(tmp_path, bases, **extra):
    return Config(tamanho=64, trabalho=str(tmp_path / "trabalho"), saidas=str(tmp_path / "saidas"),
                  pv_dir=str(bases["pv"]), pd_dir=str(bases["pd"]), pw_dir=str(bases["pw"]),
                  epocas=1, amostras=32, lote=32, todos_epocas=1, todos_amostras=32, br_epocas=1, br_amostras=32,
                  sementes=(1,), exigir_gpu=False, **extra)


def test_pipeline_completo(tmp_path, sem_rede, capsys):
    cfg = _cfg(tmp_path, sintetico.criar_bases(tmp_path / "bases"))
    ctx = p.preparar(cfg)
    d = ctx.d
    assert len(d.classes) == 53 and d.classes[:38] == CLASSES_PV and d.tem_pw
    for mascara in (d.vaz_pv, d.vaz_pd, d.vaz_pw):         # os vazamentos plantados são marcados, e não tudo é marcado
        assert 1 <= mascara.sum() < 0.5 * len(mascara)
    assert (cfg.saidas_dir / "mascaras_vazamento.npz").exists() and (cfg.saidas_dir / "ambiente.json").exists()

    p.previsualizar(ctx)
    p.etapas_plantvillage(ctx)
    p.etapas_campo(ctx)
    p.etapa_brasil(ctx)
    p.ablacao_troca_fundo(ctx)
    p.sementes(ctx)
    p.resultados(ctx)
    p.exportar(ctx)

    s = cfg.saidas_dir
    runs = ["baseline", "balanceado", "sem_troca", "campo_pd", "campo_todos", "campo_cont", "brasil", "baseline_s1", "balanceado_s1", "sem_troca_s1"]
    assert all((s / r / "done.json").exists() and (s / r / "best.pt").exists() for r in runs)
    resumo = pd.read_csv(s / "todos_os_modelos.csv", index_col="run")
    assert list(resumo.index) == ["baseline", "sem_troca", "balanceado", "campo_pd", "campo_todos", "campo_cont", "brasil"]
    for col in ("pv_acc", "pv_acc_clean", "pd_top1", "pd_ece", "pw_top1", "pw_macro_f1"):
        assert resumo[col].between(0, 1).all(), col
    brasil = json.load(open(s / "brasil" / "metrics.json"))
    assert 0 <= brasil["summary"]["new_top1"] <= 1 and 0 <= brasil["summary"]["pv_leak_to_new"] <= 1 and brasil["summary"]["new_n"] > 0
    assert len(brasil["new_class_recall"]) == 15
    cmp = pd.read_csv(s / "comparacao_modificacoes.csv")
    assert cmp["modificação"].str.startswith("1.").any() and cmp["modificação"].str.startswith("5a.").any()
    assert (cmp["IC95 inf"] <= cmp["Δ pp"]).all() and (cmp["Δ pp"] <= cmp["IC95 sup"]).all()
    assert set(pd.read_csv(s / "sementes.csv", index_col=0).index) == {"baseline", "balanceado", "sem_troca"}
    assert len(pd.read_csv(s / "tempos.csv")) == len(runs)

    cfg_modelo = json.load(open(s / "model_config.json"))
    assert len(cfg_modelo["classes"]) == 53 and cfg_modelo["tamanho"] == 64
    assert torch.load(s / "model.pt", map_location="cpu")["head.fc.weight"].shape[0] == 53

    capsys.readouterr()
    assert p.treinar is not None
    from folhas.treino import treinar
    treinar(cfg, d, "baseline", "natural", epocas=1, amostras=32, lr=cfg.lr)
    assert "já concluído" in capsys.readouterr().out         # run concluído não é treinado de novo


def test_dados_com_classes_inesperadas(tmp_path):
    bases = sintetico.criar_bases(tmp_path / "bases")
    import shutil
    shutil.rmtree(bases["pv"] / "train" / CLASSES_PV[0])
    with pytest.raises(ValueError, match="38 classes"):
        dados.carregar(Config(trabalho=str(tmp_path / "t"), saidas=str(tmp_path / "s"), pv_dir=str(bases["pv"]), pd_dir=str(bases["pd"]), pw_dir=str(bases["pw"])))


def test_fontes_fixadas_por_commit():
    assert re.search(r"/archive/[0-9a-f]{40}\.zip$", dados.URL_PD) and re.search(r"/resolve/[0-9a-f]{40}/", dados.URL_PW)


def test_ambiente_registra_versoes_e_tempos(tmp_path):
    cfg = Config(saidas=str(tmp_path))
    info = ambiente.registrar(cfg)
    assert info["pacotes"]["torch"] and (tmp_path / "ambiente.json").exists()
    (tmp_path / "x").mkdir()
    json.dump({"history": [{"sec": 60.0}, {"sec": 30.0}]}, open(tmp_path / "x" / "done.json", "w"))
    t = ambiente.tempos(cfg)
    assert t.loc[0, "minutos"] == pytest.approx(1.5) and t.loc[0, "epocas"] == 2
