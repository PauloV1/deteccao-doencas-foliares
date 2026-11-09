"""Testes das funções do núcleo (CPU, sem rede e sem dados)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from folhas import aumento, avaliacao, catalogo, comparacao, dados, modelo, rotulos, treino, vazamento
from folhas.config import Config


# ---------------------------------------------------------------- configuração e catálogo
def test_config_sobrescrita_por_ambiente(monkeypatch):
    monkeypatch.setenv("FOLHAS_CONFIG", json.dumps({"epocas": 1, "sementes": [7], "tamanho": 64}))
    c = Config()
    assert (c.epocas, c.sementes, c.tamanho) == (1, (7,), 64)


def test_config_invalida():
    with pytest.raises(ValueError):
        Config(usar_br=True, usar_plantwild=False)
    with pytest.raises(ValueError):
        Config(campo_pai="outro")


def test_todas_as_classes_tem_traducao():
    todas = set(catalogo.PD_MAP.values()) | set(catalogo.PW_MAP.values()) | set(catalogo.PW_BR.values()) | set(catalogo.PW_HORTALICAS.values())
    for c in todas:
        cultura, condicao, saudavel = rotulos.traduzir(c)
        assert cultura and condicao and isinstance(saudavel, bool)


def test_classes_do_projeto_mantem_indices_do_plantvillage():
    pv = ["A___x", "B___y"]
    todas = catalogo.classes_do_projeto(pv, {"k": "Z___novo", "j": "B___y", "i": "C___novo"})
    assert todas[:2] == pv and todas[2:] == ["C___novo", "Z___novo"]


# ---------------------------------------------------------------- aumento de dados
def cena(h=64, w=64):
    """Fundo cinza uniforme com uma elipse verde saturada no centro."""
    y, x = np.mgrid[:h, :w]
    elipse = ((x - w / 2) / (w * 0.3)) ** 2 + ((y - h / 2) / (h * 0.3)) ** 2 < 1
    img = np.full((h, w, 3), 128, np.uint8)
    img[elipse] = (30, 160, 40)
    return img, elipse.mean()


def test_mascara_fundo_acha_a_folha():
    img, frac = cena()
    x = modelo.para_float(torch.from_numpy(img)[None])
    folha = 1 - aumento.mascara_fundo(x)
    assert abs(float(folha.mean()) - frac) < 0.05
    assert float(folha[0, 0, 32, 32]) == 1.0 and float(folha[0, 0, 0, 0]) == 0.0


def test_trocar_fundo_respeita_origem_e_descarta_segmentacao_falha():
    img, _ = cena()
    x = modelo.para_float(torch.from_numpy(img)[None].repeat(4, 1, 1, 1))
    pool = torch.randint(0, 255, (5, 64, 64, 3), dtype=torch.uint8)
    so_campo = aumento.trocar_fundo(x, torch.zeros(4, dtype=torch.bool), 1.0, pool)
    assert torch.equal(so_campo, x)                                           # só fotos do PlantVillage são trocadas
    trocado = aumento.trocar_fundo(x, torch.ones(4, dtype=torch.bool), 1.0, pool)
    assert trocado.shape == x.shape and not torch.equal(trocado, x)
    assert torch.allclose(trocado[:, :, 32, 32], x[:, :, 32, 32], atol=0.05)  # a folha fica
    cinza = torch.full_like(x, 0.5)                                           # sem folha: área ~0, mantém a original
    assert torch.equal(aumento.trocar_fundo(cinza, torch.ones(4, dtype=torch.bool), 1.0, pool), cinza)


def test_preparar_treino_formato():
    xb = torch.randint(0, 255, (6, 80, 80, 3), dtype=torch.uint8)
    pool = torch.randint(0, 255, (3, 80, 80, 3), dtype=torch.uint8)
    x = aumento.preparar_treino(xb, 48, fonte=torch.zeros(6, dtype=torch.long), fundo_p=1.0, pool=pool)
    assert x.shape == (6, 3, 48, 48) and x.dtype == torch.float32 and torch.isfinite(x).all()


# ---------------------------------------------------------------- modelo
def test_carregar_pesos_expande_a_cabeca(tmp_path):
    m3 = modelo.criar(3, pretreinado=False)
    torch.save(m3.state_dict(), tmp_path / "m3.pt")
    m5 = modelo.criar(5, pretreinado=False)
    modelo.carregar_pesos(m5, tmp_path / "m3.pt")
    assert torch.equal(m5.head.fc.weight[:3], m3.head.fc.weight) and torch.equal(m5.head.fc.bias[:3], m3.head.fc.bias)
    assert m5.head.fc.weight.shape[0] == 5


def test_congelar():
    m = modelo.criar(3, pretreinado=False)
    modelo.congelar(m, 2)
    assert not any(p.requires_grad for p in m.stem.parameters()) and not any(p.requires_grad for p in m.stages[1].parameters())
    assert all(p.requires_grad for p in m.stages[3].parameters())
    modelo.congelar(m, 0)
    assert all(p.requires_grad for p in m.parameters())


# ---------------------------------------------------------------- métricas
def test_f1_macro_ignora_classes_so_previstas():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 1000)
    pred = y.copy()
    pred[rng.random(1000) < 0.3] = 99                       # erros caem numa classe que nunca é o gabarito
    from sklearn.metrics import f1_score
    com_uniao = f1_score(y, pred, average="macro")          # o cálculo ingênuo inclui a classe 99 com F1 zero
    assert avaliacao.f1_macro(y, pred) > com_uniao
    assert avaliacao.f1_macro(y, y) == 1.0


def test_ece_e_top_k():
    y = np.array([0, 1, 2, 3])
    perfeito = np.eye(4)
    assert avaliacao.ece(perfeito, y) == 0.0
    excesso = np.full((100, 2), [0.99, 0.01])
    assert avaliacao.ece(excesso, np.r_[np.zeros(50, int), np.ones(50, int)]) > 0.4     # 99% de confiança com 50% de acerto
    P = np.array([[0.1, 0.2, 0.7], [0.5, 0.4, 0.1]])
    assert avaliacao.top_k(P, np.array([1, 2]), k=2) == 0.5


def test_bootstrap_pareado():
    a = np.r_[np.ones(900, bool), np.zeros(100, bool)]
    d, lo, hi = comparacao.bootstrap(a, a)
    assert d == lo == hi == 0
    melhor = np.r_[np.ones(980, bool), np.zeros(20, bool)]
    d, lo, hi = comparacao.bootstrap(melhor, a)
    assert d == pytest.approx(8.0) and lo > 0 and hi < 12


# ---------------------------------------------------------------- dados
def test_dividir_dedupe_e_vazamento_exato():
    M = np.array(["a", "a", "b", "c", "d", "FAIL", "e", "b"])
    S = np.array([1, 1, 1, 0, 2, 1, 0, 0])
    tr, va, te, info = dados.dividir(M, S, proibidos=np.array(["e"]))
    assert list(tr) == [0, 2]                       # 'a' duplicado fica uma vez; 'FAIL' fora
    assert list(va) == [4] and list(te) == [3]      # 'b' no teste repete o treino; 'e' é proibido
    assert info["falhas"] == 1 and info["proibidas"] == 1 and info["dup_treino"] == 1


# ---------------------------------------------------------------- treino e amostragem
def _dados_falsos():
    n = 40
    Ytr = np.arange(n) % 4
    em_treino = np.arange(n) < 32
    return SimpleNamespace(n_pv=4, classes=list("abcdef"), Xtr=np.zeros((n, 8, 8, 3), np.uint8), Ytr=Ytr, em_treino=em_treino,
                           n_por_classe=np.bincount(Ytr[em_treino], minlength=4),
                           Xpd_tr=np.ones((6, 8, 8, 3), np.uint8), Ypd_tr=np.arange(6) % 4,
                           Xpw=np.full((10, 8, 8, 3), 2, np.uint8), Ypw=np.array([0, 1, 2, 3, 4, 5, 0, 1, 4, 5]), pw_tr=np.arange(10))


def test_montar_fontes_pesos_e_classes():
    cfg, d = Config(), _dados_falsos()
    f, w = treino.montar_fontes(cfg, d, "todos", nc=4)
    assert len(f) == 3 and w == list(cfg.todos_mix)
    assert (f[2].p[np.array(d.Ypw) >= 4] == 0).all()       # classes novas ficam fora do modelo de 38 classes
    f, w = treino.montar_fontes(cfg, d, "brasil", nc=6)
    assert len(f) == 3 and w[0] == cfg.br_mix[0] and sum(w) == pytest.approx(1.0)
    f, w = treino.montar_fontes(cfg, d, "balanceada", nc=4)
    contagem = np.bincount(d.Ytr, weights=f[0].p, minlength=4)
    assert np.allclose(contagem, contagem[0])                # classes equiprováveis


def test_fluxo_lotes():
    d = _dados_falsos()
    fontes, pesos = treino.montar_fontes(Config(), d, "campo", nc=4)
    lotes = list(treino.fluxo_lotes(fontes, pesos, 32, 8, np.random.default_rng(0)))
    assert len(lotes) == 4
    x, y, f = lotes[0]
    assert x.shape == (8, 8, 8, 3) and y.shape == f.shape == (8,)
    assert set(f.tolist()) <= {0, 1} and (x[f == 1] == 1).all() and (x[f == 0] == 0).all()    # fonte 1 = PlantDoc


# ---------------------------------------------------------------- vazamento
def test_duplicatas_por_pixel_sao_invariantes_a_rotacao_e_flip():
    rng = np.random.default_rng(0)
    tr = rng.integers(0, 255, (20, 32, 32, 3), dtype=np.uint8)
    te = np.stack([np.rot90(tr[3]), tr[7][:, ::-1], rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)])
    c = vazamento._corr_pixels(vazamento._pixels(tr), vazamento._pixels(te))
    assert c[0] > 0.99 and c[1] > 0.99 and c[2] < 0.9
