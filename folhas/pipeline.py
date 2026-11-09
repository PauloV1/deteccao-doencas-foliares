"""Etapas do experimento. O notebook só chama estas funções."""
import json
import shutil
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import ambiente, aumento, avaliacao, comparacao, dados, modelo
from .catalogo import BACKBONE
from .config import Config
from .treino import treinar

# nomes dos runs (pastas em `saidas/`)
NOMES = {"baseline": "baseline", "balanceado": "balanceado", "sem_troca": "sem_troca", "campo_pd": "campo_pd",
         "campo_todos": "campo_todos", "campo_cont": "campo_cont", "brasil": "brasil"}


@dataclass
class Contexto:
    cfg: Config
    d: dados.Dados


def _mostrar():
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        plt.close("all")


def _pct(df: pd.DataFrame) -> str:
    return (df * 100).round(2).to_string()


def preparar(cfg: Config | None = None) -> Contexto:
    """Verifica a GPU, baixa os dados (se preciso), monta os caches e calcula as máscaras de vazamento."""
    cfg = cfg or Config()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma"
    print(f"torch {torch.__version__} | GPU: {gpu}")
    if cfg.exigir_gpu and not torch.cuda.is_available():
        raise SystemExit("Sem GPU: no Colab, use Ambiente de execução > Alterar tipo > GPU T4.")
    cfg.saidas_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg.como_dict(), open(cfg.saidas_dir / "config.json", "w"), indent=1)
    ambiente.registrar(cfg)
    return Contexto(cfg, dados.carregar(cfg))


def previsualizar(ctx: Contexto):
    """Figuras de conferência: aumento de dados e troca de fundo (salvas em `saidas/`)."""
    cfg, d = ctx.cfg, ctx.d
    sel = np.sort(np.random.default_rng(1).choice(len(d.Ytr), 8, replace=False))
    xb = torch.from_numpy(np.ascontiguousarray(d.Xtr[sel]))
    aug = (aumento.preparar_treino(xb, cfg.tamanho) * modelo.STD + modelo.MEAN).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
    for i in range(8):
        ax[0, i].imshow(xb[i].numpy()); ax[1, i].imshow(aug[i]); ax[0, i].axis("off"); ax[1, i].axis("off")
    plt.tight_layout(); plt.savefig(cfg.saidas_dir / "aumento.png", dpi=80); _mostrar()

    x = modelo.para_float(torch.from_numpy(np.ascontiguousarray(d.Xtr[np.sort(np.random.default_rng(3).choice(len(d.Ytr), 8, replace=False))])))
    mascara = aumento.mascara_fundo(x)
    trocado = aumento.trocar_fundo(x, torch.ones(8, dtype=torch.bool, device=modelo.DEV), 1.0, d.fundos)
    fig, ax = plt.subplots(3, 8, figsize=(16, 6.4))
    for i in range(8):
        ax[0, i].imshow(x[i].permute(1, 2, 0).cpu()); ax[1, i].imshow(mascara[i, 0].cpu(), cmap="gray"); ax[2, i].imshow(trocado[i].permute(1, 2, 0).cpu())
        for r in range(3):
            ax[r, i].axis("off")
    plt.tight_layout(); plt.savefig(cfg.saidas_dir / "troca_de_fundo.png", dpi=80); _mostrar()


def _avaliar(ctx: Contexto, nome: str, graficos: bool = True) -> dict:
    """Avalia o run (uma única vez: reaproveita metrics.json se já existir)."""
    f = ctx.cfg.saidas_dir / nome / "metrics.json"
    if f.exists():
        return json.load(open(f))["summary"]
    return avaliacao.avaliar(ctx.cfg, ctx.d, nome, graficos)


def etapas_plantvillage(ctx: Contexto):
    """baseline (amostragem natural) e balanceado (classes equiprováveis), só com PlantVillage."""
    cfg, d = ctx.cfg, ctx.d
    treinar(cfg, d, NOMES["baseline"], "natural", epocas=cfg.epocas, amostras=cfg.amostras, lr=cfg.lr)
    treinar(cfg, d, NOMES["balanceado"], "balanceada", epocas=cfg.epocas, amostras=cfg.amostras, lr=cfg.lr)
    print(pd.DataFrame([_avaliar(ctx, NOMES["baseline"]), _avaliar(ctx, NOMES["balanceado"])]).set_index("run").round(4).T)


def etapas_campo(ctx: Contexto):
    """Adaptação ao campo a partir de um ponto de partida fixado a priori (`campo_pai`)."""
    cfg, d = ctx.cfg, ctx.d
    pai = NOMES[cfg.campo_pai]
    print("ponto de partida das etapas de campo:", pai)
    ep, am = (cfg.todos_epocas, cfg.todos_amostras) if cfg.campo_mesmo_orcamento else (cfg.campo_epocas, cfg.campo_amostras)
    treinar(cfg, d, NOMES["campo_pd"], "campo", epocas=ep, amostras=am, lr=cfg.campo_lr, init=pai, selecao="ultima", campo_frac=cfg.campo_frac)
    nomes = [NOMES["campo_pd"]]
    if d.tem_pw:
        treinar(cfg, d, NOMES["campo_todos"], "todos", epocas=cfg.todos_epocas, amostras=cfg.todos_amostras, lr=cfg.campo_lr, init=pai, selecao="melhor_campo")
        nomes.append(NOMES["campo_todos"])
    print(pd.DataFrame([_avaliar(ctx, n) for n in nomes]).set_index("run").round(4).T)


def etapa_brasil(ctx: Contexto):
    """Amplia a cabeça com as culturas do Brasil; o controle recebe o mesmo treino extra sem as classes novas."""
    cfg, d = ctx.cfg, ctx.d
    if not cfg.usar_br:
        print("culturas do Brasil desligadas (usar_br=False)")
        return
    treinar(cfg, d, NOMES["brasil"], "brasil", epocas=cfg.br_epocas, amostras=cfg.br_amostras, lr=cfg.campo_lr, init=NOMES["campo_todos"], selecao="melhor_campo")
    nomes = [NOMES["brasil"]]
    if cfg.br_controle:
        treinar(cfg, d, NOMES["campo_cont"], "todos", epocas=cfg.br_epocas, amostras=cfg.br_amostras, lr=cfg.campo_lr, init=NOMES["campo_todos"], selecao="melhor_campo")
        nomes.append(NOMES["campo_cont"])
    print(pd.DataFrame([_avaliar(ctx, n) for n in nomes]).set_index("run").round(4).T)


def ablacao_troca_fundo(ctx: Contexto):
    """Treina o balanceado sem troca de fundo, para medir o efeito dela."""
    cfg, d = ctx.cfg, ctx.d
    if not cfg.ablacao_fundo or cfg.fundo_p <= 0:
        return
    treinar(cfg, d, NOMES["sem_troca"], "balanceada", epocas=cfg.epocas, amostras=cfg.amostras, lr=cfg.lr, fundo_p=0.0)
    df = pd.DataFrame([_avaliar(ctx, NOMES["balanceado"]), _avaliar(ctx, NOMES["sem_troca"])]).set_index("run")
    df.to_csv(cfg.saidas_dir / "ablacao_troca_fundo.csv")
    print(df.round(4).T)


def sementes(ctx: Contexto):
    """Repete as etapas com PlantVillage com outras sementes: mede o ruído entre treinos."""
    cfg, d = ctx.cfg, ctx.d
    if not cfg.rodar_sementes:
        return
    configs = {"baseline": ("natural", cfg.fundo_p), "balanceado": ("balanceada", cfg.fundo_p)}
    if cfg.ablacao_fundo and cfg.fundo_p > 0:
        configs["sem_troca"] = ("balanceada", 0.0)
    runs = {k: [_avaliar(ctx, NOMES[k])] for k in configs}
    for s in cfg.sementes:
        for k, (amostragem, fundo) in configs.items():
            nome = f"{NOMES[k]}_s{s}"
            treinar(cfg, d, nome, amostragem, epocas=cfg.epocas, amostras=cfg.amostras, lr=cfg.lr, seed=s, fundo_p=fundo)
            runs[k].append(_avaliar(ctx, nome, graficos=False))
    df = comparacao.tabela_sementes(cfg, runs)
    print(f"média e desvio entre {len(cfg.sementes) + 1} sementes (%):")
    print(df.round(2).T.to_string())


def resultados(ctx: Contexto):
    """Avalia o que faltar, compara cada modificação com o seu controle e gera tabelas e gráfico."""
    cfg, d = ctx.cfg, ctx.d
    for nome in NOMES.values():
        if (cfg.saidas_dir / nome / "best.pt").exists():
            _avaliar(ctx, nome)
    df = comparacao.resumo_modelos(cfg, NOMES)
    cols = [c for c in ("pv_acc", "pv_acc_clean", "pv_macro_f1", "pv_minority10_recall", "pd_top1", "pd_acc_clean", "pd_ece",
                        "pw_top1", "pw_acc_clean", "pw_ece", "new_top1", "pv_leak_to_new") if c in df.columns]
    print("Métricas por modelo (%):")
    print(_pct(df[cols].T))
    cmp = comparacao.comparar(cfg, d, NOMES)
    print("\nComparações (Δ em pontos percentuais; IC95% por bootstrap):")
    print(cmp.round(2).to_string(index=False))
    print(f"\n{len(cmp)} comparações sem correção para múltiplas comparações: espere ~{0.05 * len(cmp):.0f} 'melhora/piora' por acaso.")
    comparacao.grafico_modelos(cfg, df)
    _mostrar()
    t = ambiente.tempos(cfg)
    if len(t):
        print(f"\nTempo de treino: {t.minutos.sum():.0f} min ({t.minutos.sum() / 60:.1f} h); por run em saidas/tempos.csv")
    f = cfg.saidas_dir / NOMES["brasil"] / "metrics.json"
    if f.exists():
        print("\nRecall por classe nova (teste oficial do PlantWild):")
        for c, v in sorted(json.load(open(f))["new_class_recall"].items(), key=lambda kv: kv[1]):
            print(f"  {v * 100:5.1f}%  {c}")


def exportar(ctx: Contexto):
    """Grava `model.pt` e `model_config.json` (entrada, classes) prontos para a API."""
    cfg, d = ctx.cfg, ctx.d
    final = NOMES["brasil"] if cfg.usar_br else (NOMES["campo_todos"] if cfg.usar_plantwild else NOMES["campo_pd"])
    pesos = cfg.saidas_dir / final / "best.pt"
    if not pesos.exists():
        raise FileNotFoundError(f"{pesos} não existe: rode as etapas antes de exportar")
    nc = torch.load(pesos, map_location="cpu")["head.fc.weight"].shape[0]
    shutil.copy(pesos, cfg.saidas_dir / "model.pt")
    json.dump({"tamanho": cfg.tamanho, "backbone": BACKBONE, "classes": d.classes[:nc]}, open(cfg.saidas_dir / "model_config.json", "w"), ensure_ascii=False)
    print(f"modelo final: {final} ({nc} classes) -> {cfg.saidas_dir}/model.pt e model_config.json")
