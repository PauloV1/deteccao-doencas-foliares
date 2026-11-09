"""Comparação de cada modificação com o seu controle (diferença pareada, IC95% por bootstrap) e tabelas."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_BOOT = 2000


def bootstrap(a, b, n=N_BOOT, seed=0):
    """Diferença média (a - b) em pontos percentuais e IC95%, reamostrando as imagens de teste."""
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def _ok(saida, nome):
    f = saida / nome / "ok.npz"
    return np.load(f) if f.exists() else None


def _pai(saida, nome):
    """De onde o run realmente partiu (campo `init` do done.json)."""
    f = saida / nome / "done.json"
    return json.load(open(f))["init"] if f.exists() else None


def _conjuntos(z, vaz_pv, n_pv):
    c = {"PlantVillage": z["pv"], "PlantVillage sem vazamento": z["pv"][~vaz_pv[z["sel"]]],
         "PlantDoc": z["pd"], "PlantDoc sem vazamento": z["pd"][~z["pd_leaky"]]}
    if len(z["pw"]):
        k = z["pw_y"] < n_pv                     # só as classes comuns a todos os modelos
        c["PlantWild (38 classes)"] = z["pw"][k]
        c["PlantWild sem vazamento"] = z["pw"][k & ~z["pw_leaky"]]
    return {nome: v for nome, v in c.items() if len(v)}      # subconjunto vazio não tem o que comparar


def pares(saida, nomes):
    """(rótulo, tratamento, controle). O controle da adaptação ao campo é o ponto de partida real."""
    return [
        ("1. amostragem balanceada", nomes["balanceado"], nomes["baseline"]),
        ("2. troca de fundo", nomes["balanceado"], nomes["sem_troca"]),
        ("3. + PlantDoc", nomes["campo_pd"], _pai(saida, nomes["campo_pd"])),
        ("4. + PlantWild (mesmo orçamento)", nomes["campo_todos"], nomes["campo_pd"]),
        ("5a. culturas do Brasil vs. mesmo treino sem classes novas", nomes["brasil"], nomes["campo_cont"]),
        ("5b. culturas do Brasil vs. ponto de partida", nomes["brasil"], nomes["campo_todos"]),
        ("efeito total da adaptação ao campo", nomes["campo_todos"], _pai(saida, nomes["campo_todos"])),
    ]


def comparar(cfg, d, nomes) -> pd.DataFrame:
    saida = cfg.saidas_dir
    linhas = []
    for rotulo, t, c in pares(saida, nomes):
        zt, zc = (_ok(saida, t), _ok(saida, c)) if c else (None, None)
        if zt is None or zc is None:
            print("pulando", rotulo, "— falta o run", t if zt is None else c)
            continue
        st, sc = _conjuntos(zt, d.vaz_pv, d.n_pv), _conjuntos(zc, d.vaz_pv, d.n_pv)
        for k in st:
            if k in sc and len(st[k]) == len(sc[k]):
                dif, lo, hi = bootstrap(st[k], sc[k])
                linhas.append({"modificação": rotulo, "tratamento": t, "controle": c, "teste": k, "n": len(st[k]),
                               "controle %": sc[k].mean() * 100, "tratamento %": st[k].mean() * 100, "Δ pp": dif, "IC95 inf": lo, "IC95 sup": hi,
                               "veredito": "melhora" if lo > 0 else ("piora" if hi < 0 else "sem diferença clara")})
    df = pd.DataFrame(linhas)
    df.to_csv(saida / "comparacao_modificacoes.csv", index=False)
    return df


def resumo_modelos(cfg, nomes) -> pd.DataFrame:
    """Uma linha por run avaliado, com todas as métricas (salva em todos_os_modelos.csv)."""
    ordem = [nomes[k] for k in ("baseline", "sem_troca", "balanceado", "campo_pd", "campo_todos", "campo_cont", "brasil")]
    linhas = [json.load(open(cfg.saidas_dir / n / "metrics.json"))["summary"] for n in ordem if (cfg.saidas_dir / n / "metrics.json").exists()]
    df = pd.DataFrame(linhas).set_index("run")
    df.to_csv(cfg.saidas_dir / "todos_os_modelos.csv")
    return df


def grafico_modelos(cfg, df):
    chaves = [("pv_acc", "PlantVillage"), ("pv_acc_clean", "PlantVillage\nsem vazamento"),
              ("pd_acc_clean", "PlantDoc\nsem vazamento"), ("pw_acc_clean", "PlantWild\nsem vazamento")]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(df)
    for j, nome in enumerate(df.index):
        v = [df.loc[nome, k] * 100 if k in df.columns and df.loc[nome, k] == df.loc[nome, k] else 0 for k, _ in chaves]
        ax.bar(np.arange(len(chaves)) + j * w, v, w, label=nome)
    ax.set_xticks(np.arange(len(chaves)) + 0.4 - w / 2)
    ax.set_xticklabels([r for _, r in chaves])
    ax.set_ylabel("acurácia (%)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title("Desempenho de cada modelo")
    plt.tight_layout()
    plt.savefig(cfg.saidas_dir / "comparacao_modelos.png", dpi=110)


def tabela_sementes(cfg, runs_por_config) -> pd.DataFrame:
    """Média e desvio entre sementes. `runs_por_config`: {rótulo: [metrics dict, ...]}."""
    cols = ["pv_acc", "pv_macro_f1", "pv_minority10_recall", "pd_top1", "pw_top1"]
    linhas = {}
    for rotulo, lista in runs_por_config.items():
        d = pd.DataFrame(lista)[cols] * 100
        linhas[rotulo] = pd.concat([d.mean().add_suffix(" (média)"), d.std().add_suffix(" (desvio)")])
    df = pd.DataFrame(linhas).T
    df.to_csv(cfg.saidas_dir / "sementes.csv")
    return df
