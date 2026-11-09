"""Avaliação de um modelo treinado nos três testes (PlantVillage, PlantDoc, PlantWild)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import modelo
from .modelo import DEV


def ece(P, y, bins=15) -> float:
    """Erro de calibração esperado: diferença entre a confiança do modelo e a taxa real de acerto."""
    conf, ok = P.max(1), P.argmax(1) == y
    bordas = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(bordas[:-1], bordas[1:]):
        mk = (conf > lo) & (conf <= hi)
        if mk.any():
            e += mk.mean() * abs(ok[mk].mean() - conf[mk].mean())
    return float(e)


def f1_macro(y, pred) -> float:
    """Macro-F1 só sobre as classes que existem no gabarito.
    (Sem isso, classes que aparecem apenas nas predições entram com F1 zero e derrubam a média.)"""
    return float(f1_score(y, pred, labels=np.unique(y), average="macro"))


def media(ok) -> float:
    """Acurácia de um subconjunto; NaN (sem aviso) se ele for vazio."""
    return float(ok.mean()) if len(ok) else float("nan")


def top_k(P, y, k=5) -> float:
    return float(np.mean([r in np.argsort(p)[-k:] for p, r in zip(P, y)]))


def avaliar(cfg, d, nome, graficos=True) -> dict:
    """Avalia `saidas/<nome>/best.pt`; grava metrics.json, ok.npz (acertos por amostra) e a matriz de confusão."""
    saida = cfg.saidas_dir / nome
    sd = torch.load(saida / "best.pt", map_location=DEV)
    nc = sd["head.fc.weight"].shape[0]
    m = modelo.criar(nc, pretreinado=False)
    m.load_state_dict(sd)
    n_pv, T = d.n_pv, cfg.tamanho

    sel = np.arange(len(d.Yte))
    if cfg.limite_avaliacao:
        sel = np.sort(np.random.default_rng(0).choice(len(d.Yte), cfg.limite_avaliacao, replace=False))
    P, Y = modelo.prever(m, d.Xte[sel], T), d.Yte[sel]
    pred = P.argmax(1)
    rel = classification_report(Y, pred, labels=range(n_pv), target_names=d.classes_pv, output_dict=True, zero_division=0)
    ok_pv = pred == Y
    Ppd = modelo.prever(m, d.Xpd_te, T)
    pdp = Ppd.argmax(1)
    ok_pd = pdp == d.Ypd_te
    res = {
        "run": nome,
        "pv_acc": media(ok_pv), "pv_acc_clean": media(ok_pv[~d.vaz_pv[sel]]), "pv_macro_f1": rel["macro avg"]["f1-score"],
        "pv_minority10_recall": float(np.mean([rel[d.classes_pv[i]]["recall"] for i in d.minoritarias])),
        "pd_top1": media(ok_pd), "pd_top5": top_k(Ppd, d.Ypd_te), "pd_acc_clean": media(ok_pd[~d.vaz_pd]),
        "pd_macro_f1": f1_macro(d.Ypd_te, pdp), "pd_ece": ece(Ppd, d.Ypd_te),
        "pd_healthy_vs_sick": float((d.saudavel[pdp] == d.saudavel[d.Ypd_te]).mean()),
    }
    novas = {}
    ok_pw = np.zeros(0, bool); y_pw = np.zeros(0, np.int64); vaz_pw = np.zeros(0, bool)
    if d.tem_pw:
        mk = d.Ypw_te < nc                       # só as classes que este modelo conhece
        Ppw, y_pw = modelo.prever(m, d.Xpw_te[mk], T), d.Ypw_te[mk]
        vaz_pw = d.vaz_pw[mk]
        ok_pw = Ppw.argmax(1) == y_pw
        c = y_pw < n_pv                          # as métricas pw_* usam SEMPRE as 38 classes comuns a todos os modelos
        res.update({
            "pw_top1": media(ok_pw[c]), "pw_top5": top_k(Ppw[c], y_pw[c]), "pw_acc_clean": media(ok_pw[c & ~vaz_pw]),
            "pw_macro_f1": f1_macro(y_pw[c], Ppw[c].argmax(1)), "pw_ece": ece(Ppw[c], y_pw[c]),
            "pw_healthy_vs_sick": float((d.saudavel[Ppw[c].argmax(1)] == d.saudavel[y_pw[c]]).mean()),
        })
        if nc > n_pv:                            # modelo expandido: classes novas e vazamento do PlantVillage para elas
            nova = y_pw >= n_pv
            res.update({
                "pw_all_top1": float(ok_pw.mean()), "new_top1": float(ok_pw[nova].mean()), "new_n": int(nova.sum()),
                "field_macro_f1": f1_macro(y_pw, Ppw.argmax(1)), "pv_leak_to_new": float((pred >= n_pv).mean()),
            })
            novas = {d.classes[i]: float((Ppw.argmax(1)[y_pw == i] == i).mean()) for i in range(n_pv, nc) if (y_pw == i).any()}
    np.savez(saida / "ok.npz", sel=sel, pv=ok_pv, pd=ok_pd, pd_leaky=d.vaz_pd, pw=ok_pw, pw_y=y_pw, pw_leaky=vaz_pw)
    json.dump({"summary": res, "per_class": {c: rel[c] for c in d.classes_pv}, "new_class_recall": novas}, open(saida / "metrics.json", "w"), indent=1)
    if graficos:
        cm = confusion_matrix(Y, pred, labels=range(n_pv), normalize="true")
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, cmap="Greens", xticklabels=[c.replace("___", "\n") for c in d.classes_pv], yticklabels=d.classes_pv, cbar=False)
        plt.xticks(fontsize=5); plt.yticks(fontsize=6)
        plt.title(f"Matriz de confusão (PlantVillage, teste) — {nome}")
        plt.tight_layout()
        plt.savefig(saida / "matriz_confusao.png", dpi=110)
        plt.close()
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return res
