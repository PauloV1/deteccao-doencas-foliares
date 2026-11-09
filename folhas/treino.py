"""Treino: amostragem por fonte, laço de treino com AMP, retomada por época e seleção de época por validação."""
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from . import aumento, modelo
from .modelo import AMP, DEV

AMOSTRAGENS = ("natural", "balanceada", "campo", "todos", "brasil")
SELECOES = ("melhor", "melhor_campo", "ultima")


class Fonte:
    """Um conjunto de dados com a probabilidade de sorteio de cada amostra."""

    def __init__(self, X, Y, p):
        self.X, self.Y, self.p = X, Y, p / p.sum()


def fluxo_lotes(fontes, pesos, n, lote, rng):
    """Sorteia `n` amostras (com reposição) entre as fontes e entrega lotes pré-carregados por threads.
    Cada lote traz (imagens uint8, rótulos, índice da fonte de cada amostra)."""
    fonte_de = rng.choice(len(fontes), size=n, p=np.asarray(pesos) / np.sum(pesos))
    sorteadas = np.zeros(n, dtype=np.int64)
    for k, f in enumerate(fontes):
        mk = fonte_de == k
        if mk.any():
            sorteadas[mk] = rng.choice(len(f.Y), size=int(mk.sum()), p=f.p)
    nb = n // lote

    def buscar(b):
        sl = slice(b * lote, (b + 1) * lote)
        ids, pk = fonte_de[sl], sorteadas[sl]
        xs, ys, ss = [], [], []
        for k, f in enumerate(fontes):
            sel = np.sort(pk[ids == k])
            if len(sel):
                xs.append(torch.from_numpy(np.ascontiguousarray(f.X[sel])))
                ys.append(torch.from_numpy(f.Y[sel].astype(np.int64)))
                ss.append(torch.full((len(sel),), k, dtype=torch.int64))
        x, y, s = torch.cat(xs), torch.cat(ys), torch.cat(ss)
        return (x.pin_memory(), y, s) if DEV == "cuda" else (x, y, s)

    with ThreadPoolExecutor(2) as ex:
        fila = deque(ex.submit(buscar, b) for b in range(min(4, nb)))
        prox = len(fila)
        while fila:
            yield fila.popleft().result()
            if prox < nb:
                fila.append(ex.submit(buscar, prox))
                prox += 1


def _fonte_pv(d, amostragem):
    """PlantVillage: 'natural' = como está (desbalanceado); 'balanceada' = classes equiprováveis."""
    p = d.em_treino.astype(float) if amostragem == "natural" else d.em_treino / d.n_por_classe[d.Ytr]
    return Fonte(d.Xtr, d.Ytr, p)


def montar_fontes(cfg, d, amostragem, nc, campo_frac=0.5):
    """Devolve (fontes, pesos). A fonte 0 é sempre o PlantVillage."""
    if amostragem in ("natural", "balanceada"):
        return [_fonte_pv(d, amostragem)], [1.0]
    pd_ = Fonte(d.Xpd_tr, d.Ypd_tr, np.ones(len(d.Ypd_tr)))
    if amostragem == "campo":              # PlantVillage (replay) + PlantDoc
        return [_fonte_pv(d, "natural"), pd_], [1 - campo_frac, campo_frac]
    if amostragem == "todos":              # PlantVillage (replay) + PlantDoc + PlantWild, só as classes do modelo
        p = np.zeros(len(d.Ypw))
        p[d.pw_tr[d.Ypw[d.pw_tr] < nc]] = 1.0
        return [_fonte_pv(d, "natural"), pd_, Fonte(d.Xpw, d.Ypw, p)], list(cfg.todos_mix)
    if amostragem == "brasil":             # replay + campo (PlantDoc, PlantWild) com amostragem ∝ n_classe^-br_pot
        pool = [(d.Xpd_tr, d.Ypd_tr, np.arange(len(d.Ypd_tr))), (d.Xpw, d.Ypw, d.pw_tr)]
        cont = np.bincount(np.concatenate([y[i] for _, y, i in pool]), minlength=nc).astype(float)
        fontes, massa = [_fonte_pv(d, "natural")], []
        for X, y, i in pool:
            w = np.zeros(len(y))
            w[i] = cont[y[i]] ** -cfg.br_pot            # br_pot=0: natural; 1: classes equiprováveis
            fontes.append(Fonte(X, y, w))
            massa.append(w.sum())
        return fontes, [cfg.br_mix[0]] + [cfg.br_mix[1] * m / sum(massa) for m in massa]
    raise ValueError(f"amostragem desconhecida: {amostragem}")


def treinar(cfg, d, nome, amostragem, *, epocas, amostras, lr, init=None, seed=0, selecao="melhor",
            fundo_p=None, campo_frac=0.5, mult_cabeca=5.0):
    """Treina um modelo e grava em `saidas/<nome>/` (best.pt, done.json). Se `done.json` existe, pula;
    se a sessão caiu no meio, retoma da última época.

    amostragem: 'natural' | 'balanceada' | 'campo' | 'todos' | 'brasil'
    selecao   : 'melhor' (macro-F1 na validação do PlantVillage) | 'melhor_campo' (macro-F1 na validação do PlantWild) | 'ultima'
    init      : nome de outro run de onde partir (a cabeça é expandida se tiver mais classes)
    """
    if amostragem not in AMOSTRAGENS or selecao not in SELECOES:
        raise ValueError("amostragem/selecao inválida")
    saida = cfg.saidas_dir / nome
    saida.mkdir(parents=True, exist_ok=True)
    fundo_p = cfg.fundo_p if fundo_p is None else fundo_p
    if (saida / "done.json").exists():
        print("já concluído:", nome)
        return json.load(open(saida / "done.json"))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nc = len(d.classes) if amostragem == "brasil" else d.n_pv
    m = modelo.criar(nc, pretreinado=init is None)
    if init:
        modelo.carregar_pesos(m, cfg.saidas_dir / init / "best.pt")
    modelo.congelar(m, cfg.congelar)
    cabeca = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("head")]
    corpo = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": corpo, "lr": lr}, {"params": cabeca, "lr": lr * mult_cabeca}], weight_decay=0.05)
    passos = epocas * (amostras // cfg.lote)
    aquec = max(1, int(0.05 * passos))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / aquec if s < aquec else 0.5 * (1 + math.cos(math.pi * (s - aquec) / max(1, passos - aquec))))
    scaler = torch.amp.GradScaler(DEV, enabled=AMP)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    fontes, pesos = montar_fontes(cfg, d, amostragem, nc, campo_frac)
    if selecao == "melhor_campo":
        mk = d.Ypw_va < nc
        Xfv, Yfv = d.Xpw_va[mk], d.Ypw_va[mk]
    hist, melhor, inicio = [], -1.0, 0
    if (saida / "ckpt.pt").exists():                      # retomada após queda da sessão
        ck = torch.load(saida / "ckpt.pt", map_location=DEV)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        hist, melhor, inicio = ck["hist"], ck["best"], ck["ep"] + 1
        print(f"retomando {nome} da época {inicio + 1}/{epocas}")
    n_par = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"== {nome} | {amostragem} | treináveis {n_par:.1f}M | {amostras // cfg.lote} it/época | troca de fundo p={fundo_p} | {DEV}")
    for ep in range(inicio, epocas):
        m.train()
        t0, total, vistas = time.time(), 0.0, 0
        for it, (xb, yb, fb) in enumerate(fluxo_lotes(fontes, pesos, amostras, cfg.lote, rng)):
            x = aumento.preparar_treino(xb, cfg.tamanho, fb.to(DEV), fundo_p, d.fundos)
            y = yb.to(DEV, non_blocking=True)
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
                perda = crit(m(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perda).backward(); scaler.step(opt); scaler.update(); sched.step()
            total += perda.item(); vistas += len(y)
            if it % 50 == 0:
                print(f"  ep{ep + 1} it{it} perda {perda.item():.3f} | {vistas / (time.time() - t0):.0f} img/s", flush=True)
        P = modelo.prever(m, d.Xva, cfg.tamanho)
        acc, f1 = float((P.argmax(1) == d.Yva).mean()), float(f1_score(d.Yva, P.argmax(1), average="macro"))
        hist.append({"epoch": ep, "train_loss": total / (amostras // cfg.lote), "val_acc": acc, "val_macro_f1": f1, "sec": time.time() - t0})
        print(f"ep{ep + 1}/{epocas}: perda {hist[-1]['train_loss']:.3f} | val acc {acc:.4f} | macro-F1 {f1:.4f} | {hist[-1]['sec'] / 60:.1f} min", flush=True)
        criterio = f1
        if selecao == "melhor_campo":                     # escolhe a época pela validação de campo (nunca pelo teste)
            Pf = modelo.prever(m, Xfv, cfg.tamanho)
            criterio = float(f1_score(Yfv, Pf.argmax(1), labels=np.unique(Yfv), average="macro"))
            hist[-1].update(field_val_acc=float((Pf.argmax(1) == Yfv).mean()), field_val_macro_f1=criterio)
            print(f"   validação de campo: acc {hist[-1]['field_val_acc']:.4f} | macro-F1 {criterio:.4f}", flush=True)
        if selecao != "ultima" and criterio > melhor:
            melhor = criterio
            torch.save(m.state_dict(), saida / "best.pt")
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "hist": hist, "best": melhor, "ep": ep}, saida / "ckpt.pt")
    if selecao == "ultima":
        torch.save(m.state_dict(), saida / "best.pt")
    res = {"name": nome, "sampling": amostragem, "init": init, "seed": seed, "bg_p": fundo_p, "history": hist}
    json.dump(res, open(saida / "done.json", "w"), indent=1)
    (saida / "ckpt.pt").unlink(missing_ok=True)
    return res
