"""Varredura de epocas, lr e peso do KL: que treino reproduz a Tabela 2?

Tecator (coluna 0) e Gasoline, sementes 0 a 9, receita do train_light.py.
"Reproduz" = peso da maior porta E correlacao com o prior iguais ao
publicado na 3a casa decimal.

  grade A: epocas 0..200, sem parada, lr 0.01, KL 0.001
  grade B: lr 1e-4..1e-2, 200 epocas, paciencia 30, KL 0.001
  grade C: KL 0..10, 200 epocas, paciencia 30, lr 0.01

Uso: python verificar_varredura_treino.py
"""

import os
import sys

sys.dont_write_bytecode = True  # nao criar __pycache__ no repositorio do artigo

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# PGSG_REPO: path to the published article's repository.
# (Only modification to this script: the hard-coded path became an
# environment variable, so the script runs on any machine. See README.)
PGSG = Path(os.environ.get("PGSG_REPO", Path.home() / "pgsg-repo"))
sys.path.insert(0, str(PGSG / "src"))

from data.compute_priors import compute_anova_scores
from models.band_select_net_light import BandSelectNetLight

SEP = "=" * 78
SEMENTES = range(10)
PUBLICADO = {"tecator": (0.215, 0.348), "gasoline": (0.130, 0.507)}

GRADE_A = [0, 1, 2, 5, 10, 20, 50, 100, 200]
GRADE_B = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
GRADE_C = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def titulo(t):
    print()
    print(SEP)
    print(f" {t}")
    print(SEP)
    print()


def preparar(conjunto):
    X = pd.read_csv(PGSG / f"data/raw/{conjunto}/spectra.csv").values
    y = pd.read_csv(PGSG / f"data/raw/{conjunto}/targets.csv").values
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sx, sy = StandardScaler(), StandardScaler()
    Xtr, Xte = sx.fit_transform(Xtr), sx.transform(Xte)
    ytr, yte = sy.fit_transform(ytr), sy.transform(yte)
    return Xtr, Xte, ytr[:, 0], yte[:, 0]


def treinar(prior, Xtr, ytr, Xte, yte, semente, epocas, lr, kl, paciencia):
    # receita do train_light.py; paciencia=None = sem parada antecipada
    torch.manual_seed(semente)
    m = BandSelectNetLight(Xtr.shape[1], 1, prior, 16, 0.5)
    logit0 = m.gating.gates.detach().clone().numpy()
    Xtr_t, ytr_t = torch.FloatTensor(Xtr), torch.FloatTensor(ytr)
    Xte_t, yte_t = torch.FloatTensor(Xte), torch.FloatTensor(yte)
    crit = nn.MSELoss()
    opt = optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    melhor, sem = float("inf"), 0
    for _ in range(epocas):
        m.train()
        pred, _ = m(Xtr_t)
        perda = crit(pred.squeeze(), ytr_t) + kl * m.get_kl_loss()
        opt.zero_grad()
        perda.backward()
        opt.step()
        if paciencia is None:
            continue
        m.eval()
        with torch.no_grad():
            p, _ = m(Xte_t)
            r = torch.sqrt(crit(p.squeeze(), yte_t))
        if r < melhor:
            melhor, sem = r, 0
        else:
            sem += 1
            if sem >= paciencia:
                break
    m.eval()
    with torch.no_grad():
        _, g = m(Xte_t)
    return g.numpy(), logit0, m.gating.gates.detach().numpy()


def medir(g, prior):
    return float(g.max()), float(np.corrcoef(g, prior)[0, 1])


def bate(v, pub):
    return round(v, 3) == pub


def linha(rotulo, pesos, corrs, pub):
    pesos, corrs = np.asarray(pesos), np.asarray(corrs)
    n = sum(bate(p, pub[0]) and bate(c, pub[1]) for p, c in zip(pesos, corrs))
    return (f"  {rotulo:<12} peso {np.median(pesos):.3f} "
            f"[{pesos.min():.3f}; {pesos.max():.3f}]   "
            f"corr {np.median(corrs):.3f} [{corrs.min():.3f}; {corrs.max():.3f}]"
            f"   reproduz em {n:>2}/10")


for conjunto in ("tecator", "gasoline"):
    Xtr, Xte, ytr, yte = preparar(conjunto)
    prior = compute_anova_scores(Xtr, ytr)
    pub = PUBLICADO[conjunto]
    titulo(f"{conjunto.upper()} - publicado: peso {pub[0]}, corr {pub[1]}")
    print("  mediana [minimo; maximo] em 10 sementes. 'reproduz' = peso E")
    print("  corr iguais ao publicado na 3a casa.")

    print()
    print("  GRADE A - epocas fixas, sem parada, lr 0.01, KL 0.001")
    for ep in GRADE_A:
        r = [medir(treinar(prior, Xtr, ytr, Xte, yte, s, ep, 0.01, 0.001,
                           None)[0], prior) for s in SEMENTES]
        print(linha(f"{ep} epocas", *zip(*r), pub))

    print()
    print("  GRADE B - lr, 200 epocas, paciencia 30, KL 0.001")
    for lr in GRADE_B:
        r = [medir(treinar(prior, Xtr, ytr, Xte, yte, s, 200, lr, 0.001,
                           30)[0], prior) for s in SEMENTES]
        print(linha(f"lr {lr:g}", *zip(*r), pub))

    print()
    print("  GRADE C - peso do KL, 200 epocas, paciencia 30, lr 0.01")
    for kl in GRADE_C:
        r = [medir(treinar(prior, Xtr, ytr, Xte, yte, s, 200, 0.01, kl,
                           30)[0], prior) for s in SEMENTES]
        print(linha(f"KL {kl:g}", *zip(*r), pub))

    print()
    print("  LOGITS NO PROTOCOLO DO ARTIGO (lr 0.01, 200 ep, pac. 30, KL 0.001)")
    desl, dist0, dist1 = [], [], []
    for s in SEMENTES:
        _, l0, l1 = treinar(prior, Xtr, ytr, Xte, yte, s, 200, 0.01, 0.001, 30)
        desl.append(float(np.abs(l1 - l0).max()))
        o0, o1 = np.sort(l0)[::-1], np.sort(l1)[::-1]
        dist0.append(float(o0[0] - o0[1]))
        dist1.append(float(o1[0] - o1[1]))
    desl = np.asarray(desl)
    print(f"  maior deslocamento de uma banda   mediana {np.median(desl):.2f}"
          f"  faixa [{desl.min():.2f}; {desl.max():.2f}]")
    print(f"  distancia 1o-2o logit, inicio     {min(dist0):.2f} a {max(dist0):.2f}")
    print(f"  distancia 1o-2o logit, fim        {min(dist1):.2f} a {max(dist1):.2f}")
print()
