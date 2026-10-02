"""Verificacao 3: Tabela 1 do Tecator com a coluna 0 (agua) e a 1 (gordura).

Refaz o protocolo dos scripts do artigo (preprocessing_fixed, compute_priors,
baseline_pls, train_light, train_with_vip, train_with_rf, train_random,
cross_validate, train_proper), sem alterar o repositorio deles.

Uso: python verificar_pipeline_tecator.py [--replicas N]   (padrao 10)
"""

import argparse
import os
import sys

sys.dont_write_bytecode = True  # nao criar __pycache__ no repositorio do artigo

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

# PGSG_REPO: path to the published article's repository.
# (Only modification to this script: the hard-coded path became an
# environment variable, so the script runs on any machine. See README.)
PGSG = Path(os.environ.get("PGSG_REPO", Path.home() / "pgsg-repo"))
sys.path.insert(0, str(PGSG / "src"))

from data.compute_priors import (
    compute_anova_scores,
    compute_rf_importance,
)
from models.band_select_net import BandSelectNet
from models.band_select_net_light import BandSelectNetLight

# valores publicados na Tabela 1 (RMSE, R2)
PUBLICADO = {
    "PLS": (0.296, 0.919),
    "Ours (ANOVA), teste unico": (None, 0.941),
    "Ours (ANOVA), 5-fold": (0.361, 0.862),
    "Ours (VIP), teste unico": (0.323, 0.903),
    "Ours (RF), teste unico": (0.258, 0.938),
    "Random init, teste unico": (0.373, 0.871),
    "Plain CNN, teste unico": (0.909, 0.235),
}

SEP = "=" * 78


def titulo(t):
    print()
    print(SEP)
    print(f" {t}")
    print(SEP)
    print()


def preparar(coluna):
    # mesmo split e Z-score do preprocessing_fixed.py
    X = pd.read_csv(PGSG / "data/raw/tecator/spectra.csv").values
    y = pd.read_csv(PGSG / "data/raw/tecator/targets.csv").values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sx, sy = StandardScaler(), StandardScaler()
    X_tr, X_te = sx.fit_transform(X_tr), sx.transform(X_te)
    y_tr, y_te = sy.fit_transform(y_tr), sy.transform(y_te)
    return X_tr, X_te, y_tr[:, coluna], y_te[:, coluna]


def compute_vip_scores(X, y, n_components=10):
    # copia do compute_priors.py; so muda o .ravel()[0] (o original quebra no numpy 2)
    if y.ndim > 1:
        y = y.ravel()
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    pls = PLSRegression(n_components=n_components).fit(X, y)

    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    vip = np.zeros(p)
    for i in range(p):
        peso = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2
                         for j in range(h)])
        vip[i] = np.sqrt(p * (s.T @ peso) / total_s).ravel()[0]
    return (vip - vip.min()) / (vip.max() - vip.min())


def priors(X_tr, y_tr):
    return {
        "anova": compute_anova_scores(X_tr, y_tr),
        "vip": compute_vip_scores(X_tr, y_tr),
        "rf": compute_rf_importance(X_tr, y_tr),
    }


def h_estrela(X, y, hmax=15):
    # H* pela regra de um erro-padrao em LOO, como no artigo
    from sklearn.model_selection import LeaveOneOut

    curva = []
    for H in range(1, hmax + 1):
        erros = []
        for tr, va in LeaveOneOut().split(X):
            m = PLSRegression(n_components=H).fit(X[tr], y[tr])
            erros.append(float((y[va] - m.predict(X[va]).ravel())[0]) ** 2)
        erros = np.asarray(erros)
        rmse = float(np.sqrt(erros.mean()))
        se = float(erros.std(ddof=1) / np.sqrt(len(erros)) / (2 * rmse))
        curva.append((H, rmse, se))
    H_min, rmse_min, se_min = min(curva, key=lambda t: t[1])
    limite = rmse_min + se_min
    return min(H for H, r, _ in curva if r <= limite), H_min, rmse_min, se_min


def metricas(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return rmse, float(r2_score(y_true, y_pred))


def treinar(modelo, X_tr, y_tr, X_av, y_av, n_epochs, lr, kl_weight,
            weight_decay, patience):
    # mesmo laco dos scripts do artigo (modelo final = ultima epoca)
    Xtr = torch.FloatTensor(X_tr)
    ytr = torch.FloatTensor(y_tr)
    Xav = torch.FloatTensor(X_av)
    yav = torch.FloatTensor(y_av)

    crit = nn.MSELoss()
    opt = optim.Adam(modelo.parameters(), lr=lr, weight_decay=weight_decay)

    melhor, sem_melhora = float("inf"), 0
    for _ in range(n_epochs):
        modelo.train()
        pred, _ = modelo(Xtr)
        perda = crit(pred.squeeze(), ytr) + kl_weight * modelo.get_kl_loss()
        opt.zero_grad()
        perda.backward()
        opt.step()

        modelo.eval()
        with torch.no_grad():
            p_av, _ = modelo(Xav)
            rmse_av = torch.sqrt(crit(p_av.squeeze(), yav))
        if rmse_av < melhor:
            melhor, sem_melhora = rmse_av, 0
        else:
            sem_melhora += 1
            if sem_melhora >= patience:
                break

    modelo.eval()
    with torch.no_grad():
        pred, gates = modelo(Xav)
    return metricas(yav, pred.squeeze().numpy()), gates.detach().numpy()


def leve(prior, semente):
    torch.manual_seed(semente)
    return BandSelectNetLight(n_bands=100, n_outputs=1, prior_scores=prior,
                              hidden_dim=16, dropout=0.5)


def pesada(prior, semente):
    torch.manual_seed(semente)
    return BandSelectNet(n_bands=100, n_outputs=1, prior_scores=prior)


def rodar_coluna(coluna, rotulo, conteudo, replicas):
    titulo(f"COLUNA {coluna} (rotulada '{rotulo}') = {conteudo.upper()}")

    X_tr, X_te, y_tr, y_te = preparar(coluna)
    print(f"  treino {X_tr.shape}  teste {X_te.shape}")
    print(f"  priors calculados sobre esta coluna:")
    P = priors(X_tr, y_tr)
    print()

    linhas = {}

    # PLS com H* refeito para esta coluna
    H, H_min, rmse_min, se = h_estrela(X_tr, y_tr)
    print(f"  H* por LOO+1SE: H_min={H_min}, RMSECV={rmse_min:.4f}, "
          f"SE={se:.4f} -> H*={H}")
    pls = PLSRegression(n_components=H).fit(X_tr, y_tr)
    linhas["PLS"] = [metricas(y_te, pls.predict(X_te))]
    linhas["_H"] = H

    # PLS em 5-fold, nas mesmas folds do PGSG
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    porf = []
    for tr_i, va_i in kf.split(X_tr):
        m = PLSRegression(n_components=H).fit(X_tr[tr_i], y_tr[tr_i])
        porf.append(metricas(y_tr[va_i], m.predict(X_tr[va_i])))
    linhas["PLS, 5-fold"] = [(float(np.mean([v[0] for v in porf])),
                             float(np.mean([v[1] for v in porf])))]

    # teste unico: ANOVA, VIP, RF, sem prior, CNN pesada
    receitas = [
        ("Ours (ANOVA), teste unico", P["anova"], leve, 200, 0.01, 0.001, 1e-4, 30),
        ("Ours (VIP), teste unico", P["vip"], leve, 200, 0.01, 0.001, 1e-4, 30),
        ("Ours (RF), teste unico", P["rf"], leve, 200, 0.01, 0.001, 1e-4, 30),
        # train_random.py nao tem parada antecipada (paciencia 200)
        ("Random init, teste unico", None, leve, 200, 0.01, 0.001, 1e-4, 200),
        ("Plain CNN, teste unico", P["anova"], pesada, 200, 0.01, 0.0001, 0.0, 20),
    ]
    for nome, pri, fab, ep, lr, kl, wd, pac in receitas:
        vals = []
        for s in range(replicas):
            m = fab(pri, s)
            (rmse, r2), _ = treinar(m, X_tr, y_tr, X_te, y_te, ep, lr, kl, wd, pac)
            vals.append((rmse, r2))
        linhas[nome] = vals
        print(f"  {nome:<28} ok ({replicas} replicas)")

    # 5-fold sobre o treino, como cross_validate.py
    vals, detalhe = [], []
    for s in range(replicas):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        por_fold = []
        for tr_i, va_i in kf.split(X_tr):
            m = leve(P["anova"], s)
            (rmse, r2), _ = treinar(m, X_tr[tr_i], y_tr[tr_i], X_tr[va_i],
                                    y_tr[va_i], 200, 0.01, 0.001, 1e-4, 30)
            por_fold.append((rmse, r2))
        vals.append((float(np.mean([v[0] for v in por_fold])),
                     float(np.mean([v[1] for v in por_fold]))))
        detalhe.append([v[1] for v in por_fold])
    linhas["Ours (ANOVA), 5-fold"] = vals
    linhas["_folds"] = detalhe
    print(f"  {'Ours (ANOVA), 5-fold':<28} ok ({replicas} replicas)")

    return linhas, P


def resumo(v):
    # mediana [min; max] entre replicas
    rmse = np.array([x[0] for x in v])
    r2 = np.array([x[1] for x in v])
    if len(v) == 1:
        return f"{rmse[0]:.3f}", f"{r2[0]:.3f}"
    return (f"{np.median(rmse):.3f} [{rmse.min():.2f};{rmse.max():.2f}]",
            f"{np.median(r2):.3f} [{r2.min():.2f};{r2.max():.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=10)
    a = ap.parse_args()

    print(SEP)
    print(" VERIFICACAO 3 - O PIPELINE DO TECATOR COM O ALVO CORRETO")
    print(SEP)
    print(f"\n  replicas por linha: {a.replicas}   (o pipeline original nao")
    print("  fixa semente; cada execucao dele da um numero diferente)")

    titulo("CONFERENCIA: PEGUEI OS MODELOS CERTOS?")
    print("  O manuscrito declara as contagens de parametros. Se batem, o")
    print("  mapeamento 'Ours' = leve e 'Plain CNN' = pesada esta certo.")
    print()
    for nome, m, pub in (
        ("Ours, Tecator (leve, 100 bandas)",
         BandSelectNetLight(100, 1, None, 16, 0.5), 1765),
        ("Ours, Gasoline (leve, 401 bandas)",
         BandSelectNetLight(401, 1, None, 16, 0.5), 6882),
        ("Plain CNN (pesada, 100 bandas)",
         BandSelectNet(100, 1, None), 109221),
    ):
        n = sum(p.numel() for p in m.parameters())
        print(f"  {nome:<36} {n:>8,}  manuscrito {pub:>8,}  "
              f"{'ok' if n == pub else 'DIFERE'}")
    print()
    print("  nota: a 'Plain CNN' tambem tem a camada de portas e o prior")
    print("  ANOVA - e a versao pesada do mesmo modelo, nao uma CNN sem")
    print("  gating. O rotulo da tabela e deles; reproduzi o que o")
    print("  train_proper.py faz.")

    agua, P_agua = rodar_coluna(0, "Fat", "agua", a.replicas)
    gordura, P_gord = rodar_coluna(1, "Water", "gordura", a.replicas)

    titulo("TABELA 1, TECATOR - PUBLICADO x AGUA x GORDURA REAL")
    print("  As linhas de rede sao MEDIANA [minimo;maximo] entre replicas.")
    print("  O publicado e uma medida unica, e o +- dele e entre folds.")
    print()
    print("  nota sobre o 'PLS, 5-fold': o manuscrito (linha 388) publica")
    print("  0.921 +- 0.018, mas a saida do proprio autor")
    print("  (revision_r1/one_se_output.txt) registra 0.9218 +- 0.0439 nas")
    print("  mesmas folds. A media bate; o desvio, nao.")
    print()
    ordem = ["PLS", "PLS, 5-fold", "Ours (ANOVA), teste unico",
             "Ours (ANOVA), 5-fold", "Ours (VIP), teste unico",
             "Ours (RF), teste unico", "Random init, teste unico",
             "Plain CNN, teste unico"]
    PUBLICADO["PLS, 5-fold"] = (None, 0.921)  # manuscript_R2.tex:388

    print(f"  {'':<28} {'PUBLICADO':>10} {'col 0 = agua':>24} "
          f"{'col 1 = gordura':>24}")
    print(f"  {'R2':<28}")
    for nome in ordem:
        pub = PUBLICADO[nome][1]
        _, r2_a = resumo(agua[nome])
        _, r2_g = resumo(gordura[nome])
        print(f"  {nome:<28} {pub:>10.3f} {r2_a:>24} {r2_g:>24}")
    print()
    print(f"  {'RMSE':<28}")
    for nome in ordem:
        pub = PUBLICADO[nome][0]
        pub_s = f"{pub:.3f}" if pub is not None else "---"
        rmse_a, _ = resumo(agua[nome])
        rmse_g, _ = resumo(gordura[nome])
        print(f"  {nome:<28} {pub_s:>10} {rmse_a:>24} {rmse_g:>24}")

    titulo("REPLICAS ANOMALAS NO 5-FOLD")
    print("  Antes de comparar qualquer coisa: alguma replica colapsou? Uma")
    print("  fold que colapsa move a media da replica inteira, e a media")
    print("  entre replicas depois.")
    print()
    for nome_col, d in (("col 0 = agua", agua), ("col 1 = gordura", gordura)):
        print(f"  {nome_col}:")
        houve = False
        for s, folds in enumerate(d["_folds"]):
            f = np.asarray(folds)
            if f.std(ddof=0) > 0.15 or f.min() < 0.5:
                print(f"    semente {s}: media {f.mean():.3f}  "
                      f"folds {np.round(f, 3).tolist()}")
                houve = True
        if not houve:
            print("    nenhuma")
    print()
    print("  -> e por isso que a tabela acima usa MEDIANA, nao media.")

    titulo("A DISTANCIA ENTRE O PGSG E O PLS")
    print("  O manuscrito resume como 'a performance gap of approximately")
    print("  6%' (PGSG 0.862 contra PLS 0.919). Essa conta compara 5-fold")
    print("  com teste unico. O proprio manuscrito da o numero justo na")
    print("  linha 388: PLS em 5-fold = 0.921+-0.018. Refaco das duas")
    print("  formas, para as duas colunas.")
    print()
    for nome_col, d in (("col 0 = agua", agua), ("col 1 = gordura", gordura)):
        cv = np.array([v[1] for v in d["Ours (ANOVA), 5-fold"]])
        print(f"  {nome_col}  (H* = {d['_H']})")
        for etq, ref in (("contra PLS teste unico", d["PLS"][0][1]),
                         ("contra PLS 5-fold     ", d["PLS, 5-fold"][0][1])):
            lac = (ref - cv) / ref * 100
            print(f"    {etq}  PLS {ref:.3f}   PGSG {np.median(cv):.3f}   "
                  f"lacuna mediana {np.median(lac):.1f}%  "
                  f"faixa [{lac.min():.1f};{lac.max():.1f}]")
        print()

    # mesma semente = mesmos pesos iniciais nas duas colunas, entao da pra parear
    a_cv = np.array([v[1] for v in agua["Ours (ANOVA), 5-fold"]])
    g_cv = np.array([v[1] for v in gordura["Ours (ANOVA), 5-fold"]])
    try:
        from scipy.stats import mannwhitneyu, wilcoxon
        _, p_w = wilcoxon(a_cv, g_cv)
        _, p_m = mannwhitneyu(a_cv, g_cv, alternative="two-sided")
        print(f"  PGSG 5-fold, agua x gordura: Wilcoxon pareado p={p_w:.3f}, "
              f"Mann-Whitney p={p_m:.3f}")
    except Exception:
        pass
    print(f"  medianas: agua {np.median(a_cv):.3f}  "
          f"gordura {np.median(g_cv):.3f}")
    print("  -> o PGSG praticamente nao muda. Quem muda e o PLS, que")
    print("     melhora; e e dai que vem o aumento da lacuna.")

    titulo("O PRIOR ANOVA GANHA DA INICIALIZACAO ALEATORIA?")
    print("  O artigo argumenta que sim, comparando 0.941 com 0.871 - duas")
    print("  medidas unicas. Com replicas:")
    print()
    print("  O teste e Mann-Whitney, de amostras independentes, e NAO um")
    print("  pareado por semente: a mesma semente nao emparelha nada entre")
    print("  os dois bracos, porque o braco sem prior sorteia as gates")
    print("  primeiro e as camadas seguintes acabam com pesos diferentes.")
    print()
    for nome_col, d in (("col 0 = agua", agua), ("col 1 = gordura", gordura)):
        an = np.array([v[1] for v in d["Ours (ANOVA), teste unico"]])
        rd = np.array([v[1] for v in d["Random init, teste unico"]])
        linha = (f"  {nome_col:<18} ANOVA {np.median(an):.3f}   "
                 f"aleatorio {np.median(rd):.3f}   "
                 f"dif das medianas {np.median(an) - np.median(rd):+.3f}")
        try:
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(an, rd, alternative="two-sided")
            linha += f"   Mann-Whitney p={p:.3f}"
        except Exception:
            pass
        print(linha)
    print()
    print("  ressalva: o braco ANOVA para observando o proprio teste e o")
    print("  aleatorio roda as 200 epocas. Sao criterios de parada")
    print("  diferentes, e isso favorece o braco com parada.")

    titulo("TABELA 2 - AS GATES SAO APRENDIDAS?")
    print("  Duas coisas separadas, para nao confundir o que e leitura de")
    print("  codigo com o que e medida:")
    print()
    print("  (a) POR LEITURA: o analyze_gates.py cria um modelo novo e le as")
    print("      gates. Nao ha torch.load ali, e nenhum script do pipeline")
    print("      do Tecator chama torch.save - logo nao existe modelo")
    print("      treinado para carregar. O que ele reporta e a inicializacao.")
    print()
    print("  (b) POR MEDIDA: quanto o treino move as gates? SEMENTE 0 SO -")
    print("      ilustracao. A medida em 10 sementes, e com outros lr, epocas")
    print("      e KL, esta em scripts/verificar_varredura_treino.py.")
    print()
    eixo = np.linspace(850, 1050, 100)
    for coluna, nome_col, P in ((0, "agua", P_agua), (1, "gordura", P_gord)):
        X_tr, X_te, y_tr, y_te = preparar(coluna)
        m0 = leve(P["anova"], 0)
        m0.eval()
        with torch.no_grad():
            _, g0 = m0(torch.FloatTensor(X_te))
        g0 = g0.numpy()
        m1 = leve(P["anova"], 0)
        _, g1 = treinar(m1, X_tr, y_tr, X_te, y_te, 200, 0.01, 0.001, 1e-4, 30)

        print(f"  coluna {coluna} ({nome_col}):")
        for etiqueta, g in (("sem treinar", g0), ("depois de treinar", g1)):
            top = int(np.argmax(g))
            corr = float(np.corrcoef(g, P["anova"])[0, 1])
            print(f"    {etiqueta:<18} banda topo {top:>3} "
                  f"({eixo[top]:.0f} nm, peso {g[top]:.3f})  "
                  f"corr. com o prior {corr:.3f}")

        # quanto os logits das portas andaram no treino
        l0 = m0.gating.gates.detach().numpy()
        l1 = m1.gating.gates.detach().numpy()
        mov = np.abs(l1 - l0)
        faixa = l0.max() - l0.min()
        print(f"    logits: faixa inicial {faixa:.1f} "
              f"(de {l0.min():.1f} a {l0.max():.1f})")
        print(f"            deslocamento no treino: maior {mov.max():.2f}, "
              f"mediano {np.median(mov):.2f}")
        print(f"            referencia do Adam em 200 passos a lr=0.01: ~2.0"
              f" (ordem de grandeza, nao teto rigido)")
        print(f"            distancia do 2o ao 1o colocado: "
              f"{np.sort(l0)[-1] - np.sort(l0)[-2]:.2f}")
        print()
    print("  publicado na Tabela 2: 931 nm (0.215), corr. com o prior 0.348")
    print()


if __name__ == "__main__":
    main()
