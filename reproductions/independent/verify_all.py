"""Independent reproduction of the published Tecator and Gasoline results."""
import os, sys, warnings
from pathlib import Path

PGSG = Path(os.environ.get("PGSG_REPO", Path.cwd()))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, str(PGSG / "src"))
from models.band_select_net_light import create_model_light, BandSelectNetLight


def load(ds):
    d = PGSG / "data/raw" / ds
    X = pd.read_csv(d / "spectra.csv").values
    y = pd.read_csv(d / "targets.csv").values
    return X, y


def split_scale(X, y, test_size=0.2):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    sx, sy = StandardScaler(), StandardScaler()
    return (sx.fit_transform(Xtr), sx.transform(Xte),
            sy.fit_transform(ytr), sy.transform(yte))


def anova_prior(X, y):
    f, _ = f_regression(X, y.ravel())
    return (f - f.min()) / (f.max() - f.min())


def gates_from_prior(prior):
    """Replicate SpectralGatingLayer init + forward, untrained."""
    pc = np.clip(prior, 1e-7, 1 - 1e-7)
    logits = np.log(pc / (1 - pc))
    t = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(t / 5.0, dim=0).numpy(), logits


def train_gates(Xtr, ytr, Xte, yte, prior_path, n_bands, seed,
                epochs=200, lr=0.01, kl_weight=0.001, patience=30):
    torch.manual_seed(seed)
    model = create_model_light(n_bands, 1, prior_path)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_t = torch.FloatTensor(Xtr); ytr_t = torch.FloatTensor(np.ravel(ytr))
    Xte_t = torch.FloatTensor(Xte); yte_t = torch.FloatTensor(np.ravel(yte))
    best = float("inf"); bad = 0
    for ep in range(epochs):
        model.train()
        pred, _ = model(Xtr_t)
        loss = crit(pred.squeeze(), ytr_t) + kl_weight * model.get_kl_loss()
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pt, _ = model(Xte_t)
            rmse = torch.sqrt(crit(pt.squeeze(), yte_t)).item()
        if rmse < best:
            best = rmse; bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.eval()
    with torch.no_grad():
        pt, g = model(Xte_t)
        rmse = torch.sqrt(crit(pt.squeeze(), yte_t)).item()
        ssr = ((yte_t - pt.squeeze()) ** 2).sum().item()
        sst = ((yte_t - yte_t.mean()) ** 2).sum().item()
    return g.numpy(), model.gating.gates.detach().numpy(), 1 - ssr / sst, rmse, ep + 1


def pls_1se(Xtr, ytr, max_h=15):
    """H* by one-standard-error rule over LOO, as the article describes."""
    loo = LeaveOneOut()
    n = len(Xtr)
    out = {}
    for h in range(1, max_h + 1):
        err = np.empty(n)
        for tr, te in loo.split(Xtr):
            m = PLSRegression(n_components=h).fit(Xtr[tr], ytr[tr])
            err[te] = (m.predict(Xtr[te]).ravel() - ytr[te].ravel()) ** 2
        mse = err.mean()
        rmse = np.sqrt(mse)
        se_mse = err.std(ddof=1) / np.sqrt(n)
        out[h] = (rmse, se_mse / (2 * rmse))
    hmin = min(out, key=lambda h: out[h][0])
    thr = out[hmin][0] + out[hmin][1]
    hstar = min(h for h in out if out[h][0] <= thr)
    return hstar, hmin, out


def pls_eval(Xtr, ytr, Xte, yte, h):
    m = PLSRegression(n_components=h).fit(Xtr, ytr)
    p = m.predict(Xte).ravel()
    yt = yte.ravel()
    ssr = ((yt - p) ** 2).sum(); sst = ((yt - yt.mean()) ** 2).sum()
    return 1 - ssr / sst, np.sqrt(((yt - p) ** 2).mean())


def rule(t):
    print("\n" + "=" * 74); print(" " + t); print("=" * 74)


# ----------------------------------------------------------------------
rule("1. TECATOR — which column is which")
X, Y = load("tecator")
hdr = pd.read_csv(PGSG / "data/raw/tecator/targets.csv").columns.tolist()
print(f"  spectra {X.shape}   targets {Y.shape}   header {hdr}")
for i, name in enumerate(hdr):
    print(f"   col {i}  labelled {name:<8} min {Y[:,i].min():6.2f}  max {Y[:,i].max():6.2f}  mean {Y[:,i].mean():6.2f}")
print("  canonical Tecator: fat 0.9-49.1, water 39.3-76.6, protein 11.0-21.8")

rule("2. TECATOR — PLS under the published protocol, each column")
Xtr, Xte, Ytr, Yte = split_scale(X, Y)
print(f"  train {Xtr.shape[0]} / test {Xte.shape[0]}")
print(f"  {'col':<4}{'label':<10}{'is':<10}{'H_min':<7}{'H*':<6}{'R2':<9}{'RMSE':<9}")
hstars = {}
for i, name in enumerate(hdr):
    hs, hm, tbl = pls_1se(Xtr, Ytr[:, i:i+1])
    r2, rmse = pls_eval(Xtr, Ytr[:, i:i+1], Xte, Yte[:, i:i+1], hs)
    hstars[i] = hs
    truth = {0: "water", 1: "fat", 2: "protein"}[i]
    print(f"  {i:<4}{name:<10}{truth:<10}{hm:<7}{hs:<6}{r2:<9.4f}{rmse:<9.4f}")
    if i == 0:
        print(f"       -> RMSECV at H_min = {tbl[hm][0]:.4f} +/- {tbl[hm][1]:.4f}"
              f"   (published: 0.2093 +/- 0.0269, H*=10)")
print("  published Table 1: PLS R2 = 0.919, RMSE = 0.296")

rule("3. TECATOR — PLS 5-fold on the training set")
for i in (0, 1):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s, rm = [], []
    for tr, va in kf.split(Xtr):
        a, b = pls_eval(Xtr[tr], Ytr[tr, i:i+1], Xtr[va], Ytr[va, i:i+1], hstars[i])
        r2s.append(a); rm.append(b)
    truth = {0: "water", 1: "fat"}[i]
    print(f"  col {i} ({truth}, H*={hstars[i]}): R2 = {np.mean(r2s):.4f} +/- {np.std(r2s, ddof=1):.4f}"
          f"   RMSE = {np.mean(rm):.4f}")
print("  published Section 4.2: 0.921 +/- 0.018")

rule("4. TECATOR — gates: untrained vs trained (published Table 2: 0.215, r=0.348)")
Path("priors_tmp").mkdir(parents=True, exist_ok=True)
wl = np.linspace(850, 1050, 100)
ch = (wl >= 920) & (wl <= 930)
oh = (wl >= 965) & (wl <= 975)
for i, truth in ((0, "water (the column used)"), (1, "fat")):
    pr = anova_prior(Xtr, Ytr[:, i])
    np.save(f"priors_tmp/tec_{i}.npy", pr)
    g0, lg0 = gates_from_prior(pr)
    top = int(np.argmax(g0))
    print(f"\n  --- prior from col {i} = {truth} ---")
    print(f"  UNTRAINED : band {top} ({wl[top]:.0f} nm)  weight {g0[top]:.4f}"
          f"  corr {np.corrcoef(g0, pr)[0,1]:.4f}"
          f"  C-H {100*g0[ch].sum():.2f}%  O-H {100*g0[oh].sum():.2f}%")
    srt = np.sort(lg0)
    print(f"              logit gap 1st-2nd = {srt[-1]-srt[-2]:.2f}   range {lg0.min():.2f} to {lg0.max():.2f}")
    res = []
    for s in range(10):
        g, lg, r2, rmse, ep = train_gates(Xtr, Ytr[:, i:i+1], Xte, Yte[:, i:i+1],
                                          f"priors_tmp/tec_{i}.npy", 100, s)
        res.append((g[int(np.argmax(g))], np.corrcoef(g, pr)[0, 1],
                    100*g[ch].sum(), 100*g[oh].sum(),
                    np.abs(lg - lg0).max(), r2, rmse, ep, int(np.argmax(g))))
    a = np.array([r[:8] for r in res])
    tb = [r[8] for r in res]
    print(f"  TRAINED   : top band {set(tb)}  weight [{a[:,0].min():.4f}; {a[:,0].max():.4f}]"
          f"  corr [{a[:,1].min():.4f}; {a[:,1].max():.4f}]")
    print(f"              C-H [{a[:,2].min():.2f}; {a[:,2].max():.2f}]%   O-H [{a[:,3].min():.2f}; {a[:,3].max():.2f}]%")
    print(f"              max logit displacement [{a[:,4].min():.2f}; {a[:,4].max():.2f}]   epochs run {a[:,7].min():.0f}-{a[:,7].max():.0f}")
    print(f"              test R2 median {np.median(a[:,5]):.4f} [{a[:,5].min():.4f}; {a[:,5].max():.4f}]")

rule("5. GASOLINE — target, split, H*, gates")
Xg, Yg = load("gasoline")
print(f"  spectra {Xg.shape}  targets {Yg.shape}  "
      f"octane min {Yg.min():.2f} max {Yg.max():.2f}   (article: 83.4 to 89.6)")
Xgtr, Xgte, Ygtr, Ygte = split_scale(Xg, Yg)
print(f"  train {Xgtr.shape[0]} / test {Xgte.shape[0]}   (article: 48/12)")
hs, hm, tbl = pls_1se(Xgtr, Ygtr)
r2, rmse = pls_eval(Xgtr, Ygtr, Xgte, Ygte, hs)
print(f"  H_min {hm}  H* {hs}  ->  R2 {r2:.4f}  RMSE {rmse:.4f}   (published 0.975 / 0.145, H*=5)")
for h in (5, 10):
    a, b = pls_eval(Xgtr, Ygtr, Xgte, Ygte, h)
    print(f"  forced H={h}: R2 {a:.4f}  RMSE {b:.4f}")

prg = anova_prior(Xgtr, Ygtr[:, 0])
np.save("priors_tmp/gas.npy", prg)
g0, lg0 = gates_from_prior(prg)
top = int(np.argmax(g0))
srt = np.sort(lg0)
print(f"\n  UNTRAINED : band {top} (1-based {top+1})  weight {g0[top]:.4f}"
      f"  corr {np.corrcoef(g0, prg)[0,1]:.4f}")
print(f"              logit gap 1st-2nd = {srt[-1]-srt[-2]:.2f}")
print("  published Table 2: band 154, weight 0.130, corr 0.507")
res = []
for s in range(10):
    g, lg, r2g, rmg, ep = train_gates(Xgtr, Ygtr, Xgte, Ygte, "priors_tmp/gas.npy", 401, s)
    res.append((g[int(np.argmax(g))], np.corrcoef(g, prg)[0, 1], np.abs(lg-lg0).max(), r2g, int(np.argmax(g))))
a = np.array([r[:4] for r in res])
print(f"  TRAINED   : top band {set(r[4] for r in res)}  weight [{a[:,0].min():.4f}; {a[:,0].max():.4f}]"
      f"  corr [{a[:,1].min():.4f}; {a[:,1].max():.4f}]")
print(f"              max logit displacement [{a[:,2].min():.2f}; {a[:,2].max():.2f}]"
      f"   test R2 median {np.median(a[:,3]):.4f}")
