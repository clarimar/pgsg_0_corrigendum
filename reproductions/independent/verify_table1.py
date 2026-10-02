"""Full corrected Table 1 for Tecator, over ten seeds, both target columns."""
import os, sys, warnings
from pathlib import Path

PGSG = Path(os.environ.get("PGSG_REPO", Path.cwd()))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, str(PGSG / "src"))
from models.band_select_net_light import create_model_light
from models.band_select_net import create_model

X = pd.read_csv(PGSG / "data/raw/tecator/spectra.csv").values
Y = pd.read_csv(PGSG / "data/raw/tecator/targets.csv").values
Xtr_r, Xte_r, Ytr_r, Yte_r = train_test_split(X, Y, test_size=0.2, random_state=42)
sx, sy = StandardScaler(), StandardScaler()
Xtr, Xte = sx.fit_transform(Xtr_r), sx.transform(Xte_r)
Ytr, Yte = sy.fit_transform(Ytr_r), sy.transform(Yte_r)


def priors(col):
    y = Ytr[:, col]
    f, _ = f_regression(Xtr, y)
    anova = (f - f.min()) / (f.max() - f.min())
    pls = PLSRegression(n_components=10).fit(Xtr, y)
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1); tot = s.sum()
    vip = np.array([np.sqrt(p * (s.T @ np.array(
        [(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])) / tot)[0]
        for i in range(p)])
    vip = (vip - vip.min()) / (vip.max() - vip.min())
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               random_state=42, n_jobs=-1).fit(Xtr, y)
    imp = rf.feature_importances_
    rfn = (imp - imp.min()) / (imp.max() - imp.min())
    return {"anova": anova, "vip": vip, "rf": rfn}


def run(mk, Xa, ya, Xb, yb, seed, epochs=200, lr=0.01, klw=0.001, pat=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = mk()
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xa_t, ya_t = torch.FloatTensor(Xa), torch.FloatTensor(np.ravel(ya))
    Xb_t, yb_t = torch.FloatTensor(Xb), torch.FloatTensor(np.ravel(yb))
    best, bad = float("inf"), 0
    for ep in range(epochs):
        model.train()
        pred, _ = model(Xa_t)
        loss = crit(pred.squeeze(), ya_t) + klw * model.get_kl_loss()
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pb, _ = model(Xb_t)
            r = torch.sqrt(crit(pb.squeeze(), yb_t)).item()
        if r < best: best, bad = r, 0
        else:
            bad += 1
            if bad >= pat: break
    model.eval()
    with torch.no_grad():
        pb, _ = model(Xb_t)
        rmse = torch.sqrt(crit(pb.squeeze(), yb_t)).item()
        ssr = ((yb_t - pb.squeeze()) ** 2).sum().item()
        sst = ((yb_t - yb_t.mean()) ** 2).sum().item()
    return 1 - ssr / sst, rmse


def band(vals):
    a = np.array(vals)
    return f"{np.median(a):.3f} [{a.min():.2f}; {a.max():.2f}]"


def pls_1se(Xa, ya, mx=15):
    loo, n = LeaveOneOut(), len(Xa); tbl = {}
    for h in range(1, mx + 1):
        e = np.empty(n)
        for tr, te in loo.split(Xa):
            m = PLSRegression(n_components=h).fit(Xa[tr], ya[tr])
            e[te] = (m.predict(Xa[te]).ravel() - np.ravel(ya[te])) ** 2
        rm = np.sqrt(e.mean()); tbl[h] = (rm, (e.std(ddof=1) / np.sqrt(n)) / (2 * rm))
    hmin = min(tbl, key=lambda h: tbl[h][0])
    thr = tbl[hmin][0] + tbl[hmin][1]
    return min(h for h in tbl if tbl[h][0] <= thr)


for col, label in ((0, "col 0 = water (as published)"), (1, "col 1 = fat (correct)")):
    print("\n" + "=" * 78)
    print(f" CORRECTED TABLE 1 — {label}")
    print("=" * 78)
    P = priors(col)
    Path("priors_tmp").mkdir(parents=True, exist_ok=True)
    for k, v in P.items():
        np.save(f"priors_tmp/t{col}_{k}.npy", v)
    ytr, yte = Ytr[:, col:col+1], Yte[:, col:col+1]

    h = pls_1se(Xtr, ytr)
    m = PLSRegression(n_components=h).fit(Xtr, ytr)
    p = m.predict(Xte).ravel(); yt = yte.ravel()
    r2 = 1 - ((yt-p)**2).sum() / ((yt-yt.mean())**2).sum()
    print(f"  PLS (single test, H*={h}):  R2 {r2:.4f}   RMSE {np.sqrt(((yt-p)**2).mean()):.4f}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(Xtr))
    a = []
    for tr, va in folds:
        mm = PLSRegression(n_components=h).fit(Xtr[tr], ytr[tr])
        pp = mm.predict(Xtr[va]).ravel(); yy = ytr[va].ravel()
        a.append(1 - ((yy-pp)**2).sum() / ((yy-yy.mean())**2).sum())
    print(f"  PLS 5-fold:                 R2 {np.mean(a):.4f} +/- {np.std(a, ddof=0):.4f} (ddof=0)"
          f" / {np.std(a, ddof=1):.4f} (ddof=1)")

    for pname in ("anova", "vip", "rf"):
        rs, ms = [], []
        for s in range(10):
            r, rm = run(lambda: create_model_light(100, 1, f"priors_tmp/t{col}_{pname}.npy"),
                        Xtr, ytr, Xte, yte, s)
            rs.append(r); ms.append(rm)
        print(f"  Ours ({pname.upper():<5}) single test: R2 {band(rs)}   RMSE {band(ms)}")

    rs, ms = [], []
    for s in range(10):
        r, rm = run(lambda: create_model_light(100, 1, None), Xtr, ytr, Xte, yte, s)
        rs.append(r); ms.append(rm)
    print(f"  Random init  single test:   R2 {band(rs)}   RMSE {band(ms)}")

    rs, ms = [], []
    for s in range(10):
        r, rm = run(lambda: create_model(100, 1, f"priors_tmp/t{col}_anova.npy"),
                    Xtr, ytr, Xte, yte, s)
        rs.append(r); ms.append(rm)
    print(f"  Plain CNN    single test:   R2 {band(rs)}   RMSE {band(ms)}")

    allf = []
    for s in range(10):
        fr = []
        for tr, va in folds:
            r, _ = run(lambda: create_model_light(100, 1, f"priors_tmp/t{col}_anova.npy"),
                       Xtr[tr], ytr[tr], Xtr[va], ytr[va], s)
            fr.append(r)
        allf.append(np.mean(fr))
    print(f"  Ours (ANOVA) 5-fold:        R2 {band(allf)}")
