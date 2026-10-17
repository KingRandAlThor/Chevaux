"""Backtest honnête : séparation temporelle stricte, ROI réel (rapports PMU), bootstrap par course.

python -m chevaux.backtest data/history.db [--out REPORT_DATA.json]
"""
import json
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .features import FEATURES, add_basic, add_people_stats

RNG = np.random.default_rng(12345)
LINEAR_FEATS = ["logit_p", "m_avg", "m_wins", "win_rate", "place_rate", "jockey_wr", "trainer_wr", "log_gains_pc",
                "odd_ref_ratio"]


def load(path):
    con = sqlite3.connect(path)
    df = pd.read_sql("""SELECT ru.*, ra.hippo, ra.discipline, ra.distance, ra.partants, ra.montant, ra.start_ts
        FROM runners ru JOIN races ra USING(date, r, c)""", con)
    # courses sans arrivée exploitable ou sans gagnant : exclues (incomplètes)
    ok = df.groupby(["date", "r", "c"])["finish"].transform(lambda s: (s == 1).sum() == 1)
    return df[ok].copy()


def prepare(df):
    return add_people_stats(add_basic(df))


def split_by_date(df, fr=(0.6, 0.2)):
    dates = sorted(df["date"].unique())
    a = dates[int(len(dates) * fr[0])]
    b = dates[int(len(dates) * (fr[0] + fr[1]))]
    return df[df["date"] < a], df[(df["date"] >= a) & (df["date"] < b)], df[df["date"] >= b], (a, b)


def fit_models(tr, target="win"):
    y = (tr["finish"] == 1).astype(int) if target == "win" else tr["place_div"].notna().astype(int)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
    lr.fit(tr[LINEAR_FEATS], y)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=5.0,
                                        min_samples_leaf=100, random_state=0)
    gb.fit(tr[FEATURES], y)
    return lr, gb


def renorm(df, col):
    """Normalise p à somme 1 dans la course (cas victoire)."""
    return df[col] / df.groupby(["date", "r", "c"])[col].transform("sum")


def place_div_model(tr):
    """Estime le dividende placé à partir de la cote (on ne connaît pas le rapport placé avant l'arrivée)."""
    t = tr[tr["place_div"].notna()]
    x = np.column_stack([np.log(t["odd"]), np.log(t["odd"]) ** 2, np.log(t["n_runners"])])
    y = np.log(t["place_div"])
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def est_place_div(coef, df):
    x = np.column_stack([np.ones(len(df)), np.log(df["odd"]), np.log(df["odd"]) ** 2, np.log(df["n_runners"])])
    return np.maximum(1.10, np.exp(x @ coef))


def race_ids(df):
    return df.groupby(["date", "r", "c"], sort=False).ngroup().to_numpy()


def roi_stats(df, mask, ret_col, n_boot=2000):
    """ROI à mise plate de 1 € ; IC95 et p-value par bootstrap sur les courses (clusters)."""
    sub = df[mask]
    n = len(sub)
    if n == 0:
        return {"n_bets": 0}
    rid = race_ids(sub)
    prof = sub[ret_col].to_numpy() - 1.0
    ur, inv = np.unique(rid, return_inverse=True)
    s = np.bincount(inv, weights=prof)
    c = np.bincount(inv).astype(float)
    k = len(ur)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ix = RNG.integers(0, k, k)
        boots[i] = s[ix].sum() / c[ix].sum()
    roi = prof.sum() / n
    return {"n_bets": int(n), "n_races": int(k), "roi": float(roi), "ci95": [float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5))], "p_roi_le_0": float((boots <= 0).mean()),
            "hit_rate": float((sub[ret_col] > 0).mean())}


def returns(df):
    df["ret_win"] = np.where(df["finish"] == 1, df["odd"], 0.0)
    df["ret_place"] = df["place_div"].fillna(0.0)
    return df


def pick_top(df, col, ascending=False):
    """Un partant par course : meilleur `col`."""
    r = df.groupby(["date", "r", "c"])[col].rank(method="first", ascending=ascending)
    return r == 1


def random_baseline(df, ret_col, n=300):
    """ROI d'un joueur au hasard (un cheval tiré au hasard par course) : moyenne et écart-type."""
    gid = race_ids(df)
    order = np.argsort(gid, kind="stable")
    ret = df[ret_col].to_numpy()[order]
    g = gid[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    sizes = np.diff(np.r_[starts, len(g)])
    rois = []
    for _ in range(n):
        pick = starts + (RNG.random(len(starts)) * sizes).astype(int)
        rois.append(ret[pick].mean() - 1)
    return {"roi_mean": float(np.mean(rois)), "roi_sd": float(np.std(rois)),
            "roi_all_horses": float(ret.mean() - 1)}


def calib_table(p, y, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).agg(p=("p", "mean"), y=("y", "mean"),
                                                                                 n=("y", "size"))
    return t.round(4).reset_index(drop=True).to_dict("records")


def paired_logloss_test(df, pa, pb, n_boot=1000):
    """Différence de log-loss (b - a) par course bootstrap ; >0 => a meilleur que b."""
    y = (df["finish"] == 1).to_numpy()
    la = -np.log(np.where(y, pa, 1 - pa))
    lb = -np.log(np.where(y, pb, 1 - pb))
    rid = race_ids(df)
    ur, inv = np.unique(rid, return_inverse=True)
    d = np.bincount(inv, weights=lb - la)
    c = np.bincount(inv).astype(float)
    k = len(ur)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        ix = RNG.integers(0, k, k)
        bs[i] = d[ix].sum() / c[ix].sum()
    return {"diff": float(d.sum() / c.sum()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def run(path, theta_grid=(0.0, 0.05, 0.10, 0.20, 0.30)):
    df = prepare(load(path))
    df = returns(df)
    tr, va, te, (d1, d2) = split_by_date(df)
    out = {"n_races": int(df.groupby(["date", "r", "c"]).ngroups), "n_runners": int(len(df)),
           "date_min": df["date"].min(), "date_max": df["date"].max(),
           "train": [tr["date"].min(), tr["date"].max()], "val": [va["date"].min(), va["date"].max()],
           "test": [te["date"].min(), te["date"].max()], "test_races": int(te.groupby(["date", "r", "c"]).ngroups)}

    # --- probabilité de victoire : modèles ajustés sur train uniquement ---
    lr, gb = fit_models(tr, "win")
    for d in (va, te):
        d["p_lr"] = renorm(d.assign(x=lr.predict_proba(d[LINEAR_FEATS])[:, 1]), "x")
        d["p_gb"] = renorm(d.assign(x=gb.predict_proba(d[FEATURES])[:, 1]), "x")
    yv, yt = (va["finish"] == 1).astype(int), (te["finish"] == 1).astype(int)
    out["val_logloss"] = {k: float(log_loss(yv, va[c])) for k, c in (("marche", "p_imp"), ("logreg", "p_lr"),
                                                                       ("gboost", "p_gb"))}
    out["test_logloss"] = {k: float(log_loss(yt, te[c])) for k, c in (("marche", "p_imp"), ("logreg", "p_lr"),
                                                                        ("gboost", "p_gb"))}
    out["test_brier"] = {k: float(brier_score_loss(yt, te[c])) for k, c in (("marche", "p_imp"), ("logreg", "p_lr"),
                                                                              ("gboost", "p_gb"))}
    out["test_logloss_vs_market"] = {k: paired_logloss_test(te, te[c].to_numpy(), te["p_imp"].to_numpy())
                                     for k, c in (("logreg", "p_lr"), ("gboost", "p_gb"))}
    out["calibration_test_marche"] = calib_table(te["p_imp"], yt)

    # le meilleur modèle est choisi sur VALIDATION (jamais sur le test)
    best = min(("p_lr", "p_gb"), key=lambda c: log_loss(yv, va[c]))
    out["model_choisi_sur_validation"] = best

    # --- stratégies de pari (mise plate 1 €) ---
    def block(d):
        r = {}
        r["favori_gagnant"] = roi_stats(d, pick_top(d, "p_imp"), "ret_win")
        r["favori_place"] = roi_stats(d, pick_top(d, "p_imp"), "ret_place")
        r["hasard_gagnant"] = random_baseline(d, "ret_win")
        r["hasard_place"] = random_baseline(d, "ret_place")
        r["modele_top_gagnant"] = roi_stats(d, pick_top(d, best), "ret_win")
        return r

    out["test_baselines"] = block(te)

    # seuil de value choisi sur la validation (meilleur ROI avec >= 200 paris), puis appliqué une seule fois au test
    best_vals = {}
    for kind, ret in (("gagnant", "ret_win"), ("place", "ret_place")):
        if kind == "gagnant":
            for d in (va, te):
                d["ev"] = d[best] * d["odd"]
        else:
            lr_p, gb_p = fit_models(tr, "place")
            coef = place_div_model(tr)
            for d in (va, te):
                pp = (lr_p if best == "p_lr" else gb_p).predict_proba(d[LINEAR_FEATS if best == "p_lr" else FEATURES])[:, 1]
                d["p_place"] = pp
                d["ev"] = pp * est_place_div(coef, d)
        grid = {}
        for th in theta_grid:
            m = va["ev"] > 1 + th
            grid[th] = roi_stats(va, m, ret) if m.sum() else {"n_bets": 0}
        eligible = {t: g for t, g in grid.items() if g.get("n_bets", 0) >= 200}
        th = max(eligible, key=lambda t: eligible[t]["roi"]) if eligible else max(theta_grid)
        best_vals[kind] = th
        out[f"value_{kind}"] = {"theta_choisi_sur_val": th, "val_grid": {str(k): v for k, v in grid.items()},
                                "test": roi_stats(te, te["ev"] > 1 + th, ret)}
        # aussi : value sur tous les seuils (information, NON utilisée pour choisir)
        out[f"value_{kind}"]["test_tous_seuils_info"] = {str(t): roi_stats(te, te["ev"] > 1 + t, ret)
                                                          for t in theta_grid}
    out["verdict"] = verdict(out)
    return out


def verdict(o):
    lines = []
    for k in ("gagnant", "place"):
        t = o[f"value_{k}"]["test"]
        if not t.get("n_bets"):
            lines.append(f"value {k}: aucun pari sur le test.")
            continue
        lo, hi = t["ci95"]
        if lo > 0:
            lines.append(f"value {k}: ROI {t['roi']:+.1%} avec IC95 [{lo:+.1%}, {hi:+.1%}] strictement positif.")
        else:
            lines.append(f"value {k}: ROI {t['roi']:+.1%}, IC95 [{lo:+.1%}, {hi:+.1%}] contient 0 ou est négatif : "
                         "pas d'avantage démontré.")
    return lines


if __name__ == "__main__":
    res = run(sys.argv[1])
    outp = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    txt = json.dumps(res, indent=1, ensure_ascii=False)
    if outp:
        open(outp, "w", encoding="utf-8").write(txt)
    print(txt)
