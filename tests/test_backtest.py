import sqlite3

import numpy as np
import pandas as pd

from chevaux import backtest, collect
from chevaux.features import add_basic, add_people_stats


def synth_db(path, n_days=60, per_day=12, seed=0, edge=False):
    """Marché synthétique EFFICIENT : la cote reflète la vraie proba, moins 16 % de prélèvement.
    Si edge=True, une feature cachée (musique) porte une vraie information que le marché ignore."""
    rng = np.random.default_rng(seed)
    con = collect.connect(path)
    for d in range(n_days):
        date = f"2026-01-{d % 28 + 1:02d}" if d < 28 else f"2026-{2 + (d - 28) // 28:02d}-{(d - 28) % 28 + 1:02d}"
        for r in range(per_day):
            k = int(rng.integers(8, 14))
            skill = rng.normal(size=k)
            true = np.exp(skill) / np.exp(skill).sum()
            market = true if not edge else np.exp(skill * 0.5) / np.exp(skill * 0.5).sum()
            win = rng.choice(k, p=true)
            order = np.argsort(-(skill + rng.gumbel(size=k)))
            order[0] = win if False else order[0]
            winner = order[0]
            odds = np.maximum(1.1, 0.84 / market)
            con.execute("INSERT INTO races VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (date, 1, r + 1, "X", "PLAT", 2000, 0, k, None, 20000, int(winner) + 1, float(odds[winner])))
            for j in range(k):
                fin = int(np.where(order == j)[0][0]) + 1
                m = "1a1a1a1a1a" if (edge and skill[j] > 0.5) else "5a5a5a5a5a"
                pdiv = 1.5 if fin <= 3 else None
                con.execute("INSERT INTO runners VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (date, 1, r + 1, j + 1, f"H{j}", 4, "MALES", f"J{j % 5}", f"T{j % 4}", m, 10, 1, 3, 1000.0,
                             float(odds[j]), float(odds[j]), fin, pdiv, "SANS_OEILLERES", 0))
    con.commit()
    con.close()


def test_backtest_runs_and_no_edge_on_efficient_market(tmp_path):
    p = str(tmp_path / "s.db")
    synth_db(p, n_days=56, per_day=15)
    res = backtest.run(p)
    assert res["train"][1] < res["val"][0] <= res["val"][1] < res["test"][0]  # séparation temporelle stricte
    fav = res["test_baselines"]["favori_gagnant"]
    assert fav["roi"] < 0.1  # le prélèvement empêche tout ROI positif systématique
    v = res["value_gagnant"]["test"]
    assert v.get("n_bets", 0) == 0 or v["ci95"][0] < 0.25


def test_people_stats_use_only_past():
    rows = []
    for d, w in (("2026-01-01", 1), ("2026-01-02", 0), ("2026-01-03", 1)):
        for num in (1, 2):
            rows.append({"date": d, "r": 1, "c": 1, "num": num, "jockey": "J", "trainer": "T",
                         "finish": 1 if (num == 1) == bool(w) else 2, "odd": 2.0, "musique": "1a", "nb_courses": 1,
                         "nb_wins": 0, "nb_places": 0, "gains": 0, "sexe": "MALES", "deferre": "", "odd_ref": 2.0,
                         "poids": 0, "discipline": "PLAT", "montant": 1, "age": 4, "distance": 1000})
    df = add_people_stats(add_basic(pd.DataFrame(rows)))
    first = df[df["date"] == "2026-01-01"]
    assert (first["jockey_n"] == 0).all()  # rien de connu le 1er jour
    last = df[df["date"] == "2026-01-03"]
    assert (last["jockey_n"] == 4).all()  # uniquement les 2 jours passés (2 partants chacun)
    # modifier le résultat du dernier jour ne change pas les stats du dernier jour
    df2 = df.copy()
    df2.loc[df2["date"] == "2026-01-03", "finish"] = 9
    df2 = add_people_stats(df2.drop(columns=["jockey_wr", "jockey_n", "trainer_wr", "trainer_n"]))
    assert (df2[df2["date"] == "2026-01-03"]["jockey_wr"].values == last["jockey_wr"].values).all()
