"""Features par partant, utilisées à l'identique par le backtest et par le bot.

Règle anti-fuite : les statistiques jockey/entraîneur sont calculées uniquement avec des courses
STRICTEMENT antérieures (par date) à celle du partant. La cote utilisée est la cote au moment du pari.
"""
import numpy as np
import pandas as pd

from .pmu import musique_stats

FEATURES = ["logit_p", "log_odd", "odd_rank", "n_runners", "distance", "age", "is_male", "m_n", "m_wins",
            "m_placed", "m_avg", "win_rate", "place_rate", "log_gains_pc", "log_nb", "jockey_wr", "trainer_wr",
            "jockey_n", "trainer_n", "poids", "oeil", "disc_code", "log_montant", "odd_ref_ratio"]
DISC = {"ATTELE": 0, "MONTE": 1, "PLAT": 2, "HAIE": 3, "STEEPLECHASE": 4, "CROSS": 5}
PRIOR_N = 20.0


def add_basic(df):
    """df : une ligne par partant avec colonnes de la table runners + discipline/distance/partants/montant."""
    df = df.copy()
    df = df[df["odd"].notna() & (df["odd"] > 1.0)].copy()
    g = df.groupby(["date", "r", "c"])
    inv = 1.0 / df["odd"]
    df["p_imp"] = inv / inv.groupby([df["date"], df["r"], df["c"]]).transform("sum")
    pc = df["p_imp"].clip(1e-4, 1 - 1e-4)
    df["logit_p"] = np.log(pc / (1 - pc))
    df["log_odd"] = np.log(df["odd"])
    df["odd_rank"] = g["odd"].rank(method="min")
    df["n_runners"] = g["num"].transform("count")
    ms = df["musique"].map(musique_stats).apply(pd.Series)
    df = pd.concat([df, ms], axis=1)
    nb = df["nb_courses"].fillna(0).clip(lower=0)
    df["win_rate"] = (df["nb_wins"].fillna(0) + 0.5) / (nb + 10)
    df["place_rate"] = (df["nb_places"].fillna(0) + df["nb_wins"].fillna(0) + 1.0) / (nb + 10)
    df["log_gains_pc"] = np.log1p(df["gains"].fillna(0) / (nb + 1))
    df["log_nb"] = np.log1p(nb)
    df["is_male"] = df["sexe"].isin(["MALES", "HONGRES"]).astype(float)
    df["oeil"] = df["deferre"].fillna("").str.contains("OEILLERES") & ~df["deferre"].fillna("").str.contains("SANS")
    df["oeil"] = df["oeil"].astype(float)
    df["poids"] = df["poids"].fillna(0)
    df["disc_code"] = df["discipline"].map(DISC).fillna(-1)
    df["log_montant"] = np.log1p(df["montant"].fillna(0))
    df["odd_ref_ratio"] = (df["odd_ref"].fillna(df["odd"]) / df["odd"]).clip(0.2, 5)
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def add_people_stats(df, start_stats=None):
    """Taux de victoire jockey/entraîneur avec lissage, uniquement sur le passé strict (jours antérieurs)."""
    df = df.sort_values(["date", "r", "c", "num"]).copy()
    df["win"] = (df["finish"] == 1).astype(float)
    base = float((1.0 / df["n_runners"]).mean()) if len(df) else 0.1  # indépendant des résultats
    for who, col in (("jockey", "jockey"), ("trainer", "trainer")):
        wr = np.zeros(len(df))
        cn = np.zeros(len(df))
        wins, cnt = {}, {}
        idx = 0
        for date, day in df.groupby("date", sort=True):
            keys = day[col].fillna("?").tolist()
            for j, k in enumerate(keys):
                n = cnt.get(k, 0)
                wr[idx + j] = (wins.get(k, 0) + PRIOR_N * base) / (n + PRIOR_N)
                cn[idx + j] = n
            for k, w in zip(keys, day["win"].tolist()):
                cnt[k] = cnt.get(k, 0) + 1
                wins[k] = wins.get(k, 0) + w
            idx += len(day)
        df[f"{who}_wr"] = wr
        df[f"{who}_n"] = cn
    return df


def people_tables(df):
    """Compteurs (victoires, courses) par jockey/entraîneur sur tout `df` (pour le bot en direct)."""
    t = {"base": float((df["finish"] == 1).mean())}
    for who in ("jockey", "trainer"):
        g = df.assign(w=(df["finish"] == 1).astype(float)).groupby(df[who].fillna("?"))["w"].agg(["sum", "count"])
        t[who] = {k: (float(a), int(b)) for k, (a, b) in g.iterrows()}
    return t


def apply_people(df, t):
    df = df.copy()
    for who in ("jockey", "trainer"):
        wr, cn = [], []
        for k in df[who].fillna("?"):
            w, n = t[who].get(k, (0.0, 0))
            wr.append((w + PRIOR_N * t["base"]) / (n + PRIOR_N))
            cn.append(n)
        df[f"{who}_wr"], df[f"{who}_n"] = wr, cn
    return df
