"""Modèle de production : entraîné sur tout l'historique collecté, sauvegardé dans data/model.joblib."""
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import LINEAR_FEATS, load, place_div_model, prepare, est_place_div
from .features import FEATURES, add_basic, apply_people, people_tables

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.joblib"
# Coefficients de repli (log place_div ~ 1 + log odd + log odd^2 + log n) si aucun modèle n'est disponible
DEFAULT_COEF = np.array([0.17, 0.30, 0.0, 0.0])


def train_bundle(history_path, out=MODEL_PATH):
    df = prepare(load(history_path))
    y = (df["finish"] == 1).astype(int)
    yp = df["place_div"].notna().astype(int)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)).fit(df[LINEAR_FEATS], y)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=5.0,
                                        min_samples_leaf=100, random_state=0).fit(df[FEATURES], y)
    gbp = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=5.0,
                                         min_samples_leaf=100, random_state=0).fit(df[FEATURES], yp)
    bundle = {"lr": lr, "gb": gb, "gb_place": gbp, "coef": place_div_model(df), "people": people_tables(df),
              "n_races": int(df.groupby(["date", "r", "c"]).ngroups), "trained": dt.date.today().isoformat()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    return bundle


def load_bundle(path=MODEL_PATH):
    p = Path(path)
    return joblib.load(p) if p.exists() else None


def predict(df, bundle=None):
    """df : partants à venir (colonnes de la table runners + discipline, distance, montant, finish=NaN).
    Ajoute p_win (somme 1 par course), p_place, place_odd."""
    df = df.copy()
    df["finish"] = np.nan
    df = add_basic(df)
    if df.empty:
        return df
    key = [df["date"], df["r"], df["c"]]
    if bundle is None:
        df["p_win"] = df["p_imp"]
        df["p_place"] = (1 - (1 - df["p_imp"]) ** 3).clip(upper=0.97)
        df["place_odd"] = est_place_div(DEFAULT_COEF, df)
        return df
    df = apply_people(df, bundle["people"])
    pw = 0.5 * bundle["lr"].predict_proba(df[LINEAR_FEATS])[:, 1] + 0.5 * bundle["gb"].predict_proba(df[FEATURES])[:, 1]
    df["p_win"] = pw / pd.Series(pw, index=df.index).groupby(key).transform("sum")
    df["p_place"] = bundle["gb_place"].predict_proba(df[FEATURES])[:, 1]
    df["place_odd"] = est_place_div(bundle["coef"], df)
    return df
