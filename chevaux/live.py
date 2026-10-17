"""Données en direct : courses françaises à venir, et résultats pour le règlement des paris."""
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from . import pmu

TZ = ZoneInfo("Europe/Paris")


def runner_row(date, r, c, p):
    g = p.get("gainsParticipant") or {}
    return {"date": date, "r": r, "c": c, "num": p.get("numPmu"), "name": p.get("nom"), "age": p.get("age"),
            "sexe": p.get("sexe"), "jockey": p.get("driver"), "trainer": p.get("entraineur"),
            "musique": p.get("musique"), "nb_courses": p.get("nombreCourses"), "nb_wins": p.get("nombreVictoires"),
            "nb_places": p.get("nombrePlaces"), "gains": g.get("gainsCarriere"), "odd": pmu.odds_of(p),
            "odd_ref": (p.get("dernierRapportReference") or {}).get("rapport"), "finish": p.get("ordreArrivee"),
            "place_div": None, "deferre": p.get("oeilleres"), "poids": p.get("handicapPoids")}


def upcoming(now, min_minutes=5, max_hours=14):
    """Partants des courses françaises pas encore parties (départ entre now+min et now+max_hours)."""
    d = now.astimezone(TZ).date()
    rows, meta = [], {}
    for reu in pmu.programme(d):
        if (reu.get("pays") or {}).get("code") != "FRA":
            continue
        hippo = (reu.get("hippodrome") or {}).get("libelleCourt")
        for c in reu.get("courses", []):
            ts = (c.get("heureDepart") or 0) / 1000.0
            st = str(c.get("statut", ""))
            if st.startswith(("ARRIVEE", "ANNUL")) or ts < now.timestamp() + min_minutes * 60 \
                    or ts > now.timestamp() + max_hours * 3600:
                continue
            r, cn = reu["numOfficiel"], c["numOrdre"]
            parts = [p for p in pmu.participants(d, r, cn) if p.get("statut") == "PARTANT"]
            if len(parts) < 4:
                continue
            for p in parts:
                row = runner_row(d.isoformat(), r, cn, p)
                row.update(discipline=c.get("discipline"), distance=c.get("distance"),
                           montant=c.get("montantPrix"), hippo=hippo)
                rows.append(row)
            meta[(d.isoformat(), r, cn)] = {"ts": ts, "label": f"{hippo} R{r}C{cn}", "discipline": c.get("discipline"),
                                            "distance": c.get("distance")}
    return pd.DataFrame(rows), meta


def race_result(date, r, c):
    """({num: finish}, {num: place_div}, (win_num, win_div)) ou None si la course n'est pas terminée."""
    d = dt.date.fromisoformat(date) if isinstance(date, str) else date
    parts = pmu.participants(d, r, c)
    fin = {p["numPmu"]: p.get("ordreArrivee") for p in parts if p.get("ordreArrivee")}
    rap = pmu.rapports(d, r, c)
    wn, wd = pmu.win_dividend(rap)
    if not fin or wn is None:
        return None
    return fin, pmu.place_dividends(rap), (wn, wd)
