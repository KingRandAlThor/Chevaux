"""Client minimal pour l'API publique PMU (turfinfo) + parsing.

Données publiques utilisées avec modération (délai poli entre requêtes, User-Agent explicite).
Usage strictement personnel / pédagogique : respectez les CGU du PMU.
"""
import datetime as dt
import re
import time

import requests

BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7"
UA = {"User-Agent": "Chevaux-research-bot/1.0 (usage personnel, requetes espacees)"}
DELAY = 0.3
_last = [0.0]


def get_json(path, session=None, retries=3, delay=None):
    """GET poli (espacé) avec quelques retries. Renvoie None si 404/204."""
    s = session or requests
    wait = DELAY if delay is None else delay
    for i in range(retries):
        dt_wait = wait - (time.time() - _last[0])
        if dt_wait > 0:
            time.sleep(dt_wait)
        _last[0] = time.time()
        try:
            r = s.get(BASE + path, headers=UA, timeout=25)
        except requests.RequestException:
            time.sleep(2 * (i + 1))
            continue
        if r.status_code in (204, 404):
            return None
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        time.sleep(2 * (i + 1))
    return None


def day_str(d):
    return d.strftime("%d%m%Y")


def programme(d, **kw):
    j = get_json(f"/programme/{day_str(d)}", **kw)
    return (j or {}).get("programme", {}).get("reunions", [])


def participants(d, r, c, **kw):
    j = get_json(f"/programme/{day_str(d)}/R{r}/C{c}/participants", **kw)
    return (j or {}).get("participants", [])


def rapports(d, r, c, **kw):
    return get_json(f"/programme/{day_str(d)}/R{r}/C{c}/rapports-definitifs", **kw) or []


def musique_stats(m):
    """Résume une 'musique' (ex '1p3p0a(24)DaAa') : 5 dernières places."""
    m = m if isinstance(m, str) else ""
    pos = []
    for tok in re.findall(r"([0-9DATRdatr])[a-zA-Z]?", re.sub(r"\(\d+\)", "", m)):
        pos.append(int(tok) if tok.isdigit() else None)
    pos = pos[:5]
    vals = [p for p in pos if p is not None]
    wins = sum(1 for p in vals if p == 1)
    placed = sum(1 for p in vals if 1 <= p <= 3)
    avg = sum(min(p, 10) if p else 10 for p in vals) / len(vals) if vals else 6.0
    return {"m_n": len(pos), "m_wins": wins, "m_placed": placed, "m_avg": avg}


def odds_of(p):
    """Cote (direct, sinon référence) = gain total pour 1 € misé ; None si absente."""
    for k in ("dernierRapportDirect", "dernierRapportReference"):
        x = p.get(k) or {}
        if x.get("rapport"):
            return float(x["rapport"])
    return None


def place_dividends(rap):
    """{numPmu: dividende pour 1 €} du simple placé."""
    out = {}
    for x in rap:
        if x.get("typePari") == "SIMPLE_PLACE":
            for y in x.get("rapports", []):
                try:
                    out[int(y["combinaison"])] = y["dividendePourUnEuro"] / 100.0
                except (KeyError, ValueError, TypeError):
                    pass
    return out


def win_dividend(rap):
    for x in rap:
        if x.get("typePari") == "SIMPLE_GAGNANT":
            for y in x.get("rapports", []):
                try:
                    return int(y["combinaison"]), y["dividendePourUnEuro"] / 100.0
                except (KeyError, ValueError, TypeError):
                    pass
    return None, None


def daterange(a, b):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)
