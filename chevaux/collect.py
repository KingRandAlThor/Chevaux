"""Collecte de l'historique PMU (courses françaises) dans une base sqlite, reprenable.

python -m chevaux.collect 2025-10-01 2026-10-16 [history.db]
"""
import datetime as dt
import sqlite3
import sys

from . import pmu

SCHEMA = """
CREATE TABLE IF NOT EXISTS races(date TEXT, r INTEGER, c INTEGER, hippo TEXT, discipline TEXT, distance INTEGER,
  start_ts REAL, partants INTEGER, penetro TEXT, montant REAL, win_num INTEGER, win_div REAL,
  PRIMARY KEY(date, r, c));
CREATE TABLE IF NOT EXISTS runners(date TEXT, r INTEGER, c INTEGER, num INTEGER, name TEXT, age INTEGER, sexe TEXT,
  jockey TEXT, trainer TEXT, musique TEXT, nb_courses INTEGER, nb_wins INTEGER, nb_places INTEGER, gains REAL,
  odd REAL, odd_ref REAL, finish INTEGER, place_div REAL, deferre TEXT, poids REAL,
  PRIMARY KEY(date, r, c, num));
CREATE TABLE IF NOT EXISTS done_days(date TEXT PRIMARY KEY);
"""


def connect(path):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def collect_day(con, d):
    ds = d.isoformat()
    n = 0
    for reu in pmu.programme(d):
        if (reu.get("pays") or {}).get("code") != "FRA":
            continue
        hippo = (reu.get("hippodrome") or {}).get("libelleCourt")
        for c in reu.get("courses", []):
            if not str(c.get("statut", "")).startswith("ARRIVEE"):
                continue
            r, cn = reu["numOfficiel"], c["numOrdre"]
            if con.execute("SELECT 1 FROM races WHERE date=? AND r=? AND c=?", (ds, r, cn)).fetchone():
                continue
            parts = pmu.participants(d, r, cn)
            rap = pmu.rapports(d, r, cn)
            if not parts:
                continue
            wn, wd = pmu.win_dividend(rap)
            pdv = pmu.place_dividends(rap)
            pen = c.get("penetrometre")
            pen = pen.get("valeurMesure") if isinstance(pen, dict) else None
            con.execute("INSERT OR REPLACE INTO races VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ds, r, cn, hippo, c.get("discipline"), c.get("distance"), c.get("heureDepart"),
                         c.get("nombreDeclaresPartants"), pen, c.get("montantPrix"), wn, wd))
            for p in parts:
                if p.get("statut") != "PARTANT":
                    continue
                g = p.get("gainsParticipant") or {}
                num = p.get("numPmu")
                con.execute("INSERT OR REPLACE INTO runners VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (ds, r, cn, num, p.get("nom"), p.get("age"), p.get("sexe"),
                             p.get("driver"), p.get("entraineur"), p.get("musique"), p.get("nombreCourses"),
                             p.get("nombreVictoires"), p.get("nombrePlaces"), g.get("gainsCarriere"),
                             pmu.odds_of(p), (p.get("dernierRapportReference") or {}).get("rapport"),
                             p.get("ordreArrivee"), pdv.get(num), p.get("oeilleres"), p.get("handicapPoids")))
            n += 1
    con.execute("INSERT OR REPLACE INTO done_days VALUES(?)", (ds,))
    con.commit()
    return n


def merge(out, *ins):
    con = connect(out)
    for i, p in enumerate(ins):
        con.execute(f"ATTACH DATABASE ? AS s{i}", (p,))
        for t in ("races", "runners", "done_days"):
            con.execute(f"INSERT OR REPLACE INTO {t} SELECT * FROM s{i}.{t}")
        con.commit()
        con.execute(f"DETACH DATABASE s{i}")
    return con


def main(argv):
    if argv and argv[0] == "merge":
        merge(argv[1], *argv[2:])
        return
    a = dt.date.fromisoformat(argv[0])
    b = dt.date.fromisoformat(argv[1])
    path = argv[2] if len(argv) > 2 else "history.db"
    con = connect(path)
    done = {r[0] for r in con.execute("SELECT date FROM done_days")}
    for d in pmu.daterange(a, b):
        if d.isoformat() in done:
            continue
        print(d, collect_day(con, d), "courses", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
