"""Bot Telegram Chevaux : propose des paris hippiques (PMU) sur un solde SIMULÉ, ne place JAMAIS de pari.

Usage : python -m chevaux.bot daemon        (écoute Telegram + cycles, pour GitHub Actions)
        python -m chevaux.bot once [--dry]  (un cycle)
        python -m chevaux.bot demo          (démo hors-ligne avec courses synthétiques, rien n'est envoyé)
Variables d'environnement : TELEGRAM_TOKEN, TELEGRAM_CHAT (secrets GitHub), BANKROLL, DAEMON_MINUTES, ...
"""
import datetime as dt
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from . import live, model
from .multi import build_multi, closest_multi, format_multi

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "bot.db"
TZ = ZoneInfo("Europe/Paris")
MAX_STAKE_PCT = 0.06
DRY = "--dry" in sys.argv or "demo" in sys.argv[1:2]


def load_cfg():
    cfg = {"TELEGRAM_TOKEN": "", "TELEGRAM_CHAT": "", "BANKROLL": "10", "MISSION_FROM": "12", "MISSION_TO": "14",
           "MISSION_GRACE_MIN": "10", "MISSION_PROPOSE_FROM": "9", "LISTEN_SECONDS": "25", "CYCLE_MIN": "20",
           "DAEMON_MINUTES": "345", "MIN_EDGE": "0.10", "KELLY_FRACTION": "0.25", "FORCED_STAKE_PCT": "0.02",
           "MULTI_STAKE_PCT": "0.03", "REVIVE": "0"}
    for k in cfg:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


CFG = load_cfg()
OUTBOX = []  # messages « envoyés » en mode DRY (utile pour les tests)


# ---------- base ----------
def db(path=None):
    con = sqlite3.connect(path or DB)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS bets(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, label TEXT, legs TEXT,
        odd REAL, stake REAL, status TEXT, ts REAL, placed_ts REAL, payout REAL);
    """)
    return con


def meta_get(con, k, default=None):
    r = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def meta_set(con, k, v):
    con.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))


# ---------- Telegram ----------
def tg(method, **data):
    r = requests.post(f"https://api.telegram.org/bot{CFG['TELEGRAM_TOKEN']}/{method}", json=data, timeout=40)
    r.raise_for_status()
    return r.json()["result"]


def tg_send(text, buttons=None):
    if DRY or not (CFG["TELEGRAM_TOKEN"] and CFG["TELEGRAM_CHAT"]):
        OUTBOX.append(text)
        if DRY:
            print("--- Telegram (dry) ---\n" + text + ("\n[boutons: %s]" % [b[0] for b in buttons] if buttons else ""))
        return False
    data = {"chat_id": CFG["TELEGRAM_CHAT"], "text": text}
    if buttons:
        data["reply_markup"] = {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in buttons]]}
    tg("sendMessage", **data)
    return True


# ---------- solde / mort ----------
def bankroll(con):
    b = float(meta_get(con, "bankroll", CFG["BANKROLL"]))
    return b, b <= 0


def set_bankroll(con, b):
    meta_set(con, "bankroll", round(b, 2))
    if b <= 0 and meta_get(con, "dead_notified") != "1":
        meta_set(con, "dead_notified", 1)
        tg_send("💀 Ton solde est à 0 : tu es mort. Fin de la partie. Envoie « solde 10 » pour recommencer.")
    elif b > 0:
        meta_set(con, "dead_notified", 0)


def die(con, reason):
    meta_set(con, "bankroll", 0)
    meta_set(con, "dead_notified", 1)
    tg_send(reason + "\nFin de la partie. Envoie « solde 10 » pour recommencer.")


def stats_line(con):
    r = con.execute("SELECT COUNT(*), COALESCE(SUM(status='won'),0), COALESCE(SUM(status='lost'),0), "
                    "COALESCE(SUM(CASE WHEN status='placed' THEN stake END),0) FROM bets "
                    "WHERE status IN ('placed','won','lost')").fetchone()
    return f"{r[0]} paris ({r[1]} gagnés, {r[2]} perdus, {r[3]:.2f} € en jeu)"


# ---------- courses / candidats ----------
def get_upcoming(now):
    """Renvoie (liste de courses au format multi.py). Point d'injection pour les tests."""
    df, meta = live.upcoming(now)
    return races_from_df(df, meta)


def races_from_df(df, meta, bundle="auto"):
    if df is None or df.empty:
        return []
    if bundle == "auto":
        bundle = model.load_bundle()
    pr = model.predict(df, bundle)
    races = []
    for (d, r, c), g in pr.groupby(["date", "r", "c"]):
        m = meta.get((d, r, c), {})
        races.append({"id": f"{d}|{r}|{c}", "ts": m.get("ts", 0), "label": m.get("label", f"R{r}C{c}"),
                      "runners": [{"num": int(x.num), "name": x.name_, "odd": float(x.odd), "p_win": float(x.p_win),
                                   "p_place": float(x.p_place), "place_odd": float(x.place_odd)}
                                  for x in g.rename(columns={"name": "name_"}).itertuples()]})
    return sorted(races, key=lambda x: x["ts"])


def single_candidates(races):
    out = []
    for rc in races:
        for rn in rc["runners"]:
            for kind, odd, p in (("gagnant", rn["odd"], rn["p_win"]), ("place", rn["place_odd"], rn["p_place"])):
                out.append({"race": rc["id"], "ts": rc["ts"], "label": rc["label"], "num": rn["num"],
                            "name": rn["name"], "kind": kind, "odd": odd, "p": p, "ev": p * odd})
    return out


def kelly_stake(bank, p, odd):
    b = odd - 1.0
    f = max(0.0, (p * odd - 1.0) / b) if b > 0 else 0.0
    return min(bank * MAX_STAKE_PCT, bank * f * float(CFG["KELLY_FRACTION"]))


def new_bet(con, kind, label, legs, odd, stake):
    cur = con.execute("INSERT INTO bets(kind,label,legs,odd,stake,status,ts) VALUES(?,?,?,?,?,'proposed',?)",
                      (kind, label, json.dumps(legs, ensure_ascii=False), odd, round(stake, 2), time.time()))
    con.commit()
    return cur.lastrowid


def send_proposal(con, bid, header, body):
    b = con.execute("SELECT stake FROM bets WHERE id=?", (bid,)).fetchone()
    tg_send(f"{header}\n\n{body}\n\nMise suggérée : {b[0]:.2f} € (solde simulé {bankroll(con)[0]:.2f} €)\n"
            "⚠️ Je ne parie pas : c'est à toi de jouer sur le PMU si tu veux.",
            [("✅ Placé", f"p:{bid}"), ("❌ Ignorer", f"i:{bid}")])


def leg_txt(lg):
    return f"n°{lg['num']} {lg['name']} ({lg['kind']} @ {lg['odd']:.2f})"


def propose_single(con, now, races, forced=False):
    bank, dead = bankroll(con)
    if dead:
        return None
    cands = single_candidates(races)
    if not cands:
        tg_send("Aucune course exploitable pour l'instant.")
        return None
    best = max(cands, key=lambda c: c["ev"])
    value = best["ev"] > 1 + float(CFG["MIN_EDGE"])
    if value:
        stake = max(0.10, kelly_stake(bank, best["p"], best["odd"]))
        head = f"🐎 Pari « value » : le modèle voit {best['ev'] - 1:+.0%} d'espérance (estimation, non garantie)"
    else:
        # pas d'avantage détecté : pari de mission le moins risqué (placé sur le plus probable), petite mise
        pl = [c for c in cands if c["kind"] == "place"]
        best = max(pl or cands, key=lambda c: c["p"])
        stake = max(0.10, bank * float(CFG["FORCED_STAKE_PCT"]))
        head = ("🐎 Pari de mission (aucun avantage statistique détecté, c'est le moins risqué) — "
                f"espérance estimée {best['ev'] - 1:+.0%}")
    stake = min(stake, bank * MAX_STAKE_PCT)
    t = dt.datetime.fromtimestamp(best["ts"], TZ).strftime("%H:%M")
    bid = new_bet(con, "simple", f"{best['label']} {t}", [best], best["odd"], stake)
    send_proposal(con, bid, head, f"{t} {best['label']} — {leg_txt(best)}\nProba estimée ≈ {best['p']:.0%}")
    return bid


def propose_multi(con, now, races, n, target):
    bank, dead = bankroll(con)
    if dead:
        return "💀 Je suis mort : plus de paris."
    if not (2 <= n <= 8) or target < 1.05 or target > 200:
        return "Syntaxe : « multi 4 2 » = 4 courses, cote cumulée ≈ 2 (2 à 8 courses, cote entre 1,05 et 200)."
    m = build_multi(races, n, target) or build_multi(races, n, target, tol=0.2)
    approx = False
    if not m:
        m = closest_multi(races, n, target)
        approx = True
    if not m:
        return (f"Je ne trouve pas de multi à {n} courses avec une cote ≈ {target:g} parmi les courses à venir "
                "(essaie moins de courses ou une autre cote).")
    stake = min(bank * float(CFG["MULTI_STAKE_PCT"]), bank * MAX_STAKE_PCT)
    stake = max(0.10, round(stake, 2))
    bid = new_bet(con, "multi", f"Multi {n} (≈{m['odd']:.2f})", m["legs"], m["odd"], stake)
    send_proposal(con, bid, f"🎯 Multi {n} courses, cote visée {target:g}"
                  + (f" — impossible exactement, approximation à {m['odd']:.2f}" if approx else ""),
                  format_multi(m, now.timestamp(), TZ) + "\n(cotes placé estimées, rapport réel connu après la course)")
    return None


# ---------- règlement ----------
def settle_bets(con, result_fn=live.race_result):
    for bid, kind, legs, stake in con.execute("SELECT id, kind, legs, stake FROM bets WHERE status='placed'").fetchall():
        legs = json.loads(legs)
        mult, pending, lost = 1.0, False, False
        for lg in legs:
            if "div" in lg:
                mult *= lg["div"]
                continue
            d, r, c = lg["race"].split("|")
            res = result_fn(d, int(r), int(c))
            if res is None:
                pending = True
                continue
            fin, pdiv, (wn, wd) = res
            if lg["kind"] == "gagnant":
                div = wd if fin.get(lg["num"]) == 1 else 0.0
            else:
                div = pdiv.get(lg["num"], 0.0)
            lg["div"] = div
            mult *= div
        if any(lg.get("div") == 0.0 for lg in legs):
            lost = True
        if not lost and pending:
            con.execute("UPDATE bets SET legs=? WHERE id=?", (json.dumps(legs, ensure_ascii=False), bid))
            continue
        pay = 0.0 if lost else round(stake * mult, 2)
        con.execute("UPDATE bets SET status=?, payout=?, legs=? WHERE id=?",
                    ("won" if pay > 0 else "lost", pay, json.dumps(legs, ensure_ascii=False), bid))
        b = bankroll(con)[0]
        set_bankroll(con, b + pay)
        tg_send(("🏆 Gagné" if pay > 0 else "😞 Perdu") + f" : {con.execute('SELECT label FROM bets WHERE id=?', (bid,)).fetchone()[0]}"
                f" — mise {stake:.2f} €, retour {pay:.2f} €. Solde : {bankroll(con)[0]:.2f} €")
    con.commit()


# ---------- mission ----------
def win_bounds(day):
    s = dt.datetime.combine(day, dt.time(int(CFG["MISSION_FROM"])), TZ).timestamp()
    e = dt.datetime.combine(day, dt.time(int(CFG["MISSION_TO"])), TZ).timestamp()
    return s, e


def mission_done(con, day):
    s, e = win_bounds(day)
    return con.execute("SELECT 1 FROM bets WHERE status IN ('placed','won','lost') AND placed_ts>=? AND placed_ts<?",
                       (s, e)).fetchone() is not None


def mission_step(con, now, races_fn=get_upcoming):
    """Avant 12h-14h : proposer un pari dès MISSION_PROPOSE_FROM ; fin de fenêtre sans pari placé : mort."""
    bank, dead = bankroll(con)
    if dead:
        return
    ts = now.timestamp()
    day = now.astimezone(TZ).date()
    s, e = win_bounds(day)
    key = day.isoformat()
    first_hour = dt.datetime.combine(day, dt.time(int(CFG["MISSION_PROPOSE_FROM"])), TZ).timestamp()
    if mission_done(con, day):
        return
    pending = con.execute("SELECT 1 FROM bets WHERE status='proposed' AND ts>=?", (first_hour,)).fetchone()
    if first_hour <= ts < e and not pending and meta_get(con, "mission_proposed") != key:
        meta_set(con, "mission_proposed", key)
        propose_single(con, now, races_fn(now), forced=True)
    if s <= ts < e and meta_get(con, "mission_reminded") != key and not mission_done(con, day):
        meta_set(con, "mission_reminded", key)
        tg_send(f"🚨 Mission : confirme (✅ Placé) un pari entre {CFG['MISSION_FROM']}h et {CFG['MISSION_TO']}h, "
                "sinon je meurs !")
    if ts >= e + float(CFG["MISSION_GRACE_MIN"]) * 60 and meta_get(con, "mission_checked") != key \
            and not mission_done(con, day) and ts < e + 6 * 3600:
        meta_set(con, "mission_checked", key)
        die(con, "💀 Aucun pari placé entre %sh et %sh : le bot meurt." % (CFG["MISSION_FROM"], CFG["MISSION_TO"]))


# ---------- chat ----------
HELP = ("Parle-moi simplement :\n• « multi 4 2 » – 4 courses, cote cumulée ≈ 2 (courses les plus proches)\n"
        "• « pari » – le meilleur pari du moment\n• « solde » – solde et stats\n• « mission » – état de la mission 12h-14h\n"
        "• « mes paris » – historique\n• « placé » ou « ok 0.5 @2.1 » – confirmer le dernier pari proposé\n"
        "• « solde 12.5 » – corriger le solde")


def norm(text):
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode().strip()


def handle_text(con, text, now=None, races_fn=get_upcoming):
    """Traite un message texte. Renvoie la réponse (str) ou None si déjà envoyée."""
    now = now or dt.datetime.now(TZ)
    t = norm(text)
    bank, dead = bankroll(con)
    m = re.match(r"^/?multi\s+(\d+)\s*(?:a|de|cote|@|x)?\s*(\d+(?:[.,]\d+)?)", t)
    if m:
        return propose_multi(con, now, races_fn(now), int(m.group(1)), float(m.group(2).replace(",", ".")))
    if re.match(r"^/?multi\b", t):
        return "Syntaxe : « multi 4 2 » = 4 courses, cote cumulée ≈ 2."
    m = re.match(r"^sol\w*\s+(\d+(?:[.,]\d+)?)", t)
    if m:
        set_bankroll(con, float(m.group(1).replace(",", ".")))
        for k in ("mission_checked",):
            con.execute("DELETE FROM meta WHERE k=?", (k,))
        return f"Solde : {bankroll(con)[0]:.2f} €"
    m = re.match(r"^(j ai |jai )?(place|pari[e]?|fait|c est fait|done|ok)\b(.*)$", t)
    if m and not re.match(r"^pari\s*$", t):
        b = con.execute("SELECT id, label, odd, stake FROM bets WHERE status='proposed' ORDER BY id DESC LIMIT 1").fetchone()
        if not b:
            return "Aucun pari en attente."
        odd, stake = b[2], b[3]
        for tok in m.group(3).split():
            try:
                if tok.startswith("@"):
                    odd = float(tok[1:].replace(",", "."))
                else:
                    stake = float(tok.replace(",", "."))
            except ValueError:
                pass
        return confirm_bet(con, b[0], min(stake, bank * MAX_STAKE_PCT + 1e-9), odd, now.timestamp())
    if t in ("/start", "aide", "help", "/help"):
        return HELP
    if re.search(r"\b(solde|bankroll|argent|cagnotte)\b", t):
        return ("💀 Solde : 0 € (mort). Envoie « solde 10 » pour relancer." if dead else
                f"💰 Solde : {bank:.2f} € | mise max : {bank * MAX_STAKE_PCT:.2f} €\n{stats_line(con)}")
    if re.search(r"\b(mission|aujourd|12h|14h|vivant|mort)\b", t):
        day = now.astimezone(TZ).date()
        s, e = win_bounds(day)
        if dead:
            return "💀 Je suis mort. Envoie « solde 10 » pour me relancer."
        if mission_done(con, day):
            return "✅ Mission du jour accomplie."
        if now.timestamp() < s:
            return f"🕛 Il faudra placer un pari entre {CFG['MISSION_FROM']}h et {CFG['MISSION_TO']}h."
        if now.timestamp() < e:
            return f"🚨 Mission en cours : encore {int((e - now.timestamp()) / 60)} min !"
        return "Fenêtre d'aujourd'hui terminée."
    if re.search(r"\b(mes paris|historique|resultats?)\b", t):
        rows = con.execute("SELECT label, odd, stake, status, payout FROM bets WHERE status IN ('placed','won','lost') "
                           "ORDER BY id DESC LIMIT 10").fetchall()
        icon = {"won": "🏆", "lost": "😞", "placed": "⏳"}
        return "\n".join(f"{icon[r[3]]} {r[0]} @ {r[1]:.2f} | {r[2]} € → {r[4] if r[4] is not None else '…'}"
                         for r in rows) or "Aucun pari pour l'instant."
    if re.search(r"\b(pari|parier|conseil|recommande|miser|jouer|radar|quoi)\b", t):
        if dead:
            return "💀 Je suis mort : plus de paris."
        propose_single(con, now, races_fn(now))
        return None
    return "Je n'ai pas compris. Écris « aide »."


def confirm_bet(con, bid, stake, odd, ts):
    b = con.execute("SELECT label, status FROM bets WHERE id=?", (bid,)).fetchone()
    if not b or b[1] != "proposed":
        return "Ce pari est déjà traité."
    stake = round(max(0.01, min(stake, bankroll(con)[0])), 2)
    con.execute("UPDATE bets SET status='placed', stake=?, odd=?, placed_ts=? WHERE id=?", (stake, odd, ts, bid))
    set_bankroll(con, bankroll(con)[0] - stake)
    con.commit()
    return f"✅ Enregistré : {b[0]} @ {odd:.2f} — {stake:.2f} €. Solde : {bankroll(con)[0]:.2f} €"


def poll_telegram(con, wait=0):
    if not (CFG["TELEGRAM_TOKEN"] and CFG["TELEGRAM_CHAT"]) or DRY:
        return
    try:
        ups = tg("getUpdates", offset=int(meta_get(con, "tg_offset", 0)), timeout=wait,
                 allowed_updates=["message", "callback_query"])
    except Exception as e:
        print("telegram:", e)
        time.sleep(2)
        return
    for u in ups:
        meta_set(con, "tg_offset", u["update_id"] + 1)
        cb, msg = u.get("callback_query"), u.get("message")
        chat = str(cb["message"]["chat"]["id"] if cb else (msg or {}).get("chat", {}).get("id", ""))
        if chat != str(CFG["TELEGRAM_CHAT"]):
            continue
        try:
            if cb:
                act, bid = cb["data"].split(":")
                b = con.execute("SELECT stake, odd, status FROM bets WHERE id=?", (int(bid),)).fetchone()
                if act == "p" and b:
                    reply = confirm_bet(con, int(bid), min(b[0], bankroll(con)[0] * MAX_STAKE_PCT + 1e-9), b[1], time.time())
                elif b and b[2] == "proposed":
                    con.execute("UPDATE bets SET status='ignored' WHERE id=?", (int(bid),))
                    reply = "❌ Pari ignoré."
                else:
                    reply = "Ce pari est déjà traité."
                for method, kw in (("answerCallbackQuery", {"callback_query_id": cb["id"], "text": reply[:190]}),
                                   ("editMessageReplyMarkup", {"chat_id": cb["message"]["chat"]["id"],
                                                               "message_id": cb["message"]["message_id"],
                                                               "reply_markup": {"inline_keyboard": []}})):
                    try:
                        tg(method, **kw)
                    except Exception as e:
                        print(method, e)
                tg_send(reply)
            else:
                text = (msg or {}).get("text", "")
                if text:
                    r = handle_text(con, text)
                    if r:
                        tg_send(r)
        except Exception as e:
            print("erreur traitement:", repr(e))
            tg_send("⚠️ Erreur interne, réessaie dans un instant.")
    con.commit()


# ---------- cycle / daemon ----------
def cycle(con, now=None, races_fn=get_upcoming):
    now = now or dt.datetime.now(TZ)
    if os.environ.get("REVIVE") == "1" and meta_get(con, "revived") != now.date().isoformat():
        meta_set(con, "revived", now.date().isoformat())
        set_bankroll(con, float(CFG["BANKROLL"]))
    settle_bets(con)
    mission_step(con, now, races_fn)
    con.commit()


def daemon(con):
    end = time.time() + float(CFG["DAEMON_MINUTES"]) * 60
    last = 0.0
    tg_send("🐎 Chevaux est en ligne. Écris « aide ».")
    while time.time() < end:
        if time.time() - last >= float(CFG["CYCLE_MIN"]) * 60:
            last = time.time()
            try:
                cycle(con)
            except Exception as e:
                print("cycle:", repr(e))
        poll_telegram(con, wait=int(CFG["LISTEN_SECONDS"]))


def demo_races(now, n_races=8, seed=1):
    """Courses synthétiques pour tester hors-ligne."""
    rng = np.random.default_rng(seed)
    races = []
    for i in range(n_races):
        k = int(rng.integers(8, 16))
        w = rng.dirichlet(np.ones(k) * 0.8) ** 1.0
        odds = np.maximum(1.2, 0.84 / w)
        runners = []
        for j in range(k):
            pw = float(w[j])
            runners.append({"num": j + 1, "name": f"CHEVAL{i}{j}", "odd": float(odds[j]), "p_win": pw,
                            "p_place": float(min(0.95, 1 - (1 - pw) ** 3)),
                            "place_odd": float(max(1.1, 1 + (odds[j] - 1) * 0.2))})
        races.append({"id": f"demo|{i}|1", "ts": now.timestamp() + (i + 1) * 1500, "label": f"DEMO R{i + 1}C1",
                      "runners": runners})
    return races


def main(argv):
    mode = argv[0] if argv else "daemon"
    con = db()
    if mode == "demo":
        now = dt.datetime.now(TZ)
        con = db(":memory:")
        fn = lambda n: demo_races(n)
        for q in ("multi 4 2", "pari", "solde"):
            print(">>", q)
            r = handle_text(con, q, now, fn)
            if r:
                print(r)
        return
    if mode == "once":
        cycle(con)
        poll_telegram(con)
    else:
        daemon(con)
    con.commit()


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])

