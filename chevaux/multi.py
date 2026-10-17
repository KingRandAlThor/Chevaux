"""Construction d'un « multi » : N sélections dans N courses différentes dont la cote cumulée vise une cible.

Les courses les plus proches dans le temps sont privilégiées (résultats rapides, cotes à jour).
Chaque sélection est un simple gagnant ou un simple placé ; le multi est un combiné SIMULÉ
(produit des rapports), le PMU ne propose pas de combiné de simples : il faut enchaîner les mises à la main.
"""
import itertools

TOL = 0.08          # écart relatif toléré entre cote cumulée et cible
MAX_PER_RACE = 5    # chevaux candidats par course


def leg_options(race):
    """race: {'id','ts','label','runners':[{'num','name','odd','p_win','p_place','place_odd'}]}.
    Renvoie les options (cheval, type) avec cote et probabilité."""
    opts = []
    top = sorted(race["runners"], key=lambda x: -x["p_win"])[:MAX_PER_RACE]
    for rn in top:
        opts.append({"race": race["id"], "ts": race["ts"], "label": race["label"], "num": rn["num"],
                     "name": rn["name"], "kind": "gagnant", "odd": rn["odd"], "p": rn["p_win"]})
        if rn.get("place_odd") and rn.get("p_place"):
            opts.append({"race": race["id"], "ts": race["ts"], "label": race["label"], "num": rn["num"],
                         "name": rn["name"], "kind": "place", "odd": rn["place_odd"], "p": rn["p_place"]})
    return opts


def build_multi(races, n, target, tol=TOL):
    """Cherche la combinaison de n courses (parmi les plus proches) dont le produit des cotes ≈ target.

    On élargit progressivement la fenêtre des courses (les plus proches d'abord) et on garde, dans la
    première fenêtre qui fournit une solution, celle de probabilité maximale.
    Renvoie dict(legs, odd, p) ou None."""
    races = sorted(races, key=lambda r: r["ts"])
    per = [leg_options(r) for r in races]
    per = [o for o in per if o]
    if n < 1 or len(per) < n or target <= 1.0:
        return None
    for window in range(n, len(per) + 1):
        best = None
        sub = per[:window]
        # la dernière course de la fenêtre doit servir, sinon la fenêtre précédente aurait suffi
        for combo in itertools.combinations(range(window), n):
            if window > n and (window - 1) not in combo:
                continue
            lists = [sub[i] for i in combo]
            for legs in _search(lists, target, tol):
                p = 1.0
                odd = 1.0
                for lg in legs:
                    p *= lg["p"]
                    odd *= lg["odd"]
                score = (p, -abs(odd - target))
                if best is None or score > best[0]:
                    best = (score, legs, odd, p)
        if best:
            return {"legs": list(best[1]), "odd": best[2], "p": best[3]}
    return None


def closest_multi(races, n, target, window=8):
    """Repli si aucune combinaison dans la tolérance : meilleure approximation sur les `window` courses les plus proches."""
    import math
    races = sorted(races, key=lambda r: r["ts"])
    per = [o for o in (leg_options(r) for r in races[:max(window, n)]) if o]
    if len(per) < n:
        return None
    want = math.log(target) / n
    best = None
    for combo in itertools.combinations(range(len(per)), n):
        legs = [min(per[i], key=lambda o: (abs(math.log(o["odd"]) - want), -o["p"])) for i in combo]
        odd = math.prod(l["odd"] for l in legs)
        score = (abs(math.log(odd / target)), -sum(l["ts"] for l in legs))
        if best is None or score < best[0]:
            best = (score, legs, odd)
    if not best:
        return None
    return {"legs": best[1], "odd": best[2], "p": math.prod(l["p"] for l in best[1]), "approx": True}


def _search(lists, target, tol):
    """Énumère les choix (un par course) dont le produit est dans [target(1-tol), target(1+tol)] (avec élagage)."""
    lo, hi = target * (1 - tol), target * (1 + tol)
    out = []
    # bornes pour l'élagage
    mins = [min(o["odd"] for o in L) for L in lists]
    maxs = [max(o["odd"] for o in L) for L in lists]

    def rec(i, cur, prod):
        if i == len(lists):
            if lo <= prod <= hi:
                out.append(list(cur))
            return
        rest_min = 1.0
        rest_max = 1.0
        for j in range(i, len(lists)):
            rest_min *= mins[j]
            rest_max *= maxs[j]
        if prod * rest_min > hi or prod * rest_max < lo:
            return
        for o in lists[i]:
            cur.append(o)
            rec(i + 1, cur, prod * o["odd"])
            cur.pop()

    rec(0, [], 1.0)
    return out


def format_multi(m, now_ts, tz=None, fmt_time=None):
    import datetime as dt
    lines = []
    for lg in m["legs"]:
        t = dt.datetime.fromtimestamp(lg["ts"], tz).strftime("%H:%M")
        lines.append(f"• {t} {lg['label']} — n°{lg['num']} {lg['name']} ({lg['kind']} @ {lg['odd']:.2f})")
    lines.append(f"Cote cumulée ≈ {m['odd']:.2f} | probabilité estimée ≈ {m['p']:.0%}")
    return "\n".join(lines)

