import datetime as dt
import json

import pytest

from chevaux import bot, multi, pmu
from chevaux.bot import TZ


def mk_race(i, ts, odds_place=(1.2, 1.5, 2.0), win=(2.0, 4.0, 8.0)):
    return {"id": f"2026-01-01|{i}|1", "ts": ts, "label": f"H R{i}C1",
            "runners": [{"num": j + 1, "name": f"C{i}{j}", "odd": win[j], "p_win": 0.8 / win[j],
                         "p_place": min(0.95, 2.2 / win[j]), "place_odd": odds_place[j]} for j in range(3)]}


@pytest.fixture
def con(tmp_path):
    bot.OUTBOX.clear()
    bot.DRY = True
    return bot.db(str(tmp_path / "b.db"))


def test_musique_stats():
    s = pmu.musique_stats("1p3p0a(24)DaAa")
    assert s["m_n"] == 5 and s["m_wins"] == 1 and s["m_placed"] == 2


def test_place_and_win_dividends():
    rap = [{"typePari": "SIMPLE_GAGNANT", "rapports": [{"combinaison": "4", "dividendePourUnEuro": 370}]},
           {"typePari": "SIMPLE_PLACE", "rapports": [{"combinaison": "4", "dividendePourUnEuro": 160},
                                                    {"combinaison": "5", "dividendePourUnEuro": 190}]}]
    assert pmu.win_dividend(rap) == (4, 3.7)
    assert pmu.place_dividends(rap) == {4: 1.6, 5: 1.9}


def test_multi_hits_target_with_soonest_races():
    base = 1_800_000_000
    races = [mk_race(i, base + i * 600) for i in range(8)]
    m = multi.build_multi(races, 4, 2.0)
    assert m and len(m["legs"]) == 4 and abs(m["odd"] - 2.0) <= 2.0 * multi.TOL
    assert len({lg["race"] for lg in m["legs"]}) == 4
    # fenêtre minimale : toutes les courses de la solution sont parmi les plus proches possibles
    assert max(lg["ts"] for lg in m["legs"]) <= base + 7 * 600


def test_multi_prefers_earliest_window():
    base = 1_800_000_000
    races = [mk_race(i, base + i * 600) for i in range(8)]
    m = multi.build_multi(races, 4, 2.0)
    assert {lg["race"].split("|")[1] for lg in m["legs"]} == {"0", "1", "2", "3"} or \
        max(lg["ts"] for lg in m["legs"]) < base + 7 * 600


def test_multi_impossible_returns_none():
    races = [mk_race(i, 1_800_000_000 + i) for i in range(3)]
    assert multi.build_multi(races, 4, 2.0) is None


def test_multi_command_creates_bet(con):
    now = dt.datetime(2026, 1, 1, 10, 0, tzinfo=TZ)
    races = [mk_race(i, now.timestamp() + 900 * (i + 1)) for i in range(6)]
    bot.handle_text(con, "multi 4 2", now, lambda n: races)
    row = con.execute("SELECT kind, status, legs FROM bets").fetchone()
    assert row[0] == "multi" and row[1] == "proposed" and len(json.loads(row[2])) == 4
    assert "Multi 4" in bot.OUTBOX[-1]


def test_multi_bad_syntax(con):
    now = dt.datetime(2026, 1, 1, 10, 0, tzinfo=TZ)
    assert "Syntaxe" in bot.handle_text(con, "multi 1 2", now, lambda n: [])
    assert "Syntaxe" in bot.handle_text(con, "multi", now, lambda n: [])


def test_stake_cap_and_confirm(con):
    now = dt.datetime(2026, 1, 1, 10, 0, tzinfo=TZ)
    races = [mk_race(i, now.timestamp() + 900 * (i + 1)) for i in range(6)]
    bot.handle_text(con, "pari", now, lambda n: races)
    assert con.execute("SELECT stake FROM bets").fetchone()[0] <= 0.6
    r = bot.handle_text(con, "ok 5 @1.5", now)
    assert "Enregistré" in r
    stake, = con.execute("SELECT stake FROM bets").fetchone()
    assert stake <= 0.6 + 1e-9  # plafonné à 6 % du solde
    assert bot.bankroll(con)[0] == pytest.approx(10 - stake)


def test_settle_multi_win_and_loss(con):
    legs = [{"race": "d|1|1", "num": 1, "kind": "place", "odd": 1.2, "name": "A", "label": "x", "ts": 0},
            {"race": "d|2|1", "num": 2, "kind": "gagnant", "odd": 2.0, "name": "B", "label": "y", "ts": 0}]
    bid = bot.new_bet(con, "multi", "m", legs, 2.4, 1.0)
    con.execute("UPDATE bets SET status='placed' WHERE id=?", (bid,))
    bot.set_bankroll(con, 9.0)

    def res(d, r, c):
        return {1: ({1: 2, 2: 1}, {1: 1.3, 2: 1.1}, (2, 3.0))}.get(r) if r == 1 else ({2: 1, 1: 2}, {}, (2, 3.5))

    bot.settle_bets(con, res)
    assert con.execute("SELECT status, payout FROM bets").fetchone() == ("won", 4.55)  # 1.0 * 1.3 * 3.5
    assert bot.bankroll(con)[0] == pytest.approx(13.55)


def test_settle_loss_early(con):
    legs = [{"race": "d|1|1", "num": 9, "kind": "place", "odd": 1.2, "name": "A", "label": "x", "ts": 0},
            {"race": "d|2|1", "num": 2, "kind": "gagnant", "odd": 2.0, "name": "B", "label": "y", "ts": 0}]
    bid = bot.new_bet(con, "multi", "m", legs, 2.4, 1.0)
    con.execute("UPDATE bets SET status='placed' WHERE id=?", (bid,))
    bot.settle_bets(con, lambda d, r, c: ({1: 1}, {1: 1.2}, (1, 2.0)) if r == 1 else None)
    assert con.execute("SELECT status FROM bets").fetchone()[0] == "lost"


def test_death_when_balance_zero(con):
    bot.set_bankroll(con, 0)
    assert bot.bankroll(con)[1] is True
    assert any("mort" in m for m in bot.OUTBOX)
    assert "mort" in bot.handle_text(con, "pari", dt.datetime(2026, 1, 1, 10, tzinfo=TZ), lambda n: [])


def test_mission_proposes_then_dies(con):
    day = dt.date(2026, 1, 1)
    races = lambda n: [mk_race(i, n.timestamp() + 900 * (i + 1)) for i in range(4)]
    bot.mission_step(con, dt.datetime.combine(day, dt.time(9, 5), TZ), races)
    assert con.execute("SELECT COUNT(*) FROM bets WHERE status='proposed'").fetchone()[0] == 1
    bot.mission_step(con, dt.datetime.combine(day, dt.time(14, 30), TZ), races)  # rien placé -> mort
    assert bot.bankroll(con)[1] is True


def test_mission_survives_if_placed_in_window(con):
    day = dt.date(2026, 1, 1)
    races = lambda n: [mk_race(i, n.timestamp() + 900 * (i + 1)) for i in range(4)]
    bot.mission_step(con, dt.datetime.combine(day, dt.time(9, 5), TZ), races)
    bid = con.execute("SELECT id FROM bets").fetchone()[0]
    bot.confirm_bet(con, bid, 0.2, 1.3, dt.datetime.combine(day, dt.time(12, 30), TZ).timestamp())
    bot.mission_step(con, dt.datetime.combine(day, dt.time(14, 30), TZ), races)
    assert bot.bankroll(con)[1] is False


def test_never_places_bets_itself():
    import inspect
    src = inspect.getsource(bot)
    assert "parier.pmu" not in src and "turf-fr" not in src
