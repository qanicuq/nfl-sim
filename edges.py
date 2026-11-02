"""Compare the simulator with Kalshi and Polymarket prices for one game.

Usage: python edges.py AWAY HOME     (example: python edges.py NE MIA)

Game lines come from the backtested team model in nfl_sim.py.
Player props come from each player's own recent games. That part has
never been backtested, so every player result is marked "untested".
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import nfl_sim

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
POLY = "https://gamma-api.polymarket.com"
STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv"
)
INJURY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "injuries/injuries_{season}.csv"
)
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_TEAM = {"WSH": "WAS", "LAR": "LA"}
ESPN_STATUS = {"Injured Reserve": "Out", "Suspended": "Out", "Physically Unable to Perform": "Out"}
N_SIMS = 10000
PLAYER_GAMES = 8  # recent games used per player
MIN_PLAYER_GAMES = 3
FLAG = 0.05  # flag edges of 5 points or more after fees

# Kalshi series -> stat simulated for the player in that market.
KALSHI_PLAYER_SERIES = {
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLRSHYDS": "rushing_yards",
    "KXNFLRECYDS": "receiving_yards",
    "KXNFLRRYDS": "rush_rec_yards",
    "KXNFLREC": "receptions",
    "KXNFLPASSTDS": "passing_tds",
    "KXNFLPASSINT": "passing_interceptions",
    "KXNFLPASSATT": "attempts",
    "KXNFLPASSCOMP": "completions",
    "KXNFLRSHATT": "carries",
    "KXNFLTD": "touchdowns",
}
KALSHI_GAME_SERIES = ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLTEAMTOTAL"]

# Polymarket sportsMarketType -> (stat, fixed line or None to use market line)
POLY_PLAYER_TYPES = {
    "passing_yards": ("passing_yards", None),
    "rushing_yards": ("rushing_yards", None),
    "receiving_yards": ("receiving_yards", None),
    "rushing_receiving_yards": ("rush_rec_yards", None),
    "receptions": ("receptions", None),
    "passing_touchdowns": ("passing_tds", None),
    "passing_interceptions": ("passing_interceptions", None),
    "interceptions": ("passing_interceptions", None),
    "passing_attempts": ("attempts", None),
    "completions": ("completions", None),
    "passing_completions": ("completions", None),
    "rushing_attempts": ("carries", None),
    "anytime_touchdowns": ("touchdowns", 0.5),
    "two_plus_touchdowns": ("touchdowns", 1.5),
}
COUNT_STATS = {"passing_tds", "passing_interceptions", "touchdowns"}
VOLUME_STATS = {"receptions", "carries", "attempts", "completions"}
PASS_STATS = {"passing_yards", "attempts", "completions", "passing_tds", "passing_interceptions"}
ROLE_VOLUME = {
    "rushing_yards": ["carries"],
    "carries": ["carries"],
    "receiving_yards": ["targets"],
    "receptions": ["targets"],
    "rush_rec_yards": ["carries", "targets"],
}


def get_json(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("too many retries")


def num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if 0 < out < 1 else None


def kalshi_fee(price, multiplier):
    # Kalshi quadratic taker fee per contract (Kalshi rounds the order total up to a cent).
    return 0.07 * multiplier * price * (1 - price)


def poly_fee(price, rate):
    # Polymarket taker fee per share: rate x p x (1 - p).
    return rate * price * (1 - price)


def norm_name(name):
    name = name.lower().replace(".", "").replace("'", "")
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", name)
    return " ".join(name.split())


# ---------- player model ----------

def load_player_games(season, week):
    frames = []
    for year in (season - 1, season):
        try:
            frames.append(pd.read_csv(STATS_URL.format(season=year), low_memory=False))
        except Exception:
            print(f"player stats for {year} not available")
    stats = pd.concat(frames, ignore_index=True)
    stats = stats[(stats["season"] < season) | (stats["week"] < week)]
    stats = stats[stats["season"] >= season - 1].copy()
    numbers = stats.select_dtypes("number").columns
    stats[numbers] = stats[numbers].fillna(0)
    stats = stats.assign(
        rush_rec_yards=stats["rushing_yards"] + stats["receiving_yards"],
        touchdowns=stats["rushing_tds"]
        + stats["receiving_tds"]
        + stats["special_teams_tds"]
        + stats["fumble_recovery_tds"],
        key=stats["player_display_name"].fillna("").map(norm_name),
    )
    return stats.sort_values(["season", "week"])


def load_injuries(season, week, teams):
    # This week's injury report: player name -> Out, Doubtful or Questionable.
    try:
        report = pd.read_csv(INJURY_URL.format(season=season), low_memory=False)
    except Exception:
        print("injury report not available")
        return {}
    report = report[
        (report["season"] == season)
        & (report["week"] == week)
        & (report["team"].isin(teams))
        & report["report_status"].notna()
    ]
    return {norm_name(str(n)): s for n, s in zip(report["full_name"], report["report_status"])}


def load_espn_injuries(away, home, today):
    # Today's ESPN injury report for this game: player name -> status.
    try:
        board = get_json(f"{ESPN}/scoreboard?dates={today:%Y%m%d}")
    except Exception:
        print("ESPN scoreboard not available")
        return {}
    for event in board.get("events", []):
        codes = {
            ESPN_TEAM.get(c["team"]["abbreviation"], c["team"]["abbreviation"])
            for c in event["competitions"][0]["competitors"]
        }
        if codes != {away, home}:
            continue
        try:
            summary = get_json(f"{ESPN}/summary?event={event['id']}")
        except Exception:
            print("ESPN injury report not available")
            return {}
        report = {}
        for team in summary.get("injuries", []):
            for item in team.get("injuries", []):
                name = (item.get("athlete") or {}).get("displayName")
                status = item.get("status") or ""
                if name:
                    report[norm_name(name)] = ESPN_STATUS.get(status, status)
        return report
    print("ESPN game not found")
    return {}


def draw(recent, stat, rng, n):
    mean = recent.mean()
    if mean <= 0:
        return np.zeros(n)
    if stat in COUNT_STATS:
        return rng.poisson(mean, n).astype(float)
    var = max(recent.var(ddof=1), mean)
    if stat in VOLUME_STATS:
        if var <= mean:
            return rng.poisson(mean, n).astype(float)
        size = mean * mean / (var - mean)
        return rng.negative_binomial(size, size / (size + mean), n).astype(float)
    # Yards: a gamma curve allows the occasional big game that a bell curve misses.
    shape = mean * mean / var
    return np.rint(rng.gamma(shape, var / mean, n))


def starter_rows(stats, rows):
    # Passing stats come from games the QB started (15+ attempts). A QB with fewer
    # than 3 starts gets his team's starting-QB line from each recent game instead.
    starts = rows[rows["attempts"] >= 15]
    if len(starts) >= MIN_PLAYER_GAMES or rows.empty:
        return starts
    team_rows = stats[stats["team"] == rows["team"].iloc[-1]]
    best = team_rows.sort_values("attempts").groupby(["season", "week"]).tail(1)
    return best.sort_values(["season", "week"])


def td_sims(stats, rows, rng):
    # Touchdowns are rare, so 8 games say little. Use his average touches
    # (carries + targets) times the league touchdown rate per touch for his position.
    recent = rows.tail(PLAYER_GAMES)
    touches = float((recent["carries"] + recent["targets"]).mean())
    same = stats[stats["position"] == rows["position"].iloc[-1]]
    volume = float((same["carries"] + same["targets"]).sum())
    rate = float(same["touchdowns"].sum()) / volume if volume > 0 else 0.0
    return rng.poisson(touches * rate, N_SIMS).astype(float)


def player_sims(stats, name, stat, rng, cache):
    key = (norm_name(name), stat)
    if key in cache:
        return cache[key]
    rows = stats[stats["key"] == key[0]]
    if stat in PASS_STATS:
        rows = starter_rows(stats, rows)
    recent = rows[stat].tail(PLAYER_GAMES).to_numpy(dtype=float)
    if len(recent) < MIN_PLAYER_GAMES:
        cache[key] = (None, len(recent))
    elif stat == "touchdowns":
        cache[key] = (td_sims(stats, rows, rng), len(recent))
    else:
        cache[key] = (draw(recent, stat, rng, N_SIMS), len(recent))
    return cache[key]


# The raw player sims are too sure of themselves (checked on past games), so
# each stat gets a correction fitted on LAST season only and checked on this
# season's games so far: corrected = 1 / (1 + exp(-(a + b * logit(raw)))).
STAT_FLOOR = {
    "passing_yards": 50, "rushing_yards": 5, "receiving_yards": 5, "rush_rec_yards": 5,
    "receptions": 1, "attempts": 10, "completions": 5, "carries": 1,
    "passing_tds": 0.2, "passing_interceptions": 0.2, "touchdowns": 0.1,
}
MULTIPLES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
CALIBRATION = {}
PROP_CAL_FILE = "prop_calibration.json"
PROP_CAL = None


def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def fit_line(x, y):
    # Logistic regression y ~ a + b*x by Newton's method.
    a, b = 0.0, 1.0
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a + b * x)))
        grad = X.T @ (y - p)
        hess = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(hess, grad)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def calibration_cases(stats, rng):
    cases = []
    for _, games in stats.groupby("player_id", sort=False):
        for stat, floor in STAT_FLOOR.items():
            values = games[stat].to_numpy(dtype=float)
            seasons = games["season"].to_numpy()
            for i in range(MIN_PLAYER_GAMES, len(values)):
                recent = values[max(0, i - PLAYER_GAMES):i]
                mean = recent.mean()
                if mean < floor:
                    continue
                sims = draw(recent, stat, rng, 2000)
                if stat in COUNT_STATS:
                    lines = (0.5, 1.5, 2.5)
                else:
                    lines = sorted({np.floor(mean * k) + 0.5 for k in MULTIPLES})
                for line in lines:
                    cases.append((stat, seasons[i], float(np.mean(sims > line)), float(values[i] > line)))
    return pd.DataFrame(cases, columns=["stat", "season", "raw", "hit"])


def fit_calibration(stats, season, rng):
    cases = calibration_cases(stats, rng)
    report = []
    for stat in STAT_FLOOR:
        train = cases[(cases["stat"] == stat) & (cases["season"] < season)]
        test = cases[(cases["stat"] == stat) & (cases["season"] == season)]
        if len(train) < 200:
            continue
        a, b = fit_line(logit(train["raw"].to_numpy()), train["hit"].to_numpy())
        CALIBRATION[stat] = (a, b)
        if len(test):
            fixed = 1 / (1 + np.exp(-(a + b * logit(test["raw"].to_numpy()))))
            report.append((
                stat, len(test),
                float(np.mean((test["raw"] - test["hit"]) ** 2)),
                float(np.mean((fixed - test["hit"]) ** 2)),
                float(np.mean((test["hit"].mean() - test["hit"]) ** 2)),
            ))
    return report


def player_raw_prob(sims, line):
    return float(np.mean(sims > line))


def player_prob(sims, stat, line):
    # Prefer the correction fitted on real Kalshi lines (prop_backtest.py writes it).
    global PROP_CAL
    if PROP_CAL is None:
        try:
            with open(PROP_CAL_FILE, encoding="utf-8") as handle:
                PROP_CAL = json.load(handle)
        except FileNotFoundError:
            PROP_CAL = {}
    raw = player_raw_prob(sims, line)
    if PROP_CAL:
        a, b = PROP_CAL.get(stat, PROP_CAL["_all"])
    elif stat in CALIBRATION:
        a, b = CALIBRATION[stat]
    else:
        return None
    return float(1 / (1 + np.exp(-(a + b * logit(raw)))))


def role_scale(stats, name, stat, sims, injuries):
    # Give a player his share of an Out or Doubtful teammate's carries or targets.
    # Only a teammate who played the team's last game counts; otherwise recent games already show it.
    cols = ROLE_VOLUME.get(stat)
    key = norm_name(name)
    rows = stats[stats["key"] == key]
    if cols is None or rows.empty:
        return sims, 1.0
    team = rows["team"].iloc[-1]
    position = rows["position"].iloc[-1]
    team_rows = stats[stats["team"] == team]
    weeks = team_rows[["season", "week"]].drop_duplicates().tail(PLAYER_GAMES)
    recent = team_rows.merge(weeks, on=["season", "week"])
    recent = recent[recent["position"] == position]
    last = weeks.iloc[-1]
    played_last = set(recent.loc[(recent["season"] == last["season"]) & (recent["week"] == last["week"]), "key"])
    volume = recent[cols].sum(axis=1).groupby(recent["key"]).sum()
    total = volume.sum()
    if total <= 0 or volume.get(key, 0) <= 0:
        return sims, 1.0
    share = volume / total
    out = [k for k in share.index if injuries.get(k, "") in ("Out", "Doubtful") and k in played_last]
    out_share = float(share[out].sum())
    left = 1.0 - out_share
    if out_share <= 0 or left <= 0:
        return sims, 1.0
    factor = min(1.0 + out_share / left, 2.0)
    return np.rint(sims * factor), factor


# ---------- game model ----------

def game_sims(games, row, rng):
    with open("settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)
    home_exp, away_exp = nfl_sim.expected_points(
        games, row, int(settings["N"]), settings["H"]
    )
    if home_exp is None:
        sys.exit("not enough past games to rate one of the teams")
    S = float(settings["S"])
    home = np.maximum(np.rint(rng.normal(home_exp, S, N_SIMS)), 0)
    away = np.maximum(np.rint(rng.normal(away_exp, S, N_SIMS)), 0)
    return home, away


def win_prob(margin):
    # Both sites settle a tie at 50 cents.
    return float(np.mean(margin > 0) + 0.5 * np.mean(margin == 0))


# ---------- markets ----------

def kalshi_markets(event_code, away, home, sims, stats, rng, cache, out, skipped, injuries):
    series_info = get_json(f"{KALSHI}/series?category=Sports")
    fee_mult = {
        item["ticker"]: float(item.get("fee_multiplier") or 1)
        for item in series_info.get("series", [])
    }
    team_of = {}
    margin = {home: sims["home"] - sims["away"], away: sims["away"] - sims["home"]}
    points = {home: sims["home"], away: sims["away"]}
    total = sims["home"] + sims["away"]

    for series in KALSHI_GAME_SERIES + list(KALSHI_PLAYER_SERIES):
        time.sleep(0.5)
        data = get_json(f"{KALSHI}/markets?event_ticker={series}-{event_code}&limit=1000")
        markets = data.get("markets", [])
        if not markets:
            skipped.append(f"Kalshi | no {series} markets for {event_code}")
        if series == "KXNFLGAME":
            for m in markets:
                team_of[m["custom_strike"]["football_team"]] = m["ticker"].rsplit("-", 1)[1]
        for m in markets:
            if m.get("status") != "active":
                continue
            title = m["title"]
            strike = m.get("floor_strike")
            team = team_of.get((m.get("custom_strike") or {}).get("football_team"))
            kind = "game"
            proj = None
            group = title
            if series == "KXNFLGAME":
                p_yes = win_prob(margin[team])
                proj = float(margin[team].mean())
            elif series == "KXNFLSPREAD" and team in margin:
                p_yes = float(np.mean(margin[team] > strike))
                proj = float(margin[team].mean())
            elif series == "KXNFLTOTAL":
                p_yes = float(np.mean(total > strike))
                proj = float(total.mean())
            elif series == "KXNFLTEAMTOTAL" and team in points:
                p_yes = float(np.mean(points[team] > strike))
                proj = float(points[team].mean())
            elif series in KALSHI_PLAYER_SERIES and ":" in title:
                kind = "player (untested)"
                name = title.split(":")[0].strip()
                group = f"{norm_name(name)}|{KALSHI_PLAYER_SERIES[series]}"
                status = injuries.get(norm_name(name), "")
                if status in ("Out", "Doubtful"):
                    skipped.append(f"Kalshi | {title} | injury report: {status}")
                    continue
                if status == "Questionable":
                    kind = "player (untested, Q)"
                player, games_used = player_sims(
                    stats, name, KALSHI_PLAYER_SERIES[series], rng, cache
                )
                if player is None:
                    skipped.append(f"Kalshi | {title} | only {games_used} past games")
                    continue
                player, factor = role_scale(stats, name, KALSHI_PLAYER_SERIES[series], player, injuries)
                if factor > 1:
                    kind = kind.replace(")", f", role x{factor:.2f})")
                proj = float(player.mean())
                p_yes = player_raw_prob(player, strike)
                if p_yes is None:
                    skipped.append(f"Kalshi | {title} | no calibration for this stat")
                    continue
            else:
                skipped.append(f"Kalshi | {title} | not modeled")
                continue
            mult = fee_mult.get(series, 1.0)
            for side, prob, ask in (
                ("YES", p_yes, num(m.get("yes_ask_dollars"))),
                ("NO", 1 - p_yes, num(m.get("no_ask_dollars"))),
            ):
                if ask is None:
                    continue
                fee = kalshi_fee(ask, mult)
                out.append({
                    "site": "Kalshi", "market": title, "side": side, "kind": kind,
                    "price": ask, "fee": fee, "sim": prob, "edge": prob - ask - fee,
                    "volume": m.get("volume_fp") or "",
                    "proj": proj,
                    "group": group,
                })


def poly_markets(slug_base, away, home, names, sims, stats, rng, cache, out, skipped, injuries):
    margin = {home: sims["home"] - sims["away"], away: sims["away"] - sims["home"]}
    points = {home: sims["home"], away: sims["away"]}
    total = sims["home"] + sims["away"]
    found = 0
    for suffix in ("", "-player-props", "-more-markets", "-totals", "-game-props"):
        events = get_json(f"{POLY}/events?slug={slug_base}{suffix}")
        for event in events:
            for m in event.get("markets", []):
                if m.get("closed") or not m.get("active", True):
                    continue
                found += 1
                question = m.get("question", "")
                kind_type = m.get("sportsMarketType") or ""
                outcomes = m.get("outcomes") or "[]"
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                line = m.get("line")
                if len(outcomes) != 2:
                    skipped.append(f"Polymarket | {question} | not a two-way market")
                    continue
                first = outcomes[0]
                kind = "game"
                p_first = None
                proj = None
                group = question
                if kind_type == "moneyline":
                    team = names.get(first)
                    if team:
                        p_first = win_prob(margin[team])
                        proj = float(margin[team].mean())
                elif kind_type == "spreads" and line is not None:
                    team = names.get(first)
                    if team:
                        p_first = float(np.mean(margin[team] > -float(line)))
                        proj = float(margin[team].mean())
                elif kind_type == "totals" and line is not None and first.lower() == "over":
                    p_first = float(np.mean(total > float(line)))
                    proj = float(total.mean())
                elif kind_type in POLY_PLAYER_TYPES and ":" in question:
                    kind = "player (untested)"
                    stat, fixed = POLY_PLAYER_TYPES[kind_type]
                    cut = fixed if fixed is not None else line
                    if cut is not None and first.lower() in ("over", "yes"):
                        name = question.split(":")[0].strip()
                        group = f"{norm_name(name)}|{stat}"
                        status = injuries.get(norm_name(name), "")
                        if status in ("Out", "Doubtful"):
                            skipped.append(f"Polymarket | {question} | injury report: {status}")
                            continue
                        if status == "Questionable":
                            kind = "player (untested, Q)"
                        player, games_used = player_sims(
                            stats, name, stat, rng, cache
                        )
                        if player is None:
                            skipped.append(f"Polymarket | {question} | only {games_used} past games")
                            continue
                        player, factor = role_scale(stats, name, stat, player, injuries)
                        if factor > 1:
                            kind = kind.replace(")", f", role x{factor:.2f})")
                        proj = float(player.mean())
                        p_first = player_raw_prob(player, float(cut))
                if p_first is None:
                    skipped.append(f"Polymarket | {question} | not modeled ({kind_type})")
                    continue
                rate = float((m.get("feeSchedule") or {}).get("rate", 0.05)) if m.get("feesEnabled") else 0.0
                best_ask = num(m.get("bestAsk"))
                best_bid = num(m.get("bestBid"))
                second_ask = num(1 - best_bid) if best_bid is not None else None
                for side, prob, ask in (
                    (first, p_first, best_ask),
                    (outcomes[1], 1 - p_first, second_ask),
                ):
                    if ask is None:
                        continue
                    fee = poly_fee(ask, rate)
                    out.append({
                        "site": "Polymarket", "market": question, "side": side, "kind": kind,
                        "price": ask, "fee": fee, "sim": prob, "edge": prob - ask - fee,
                        "volume": m.get("volumeNum") or m.get("volume") or "",
                        "proj": proj,
                        "group": group,
                    })
    if found == 0:
        skipped.append(f"Polymarket | no open markets found for {slug_base}")


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python edges.py AWAY HOME")
    away, home = sys.argv[1].upper(), sys.argv[2].upper()
    today = datetime.now(ZoneInfo("America/New_York")).date()

    games = nfl_sim.load_games()
    match = games[
        (games["away_team"] == away)
        & (games["home_team"] == home)
        & (games["gameday"] == pd.Timestamp(today))
    ]
    if match.empty:
        sys.exit(f"no {away} @ {home} game today ({today})")
    row = match.iloc[0].to_dict()
    now = datetime.now(ZoneInfo("America/New_York"))
    if pd.notna(row["gametime"]) and now.strftime("%H:%M") >= row["gametime"]:
        # After kickoff the prices are live in-game prices; a pre-game model can't be compared.
        nfl_sim._send_push(f"{away} @ {home}: game already started, nothing compared")
        sys.exit(f"{away} @ {home} kicked off at {row['gametime']} New York time")

    rng = np.random.default_rng(42)
    home_pts, away_pts = game_sims(games, row, rng)
    sims = {"home": home_pts, "away": away_pts}
    stats = load_player_games(int(row["season"]), int(row["week"]))
    calibration = fit_calibration(stats, int(row["season"]), rng)
    injuries = load_injuries(int(row["season"]), int(row["week"]), {away, home})
    injuries.update(load_espn_injuries(away, home, today))
    cache = {}
    out = []
    skipped = []

    event_code = today.strftime("%y%b%d").upper() + away + home
    kalshi_markets(event_code, away, home, sims, stats, rng, cache, out, skipped, injuries)

    slug_base = f"nfl-{away.lower()}-{home.lower()}-{today.isoformat()}"
    poly_markets(slug_base, away, home, NICKNAMES, sims, stats, rng, cache, out, skipped, injuries)

    write_report(away, home, today, sims, out, skipped, calibration)


# Polymarket names teams by nickname.
NICKNAMES = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Rams": "LA", "Chargers": "LAC", "Raiders": "LV", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "Seahawks": "SEA",
    "49ers": "SF", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
}


def american(p):
    # Chance (0-1) -> American odds, like -122 or +133.
    if p <= 0 or p >= 1:
        return "n/a"
    if p >= 0.5:
        return f"-{round(100 * p / (1 - p))}"
    return f"+{round(100 * (1 - p) / p)}"


def pct(value):
    return f"{100 * value:.1f}"


def write_report(away, home, today, sims, out, skipped, calibration):
    out.sort(key=lambda r: r["edge"], reverse=True)
    flagged = [r for r in out if r["edge"] >= FLAG]
    top10 = []
    seen = set()
    for r in out:
        if "Q" in r["kind"] or r["group"] in seen:
            continue
        seen.add(r["group"])
        top10.append(r)
        if len(top10) == 10:
            break
    margin = sims["home"] - sims["away"]
    lines = [
        f"# Edges: {away} @ {home}, {today}",
        "",
        f"Run at {datetime.now(ZoneInfo('America/New_York')):%Y-%m-%d %I:%M %p} New York time. "
        "Prices are the ask (what you would pay right now), and fees are taken out.",
        "",
        f"Sim: {home} wins {pct(win_prob(margin))}%, average score "
        f"{away} {sims['away'].mean():.1f} - {home} {sims['home'].mean():.1f}.",
        "",
        "Read this first:",
        "- Game lines: this model was LESS accurate than the Vegas line in the 2022-2025 test and lost money.",
        "- Player props: the hit count is straight from the 10,000 sims of that player's last "
        f"{PLAYER_GAMES} games. On past Kalshi props these sims did NOT beat the market.",
        "- If a teammate at the same position is Out or Doubtful (ESPN injury report, read when this ran), "
        "the player gets his share of that teammate's carries or targets (shown as role x1.40). "
        "A backup QB uses his team's starting-QB games.",
        "- Market odds include the site's fee.",
        "- Touchdowns: his recent touches (carries + targets) times the league touchdown rate for his position.",
        "- Edge = sim chance - price - fee. Flagged = 5 points or more.",
        "- The Top 10 shows only the best line for each player and stat.",
        "- Players listed Out or Doubtful on the injury report are left out. Questionable players are marked Q and kept out of the Top 10.",
        "- Sim projected = the 10,000-sim average: the stat for player props, the team's winning margin for winner and spread markets (negative means it loses), total points for totals, the team's points for team totals.",
        "",
        "## Top 10 (most edge first)",
        "",
        "| rank | site | market | buy | sim projected | hit in sims | sim fair odds | price | market odds | edge (pts) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {i} | {r['site']} | {r['market']} | {r['side']} | {r['proj']:.1f} | "
            f"{round(r['sim'] * N_SIMS):,} of {N_SIMS:,} | {american(r['sim'])} | "
            f"{r['price']:.2f} | {american(r['price'] + r['fee'])} | {100 * r['edge']:.1f} |"
            for i, r in enumerate(top10, 1)
        ],
        "",
        f"## Flagged ({len(flagged)})",
        "",
        "| site | market | buy | price | fee | sim % | edge (pts) | kind | volume |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in flagged:
        lines.append(
            f"| {r['site']} | {r['market']} | {r['side']} | {r['price']:.2f} | {r['fee']:.3f} | "
            f"{pct(r['sim'])} | {100 * r['edge']:.1f} | {r['kind']} | {r['volume']} |"
        )
    lines += [
        "",
        f"## Every priced side ({len(out)})",
        "",
        "| site | market | buy | price | sim % | edge (pts) | kind |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in out:
        lines.append(
            f"| {r['site']} | {r['market']} | {r['side']} | {r['price']:.2f} | "
            f"{pct(r['sim'])} | {100 * r['edge']:.1f} | {r['kind']} |"
        )
    lines += [
        "",
        "## Player model check (this season's games, not used to fit)",
        "",
        "Score = average squared miss (lower is better). Guess-the-average is what you get "
        "by always saying the overall hit rate.",
        "",
        "| stat | cases | raw sim | corrected | guess-the-average |",
        "|---|---:|---:|---:|---:|",
    ]
    for stat, count, raw, fixed, base in calibration:
        lines.append(f"| {stat} | {count} | {raw:.4f} | {fixed:.4f} | {base:.4f} |")
    lines += ["", f"## Skipped ({len(skipped)})", ""]
    lines += [f"- {item}" for item in skipped]
    lines.append("")
    with open("edges.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    alert = [
        f"{i}. {r['site']}: {r['market']} -> {r['side']} | "
        f"sims {round(r['sim'] * N_SIMS):,}/{N_SIMS:,} (fair {american(r['sim'])}) | "
        f"price {r['price']:.2f} ({american(r['price'] + r['fee'])}) | "
        f"proj {r['proj']:.1f} | edge {100 * r['edge']:.1f}"
        for i, r in enumerate(top10, 1)
    ]
    message = f"{away} @ {home} Top 10 (player sims unproven vs market)\n" + "\n".join(alert)
    nfl_sim._send_push(message)


if __name__ == "__main__":
    main()
