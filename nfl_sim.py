import json
import os
import sys
import urllib.request
from bisect import bisect_left
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
WIN_UNITS = 0.909
N_SIMS = 10000


def load_games():
    games = pd.read_csv(GAMES_URL)
    games["gameday"] = pd.to_datetime(games["gameday"])
    # Moved teams keep their history under their current code.
    moved = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
    games["home_team"] = games["home_team"].replace(moved)
    games["away_team"] = games["away_team"].replace(moved)
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return games


def team_ratings_before(games, team, date, N):
    # Completed games only. Nothing on or after `date` is used.
    cached = getattr(team_ratings_before, "_hist", None)
    cached_id = getattr(team_ratings_before, "_hist_id", None)
    if cached is None or cached_id != id(games):
        hist = {}
        for row in games.itertuples(index=False):
            if pd.isna(row.home_score) or pd.isna(row.away_score):
                continue
            hist.setdefault(row.home_team, []).append(
                (row.gameday, float(row.home_score), float(row.away_score))
            )
            hist.setdefault(row.away_team, []).append(
                (row.gameday, float(row.away_score), float(row.home_score))
            )
        team_ratings_before._hist = hist
        team_ratings_before._hist_id = id(games)
    else:
        hist = cached

    history = hist.get(team)
    if not history:
        return None, None
    cut = bisect_left(history, (date,))
    prior = history[max(0, cut - N) : cut]
    if not prior:
        return None, None
    scored = sum(item[1] for item in prior) / len(prior)
    allowed = sum(item[2] for item in prior) / len(prior)
    return scored, allowed


def expected_points(games, game_row, N, H):
    if game_row["location"] == "Neutral":
        H = 0
    home_scored, home_allowed = team_ratings_before(
        games, game_row["home_team"], game_row["gameday"], N
    )
    away_scored, away_allowed = team_ratings_before(
        games, game_row["away_team"], game_row["gameday"], N
    )
    if home_scored is None or away_scored is None:
        return None, None
    home_exp = (home_scored + away_allowed) / 2 + H / 2
    away_exp = (away_scored + home_allowed) / 2 - H / 2
    return home_exp, away_exp


def simulate(home_exp, away_exp, spread_line, total_line, S, rng):
    home_scores = np.maximum(np.rint(rng.normal(home_exp, S, N_SIMS)), 0)
    away_scores = np.maximum(np.rint(rng.normal(away_exp, S, N_SIMS)), 0)
    margins = home_scores - away_scores
    totals = home_scores + away_scores
    home_win_pct = 100.0 * (
        np.sum(margins > 0) + 0.5 * np.sum(margins == 0)
    ) / N_SIMS
    avg_margin = float(np.mean(margins))
    avg_total = float(np.mean(totals))
    cover_decided = margins != spread_line
    if np.any(cover_decided):
        cover_pct = 100.0 * float(np.mean(margins[cover_decided] > spread_line))
    else:
        cover_pct = 50.0
    over_decided = totals != total_line
    if np.any(over_decided):
        over_pct = 100.0 * float(np.mean(totals[over_decided] > total_line))
    else:
        over_pct = 50.0
    return (
        float(home_win_pct),
        avg_margin,
        avg_total,
        float(cover_pct),
        float(over_pct),
    )


def _spread_units(result, spread_line, side):
    if result == spread_line:
        return 0.0, "push"
    home_covers = result > spread_line
    won = home_covers if side == "home" else not home_covers
    if won:
        return WIN_UNITS, "win"
    return -1.0, "loss"


def _total_units(points, total_line, side):
    if points == total_line:
        return 0.0, "push"
    went_over = points > total_line
    won = went_over if side == "over" else not went_over
    if won:
        return WIN_UNITS, "win"
    return -1.0, "loss"


def _pick(cover_pct, over_pct):
    picks = []
    if cover_pct is not None:
        if cover_pct >= 55:
            picks.append("home")
        elif cover_pct <= 45:
            picks.append("away")
    if over_pct is not None:
        if over_pct >= 55:
            picks.append("over")
        elif over_pct <= 45:
            picks.append("under")
    if not picks:
        return "no bet"
    return ", ".join(picks)


def _blank_month():
    return {
        "spread_bets": 0,
        "spread_wins": 0,
        "spread_losses": 0,
        "spread_units": 0.0,
        "total_bets": 0,
        "total_wins": 0,
        "total_losses": 0,
        "total_units": 0.0,
        "baseline1_units": 0.0,
        "baseline2_units": 0.0,
    }


def _add_grade(bucket, kind, units, label):
    bucket[kind + "_bets"] += 1
    bucket[kind + "_units"] += units
    if label == "win":
        bucket[kind + "_wins"] += 1
    elif label == "loss":
        bucket[kind + "_losses"] += 1


def backtest():
    games = load_games()
    played = games["home_score"].notna() & games["away_score"].notna()
    train = games[
        games["season"].between(2012, 2021) & played & games["result"].notna()
    ]
    train_scores = np.concatenate(
        [
            train["home_score"].to_numpy(dtype=float),
            train["away_score"].to_numpy(dtype=float),
        ]
    )
    S = float(np.std(train_scores, ddof=1))

    train_rows = train[
        ["home_team", "away_team", "gameday", "location", "result"]
    ].to_dict("records")

    best_mae = None
    best_n = None
    best_h = None
    for N in (4, 8, 16):
        for H in (0, 1, 2, 3):
            errors = []
            for game_row in train_rows:
                home_exp, away_exp = expected_points(games, game_row, N, H)
                if home_exp is None:
                    continue
                errors.append(abs((home_exp - away_exp) - game_row["result"]))
            mae = float(np.mean(errors))
            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_n = N
                best_h = H

    test = games[
        games["season"].between(2022, 2025)
        & played
        & games["result"].notna()
        & games["spread_line"].notna()
        & games["total_line"].notna()
    ]
    test_rows = test[
        [
            "game_id",
            "gameday",
            "home_team",
            "away_team",
            "location",
            "home_score",
            "away_score",
            "result",
            "spread_line",
            "total_line",
        ]
    ].to_dict("records")

    rng = np.random.default_rng(42)
    months = {}
    line_errors = []
    model_errors = []

    for game_row in test_rows:
        home_exp, away_exp = expected_points(games, game_row, best_n, best_h)
        if home_exp is None:
            continue
        result = float(game_row["result"])
        spread_line = float(game_row["spread_line"])
        total_line = float(game_row["total_line"])
        points = float(game_row["home_score"]) + float(game_row["away_score"])
        home_win_pct, avg_margin, avg_total, cover_pct, over_pct = simulate(
            home_exp, away_exp, spread_line, total_line, S, rng
        )
        del home_win_pct, avg_total
        month = f"{game_row['gameday'].year:04d}-{game_row['gameday'].month:02d}"
        if month not in months:
            months[month] = _blank_month()
        bucket = months[month]

        pick = _pick(cover_pct, over_pct)
        for side in pick.split(", "):
            if side == "home" or side == "away":
                units, label = _spread_units(result, spread_line, side)
                _add_grade(bucket, "spread", units, label)
            elif side == "over" or side == "under":
                units, label = _total_units(points, total_line, side)
                _add_grade(bucket, "total", units, label)

        base_units, _label = _spread_units(result, spread_line, "home")
        bucket["baseline1_units"] += base_units
        base_units, _label = _total_units(points, total_line, "over")
        bucket["baseline2_units"] += base_units

        line_errors.append(abs(spread_line - result))
        model_errors.append(abs(avg_margin - result))

    line_mae = float(np.mean(line_errors))
    model_mae = float(np.mean(model_errors))

    totals = _blank_month()
    for bucket in months.values():
        for key in totals:
            totals[key] += bucket[key]

    def win_rate(wins, losses):
        decided = wins + losses
        if decided == 0:
            return "n/a"
        return f"{100.0 * wins / decided:.1f}%"

    lines = [
        "# Backtest results",
        "",
        f"Chosen N: {best_n}",
        f"Chosen H: {best_h}",
        f"S: {S:.4f}",
        "",
        "break-even win rate at -110 is 52.4%",
        "",
        "## Month by month",
        "",
        "| year-month | spread bets | spread wins | spread units | total bets | total wins | total units | baseline 1 units | baseline 2 units |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for month in sorted(months):
        bucket = months[month]
        lines.append(
            "| {month} | {spread_bets} | {spread_wins} | {spread_units:.3f} | {total_bets} | {total_wins} | {total_units:.3f} | {baseline1_units:.3f} | {baseline2_units:.3f} |".format(
                month=month, **bucket
            )
        )
    lines.extend(
        [
            "",
            "## Full period",
            "",
            f"Spread bets: {totals['spread_bets']}",
            f"Spread wins: {totals['spread_wins']}",
            f"Spread losses: {totals['spread_losses']}",
            f"Spread pushes: {totals['spread_bets'] - totals['spread_wins'] - totals['spread_losses']}",
            f"Spread units: {totals['spread_units']:.3f}",
            f"Spread win rate: {win_rate(totals['spread_wins'], totals['spread_losses'])}",
            "",
            f"Total bets: {totals['total_bets']}",
            f"Total wins: {totals['total_wins']}",
            f"Total losses: {totals['total_losses']}",
            f"Total pushes: {totals['total_bets'] - totals['total_wins'] - totals['total_losses']}",
            f"Total units: {totals['total_units']:.3f}",
            f"Total win rate: {win_rate(totals['total_wins'], totals['total_losses'])}",
            "",
            f"Baseline 1 units: {totals['baseline1_units']:.3f}",
            f"Baseline 2 units: {totals['baseline2_units']:.3f}",
            "",
            f"Mean absolute error of spread_line versus the real result: {line_mae:.3f}",
            f"Mean absolute error of the model average margin versus the real result: {model_mae:.3f}",
            "",
        ]
    )
    with open("backtest_results.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    with open("settings.json", "w", encoding="utf-8") as handle:
        json.dump({"N": best_n, "H": best_h, "S": S}, handle)
        handle.write("\n")


def _send_push(message):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("push failed")
        return
    url = "https://ntfy.sh/" + topic
    request = urllib.request.Request(
        url, data=message.encode("utf-8"), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except Exception:
        print("push failed")


def predict():
    with open("settings.json", encoding="utf-8") as handle:
        settings = json.load(handle)
    N = int(settings["N"])
    H = settings["H"]
    S = float(settings["S"])

    games = load_games()
    today = datetime.now(ZoneInfo("America/New_York")).date()
    last_day = today + timedelta(days=7)
    start = pd.Timestamp(today)
    end = pd.Timestamp(last_day)
    upcoming = games[
        games["result"].isna()
        & (games["gameday"] >= start)
        & (games["gameday"] <= end)
    ]
    if upcoming.empty:
        print("no games")
        return

    rows = upcoming[
        [
            "game_id",
            "gameday",
            "home_team",
            "away_team",
            "location",
            "spread_line",
            "total_line",
        ]
    ].to_dict("records")

    rng = np.random.default_rng(42)
    output_rows = []
    alert_lines = []
    for game_row in rows:
        home_exp, away_exp = expected_points(games, game_row, N, H)
        if home_exp is None:
            continue
        has_spread = pd.notna(game_row["spread_line"])
        has_total = pd.notna(game_row["total_line"])
        spread_line = float(game_row["spread_line"]) if has_spread else 0.0
        total_line = float(game_row["total_line"]) if has_total else 0.0
        home_win_pct, avg_margin, _avg_total, cover_pct, over_pct = simulate(
            home_exp, away_exp, spread_line, total_line, S, rng
        )
        if not has_spread:
            cover_pct = None
        if not has_total:
            over_pct = None
        pick = _pick(cover_pct, over_pct)
        game = (
            f"{game_row['away_team']} @ {game_row['home_team']} "
            f"{game_row['gameday'].date()} {game_row['game_id']}"
        )
        output_rows.append(
            {
                "game": game,
                "home_win_pct": home_win_pct,
                "avg_margin": avg_margin,
                "spread_line": spread_line if has_spread else None,
                "cover_pct": cover_pct,
                "total_line": total_line if has_total else None,
                "over_pct": over_pct,
                "pick": pick,
            }
        )
        if pick != "no bet":
            alert_lines.append(f"{game}: {pick}")

    def cell(value, digits):
        if value is None:
            return ""
        return f"{value:.{digits}f}"

    lines = [
        "# Predictions",
        "",
        "| game | home win % | average margin | spread_line | cover % | total_line | over % | pick |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in output_rows:
        lines.append(
            "| {game} | {home} | {margin} | {spread} | {cover} | {total} | {over} | {pick} |".format(
                game=row["game"],
                home=cell(row["home_win_pct"], 1),
                margin=cell(row["avg_margin"], 2),
                spread=cell(row["spread_line"], 1),
                cover=cell(row["cover_pct"], 1),
                total=cell(row["total_line"], 1),
                over=cell(row["over_pct"], 1),
                pick=row["pick"],
            )
        )
    lines.append("")
    with open("predictions.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    if alert_lines:
        _send_push("\n".join(alert_lines))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "backtest":
        backtest()
    elif len(sys.argv) == 2 and sys.argv[1] == "predict":
        predict()
    else:
        sys.exit(1)
