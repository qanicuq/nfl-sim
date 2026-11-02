"""Test the player-prop model on past Kalshi prices. Usage: python prop_backtest.py download | test"""
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import edges
import nfl_sim

KALSHI = edges.KALSHI
DATA = "data/kalshi_props.csv.gz"
JUDGE_FROM = "2026-10"
# Kalshi spellings of nflverse codes (Kalshi switched the Rams from LA to LAR in 2026).
KALSHI_TEAM = {"LA": ("LA", "LAR"), "JAX": ("JAX", "JAC")}
ET = ZoneInfo("America/New_York")
# English month names so the event key matches Kalshi (NOV) on any locale.
MONTHS = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)
COLUMNS = [
    "event_ticker", "series", "ticker", "title", "player", "strike", "result",
    "season", "week", "away", "home", "kickoff", "yes_ask", "yes_bid",
]
RESULT_COLUMNS = [
    "month", "ticker", "stat", "raw", "result", "yes_price", "no_price",
    "yes_fee", "no_fee", "yes_profit", "no_profit",
]
SUMMARY_COLUMNS = [
    "month", "markets", "model bets", "model YES bets", "model NO bets",
    "model wins", "model profit", "always-YES profit", "always-NO profit",
]
# Lower bound included. Upper bound excluded, except 1.0 stays in the last bucket.
BUCKETS = (
    ("0-0.2", 0.0, 0.2, False),
    ("0.2-0.4", 0.2, 0.4, False),
    ("0.4-0.6", 0.4, 0.6, False),
    ("0.6-0.8", 0.6, 0.8, False),
    ("0.8-1.0", 0.8, 1.0, True),
)


def kickoff_lookup(games):
    kept = games[games["gametime"].notna()]
    kept = kept[kept["gametime"].astype(str).str.strip() != ""]
    lookup = {}
    for row in kept.itertuples(index=False):
        day = row.gameday
        hour, minute = str(row.gametime).strip().split(":")[:2]
        away = row.away_team
        home = row.home_team
        date = f"{int(day.year) % 100:02d}{MONTHS[int(day.month) - 1]}{int(day.day):02d}"
        kickoff = datetime(
            int(day.year), int(day.month), int(day.day),
            int(hour), int(minute), tzinfo=ET,
        )
        for a in KALSHI_TEAM.get(away, (away,)):
            for h in KALSHI_TEAM.get(home, (home,)):
                lookup[date + a + h] = (int(row.season), int(row.week), away, home, kickoff)
    return lookup


def _close_dollars(book):
    if not isinstance(book, dict):
        return None
    raw = book.get("close_dollars")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _settled_event_tickers(series):
    cursor = ""
    while True:
        url = f"{KALSHI}/events?series_ticker={series}&status=settled&limit=200"
        if cursor:
            url += "&cursor=" + cursor
        payload = edges.get_json(url)
        time.sleep(0.3)
        for event in payload.get("events") or []:
            ticker = event.get("event_ticker")
            if ticker:
                yield ticker
        cursor = payload.get("cursor") or ""
        if not cursor:
            break


def download():
    games = nfl_sim.load_games()
    lookup = kickoff_lookup(games)
    rows = []
    unmatched = []
    seen = 0
    for series in edges.KALSHI_PLAYER_SERIES:
        for event_ticker in _settled_event_tickers(series):
            seen += 1
            code = event_ticker.split("-", 1)[1] if "-" in event_ticker else ""
            found = lookup.get(code)
            if found is None:
                unmatched.append(event_ticker)
            else:
                season, week, away, home, kickoff = found
                entry = int(kickoff.timestamp()) - 3600
                payload = edges.get_json(
                    f"{KALSHI}/markets?event_ticker={event_ticker}&limit=1000"
                )
                time.sleep(0.3)
                markets = payload.get("markets") or []
                info = {}
                tickers = []
                for market in markets:
                    ticker = market.get("ticker")
                    if not ticker:
                        continue
                    info[ticker] = market
                    tickers.append(ticker)
                for start in range(0, len(tickers), 100):
                    group = tickers[start:start + 100]
                    url = (
                        f"{KALSHI}/markets/candlesticks?market_tickers={','.join(group)}"
                        f"&start_ts={entry - 6 * 3600}&end_ts={entry}&period_interval=60"
                    )
                    candles = edges.get_json(url)
                    time.sleep(0.3)
                    for item in candles.get("markets") or []:
                        ticker = item.get("market_ticker")
                        market = info.get(ticker)
                        if market is None:
                            continue
                        chosen = None
                        for candle in item.get("candlesticks") or []:
                            ts = candle.get("end_period_ts")
                            if ts is None or ts > entry:
                                continue
                            if chosen is None or ts >= chosen["end_period_ts"]:
                                chosen = candle
                        if chosen is None:
                            continue
                        yes_ask = _close_dollars(chosen.get("yes_ask"))
                        yes_bid = _close_dollars(chosen.get("yes_bid"))
                        if yes_ask is None or yes_bid is None:
                            continue
                        title = market.get("title") or ""
                        rows.append({
                            "event_ticker": event_ticker,
                            "series": series,
                            "ticker": ticker,
                            "title": title,
                            "player": title.split(":", 1)[0].strip(),
                            "strike": market.get("floor_strike"),
                            "result": market.get("result"),
                            "season": season,
                            "week": week,
                            "away": away,
                            "home": home,
                            "kickoff": kickoff.isoformat(),
                            "yes_ask": yes_ask,
                            "yes_bid": yes_bid,
                        })
            if seen % 50 == 0:
                print(f"{seen} events, {len(rows)} rows")
    os.makedirs("data", exist_ok=True)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame.to_csv(DATA, index=False, compression="gzip")
    print(f"saved {len(frame)} rows")
    print(f"unmatched events: {len(unmatched)}")
    for ticker in unmatched[:20]:
        print(ticker)


def _summary(label, part):
    return {
        "month": label,
        "markets": int(len(part)),
        "model bets": int((part["bet"] != "").sum()),
        "model YES bets": int((part["bet"] == "YES").sum()),
        "model NO bets": int((part["bet"] == "NO").sum()),
        "model wins": int((part["model_profit"] > 0).sum()),
        "model profit": float(part["model_profit"].sum()),
        "always-YES profit": float(part["yes_profit"].sum()),
        "always-NO profit": float(part["no_profit"].sum()),
    }


def _markdown_table(headers, records):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    money = {"model profit", "always-YES profit", "always-NO profit"}
    for record in records:
        cells = []
        for header in headers:
            value = record[header]
            if header in money:
                cells.append(f"{float(value):.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _check_rows(table):
    bets = table[table["bet"] != ""].copy()
    if len(bets):
        yes_side = bets["bet"].to_numpy() == "YES"
        bets["chance"] = np.where(yes_side, bets["p"], 1.0 - bets["p"])
        bets["won"] = np.where(
            yes_side,
            bets["result"].to_numpy() == "yes",
            bets["result"].to_numpy() == "no",
        )
    rows = []
    for label, low, high, include_high in BUCKETS:
        if len(bets) == 0:
            chosen = bets
        elif include_high:
            chosen = bets[(bets["chance"] >= low) & (bets["chance"] <= high)]
        else:
            chosen = bets[(bets["chance"] >= low) & (bets["chance"] < high)]
        count = int(len(chosen))
        if count == 0:
            average = "n/a"
            won = "n/a"
        else:
            average = f"{float(chosen['chance'].mean()):.3f}"
            won = f"{float(np.mean(chosen['won'])):.3f}"
        rows.append({
            "bucket": label,
            "bets": count,
            "average chance": average,
            "actually won": won,
        })
    return rows


def _write_report(month_rows, totals, check_rows):
    lines = [
        "Judge the model ONLY on the JUDGE row. If model profit there is not above both baselines, the player model has no proven edge.",
        "",
        "# Prop backtest",
        "",
        "The entry price is the ask 1 hour before kickoff, with Kalshi fees taken out.",
        "",
        "Only one market per player and stat is used (the one closest to 50 cents).",
        "",
        "## Month by month",
        "",
    ]
    lines.extend(_markdown_table(SUMMARY_COLUMNS, month_rows))
    lines.extend(["", "## Totals", ""])
    lines.extend(_markdown_table(SUMMARY_COLUMNS, totals))
    lines.extend(["", "## Check", ""])
    lines.extend(_markdown_table(
        ["bucket", "bets", "average chance", "actually won"], check_rows,
    ))
    lines.append("")
    with open("prop_backtest_results.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def fit_prop_calibration(setup):
    # Fit chance = 1 / (1 + exp(-(a + b * logit(raw)))) on SETUP months only, per stat.
    hit = (setup["result"] == "yes").to_numpy(dtype=float)
    cal = {"_all": list(edges.fit_line(edges.logit(setup["raw"].to_numpy()), hit))}
    for stat, part in setup.groupby("stat"):
        if len(part) >= 150:
            part_hit = (part["result"] == "yes").to_numpy(dtype=float)
            cal[stat] = list(edges.fit_line(edges.logit(part["raw"].to_numpy()), part_hit))
    with open(edges.PROP_CAL_FILE, "w", encoding="utf-8") as handle:
        json.dump(cal, handle, indent=1)
    return cal


def score(table):
    setup = table[table["month"] < JUDGE_FROM]
    cal = fit_prop_calibration(setup)
    ab = np.array([cal.get(stat, cal["_all"]) for stat in table["stat"]])
    table["p"] = 1 / (1 + np.exp(-(ab[:, 0] + ab[:, 1] * edges.logit(table["raw"].to_numpy()))))
    yes_edge = table["p"] - table["yes_price"] - table["yes_fee"]
    no_edge = (1 - table["p"]) - table["no_price"] - table["no_fee"]
    take_yes = (yes_edge >= no_edge) & (yes_edge >= edges.FLAG)
    take_no = (no_edge > yes_edge) & (no_edge >= edges.FLAG)
    table["bet"] = np.where(take_yes, "YES", np.where(take_no, "NO", ""))
    table["model_profit"] = np.where(take_yes, table["yes_profit"], np.where(take_no, table["no_profit"], 0.0))
    return table


def test():
    data = pd.read_csv(DATA)
    data["mid"] = (data["yes_ask"] + data["yes_bid"]) / 2
    data["month"] = data["kickoff"].astype(str).str.slice(0, 7)
    rng = np.random.default_rng(42)
    results = []
    ordered = data.sort_values(["season", "week", "event_ticker", "ticker"])
    for (season, week), week_rows in ordered.groupby(["season", "week"], sort=True):
        season = int(season)
        week = int(week)
        stats = edges.load_player_games(season, week)
        for _event_ticker, event in week_rows.groupby("event_ticker", sort=True):
            away = event["away"].iloc[0]
            home = event["home"].iloc[0]
            injuries = edges.load_injuries(season, week, {away, home})
            cache = {}
            for player, markets in event.groupby("player", sort=True):
                status = injuries.get(edges.norm_name(str(player)), "")
                if status == "Out" or status == "Doubtful":
                    continue
                market = markets.loc[(markets["mid"] - 0.50).abs().idxmin()]
                stat = edges.KALSHI_PLAYER_SERIES[market["series"]]
                sims, games_used = edges.player_sims(stats, str(player), stat, rng, cache)
                if sims is None:
                    continue
                strike = market["strike"]
                if pd.isna(strike):
                    continue
                raw = edges.player_raw_prob(sims, float(strike))
                yes_price = float(market["yes_ask"])
                no_price = 1.0 - float(market["yes_bid"])
                if not (0 < yes_price < 1 and 0 < no_price < 1):
                    continue
                yes_fee = edges.kalshi_fee(yes_price, 1.0)
                no_fee = edges.kalshi_fee(no_price, 1.0)
                result = market["result"]
                yes_profit = (1.0 if result == "yes" else 0.0) - yes_price - yes_fee
                no_profit = (1.0 if result == "no" else 0.0) - no_price - no_fee
                results.append({
                    "month": market["month"],
                    "ticker": market["ticker"],
                    "stat": stat,
                    "raw": raw,
                    "result": result,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "yes_fee": yes_fee,
                    "no_fee": no_fee,
                    "yes_profit": float(yes_profit),
                    "no_profit": float(no_profit),
                })
    table = pd.DataFrame(results, columns=RESULT_COLUMNS)
    table = score(table)
    if len(table):
        month_rows = [
            _summary(month, part)
            for month, part in table.groupby("month", sort=True)
        ]
        setup = table[table["month"] < JUDGE_FROM]
        judge = table[table["month"] >= JUDGE_FROM]
    else:
        month_rows = []
        setup = table
        judge = table
    totals = [_summary("SETUP", setup), _summary("JUDGE", judge)]
    check_rows = _check_rows(table)
    _write_report(month_rows, totals, check_rows)
    judge_row = totals[1]
    message = (
        "Prop backtest done. JUDGE: model {model:.2f}, always-YES {yes:.2f}, always-NO {no:.2f}"
    ).format(
        model=judge_row["model profit"],
        yes=judge_row["always-YES profit"],
        no=judge_row["always-NO profit"],
    )
    print(message)
    nfl_sim._send_push(message)


if __name__ == "__main__":
    usage = "Usage: python prop_backtest.py download | test"
    if len(sys.argv) < 2:
        sys.exit(usage)
    command = sys.argv[1]
    if command == "download":
        download()
    elif command == "test":
        test()
    else:
        sys.exit(usage)
