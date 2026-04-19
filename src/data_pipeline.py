"""
data_pipeline.py
Hormuz Shock Propagation Model

Pulls real Brent Crude daily prices from Yahoo Finance and builds a
structured conflict events timeline for the 2026 Iran war. The output
is a clean daily panel saved to data/processed/ that feeds directly
into the modelling notebooks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime


def fetch_brent_crude(start="2024-01-01", end=None):
    # Pull daily OHLCV for Brent Crude front-month futures from Yahoo Finance.
    # BZ=F is the standard ticker. We strip timezone info to keep the index clean.
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    ticker = yf.Ticker("BZ=F")
    df = ticker.history(start=start, end=end, interval="1d")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[df["close"] > 0].dropna(subset=["close"])

    print(f"  Brent crude: {len(df)} trading days [{df.index.min().date()} to {df.index.max().date()}]")
    return df


# Each event is scored on escalation (1-5) and de-escalation (1-5).
# Sources: Wikipedia 2026 Iran war, Britannica, Al Jazeera.
# These two dimensions are kept separate rather than collapsed into a single
# score because the model treats them as distinct signals.

CONFLICT_EVENTS = [
    {
        "date": "2025-10-01",
        "event": "US-Iran nuclear talks collapse; new sanctions imposed",
        "category": "diplomatic",
        "escalation": 2, "de_escalation": 0,
        "hormuz_threat": False,
    },
    {
        "date": "2025-12-15",
        "event": "Israel strikes Iranian proxy sites in Syria",
        "category": "military",
        "escalation": 2, "de_escalation": 0,
        "hormuz_threat": False,
    },
    {
        "date": "2026-01-20",
        "event": "US deploys carrier strike group to Persian Gulf",
        "category": "military",
        "escalation": 3, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-02-10",
        "event": "Iran conducts large-scale Hormuz closure drill",
        "category": "military",
        "escalation": 3, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-02-28",
        "event": "Operation Epic Fury begins - US and Israel strike Iran; Khamenei killed",
        "category": "military",
        "escalation": 5, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-01",
        "event": "Iran launches hundreds of drones and ballistic missiles across the region",
        "category": "military",
        "escalation": 5, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-01",
        "event": "Dubai International Airport struck by drone; flights suspended",
        "category": "infrastructure",
        "escalation": 4, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-02",
        "event": "Strait of Hormuz closure - Iran bars commercial shipping",
        "category": "economic",
        "escalation": 5, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-03",
        "event": "Israel strikes Assembly of Experts building; leadership succession disrupted",
        "category": "military",
        "escalation": 4, "de_escalation": 0,
        "hormuz_threat": False,
    },
    {
        "date": "2026-03-05",
        "event": "Mojtaba Khamenei appointed as new supreme leader",
        "category": "political",
        "escalation": 2, "de_escalation": 0,
        "hormuz_threat": False,
    },
    {
        "date": "2026-03-07",
        "event": "US naval blockade of Iranian ports begins",
        "category": "military",
        "escalation": 4, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-10",
        "event": "USS Charlotte sinks Iranian frigate IRIS Dena in Indian Ocean",
        "category": "military",
        "escalation": 4, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-20",
        "event": "Trump postpones strikes on Iranian power plants; peace talks hinted",
        "category": "diplomatic",
        "escalation": 0, "de_escalation": 3,
        "hormuz_threat": False,
    },
    {
        "date": "2026-03-24",
        "event": "Iran rejects US 15-point peace plan; ceasefire conditional on Lebanon",
        "category": "diplomatic",
        "escalation": 2, "de_escalation": 0,
        "hormuz_threat": False,
    },
    {
        "date": "2026-03-28",
        "event": "Houthis fire ballistic missiles at Israel; 2,500 US Marines arrive in region",
        "category": "military",
        "escalation": 3, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-04-05",
        "event": "Iran partially reopens Strait of Hormuz for non-hostile commercial vessels",
        "category": "economic",
        "escalation": 0, "de_escalation": 4,
        "hormuz_threat": False,
    },
    {
        "date": "2026-04-11",
        "event": "UK and France announce multinational Hormuz freedom-of-navigation mission",
        "category": "diplomatic",
        "escalation": 1, "de_escalation": 2,
        "hormuz_threat": False,
    },
    {
        "date": "2026-04-17",
        "event": "Iran reopens Strait of Hormuz fully; Brent falls sharply",
        "category": "economic",
        "escalation": 0, "de_escalation": 4,
        "hormuz_threat": False,
    },
    {
        "date": "2026-04-18",
        "event": "Iran closes Strait again pending US port blockade lift",
        "category": "economic",
        "escalation": 3, "de_escalation": 0,
        "hormuz_threat": True,
    },
]


def build_events_dataframe():
    df = pd.DataFrame(CONFLICT_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_daily_panel(brent, events):
    # Start from the price data and join everything onto it.
    # We use the price index as the calendar so we only have trading days.
    panel = brent.copy()
    panel["daily_return"] = np.log(panel["close"] / panel["close"].shift(1))

    # Aggregate events to daily totals so multiple events on the same day stack correctly
    esc = (
        events.groupby("date")
        .agg(
            escalation_score=("escalation", "sum"),
            de_escalation_score=("de_escalation", "sum"),
            hormuz_threat=("hormuz_threat", "max"),
            n_events=("event", "count"),
        )
        .astype({"hormuz_threat": int})
    )

    panel = panel.join(esc, how="left")
    panel[["escalation_score", "de_escalation_score", "hormuz_threat", "n_events"]] = (
        panel[["escalation_score", "de_escalation_score", "hormuz_threat", "n_events"]].fillna(0)
    )

    panel["net_conflict_raw"] = panel["escalation_score"] - panel["de_escalation_score"]
    panel["net_conflict_7d"]  = panel["net_conflict_raw"].rolling(7, min_periods=1).sum()

    panel["war_period"] = (panel.index >= pd.Timestamp("2026-02-28")).astype(int)

    # Hormuz closure windows confirmed from news reporting
    closure_windows = [
        ("2026-03-02", "2026-04-04"),
        ("2026-04-18", "2099-12-31"),
    ]
    panel["hormuz_closed"] = 0
    for start, end in closure_windows:
        panel.loc[
            (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end)),
            "hormuz_closed",
        ] = 1

    panel["ma_5"]           = panel["close"].rolling(5).mean()
    panel["ma_20"]          = panel["close"].rolling(20).mean()
    panel["volatility_10d"] = panel["daily_return"].rolling(10).std()

    for lag in [1, 2, 3, 5]:
        panel[f"return_lag_{lag}"] = panel["daily_return"].shift(lag)
        panel[f"close_lag_{lag}"]  = panel["close"].shift(lag)

    panel = panel.dropna(subset=["daily_return"])
    return panel


if __name__ == "__main__":
    print("Hormuz Shock Propagation Model - Data Pipeline\n")

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("[1] Fetching Brent Crude prices from Yahoo Finance...")
    brent = fetch_brent_crude(start="2024-01-01")
    brent.to_csv("data/raw/brent_crude_daily.csv")
    print("    Saved to data/raw/brent_crude_daily.csv")

    print("\n[2] Building conflict events table...")
    events = build_events_dataframe()
    events.to_csv("data/raw/conflict_events.csv", index=False)
    print(f"    {len(events)} events saved to data/raw/conflict_events.csv")

    print("\n[3] Building daily panel...")
    panel = build_daily_panel(brent, events)
    panel.to_csv("data/processed/daily_panel.csv")
    print(f"    Shape: {panel.shape}")
    print(f"    Columns: {list(panel.columns)}")
    print("    Saved to data/processed/daily_panel.csv")

    print("\n[4] War period summary:")
    war = panel[panel["war_period"] == 1]
    print(f"    Trading days: {len(war)}")
    print(f"    Price range: ${war['close'].min():.2f} to ${war['close'].max():.2f}")
    print(f"    Mean daily return: {war['daily_return'].mean()*100:.3f}%")
    print(f"    Mean 10d volatility: {war['volatility_10d'].mean()*100:.3f}%")

    print("\nDone.")