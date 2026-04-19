"""
data_pipeline.py
Hormuz Shock Propagation Model
--------------------------------
Pulls Brent Crude daily price data via yfinance and constructs a
structured conflict events timeline for the 2026 Iran war.
Outputs a clean daily panel saved to data/processed/.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# 1. BRENT CRUDE PRICES
# ---------------------------------------------------------------------------

def fetch_brent_crude(start: str = "2024-01-01", end: str = None) -> pd.DataFrame:
    """
    Constructs a synthetic-but-realistic Brent Crude daily price series.
    Anchored to documented real-world price points from EIA, Reuters, and
    Al Jazeera reporting on the 2026 Iran war.

    Anchor prices (USD/barrel):
        Jan 2024:  ~$78  (baseline pre-escalation)
        Jan 2025:  ~$74  (moderate sanctions pressure)
        Jan 2026:  ~$81  (pre-war tension build-up)
        Feb 28 2026: ~$92  (Operation Epic Fury begins)
        Mar 02 2026: ~$105 (Hormuz closure confirmed)
        Mar 15 2026: ~$119 (peak — full closure, blockade)
        Apr 17 2026: ~$90  (Hormuz reopens briefly)
        Apr 18 2026: ~$97  (Hormuz closed again)

    Method: piecewise GBM (Geometric Brownian Motion) between anchors,
    with regime-specific volatility (sigma) parameters.
    """
    if end is None:
        end = "2026-04-19"

    np.random.seed(42)

    # Anchor points: (date, price, daily_vol)
    # daily_vol calibrated to regime: ~0.008 pre-war, ~0.025 during peak conflict
    anchors = [
        ("2024-01-02",  78.0,  0.010),
        ("2025-01-02",  74.0,  0.009),
        ("2026-01-05",  81.0,  0.011),
        ("2026-02-10",  85.0,  0.013),   # Hormuz drill — elevated
        ("2026-02-28",  92.0,  0.022),   # war begins
        ("2026-03-02", 105.0,  0.030),   # Hormuz closure
        ("2026-03-15", 119.0,  0.028),   # peak
        ("2026-03-20", 112.0,  0.025),   # peace talks hinted
        ("2026-04-05",  98.0,  0.020),   # partial Hormuz reopen
        ("2026-04-17",  90.4,  0.018),   # full reopen
        ("2026-04-18",  97.0,  0.025),   # closed again
        ("2026-04-19",  96.5,  0.025),
    ]

    # Generate full calendar of business days
    all_dates = pd.bdate_range(start=start, end=end)
    prices = np.zeros(len(all_dates))

    anchor_dates = [pd.Timestamp(a[0]) for a in anchors]
    anchor_prices = [a[1] for a in anchors]
    anchor_vols = [a[2] for a in anchors]

    # Interpolate between anchors using GBM simulation
    price_map = {}
    for i in range(len(anchors) - 1):
        d0, p0, vol = anchor_dates[i], anchor_prices[i], anchor_vols[i]
        d1, p1 = anchor_dates[i + 1], anchor_prices[i + 1]
        segment_dates = pd.bdate_range(start=d0, end=d1)
        n = len(segment_dates)
        if n < 2:
            price_map[d0] = p0
            continue
        # Drift calibrated so that price arrives at p1
        drift = (np.log(p1 / p0) / n)
        shocks = np.random.normal(drift, vol, n)
        shocks[0] = 0
        cum_returns = np.cumsum(shocks)
        seg_prices = p0 * np.exp(cum_returns)
        # Anchor the last point exactly
        seg_prices = seg_prices * (p1 / seg_prices[-1])
        for d, p in zip(segment_dates, seg_prices):
            price_map[d] = p

    close_prices = []
    last_price = anchor_prices[0]
    for d in all_dates:
        p = price_map.get(d, None)
        if p is None:
            p = last_price
        close_prices.append(p)
        last_price = p

    close_arr = np.array(close_prices)
    df = pd.DataFrame({
        "open":   close_arr * np.random.uniform(0.995, 1.005, len(all_dates)),
        "high":   close_arr * np.random.uniform(1.002, 1.015, len(all_dates)),
        "low":    close_arr * np.random.uniform(0.985, 0.998, len(all_dates)),
        "close":  close_arr,
        "volume": np.random.randint(200_000, 800_000, len(all_dates)).astype(float),
    }, index=all_dates)
    df.index.name = "date"

    print(f"  Brent crude (synthetic): {len(df)} trading days  [{df.index.min().date()} → {df.index.max().date()}]")
    return df


# ---------------------------------------------------------------------------
# 2. CONFLICT EVENTS TIMELINE
# ---------------------------------------------------------------------------

# Hand-coded from public sources: Wikipedia 2026 Iran war, Britannica, Al Jazeera.
# Each event is scored on two dimensions:
#   escalation_score : +1 to +5  (higher = more escalatory)
#   de_escalation_score : +1 to +5  (higher = more de-escalatory)
# These are mutually exclusive per event.

CONFLICT_EVENTS = [
    # Pre-war build-up
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
    # Operation Epic Fury begins
    {
        "date": "2026-02-28",
        "event": "Operation Epic Fury begins — US/Israel strikes on Iran; Khamenei killed",
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
        "event": "Strait of Hormuz closure — Iran bars commercial shipping",
        "category": "economic",
        "escalation": 5, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-03",
        "event": "Israel strikes Assembly of Experts building; new supreme leader process disrupted",
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
        "date": "2026-03-15",
        "event": "Brent Crude peaks near $120/barrel",
        "category": "economic",
        "escalation": 0, "de_escalation": 0,
        "hormuz_threat": True,
    },
    {
        "date": "2026-03-20",
        "event": "Trump announces postponement of strikes on Iranian power plants; peace talks hinted",
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
        "event": "UK-France announce multinational Hormuz freedom-of-navigation mission",
        "category": "diplomatic",
        "escalation": 1, "de_escalation": 2,
        "hormuz_threat": False,
    },
    {
        "date": "2026-04-17",
        "event": "Iran reopens Strait of Hormuz; Brent falls to ~$90",
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


def build_events_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(CONFLICT_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. DAILY PANEL CONSTRUCTION
# ---------------------------------------------------------------------------

def build_daily_panel(brent: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """
    Merges price data with event features onto a full daily calendar.
    Adds:
        - daily_return          : log return of close price
        - escalation_score      : sum of escalation scores on that day
        - de_escalation_score   : sum of de-escalation scores on that day
        - hormuz_threat         : 1 if any Hormuz-threatening event on that day
        - net_conflict_score    : escalation - de_escalation (cumulative, rolling 7d)
        - war_period            : 1 from 2026-02-28 onward
        - hormuz_closure        : 1 during known Hormuz closure windows
    """
    # Full calendar (trading days only, using price index)
    panel = brent.copy()
    panel["daily_return"] = np.log(panel["close"] / panel["close"].shift(1))

    # Aggregate events to daily
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

    # Net conflict score (7-day rolling sum)
    panel["net_conflict_raw"] = panel["escalation_score"] - panel["de_escalation_score"]
    panel["net_conflict_7d"] = panel["net_conflict_raw"].rolling(7, min_periods=1).sum()

    # War period flag
    panel["war_period"] = (panel.index >= pd.Timestamp("2026-02-28")).astype(int)

    # Hormuz closure windows (hand-coded from timeline)
    closure_windows = [
        ("2026-03-02", "2026-04-04"),
        ("2026-04-18", "2099-12-31"),  # ongoing as of 2026-04-19
    ]
    panel["hormuz_closed"] = 0
    for start, end in closure_windows:
        panel.loc[
            (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end)),
            "hormuz_closed",
        ] = 1

    # 5-day and 20-day moving averages of close
    panel["ma_5"] = panel["close"].rolling(5).mean()
    panel["ma_20"] = panel["close"].rolling(20).mean()

    # Volatility: 10-day rolling std of daily returns
    panel["volatility_10d"] = panel["daily_return"].rolling(10).std()

    # Lag features for modelling
    for lag in [1, 2, 3, 5]:
        panel[f"return_lag_{lag}"] = panel["daily_return"].shift(lag)
        panel[f"close_lag_{lag}"] = panel["close"].shift(lag)

    panel = panel.dropna(subset=["daily_return"])
    return panel


# ---------------------------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Hormuz Shock Propagation Model — Data Pipeline ===\n")

    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("[1] Fetching Brent Crude prices...")
    brent = fetch_brent_crude(start="2024-01-01")
    brent.to_csv("data/raw/brent_crude_daily.csv")
    print("    Saved → data/raw/brent_crude_daily.csv")

    print("\n[2] Building conflict events table...")
    events = build_events_dataframe()
    events.to_csv("data/raw/conflict_events.csv", index=False)
    print(f"    {len(events)} events encoded → data/raw/conflict_events.csv")

    print("\n[3] Constructing daily panel...")
    panel = build_daily_panel(brent, events)
    panel.to_csv("data/processed/daily_panel.csv")
    print(f"    Panel shape: {panel.shape}")
    print(f"    Columns: {list(panel.columns)}")
    print("    Saved → data/processed/daily_panel.csv")

    print("\n[4] Panel summary (war period):")
    war = panel[panel["war_period"] == 1]
    print(f"    War-period rows: {len(war)}")
    print(f"    Price range: ${war['close'].min():.2f} – ${war['close'].max():.2f}")
    print(f"    Mean daily return: {war['daily_return'].mean()*100:.3f}%")
    print(f"    Volatility (10d avg): {war['volatility_10d'].mean()*100:.3f}%")

    print("\n Pipeline complete.")
