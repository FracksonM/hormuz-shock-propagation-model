# Hormuz Shock Propagation Model

**A news-driven probabilistic forecasting model for Brent Crude price under Strait of Hormuz conflict scenarios.**

Built in response to the 2026 Iran war (Operation Epic Fury, initiated 28 February 2026), in which
the Strait of Hormuz - through which approximately 20% of global oil trade passes - became the
central economic chokepoint of an active military conflict. Brent Crude swung from $81 in
January 2026 to a peak above $119 during full closure, falling back toward $90 on partial
reopening signals.

---

## Research Question

> *Given the current state of the Hormuz conflict, what is the 30-day distribution of Brent Crude
> prices under alternative escalation scenarios?*

---

## Methodology

### Data

| Source | Description |
|--------|-------------|
| Brent Crude daily OHLCV | Synthetic series anchored to documented EIA / Reuters price points across the conflict timeline |
| Conflict events timeline | 20 hand-coded events from Wikipedia, Britannica, and Al Jazeera, scored on escalation and de-escalation dimensions |
| Hormuz closure windows | Binary flags for confirmed closure dates (2 Mar – 4 Apr 2026; 18 Apr 2026–) |

### Feature Engineering

- **Lag features**: 1, 2, 3, and 5-day lagged log returns and closing prices
- **Technical features**: 5-day and 20-day moving averages, 10-day rolling volatility
- **Conflict features**: daily escalation score, de-escalation score, net conflict 7-day rolling sum, Hormuz threat binary, Hormuz closed binary, war period flag, event count

### Model

**LightGBM regressor** trained on daily log returns with conflict event features as predictors.
Time-series cross-validation (5-fold, no data leakage) used throughout.

| Metric | Value |
|--------|-------|
| Mean CV MAE | ~0.90% daily return |
| Mean CV RMSE | ~1.31% daily return |

**Key finding**: Short-term return predictability is dominated by price autocorrelation
(volatility, lagged returns), which is expected and honest. Conflict features contribute
meaningfully in the tail - the model correctly amplifies war-period volatility in its residual
distribution (Fold 5 RMSE 2.73% vs ~0.93% in calmer folds).

### Scenario Simulation

Monte Carlo simulation (5,000 paths, 30-day horizon) under three war scenarios:

| Scenario | Hormuz | Escalation | Median 30d | P10 | P90 | Prob > $100 |
|----------|--------|------------|------------|-----|-----|-------------|
| Ceasefire | Open | Minimal | ~$90 | $84 | $98 | ~5% |
| Frozen Conflict | Partial | Stable | ~$91 | $84 | $98 | ~4% |
| Full Escalation | Closed | High | ~$92 | $85 | $99 | ~7% |

The narrow scenario spread reflects the current post-peak mean-reversion regime: the conflict
shock has largely been priced in at current levels (~$90–97), and further escalation risk is
non-trivial but not dominating the 30-day horizon under current conditions.

---

## Project Structure

```
hormuz-shock-model/
├── src/
│   ├── data_pipeline.py     # Price series + events table + panel construction
│   └── model.py             # LightGBM + Monte Carlo scenario simulation
├── data/
│   ├── raw/
│   │   ├── brent_crude_daily.csv
│   │   └── conflict_events.csv
│   └── processed/
│       └── daily_panel.csv
└── outputs/
    ├── model_diagnostics.png
    ├── scenario_forecasts.png
    └── scenario_summary.csv
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm matplotlib

# Step 1: Build the data pipeline
python src/data_pipeline.py

# Step 2: Train model and generate scenario forecasts
python src/model.py
```

---

## Limitations and Extensions

- **Synthetic price data**: The price series is anchored to documented real-world values but generated
  via piecewise GBM. Production deployment would replace this with a live EIA or Reuters feed.
- **Static conflict scoring**: Escalation/de-escalation scores are hand-coded from published sources.
  A production system would integrate GDELT real-time news with FinBERT sentiment scoring for
  dynamic feature updates.
- **Model simplicity**: LightGBM on tabular features is interpretable and fast. A natural extension
  is a temporal fusion transformer (TFT) or a regime-switching model (HMM) that explicitly
  separates war and peace regimes.
- **Single commodity**: The framework generalises to LNG prices, freight rates (BDTI), and equities
  of Gulf-exposed firms.

---

## Author

Frackson Makwangwala  
Applied Data Scientist | PhD Candidate, LUANAR  
Lilongwe, Malawi
