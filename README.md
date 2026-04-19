# Hormuz Shock Propagation Model

A news-driven probabilistic forecasting model for Brent Crude price under
Strait of Hormuz conflict scenarios.

Built in response to the 2026 Iran war (Operation Epic Fury, initiated 28
February 2026), in which the Strait of Hormuz - through which approximately
20% of global oil trade passes - became the central economic chokepoint of
an active military conflict. Brent Crude moved from $81 in January 2026 to
a peak above $118 during full closure, falling back toward $90 on partial
reopening signals.

## Research Question

Given the current state of the Hormuz conflict, what is the 30-day
distribution of Brent Crude prices under alternative escalation scenarios?

## Data

| Source | Description |
|--------|-------------|
| Brent Crude daily OHLCV | Live prices via yfinance (BZ=F), Jan 2024 to present |
| Conflict events timeline | 19 hand-coded events from Wikipedia, Britannica, and Al Jazeera, scored on escalation and de-escalation dimensions |
| Hormuz closure windows | Binary flags for confirmed closure dates (2 Mar to 4 Apr 2026; 18 Apr 2026 onward) |

## Feature Engineering

Three groups of features feed the model:

**Price and momentum**: 1, 2, 3, and 5-day lagged log returns and closing
prices, 5-day and 20-day moving averages, 10-day rolling volatility.

**Conflict features**: daily escalation score, de-escalation score, net
conflict 7-day rolling sum, Hormuz threat binary, Hormuz closed binary, war
period flag, event count per day.

## Model

LightGBM regressor trained on daily log returns. Time-series cross-validation
with 5 folds and no data leakage throughout.

| Metric | Value |
|--------|-------|
| Mean CV MAE | 1.65% daily return |
| Mean CV RMSE | 2.27% daily return |
| Fold 5 RMSE (war period) | 4.17% |

Fold 5 covers the war regime. The model was trained entirely on pre-war data
for that fold and validated on the conflict period - the higher error is
expected and reflects genuine regime shift rather than model failure.
Volatility_10d is the dominant feature, confirming that the model is
primarily a volatility regime detector rather than a directional price
predictor. That is the right tool for this problem.

## Scenario Simulation

Monte Carlo simulation with 5,000 paths over a 30-day horizon under three
scenarios:

| Scenario | Hormuz | Median 30d | P10 | P90 | Prob above $100 |
|----------|--------|------------|-----|-----|-----------------|
| Ceasefire | Open | $93.5 | $84.2 | $103.4 | 19.6% |
| Frozen Conflict | Partial | $86.8 | $78.3 | $96.4 | 4.1% |
| Full Escalation | Closed | $94.9 | $85.7 | $105.3 | 26.1% |

The scenario spread is meaningful. Frozen conflict produces a bearish drift
as prolonged disruption suppresses demand expectations. Full escalation holds
a risk premium. The fan charts in outputs/ show the full distributional
picture including downside risk.

## Project Structure

```
hormuz-shock-propagation-model/
├── src/
│   ├── data_pipeline.py     # Live price pull, events table, panel construction
│   └── model.py             # LightGBM training, Monte Carlo simulation, outputs
├── data/
│   ├── raw/
│   │   ├── brent_crude_daily.csv
│   │   └── conflict_events.csv
│   └── processed/
│       └── daily_panel.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_and_scenarios.ipynb
└── outputs/
    ├── model_diagnostics.png
    ├── scenario_forecasts.png
    └── scenario_summary.csv
```

## How to Run

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib yfinance

python src/data_pipeline.py
python src/model.py
```

## Limitations

The war regime sample is 34 trading days. That is a thin basis for
regime-specific parameter estimation. The model is calibrated primarily on
the pre-war period and extrapolates into the conflict regime using the
conflict feature structure. Treat scenario outputs as risk characterisation
rather than price forecasts.

Conflict scoring is static and hand-coded. A production version would
replace this with a GDELT real-time feed scored using FinBERT sentiment,
updating daily as new events are reported.

The framework extends naturally to LNG prices, Baltic Dirty Tanker Index
freight rates, and equities of Gulf-exposed firms.

## Author

Frackson Makwangwala  
Applied Data Scientist | PhD Candidate, LUANAR  
Lilongwe, Malawi
