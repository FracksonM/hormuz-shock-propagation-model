"""
model.py
Hormuz Shock Propagation Model

Trains a LightGBM model to predict daily Brent Crude log returns using
conflict event features alongside standard price lags. Then runs a Monte
Carlo simulation to produce 30-day price distributions under three war
scenarios: ceasefire, frozen conflict, and full escalation.

Run data_pipeline.py first to generate data/processed/daily_panel.csv.

Outputs saved to outputs/:
    model_diagnostics.png
    scenario_forecasts.png
    scenario_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import os

os.makedirs("outputs", exist_ok=True)


panel = pd.read_csv("data/processed/daily_panel.csv", index_col="date", parse_dates=True)
print(f"Panel loaded: {panel.shape}")


FEATURES = [
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "close_lag_1", "close_lag_2",
    "volatility_10d",
    "escalation_score", "de_escalation_score",
    "net_conflict_7d",
    "hormuz_threat", "hormuz_closed",
    "war_period",
    "n_events",
    "ma_5", "ma_20",
]
TARGET = "daily_return"

df_model = panel[FEATURES + [TARGET, "close"]].dropna()
X = df_model[FEATURES].values
y = df_model[TARGET].values

print(f"Training rows: {len(df_model)}, features: {len(FEATURES)}")


lgb_params = {
    "objective":         "regression",
    "n_estimators":      300,
    "learning_rate":     0.03,
    "max_depth":         4,
    "num_leaves":        15,
    "min_child_samples": 10,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "verbose":           -1,
    "random_state":      42,
}

tscv = TimeSeriesSplit(n_splits=5)
cv_maes, cv_rmses = [], []

print("\nTime-series cross-validation (5 folds):")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X_tr, y_tr,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(30, verbose=False),
                     lgb.log_evaluation(period=-1)])

    preds = m.predict(X_val)
    mae   = mean_absolute_error(y_val, preds)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    cv_maes.append(mae)
    cv_rmses.append(rmse)
    print(f"  Fold {fold}: MAE={mae*100:.4f}%  RMSE={rmse*100:.4f}%")

print(f"\n  Mean CV MAE : {np.mean(cv_maes)*100:.4f}%")
print(f"  Mean CV RMSE: {np.mean(cv_rmses)*100:.4f}%")


# Train on the full dataset for the final model and scenario simulation
final_model = lgb.LGBMRegressor(**lgb_params)
final_model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])
in_sample_preds = final_model.predict(X)
residuals = y - in_sample_preds

importance_df = pd.DataFrame({
    "feature":    FEATURES,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 8 features by importance:")
for _, row in importance_df.head(8).iterrows():
    print(f"  {row['feature']:<25} {row['importance']:.0f}")


N_SIMS  = 5000
HORIZON = 30

last_row     = df_model.iloc[-1]
last_close   = last_row["close"]
last_vol     = last_row["volatility_10d"]
residual_std = residuals.std()

# Each scenario fixes the conflict feature state for all 30 forecast steps.
# Noise is drawn from the in-sample residual distribution at each step.
SCENARIOS = {
    "Ceasefire": {
        "escalation_score":    0,
        "de_escalation_score": 4,
        "hormuz_threat":       0,
        "hormuz_closed":       0,
        "war_period":          1,
        "n_events":            1,
        "net_conflict_7d":    -8.0,
        "description": "Ceasefire agreed; Hormuz reopens fully",
        "color": "#2ecc71",
    },
    "Frozen Conflict": {
        "escalation_score":    1,
        "de_escalation_score": 1,
        "hormuz_threat":       1,
        "hormuz_closed":       1,
        "war_period":          1,
        "n_events":            1,
        "net_conflict_7d":     2.0,
        "description": "Stalemate; Hormuz partially open with disruption",
        "color": "#f39c12",
    },
    "Full Escalation": {
        "escalation_score":    4,
        "de_escalation_score": 0,
        "hormuz_threat":       1,
        "hormuz_closed":       1,
        "war_period":          1,
        "n_events":            2,
        "net_conflict_7d":    12.0,
        "description": "Iran strikes regional infrastructure; full Hormuz closure",
        "color": "#e74c3c",
    },
}

scenario_results = {}

for scenario_name, sp in SCENARIOS.items():
    paths = np.zeros((N_SIMS, HORIZON + 1))
    paths[:, 0] = last_close

    for t in range(1, HORIZON + 1):
        prev_close_mean = paths[:, t-1].mean()

        feat_vec = {
            "return_lag_1":        in_sample_preds[-1] if t == 1 else 0.0,
            "return_lag_2":        in_sample_preds[-2] if t <= 2 else 0.0,
            "return_lag_3":        in_sample_preds[-3] if t <= 3 else 0.0,
            "return_lag_5":        in_sample_preds[-5] if t <= 5 else 0.0,
            "close_lag_1":         prev_close_mean,
            "close_lag_2":         paths[:, max(0, t-2)].mean(),
            "volatility_10d":      last_vol * (1.2 if sp["hormuz_closed"] else 0.9),
            "escalation_score":    sp["escalation_score"],
            "de_escalation_score": sp["de_escalation_score"],
            "net_conflict_7d":     sp["net_conflict_7d"],
            "hormuz_threat":       sp["hormuz_threat"],
            "hormuz_closed":       sp["hormuz_closed"],
            "war_period":          1,
            "n_events":            sp["n_events"],
            "ma_5":                prev_close_mean,
            "ma_20":               prev_close_mean * 0.98,
        }
        x_pred           = np.array([[feat_vec[f] for f in FEATURES]])
        predicted_return = final_model.predict(x_pred)[0]
        noise            = np.random.normal(0, residual_std, N_SIMS)
        paths[:, t]      = paths[:, t-1] * np.exp(predicted_return + noise)

    scenario_results[scenario_name] = paths
    final_prices = paths[:, -1]
    print(f"  {scenario_name:<20} P10=${np.percentile(final_prices,10):.1f}  "
          f"Median=${np.median(final_prices):.1f}  "
          f"P90=${np.percentile(final_prices,90):.1f}")


# Figure 1: model diagnostics - four panels
fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
fig1.patch.set_facecolor("#0f1117")
for ax in axes.flat:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#cccccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

ax = axes[0, 0]
pre = df_model[df_model["war_period"] == 0]
war = df_model[df_model["war_period"] == 1]
ax.plot(pre.index, pre["close"], color="#5b8dee", linewidth=1.2, label="Pre-war")
ax.plot(war.index, war["close"], color="#e74c3c", linewidth=1.8, label="War period")
ax.axvline(pd.Timestamp("2026-02-28"), color="#f39c12", linestyle="--", linewidth=1, alpha=0.8)
ax.annotate("Op. Epic Fury\n28 Feb",
            xy=(pd.Timestamp("2026-02-28"), war["close"].iloc[0]),
            xytext=(pd.Timestamp("2026-01-10"), 108),
            arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.2),
            color="#f39c12", fontsize=7.5)
ax.set_title("Brent Crude - Price Series", color="white", fontsize=11, pad=8)
ax.set_ylabel("USD / barrel", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")

ax = axes[0, 1]
top8 = importance_df.head(8)
bar_colors = [
    "#e74c3c" if any(k in f for k in ["hormuz", "escalat", "conflict", "war", "n_events"])
    else "#2ecc71" if any(k in f for k in ["ma_"])
    else "#5b8dee"
    for f in top8["feature"]
]
ax.barh(top8["feature"], top8["importance"], color=bar_colors, height=0.6)
ax.invert_yaxis()
ax.set_title("Feature Importance (LightGBM)", color="white", fontsize=11, pad=8)
ax.set_xlabel("Gain", color="#aaaaaa", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
blue_p  = mpatches.Patch(color="#5b8dee", label="Price / return lags")
red_p   = mpatches.Patch(color="#e74c3c", label="Conflict features")
green_p = mpatches.Patch(color="#2ecc71", label="Technical")
ax.legend(handles=[blue_p, red_p, green_p], fontsize=7.5, framealpha=0.3, labelcolor="white")

ax = axes[1, 0]
folds = range(1, 6)
ax.bar([f - 0.2 for f in folds], [m * 100 for m in cv_maes],
       width=0.35, color="#5b8dee", label="MAE (%)", alpha=0.85)
ax.bar([f + 0.2 for f in folds], [r * 100 for r in cv_rmses],
       width=0.35, color="#e74c3c", label="RMSE (%)", alpha=0.85)
ax.set_title("Cross-validation Performance (5-fold TS)", color="white", fontsize=11, pad=8)
ax.set_xlabel("Fold", color="#aaaaaa", fontsize=9)
ax.set_ylabel("Error (%)", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")

ax = axes[1, 1]
ax.hist(residuals * 100, bins=40, color="#5b8dee",
        edgecolor="#0f1117", alpha=0.85, density=True)
mu, sigma = residuals.mean() * 100, residuals.std() * 100
x_range   = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
normal_fit = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma)**2)
ax.plot(x_range, normal_fit, color="#f39c12", linewidth=1.5, label="Normal fit")
ax.set_title("Model Residuals Distribution", color="white", fontsize=11, pad=8)
ax.set_xlabel("Residual daily return (%)", color="#aaaaaa", fontsize=9)
ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")

fig1.suptitle("Hormuz Shock Propagation Model - Diagnostics",
              color="white", fontsize=14, fontweight="bold", y=1.01)
fig1.tight_layout()
fig1.savefig("outputs/model_diagnostics.png", dpi=150,
             bbox_inches="tight", facecolor="#0f1117")
print("\nSaved to outputs/model_diagnostics.png")


# Figure 2: scenario fan charts
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig2.patch.set_facecolor("#0f1117")
forecast_days = np.arange(0, HORIZON + 1)
summary_rows  = []

for ax, (scenario_name, paths) in zip(axes2, scenario_results.items()):
    sp    = SCENARIOS[scenario_name]
    color = sp["color"]
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#cccccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    p10 = np.percentile(paths, 10, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p90 = np.percentile(paths, 90, axis=0)

    ax.fill_between(forecast_days, p10, p90, alpha=0.15, color=color, label="P10-P90")
    ax.fill_between(forecast_days, p25, p75, alpha=0.30, color=color, label="P25-P75")
    ax.plot(forecast_days, p50, color=color, linewidth=2.2, label="Median")
    ax.axhline(last_close, color="white", linestyle="--", linewidth=0.9,
               alpha=0.5, label=f"Current ${last_close:.1f}")
    ax.axhline(100, color="#f39c12", linestyle=":", linewidth=0.8,
               alpha=0.6, label="$100 mark")

    ax.set_title(scenario_name, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Trading days ahead", color="#aaaaaa", fontsize=9)
    ax.text(0.5, 0.04, sp["description"], transform=ax.transAxes,
            color="#aaaaaa", fontsize=7.5, ha="center", style="italic")
    ax.legend(fontsize=7.5, framealpha=0.25, labelcolor="white", loc="upper right")
    ax.grid(True)

    if ax == axes2[0]:
        ax.set_ylabel("Brent Crude (USD/barrel)", color="#aaaaaa", fontsize=9)

    final_prices = paths[:, -1]
    summary_rows.append({
        "scenario":       scenario_name,
        "current_price":  round(last_close, 2),
        "median_30d":     round(np.median(final_prices), 2),
        "p10_30d":        round(np.percentile(final_prices, 10), 2),
        "p90_30d":        round(np.percentile(final_prices, 90), 2),
        "prob_above_100": round((final_prices > 100).mean() * 100, 1),
        "prob_above_120": round((final_prices > 120).mean() * 100, 1),
        "prob_below_80":  round((final_prices < 80).mean() * 100, 1),
    })

fig2.suptitle(
    "Hormuz Shock Propagation Model - 30-Day Brent Crude Scenarios\n"
    "Monte Carlo Fan Charts (5,000 simulations per scenario)",
    color="white", fontsize=13, fontweight="bold", y=1.02,
)
fig2.tight_layout()
fig2.savefig("outputs/scenario_forecasts.png", dpi=150,
             bbox_inches="tight", facecolor="#0f1117")
print("Saved to outputs/scenario_forecasts.png")


summary = pd.DataFrame(summary_rows)
summary.to_csv("outputs/scenario_summary.csv", index=False)
print("\nSaved to outputs/scenario_summary.csv")
print("\nScenario summary:")
print(summary.to_string(index=False))
print("\nDone.")