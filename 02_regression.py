"""
02 - Regression: predict a company's return on assets.

A warm-up before predicting bankruptcy. The target here is a number:
net profit / total assets (return on assets, ROA). Three ratios are the inputs.
We compare an always-the-average baseline, a straight line (linear regression)
and a curve (degree-2 polynomial), then check the errors the scores hide.

Run: python 02_regression.py
Figures: figures/02_regression/
Lesson: Aula 05 (linear and non-linear regression)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import COLORS, SEED, load_data, readable, save_figure

FIGURES = Path(__file__).stem
ROA = "net_profit_to_total_assets"  # the regression target
FEATURES = ["sales_to_total_assets", "equity_to_total_assets", "log_total_assets"]
# Fixed plausible ranges, decided before splitting: nothing is learned from the test set.
PLAUSIBLE = {
    "net_profit_to_total_assets": (-1, 1),
    "sales_to_total_assets": (0, 10),
    "equity_to_total_assets": (-1, 1),
}
BEST = "Polynomial (degree 2)"


def prepare(data):
    """Keep complete rows whose ratios fall inside the plausible ranges."""
    rows = data[[ROA] + FEATURES].dropna()
    for column, (low, high) in PLAUSIBLE.items():
        rows = rows[rows[column].between(low, high)]
    return rows


def make_models():
    return {
        "Baseline": DummyRegressor(strategy="mean"),
        "Linear": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        BEST: Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression()),
        ]),
    }


def error_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_target_and_features(rows):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    axes[0].hist(rows[ROA], bins=60, color=COLORS["operating"])
    axes[0].set(title="Target: return on assets", xlabel=readable(ROA), ylabel="Companies")
    for ax, feature in zip(axes[1:], FEATURES):
        ax.scatter(rows[feature], rows[ROA], s=3, alpha=0.15, color=COLORS["operating"])
        ax.set(title=f"{readable(feature, width=24)} vs ROA", xlabel=readable(feature),
               ylabel=readable(ROA))
    save_figure(fig, FIGURES, "01_target_and_features.png")


def plot_test_errors(test_metrics):
    fig, ax = plt.subplots(figsize=(8, 4))
    test_metrics[["MAE", "RMSE"]].plot.bar(ax=ax, rot=0,
                                          color=[COLORS["operating"], COLORS["validation"]])
    ax.set(title="Test error by model (lower is better)",
           ylabel="Error (share of total assets)", xlabel="")
    save_figure(fig, FIGURES, "02_test_error_by_model.png")


def plot_actual_vs_predicted(y_test, predicted):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, predicted, s=5, alpha=0.15, color=COLORS["operating"])
    limits = [y_test.min(), y_test.max()]
    ax.plot(limits, limits, "--", color=COLORS["bankrupt"], label="Perfect prediction")
    ax.set(title=f"{BEST}: actual vs predicted ROA", xlabel="Actual", ylabel="Predicted")
    ax.legend()
    save_figure(fig, FIGURES, "03_actual_vs_predicted.png")


def plot_residuals(predicted, residuals):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(predicted, residuals, s=5, alpha=0.15, color=COLORS["operating"])
    axes[0].axhline(0, color=COLORS["bankrupt"], linestyle="--")
    axes[0].set(title="Residuals vs predicted", xlabel="Predicted",
                ylabel="Residual (actual - predicted)")
    axes[1].hist(residuals, bins=60, color=COLORS["test"])
    axes[1].axvline(0, color=COLORS["bankrupt"], linestyle="--")
    axes[1].set(title="Distribution of residuals", xlabel="Residual", ylabel="Companies")
    save_figure(fig, FIGURES, "04_residuals.png")


def main():
    rows = prepare(load_data())
    print(f"Rows used: {len(rows):,} of 43,405 (complete and inside the plausible ranges)")
    plot_target_and_features(rows)

    X_train, X_test, y_train, y_test = train_test_split(
        rows[FEATURES], rows[ROA], test_size=0.20, random_state=SEED)
    print(f"Training: {len(X_train):,} | test: {len(X_test):,}")

    models = make_models()
    for model in models.values():
        model.fit(X_train, y_train)

    test_metrics = pd.DataFrame(
        {name: error_metrics(y_test, model.predict(X_test)) for name, model in models.items()}
    ).T
    print("\nTest set:")
    print(test_metrics.round(4).to_string())

    linear = models["Linear"].named_steps["model"]
    coefficients = pd.Series(linear.coef_, index=FEATURES)
    print("\nLinear model on standardized inputs (change in ROA per one standard deviation):")
    print(coefficients.round(4).to_string())
    print(f"Intercept: {linear.intercept_:.4f}")

    train_vs_test = pd.DataFrame({
        name: {
            "MAE train": mean_absolute_error(y_train, model.predict(X_train)),
            "MAE test": mean_absolute_error(y_test, model.predict(X_test)),
            "RMSE train": root_mean_squared_error(y_train, model.predict(X_train)),
            "RMSE test": root_mean_squared_error(y_test, model.predict(X_test)),
        }
        for name, model in models.items()
    }).T
    print("\nTraining vs test error (similar values: no overfitting; both high: underfitting):")
    print(train_vs_test.round(4).to_string())

    predicted = models[BEST].predict(X_test)
    plot_test_errors(test_metrics)
    plot_actual_vs_predicted(y_test, predicted)
    plot_residuals(predicted, y_test - predicted)
    print(f"\nFigures saved in figures/{FIGURES}/")

    return {
        "rows_used": len(rows),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_metrics": test_metrics,
        "coefficients": coefficients,
        "intercept": float(linear.intercept_),
        "train_vs_test": train_vs_test,
    }


if __name__ == "__main__":
    main()
