"""
04 - First model comparison.

The project's first full comparison: always-operating, logistic regression,
weighted logistic regression, random forest and gradient boosting on one
75/25 split. Accuracy barely separates them; F1 and PR-AUC do. Then it trains
and tests boosting inside each forecast-horizon group on its own, lists the
inputs the forest uses most, and prints the companies boosting ranks as most
at risk.

Kept exactly as first run, so these scores are NOT comparable with scripts
05-07: duplicate records are kept and `year` (the horizon group) is one of the
inputs. 07_compare_models.py removes duplicates, leaves `year` out and keeps a
separate validation set for every choice.

Run: python 04_first_model_comparison.py
Figures: figures/04_first_model_comparison/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score,
                             average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (COLORS, GROUP, SEED, TARGET, financial_columns, load_data, readable,
                    save_figure)

FIGURES = Path(__file__).stem
LABELS = ["Operating", "Bankrupt"]
RATIO = "net_profit_to_total_assets"  # the ratio used to show outliers


def make_logistic(class_weight=None):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight=class_weight)),
    ])


def make_models():
    return {
        "Always operating": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_logistic(),
        "Weighted logistic regression": make_logistic(class_weight="balanced"),
        "Random forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                             n_jobs=-1, random_state=SEED)),
        ]),
        "Gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def evaluate(model, X, y):
    """Scores at the default 0.50 cutoff, plus two ranking scores (PR-AUC and ROC-AUC)."""
    predicted = model.predict(X)
    scores = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, predicted),
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted),
        "F1": f1_score(y, predicted, zero_division=0),
        "PR-AUC": average_precision_score(y, scores),
        "ROC-AUC": roc_auc_score(y, scores),
    }


def check_by_horizon(data):
    """Train and test boosting inside each horizon group, so no record crosses groups.

    The CSV has no company IDs, and one company can appear in several groups.
    If scores inside each group are close to the pooled scores, that overlap is
    not inflating the results much.
    """
    ratios = financial_columns(data)
    rows = {}
    for year, group in data.groupby(GROUP):
        X_train, X_test, y_train, y_test = train_test_split(
            group[ratios], group[TARGET], test_size=0.25, stratify=group[TARGET],
            random_state=SEED)
        model = HistGradientBoostingClassifier(random_state=SEED).fit(X_train, y_train)
        rows[year] = {"n": len(group), **evaluate(model, X_test, y_test)}
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_class_balance(y):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(LABELS, counts, color=[COLORS["operating"], COLORS["bankrupt"]])
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(y):.1%})" for n in counts], padding=4)
    ax.set(title="Class balance", ylabel="Records", ylim=(0, counts.max() * 1.18))
    save_figure(fig, FIGURES, "01_class_balance.png")


def plot_rate_by_year(data):
    rate = data.groupby(GROUP)[TARGET].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(rate.index.astype(str), rate, color=COLORS["bankrupt"])
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set(title="Bankruptcy rate by forecast-horizon group",
           xlabel="Horizon group (1 = five years ahead, 5 = one year ahead)",
           ylabel="Bankrupt records (%)")
    save_figure(fig, FIGURES, "02_bankruptcy_rate_by_year.png")


def plot_outliers(values):
    """A boxplot shows the extreme values; a clipped histogram shows the typical ones."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot(values)
    ax.set(title=f"{readable(RATIO)}: extreme outliers", ylabel="Value", xticks=[])
    save_figure(fig, FIGURES, "03_net_profit_boxplot.png")

    low, high = values.quantile([0.01, 0.99])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values.clip(low, high), bins=30, color=COLORS["operating"])
    ax.set(title=f"{readable(RATIO)} (clipped to the 1st-99th percentiles)",
           xlabel="Value", ylabel="Records")
    save_figure(fig, FIGURES, "04_net_profit_histogram.png")


def plot_confusion(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, labels=[0, 1], display_labels=LABELS, cmap="Blues",
        colorbar=False, values_format=",d", ax=ax)
    ax.set(title="Logistic regression", xlabel="Predicted", ylabel="Actual")
    save_figure(fig, FIGURES, "05_confusion_logistic.png")


def plot_accuracy_vs_f1(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    results["Accuracy"].plot.barh(ax=axes[0], color=COLORS["context"], xlim=(0.7, 1.0))
    axes[0].set(title="Accuracy: every model looks fine", xlabel="Accuracy")
    results["F1"].plot.barh(ax=axes[1], color=COLORS["bankrupt"])
    axes[1].set(title="F1 for bankruptcies: the models differ a lot", xlabel="F1")
    for ax in axes:
        ax.invert_yaxis()
    save_figure(fig, FIGURES, "06_accuracy_vs_f1.png")


def plot_precision_recall(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        if name != "Always operating":
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.axhline(y_test.mean(), linestyle="--", color=COLORS["context"],
               label=f"No skill ({y_test.mean():.3f})")
    ax.set(title="Precision-recall curves on the test set")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, FIGURES, "07_precision_recall.png")


def plot_importance(importance):
    top = importance.head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([readable(name, width=45) for name in top.index], top, color=COLORS["operating"])
    ax.set(title="Random forest: 15 most used inputs", xlabel="Split importance")
    save_figure(fig, FIGURES, "08_forest_importance.png")


def main():
    data = load_data()
    X = data.drop(columns=[TARGET])  # keeps `year`, as in the first run
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"Training: {len(X_train):,} records, {int(y_train.sum())} bankruptcies")
    print(f"Test: {len(X_test):,} records, {int(y_test.sum())} bankruptcies")
    print(f"Duplicate records (kept here): {int(data.duplicated().sum())}")

    plot_class_balance(y)
    plot_rate_by_year(data)
    plot_outliers(X_train[RATIO].dropna())

    models = make_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
    results = pd.DataFrame.from_dict(
        {name: evaluate(model, X_test, y_test) for name, model in models.items()}, orient="index")
    print("\nTest set (0.50 cutoff for the first four columns):")
    print(results.round(4).to_string())

    by_year = check_by_horizon(data)
    print("\nGradient boosting trained and tested inside each horizon group:")
    print(by_year.round(4).to_string())

    forest = models["Random forest"].named_steps["model"]
    importance = pd.Series(forest.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    print("\nInputs the random forest uses most:")
    print(importance.head(15).round(4).to_string())

    boosting = models["Gradient boosting"]
    riskiest = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": boosting.predict(X_test),
        "score": boosting.predict_proba(X_test)[:, 1].round(4),
    }, index=X_test.index.rename("record")).sort_values("score", ascending=False)
    print("\nCompanies gradient boosting ranks as most at risk:")
    print(riskiest.head(20).to_string())

    plot_confusion(models["Logistic regression"], X_test, y_test)
    plot_accuracy_vs_f1(results)
    plot_precision_recall(models, X_test, y_test)
    plot_importance(importance)
    print(f"\nFigures saved in figures/{FIGURES}/")

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_bankruptcies": int(y_test.sum()),
        "results": results,
        "by_year": by_year,
        "importance": importance,
    }


if __name__ == "__main__":
    main()
