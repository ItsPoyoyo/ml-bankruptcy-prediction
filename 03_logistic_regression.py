"""
03 - Logistic regression and the 95% accuracy trap.

Predicts bankruptcy (class 1) from the 64 financial ratios. About 95 in 100
records are operating companies, so a model that nearly always answers
"operating" scores about 95% accuracy while catching almost no bankruptcies.
Giving the rare class more weight changes that, at the cost of many false
alarms. The confusion matrices show what accuracy hides.

Setup, as in the original activity: all 43,405 records (duplicates kept),
`year` left out of the inputs, one stratified 80/20 train/test split.

Run: python 03_logistic_regression.py
Figures: figures/03_logistic_regression/
Lesson: Aula 06 (classification)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (COLORS, SEED, TARGET, count_outcomes, financial_columns, load_data,
                    save_figure)

FIGURES = Path(__file__).stem
LABELS = ["Operating", "Bankrupt"]


def make_logistic(class_weight=None):
    """Median filling and scaling learn from the training data only, inside the pipeline."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight=class_weight)),
    ])


def evaluate(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        **count_outcomes(y_true, y_pred),
    }


def plot_class_balance(y):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(LABELS, counts, color=[COLORS["operating"], COLORS["bankrupt"]])
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(y):.1%})" for n in counts], padding=4)
    ax.set(title="Only about 5 in 100 records are bankruptcies", ylabel="Records",
           ylim=(0, counts.max() * 1.18))
    save_figure(fig, FIGURES, "01_class_balance.png")


def plot_confusion(y_true, y_pred, title, name):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1], display_labels=LABELS, cmap="Blues",
        colorbar=False, values_format=",d", ax=ax)
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    save_figure(fig, FIGURES, name)


def plot_metrics(results):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    results[["Accuracy", "Recall", "F1"]].plot.bar(
        ax=ax, rot=0, color=[COLORS["context"], COLORS["bankrupt"], COLORS["test"]])
    ax.set(title="High accuracy does not mean a useful model", ylabel="Score", xlabel="",
           ylim=(0, 1.05))
    ax.legend(ncol=3, loc="upper center")
    save_figure(fig, FIGURES, "04_metrics_by_model.png")


def main():
    data = load_data()
    X, y = data[financial_columns(data)], data[TARGET]
    print(f"Records: {len(X):,} | inputs: {X.shape[1]} ratios | bankruptcy share: {y.mean():.4f}")
    plot_class_balance(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    print(f"Training: {len(X_train):,} | test: {len(X_test):,} "
          f"with {int(y_test.sum())} bankruptcies")

    models = {
        "Always operating": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": make_logistic(),
        "Weighted logistic regression": make_logistic(class_weight="balanced"),
    }
    predictions = {name: model.fit(X_train, y_train).predict(X_test)
                   for name, model in models.items()}
    results = pd.DataFrame.from_dict(
        {name: evaluate(y_test, pred) for name, pred in predictions.items()}, orient="index")
    print("\nTest set:")
    print(results.round(3).to_string())

    plot_confusion(y_test, predictions["Logistic regression"],
                   "Logistic regression", "02_confusion_logistic.png")
    plot_confusion(y_test, predictions["Weighted logistic regression"],
                   "Weighted logistic regression", "03_confusion_weighted.png")
    plot_metrics(results)

    plain = results.loc["Logistic regression"]
    weighted = results.loc["Weighted logistic regression"]
    print(f"\nLogistic regression caught {int(plain['caught'])} of {int(y_test.sum())} "
          f"bankruptcies with {plain['Accuracy']:.1%} accuracy.")
    print(f"With class weights it caught {int(weighted['caught'])}, "
          f"but raised {int(weighted['false_alarms']):,} false alarms.")
    print(f"Figures saved in figures/{FIGURES}/")

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_bankruptcies": int(y_test.sum()),
        "results": results,
    }


if __name__ == "__main__":
    main()
