"""
06 - Random forest: can a forest catch bankruptcies without drowning in false alarms?

Compares an ordinary forest with a class-weighted forest (200 trees each, same
settings, only the class weights differ) using all 64 ratios. The forest with
the better average precision on three training folds is chosen, its alert
threshold is chosen on validation data, and only then is it scored once on
the final test, next to the always-operating baseline.

Setup: exact duplicate records removed; 60% training, 20% validation, 20%
final test, stratified. Missing values are filled with training medians and
flagged, inside the pipeline, so the forest can also learn from the gaps.

Run: python 06_random_forest.py
Figures: figures/06_random_forest/
Lesson: Aula 10 (random forest)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, fbeta_score, make_scorer,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline

from common import (COLORS, SEED, TARGET, count_outcomes, financial_columns, load_data,
                    readable, save_figure, split_60_20_20)

FIGURES = Path(__file__).stem
N_TREES = 200
N_JOBS = 2
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
CV_SCORING = {
    "ap": "average_precision",
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "accuracy": "accuracy",
}


def make_forest(class_weight):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2, max_features="sqrt",
                               class_weight=class_weight, random_state=SEED, n_jobs=N_JOBS),
    )


def cross_validate_forests(models, X_train, y_train):
    """Mean scores over three training folds. Training AP shows how optimistic fitting is."""
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = {}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=folds, scoring=CV_SCORING,
                                return_train_score=True, n_jobs=1)
        rows[name] = {
            "Training AP": scores["train_ap"].mean(),
            "CV AP": scores["test_ap"].mean(),
            "Fold SD": scores["test_ap"].std(ddof=1),
            "CV recall at 0.50": scores["test_recall"].mean(),
            "CV F1 at 0.50": scores["test_f1"].mean(),
            "CV accuracy at 0.50": scores["test_accuracy"].mean(),
        }
        print(f"  {name}: three training folds done")
    return pd.DataFrame.from_dict(rows, orient="index")


def choose_threshold(y_valid, valid_scores):
    """Best F2 on validation (catching bankruptcies counts more than precision).

    Ties go to higher precision, then to the higher threshold.
    """
    rows = []
    for threshold in THRESHOLDS:
        predicted = (valid_scores >= threshold).astype(int)
        rows.append({
            "Threshold": threshold,
            "Precision": precision_score(y_valid, predicted, zero_division=0),
            "Recall": recall_score(y_valid, predicted, zero_division=0),
            "F1": f1_score(y_valid, predicted, zero_division=0),
            "F2": fbeta_score(y_valid, predicted, beta=2, zero_division=0),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(["F2", "Precision", "Threshold"], ascending=False).iloc[0]
    return float(best["Threshold"]), table


def summarize(y_test, predicted, scores):
    return {
        "Accuracy": accuracy_score(y_test, predicted),
        "Balanced accuracy": balanced_accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "F2": fbeta_score(y_test, predicted, beta=2, zero_division=0),
        "AP": average_precision_score(y_test, scores),
        **count_outcomes(y_test, predicted),
    }


def input_importance(pipeline, columns):
    """Split importance per ratio, adding the importance of its missing-value flag."""
    imputer = pipeline.named_steps["simpleimputer"]
    forest = pipeline.named_steps["randomforestclassifier"]
    names = pd.Index(imputer.get_feature_names_out(columns))
    importance = pd.Series(forest.feature_importances_,
                           index=names.str.replace("missingindicator_", "", regex=False))
    return importance.groupby(level=0).sum().sort_values(ascending=False)


def plot_class_balance(y):
    counts = y.value_counts().reindex([0, 1])
    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.barh(["Operating", "Bankrupt"], counts.to_numpy(),
                   color=[COLORS["context"], COLORS["bankrupt"]], height=0.5)
    ax.bar_label(bars, labels=[f"{n:,}  ({n / len(y):.2%})" for n in counts], padding=8)
    ax.invert_yaxis()
    ax.set(title="Bankruptcies are fewer than 1 in 20 records", xlabel="Records",
           xlim=(0, counts.max() * 1.35))
    save_figure(fig, FIGURES, "01_class_balance.png")


def plot_cross_validation(cv, best_name, no_skill):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.barh(cv.index, cv["CV AP"], xerr=cv["Fold SD"], capsize=4, height=0.5,
                   color=[COLORS["operating"] if name == best_name else COLORS["context"]
                          for name in cv.index])
    ax.bar_label(bars, labels=[f"  {value:.3f}" for value in cv["CV AP"]], padding=24)
    ax.axvline(no_skill, color="0.2", linestyle=":", label=f"No-skill reference: {no_skill:.3f}")
    ax.set(title="Forests compared on three unseen training folds",
           xlabel="Average precision (higher is better)", xlim=(0, 1))
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, FIGURES, "02_cross_validation.png")


def plot_thresholds(table, threshold):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for metric, color, style in [("Precision", COLORS["context"], "--"),
                                 ("Recall", COLORS["bankrupt"], "-"),
                                 ("F2", "0.2", "-.")]:
        ax.plot(table["Threshold"], table[metric], color=color, linestyle=style, linewidth=2,
                label=metric)
    ax.axvline(threshold, color="0.2", linewidth=1, linestyle=":",
               label=f"Chosen threshold: {threshold:.2f}")
    ax.set(title="The alert threshold trades detection for false alarms",
           xlabel="Alert threshold", ylabel="Validation score", xlim=(0, 1), ylim=(0, 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="best", ncol=2)
    save_figure(fig, FIGURES, "03_validation_threshold.png")


def plot_accuracy_and_recall(test):
    positions = np.arange(len(test))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for position, (_, row) in zip(positions, test.iterrows()):
        ax.plot([row["Recall"], row["Accuracy"]], [position, position], color="0.9",
                linewidth=3, zorder=1)
        ax.annotate(f"{row['Recall']:.1%}", (row["Recall"], position), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=10)
        ax.annotate(f"{row['Accuracy']:.1%}", (row["Accuracy"], position), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=10)
    ax.scatter(test["Accuracy"], positions, color=COLORS["context"], s=65, label="Accuracy",
               zorder=3)
    ax.scatter(test["Recall"], positions, color=COLORS["bankrupt"], s=65,
               label="Bankruptcy recall", zorder=4)
    ax.set_yticks(positions, test.index)
    ax.set(title="Read accuracy next to bankruptcy detection", xlabel="Final test score",
           xlim=(-0.04, 1.07), ylim=(len(test) - 0.45, -0.55))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.21), ncol=2)
    save_figure(fig, FIGURES, "04_accuracy_and_recall.png")


def plot_confusion(y_test, predicted, title):
    fig, ax = plt.subplots(figsize=(6.7, 5.4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=[0, 1], display_labels=["Operating", "Bankrupt"],
        cmap="Blues", colorbar=False, values_format=",d", ax=ax, text_kw={"fontsize": 17})
    ax.set(title=title, xlabel="Predicted", ylabel="Actual")
    save_figure(fig, FIGURES, "05_confusion_matrix.png")


def plot_precision_recall(y_test, scores, chosen, label):
    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.step(recall, precision, where="post", color=COLORS["operating"], linewidth=2,
            label=f"{label} (AP {chosen['AP']:.3f})")
    ax.axhline(y_test.mean(), linestyle=":", color=COLORS["context"],
               label=f"No-skill reference: {y_test.mean():.3f}")
    ax.scatter(chosen["Recall"], chosen["Precision"], color="0.2", s=55, zorder=5,
               label="Chosen threshold")
    ax.set(title="Useful alerts versus bankruptcies caught", xlabel="Bankruptcy recall",
           ylabel="Alert precision", xlim=(0, 1), ylim=(0, 1.04))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, FIGURES, "06_precision_recall.png")


def plot_top_inputs(importance):
    top = importance.head(8).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh([readable(name, width=43) for name in top.index], top.to_numpy(), height=0.6,
                   color=[COLORS["operating"] if value == top.max() else COLORS["context"]
                          for value in top])
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in top], padding=5)
    ax.set(title="Inputs the chosen forest uses most",
           xlabel="Share of split importance, including missing-value flags",
           xlim=(0, top.max() * 1.28))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    save_figure(fig, FIGURES, "07_top_inputs.png")


def main():
    data = load_data().drop_duplicates()
    columns = financial_columns(data)
    X, y = data[columns], data[TARGET].astype(int)
    print(f"Records after removing exact duplicates: {len(data):,} | inputs: {len(columns)}")
    print(f"Always-operating accuracy: {(y == 0).mean():.2%}")
    plot_class_balance(y)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_60_20_20(X, y)
    print(f"Training / validation / test: {len(X_train):,} / {len(X_valid):,} / {len(X_test):,}")

    models = {"Ordinary forest": make_forest(None), "Weighted forest": make_forest("balanced")}
    print("\nCross-validation on the training part:")
    cv = cross_validate_forests(models, X_train, y_train)
    best_name = cv["CV AP"].idxmax()
    print(cv.round(3).to_string())
    print(f"Chosen on training folds: {best_name}")
    plot_cross_validation(cv, best_name, y_train.mean())

    fitted = {name: clone(model).fit(X_train, y_train) for name, model in models.items()}
    chosen_model = fitted[best_name]
    threshold, threshold_table = choose_threshold(y_valid, chosen_model.predict_proba(X_valid)[:, 1])
    print(f"\nThreshold chosen on validation (best F2): {threshold:.2f}")
    print(threshold_table[threshold_table["Threshold"].isin([0.50, threshold])]
          .set_index("Threshold").round(3).to_string())
    plot_thresholds(threshold_table, threshold)

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    scores = {name: model.predict_proba(X_test)[:, 1] for name, model in fitted.items()}
    tuned_label = f"{best_name} (tuned)"
    test = pd.DataFrame.from_dict({
        "Always operating": summarize(y_test, baseline.predict(X_test),
                                      baseline.predict_proba(X_test)[:, 1]),
        "Ordinary forest (0.50)": summarize(y_test, (scores["Ordinary forest"] >= 0.50).astype(int),
                                            scores["Ordinary forest"]),
        "Weighted forest (0.50)": summarize(y_test, (scores["Weighted forest"] >= 0.50).astype(int),
                                            scores["Weighted forest"]),
        tuned_label: summarize(y_test, (scores[best_name] >= threshold).astype(int),
                               scores[best_name]),
    }, orient="index")
    print(f"\nFinal test ({len(y_test):,} records, {int(y_test.sum())} bankruptcies):")
    print(test.round(3).to_string())
    print("AP does not change with the threshold: it measures the ranking, not the alerts.")

    tuned = test.loc[tuned_label]
    plot_accuracy_and_recall(test)
    plot_confusion(y_test, (scores[best_name] >= threshold).astype(int),
                   f"Final test: {best_name.lower()}, threshold {threshold:.2f}")
    plot_precision_recall(y_test, scores[best_name], tuned, best_name)
    plot_top_inputs(input_importance(chosen_model, columns))
    print(f"\nCaught {int(tuned['caught'])} of {int(y_test.sum())} bankruptcies, "
          f"missed {int(tuned['missed'])}, raised {int(tuned['false_alarms'])} false alarms.")
    print(f"Figures saved in figures/{FIGURES}/")

    return {"cv": cv, "best_name": best_name, "threshold": threshold, "test": test}


if __name__ == "__main__":
    main()
