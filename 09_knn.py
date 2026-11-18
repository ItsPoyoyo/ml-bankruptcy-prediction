"""
09 - KNN on the bankruptcy data: neighbors vote.

Aula 12 applied to the project's question: classify bankruptcy from the 64
financial ratios with K-nearest neighbors. The lessons from the slides are
tested one by one: the scale of the ratios changes the distances, K changes
how local the decision is, and weights and the distance metric change who
gets to vote. A grid of 2 scales x 5 values of K x 2 weights x 2 metrics =
40 configurations is compared on validation data (never on the final test),
ranked by average precision because the classes are very unbalanced. The
winner is then rechecked on the final test against the "always operating"
baseline and the weighted logistic regression from script 03.

Setup: exact duplicates removed; 60% training, 20% validation, 20% final
test, stratified; median filling and the chosen scale live inside each
pipeline, so they learn from training data only.

Run: python 09_knn.py  (about two minutes)
Figures: figures/09_knn/
Lesson: Aula 12 (KNN)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (COLORS, SEED, TARGET, count_outcomes, financial_columns, load_data,
                    save_figure, split_60_20_20)

FIGURES = Path(__file__).stem
KS = [1, 3, 5, 7, 9]
SCALES = ["Raw ratios", "Standardized"]
WEIGHTS = ["uniform", "distance"]
METRICS = ["euclidean", "manhattan"]
CUTOFFS = np.arange(0.01, 1.00, 0.01)


def build_knn(scale, k, weights, metric):
    """Median filling first, then the optional scaling, then the KNN."""
    steps = [SimpleImputer(strategy="median")]
    steps.append(StandardScaler() if scale == "Standardized" else "passthrough")
    steps.append(KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, n_jobs=2))
    return make_pipeline(*steps)


def evaluate_configs(X_train, y_train, X_valid, y_valid):
    """Fit all 40 configurations and score them on validation data."""
    rows = []
    for scale in SCALES:
        for k in KS:
            for weights in WEIGHTS:
                for metric in METRICS:
                    model = build_knn(scale, k, weights, metric).fit(X_train, y_train)
                    scores = model.predict_proba(X_valid)[:, 1]
                    rows.append({
                        "Scale": scale, "K": k, "Weights": weights, "Metric": metric,
                        "Validation AP": average_precision_score(y_valid, scores),
                    })
                    print(f"  {scale} | K={k} | weights={weights} | metric={metric}: done")
    return pd.DataFrame(rows)


def choose_cutoff(y_valid, valid_scores):
    """The cutoff with the best F2 on validation (recall counts more than precision)."""
    f2 = []
    for cutoff in CUTOFFS:
        alerts = valid_scores >= cutoff
        precision = precision_score(y_valid, alerts, zero_division=0)
        recall = recall_score(y_valid, alerts)
        f2.append((5 * precision * recall) / (4 * precision + recall)
                  if precision + recall else 0)
    return float(CUTOFFS[int(np.argmax(f2))]), np.array(f2)


def result_row(y_test, predicted, scores):
    return {
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "Average precision": average_precision_score(y_test, scores),
        **count_outcomes(y_test, predicted),
    }


def plot_heatmap(results, best):
    """Validation AP for every configuration, K rows x 8 setting columns."""
    order = [(scale, weights, metric) for scale in SCALES for weights in WEIGHTS
             for metric in METRICS]
    columns = [f"{scale[:4].lower()}|{weights[:4]}|{metric[:4]}"
               for scale, weights, metric in order]
    table = results.pivot_table(index="K", columns=["Scale", "Weights", "Metric"],
                                values="Validation AP")
    table = table[[tuple(column) for column in order]]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    image = ax.imshow(table.values, cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, label="Validation average precision")
    ax.set_xticks(range(len(order)), columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(KS)), KS)
    for row in range(len(KS)):
        for column in range(len(order)):
            value = table.values[row, column]
            best_here = abs(value - best["Validation AP"]) < 1e-12
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if value > table.values.mean() else "black",
                    fontweight="bold" if best_here else "normal")
    ax.set(title="All 40 configurations on validation data", ylabel="K")
    save_figure(fig, FIGURES, "01_validation_heatmap.png")


def plot_scale_effect(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for scale, color in zip(SCALES, [COLORS["context"], COLORS["test"]]):
        by_k = results[results["Scale"] == scale].groupby("K")["Validation AP"].max()
        ax.plot(by_k.index, by_k.values, "o-", color=color, label=scale)
        ax.set_xticks(KS)
    ax.set(title="The unit of measurement must not decide the neighbors",
           xlabel="K", ylabel="Best validation AP (any weights/metric)", ylim=(0, None))
    ax.legend()
    save_figure(fig, FIGURES, "02_scale_effect.png")


def plot_final(final):
    fig, ax = plt.subplots(figsize=(11, 5))
    shown = final[["Accuracy", "Precision", "Recall", "F1"]]
    shown.plot.bar(ax=ax, rot=0, color=[COLORS["context"], COLORS["operating"],
                                        COLORS["bankrupt"], COLORS["test"]])
    ax.set(title="Final test: accuracy hides the bankruptcies", ylabel="Score", xlabel="",
           ylim=(0, 1.05))
    ax.legend(ncol=4, loc="upper center")
    save_figure(fig, FIGURES, "03_final_test_metrics.png")


def plot_confusions(y_test, predictions, names, title):
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 4.5))
    for ax, name in zip(np.atleast_1d(axes), names):
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions[name], labels=[0, 1],
            display_labels=["Operating", "Bankrupt"], cmap="Blues", colorbar=False,
            values_format=",d", ax=ax)
        ax.set_title(name, fontsize=11)
    fig.suptitle(title, fontsize=13)
    save_figure(fig, FIGURES, "04_confusion_matrices.png")


def plot_cutoff(cutoffs, f2, cutoff):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cutoffs, f2, color=COLORS["test"])
    ax.axvline(cutoff, color="black", linestyle="--", label=f"Chosen cutoff: {cutoff:.2f}")
    ax.set(title="The vote share cutoff is tuned on validation, never on the test",
           xlabel="Alert cutoff (predicted probability of bankruptcy)", ylabel="Validation F2")
    ax.legend()
    save_figure(fig, FIGURES, "05_cutoff.png")


def main():
    data = load_data()
    clean = data.drop_duplicates()
    X = clean[financial_columns(clean)]
    y = clean[TARGET].astype(int)
    print(f"Records: {len(clean):,} (removed {len(data) - len(clean)} duplicates) "
          f"| inputs: {X.shape[1]} ratios | bankruptcy share: {y.mean():.4f}")
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_60_20_20(X, y)
    print(f"Training: {len(X_train):,} | validation: {len(X_valid):,} | "
          f"final test: {len(X_test):,} with {int(y_test.sum())} bankruptcies")

    print("\nGrid search on validation data (40 configurations):")
    results = evaluate_configs(X_train, y_train, X_valid, y_valid)
    best = results.loc[results["Validation AP"].idxmax()]
    print(results.round(3).to_string(index=False))
    print(f"\nChosen before opening the test: {best['Scale']} | K={best['K']} | "
          f"weights={best['Weights']} | metric={best['Metric']} "
          f"(validation AP {best['Validation AP']:.3f})")
    plot_heatmap(results, best)
    plot_scale_effect(results)

    knn = build_knn(best["Scale"], best["K"], best["Weights"], best["Metric"])
    knn.fit(X_train, y_train)
    valid_scores = knn.predict_proba(X_valid)[:, 1]
    cutoff, f2 = choose_cutoff(y_valid, valid_scores)
    print(f"Cutoff tuned on validation (best F2): {cutoff:.2f}")
    plot_cutoff(CUTOFFS, f2, cutoff)

    weighted_logistic = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_train, y_train)
    always_operating = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    test_scores = knn.predict_proba(X_test)[:, 1]
    predictions = {
        "Always operating": always_operating.predict(X_test),
        f"KNN (K={best['K']}) at 0.50": test_scores >= 0.50,
        f"KNN (K={best['K']}) at cutoff {cutoff:.2f}": test_scores >= cutoff,
        "Weighted logistic": weighted_logistic.predict(X_test),
    }
    scores = {
        "Always operating": np.zeros(len(y_test)),
        "Weighted logistic": weighted_logistic.predict_proba(X_test)[:, 1],
        **{name: test_scores for name in predictions if "KNN" in name},
    }
    final = pd.DataFrame.from_dict(
        {name: result_row(y_test, predictions[name], scores[name]) for name in predictions},
        orient="index")
    print(f"\nFinal test ({len(y_test):,} records, {int(y_test.sum())} bankruptcies):")
    print(final.round(3).to_string())
    plot_final(final)
    plot_confusions(y_test, predictions, ["Always operating", "Weighted logistic",
                                          f"KNN (K={best['K']}) at cutoff {cutoff:.2f}"],
                    "Same test records: the bottom row shows the bankruptcies")

    tuned = final.loc[f"KNN (K={best['K']}) at cutoff {cutoff:.2f}"]
    print(f"\nKNN caught {int(tuned['caught'])} of {int(y_test.sum())} bankruptcies "
          f"with {int(tuned['false_alarms']):,} false alarms at cutoff {cutoff:.2f}.")
    print("Every prediction costs a search through the training records: prediction time "
          "grows with the data, and the scores are vote shares, not calibrated chances.")
    print(f"Figures saved in figures/{FIGURES}/")

    return {"results": results, "best": best, "cutoff": cutoff, "final": final}


if __name__ == "__main__":
    main()
