"""
07 - Compare the models: what each change bought.

The project's final comparison, all on the same records. Starting from the
first logistic regression, each step adds one change: class weights, then
missing-value flags and shrunken extreme values, then two tree ensembles
(random forest and gradient boosting). Models are ranked by average precision
on three training folds, the winner's alert cutoff is chosen on validation
data, and only then does every model face the same final test. It ends by
showing which inputs the winner relies on and whether its scores can be read
as probabilities.

Setup: exact duplicates removed; 60% training, 20% validation, 20% final
test, stratified; `year` left out of the inputs. All filling and scaling
happen inside each pipeline, so they learn from training data only.

Run: python 07_compare_models.py
Figures: figures/07_compare_models/
Lesson: Aulas 06-11 (classification, trees, random forest, gradient boosting)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score,
                             average_precision_score, brier_score_loss, f1_score, fbeta_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from common import (COLORS, SEED, TARGET, count_outcomes, financial_columns, load_data,
                    readable, save_figure, split_60_20_20)

FIGURES = Path(__file__).stem
CUTOFFS = np.arange(0.01, 0.96, 0.01)
SHOWN = ["Old logistic", "Chosen model at 0.50", "Chosen model at tuned cutoff"]


def shrink_extremes(values):
    """Signed log: shrinks huge ratios, keeps negative values negative."""
    return np.sign(values) * np.log1p(np.abs(values))


def logistic(class_weight=None):
    return LogisticRegression(solver="liblinear", class_weight=class_weight, max_iter=2000,
                              random_state=SEED)


def make_models():
    """Each model adds one change to the one before it."""
    return {
        "Old logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), logistic()),
        "Add class weights": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), logistic("balanced")),
        "Handle gaps and extremes": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            FunctionTransformer(shrink_extremes), RobustScaler(), logistic("balanced")),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced",
                                   random_state=SEED, n_jobs=2)),
        "Boosting": HistGradientBoostingClassifier(max_iter=150, class_weight="balanced",
                                                   random_state=SEED),
    }


def cross_validate_models(models, X_train, y_train):
    """Average precision on three training folds: how well bankruptcies are ranked first."""
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=folds,
                                 scoring="average_precision", n_jobs=1)
        rows[name] = {"Mean AP": scores.mean(), "Split SD": scores.std(),
                      "Split 1": scores[0], "Split 2": scores[1], "Split 3": scores[2]}
        print(f"  {name}: done")
    return pd.DataFrame.from_dict(rows, orient="index")


def choose_cutoff(y_valid, valid_scores):
    """The first cutoff with the best F2 on validation (recall counts more than precision)."""
    rows = []
    for cutoff in CUTOFFS:
        alerts = valid_scores >= cutoff
        rows.append({"Cutoff": cutoff,
                     "Precision": precision_score(y_valid, alerts, zero_division=0),
                     "Recall": recall_score(y_valid, alerts),
                     "F2": fbeta_score(y_valid, alerts, beta=2, zero_division=0)})
    table = pd.DataFrame(rows)
    return table.loc[table["F2"].idxmax(), "Cutoff"], table


def result_row(y_test, predicted, scores):
    return {
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "Average precision": average_precision_score(y_test, scores),
        **count_outcomes(y_test, predicted),
    }


def plot_class_balance(y):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(["Operating", "Bankrupt"], counts,
                  color=[COLORS["operating"], COLORS["bankrupt"]])
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(y):.1%})" for n in counts], padding=5)
    ax.set(title="Most records are not bankruptcies", ylabel="Records",
           ylim=(0, counts.max() * 1.18))
    save_figure(fig, FIGURES, "01_class_balance.png")


def plot_missing(data):
    missing = data[financial_columns(data)].isna().mean().mul(100)
    top = missing.nlargest(5).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh([readable(name, width=34) for name in top.index], top,
                   color=COLORS["validation"])
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set(title="Some financial values are often missing", xlabel="Missing values (%)",
           xlim=(0, 55))
    save_figure(fig, FIGURES, "02_missing_values.png")


def plot_splits(sizes):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(sizes.index, sizes["Records"],
                  color=[COLORS["operating"], COLORS["validation"], COLORS["test"]])
    ax.bar_label(bars, fmt="%.0f", padding=4)
    ax.set(title="Three separate jobs for the data", ylabel="Records",
           ylim=(0, sizes["Records"].max() * 1.2))
    save_figure(fig, FIGURES, "03_data_splits.png")


def plot_shrink(ratio):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(ratio, bins=60, color=COLORS["operating"])
    axes[0].set(title="Original ratio", xlabel="Net profit / total assets", ylabel="Records")
    axes[1].hist(shrink_extremes(ratio), bins=60, color=COLORS["test"])
    axes[1].set(title="After shrinking extreme values", xlabel="Signed log of the ratio",
                ylabel="Records")
    save_figure(fig, FIGURES, "04_shrink_extremes.png")


def plot_cross_validation(cv, best_name, no_skill):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS["test"] if name == best_name else COLORS["operating"] for name in cv.index]
    bars = ax.barh(cv.index, cv["Mean AP"], xerr=cv["Split SD"], color=colors, capsize=4)
    ax.bar_label(bars, fmt="%.3f", padding=6)
    ax.axvline(no_skill, color=COLORS["bankrupt"], linestyle="--", label="No-skill reference")
    ax.set(title="Which model ranks bankruptcies best?",
           xlabel="Average precision on three training folds (higher is better)", xlim=(0, 1))
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    save_figure(fig, FIGURES, "05_cv_average_precision.png")


def plot_cutoffs(table, cutoff):
    fig, ax = plt.subplots(figsize=(9, 4))
    for metric, color in [("Precision", COLORS["operating"]), ("Recall", COLORS["bankrupt"]),
                          ("F2", COLORS["test"])]:
        ax.plot(table["Cutoff"], table[metric], label=metric, color=color)
    ax.axvline(cutoff, color="black", linestyle="--", label=f"Chosen cutoff: {cutoff:.2f}")
    ax.set(title="The cutoff changes the balance of mistakes", xlabel="Alert cutoff",
           ylabel="Validation score", ylim=(0, 1.05))
    ax.legend(loc="best")
    save_figure(fig, FIGURES, "06_cutoff.png")


def plot_confusions(y_test, predictions):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axes, SHOWN):
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions[name], labels=[0, 1], display_labels=["Operating", "Bankrupt"],
            cmap="Blues", colorbar=False, ax=ax, values_format=",d")
        ax.set_title(name.replace("Chosen model", "New model").replace(" at ", ": "),
                     fontsize=11)
    fig.suptitle("Same test records: the bottom row shows the bankruptcies", fontsize=13)
    save_figure(fig, FIGURES, "07_confusion_matrices.png")


def plot_metrics(final):
    shown = final.loc[SHOWN, ["Accuracy", "Precision", "Recall", "F1"]].copy()
    shown.index = ["Old approach", "New model: 0.50", "New model: tuned cutoff"]
    fig, ax = plt.subplots(figsize=(11, 5))
    shown.plot.bar(ax=ax, rot=0, color=[COLORS["context"], COLORS["operating"],
                                        COLORS["bankrupt"], COLORS["test"]])
    ax.set(title="Show detection and false alarms, not accuracy alone", ylabel="Score",
           xlabel="", ylim=(0, 1.1))
    ax.legend(ncol=4, loc="upper center")
    save_figure(fig, FIGURES, "08_metrics.png")


def plot_precision_recall(y_test, old_scores, new_scores, best_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    PrecisionRecallDisplay.from_predictions(y_test, old_scores, name="Old logistic", ax=ax)
    PrecisionRecallDisplay.from_predictions(y_test, new_scores, name=best_name, ax=ax)
    ax.axhline(y_test.mean(), color="gray", linestyle="--", label="No-skill reference")
    ax.set(title="Does the new model rank bankruptcies better?", xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(loc="upper right")
    save_figure(fig, FIGURES, "09_precision_recall.png")


def plot_importance(top):
    top = top.sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh([readable(name, width=38) for name in top.index], top, color=COLORS["operating"])
    ax.axvline(0, color="gray", linewidth=1)
    ax.set(title="Which inputs matter to the chosen model?",
           xlabel="Drop in average precision after shuffling the input")
    save_figure(fig, FIGURES, "10_permutation_importance.png")


def plot_calibration(y_test, scores_by_model):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, scores in scores_by_model.items():
        observed, predicted = calibration_curve(y_test, scores, n_bins=6, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect match")
    ax.set(title="Do scores match observed bankruptcy rates?", xlabel="Average model score",
           ylabel="Observed bankruptcy rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save_figure(fig, FIGURES, "11_calibration.png")


def main():
    data = load_data()
    print(f"Records: {len(data):,} | financial columns: {len(financial_columns(data))}")
    print(f"Exact duplicates: {int(data.duplicated().sum())} | "
          f"rows with missing values: {data.isna().any(axis=1).mean():.1%}")
    plot_class_balance(data[TARGET])
    plot_missing(data)

    clean = data.drop_duplicates()
    X = clean[financial_columns(clean)]
    y = clean[TARGET].astype(int)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_60_20_20(X, y)
    sizes = pd.DataFrame({
        "Records": [len(X_train), len(X_valid), len(X_test)],
        "Bankruptcies": [int(y_train.sum()), int(y_valid.sum()), int(y_test.sum())],
    }, index=["Training", "Validation", "Final test"])
    print(f"\nExact duplicates removed: {len(data) - len(clean)}")
    print(sizes.to_string())
    plot_splits(sizes)
    plot_shrink(X_train["net_profit_to_total_assets"].dropna())

    models = make_models()
    print("\nAverage precision on three training folds:")
    cv = cross_validate_models(models, X_train, y_train)
    best_name = cv["Mean AP"].idxmax()
    print(cv.round(3).to_string())
    print(f"Chosen on training folds, before opening the final test: {best_name}")
    plot_cross_validation(cv, best_name, y_train.mean())

    best_model = models[best_name].fit(X_train, y_train)
    old_logistic = models["Old logistic"].fit(X_train, y_train)
    cutoff, cutoff_table = choose_cutoff(y_valid, best_model.predict_proba(X_valid)[:, 1])
    print(f"Cutoff chosen on validation (best F2): {cutoff:.2f}")
    plot_cutoffs(cutoff_table, cutoff)

    always_operating = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    old_scores = old_logistic.predict_proba(X_test)[:, 1]
    new_scores = best_model.predict_proba(X_test)[:, 1]
    predictions = {
        "Always operating": always_operating.predict(X_test),
        "Old logistic": old_logistic.predict(X_test),
        "Chosen model at 0.50": new_scores >= 0.50,
        "Chosen model at tuned cutoff": new_scores >= cutoff,
    }
    scores = {
        "Always operating": always_operating.predict_proba(X_test)[:, 1],
        "Old logistic": old_scores,
        "Chosen model at 0.50": new_scores,
        "Chosen model at tuned cutoff": new_scores,
    }
    final = pd.DataFrame.from_dict(
        {name: result_row(y_test, predictions[name], scores[name]) for name in predictions},
        orient="index")
    print(f"\nFinal test ({len(y_test):,} records, {int(y_test.sum())} bankruptcies):")
    print(final.round(3).to_string())
    plot_confusions(y_test, predictions)
    plot_metrics(final)
    plot_precision_recall(y_test, old_scores, new_scores, best_name)

    old, tuned = final.loc["Old logistic"], final.loc["Chosen model at tuned cutoff"]
    print(f"\nOld logistic: caught {int(old['caught'])}, missed {int(old['missed'])}, "
          f"{int(old['false_alarms'])} false alarms.")
    print(f"{best_name} at cutoff {cutoff:.2f}: caught {int(tuned['caught'])}, "
          f"missed {int(tuned['missed'])}, {int(tuned['false_alarms'])} false alarms.")

    importance = permutation_importance(best_model, X_valid, y_valid,
                                        scoring="average_precision", n_repeats=3,
                                        random_state=SEED, n_jobs=1)
    top = pd.Series(importance.importances_mean, index=X.columns).nlargest(8)
    print("\nInputs the chosen model relies on (drop in validation AP when shuffled):")
    print(top.round(4).to_string())
    plot_importance(top)

    brier_old = brier_score_loss(y_test, old_scores)
    brier_new = brier_score_loss(y_test, new_scores)
    plot_calibration(y_test, {"Old logistic": old_scores, best_name: new_scores})
    print(f"\nBrier score (lower is better): old {brier_old:.4f}, new {brier_new:.4f}")
    print("Better ranking did not give better probabilities: the scores need calibration "
          "before they are read as real bankruptcy chances.")
    print(f"Figures saved in figures/{FIGURES}/")

    return {
        "cv": cv,
        "best_name": best_name,
        "cutoff": cutoff,
        "final": final,
        "brier_old": brier_old,
        "brier_new": brier_new,
    }


if __name__ == "__main__":
    main()
