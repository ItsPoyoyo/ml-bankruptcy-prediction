"""
05 - Decision trees on two ratios.

Follows the course's decision-tree lesson with our data. Two inputs (net
profit / total assets and total liabilities / total assets) keep the tree's
rules drawable. Steps: a one-question tree, what changing one training label
does, how a regression leaf predicts an average, then picking the depth
(1 to 5) on validation data before one final test.

The result is the lesson's warning: chosen by accuracy, the winner is the
depth-1 tree. It predicts "operating" for every company, scores 95.2% on the
test and catches none of the 417 bankruptcies.

Run: python 05_decision_tree.py
Figures: figures/05_decision_tree/
Lesson: Aula 09 (decision trees)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from common import COLORS, SEED, TARGET, count_outcomes, load_data, save_figure, split_60_20_20

FIGURES = Path(__file__).stem
FEATURES = ["net_profit_to_total_assets", "total_liabilities_to_total_assets"]
FEATURE_LABELS = ["Net profit / assets", "Liabilities / assets"]
CLASS_LABELS = ["Operating", "Bankrupt"]
# Two made-up companies, not test records: a healthy one and a struggling one.
PROFILES = np.array([[0.05, 0.40], [-0.10, 0.90]])
DEPTHS = [1, 2, 3, 4, 5]


def fit_tree(X, y, depth):
    return DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(X, y)


def root_rule(tree):
    """Index of the input the first question uses, and its cutoff."""
    return int(tree.tree_.feature[0]), float(tree.tree_.threshold[0])


def compare_depths(X_train, y_train, X_valid, y_valid):
    """Fit depths 1-5; choose the most correct validation answers, ties to the smaller depth."""
    rows, trees = [], {}
    for depth in DEPTHS:
        tree = fit_tree(X_train, y_train, depth)
        trees[depth] = tree
        valid_predicted = tree.predict(X_valid)
        rows.append({
            "depth": depth,
            "train_correct": int((tree.predict(X_train) == y_train).sum()),
            "valid_correct": int((valid_predicted == y_valid).sum()),
            "valid_recall": recall_score(y_valid, valid_predicted),
            "valid_precision": precision_score(y_valid, valid_predicted, zero_division=0),
        })
    table = pd.DataFrame(rows)
    best = table.sort_values(["valid_correct", "depth"], ascending=[False, True]).iloc[0]
    return table, trees, int(best["depth"])


def regression_leaf_example(clean, train_index):
    """The first four complete training records: log total assets -> net profit / assets."""
    example = clean.loc[train_index, ["log_total_assets", "net_profit_to_total_assets"]]
    example = example.dropna().head(4)
    regressor = DecisionTreeRegressor(max_depth=1, random_state=SEED)
    regressor.fit(example[["log_total_assets"]].to_numpy(), example["net_profit_to_total_assets"])
    return example, regressor


def draw_points(ax, X, labels, limits):
    for value, name in enumerate(CLASS_LABELS):
        mask = labels == value
        ax.scatter(X[mask, 0], X[mask, 1], s=8, alpha=0.25, color=COLORS[name.lower()],
                   label=name)
    ax.set(xlabel=FEATURE_LABELS[0], ylabel=FEATURE_LABELS[1], xlim=limits[0], ylim=limits[1])


def draw_split(ax, tree, X, labels, limits):
    """Training records as points, the tree's first question as a dashed line.

    Axes are zoomed to the 1st-99th percentiles; every record is still used for fitting.
    """
    draw_points(ax, X, labels, limits)
    feature, threshold = root_rule(tree)
    line = ax.axvline if feature == 0 else ax.axhline
    line(threshold, color="black", linestyle="--", label="Root split")
    ax.legend(loc="upper left", fontsize=9)


def plot_classes(y):
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASS_LABELS, counts, color=[COLORS["operating"], COLORS["bankrupt"]],
                  width=0.55)
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(y):.1%})" for n in counts], padding=5)
    ax.set(title="Our classes are very uneven", ylabel="Records", ylim=(0, counts.max() * 1.18))
    save_figure(fig, FIGURES, "01_classes.png")


def plot_one_question(stump, X_train, y_train, limits):
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_tree(stump, feature_names=FEATURE_LABELS, class_names=CLASS_LABELS, filled=True,
              rounded=True, fontsize=11, precision=4, ax=ax)
    ax.set_title("One question, two possible paths", pad=16)
    save_figure(fig, FIGURES, "02_one_question_tree.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    draw_split(ax, stump, X_train, y_train, limits)
    ax.set_title("The root split on the training data")
    save_figure(fig, FIGURES, "03_one_question_split.png")


def plot_changed_label(tree, X_train, labels, limits):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_tree(tree, feature_names=FEATURE_LABELS, class_names=CLASS_LABELS, filled=True,
              rounded=True, fontsize=9, precision=4, ax=axes[0])
    axes[0].set_title("Tree after changing one label")
    draw_split(axes[1], tree, X_train, labels, limits)
    axes[1].set_title("Split after changing one label")
    save_figure(fig, FIGURES, "04_changed_label.png")


def plot_regression_leaf(example, regressor):
    x = example[["log_total_assets"]].to_numpy()
    grid = np.linspace(x.min() - 0.1, x.max() + 0.1, 300).reshape(-1, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_tree(regressor, feature_names=["Log total assets"], filled=True, rounded=True,
              fontsize=10, precision=4, ax=axes[0])
    axes[0].set_title("A regression leaf predicts an average")
    axes[1].scatter(x[:, 0], example["net_profit_to_total_assets"], s=80,
                    color=COLORS["operating"], label="Real records", zorder=3)
    axes[1].plot(grid[:, 0], regressor.predict(grid), color=COLORS["bankrupt"], linewidth=2,
                 label="Leaf prediction")
    axes[1].set(xlabel="Log total assets", ylabel="Net profit / assets",
                title="One predicted value in each region")
    axes[1].legend()
    save_figure(fig, FIGURES, "05_regression_leaf.png")


def plot_depths(depths, best_depth, n_train, n_valid):
    train_pct = 100 * depths["train_correct"] / n_train
    valid_pct = 100 * depths["valid_correct"] / n_valid
    best = depths.set_index("depth").loc[best_depth]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(depths["depth"], train_pct, "o-", label="Training", color=COLORS["operating"])
    axes[0].plot(depths["depth"], valid_pct, "o-", label="Validation",
                 color=COLORS["validation"])
    axes[0].scatter(best_depth, 100 * best["valid_correct"] / n_valid, s=220, marker="*",
                    color=COLORS["bankrupt"], label=f"Chosen: depth {best_depth}", zorder=5)
    axes[0].set(title="The lesson's rule: most correct answers", xlabel="Maximum depth",
                ylabel="Accuracy (%)", xticks=DEPTHS)
    axes[0].legend()
    axes[1].plot(depths["depth"], 100 * depths["valid_recall"], "o-", color=COLORS["bankrupt"])
    axes[1].set(title="Also check the bankruptcies caught", xlabel="Maximum depth",
                ylabel="Validation recall (%)", xticks=DEPTHS, ylim=(0, 100))
    save_figure(fig, FIGURES, "06_depth_comparison.png")


def plot_test_results(tree, best_depth, X_test, y_test, predicted, limits):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=[0, 1], display_labels=CLASS_LABELS, cmap="Blues",
        colorbar=False, values_format=",d", ax=ax)
    ax.set(title=f"Chosen tree: depth {best_depth}", xlabel="Predicted", ylabel="Actual")
    save_figure(fig, FIGURES, "07_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(max(11, 3.8 * best_depth), max(5, 3 + 1.8 * best_depth)))
    plot_tree(tree, feature_names=FEATURE_LABELS, class_names=CLASS_LABELS, filled=True,
              rounded=True, fontsize=11 if best_depth <= 2 else 8, precision=3, ax=ax)
    ax.set_title(f"The chosen tree, depth {best_depth}", fontsize=15, pad=18)
    save_figure(fig, FIGURES, "08_chosen_tree.png")

    xx, yy = np.meshgrid(np.linspace(*limits[0], 240), np.linspace(*limits[1], 240))
    region = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(xx, yy, region, levels=[-0.5, 0.5, 1.5],
                colors=[COLORS["operating"], COLORS["bankrupt"]], alpha=0.15)
    draw_points(ax, X_test, y_test, limits)
    ax.set_title("The chosen tree's regions on the final test")
    ax.legend(loc="upper left")
    save_figure(fig, FIGURES, "09_decision_regions.png")


def plot_baseline(y_test, predicted, best_depth):
    baseline = np.zeros_like(y_test)
    summary = pd.DataFrame.from_dict({
        name: {
            "Accuracy": accuracy_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
        }
        for name, pred in [("Always operating", baseline), (f"Tree: depth {best_depth}", predicted)]
    }, orient="index")
    print("\nFinal test against the always-operating baseline:")
    print(summary.round(3).to_string())
    fig, ax = plt.subplots(figsize=(9, 5))
    summary.plot.bar(ax=ax, rot=0,
                     color=[COLORS["context"], COLORS["bankrupt"], COLORS["operating"]])
    ax.set(title="Compare with the simplest possible baseline", ylabel="Score", xlabel="",
           ylim=(0, 1.1))
    ax.legend(loc="upper center", ncol=3)
    save_figure(fig, FIGURES, "10_baseline_comparison.png")


def main():
    data = load_data()
    clean = data.drop_duplicates()
    print(f"Records: {len(data):,} | exact duplicates removed: {len(data) - len(clean)} "
          f"| used: {len(clean):,}")
    X = clean[FEATURES]
    y = clean[TARGET].astype(int).to_numpy()
    plot_classes(y)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_60_20_20(X, y)
    train_index = X_train.index
    split_sizes = (len(y_train), len(y_valid), len(y_test))
    split_bankruptcies = (int(y_train.sum()), int(y_valid.sum()), int(y_test.sum()))
    print(f"Training / validation / test: {split_sizes} records, "
          f"{split_bankruptcies} bankruptcies")

    imputer = SimpleImputer(strategy="median")  # medians learned from training only
    X_train = imputer.fit_transform(X_train)
    X_valid = imputer.transform(X_valid)
    X_test = imputer.transform(X_test)
    medians = tuple(float(value) for value in imputer.statistics_)
    print("Training medians:", ", ".join(
        f"{label} = {value:.4f}" for label, value in zip(FEATURE_LABELS, medians)))
    limits = (np.quantile(X_train[:, 0], [0.01, 0.99]), np.quantile(X_train[:, 1], [0.01, 0.99]))

    stump = fit_tree(X_train, y_train, depth=1)
    feature, threshold = root_rule(stump)
    print(f"\nOne-question tree: is {FEATURES[feature]} <= {threshold:.5f}?")
    for profile, answer in zip(PROFILES, stump.predict(PROFILES)):
        print(f"  made-up company {profile.tolist()} -> {CLASS_LABELS[answer]}")
    plot_one_question(stump, X_train, y_train, limits)

    changed = y_train.copy()
    changed[np.flatnonzero(y_train == 0)[0]] = 1  # one operating record relabelled bankrupt
    changed_stump = fit_tree(X_train, changed, depth=1)
    new_feature, new_threshold = root_rule(changed_stump)
    same_rule = bool(feature == new_feature and np.isclose(threshold, new_threshold))
    print(f"After changing one training label the root is "
          f"{FEATURES[new_feature]} <= {new_threshold:.5f} "
          f"({'unchanged' if same_rule else 'changed'})")
    plot_changed_label(changed_stump, X_train, changed, limits)

    example, regressor = regression_leaf_example(clean, train_index)
    leaf_predictions = regressor.predict(example[["log_total_assets"]].to_numpy())
    print("\nA regression leaf predicts the average of its records:")
    print(example.assign(leaf_prediction=leaf_predictions).to_string())
    plot_regression_leaf(example, regressor)

    depths, trees, best_depth = compare_depths(X_train, y_train, X_valid, y_valid)
    print("\nDepths 1-5 on validation:")
    print(depths.round(3).to_string(index=False))
    print(f"Chosen: depth {best_depth} "
          f"({depths.set_index('depth').loc[best_depth, 'valid_correct']}/{len(y_valid)} correct)")
    plot_depths(depths, best_depth, len(y_train), len(y_valid))

    best = trees[best_depth]
    predicted = best.predict(X_test)
    test_correct = int((predicted == y_test).sum())
    outcomes = count_outcomes(y_test, predicted)
    print(f"\nFinal test: {test_correct}/{len(y_test)} correct "
          f"({accuracy_score(y_test, predicted):.1%}); caught {outcomes['caught']} "
          f"of {int(y_test.sum())} bankruptcies, {outcomes['false_alarms']} false alarms")
    plot_test_results(best, best_depth, X_test, y_test, predicted, limits)
    plot_baseline(y_test, predicted, best_depth)
    print(f"\nFigures saved in figures/{FIGURES}/")

    return {
        "rows_used": len(clean),
        "split_sizes": split_sizes,
        "split_bankruptcies": split_bankruptcies,
        "medians": medians,
        "root_feature": FEATURES[feature],
        "root_threshold": threshold,
        "label_change_same_rule": same_rule,
        "leaf_predictions": leaf_predictions,
        "depths": depths,
        "best_depth": best_depth,
        "test_correct": test_correct,
        "test_outcomes": outcomes,
    }


if __name__ == "__main__":
    main()
