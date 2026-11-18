"""
10 - A neural network (MLP) on the bankruptcy data.

Aula 13 applied to the project's question. A dense multilayer perceptron
receives the 64 financial ratios, passes them through a hidden layer with a
non-linear activation (sigmoid, as in the lesson, or ReLU) and answers with
a probability for the bankruptcy class. Following the lesson, training is
watched at checkpoints (epochs 1, 10, 50, 100, 200, 400, 800 and 1,200); every
snapshot is scored on validation data, and the checkpoint with the best
average precision (fewest epochs on ties) is the one that faces the final
test. The lesson ranks checkpoints by validation accuracy, which works on
its balanced synthetic digits; here the validation set keeps the natural
4.8% share, so the script shows both numbers and selects by average
precision. The lesson stops at 400 epochs; the extra checkpoints at 800 and
1,200 show whether the network kept learning past that point. Five
configurations are tested: hidden size, one or two hidden layers, activation,
and the strength of the L2 regularization that keeps the weights small.

Setup: exact duplicates removed; 60% training, 20% validation, 20% final
test, stratified; median filling and standardization fitted once on the
training data only. The MLP has no class weights, so the bankrupt records
are repeated ten times in the training data (validation and test stay
untouched) to keep the network from answering "operating" all the time.
Each epoch is one fit call of the warm-started network, exactly like
watching the lesson's epoch counter.

Run: python 10_mlp.py  (about thirty minutes)
Figures: figures/10_mlp/
Lesson: Aula 13 (neural networks, MLP)
"""
from copy import deepcopy
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (COLORS, SEED, TARGET, financial_columns, load_data,
                    save_figure, split_60_20_20)

FIGURES = Path(__file__).stem
CUTOFFS = np.arange(0.01, 1.00, 0.01)
CHECKPOINTS = [1, 10, 50, 100, 200, 400, 800, 1200]
CONFIGS = {
    "Sigmoid 20 (lesson setup)": dict(hidden_layer_sizes=(20,), activation="logistic",
                                      alpha=0.01),
    "ReLU 20": dict(hidden_layer_sizes=(20,), activation="relu", alpha=0.01),
    "ReLU 64": dict(hidden_layer_sizes=(64,), activation="relu", alpha=0.01),
    "ReLU 20-10 (two layers)": dict(hidden_layer_sizes=(20, 10), activation="relu",
                                    alpha=0.01),
    "ReLU 64, weak regularization": dict(hidden_layer_sizes=(64,), activation="relu",
                                         alpha=0.0001),
}


def make_mlp(**settings):
    """The network itself; preprocessing is fitted once before the epoch loop."""
    return MLPClassifier(random_state=SEED, learning_rate_init=0.001, batch_size=2048,
                         warm_start=True, max_iter=1, **settings)


def fit_preprocess(X_train):
    """Median filling and standardization fitted on training data only."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X_train)


def balance_training(X_train, y_train, repeat=10):
    """Repeat bankrupt records ten times in the training data only.

    The MLP has no class weights, so balancing happens in the data. The
    validation and test sets keep the natural share.
    """
    bankrupt = X_train[y_train == 1]
    X_extra = pd.concat([bankrupt] * (repeat - 1), ignore_index=True)
    y_extra = pd.Series(1, index=range(len(X_extra)))
    return pd.concat([X_train, X_extra], ignore_index=True), pd.concat(
        [y_train.reset_index(drop=True), y_extra], ignore_index=True)


def train_config(name, settings, X_train, y_train, X_valid, y_valid):
    """Train to the last checkpoint and score every checkpoint on validation data."""
    model = make_mlp(**settings)
    snapshots, history = {}, []
    for epoch in range(1, CHECKPOINTS[-1] + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)  # warm_start: one more epoch each call
        if epoch in CHECKPOINTS:
            scores = model.predict_proba(X_valid)[:, 1]
            snapshots[epoch] = deepcopy(model)
            history.append({
                "Epoch": epoch,
                "Validation AP": average_precision_score(y_valid, scores),
                "Validation accuracy": accuracy_score(y_valid, scores >= 0.5),
                "Training loss": model.loss_,
            })
            print(f"    epoch {epoch}: AP {history[-1]['Validation AP']:.3f}, "
                  f"accuracy {history[-1]['Validation accuracy']:.1%}", flush=True)
    history = pd.DataFrame(history)
    best = history.sort_values(["Validation AP", "Epoch"], ascending=[False, True]).iloc[0]
    return snapshots, history, int(best["Epoch"])


def train_configs(X_train, y_train, X_valid, y_valid):
    """Train every configuration; keep snapshots, learning curves and the choice."""
    snapshots, histories, rows = {}, {}, []
    for name, settings in CONFIGS.items():
        print(f"  {name}: training...", flush=True)
        config_snapshots, history, best_epoch = train_config(
            name, settings, X_train, y_train, X_valid, y_valid)
        snapshots[name] = config_snapshots
        histories[name] = history
        rows.append({
            "Configuration": name,
            "Chosen epoch": best_epoch,
            "Validation AP": history.set_index("Epoch").loc[best_epoch, "Validation AP"],
            "Validation accuracy":
                history.set_index("Epoch").loc[best_epoch, "Validation accuracy"],
        })
        print(f"  {name}: done (checkpoint {best_epoch})")
    return snapshots, histories, pd.DataFrame(rows).set_index("Configuration")


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
    """Every metric worth reading, plus the four confusion-matrix counts."""
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test, predicted, labels=[0, 1]).ravel()
    precision = precision_score(y_test, predicted, zero_division=0)
    recall = recall_score(y_test, predicted)
    return {
        "Accuracy": accuracy_score(y_test, predicted),
        "Balanced accuracy": balanced_accuracy_score(y_test, predicted),
        "Precision": precision,
        "Recall": recall,
        "Specificity": true_negative / (true_negative + false_positive)
        if true_negative + false_positive else 0,
        "F1": f1_score(y_test, predicted, zero_division=0),
        "F2": (5 * precision * recall) / (4 * precision + recall)
        if precision + recall else 0,
        "Average precision": average_precision_score(y_test, scores),
        "ROC AUC": roc_auc_score(y_test, scores),
        "MCC": matthews_corrcoef(y_test, predicted),
        "TP": int(true_positive),
        "FP": int(false_positive),
        "TN": int(true_negative),
        "FN": int(false_negative),
    }


def plot_checkpoints(histories, best_name, best_epoch):
    """Validation scores at every checkpoint, per configuration."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
    for name, history in histories.items():
        chosen = name == best_name
        color = COLORS["bankrupt"] if chosen else COLORS["context"]
        axes[0].plot(history["Epoch"], history["Validation AP"], "o-", color=color,
                     linewidth=2 if chosen else 1, alpha=1 if chosen else 0.55,
                     label=name, zorder=5 if chosen else 2)
        axes[1].plot(history["Epoch"], history["Validation accuracy"], "o-", color=color,
                     linewidth=2 if chosen else 1, alpha=1 if chosen else 0.55,
                     zorder=5 if chosen else 2)
    best_history = histories[best_name]
    best_ap = best_history.set_index("Epoch").loc[best_epoch, "Validation AP"]
    axes[0].scatter(best_epoch, best_ap, s=200, marker="*", color=COLORS["bankrupt"],
                    zorder=6, label=f"Chosen: epoch {best_epoch}")
    for ax, metric in zip(axes, ["Validation AP", "Validation accuracy"]):
        ax.set(xlabel="Epoch", ylabel=metric, xticks=CHECKPOINTS)
        ax.grid(alpha=0.2)
    axes[0].set_title("Average precision at each checkpoint (chosen by this)")
    axes[1].set_title("Accuracy at each checkpoint (the lesson's metric)")
    axes[0].legend(fontsize=8, loc="best")
    save_figure(fig, FIGURES, "01_checkpoints.png")


def plot_configs(validation, best_name):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [COLORS["bankrupt"] if name == best_name else COLORS["operating"]
              for name in validation.index]
    labels = [f"{name}\n(epoch {epoch})" for name, epoch in
              zip(validation.index, validation["Chosen epoch"])]
    bars = ax.barh(labels, validation["Validation AP"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=6)
    ax.set(title="Which configuration ranks bankruptcies best? (chosen on validation)",
           xlabel="Validation average precision (higher is better)", xlim=(0, 1))
    ax.invert_yaxis()
    save_figure(fig, FIGURES, "02_config_comparison.png")


def plot_cutoff(cutoffs, f2, cutoff):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cutoffs, f2, color=COLORS["test"])
    ax.axvline(cutoff, color="black", linestyle="--", label=f"Chosen cutoff: {cutoff:.2f}")
    ax.set(title="The alert cutoff is tuned on validation, never on the test",
           xlabel="Alert cutoff (predicted probability of bankruptcy)", ylabel="Validation F2")
    ax.legend()
    save_figure(fig, FIGURES, "03_cutoff.png")


def plot_confusion(y_test, predicted, name):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, labels=[0, 1], display_labels=["Operating", "Bankrupt"],
        cmap="Blues", colorbar=False, values_format=",d", ax=ax)
    ax.set(title=name, xlabel="Predicted", ylabel="Actual")
    save_figure(fig, FIGURES, "04_confusion_best.png")


def plot_final(final):
    fig, ax = plt.subplots(figsize=(12, 5))
    columns = ["Accuracy", "Balanced accuracy", "Precision", "Recall", "F1"]
    final[columns].plot.bar(
        ax=ax, rot=0, color=[COLORS["context"], COLORS["validation"], COLORS["operating"],
                             COLORS["bankrupt"], COLORS["test"]])
    ax.set(title="Final test: accuracy hides the bankruptcies", ylabel="Score", xlabel="",
           ylim=(0, 1.05))
    ax.legend(ncol=5, loc="upper center")
    save_figure(fig, FIGURES, "05_final_test_metrics.png")


def plot_scores_by_class(y_test, scores):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 25)
    ax.hist(scores[y_test == 0], bins=bins, alpha=0.7, color=COLORS["operating"],
            label="Still operating", density=True)
    ax.hist(scores[y_test == 1], bins=bins, alpha=0.7, color=COLORS["bankrupt"],
            label="Went bankrupt", density=True)
    ax.set(title="How confident are the scores? (a score is not a measured chance)",
           xlabel="Predicted probability of bankruptcy", ylabel="Share of records")
    ax.legend()
    save_figure(fig, FIGURES, "06_scores_by_class.png")


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

    X_train_balanced, y_train_balanced = balance_training(X_train, y_train)
    print(f"Bankrupt records repeated 10x for training: {len(X_train_balanced):,} records, "
          f"bankruptcy share {(y_train_balanced == 1).mean():.1%} (validation and test stay "
          "natural)")
    preprocess = fit_preprocess(X_train)
    train_matrix = preprocess.transform(X_train_balanced)
    valid_matrix = preprocess.transform(X_valid)
    test_matrix = preprocess.transform(X_test)
    print("Median filling and standardization fitted once on training data only.")

    print(f"\nTraining to {CHECKPOINTS[-1]} epochs, snapshots at {CHECKPOINTS}:")
    snapshots, histories, validation = train_configs(train_matrix, y_train_balanced,
                                                     valid_matrix, y_valid)
    best_name = validation["Validation AP"].idxmax()
    best_epoch = int(validation.loc[best_name, "Chosen epoch"])
    print(validation.round(3).to_string())
    print(f"\nChosen before opening the test: {best_name}, checkpoint at epoch {best_epoch}")
    plot_checkpoints(histories, best_name, best_epoch)
    plot_configs(validation, best_name)

    best_model = snapshots[best_name][best_epoch]
    valid_scores = best_model.predict_proba(valid_matrix)[:, 1]
    cutoff, f2 = choose_cutoff(y_valid, valid_scores)
    print(f"Cutoff tuned on validation (best F2): {cutoff:.2f}")
    plot_cutoff(CUTOFFS, f2, cutoff)

    weighted_logistic = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_train, y_train)
    always_operating = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    test_scores = best_model.predict_proba(test_matrix)[:, 1]
    predictions = {
        "Always operating": always_operating.predict(X_test),
        "Weighted logistic": weighted_logistic.predict(X_test),
        f"MLP at cutoff {cutoff:.2f}": test_scores >= cutoff,
    }
    scores = {
        "Always operating": np.zeros(len(y_test)),
        "Weighted logistic": weighted_logistic.predict_proba(X_test)[:, 1],
        f"MLP at cutoff {cutoff:.2f}": test_scores,
    }
    final = pd.DataFrame.from_dict(
        {name: result_row(y_test, predictions[name], scores[name]) for name in predictions},
        orient="index")
    metrics = ["Accuracy", "Balanced accuracy", "Precision", "Recall", "Specificity",
               "F1", "F2", "Average precision", "ROC AUC", "MCC"]
    counts = ["TP", "FP", "TN", "FN"]
    print(f"\nFinal test ({len(y_test):,} records, {int(y_test.sum())} bankruptcies):")
    print(final[metrics].round(3).to_string())
    print("\nConfusion counts on the final test (TP = caught bankruptcies, FP = false alarms, "
          "TN = quiet operating companies, FN = missed bankruptcies):")
    print(final[counts].to_string())
    plot_confusion(y_test, predictions[f"MLP at cutoff {cutoff:.2f}"],
                   f"{best_name}, epoch {best_epoch}, cutoff {cutoff:.2f}")
    plot_final(final)
    plot_scores_by_class(y_test, test_scores)

    tuned = final.loc[f"MLP at cutoff {cutoff:.2f}"]
    brier = brier_score_loss(y_test, test_scores)
    last_ap = histories[best_name].set_index("Epoch").loc[CHECKPOINTS[-1], "Validation AP"]
    print(f"\nMLP caught {int(tuned['TP'])} of {int(y_test.sum())} bankruptcies "
          f"with {int(tuned['FP']):,} false alarms at cutoff {cutoff:.2f} "
          f"(missed {int(tuned['FN'])}, quiet companies correctly left alone {int(tuned['TN']):,}).")
    print(f"Validation AP of the chosen checkpoint: "
          f"{histories[best_name].set_index('Epoch').loc[best_epoch, 'Validation AP']:.3f} "
          f"(epoch {best_epoch}) vs {last_ap:.3f} at epoch {CHECKPOINTS[-1]}.")
    print(f"Brier score of the MLP scores: {brier:.4f} (lower is better; 0.25 = always guessing)")
    print("High scores come from softmax outputs, not from measured frequencies: the same "
          "calibration caveat as in script 07 applies here.")
    print(f"Figures saved in figures/{FIGURES}/")

    return {"validation": validation, "best_name": best_name, "best_epoch": best_epoch,
            "cutoff": cutoff, "final": final, "brier": brier}


if __name__ == "__main__":
    main()
