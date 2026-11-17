"""
Shared helpers for the bankruptcy scripts.

Every numbered script imports this file. It finds the dataset and the figures
folder relative to this file, so the scripts run from any folder, and it sets
one plotting style for all figures. Figures are saved, never shown on screen.
"""
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")  # write figures to files; never open a window
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "polish_bankruptcy.csv"
FIGURES_DIR = ROOT / "figures"
SEED = 42
TARGET = "class"  # 1 = went bankrupt, 0 = still operating
GROUP = "year"  # forecast-horizon group 1 to 5 (5 to 1 years ahead), not a calendar year
COLORS = {
    "operating": "#4278A6",
    "bankrupt": "#C54D45",
    "validation": "#CF8A3E",
    "test": "#40886A",
    "context": "#87959D",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "font.size": 11,
})
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def load_data():
    """Read the dataset: 43,405 records with 64 ratios plus `year` and `class`."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Keep data/polish_bankruptcy.csv next to the scripts."
        )
    return pd.read_csv(DATA_PATH)


def financial_columns(data):
    """The 64 financial ratio columns: every column except `year` and `class`."""
    return [column for column in data.columns if column not in (GROUP, TARGET)]


def split_60_20_20(X, y, seed=SEED):
    """Stratified split into 60% training, 20% validation and 20% final test.

    Returns X_train, X_valid, X_test, y_train, y_valid, y_test.
    """
    X_work, X_test, y_work, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_work, y_work, test_size=0.25, stratify=y_work, random_state=seed)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def count_outcomes(y_true, y_pred):
    """Count caught bankruptcies, missed bankruptcies and false alarms."""
    _, false_alarms, missed, caught = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"caught": int(caught), "missed": int(missed), "false_alarms": int(false_alarms)}


def readable(name, width=40):
    """Short label for a column, e.g. 'net_profit_to_total_assets' -> 'net profit / total assets'."""
    label = name.replace("_to_", " / ").replace("_plus_", " + ")
    label = label.replace("_minus_", " minus ").replace("_", " ")
    return textwrap.fill(label, width=width)


def save_figure(fig, folder, name):
    """Save a figure as figures/<folder>/<name>, close it and return its path."""
    path = FIGURES_DIR / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
