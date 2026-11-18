"""
08 - Grouping companies with K-means (unsupervised).

Aula 08 applied to the bankruptcy data: the 64 financial ratios are clustered
without looking at the bankruptcy label. Two preprocessing settings (plain
standardization, and shrinking extreme values first) are compared for
k = 2 to 7, scored with inertia (elbow) and silhouette, and the chosen
solution is described afterwards: cluster sizes, bankruptcy share per group,
which ratios set the groups apart, and how stable the groups are when the
initialization changes. The label is used only to describe the groups, never
to build them.

Setup: exact duplicates removed; median filling learned from the clustered
data; `year` and `class` left out of the inputs.

Run: python 08_kmeans.py
Figures: figures/08_kmeans/
Lesson: Aula 08 (clustering)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from common import COLORS, SEED, TARGET, financial_columns, load_data, readable, save_figure

FIGURES = Path(__file__).stem
KS = range(2, 8)
SILHOUETTE_SAMPLE = 5_000
SEEDS = [SEED, 1, 7, 123, 2026]
PROFILE_RATIOS = [
    "net_profit_to_total_assets",
    "total_liabilities_to_total_assets",
    "working_capital_to_total_assets",
    "retained_earnings_to_total_assets",
    "ebit_to_total_assets",
    "equity_to_total_assets",
]


def shrink_extremes(values):
    """Signed log: shrinks huge ratios, keeps negative values negative."""
    return np.sign(values) * np.log1p(np.abs(values))


def make_scalers():
    """Two settings for the distances, as tested in the lesson."""
    return {
        "Standard scaled": StandardScaler(),
        "Shrunk + robust": [FunctionTransformer(shrink_extremes), RobustScaler()],
    }


def prepare_inputs(data, setting):
    """Median filling, then one of the two scaling settings."""
    filled = SimpleImputer(strategy="median").fit_transform(data)
    if not isinstance(setting, (list, tuple)):  # a single scaler
        return setting.fit_transform(filled)
    for step in setting:  # shrink, then scale
        filled = step.fit_transform(filled)
    return filled


def try_ks(X_scaled):
    """Fit every k with both settings and score inertia plus silhouette."""
    rows = []
    for setting_name, setting in make_scalers().items():
        X = prepare_inputs(X_scaled, setting)
        for k in KS:
            model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
            sample = np.random.default_rng(SEED).choice(len(X), SILHOUETTE_SAMPLE, replace=False)
            unique = np.unique(model.labels_[sample])
            sil = (silhouette_score(X[sample], model.labels_[sample])
                   if len(unique) >= 2 else np.nan)
            largest_share = pd.Series(model.labels_).value_counts(normalize=True).max()
            rows.append({"Setting": setting_name, "k": k, "Inertia": model.inertia_,
                         "Silhouette": sil, "Largest share": largest_share})
            print(f"  {setting_name} | k={k}: done")
    return pd.DataFrame(rows)


def stability_check(X, k):
    """Refit with different seeds; agreement between runs is 1 when identical."""
    labels_base = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X).labels_
    agreements = [adjusted_rand_score(labels_base,
                                      KMeans(n_clusters=k, n_init=10,
                                             random_state=seed).fit(X).labels_)
                  for seed in SEEDS[1:]]
    return agreements


def plot_scores(scores):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for setting_name, group in scores.groupby("Setting", sort=False):
        for ax, metric in zip(axes, ["Inertia", "Silhouette"]):
            ax.plot(group["k"], group[metric], "o-", label=setting_name)
            ax.set_xticks(list(KS))
            ax.set(title=metric, xlabel="k", ylabel=metric)
            ax.grid(alpha=0.2)
    axes[0].set(title="Elbow: where does the gain level off?")
    axes[1].set(title="Silhouette: are the groups separated?")
    axes[0].legend()
    save_figure(fig, FIGURES, "01_inertia_silhouette.png")


def plot_groups(sizes, share, overall):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bars = axes[0].bar(sizes.index.astype(str), sizes, color=COLORS["operating"])
    axes[0].bar_label(bars, fmt="%d", padding=4)
    axes[0].set(title="Group sizes", xlabel="Cluster", ylabel="Records")
    bars = axes[1].bar(sizes.index.astype(str), share, color=COLORS["bankrupt"])
    axes[1].bar_label(bars, labels=[f"{value:.1%}" for value in share], padding=4)
    axes[1].axhline(overall, color="gray", linestyle="--",
                    label=f"Overall: {overall:.1%}")
    axes[1].set(title="Bankruptcy share inside each group", xlabel="Cluster", ylabel="Share")
    axes[1].set_ylim(0, share.max() * 1.3)
    axes[1].legend()
    save_figure(fig, FIGURES, "02_group_sizes_and_bankruptcy.png")


def plot_profile(profile):
    """Mean of a few ratios per group, in original units."""
    ratios = [column for column in PROFILE_RATIOS if column in profile.columns]
    width = len(ratios)
    fig, axes = plt.subplots(1, width, figsize=(3 * width, 4.2), sharey=False)
    clusters = profile.index.astype(str)
    for ax, column in zip(axes, ratios):
        bars = ax.bar(clusters, profile[column], color=COLORS["validation"])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set(title=readable(column, width=18), xlabel="Cluster")
    fig.suptitle("What sets the groups apart (mean ratio per group)", fontsize=13)
    save_figure(fig, FIGURES, "03_group_profile.png")


def plot_stability(ari):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(ari.index, ari["Mean ARI"], xerr=ari["ARI SD"], color=COLORS["test"],
                   capsize=4)
    ax.bar_label(bars, fmt="%.3f", padding=6)
    ax.set(title="Same data, different initializations: how much do the groups agree?",
           xlabel="Adjusted Rand index between runs (1 = identical)", xlim=(0, 1.05))
    ax.invert_yaxis()
    save_figure(fig, FIGURES, "04_stability.png")


def main():
    data = load_data()
    print(f"Records: {len(data):,} | financial columns: {len(financial_columns(data))}")
    clean = data.drop_duplicates()
    print(f"Exact duplicates removed: {len(data) - len(clean)}")

    X = clean[financial_columns(clean)].copy()
    bankruptcy_share = clean[TARGET].mean()

    print("\nFitting k = 2..7 with two settings:")
    scores = try_ks(X)
    print(scores.round(3).to_string(index=False))
    plot_scores(scores)

    usable = scores[scores["Largest share"] < 0.99]
    degenerate = len(scores) - len(usable)
    if degenerate:
        print(f"\n{degenerate} solutions are degenerate: one group holds almost every record "
              "while the rest isolate a handful of extreme values. Those groups say nothing "
              "about typical companies, so they are not eligible.")
    if usable["Silhouette"].isna().all():
        raise RuntimeError("No usable clustering was found; check the preprocessing.")
    best = usable.loc[usable["Silhouette"].idxmax()]
    k_chosen = int(best["k"])
    setting_name = best["Setting"]
    print(f"\nBest usable silhouette: k={k_chosen} with '{setting_name}' "
          f"({best['Silhouette']:.3f})")
    print("The elbow and the silhouette are hints, not verdicts: the groups still need "
          "to make sense.")

    setting = make_scalers()[setting_name]
    X_scaled = prepare_inputs(X, setting)
    model = KMeans(n_clusters=k_chosen, n_init=10, random_state=SEED).fit(X_scaled)
    labels = pd.Series(model.labels_.astype(int), name="Cluster")
    print(f"\nGroup sizes:\n{labels.value_counts().sort_index().to_string()}")

    described = pd.DataFrame({"Cluster": labels, TARGET: clean[TARGET].values})
    sizes = labels.value_counts().sort_index()
    share = described.groupby("Cluster")[TARGET].mean()
    plot_groups(sizes, share, bankruptcy_share)
    print(f"\nBankruptcy share inside each group:\n{share.map('{:.1%}'.format).to_string()}")
    print(f"Overall share: {bankruptcy_share:.1%}")

    profile = pd.concat([X, labels], axis=1).groupby("Cluster")[PROFILE_RATIOS].mean()
    print(f"\nMean of a few ratios per group (original units):\n{profile.round(3).to_string()}")
    plot_profile(profile)

    print("\nStability across initializations (4 extra fits):")
    ari_runs = pd.Series(stability_check(X_scaled, k_chosen))
    for seed, value in zip(SEEDS[1:], ari_runs):
        print(f"  seed {seed}: ARI {value:.3f}")
    ari = pd.DataFrame({"Mean ARI": [np.mean(ari_runs)], "ARI SD": [np.std(ari_runs)]},
                       index=[f"k={k_chosen}, {setting_name}"])
    plot_stability(ari)

    print(f"\nFigures saved in figures/{FIGURES}/")
    return {"scores": scores, "k": k_chosen, "setting": setting_name, "sizes": sizes,
            "share": share, "profile": profile, "stability": ari}


if __name__ == "__main__":
    main()
