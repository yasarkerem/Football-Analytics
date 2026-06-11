"""
Visualization utilities for Tactical Fingerprints analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def plot_cluster_scatter(
    labelled: pd.DataFrame,
    teams_df: pd.DataFrame | None = None,
    title: str = "Tactical Fingerprints — 2D Embedding",
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    n_clusters = labelled["fingerprint"].nunique()
    palette = cm.get_cmap("tab10", n_clusters)

    for fp_id, grp in labelled.groupby("fingerprint"):
        ax.scatter(
            grp["embed_x"], grp["embed_y"],
            color=palette(fp_id),
            label=f"Fingerprint {fp_id}",
            alpha=0.75, s=80, edgecolors="white", linewidths=0.5,
        )
        if teams_df is not None:
            merged = grp.merge(teams_df[["teamId", "name"]], on="teamId", how="left")
            for _, row in merged.iterrows():
                ax.annotate(row.get("name", ""), (row["embed_x"], row["embed_y"]),
                            fontsize=6, alpha=0.7)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best", fontsize=9)
    sns.despine()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_fingerprint_radar(
    labelled: pd.DataFrame,
    feature_cols: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """Radar chart of mean feature values per fingerprint cluster."""
    cluster_means = labelled.groupby("fingerprint")[feature_cols].mean()
    # Normalize 0-1 for radar display
    normalized = (cluster_means - cluster_means.min()) / (
        cluster_means.max() - cluster_means.min() + 1e-9
    )

    n_vars = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(
        1, len(cluster_means),
        figsize=(4 * len(cluster_means), 4),
        subplot_kw=dict(polar=True),
    )
    if len(cluster_means) == 1:
        axes = [axes]

    palette = cm.get_cmap("tab10", len(cluster_means))

    for ax, (fp_id, row) in zip(axes, normalized.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, color=palette(fp_id), linewidth=2)
        ax.fill(angles, values, color=palette(fp_id), alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_cols, size=7)
        ax.set_title(f"FP {fp_id}", size=11, fontweight="bold", pad=12)
        ax.set_ylim(0, 1)

    plt.suptitle("Tactical Fingerprint Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_outcome_by_fingerprint(
    labelled: pd.DataFrame,
    metric: str = "win_rate",
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = sorted(labelled["fingerprint"].unique())
    data  = [labelled[labelled["fingerprint"] == fp][metric].dropna().values for fp in order]

    ax.boxplot(data, labels=[f"FP {fp}" for fp in order], patch_artist=True,
               boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax.set_title(f"{metric} by Tactical Fingerprint", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fingerprint")
    ax.set_ylabel(metric)
    sns.despine()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pca_variance(explained_variance: np.ndarray, save_path: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(explained_variance) + 1), explained_variance * 100,
            marker="o", color="#2196F3")
    ax.axhline(90, linestyle="--", color="gray", alpha=0.6, label="90% threshold")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA Explained Variance", fontsize=12, fontweight="bold")
    ax.legend()
    sns.despine()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_silhouette_curve(eval_df: pd.DataFrame, save_path: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eval_df["k"], eval_df["silhouette"], marker="o", color="#4CAF50", label="Silhouette")
    ax2 = ax.twinx()
    ax2.plot(eval_df["k"], eval_df["davies_bouldin"], marker="s",
             color="#F44336", linestyle="--", label="Davies-Bouldin")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score", color="#4CAF50")
    ax2.set_ylabel("Davies-Bouldin Score", color="#F44336")
    ax.set_title("Cluster Quality Metrics", fontsize=12, fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    sns.despine()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
