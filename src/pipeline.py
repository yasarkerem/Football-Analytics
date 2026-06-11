"""
End-to-end Tactical Fingerprints pipeline.

Usage:
    python -m src.pipeline [--competitions England Spain] [--clusters 5] [--no-umap]
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

from src.data.loader import load_events, load_matches, load_teams
from src.data.preprocessor import preprocess, filter_passes
from src.features.event_features import extract_event_features
from src.network.builder import build_all_networks
from src.network.metrics import compute_network_metrics
from src.features.aggregator import aggregate_team_profiles, build_fingerprint_matrix
from src.clustering.fingerprints import build_fingerprint_report
from src.visualization.plots import (
    plot_cluster_scatter,
    plot_fingerprint_radar,
    plot_outcome_by_fingerprint,
    plot_pca_variance,
    plot_silhouette_curve,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR   = Path(__file__).resolve().parents[1] / "data" / "results"
REPORTS_DIR   = Path(__file__).resolve().parents[1] / "reports"

for d in [PROCESSED_DIR, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def run(competitions=None, n_clusters=5, use_umap=True):
    print("=" * 60)
    print("Tactical Fingerprints Pipeline")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    events_df  = load_events(competitions)
    matches_df = load_matches(competitions)
    teams_df   = load_teams()
    print(f"  Events:  {len(events_df):,} rows")
    print(f"  Matches: {len(matches_df):,} rows")

    # ── 2. Preprocess ────────────────────────────────────────────
    print("\n[2/7] Preprocessing events...")
    events_clean = preprocess(events_df, matches_df)
    events_clean.to_parquet(PROCESSED_DIR / "events_clean.parquet", index=False)
    print(f"  Clean events: {len(events_clean):,} rows")

    # ── 3. Extract event features ────────────────────────────────
    print("\n[3/7] Extracting event features...")
    event_feats = extract_event_features(events_clean)
    event_feats.to_parquet(PROCESSED_DIR / "event_features.parquet", index=False)
    print(f"  Feature rows: {len(event_feats):,}")

    # ── 4. Build passing networks ────────────────────────────────
    print("\n[4/7] Building passing networks...")
    passes = filter_passes(events_clean, accurate_only=True)
    networks = build_all_networks(passes)
    print(f"  Networks built: {len(networks):,}")

    # ── 5. Compute network metrics ───────────────────────────────
    print("\n[5/7] Computing network metrics...")
    net_metrics = compute_network_metrics(networks)
    net_metrics.to_parquet(PROCESSED_DIR / "network_metrics.parquet", index=False)
    print(f"  Metric rows: {len(net_metrics):,}")

    # ── 6. Aggregate team profiles ───────────────────────────────
    print("\n[6/7] Aggregating team profiles...")
    combined = event_feats.merge(net_metrics, on=["matchId", "teamId"], how="inner")
    profiles = aggregate_team_profiles(combined, matches_df)
    profiles.to_parquet(RESULTS_DIR / "team_profiles.parquet", index=False)
    print(f"  Team profiles: {len(profiles):,}")

    X, y = build_fingerprint_matrix(profiles)
    feature_cols = [c for c in X.columns if c not in {"teamId", "competition"}]

    # ── 7. Discover fingerprints ─────────────────────────────────
    print(f"\n[7/7] Discovering Tactical Fingerprints (k={n_clusters})...")
    result = build_fingerprint_report(
        profiles=profiles.merge(y[["teamId", "competition", "win_rate",
                                    "points_per_match", "goalDiff"]],
                                on=["teamId", "competition"], how="left"),
        feature_cols=feature_cols,
        n_clusters=n_clusters,
        use_umap=use_umap,
    )

    labelled = result["profiles_labelled"]
    labelled = labelled.merge(teams_df[["teamId", "name"]], on="teamId", how="left")
    labelled.to_parquet(RESULTS_DIR / "fingerprints.parquet", index=False)
    labelled.to_csv(RESULTS_DIR / "fingerprints.csv", index=False)

    # Save model artifacts
    with open(RESULTS_DIR / "pipeline_artifacts.pkl", "wb") as f:
        pickle.dump({"scaler": result["scaler"], "pca": result["pca"]}, f)

    # ── Plots ────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_pca_variance(result["pca_explained"],
                      save_path=str(REPORTS_DIR / "pca_variance.png"))
    plot_silhouette_curve(result["eval_df"],
                          save_path=str(REPORTS_DIR / "cluster_quality.png"))
    plot_cluster_scatter(labelled, teams_df=teams_df,
                         save_path=str(REPORTS_DIR / "fingerprint_scatter.png"))
    plot_fingerprint_radar(labelled, feature_cols=feature_cols[:8],
                           save_path=str(REPORTS_DIR / "fingerprint_radar.png"))
    plot_outcome_by_fingerprint(labelled, metric="win_rate",
                                save_path=str(REPORTS_DIR / "win_rate_by_fp.png"))
    plot_outcome_by_fingerprint(labelled, metric="points_per_match",
                                save_path=str(REPORTS_DIR / "ppm_by_fp.png"))

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Fingerprint Summary")
    print("=" * 60)
    summary = labelled.groupby("fingerprint").agg(
        n_teams=("teamId", "nunique"),
        win_rate=("win_rate", "mean"),
        points_per_match=("points_per_match", "mean"),
        avg_pass_accuracy=("pass_accuracy", "mean"),
        avg_long_ball=("long_ball_ratio", "mean"),
        avg_density=("density", "mean"),
        avg_centralization=("centralization", "mean"),
    ).round(3)
    print(summary.to_string())

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Plots saved to:   {REPORTS_DIR}")
    return labelled, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tactical Fingerprints Pipeline")
    parser.add_argument("--competitions", nargs="+", default=None,
                        help="Competitions to include (default: all)")
    parser.add_argument("--clusters", type=int, default=5,
                        help="Number of fingerprint clusters (default: 5)")
    parser.add_argument("--no-umap", action="store_true",
                        help="Use PCA 2D projection instead of UMAP")
    args = parser.parse_args()

    run(
        competitions=args.competitions,
        n_clusters=args.clusters,
        use_umap=not args.no_umap,
    )
