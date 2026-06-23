# Football Analytics — Tactical Fingerprints

## Project Goal
Discover latent tactical identities ("Tactical Fingerprints") from Wyscout event data
and passing network structures. Descriptive + prescriptive analysis for coaches/analysts.

## Dataset
**Source:** Wyscout public dataset (figshare collection 4415000/5)
**Scope:** ~3M events across 7 competitions (Serie A, EPL, La Liga, Ligue 1, Bundesliga, Euro 2016, World Cup 2018)

### Data Location
All raw files must be placed under `data/raw/` (gitignored):
```
data/raw/
  events/   events_England.json, events_Italy.json, events_Spain.json,
            events_France.json, events_Germany.json,
            events_European_Championship.json, events_World_Cup.json
  matches/  matches_*.json  (same 7 competitions)
  players.json, teams.json, coaches.json, referees.json
  playerank.json, eventid2name.csv, tags2name.csv
```

### Key Schema Facts
- **Events:** `matchId, teamId, playerId, eventId, subEventId, matchPeriod, eventSec, positions[{x,y}], tags[{id}]`
- **Positions:** Wyscout normalised coordinates — x∈[0,100] (0=own goal, 100=opponent goal), y∈[0,100]
- **Pass eventId = 8** (subEvents: 80=cross, 85=simple, 86=smart, 84=launch/long ball)
- **Tag 1801 = accurate, 1802 = not accurate, 1901 = counter_attack, 703 = duel won**
- **Matches:** `teamsData` keyed by teamId string; `winner=0` means draw

## Architecture

```
src/
  utils/constants.py      — all magic numbers (event IDs, tag IDs, zone boundaries)
  data/
    loader.py             — load_events(), load_matches(), load_players(), load_teams(),
                            load_playerank()
    preprocessor.py       — preprocess() → clean + tag flags + zones + outcome labels
                            filter_passes(df, accurate_only) → pass subset
  features/
    event_features.py     — extract_event_features() → 20 tactical features per (matchId, teamId)
    aggregator.py         — aggregate_team_profiles() → season-level team vectors + outcomes
                            build_fingerprint_matrix() → (X, y) split for clustering
  network/
    builder.py            — build_all_networks() → dict[(matchId,teamId)] = nx.DiGraph
    metrics.py            — compute_network_metrics() → density, clustering, centralization, etc.
  clustering/
    fingerprints.py       — scale_features(), run_pca(), find_optimal_clusters()
                            build_fingerprint_report() → scale→PCA→UMAP→AgglomerativeClustering
  visualization/
    plots.py              — plot_cluster_scatter(), plot_fingerprint_radar(),
                            plot_outcome_by_fingerprint(), plot_pca_variance(),
                            plot_silhouette_curve()
  pipeline.py             — end-to-end CLI runner
```

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (all competitions, 5 clusters)
python -m src.pipeline

# Subset for speed
python -m src.pipeline --competitions England Spain Germany --clusters 5

# Skip UMAP (faster, uses PCA 2D projection instead)
python -m src.pipeline --no-umap
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/01_tactical_fingerprints.ipynb` | Original step-by-step walkthrough |
| `notebooks/03_full_analysis.ipynb` | **Primary notebook — full end-to-end analysis, Run All** |

### `03_full_analysis.ipynb` (Google Colab)
Designed for Google Colab with Drive-backed data. Covers all pipeline stages in a single notebook:

1. **Setup** — Drive mount, repo clone/pull, `pip install`
2. **Load** — `load_events()`, `load_matches()`, `load_teams()`, `load_players()`, `load_playerank()`
3. **EDA** — event type distribution, spatial heatmaps per competition, match outcome pie, player role bar chart, pass accuracy by competition
4. **Preprocess** — `preprocess()` + `filter_passes()`
5. **Event Features** — `extract_event_features()` + correlation heatmap
6. **Passing Networks** — `build_all_networks()` + `compute_network_metrics()` + sample network viz + metric distributions
7. **Team Profiles** — `aggregate_team_profiles()` + top-3 teams per competition + feature vs win-rate scatter grid
8. **PCA** — variance curve + top feature loadings per PC
9. **Optimal k** — silhouette curve via `find_optimal_clusters()`
10. **Fingerprints** — `build_fingerprint_report()` (UMAP scatter + radar + outcome boxplots + feature heatmap)
11. **Cross-competition** — fingerprint distribution stacked bar + tactical style comparison
12. **Save** — all artefacts written to `DRIVE_OUT` (`fingerprints.csv/.parquet`, `pipeline_artifacts.pkl`, all PNGs)

**Key config variables at top of notebook:**
```python
DRIVE_DATA   = '/content/drive/MyDrive/Football_Events_SDS/Data'
DRIVE_OUT    = '/content/drive/MyDrive/Football_Events_SDS/Results'
COMPETITIONS = ['England','Spain','Italy','France','Germany',
                'European_Championship','World_Cup']
N_CLUSTERS   = 5
```

## Output Files

### CLI pipeline (`data/`, `reports/`)
```
data/processed/
  events_clean.parquet       — preprocessed events
  event_features.parquet     — per (matchId, teamId) tactical features
  network_metrics.parquet    — per (matchId, teamId) graph metrics
data/results/
  team_profiles.parquet      — aggregated team-season vectors
  fingerprints.parquet       — team profiles with cluster assignments
  fingerprints.csv           — same, human-readable
  pipeline_artifacts.pkl     — scaler + PCA objects
reports/
  fingerprint_scatter.png, fingerprint_radar.png,
  win_rate_by_fp.png, ppm_by_fp.png,
  pca_variance.png, cluster_quality.png
```

### Notebook outputs — committed to `Results/`
```
Results/
  fingerprints.csv                  — labelled team profiles (human-readable)
  fingerprint_summary.csv           — per-cluster mean stats
  eda_event_distribution.png        — event type + events per competition
  eda_spatial_heatmap.png           — pitch density heatmap per competition
  eda_match_outcomes.png            — goals distribution + result pie
  eda_player_roles.png              — players by position
  eda_pass_accuracy.png             — pass accuracy by competition
  feature_correlation.png           — 20-feature correlation heatmap
  sample_passing_network.png        — sample team pass graph
  network_metric_distributions.png  — 6-panel network metric histograms
  feature_vs_win_rate.png           — 8 key features vs win rate scatter
  pca_variance.png                  — cumulative explained variance
  pca_loadings.png                  — top feature loadings for PC1–3
  cluster_quality.png               — silhouette curve
  fingerprint_scatter.png           — UMAP scatter coloured by fingerprint
  fingerprint_radar.png             — radar chart per cluster
  win_rate_y_by_fp.png              — win rate boxplot by fingerprint
  points_per_match_y_by_fp.png      — points per match boxplot by fingerprint
  goalDiff_y_by_fp.png              — goal difference boxplot by fingerprint
  fingerprint_feature_heatmap.png   — normalised feature means per cluster
  fingerprint_by_competition.png    — stacked bar of FP distribution per league
  style_by_competition.png          — tactical style bar chart per league
```

## Feature Groups (20 event + 10 network = 30 total)

| Group | Features |
|---|---|
| Volume | n_events, n_passes, n_shots, n_duels |
| Efficiency | pass_accuracy, shot_accuracy, duel_win_rate |
| Spatial | avg_pos_x/y, std_pos_x/y |
| Zone | pass_def_third, pass_mid_third, pass_att_third |
| Style | cross_ratio, simple_pass_ratio, smart_pass_ratio, high_pass_ratio, long_ball_ratio, counter_rate, dangerous_rate |
| Network | density, clustering_coef, weighted_clustering, centralization, top_player_share, avg_shortest_path, diameter, reciprocity, n_nodes, n_edges |

## Key Design Decisions
- Passing networks use **consecutive pass sequences** (sorted by matchPeriod + eventSec) as a proxy for passer→receiver, since Wyscout events do not encode the receiver directly.
- **Accurate passes only** (tag 1801) used for network construction — noise reduction.
- Fingerprints cluster at **team-season level** (one vector per team per competition), not per-match.
- PCA (10 components) feeds into AgglomerativeClustering (Ward linkage); UMAP used for 2D visualisation only.
