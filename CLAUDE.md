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
    loader.py             — load_events(), load_matches(), load_players(), load_teams()
    preprocessor.py       — preprocess() → clean + tag flags + zones + outcome labels
  features/
    event_features.py     — extract_event_features() → 20 tactical features per (matchId, teamId)
    aggregator.py         — aggregate_team_profiles() → season-level team vectors + outcomes
  network/
    builder.py            — build_all_networks() → dict[(matchId,teamId)] = nx.DiGraph
    metrics.py            — compute_network_metrics() → density, clustering, centralization, etc.
  clustering/
    fingerprints.py       — build_fingerprint_report() → scale→PCA→UMAP→AgglomerativeClustering
  visualization/
    plots.py              — scatter, radar, outcome boxplot, PCA variance, silhouette
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

## Notebook
`notebooks/01_tactical_fingerprints.ipynb` — interactive step-by-step walkthrough.
Set `COMPETITIONS` list at the top to control which leagues to load.

## Output Files
```
data/processed/
  events_clean.parquet    — preprocessed events
  event_features.parquet  — per (matchId, teamId) tactical features
  network_metrics.parquet — per (matchId, teamId) graph metrics
data/results/
  team_profiles.parquet   — aggregated team-season vectors
  fingerprints.parquet    — team profiles with cluster assignments
  fingerprints.csv        — same, human-readable
  pipeline_artifacts.pkl  — scaler + PCA objects
reports/
  fingerprint_scatter.png, fingerprint_radar.png,
  win_rate_by_fp.png, ppm_by_fp.png,
  pca_variance.png, cluster_quality.png
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
