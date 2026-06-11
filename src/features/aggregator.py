"""
Aggregate per-match features into per-team season-level profiles
and attach match outcome statistics.
"""
import pandas as pd
import numpy as np


def aggregate_team_profiles(
    feature_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-match-team features to per-team mean profiles.
    Also computes win_rate, avg_goals_for, avg_goals_against, points_per_match.
    """
    # Attach outcome info
    home = matches_df[["matchId", "homeTeamId", "awayTeamId",
                        "homeScore", "awayScore", "winner", "competition"]].copy()
    home["teamId"]       = home["homeTeamId"]
    home["goalsFor"]     = home["homeScore"]
    home["goalsAgainst"] = home["awayScore"]

    away = matches_df[["matchId", "homeTeamId", "awayTeamId",
                        "homeScore", "awayScore", "winner", "competition"]].copy()
    away["teamId"]       = away["awayTeamId"]
    away["goalsFor"]     = away["awayScore"]
    away["goalsAgainst"] = away["homeScore"]

    outcomes = pd.concat([home, away], ignore_index=True)[
        ["matchId", "teamId", "goalsFor", "goalsAgainst", "winner", "competition"]
    ]
    outcomes["win"]   = (outcomes["winner"] == outcomes["teamId"]).astype(int)
    outcomes["draw"]  = (outcomes["winner"] == 0).astype(int)
    outcomes["loss"]  = (~outcomes["win"].astype(bool) & ~outcomes["draw"].astype(bool)).astype(int)
    outcomes["points"] = outcomes["win"] * 3 + outcomes["draw"]
    outcomes["goalDiff"] = outcomes["goalsFor"] - outcomes["goalsAgainst"]

    merged = feature_df.merge(
        outcomes[["matchId", "teamId", "goalsFor", "goalsAgainst",
                  "win", "draw", "loss", "points", "goalDiff", "competition"]],
        on=["matchId", "teamId"],
        how="left",
    )

    # Tactical feature columns (numeric only, exclude id/label cols)
    exclude = {"matchId", "teamId", "goalsFor", "goalsAgainst",
               "win", "draw", "loss", "points", "goalDiff", "competition"}
    feature_cols = [c for c in merged.columns if c not in exclude]

    # Group by team (and competition to keep context)
    group_cols = ["teamId", "competition"]
    agg_dict = {col: "mean" for col in feature_cols}
    agg_dict.update({
        "win":        "mean",
        "draw":       "mean",
        "loss":       "mean",
        "points":     "mean",
        "goalDiff":   "mean",
        "goalsFor":   "mean",
        "goalsAgainst": "mean",
    })

    profiles = merged.groupby(group_cols).agg(agg_dict).reset_index()
    profiles.rename(columns={"win": "win_rate", "points": "points_per_match"}, inplace=True)

    return profiles


def build_fingerprint_matrix(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the profiles DataFrame into:
        X  — feature matrix (tactical + network metrics)
        y  — outcome labels (win_rate, points_per_match, goalDiff)
    """
    outcome_cols = {"win_rate", "win", "draw", "loss",
                    "points_per_match", "goalDiff", "goalsFor", "goalsAgainst"}
    id_cols      = {"teamId", "competition"}
    feature_cols = [c for c in profiles.columns
                    if c not in outcome_cols and c not in id_cols]

    X = profiles[["teamId", "competition"] + feature_cols].copy()
    y = profiles[["teamId", "competition"] +
                 [c for c in outcome_cols if c in profiles.columns]].copy()

    return X, y
