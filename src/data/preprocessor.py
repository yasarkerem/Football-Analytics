"""
Cleans and enriches the raw events DataFrame.
"""
import pandas as pd
import numpy as np
from src.utils.constants import (
    TAG_ACCURATE, TAG_NOT_ACCURATE, TAG_COUNTER, TAG_DANGEROUS,
    EVENT_PASS, EVENT_SHOT, EVENT_DUEL, EVENT_FREE_KICK,
    DEFENSIVE_THIRD_END, MIDDLE_THIRD_END,
    PITCH_LENGTH, PITCH_WIDTH,
)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """Drop events with missing spatial or identity data."""
    df = df.dropna(subset=["pos_x", "pos_y", "teamId", "playerId", "matchId"])
    df = df[df["pos_x"].between(0, 100) & df["pos_y"].between(0, 100)]
    return df.reset_index(drop=True)


def add_tag_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the tags list into boolean flag columns."""
    df = df.copy()
    df["accurate"]       = df["tags"].apply(lambda t: TAG_ACCURATE in t)
    df["not_accurate"]   = df["tags"].apply(lambda t: TAG_NOT_ACCURATE in t)
    df["counter_attack"] = df["tags"].apply(lambda t: TAG_COUNTER in t)
    df["dangerous"]      = df["tags"].apply(lambda t: TAG_DANGEROUS in t)
    return df


def add_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a pitch zone based on x-coordinate.
    Wyscout x=0 is the team's own goal line; x=100 is the opponent's.
    """
    df = df.copy()
    bins   = [0, DEFENSIVE_THIRD_END, MIDDLE_THIRD_END, 100.0]
    labels = ["defensive_third", "middle_third", "attacking_third"]
    df["zone"] = pd.cut(df["pos_x"], bins=bins, labels=labels, include_lowest=True)
    return df


def filter_passes(df: pd.DataFrame, accurate_only: bool = True) -> pd.DataFrame:
    """Return only pass events, optionally filtered to accurate passes."""
    passes = df[df["eventId"] == EVENT_PASS].copy()
    if accurate_only:
        passes = passes[passes["accurate"]]
    return passes.reset_index(drop=True)


def filter_shots(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["eventId"] == EVENT_SHOT].copy().reset_index(drop=True)


def add_match_outcome(
    events: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach win/draw/loss label for each (matchId, teamId) pair.

    Outcome from the perspective of teamId:
        'win', 'draw', 'loss'
    """
    match_info = matches[["matchId", "homeTeamId", "awayTeamId",
                           "homeScore", "awayScore", "winner"]].copy()

    def outcome_for_team(row):
        if row["winner"] == 0:
            return "draw"
        if row["winner"] == row["teamId"]:
            return "win"
        return "loss"

    # Build a flat (matchId, teamId, outcome, goalsFor, goalsAgainst) table
    home_rows = match_info.copy()
    home_rows["teamId"]        = home_rows["homeTeamId"]
    home_rows["goalsFor"]      = home_rows["homeScore"]
    home_rows["goalsAgainst"]  = home_rows["awayScore"]

    away_rows = match_info.copy()
    away_rows["teamId"]        = away_rows["awayTeamId"]
    away_rows["goalsFor"]      = away_rows["awayScore"]
    away_rows["goalsAgainst"]  = away_rows["homeScore"]

    flat = pd.concat([home_rows, away_rows], ignore_index=True)
    flat["outcome"] = flat.apply(outcome_for_team, axis=1)
    flat = flat[["matchId", "teamId", "outcome", "goalsFor", "goalsAgainst"]]
    flat["goalDiff"] = flat["goalsFor"] - flat["goalsAgainst"]

    return events.merge(flat, on=["matchId", "teamId"], how="left")


def preprocess(
    events: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = clean_events(events)
    df = add_tag_flags(df)
    df = add_zone(df)
    df = add_match_outcome(df, matches)
    return df
