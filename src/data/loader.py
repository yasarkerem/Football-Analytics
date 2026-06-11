"""
Loads raw Wyscout JSON files from the data/raw directory.
"""
import json
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.constants import COMPETITION_FILES, MATCH_FILES

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _load_json(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_events(competitions: list[str] | None = None) -> pd.DataFrame:
    """
    Load and concatenate event data for the specified competitions.
    If competitions is None, all available competitions are loaded.

    Returns a DataFrame with columns:
        matchId, teamId, playerId, eventId, subEventId,
        eventName, subEventName, matchPeriod, eventSec,
        pos_x, pos_y, tags, competition
    """
    targets = competitions or list(COMPETITION_FILES.keys())
    frames = []

    for comp in tqdm(targets, desc="Loading events"):
        fname = COMPETITION_FILES[comp]
        fpath = RAW_DIR / "events" / fname
        if not fpath.exists():
            print(f"[WARN] Missing: {fpath}")
            continue

        raw = _load_json(fpath)
        rows = []
        for ev in raw:
            pos = ev.get("positions", [{}])
            start = pos[0] if pos else {}
            rows.append({
                "matchId":      ev.get("matchId"),
                "teamId":       ev.get("teamId"),
                "playerId":     ev.get("playerId"),
                "eventId":      ev.get("eventId"),
                "subEventId":   ev.get("subEventId"),
                "eventName":    ev.get("eventName", ""),
                "subEventName": ev.get("subEventName", ""),
                "matchPeriod":  ev.get("matchPeriod"),
                "eventSec":     ev.get("eventSec"),
                "pos_x":        start.get("x"),
                "pos_y":        start.get("y"),
                "tags":         [t["id"] for t in ev.get("tags", [])],
                "competition":  comp,
            })
        frames.append(pd.DataFrame(rows))

    if not frames:
        raise FileNotFoundError(
            f"No event files found under {RAW_DIR / 'events'}. "
            "Please copy the Wyscout JSON files there first."
        )

    df = pd.concat(frames, ignore_index=True)
    df["matchId"]  = df["matchId"].astype("Int64")
    df["teamId"]   = df["teamId"].astype("Int64")
    df["playerId"] = df["playerId"].astype("Int64")
    return df


def load_matches(competitions: list[str] | None = None) -> pd.DataFrame:
    """
    Load and flatten match data.

    Returns a DataFrame with columns:
        matchId, competitionId, seasonId, dateutc, gameweek,
        homeTeamId, awayTeamId, homeScore, awayScore, winner,
        duration, competition
    """
    targets = competitions or list(MATCH_FILES.keys())
    frames = []

    for comp in targets:
        fname = MATCH_FILES[comp]
        fpath = RAW_DIR / "matches" / fname
        if not fpath.exists():
            print(f"[WARN] Missing: {fpath}")
            continue

        raw = _load_json(fpath)
        rows = []
        for m in raw:
            teams = m.get("teamsData", {})
            home_id = next((int(k) for k, v in teams.items() if v["side"] == "home"), None)
            away_id = next((int(k) for k, v in teams.items() if v["side"] == "away"), None)
            home_score = teams.get(str(home_id), {}).get("score") if home_id else None
            away_score = teams.get(str(away_id), {}).get("score") if away_id else None

            rows.append({
                "matchId":       m.get("wyId"),
                "competitionId": m.get("competitionId"),
                "seasonId":      m.get("seasonId"),
                "dateutc":       m.get("dateutc"),
                "gameweek":      m.get("gameweek"),
                "homeTeamId":    home_id,
                "awayTeamId":    away_id,
                "homeScore":     home_score,
                "awayScore":     away_score,
                "winner":        m.get("winner"),
                "duration":      m.get("duration"),
                "competition":   comp,
            })
        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)
    df["matchId"] = df["matchId"].astype("Int64")
    return df


def load_players() -> pd.DataFrame:
    path = RAW_DIR / "players.json"
    raw = _load_json(path)
    rows = []
    for p in raw:
        rows.append({
            "playerId":    p.get("wyId"),
            "shortName":   p.get("shortName"),
            "role":        p.get("role", {}).get("code2"),
            "foot":        p.get("foot"),
            "height":      p.get("height"),
            "weight":      p.get("weight"),
            "birthDate":   p.get("birthDate"),
            "currentTeamId": p.get("currentTeamId"),
            "nationality": p.get("passportArea", {}).get("alpha3code"),
        })
    return pd.DataFrame(rows)


def load_teams() -> pd.DataFrame:
    path = RAW_DIR / "teams.json"
    raw = _load_json(path)
    rows = []
    for t in raw:
        rows.append({
            "teamId":       t.get("wyId"),
            "name":         t.get("name"),
            "officialName": t.get("officialName"),
            "city":         t.get("city"),
            "area":         t.get("area", {}).get("name"),
            "teamType":     t.get("type"),
        })
    return pd.DataFrame(rows)


def load_playerank() -> pd.DataFrame:
    path = RAW_DIR / "playerank.json"
    raw = _load_json(path)
    return pd.DataFrame(raw)
