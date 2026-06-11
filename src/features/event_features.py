"""
Extract team-match level tactical features from event data.
"""
import numpy as np
import pandas as pd
from src.utils.constants import (
    EVENT_PASS, EVENT_SHOT, EVENT_DUEL, EVENT_FREE_KICK, EVENT_SAVE,
    PASS_CROSS, PASS_SIMPLE, PASS_SMART, PASS_HIGH, PASS_LAUNCH,
    TAG_ACCURATE, TAG_COUNTER, TAG_DANGEROUS,
)


def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def extract_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (matchId, teamId) tactical features from preprocessed events.

    Feature groups:
        - Volume: total events, passes, shots, duels, fouls
        - Efficiency: pass accuracy, shot accuracy, duel win rate
        - Spatial: avg positions, zone distribution, directness
        - Style: counter-attack rate, long ball ratio, cross ratio, smart pass ratio
    """
    results = []

    for (match_id, team_id), grp in df.groupby(["matchId", "teamId"]):
        passes   = grp[grp["eventId"] == EVENT_PASS]
        shots    = grp[grp["eventId"] == EVENT_SHOT]
        duels    = grp[grp["eventId"] == EVENT_DUEL]

        n_events  = len(grp)
        n_passes  = len(passes)
        n_shots   = len(shots)
        n_duels   = len(duels)

        acc_passes = passes["accurate"].sum() if "accurate" in passes.columns else 0
        acc_shots  = shots["accurate"].sum()  if "accurate" in shots.columns  else 0
        won_duels  = (duels["tags"].apply(lambda t: 703 in t)).sum()  # tag 703 = won

        # Spatial averages
        avg_x = grp["pos_x"].mean()
        avg_y = grp["pos_y"].mean()
        std_x = grp["pos_x"].std()
        std_y = grp["pos_y"].std()

        # Zone distribution (passes only for tactical relevance)
        def zone_ratio(zone_label):
            if "zone" not in passes.columns or n_passes == 0:
                return 0.0
            return (passes["zone"] == zone_label).sum() / n_passes

        # Pass sub-type ratios
        def pass_subtype_ratio(subtype_id):
            return _safe_div((passes["subEventId"] == subtype_id).sum(), n_passes)

        cross_ratio      = pass_subtype_ratio(PASS_CROSS)
        simple_ratio     = pass_subtype_ratio(PASS_SIMPLE)
        smart_ratio      = pass_subtype_ratio(PASS_SMART)
        high_pass_ratio  = pass_subtype_ratio(PASS_HIGH)
        long_ball_ratio  = pass_subtype_ratio(PASS_LAUNCH)

        counter_rate  = _safe_div(grp.get("counter_attack", pd.Series(dtype=bool)).sum(), n_events)
        dangerous_rate = _safe_div(grp.get("dangerous", pd.Series(dtype=bool)).sum(), n_events)

        results.append({
            "matchId":   match_id,
            "teamId":    team_id,
            # Volume
            "n_events":  n_events,
            "n_passes":  n_passes,
            "n_shots":   n_shots,
            "n_duels":   n_duels,
            # Efficiency
            "pass_accuracy":  _safe_div(acc_passes, n_passes),
            "shot_accuracy":  _safe_div(acc_shots, n_shots),
            "duel_win_rate":  _safe_div(won_duels, n_duels),
            # Spatial
            "avg_pos_x":  avg_x,
            "avg_pos_y":  avg_y,
            "std_pos_x":  std_x,
            "std_pos_y":  std_y,
            # Zone ratios (from passes)
            "pass_def_third":  zone_ratio("defensive_third"),
            "pass_mid_third":  zone_ratio("middle_third"),
            "pass_att_third":  zone_ratio("attacking_third"),
            # Style
            "cross_ratio":      cross_ratio,
            "simple_pass_ratio": simple_ratio,
            "smart_pass_ratio":  smart_ratio,
            "high_pass_ratio":   high_pass_ratio,
            "long_ball_ratio":   long_ball_ratio,
            "counter_rate":      counter_rate,
            "dangerous_rate":    dangerous_rate,
        })

    return pd.DataFrame(results)
