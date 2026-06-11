"""
Build passing networks for each (matchId, teamId) pair.
"""
import networkx as nx
import pandas as pd
from src.utils.constants import EVENT_PASS


def build_passing_network(passes: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed weighted passing network from a slice of pass events
    for a single team in a single match.

    Nodes   = playerIds
    Edges   = directed passes (passer → receiver inferred from consecutive events)
    Weight  = number of passes on that edge
    """
    G = nx.DiGraph()

    # Sort by period then time to get pass sequence
    passes = passes.sort_values(["matchPeriod", "eventSec"])
    player_seq = passes["playerId"].tolist()

    for i in range(len(player_seq) - 1):
        src = player_seq[i]
        dst = player_seq[i + 1]
        if src == dst:
            continue
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1
        else:
            G.add_edge(src, dst, weight=1)

    return G


def build_all_networks(passes_df: pd.DataFrame) -> dict[tuple, nx.DiGraph]:
    """
    Build passing networks for all (matchId, teamId) combinations.

    Returns a dict keyed by (matchId, teamId).
    """
    networks = {}
    for (match_id, team_id), grp in passes_df.groupby(["matchId", "teamId"]):
        if len(grp) < 5:
            continue
        networks[(match_id, team_id)] = build_passing_network(grp)
    return networks
