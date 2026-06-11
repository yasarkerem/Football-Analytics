"""
Compute graph-level network metrics for each passing network.
"""
import networkx as nx
import numpy as np
import pandas as pd


def _safe(fn, default=0.0):
    try:
        return fn()
    except Exception:
        return default


def compute_network_metrics(networks: dict) -> pd.DataFrame:
    """
    For each (matchId, teamId) network, compute:
        - density
        - clustering_coefficient
        - centralization (degree)
        - avg_shortest_path (on largest weakly-connected component)
        - diameter (on largest WCC)
        - n_nodes, n_edges
        - top_player_share (fraction of passes through most central player)
        - reciprocity
        - weighted_clustering
    """
    rows = []

    for (match_id, team_id), G in networks.items():
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        if n_nodes < 2:
            continue

        UG = G.to_undirected()

        density = _safe(lambda: nx.density(G))
        clustering = _safe(lambda: nx.average_clustering(UG))
        reciprocity = _safe(lambda: nx.reciprocity(G))
        weighted_clustering = _safe(lambda: nx.average_clustering(UG, weight="weight"))

        # Degree centralization: (max_degree - avg_degree) / (n-1)
        degrees = dict(G.degree())
        if degrees:
            deg_vals = list(degrees.values())
            max_deg = max(deg_vals)
            avg_deg = np.mean(deg_vals)
            centralization = (max_deg - avg_deg) / (n_nodes - 1) if n_nodes > 1 else 0.0
        else:
            centralization = 0.0

        # Top player share by betweenness centrality
        try:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            top_player_share = max(bc.values()) if bc else 0.0
        except Exception:
            top_player_share = 0.0

        # Shortest path on largest WCC
        try:
            wcc = max(nx.weakly_connected_components(G), key=len)
            sub = G.subgraph(wcc)
            avg_shortest_path = nx.average_shortest_path_length(sub)
            diameter = nx.diameter(sub.to_undirected())
        except Exception:
            avg_shortest_path = 0.0
            diameter = 0

        rows.append({
            "matchId":             match_id,
            "teamId":              team_id,
            "n_nodes":             n_nodes,
            "n_edges":             n_edges,
            "density":             density,
            "clustering_coef":     clustering,
            "weighted_clustering": weighted_clustering,
            "centralization":      centralization,
            "top_player_share":    top_player_share,
            "avg_shortest_path":   avg_shortest_path,
            "diameter":            diameter,
            "reciprocity":         reciprocity,
        })

    return pd.DataFrame(rows)
