"""
Dimensionality reduction and clustering to discover Tactical Fingerprints.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def scale_features(X_raw: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X_raw), scaler


def run_pca(X_scaled: np.ndarray, n_components: int = 10) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def run_umap(X_scaled: np.ndarray, n_components: int = 2, **kwargs):
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is not installed. Run: pip install umap-learn")
    reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
    return reducer.fit_transform(X_scaled), reducer


def find_optimal_clusters(
    X_reduced: np.ndarray,
    k_range: range = range(2, 9),
    linkage: str = "ward",
) -> pd.DataFrame:
    """
    Evaluate hierarchical clustering across a range of k values.
    Returns a DataFrame with silhouette and Davies-Bouldin scores per k.
    """
    records = []
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X_reduced)
        sil = silhouette_score(X_reduced, labels)
        db  = davies_bouldin_score(X_reduced, labels)
        records.append({"k": k, "silhouette": sil, "davies_bouldin": db})
    return pd.DataFrame(records)


def cluster_fingerprints(
    X_reduced: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(X_reduced)


def build_fingerprint_report(
    profiles: pd.DataFrame,
    feature_cols: list[str],
    n_pca_components: int = 10,
    n_clusters: int = 5,
    use_umap: bool = True,
) -> dict:
    """
    Full pipeline: scale → PCA → (UMAP) → cluster → attach labels.

    Returns a dict with:
        profiles_labelled  — original profiles with cluster assignments
        pca_explained      — cumulative explained variance
        eval_df            — silhouette/DB scores across k values
        embed_2d           — 2D embedding coordinates (UMAP or PCA[:2])
    """
    X_raw = profiles[feature_cols].fillna(0).values
    X_scaled, scaler = scale_features(X_raw)

    X_pca, pca = run_pca(X_scaled, n_components=n_pca_components)
    pca_explained = np.cumsum(pca.explained_variance_ratio_)

    if use_umap and UMAP_AVAILABLE:
        X_embed, _ = run_umap(X_scaled, n_components=2)
    else:
        X_embed = X_pca[:, :2]

    eval_df = find_optimal_clusters(X_pca)
    labels  = cluster_fingerprints(X_pca, n_clusters=n_clusters)

    labelled = profiles.copy()
    labelled["fingerprint"]  = labels
    labelled["embed_x"]      = X_embed[:, 0]
    labelled["embed_y"]      = X_embed[:, 1]

    return {
        "profiles_labelled": labelled,
        "pca_explained":     pca_explained,
        "eval_df":           eval_df,
        "embed_2d":          X_embed,
        "scaler":            scaler,
        "pca":               pca,
    }
