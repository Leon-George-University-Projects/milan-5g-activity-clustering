"""
Milan Telecom Dataset - Clustering Analysis
============================================
Dataset: sms-call-internet-mi-2013-{11,12}-XX  (62 days — Nov + Dec 2013)
Columns: grid_id, timestamp, country_code, sms_in, sms_out, call_in, call_out, internet_activity

Pipeline:
  1. Load ALL daily files from DATA_DIR (all months, all days)
  2. Aggregate per 10-SECOND interval (timestamp // 10_000 * 10_000)
  3. Group grid cells into groups of 6  →  effective resolution: 1 minute per super-cell
  4. Feature engineering (spatial + temporal)
  5. K-Means + Hierarchical Clustering
  6. Evaluation metrics + visualizations
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, to_rgba
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.impute import SimpleImputer
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"       # subfolder containing all daily .txt files
OUTPUT_DIR = "."          # where to save figures
N_CLUSTERS = 5            # number of clusters
GRID_GROUP = 6            # merge every 6 grid cells → 1-minute effective resolution
TEN_SEC_MS = 10_000       # 10 seconds in milliseconds  (6 × 10s = 1 min per super-cell)
RANDOM_STATE = 42
CHUNKSIZE = 2_000_000  # rows per read_csv chunk; lower this if RAM is still limited

PALETTE = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261",
           "#6A4C93","#1982C4","#8AC926","#FF595E","#FFCA3A"]

# Visualization-only palette: labels remain size-aligned, colors show activity intensity.
ACTIVITY_PALETTE = [
    "#E63946",  # highest average activity
    "#F4A261",  # 2nd highest
    "#E9C46A",  # 3rd highest
    "#2A9D8F",  # 4th highest
    "#457B9D",  # 5th highest
    "#6A4C93",
    "#1982C4",
    "#8AC926",
    "#FF595E",
    "#FFCA3A",
]

# ─── MEMORY-EFFICIENT LOADING + AGGREGATION ───────────────────────────────────
def _find_data_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "sms-call-internet-mi-*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No data files found in '{data_dir}'.\n"
            "Place the sms-call-internet-mi-*.txt files in the data/ subfolder."
        )
    return files


def _build_grid_map(files, group_size: int = GRID_GROUP) -> dict:
    """Build the same grid_id -> super_cell mapping as the old group_grids()."""
    unique_grids = set()
    for f in files:
        for chunk in pd.read_csv(
            f, sep="\t", header=None, usecols=[0], names=["grid_id"],
            dtype={"grid_id": "int32"}, chunksize=CHUNKSIZE,
        ):
            unique_grids.update(chunk["grid_id"].dropna().unique().tolist())
    unique_grids = sorted(unique_grids)
    return {g: (i // group_size) for i, g in enumerate(unique_grids)}


def _reduce_partials(partials):
    """Merge partial aggregated chunks to keep memory usage stable."""
    if len(partials) == 1:
        return partials[0]
    return (
        pd.concat(partials, ignore_index=True)
          .groupby(["super_cell", "ts_10sec"], as_index=False, sort=False)
          [["sms_in", "sms_out", "call_in", "call_out", "internet"]]
          .sum()
    )


def load_aggregate_and_group(data_dir: str, group_size: int = GRID_GROUP) -> pd.DataFrame:
    """Memory-efficient equivalent of load_data -> aggregate_10sec -> group_grids."""
    files = _find_data_files(data_dir)
    print(f"  Found {len(files)} file(s)")

    print("  Building grid_id → super_cell map...")
    grid_map = _build_grid_map(files, group_size)
    print(f"  Grid map: {len(grid_map):,} grid cells → {max(grid_map.values()) + 1:,} super-cells")

    col_names = ["grid_id", "timestamp", "country_code", "sms_in", "sms_out", "call_in", "call_out", "internet"]
    usecols = ["grid_id", "timestamp", "sms_in", "sms_out", "call_in", "call_out", "internet"]
    dtypes = {
        "grid_id": "int32", "timestamp": "int64",
        "sms_in": "float32", "sms_out": "float32",
        "call_in": "float32", "call_out": "float32", "internet": "float32",
    }

    partials = []
    total_rows = 0

    for file_idx, f in enumerate(files, start=1):
        print(f"  Reading + aggregating file {file_idx}/{len(files)}: {os.path.basename(f)}")
        reader = pd.read_csv(
            f, sep="\t", header=None, names=col_names, usecols=usecols,
            dtype=dtypes, chunksize=CHUNKSIZE,
        )
        for chunk in reader:
            total_rows += len(chunk)
            chunk["ts_10sec"] = (chunk["timestamp"] // TEN_SEC_MS) * TEN_SEC_MS
            chunk["super_cell"] = chunk["grid_id"].map(grid_map).astype("int32")

            part = (
                chunk.groupby(["super_cell", "ts_10sec"], as_index=False, sort=False)
                     [["sms_in", "sms_out", "call_in", "call_out", "internet"]]
                     .sum()
            )
            partials.append(part)

            if len(partials) >= 20:
                partials = [_reduce_partials(partials)]

            del chunk, part

    print(f"  Raw rows processed: {total_rows:,}")
    print("  Final aggregation across chunks...")
    grouped = _reduce_partials(partials)
    grouped["datetime"] = pd.to_datetime(grouped["ts_10sec"], unit="ms")
    grouped["total_activity"] = grouped[["sms_in", "sms_out", "call_in", "call_out", "internet"]].sum(axis=1)

    # Keep the spatial mapping so we can later draw the Milan grid map.
    # grid_id in the Milano CDR dataset follows a 100x100 spatial grid.
    grouped.attrs["grid_map"] = grid_map

    print(f"  After 10-sec aggregation + grid grouping (/{group_size}): {len(grouped):,} rows | "
          f"{grouped['super_cell'].nunique()} super-cells")
    return grouped


# ─── 1. LOAD DATA ──────────────────────────────────────────────────────────────
def load_data(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "sms-call-internet-mi-*.txt")))
    if not files:
        raise FileNotFoundError(
            f"No data files found in '{data_dir}'.\n"
            "Place the sms-call-internet-mi-*.txt files in the data/ subfolder."
        )
    print(f"  Found {len(files)} file(s)")

    col_names = ["grid_id", "timestamp", "country_code",
                 "sms_in", "sms_out", "call_in", "call_out", "internet"]
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep="\t", header=None, names=col_names)
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    print(f"  Raw rows: {len(raw):,}")
    return raw


# ─── 2. AGGREGATE PER 10-MIN INTERVAL ─────────────────────────────────────────
def aggregate_10sec(raw: pd.DataFrame) -> pd.DataFrame:
    raw["ts_10sec"] = (raw["timestamp"] // TEN_SEC_MS) * TEN_SEC_MS
    agg = (
        raw.groupby(["grid_id", "ts_10sec"], as_index=False)
           [["sms_in","sms_out","call_in","call_out","internet"]]
           .sum()
    )
    agg["datetime"] = pd.to_datetime(agg["ts_10sec"], unit="ms")
    agg["total_activity"] = agg[["sms_in","sms_out","call_in","call_out","internet"]].sum(axis=1)
    print(f"  After 10-sec aggregation: {len(agg):,} rows")
    return agg


# ─── 3. GROUP GRID CELLS (every 6) ─────────────────────────────────────────────
def group_grids(agg: pd.DataFrame, group_size: int = 6) -> pd.DataFrame:
    unique_grids = sorted(agg["grid_id"].unique())
    grid_map = {g: (i // group_size) for i, g in enumerate(unique_grids)}
    agg["super_cell"] = agg["grid_id"].map(grid_map)
    grouped = (
        agg.groupby(["super_cell", "ts_10sec"], as_index=False)
           [["sms_in","sms_out","call_in","call_out","internet","total_activity"]]
           .sum()
    )
    grouped["datetime"] = pd.to_datetime(grouped["ts_10sec"], unit="ms")
    print(f"  After grid grouping (/{group_size}): {len(grouped):,} rows | "
          f"{grouped['super_cell'].nunique()} super-cells")
    return grouped


# ─── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(grouped: pd.DataFrame) -> pd.DataFrame:
    df = grouped.copy()
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek   # 0=Mon
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Activity-weighted temporal activity.
    # rush/night/weekend ratios now depend on the real traffic volume in the dataset,
    # not only on the number of timestamps that belong to each binary time category.
    df["rush_activity"]    = df["total_activity"] * df["is_rush_hour"]
    df["night_activity"]   = df["total_activity"] * df["is_night"]
    df["weekend_activity"] = df["total_activity"] * df["is_weekend"]

    feat = df.groupby("super_cell").agg(
        mean_sms_in        = ("sms_in",   "mean"),
        mean_sms_out       = ("sms_out",  "mean"),
        mean_call_in       = ("call_in",  "mean"),
        mean_call_out      = ("call_out", "mean"),
        mean_internet      = ("internet", "mean"),
        std_internet       = ("internet", "std"),
        mean_total         = ("total_activity", "mean"),
        std_total          = ("total_activity", "std"),
        peak_total_act     = ("total_activity", "max"),
        total_activity_sum = ("total_activity", "sum"),
        rush_activity_sum  = ("rush_activity", "sum"),
        night_activity_sum = ("night_activity", "sum"),
        weekend_activity_sum = ("weekend_activity", "sum"),
    ).reset_index()

    period_counts = df.groupby("super_cell").agg(
        rush_slots    = ("is_rush_hour", "sum"),
        night_slots   = ("is_night", "sum"),
        weekend_slots = ("is_weekend", "sum"),
    ).reset_index()
    feat = feat.merge(period_counts, on="super_cell", how="left")

    denom = feat["total_activity_sum"] + 1e-9
    feat["rush_ratio"]    = feat["rush_activity_sum"] / denom
    feat["night_ratio"]   = feat["night_activity_sum"] / denom
    feat["weekend_ratio"] = feat["weekend_activity_sum"] / denom

    # True temporal intensity metrics from the dataset: average activity only
    # inside rush/night/weekend slots for each super-cell.
    feat["peak_rush_mean"]    = feat["rush_activity_sum"] / (feat["rush_slots"] + 1e-9)
    feat["peak_night_mean"]   = feat["night_activity_sum"] / (feat["night_slots"] + 1e-9)
    feat["peak_weekend_mean"] = feat["weekend_activity_sum"] / (feat["weekend_slots"] + 1e-9)

    feat["cv_total"] = feat["std_total"] / (feat["mean_total"] + 1e-9)
    feat["internet_share"] = feat["mean_internet"] / (feat["mean_total"] + 1e-9)

    feat = feat.drop(columns=[
        "total_activity_sum", "rush_activity_sum", "night_activity_sum",
        "weekend_activity_sum", "rush_slots", "night_slots", "weekend_slots"
    ])

    print(f"  Feature matrix: {feat.shape}")
    return feat, df


# ─── 5. PREPROCESS ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "mean_sms_in", "mean_sms_out", "mean_call_in", "mean_call_out",
    "mean_internet", "std_internet", "mean_total", "std_total",
    "peak_total_act", "peak_rush_mean", "peak_night_mean", "peak_weekend_mean",
    "rush_ratio", "night_ratio", "weekend_ratio",
    "cv_total", "internet_share"
]
def preprocess(feat: pd.DataFrame):
    X_raw = feat[FEATURE_COLS].values
    imp   = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_raw)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_imp)
    return X_sc


# ─── 6. CLUSTERING ─────────────────────────────────────────────────────────────
def run_kmeans(X, k=N_CLUSTERS):
    km = KMeans(n_clusters=k, n_init=15, max_iter=300, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    return labels, km

def run_hierarchical(X, k=N_CLUSTERS):
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hc.fit_predict(X)
    return labels

def relabel_clusters_by_size(labels, name=""):
    """Relabel clusters by descending size so C0 is always the largest cluster.

    This does NOT change the clustering result or any metric. It only renames
    cluster IDs, making K-Means and Hierarchical plots easier to compare.
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)

    # Larger clusters get smaller label IDs. Tie-break by old label for stability.
    ordered_old_labels = [old for old, _ in sorted(zip(unique, counts), key=lambda x: (-x[1], x[0]))]
    mapping = {old: new for new, old in enumerate(ordered_old_labels)}
    relabeled = np.array([mapping[x] for x in labels], dtype=int)

    if name:
        mapping_txt = ", ".join([f"old C{old}->new C{new}" for old, new in mapping.items()])
        print(f"  [{name}] Cluster labels reordered by size: {mapping_txt}")

    return relabeled, mapping

def make_activity_color_map(feat: pd.DataFrame, labels, name: str = ""):
    """Assign colors by cluster average activity without changing labels/results.

    Cluster IDs remain exactly as produced by relabel_clusters_by_size()
    (C0 = largest, C1 = second largest, etc.). Only the color is chosen by
    mean_total activity: highest activity -> red, second -> orange, etc.
    """
    tmp = feat[["super_cell", "mean_total"]].copy()
    tmp["cluster"] = np.asarray(labels)
    activity_rank = tmp.groupby("cluster")["mean_total"].mean().sort_values(ascending=False)

    color_map = {
        int(cluster): ACTIVITY_PALETTE[i % len(ACTIVITY_PALETTE)]
        for i, cluster in enumerate(activity_rank.index)
    }

    if name:
        txt = ", ".join(
            f"C{int(cluster)}={value:.3f}->{color_map[int(cluster)]}"
            for cluster, value in activity_rank.items()
        )
        print(f"  [{name}] Cluster colors by mean_total activity: {txt}")

    return color_map


def cluster_color(color_map, cluster):
    return (color_map or {}).get(int(cluster), PALETTE[int(cluster) % len(PALETTE)])


def evaluate(X, labels, name):
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    print(f"  [{name}] Silhouette={sil:.3f}  Davies-Bouldin={db:.3f}  Calinski-Harabasz={ch:.1f}")
    return {"Silhouette ↑": sil, "Davies-Bouldin ↓": db, "Calinski-Harabasz ↑": ch}


# ─── 7. PLOTS ──────────────────────────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "figure.facecolor": "#0D1117",
        "axes.facecolor":   "#161B22",
        "axes.edgecolor":   "#30363D",
        "axes.labelcolor":  "#C9D1D9",
        "xtick.color":      "#8B949E",
        "ytick.color":      "#8B949E",
        "text.color":       "#C9D1D9",
        "grid.color":       "#21262D",
        "grid.linestyle":   "--",
        "font.family":      "monospace",
        "axes.titlesize":   12,
        "axes.labelsize":   10,
    })

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    plt.close(fig)


# ── Fig 1: Elbow + Silhouette for K selection ──────────────────────────────────
def plot_k_selection(X):
    ks   = range(2, 10)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lbl))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#0D1117")
    fig.suptitle("K-Means — Επιλογή Βέλτιστου K", fontsize=14, color="#C9D1D9", y=1.02)

    axes[0].plot(ks, inertias, "o-", color="#E63946", lw=2, ms=7)
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
    axes[0].grid(True)

    axes[1].plot(ks, sils, "o-", color="#2A9D8F", lw=2, ms=7)
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Score")
    axes[1].axvline(N_CLUSTERS, color="#E9C46A", ls="--", lw=1.5, label=f"K={N_CLUSTERS}")
    axes[1].legend(facecolor="#161B22", edgecolor="#30363D")
    axes[1].grid(True)

    fig.tight_layout()
    save(fig, "fig1_k_selection.png")


# ── Fig 2: PCA 2D scatter – both algorithms side by side ──────────────────────
def plot_pca_scatter(X, km_labels, hc_labels, km_color_map=None, hc_color_map=None):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xp  = pca.fit_transform(X)
    ev  = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0D1117")
    fig.suptitle("PCA Προβολή Clusters (2 κύριες συνιστώσες)", fontsize=14, color="#C9D1D9")

    for ax, labels, title, color_map in zip(axes, [km_labels, hc_labels], ["K-Means", "Hierarchical (Ward)"], [km_color_map or {}, hc_color_map or {}]):
        for c in range(N_CLUSTERS):
            mask = labels == c
            ax.scatter(Xp[mask, 0], Xp[mask, 1],
                       color=cluster_color(color_map, c), alpha=0.8, s=60, label=f"C{c}", edgecolors="none")
        ax.set_title(title); ax.grid(True)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
        ax.legend(facecolor="#161B22", edgecolor="#30363D",
                  markerscale=1.2, fontsize=8)

    fig.tight_layout()
    save(fig, "fig2_pca_scatter.png")


# ── Fig 3: Dendrogram (Ward) ────────────────────────────────────────────────────
def plot_dendrogram(X):
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(13, 5), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-N_CLUSTERS, 2],
               above_threshold_color="#8B949E",
               link_color_func=lambda k: PALETTE[k % len(PALETTE)])
    ax.set_title("Dendrogram — Ward Linkage", fontsize=13)
    ax.set_xlabel("Super-cells"); ax.set_ylabel("Distance")
    ax.axhline(Z[-N_CLUSTERS, 2], color="#E63946", ls="--", lw=1.5,
               label=f"Cut → {N_CLUSTERS} clusters")
    ax.legend(facecolor="#161B22", edgecolor="#30363D")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig3_dendrogram.png")


# ── Fig 4: Cluster metrics bar chart ───────────────────────────────────────────
def plot_metrics(metrics_km, metrics_hc):
    """Plot clustering evaluation metrics in two separate charts.

    Silhouette and Davies-Bouldin have small values, while Calinski-Harabasz
    is usually much larger. Splitting them avoids hiding the small-scale metrics.
    """
    # --- Fig 4a: small-scale metrics ---
    small_metrics = ["Silhouette ↑", "Davies-Bouldin ↓"]
    km_small = [metrics_km[m] for m in small_metrics]
    hc_small = [metrics_hc[m] for m in small_metrics]

    x = np.arange(len(small_metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0D1117")
    ax.bar(x - w/2, km_small, w, label="K-Means", color="#E63946", alpha=0.85)
    ax.bar(x + w/2, hc_small, w, label="Hierarchical", color="#457B9D", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(small_metrics, fontsize=9)
    ax.set_title("Μετρικές Αξιολόγησης — Silhouette & Davies-Bouldin", fontsize=13)
    ax.legend(facecolor="#161B22", edgecolor="#30363D")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig4a_metrics_silhouette_davies.png")

    # --- Fig 4b: large-scale metric ---
    large_metrics = ["Calinski-Harabasz ↑"]
    km_large = [metrics_km[m] for m in large_metrics]
    hc_large = [metrics_hc[m] for m in large_metrics]

    x = np.arange(len(large_metrics))

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0D1117")
    ax.bar(x - w/2, km_large, w, label="K-Means", color="#E63946", alpha=0.85)
    ax.bar(x + w/2, hc_large, w, label="Hierarchical", color="#457B9D", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(large_metrics, fontsize=9)
    ax.set_title("Μετρική Αξιολόγησης — Calinski-Harabasz", fontsize=13)
    ax.legend(facecolor="#161B22", edgecolor="#30363D")
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig4b_metrics_calinski_harabasz.png")


# ── Fig 5: Cluster profiles (radar / bar per feature) ─────────────────────────
def plot_cluster_profiles(feat, km_labels, color_map=None):
    feat2 = feat.copy()
    feat2["cluster"] = km_labels

    profile_cols = ["mean_total", "mean_internet", "mean_call_in", "mean_sms_in",
                    "peak_rush_mean", "peak_night_mean", "peak_weekend_mean",
                    "rush_ratio", "night_ratio", "weekend_ratio", "cv_total"]
    profile = feat2.groupby("cluster")[profile_cols].mean()

    # Normalize 0-1 for readability
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0D1117")
    bottom = np.zeros(len(profile_cols))
    x = np.arange(len(profile_cols))
    width = 0.15
    for i, (idx, row) in enumerate(profile_norm.iterrows()):
        ax.bar(x + i*width, row.values, width, label=f"Cluster {idx}",
               color=cluster_color(color_map, idx), alpha=0.85)
    ax.set_xticks(x + width * (N_CLUSTERS-1)/2)
    ax.set_xticklabels([c.replace("mean_","").replace("_"," ") for c in profile_cols],
                       rotation=25, ha="right", fontsize=9)
    ax.set_title("Προφίλ Clusters — K-Means (normalized)", fontsize=13)
    ax.set_ylabel("Normalized value")
    ax.legend(facecolor="#161B22", edgecolor="#30363D", fontsize=8)
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig5_cluster_profiles.png")


# ── Fig 6: Temporal activity per cluster (hourly heatmap) ─────────────────────
def plot_temporal_heatmap(df_time, feat, km_labels):
    feat2 = feat.copy()
    feat2["cluster"] = km_labels
    df_time = df_time.merge(feat2[["super_cell","cluster"]], on="super_cell", how="left")

    hourly = df_time.groupby(["cluster","hour"])["total_activity"].mean().reset_index()

    pivot = hourly.pivot(index="cluster", columns="hour", values="total_activity").fillna(0)

    fig, ax = plt.subplots(figsize=(14, 4), facecolor="#0D1117")
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(N_CLUSTERS))
    ax.set_yticklabels([f"Cluster {i}" for i in pivot.index])
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24), fontsize=8)
    ax.set_xlabel("Ώρα Ημέρας"); ax.set_title("Μέση Δραστηριότητα ανά Ώρα & Cluster", fontsize=13)
    plt.colorbar(im, ax=ax, label="Avg Activity")
    fig.tight_layout()
    save(fig, "fig6_temporal_heatmap.png")


# ── Fig 7: Weekly pattern per cluster ─────────────────────────────────────────
def plot_weekly_pattern(df_time, feat, km_labels, color_map=None):
    feat2 = feat.copy()
    feat2["cluster"] = km_labels
    df_time = df_time.merge(feat2[["super_cell","cluster"]], on="super_cell", how="left")

    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    weekly = df_time.groupby(["cluster","day_of_week"])["total_activity"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0D1117")
    for c in range(N_CLUSTERS):
        sub = weekly[weekly["cluster"]==c].sort_values("day_of_week")
        ax.plot(sub["day_of_week"], sub["total_activity"],
                "o-", color=cluster_color(color_map, c), lw=2, ms=6, label=f"Cluster {c}")
    ax.set_xticks(range(7)); ax.set_xticklabels(day_names)
    ax.set_title("Εβδομαδιαίο Μοτίβο Δραστηριότητας ανά Cluster", fontsize=13)
    ax.set_ylabel("Μέση Δραστηριότητα")
    ax.legend(facecolor="#161B22", edgecolor="#30363D", fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    save(fig, "fig7_weekly_pattern.png")


# ── Fig 8: Internet vs Voice ratio per cluster ────────────────────────────────
def plot_activity_composition(feat, km_labels):
    feat2 = feat.copy()
    feat2["cluster"] = km_labels
    comp = feat2.groupby("cluster")[["mean_internet","mean_call_in","mean_call_out",
                                      "mean_sms_in","mean_sms_out"]].mean()
    comp_norm = comp.div(comp.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0D1117")
    colors_stack = ["#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261"]
    bottom = np.zeros(N_CLUSTERS)
    for i, col in enumerate(comp_norm.columns):
        ax.bar(comp_norm.index, comp_norm[col], bottom=bottom,
               label=col.replace("mean_",""), color=colors_stack[i], alpha=0.9)
        bottom += comp_norm[col].values

    ax.set_xticks(range(N_CLUSTERS))
    ax.set_xticklabels([f"C{i}" for i in range(N_CLUSTERS)])
    ax.set_ylabel("% Σύνθεσης")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Σύνθεση Δραστηριότητας ανά Cluster (%)", fontsize=13)
    ax.legend(facecolor="#161B22", edgecolor="#30363D", fontsize=9,
              loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig8_activity_composition.png")


# ── Fig 9: Distribution of cluster sizes ──────────────────────────────────────
def plot_cluster_sizes(km_labels, hc_labels, km_color_map=None, hc_color_map=None):
    from collections import Counter
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#0D1117")
    for ax, labels, title, color_map in zip(axes, [km_labels, hc_labels], ["K-Means","Hierarchical"], [km_color_map or {}, hc_color_map or {}]):
        cnt = Counter(labels)
        ks  = sorted(cnt.keys())
        vs  = [cnt[k] for k in ks]
        bars = ax.bar([f"C{k}" for k in ks], vs,
                      color=[cluster_color(color_map, k) for k in ks], alpha=0.9)
        for bar, v in zip(bars, vs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(f"Μέγεθος Clusters — {title}", fontsize=11)
        ax.set_ylabel("# Super-cells"); ax.grid(True, axis="y")
    fig.tight_layout()
    save(fig, "fig9_cluster_sizes.png")


# ── Fig 10: Milan 100x100 spatial grid map with clusters ─────────────────────
def _grid_id_to_xy(grid_id: int, side: int = 100):
    """Convert 1-based Milano CDR grid_id to 0-based x/y coordinates on a 100x100 grid."""
    gid0 = int(grid_id) - 1
    return gid0 % side, gid0 // side


def _find_milan_background_image():
    """
    Find the Milan map image in ./Images/City_of_Milan.* relative to this script.
    Supported names: City_of_Milan, City_of_Milan.png, .jpg, .jpeg, .webp.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    images_dir = os.path.join(script_dir, "Images")
    candidates = [
        os.path.join(images_dir, "City_of_Milan"),
        os.path.join(images_dir, "City_of_Milan.png"),
        os.path.join(images_dir, "City_of_Milan.jpg"),
        os.path.join(images_dir, "City_of_Milan.jpeg"),
        os.path.join(images_dir, "City_of_Milan.webp"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def plot_milan_cluster_map(feat, km_labels, grid_map=None, side: int = 100, cluster_alpha: float = 0.45, color_map=None):
    """
    Plot clusters on the original aligned 100x100 Milano grid over a Milan map image.

    Put the background map here, next to the script:
        Images/City_of_Milan.png

    Clustering is still trained on super-cells, but for visualization each original
    grid cell is drawn at its real 100x100 grid position and colored according to
    the cluster of the super-cell it belongs to.
    """
    feat2 = feat[["super_cell"]].copy()
    feat2["cluster"] = km_labels
    cluster_by_supercell = dict(zip(feat2["super_cell"], feat2["cluster"]))

    # grid_img[y, x] contains the cluster id for the original 100x100 Milano cell.
    grid_img = np.full((side, side), np.nan, dtype=float)

    if grid_map:
        for grid_id, super_cell in grid_map.items():
            cluster = cluster_by_supercell.get(super_cell)
            if cluster is None:
                continue
            x, y = _grid_id_to_xy(grid_id, side=side)
            if 0 <= x < side and 0 <= y < side:
                grid_img[y, x] = cluster
    else:
        # Fallback for legacy path: draw available super-cells in a square grid.
        for super_cell, cluster in cluster_by_supercell.items():
            x = int(super_cell) % side
            y = int(super_cell) // side
            if 0 <= x < side and 0 <= y < side:
                grid_img[y, x] = cluster

    cmap = ListedColormap([cluster_color(color_map, c) for c in range(N_CLUSTERS)])
    masked_grid = np.ma.masked_invalid(grid_img)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0D1117")
    ax.set_facecolor("#10151C")

    # 1) Draw Milan map behind the cluster layer.
    map_path = _find_milan_background_image()
    if map_path is not None:
        try:
            bg = plt.imread(map_path)
            ax.imshow(
                bg,
                origin="upper",
                interpolation="bilinear",
                extent=[0, side, side, 0],
                alpha=1.0,
                zorder=0,
            )
            print(f"  Milan background map loaded → {map_path}")
        except Exception as exc:
            print(f"  Warning: could not read Milan background image ({map_path}): {exc}")
    else:
        print("  Warning: Images/City_of_Milan image not found. Plotting clusters without background map.")

    # 2) Draw the clusters as a transparent overlay so the map remains visible.
    ax.imshow(
        masked_grid,
        cmap=cmap,
        vmin=-0.5,
        vmax=N_CLUSTERS - 0.5,
        origin="upper",
        interpolation="nearest",
        extent=[0, side, side, 0],
        alpha=cluster_alpha,
        zorder=1,
    )

    # 3) Draw a thin aligned 100x100 grid on top.
    ax.set_xticks(np.arange(0, side + 1, 10))
    ax.set_yticks(np.arange(0, side + 1, 10))
    ax.set_xticks(np.arange(0, side + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, side + 1, 1), minor=True)
    ax.grid(which="major", color="#1F2933", linewidth=0.45, alpha=0.55, zorder=2)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.08, alpha=0.35, zorder=2)

    ax.set_title("Χάρτης Milano 100×100 Grid — K-Means Clusters", fontsize=13)
    ax.set_xlabel("Grid X / West → East")
    ax.set_ylabel("Grid Y / North → South")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)

    legend_items = [
        Patch(facecolor=cluster_color(color_map, c), edgecolor="#0D1117",
              alpha=cluster_alpha, label=f"Cluster {c}")
        for c in range(N_CLUSTERS)
    ]
    ax.legend(handles=legend_items, facecolor="#F8F9FA", edgecolor="#30363D",
              labelcolor="#111111", fontsize=8, loc="upper right", bbox_to_anchor=(1.24, 1.0))

    ax.text(
        0.01, -0.08,
        "Κάθε μικρό τετράγωνο είναι original grid cell του Milano 100×100 grid. "
        f"Τα cluster χρώματα είναι διαφανή (alpha={cluster_alpha}) πάνω στον χάρτη.",
        transform=ax.transAxes,
        fontsize=8,
        color="#8B949E",
        ha="left",
        va="top"
    )

    fig.tight_layout()
    save(fig, "fig10_milan_100x100_cluster_map.png")


# ── Fig 10B: Real Milano GeoJSON grid map with clusters ──────────────────────
def _find_milano_grid_geojson():
    """
    Find the official Milano Grid GeoJSON.

    Recommended location next to the script:
        data/milano-grid.geojson

    Also supports:
        milano-grid.geojson
        Data/milano-grid.geojson
        Images/milano-grid.geojson
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [
        os.path.join(script_dir, "data", "milano-grid.geojson"),
        os.path.join(script_dir, "Data", "milano-grid.geojson"),
        os.path.join(script_dir, "milano-grid.geojson"),
        os.path.join(script_dir, "Images", "milano-grid.geojson"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _iter_geojson_polygons(feature):
    """Yield exterior rings from Polygon or MultiPolygon GeoJSON geometries."""
    geom = feature.get("geometry", {})
    gtype = geom.get("type")
    coords = geom.get("coordinates", [])

    if gtype == "Polygon":
        if coords:
            yield coords[0]
    elif gtype == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0]


def plot_milan_geojson_cluster_map(feat, km_labels, grid_map=None, cluster_alpha: float = 0.48, color_map=None):
    """
    Plot K-Means clusters on the official Milano Grid GeoJSON polygons.

    This is more geographically correct than the simple 100x100 matrix plot because
    each cell is drawn from its real polygon geometry from milano-grid.geojson.

    Required file:
        data/milano-grid.geojson

    Optional background image:
        Images/City_of_Milan.png
    """
    geojson_path = _find_milano_grid_geojson()
    if geojson_path is None:
        print("  Warning: milano-grid.geojson not found. Skipping GeoJSON cluster map.")
        return

    feat2 = feat[["super_cell"]].copy()
    feat2["cluster"] = km_labels
    cluster_by_supercell = dict(zip(feat2["super_cell"], feat2["cluster"]))

    cluster_by_grid = {}
    if grid_map:
        for grid_id, super_cell in grid_map.items():
            cluster = cluster_by_supercell.get(super_cell)
            if cluster is not None:
                cluster_by_grid[int(grid_id)] = int(cluster)
    else:
        # Fallback: assumes grid_id maps directly to super_cell. Usually not used.
        for super_cell, cluster in cluster_by_supercell.items():
            cluster_by_grid[int(super_cell) + 1] = int(cluster)

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    polygons = []
    facecolors = []
    all_x, all_y = [], []

    for feature in gj.get("features", []):
        props = feature.get("properties", {})
        cell_id = props.get("cellId", props.get("cellid", props.get("grid_id", feature.get("id"))))
        try:
            cell_id = int(cell_id)
        except Exception:
            continue

        cluster = cluster_by_grid.get(cell_id)
        for ring in _iter_geojson_polygons(feature):
            poly = [(float(x), float(y)) for x, y in ring]
            if len(poly) < 3:
                continue
            polygons.append(poly)
            all_x.extend([p[0] for p in poly])
            all_y.extend([p[1] for p in poly])
            if cluster is None:
                facecolors.append((0, 0, 0, 0))
            else:
                facecolors.append(to_rgba(cluster_color(color_map, cluster), alpha=cluster_alpha))

    if not polygons:
        print("  Warning: GeoJSON loaded but no valid polygons were found.")
        return

    minx, maxx = min(all_x), max(all_x)
    miny, maxy = min(all_y), max(all_y)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#0D1117")
    ax.set_facecolor("#F5F5F2")

    # Optional visual background map. It is stretched to the GeoJSON bounds.
    map_path = _find_milan_background_image()
    if map_path is not None:
        try:
            bg = plt.imread(map_path)
            ax.imshow(
                bg,
                extent=[minx, maxx, miny, maxy],
                origin="upper",
                interpolation="bilinear",
                alpha=1.0,
                zorder=0,
            )
            print(f"  Milan background map loaded → {map_path}")
        except Exception as exc:
            print(f"  Warning: could not read Milan background image ({map_path}): {exc}")

    collection = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=(1, 1, 1, 0.28),
        linewidths=0.08,
        closed=True,
        zorder=1,
    )
    ax.add_collection(collection)

    pad_x = (maxx - minx) * 0.01
    pad_y = (maxy - miny) * 0.01
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Milano Grid GeoJSON — K-Means Clusters", fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_items = [
        Patch(facecolor=cluster_color(color_map, c), edgecolor="#30363D",
              alpha=cluster_alpha, label=f"Cluster {c}")
        for c in range(N_CLUSTERS)
    ]
    ax.legend(handles=legend_items, facecolor="#F8F9FA", edgecolor="#30363D",
              labelcolor="#111111", fontsize=8, loc="upper right", bbox_to_anchor=(1.22, 1.0))

    ax.text(
        0.01, -0.08,
        f"Official Milano Grid GeoJSON polygons colored by cluster. Transparency alpha={cluster_alpha}.",
        transform=ax.transAxes,
        fontsize=8,
        color="#8B949E",
        ha="left",
        va="top",
    )

    fig.tight_layout()
    save(fig, "fig10b_milan_geojson_cluster_map.png")


# ── Fig 11: Summary dashboard ─────────────────────────────────────────────────
def plot_dashboard(feat, km_labels, metrics_km, metrics_hc, color_map=None):
    feat2 = feat.copy()
    feat2["cluster"] = km_labels

    fig = plt.figure(figsize=(16, 10), facecolor="#0D1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Top-left: mean_total per cluster
    ax1 = fig.add_subplot(gs[0, 0])
    c_mean = feat2.groupby("cluster")["mean_total"].mean()
    ax1.bar([f"C{i}" for i in c_mean.index], c_mean.values,
            color=[cluster_color(color_map, i) for i in c_mean.index], alpha=0.9)
    ax1.set_title("Μέση Συνολική Δραστηριότητα"); ax1.grid(True, axis="y")

    # Top-middle: internet share
    ax2 = fig.add_subplot(gs[0, 1])
    c_int = feat2.groupby("cluster")["internet_share"].mean() * 100
    ax2.bar([f"C{i}" for i in c_int.index], c_int.values,
            color=[cluster_color(color_map, i) for i in c_int.index], alpha=0.9)
    ax2.set_title("Internet Share (%)"); ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.grid(True, axis="y")

    # Top-right: CV (variability)
    ax3 = fig.add_subplot(gs[0, 2])
    c_cv = feat2.groupby("cluster")["cv_total"].mean()
    ax3.bar([f"C{i}" for i in c_cv.index], c_cv.values,
            color=[cluster_color(color_map, i) for i in c_cv.index], alpha=0.9)
    ax3.set_title("Μεταβλητότητα (CV)"); ax3.grid(True, axis="y")

    # Bottom-left: activity-weighted rush vs night percentages
    ax4 = fig.add_subplot(gs[1, 0])
    c_rush  = feat2.groupby("cluster")["rush_ratio"].mean() * 100
    c_night = feat2.groupby("cluster")["night_ratio"].mean() * 100
    x = np.arange(N_CLUSTERS); w = 0.35
    ax4.bar(x-w/2, c_rush.values,  w, label="Rush activity", color="#E63946", alpha=0.85)
    ax4.bar(x+w/2, c_night.values, w, label="Night activity", color="#457B9D", alpha=0.85)
    ax4.set_xticks(x); ax4.set_xticklabels([f"C{i}" for i in range(N_CLUSTERS)])
    ax4.set_title("Rush vs Night Activity (%)")
    ax4.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax4.legend(facecolor="#161B22", edgecolor="#30363D", fontsize=8)
    ax4.grid(True, axis="y")

    # Bottom-middle: metrics comparison table
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis("off")
    metric_names = ["Silhouette ↑","Davies-Bouldin ↓","Calinski-Harabasz ↑"]
    table_data = [[m, f"{metrics_km[m]:.3f}", f"{metrics_hc[m]:.3f}"] for m in metric_names]
    tbl = ax5.table(cellText=table_data,
                    colLabels=["Metric","K-Means","Hierarchical"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1C2128" if row == 0 else "#161B22")
        cell.set_edgecolor("#30363D")
        cell.set_text_props(color="#C9D1D9")
    ax5.set_title("Μετρικές Αξιολόγησης", y=0.92)

    # Bottom-right: weekend ratio
    ax6 = fig.add_subplot(gs[1, 2])
    c_we = feat2.groupby("cluster")["weekend_ratio"].mean() * 100
    ax6.bar([f"C{i}" for i in c_we.index], c_we.values,
            color=[cluster_color(color_map, i) for i in c_we.index], alpha=0.9)
    ax6.set_title("Weekend Activity (%)"); ax6.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax6.grid(True, axis="y")

    fig.suptitle("Milan Telecom — Dashboard Clustering (K-Means)", fontsize=15, color="#C9D1D9", y=1.01)
    save(fig, "fig10_dashboard.png")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    set_style()

    print("\n=== 1-3. Φόρτωση, aggregation ανά 10 δευτερόλεπτα και ομαδοποίηση grids (×6) ===")
    grouped = load_aggregate_and_group(DATA_DIR, GRID_GROUP)

    print("\n=== 4. Feature engineering ===")
    feat, df_time = build_features(grouped)

    print("\n=== 5. Προεπεξεργασία ===")
    X = preprocess(feat)

    print("\n=== 6. Επιλογή K (Elbow + Silhouette) ===")
    plot_k_selection(X)

    print("\n=== 7. Clustering ===")
    km_labels, km_model = run_kmeans(X, N_CLUSTERS)
    hc_labels            = run_hierarchical(X, N_CLUSTERS)

    # Rename cluster IDs consistently by size, for clearer K-Means vs Hierarchical comparison.
    # Example: C0 is the largest cluster in both algorithms, C1 the second largest, etc.
    # This is only a label permutation; it does not alter the clustering assignments or metrics.
    km_labels, km_label_map = relabel_clusters_by_size(km_labels, "K-Means")
    hc_labels, hc_label_map = relabel_clusters_by_size(hc_labels, "Hierarchical")

    # Labels stay size-aligned; colors show average activity intensity.
    km_color_map = make_activity_color_map(feat, km_labels, "K-Means")
    hc_color_map = make_activity_color_map(feat, hc_labels, "Hierarchical")

    print("\n=== 8. Αξιολόγηση ===")
    metrics_km = evaluate(X, km_labels, "K-Means")
    metrics_hc = evaluate(X, hc_labels, "Hierarchical")

    print("\n=== 9. Οπτικοποίηση ===")
    plot_pca_scatter(X, km_labels, hc_labels, km_color_map, hc_color_map)
    plot_dendrogram(X)
    plot_metrics(metrics_km, metrics_hc)
    plot_cluster_profiles(feat, km_labels, km_color_map)
    plot_temporal_heatmap(df_time, feat, km_labels)
    plot_weekly_pattern(df_time, feat, km_labels, km_color_map)
    plot_activity_composition(feat, km_labels)
    plot_cluster_sizes(km_labels, hc_labels, km_color_map, hc_color_map)
    plot_milan_cluster_map(feat, km_labels, grouped.attrs.get("grid_map"), color_map=km_color_map)
    plot_milan_geojson_cluster_map(feat, km_labels, grouped.attrs.get("grid_map"), color_map=km_color_map)
    plot_dashboard(feat, km_labels, metrics_km, metrics_hc, km_color_map)

    print("\n=== Χαρακτηρισμός Clusters (K-Means) ===")
    feat2 = feat.copy()
    feat2["cluster"] = km_labels
    profile = feat2.groupby("cluster")[[
        "mean_total", "internet_share", "peak_rush_mean", "peak_night_mean",
        "peak_weekend_mean", "rush_ratio", "night_ratio", "weekend_ratio", "cv_total"
    ]].mean()

    profile_print = profile.copy()
    for col in ["internet_share", "rush_ratio", "night_ratio", "weekend_ratio"]:
        profile_print[col] = profile_print[col] * 100
    print(profile_print.round(3).to_string())

    print("\n✅ Ολοκληρώθηκε! Αρχεία PNG, μαζί με τον χάρτη clusters του Milano grid, αποθηκεύτηκαν στον τρέχοντα φάκελο.\n")


if __name__ == "__main__":
    main()
