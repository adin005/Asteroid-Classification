"""
recreate_asteroid_gnn_paper.py
==============================================================================
Re-implementation of the pipeline in:

  "Explainable Deep-Learning Based Potentially Hazardous Asteroids
   Classification Using Graph Neural Networks"  (Boukar, arXiv 2504.18605, 2025)

What it runs (same models as the paper's Table III):
    GNN (SMOTE)      3 x GCN layers -> 1 GAT layer -> output, trained on SMOTE data
    GNN (Original)   same GNN trained on the original imbalanced data
    MLP              plain neural network (no graph)
    GAT              graph attention network baseline
    iForest          Isolation Forest (anomaly detection)
  Extra (NOT in the paper, marked with *):
    Random Forest*, XGBoost*   -> also exported as .pkl files for your web app

Dataset: put the Kaggle "Asteroid Dataset" CSV next to this script and name it
         dataset.csv     (columns used: pha, e, a, q, i, tp, H, diameter, albedo, moid)

------------------------------------------------------------------------------
ONE-TIME SETUP (run these in a terminal / PowerShell, one line at a time)
------------------------------------------------------------------------------
    python -m venv venv
    venv\\Scripts\\Activate.ps1            (Windows PowerShell)
    source venv/bin/activate              (macOS / Linux)

    pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn tqdm rich joblib xgboost
    pip install torch
    pip install torch_geometric

  Notes:
  * Only if you have an NVIDIA GPU: get the matching torch command from
    https://pytorch.org/get-started/locally/ instead of `pip install torch`.
  * If the deep-learning install fails, set ENABLE_DEEP_MODELS = False below;
    the rest of the script (iForest, Random Forest, XGBoost, plots) still runs.

RUN:
    python recreate_asteroid_gnn_paper.py

Everything is saved in the  results/  folder (figures, metrics CSV, models).
Expected run time on a laptop CPU with the default 200,000-row subset: roughly 15-45 minutes
(most of it is the three graph models). Set SAMPLE_SIZE = 50_000 for a quick first test run.
==============================================================================
"""

# ==============================================================================
# CONFIG  -- the only section you normally need to touch
# ==============================================================================
DATA_PATH = "dataset.csv"        # the Kaggle asteroid CSV, renamed to dataset.csv
OUTPUT_DIR = "results"           # figures / metrics / models are saved here

# The paper used the full ~938k rows on a 300 GB RAM machine. A laptop cannot
# hold that graph in memory, so we use a stratified random subset by default.
#   200_000 -> fine for 8-16 GB RAM  |  500_000 -> 16-32 GB RAM  |  None -> all rows (32 GB+)
SAMPLE_SIZE = 200_000

RANDOM_STATE = 42
TEST_SIZE = 0.20                 # paper: 80 / 20 split, stratified
VAL_FRACTION = 0.10              # part of the TRAIN split used for early stopping
K_NEIGHBORS = 5                  # paper: k = 5 nearest neighbours per asteroid
SMOTE_RATIO = 1.0                # 1.0 = balance classes fully (lower it, e.g. 0.3, to save memory)

# --- GNN training (paper: 100 epochs, Adam, early stopping patience 10) ---
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
HIDDEN_DIM = 128                 # paper: 128-dim after the first GCN layer
GAT_OUT_DIM = 64                 # paper: 64-dim output of the attention layer
GAT_HEADS = 4
DROPOUT = 0.3
# Paper: "combination of binary cross-entropy and focal loss" (weights not published)
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
LOSS_MIX = 0.5                   # 0.5 * BCE + 0.5 * Focal

ENABLE_DEEP_MODELS = True        # False = skip GNN / GAT / MLP (needs torch otherwise)
BASELINES_USE_SMOTE = False      # paper does not say; False matches its low baseline recall
IFOREST_CONTAMINATION = 0.03     # back-calculated from the paper's iForest precision/recall
RUN_TREE_REFERENCE = True        # extra Random Forest + XGBoost (also saved as .pkl)

# The paper's feature list includes `moid` and `H`, which are the two quantities that
# DEFINE a "potentially hazardous asteroid" (MOID <= 0.05 AU and H <= 22). That is label
# leakage. False = faithful to the paper. True = drop them for an honest, harder task.
REMOVE_LABEL_DEFINING_FEATURES = False

SHOW_PLOTS = True                # False = only save PNGs, do not open windows
N_JOBS = -1                      # CPU threads for kNN / forests (set 1 if you see threading errors)

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import joblib

import matplotlib
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from tqdm.auto import tqdm
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"blue": "#2b5c8a", "orange": "#e07a1f", "teal": "#2a9d8f",
           "red": "#c8443a", "grey": "#8d99a6", "purple": "#7b5ea7"}
SPACE_CMAP = LinearSegmentedColormap.from_list("space", ["#f4f7fb", "#8fb3d9", "#1f3a5f"])

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Numbers reported in the paper (Table III, hazardous class): precision, recall, F1
PAPER = OrderedDict([
    ("GNN (SMOTE)",    (0.24, 0.78, 0.37)),
    ("GNN (Original)", (0.28, 0.37, 0.32)),
    ("MLP",            (0.20, 0.41, 0.27)),
    ("GAT",            (0.11, 0.17, 0.14)),
    ("iForest",        (0.04, 0.57, 0.08)),
])

# The 9 features the paper names (orbital + physical). Its full list is not printed in the text.
FEATURES = ["e", "a", "q", "i", "tp", "H", "diameter", "albedo", "moid"]
if REMOVE_LABEL_DEFINING_FEATURES:
    FEATURES = [f for f in FEATURES if f not in ("moid", "H")]
LABEL = "pha"

T_START = time.time()


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


# ==============================================================================
# Small helpers: metrics + result registry
# ==============================================================================
RESULTS = OrderedDict()   # model name -> dict of metrics (+ scores for plotting)


def add_result(name, y_true, score, pred=None, threshold=0.5, note=""):
    """Store precision/recall/F1/AUC/confusion counts for the HAZARDOUS class (label 1)."""
    if pred is None:
        pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    RESULTS[name] = dict(
        precision=precision_score(y_true, pred, zero_division=0),
        recall=recall_score(y_true, pred, zero_division=0),
        f1=f1_score(y_true, pred, zero_division=0),
        accuracy=accuracy_score(y_true, pred),
        roc_auc=roc_auc_score(y_true, score),
        pr_auc=average_precision_score(y_true, score),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        threshold=threshold, note=note, score=np.asarray(score), y_true=np.asarray(y_true),
    )


def best_f1_threshold(y_val, p_val):
    """Threshold that maximises F1 on the validation set (never looks at the test set)."""
    if y_val.sum() == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1[:-1]))])


def print_model_row(name):
    r = RESULTS[name]
    console.print(f"  [bold]{name}[/bold]  precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
                  f"F1={r['f1']:.3f}  ROC-AUC={r['roc_auc']:.3f}  PR-AUC={r['pr_auc']:.3f}")


# ==============================================================================
# Deep-learning pieces (only defined when torch + PyG are available)
# ==============================================================================
def build_graph_arrays(X_train, y_train, X_val, y_val, X_test, y_test, k):
    """
    Numpy-only graph construction (paper, Section III-B):
      * nodes  = asteroids (train + val + test, in that order)
      * edges  = k nearest neighbours in standardised feature space
      * edge weight = 1 / (1 + distance)          (used by the GCN layers)
      * edge attr   = feature difference of the two endpoints   (used by the GAT layer)
    """
    X_all = np.vstack([X_train, X_val, X_test]).astype(np.float32)
    y_all = np.concatenate([y_train, y_val, y_test]).astype(np.int64)
    n_tr, n_va, n_te = len(X_train), len(X_val), len(X_test)
    n = len(X_all)

    index = NearestNeighbors(n_neighbors=k + 1, n_jobs=N_JOBS).fit(X_all)
    dist, idx = index.kneighbors(X_all)                      # column 0 is normally the point itself

    is_self = idx == np.arange(n)[:, None]
    is_self[~is_self.any(axis=1), -1] = True                 # duplicate points: drop last column instead
    keep = ~is_self
    idx = idx[keep].reshape(n, k)
    dist = dist[keep].reshape(n, k)

    src = idx.reshape(-1)                                    # neighbour  -> node
    dst = np.repeat(np.arange(n), k)
    edge_index = np.stack([src, dst]).astype(np.int64)
    edge_weight = (1.0 / (1.0 + dist)).reshape(-1).astype(np.float32)
    edge_attr = (X_all[src] - X_all[dst]).astype(np.float32)

    train_mask = np.zeros(n, dtype=bool); train_mask[:n_tr] = True
    val_mask = np.zeros(n, dtype=bool);   val_mask[n_tr:n_tr + n_va] = True
    test_mask = np.zeros(n, dtype=bool);  test_mask[n_tr + n_va:] = True
    return dict(x=X_all, y=y_all, edge_index=edge_index, edge_weight=edge_weight,
                edge_attr=edge_attr, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


if ENABLE_DEEP_MODELS:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GATConv, GCNConv
    except ImportError as exc:
        console.print(Panel(
            f"[bold red]Could not import torch / torch_geometric[/bold red]\n{exc}\n\n"
            "Install them with:\n    pip install torch\n    pip install torch_geometric\n\n"
            "Or set ENABLE_DEEP_MODELS = False at the top of this file to run without them.",
            title="Missing deep-learning libraries", border_style="red"))
        sys.exit(1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_STATE)

    class BCEFocalLoss(nn.Module):
        """LOSS_MIX * BCE + (1 - LOSS_MIX) * Focal loss (paper: 'BCE and Focal Loss')."""

        def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, mix=LOSS_MIX):
            super().__init__()
            self.alpha, self.gamma, self.mix = alpha, gamma, mix

        def forward(self, logits, targets):
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * (1 - p_t) ** self.gamma * bce
            return (self.mix * bce + (1 - self.mix) * focal).mean()

    class HazardGNN(nn.Module):
        """3 x GCNConv (128-d) -> GATConv attention layer (64-d) -> linear output (sigmoid at inference)."""

        def __init__(self, in_dim, edge_dim, hidden=HIDDEN_DIM, gat_out=GAT_OUT_DIM,
                     heads=GAT_HEADS, dropout=DROPOUT):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.conv3 = GCNConv(hidden, hidden)
            self.gat = GATConv(hidden, gat_out // heads, heads=heads, edge_dim=edge_dim, dropout=0.1)
            self.out = nn.Linear(gat_out, 1)
            self.dropout = dropout

        def forward(self, data):
            x, ei, ew, ea = data.x, data.edge_index, data.edge_weight, data.edge_attr
            x = F.dropout(F.relu(self.conv1(x, ei, ew)), p=self.dropout, training=self.training)
            x = F.dropout(F.relu(self.conv2(x, ei, ew)), p=self.dropout, training=self.training)
            x = F.relu(self.conv3(x, ei, ew))
            x = F.elu(self.gat(x, ei, edge_attr=ea))
            return self.out(x).squeeze(-1)                  # node-level logit (one per asteroid)

    class PlainGAT(nn.Module):
        """Baseline: two attention layers, no edge features."""

        def __init__(self, in_dim, dropout=DROPOUT):
            super().__init__()
            self.gat1 = GATConv(in_dim, 32, heads=4, dropout=0.1)
            self.gat2 = GATConv(128, 16, heads=4, dropout=0.1)
            self.out = nn.Linear(64, 1)
            self.dropout = dropout

        def forward(self, data):
            x = F.dropout(F.elu(self.gat1(data.x, data.edge_index)), p=self.dropout, training=self.training)
            x = F.elu(self.gat2(x, data.edge_index))
            return self.out(x).squeeze(-1)

    class MLP(nn.Module):
        """Baseline: ignores the graph completely, looks at each asteroid in isolation."""

        def __init__(self, in_dim, dropout=DROPOUT):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, data):
            return self.net(data.x).squeeze(-1)

    def build_graph_torch(*args):
        arrs = build_graph_arrays(*args, K_NEIGHBORS)
        data = Data(
            x=torch.from_numpy(arrs["x"]),
            edge_index=torch.from_numpy(arrs["edge_index"]),
            edge_weight=torch.from_numpy(arrs["edge_weight"]),
            edge_attr=torch.from_numpy(arrs["edge_attr"]),
            y=torch.from_numpy(arrs["y"]),
        )
        data.train_mask = torch.from_numpy(arrs["train_mask"])
        data.val_mask = torch.from_numpy(arrs["val_mask"])
        data.test_mask = torch.from_numpy(arrs["test_mask"])
        return data.to(DEVICE)

    def train_node_model(model, data, name):
        """Full-batch training with BCE+Focal loss, Adam, early stopping on validation PR-AUC."""
        model = model.to(DEVICE)
        criterion = BCEFocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        y = data.y.float()
        val_y = data.y[data.val_mask].cpu().numpy()
        history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}
        best_score, best_epoch, best_state, wait = -np.inf, 0, None, 0

        bar = tqdm(range(1, EPOCHS + 1), desc=f"{name:<15}", ncols=105)
        for epoch in bar:
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(data)[data.train_mask], y[data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(data)
                val_loss = criterion(logits[data.val_mask], y[data.val_mask]).item()
                val_prob = torch.sigmoid(logits[data.val_mask]).cpu().numpy()
            val_pr = average_precision_score(val_y, val_prob) if val_y.sum() > 0 else 0.0

            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            history["val_pr_auc"].append(val_pr)
            bar.set_postfix(loss=f"{loss.item():.4f}", val_loss=f"{val_loss:.4f}", val_PR=f"{val_pr:.3f}")

            if val_pr > best_score + 1e-4:
                best_score, best_epoch, wait = val_pr, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        bar.close()
        model.load_state_dict(best_state)
        console.print(f"    stopped after epoch {epoch}; best epoch = {best_epoch} "
                      f"(validation PR-AUC {best_score:.3f})")
        return model, history, best_epoch

    @torch.no_grad()
    def predict_nodes(model, data):
        model.eval()
        return torch.sigmoid(model(data)).cpu().numpy()

    @torch.no_grad()
    def permutation_importance(model, data, feature_names, n_repeats=3):
        """Drop in test PR-AUC when one feature is shuffled across the test asteroids."""
        test_idx = data.test_mask.nonzero(as_tuple=True)[0]
        test_np = test_idx.cpu().numpy()
        y_test = data.y[test_idx].cpu().numpy()
        base = average_precision_score(y_test, predict_nodes(model, data)[test_np])
        rng = np.random.default_rng(RANDOM_STATE)
        x_orig = data.x.clone()
        out = {}
        for j, fname in enumerate(feature_names):
            drops = []
            for _ in range(n_repeats):
                perm = torch.as_tensor(rng.permutation(len(test_np)), device=data.x.device)
                data.x = x_orig.clone()
                data.x[test_idx, j] = x_orig[test_idx[perm], j]
                p = predict_nodes(model, data)[test_np]
                drops.append(base - average_precision_score(y_test, p))
            out[fname] = float(np.mean(drops))
        data.x = x_orig
        return out


# ==============================================================================
# STEP 1 - LOAD DATA
# ==============================================================================
console.print(Panel.fit(
    "[bold cyan]Recreating: Explainable DL-Based Potentially Hazardous Asteroid Classification (GNN)[/bold cyan]\n"
    f"dataset file: {DATA_PATH}   |   sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'ALL rows'}   |   "
    f"deep models: {'ON' if ENABLE_DEEP_MODELS else 'OFF'}   |   "
    f"device: {DEVICE if ENABLE_DEEP_MODELS else 'cpu (sklearn only)'}\n"
    f"features ({len(FEATURES)}): {', '.join(FEATURES)}",
    border_style="cyan"))

console.rule("[bold cyan]STEP 1  Load the dataset")
if not os.path.exists(DATA_PATH):
    console.print(Panel(f"[bold red]Cannot find '{DATA_PATH}'[/bold red]\n"
                        "Put the Kaggle asteroid CSV in the same folder as this script and rename it dataset.csv\n"
                        f"Current folder: {os.getcwd()}", border_style="red"))
    sys.exit(1)

header = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
needed = FEATURES + [LABEL]
absent = [c for c in needed if c not in header]
if absent:
    console.print(Panel(f"[bold red]These columns are missing from the CSV:[/bold red] {absent}\n"
                        f"Columns found: {header}", border_style="red"))
    sys.exit(1)

df = pd.read_csv(DATA_PATH, usecols=needed, low_memory=False)
n_raw = len(df)
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["label"] = (df[LABEL].astype(str).str.strip().str.upper()
               .map({"Y": 1, "N": 0, "TRUE": 1, "FALSE": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}))
n_unlabeled = int(df["label"].isna().sum())
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)
n_valid = len(df)

t = Table(title="Dataset summary (compare with the paper)", box=box.SIMPLE_HEAVY, header_style="bold")
t.add_column("Item"); t.add_column("This run", justify="right"); t.add_column("Paper", justify="right")
t.add_row("Rows in CSV", f"{n_raw:,}", "958,524")
t.add_row("Rows with a known pha label", f"{n_valid:,}", "about 938,600 (implied by test size)")
t.add_row("Hazardous asteroids", f"{int(df['label'].sum()):,}", "about 2,070")
t.add_row("Hazardous rate", f"{df['label'].mean() * 100:.3f}%", "0.22%")
console.print(t)
console.print(f"  (dropped {n_unlabeled:,} rows whose pha label is empty)")

mt = Table(title="Missing values per feature", box=box.SIMPLE, header_style="bold")
mt.add_column("Feature"); mt.add_column("Missing", justify="right")
for col in FEATURES:
    mt.add_row(col, f"{df[col].isna().mean() * 100:.1f}%")
console.print(mt)
console.print("  Paper: diameter and albedo are missing for over 85% of asteroids -> filled with the median.")

if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
    df, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df["label"], random_state=RANDOM_STATE)
    console.print(f"  Using a stratified random subset of {len(df):,} rows "
                  f"({int(df['label'].sum())} hazardous, same 0.22%-style rate).")


def fig_feature_distributions():
    n = len(FEATURES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, FEATURES):
        s = df[f].dropna()
        if len(s):
            lo, hi = s.quantile([0.01, 0.99]).values
            s = s[(s >= lo) & (s <= hi)]
        ax.hist(s, bins=60, color=PALETTE["blue"], alpha=0.85)
        ax.set_title(f"{f}   ({df[f].isna().mean() * 100:.0f}% missing)", fontsize=10)
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Feature distributions (1st-99th percentile, before scaling)  |  paper Fig. 2",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig1_feature_distributions.png")


fig_feature_distributions()

# ==============================================================================
# STEP 2 - SPLIT, IMPUTE, SCALE, SMOTE
# ==============================================================================
console.rule("[bold cyan]STEP 2  Split, impute (median), scale, SMOTE")
X = df[FEATURES].values.astype(np.float64)
y = df["label"].values.astype(int)

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(X_tr_raw, y_tr, test_size=VAL_FRACTION, stratify=y_tr,
                                                  random_state=RANDOM_STATE)

imputer = SimpleImputer(strategy="median").fit(X_tr_raw)                  # fitted on TRAIN only
scaler = StandardScaler().fit(imputer.transform(X_tr_raw))


def prep(a):
    return scaler.transform(imputer.transform(a)).astype(np.float32)


X_tr_s, X_va_s, X_te_s = prep(X_tr_raw), prep(X_va_raw), prep(X_te_raw)

smote = SMOTE(sampling_strategy=SMOTE_RATIO, k_neighbors=min(5, int(y_tr.sum()) - 1), random_state=RANDOM_STATE)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_s, y_tr)                       # TRAIN only, never val/test
X_tr_sm = X_tr_sm.astype(np.float32)

st = Table(title="Split sizes", box=box.SIMPLE_HEAVY, header_style="bold")
st.add_column("Part"); st.add_column("Rows", justify="right"); st.add_column("Hazardous", justify="right")
st.add_column("Rate", justify="right")
for nm, yy in [("Train (original)", y_tr), ("Train (after SMOTE)", y_tr_sm), ("Validation", y_va), ("Test", y_te)]:
    st.add_row(nm, f"{len(yy):,}", f"{int(yy.sum()):,}", f"{yy.mean() * 100:.2f}%")
console.print(st)
console.print(f"  Test set has {int(y_te.sum())} hazardous asteroids (paper: 413 out of 187,721). "
              "With a subset this number is much smaller, so the metrics are noisier.")


def fig_class_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, yy, title in [(axes[0], y_tr, "Training set BEFORE SMOTE  (paper Fig. 3)"),
                          (axes[1], y_tr_sm, "Training set AFTER SMOTE  (paper Fig. 4)")]:
        counts = [int((yy == 0).sum()), int((yy == 1).sum())]
        bars = ax.bar(["Non-hazardous", "Hazardous"], counts, color=[PALETTE["blue"], PALETTE["orange"]])
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("asteroids")
    fig.tight_layout()
    savefig(fig, "fig2_class_distribution_smote.png")


fig_class_distribution()

# ==============================================================================
# STEP 3 - DEEP MODELS ON THE KNN GRAPH
# ==============================================================================
HISTORY = {}
GNN_MODEL = None
GNN_GRAPH_FOR_IMPORTANCE = None
IMPORTANCES = None

if ENABLE_DEEP_MODELS:
    console.rule(f"[bold cyan]STEP 3  Graph models  (device: {DEVICE})")

    # ---- 3a. GNN trained on SMOTE-balanced training nodes --------------------------------
    with console.status("Building the k-nearest-neighbour graph (SMOTE version) ..."):
        t0 = time.time()
        graph_sm = build_graph_torch(X_tr_sm, y_tr_sm, X_va_s, y_va, X_te_s, y_te)
    console.print(f"  graph (SMOTE): {graph_sm.num_nodes:,} nodes, {graph_sm.num_edges:,} edges "
                  f"(built in {time.time() - t0:.0f}s)")

    torch.manual_seed(RANDOM_STATE)
    gnn = HazardGNN(in_dim=len(FEATURES), edge_dim=len(FEATURES))
    gnn, hist, best_ep = train_node_model(gnn, graph_sm, "GNN (SMOTE)")
    HISTORY["GNN (SMOTE)"] = (hist, best_ep)
    probs = predict_nodes(gnn, graph_sm)
    m_test, m_val = graph_sm.test_mask.cpu().numpy(), graph_sm.val_mask.cpu().numpy()
    add_result("GNN (SMOTE)", y_te, probs[m_test])
    print_model_row("GNN (SMOTE)")

    thr = best_f1_threshold(y_va, probs[m_val])
    add_result("GNN (SMOTE) tuned threshold*", y_te, probs[m_test], threshold=thr,
               note=f"threshold {thr:.2f} picked on validation set")
    print_model_row("GNN (SMOTE) tuned threshold*")

    with console.status("Computing feature importance (permutation on the test nodes) ..."):
        IMPORTANCES = permutation_importance(gnn, graph_sm, FEATURES)
    GNN_MODEL = gnn
    torch.save(gnn.state_dict(), os.path.join(OUTPUT_DIR, "gnn_smote_weights.pt"))

    # ---- 3b. graph without SMOTE (used by "GNN (Original)" and, by default, the baselines) ----
    del graph_sm
    with console.status("Building the k-nearest-neighbour graph (original, imbalanced version) ..."):
        graph_or = build_graph_torch(X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te)
    m_test = graph_or.test_mask.cpu().numpy()

    torch.manual_seed(RANDOM_STATE)
    gnn_o = HazardGNN(in_dim=len(FEATURES), edge_dim=len(FEATURES))
    gnn_o, hist, best_ep = train_node_model(gnn_o, graph_or, "GNN (Original)")
    HISTORY["GNN (Original)"] = (hist, best_ep)
    add_result("GNN (Original)", y_te, predict_nodes(gnn_o, graph_or)[m_test])
    print_model_row("GNN (Original)")

    base_graph = graph_or
    if BASELINES_USE_SMOTE:
        del graph_or
        base_graph = build_graph_torch(X_tr_sm, y_tr_sm, X_va_s, y_va, X_te_s, y_te)
        m_test = base_graph.test_mask.cpu().numpy()

    # ---- 3c. MLP baseline (no graph) ----
    torch.manual_seed(RANDOM_STATE)
    mlp, hist, best_ep = train_node_model(MLP(len(FEATURES)), base_graph, "MLP")
    HISTORY["MLP"] = (hist, best_ep)
    add_result("MLP", y_te, predict_nodes(mlp, base_graph)[m_test])
    print_model_row("MLP")

    # ---- 3d. GAT baseline ----
    torch.manual_seed(RANDOM_STATE)
    gat, hist, best_ep = train_node_model(PlainGAT(len(FEATURES)), base_graph, "GAT")
    HISTORY["GAT"] = (hist, best_ep)
    add_result("GAT", y_te, predict_nodes(gat, base_graph)[m_test])
    print_model_row("GAT")
    del base_graph

# ==============================================================================
# STEP 4 - ISOLATION FOREST (paper baseline) + TREE MODELS (extra)
# ==============================================================================
console.rule("[bold cyan]STEP 4  Isolation Forest and tree-based reference models")
iso = IsolationForest(n_estimators=200, contamination=IFOREST_CONTAMINATION,
                      random_state=RANDOM_STATE, n_jobs=N_JOBS).fit(X_tr_s)
iso_score = -iso.score_samples(X_te_s)                       # higher = more anomalous
iso_pred = (iso.predict(X_te_s) == -1).astype(int)
add_result("iForest", y_te, iso_score, pred=iso_pred)
print_model_row("iForest")

if RUN_TREE_REFERENCE:
    rf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                   ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                    class_weight="balanced_subsample",
                                                    n_jobs=N_JOBS, random_state=RANDOM_STATE))])
    rf.fit(X_tr_raw, y_tr)
    add_result("Random Forest*", y_te, rf.predict_proba(X_te_raw)[:, 1])
    print_model_row("Random Forest*")
    joblib.dump({"pipeline": rf, "features": FEATURES}, os.path.join(OUTPUT_DIR, "asteroid_random_forest.pkl"))

    try:
        from xgboost import XGBClassifier
        pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
        xgb = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("model", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                subsample=0.8, colsample_bytree=0.8,
                                                scale_pos_weight=pos_weight, eval_metric="logloss",
                                                tree_method="hist", n_jobs=N_JOBS,
                                                random_state=RANDOM_STATE))])
        xgb.fit(X_tr_raw, y_tr)
        add_result("XGBoost*", y_te, xgb.predict_proba(X_te_raw)[:, 1])
        print_model_row("XGBoost*")
        joblib.dump({"pipeline": xgb, "features": FEATURES}, os.path.join(OUTPUT_DIR, "asteroid_xgboost.pkl"))
    except ImportError:
        console.print("  xgboost is not installed -> skipped (pip install xgboost)")

joblib.dump({"imputer": imputer, "scaler": scaler, "features": FEATURES},
            os.path.join(OUTPUT_DIR, "preprocessing.pkl"))

# ==============================================================================
# STEP 5 - RESULTS TABLES
# ==============================================================================
console.rule("[bold cyan]STEP 5  Results on the TEST set (hazardous class)")

rt = Table(title="All models (metrics are for the hazardous class; * = not in the paper)",
           box=box.SIMPLE_HEAVY, header_style="bold")
for c in ["Model", "Prec", "Recall", "F1", "ROC", "PR-AUC", "Acc", "TP", "FP", "FN", "TN"]:
    if c == "Model":
        rt.add_column(c, justify="left", no_wrap=True, min_width=30)
    else:
        rt.add_column(c, justify="right", no_wrap=True)
for name, r in RESULTS.items():
    rt.add_row(name, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}",
               f"{r['pr_auc']:.3f}", f"{r['accuracy']:.4f}", f"{r['tp']:,}", f"{r['fp']:,}", f"{r['fn']:,}",
               f"{r['tn']:,}")
console.print(rt)

pt = Table(title="This run vs the paper (paper Table III)", box=box.SIMPLE_HEAVY, header_style="bold")
for c in ["Model", "Paper P", "Paper R", "Paper F1", "Ours P", "Ours R", "Ours F1", "F1 diff"]:
    if c == "Model":
        pt.add_column(c, justify="left", no_wrap=True, min_width=16)
    else:
        pt.add_column(c, justify="right", no_wrap=True)
for name, (pp, pr_, pf) in PAPER.items():
    if name in RESULTS:
        r = RESULTS[name]
        d = r["f1"] - pf
        colour = "green" if abs(d) <= 0.05 else ("yellow" if abs(d) <= 0.15 else "red")
        pt.add_row(name, f"{pp:.2f}", f"{pr_:.2f}", f"{pf:.2f}", f"{r['precision']:.2f}", f"{r['recall']:.2f}",
                   f"{r['f1']:.2f}", f"[{colour}]{d:+.2f}[/{colour}]")
console.print(pt)
console.print("  Paper (full data): TN 186,287 | FP 1,021 | FN 92 | TP 321 | accuracy 0.99 | ROC-AUC 0.99")

if IMPORTANCES:
    it = Table(title="Feature importance of the GNN (drop in test PR-AUC when the feature is shuffled)",
               box=box.SIMPLE, header_style="bold")
    it.add_column("Feature"); it.add_column("Importance", justify="right")
    for f, v in sorted(IMPORTANCES.items(), key=lambda kv: -kv[1]):
        it.add_row(f, f"{v:.4f}")
    console.print(it)
    console.print("  Paper's top features: albedo (0.0704), perihelion distance q (0.0305), semi-major axis a (0.0303)")

# ==============================================================================
# STEP 6 - FIGURES
# ==============================================================================
console.rule("[bold cyan]STEP 6  Figures")
primary = "GNN (SMOTE)" if "GNN (SMOTE)" in RESULTS else ("Random Forest*" if "Random Forest*" in RESULTS else "iForest")

# -- confusion matrix (paper Fig. 6)
r = RESULTS[primary]
cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
fig, ax = plt.subplots(figsize=(5.6, 4.6))
sns.heatmap(np.log10(cm + 1), annot=cm, fmt=",d", cmap=SPACE_CMAP, cbar=False, ax=ax,
            xticklabels=["Predicted\nnon-hazardous", "Predicted\nhazardous"],
            yticklabels=["Actually\nnon-hazardous", "Actually\nhazardous"], linewidths=2, linecolor="white",
            annot_kws={"fontsize": 13, "fontweight": "bold"})
ax.set_title(f"Confusion matrix - {primary}\nprecision {r['precision']:.2f} | recall {r['recall']:.2f} | "
             f"F1 {r['f1']:.2f}   (paper Fig. 6)", fontsize=11)
fig.tight_layout()
savefig(fig, "fig3_confusion_matrix.png")

# -- ROC + precision-recall curves (paper Fig. 7 + extra)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
curve_items = [(n, r) for n, r in RESULTS.items() if "tuned threshold" not in n]   # same curve as the untuned model
colors = sns.color_palette("tab10", n_colors=len(curve_items))
prevalence = float(np.mean(next(iter(RESULTS.values()))["y_true"]))
for (name, r), col in zip(curve_items, colors):
    fpr, tpr, _ = roc_curve(r["y_true"], r["score"])
    prc, rec, _ = precision_recall_curve(r["y_true"], r["score"])
    axes[0].plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC {r['roc_auc']:.3f})")
    axes[1].plot(rec, prc, color=col, lw=2, label=f"{name} (AP {r['pr_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "--", color=PALETTE["grey"])
axes[0].set(title="ROC curve  (paper Fig. 7)", xlabel="False positive rate", ylabel="True positive rate")
axes[1].axhline(prevalence, ls="--", color=PALETTE["grey"], label=f"random guess ({prevalence:.4f})")
axes[1].set(title="Precision-recall curve  (more honest for rare classes)", xlabel="Recall", ylabel="Precision")
axes[0].legend(fontsize=8, loc="lower right")
axes[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
savefig(fig, "fig4_roc_and_pr_curves.png")

# -- training curves (paper Fig. 8)
if "GNN (SMOTE)" in HISTORY:
    hist, best_ep = HISTORY["GNN (SMOTE)"]
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ep, hist["train_loss"], color=PALETTE["blue"], lw=2, label="training loss")
    axes[0].plot(ep, hist["val_loss"], color=PALETTE["orange"], lw=2, label="validation loss")
    axes[0].axvline(best_ep, ls="--", color=PALETTE["grey"], label=f"best epoch ({best_ep})")
    axes[0].set(title="Training and validation loss  (paper Fig. 8)", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(ep, hist["val_pr_auc"], color=PALETTE["teal"], lw=2)
    axes[1].axvline(best_ep, ls="--", color=PALETTE["grey"])
    axes[1].set(title="Validation PR-AUC (used for early stopping)", xlabel="epoch", ylabel="PR-AUC")
    fig.tight_layout()
    savefig(fig, "fig5_training_curves.png")

# -- feature importance (paper Fig. 9)
if IMPORTANCES:
    items = sorted(IMPORTANCES.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh([k for k, _ in items], [v for _, v in items], color=PALETTE["purple"])
    ax.set(title="GNN feature importance (permutation, drop in PR-AUC)  |  paper Fig. 9", xlabel="importance")
    fig.tight_layout()
    savefig(fig, "fig6_feature_importance.png")

# -- our results vs the paper
common = [n for n in PAPER if n in RESULTS]
if common:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, (k, label) in zip(axes, [(0, "Precision"), (1, "Recall"), (2, "F1-score")]):
        w = 0.38
        xs = np.arange(len(common))
        ax.bar(xs - w / 2, [PAPER[n][k] for n in common], w, color=PALETTE["grey"], label="paper")
        ax.bar(xs + w / 2, [RESULTS[n][["precision", "recall", "f1"][k]] for n in common], w,
               color=PALETTE["blue"], label="this run")
        ax.set_xticks(xs)
        ax.set_xticklabels(common, rotation=25, ha="right", fontsize=9)
        ax.set_title(f"{label} (hazardous class)")
    axes[0].legend()
    fig.suptitle("Recreation vs paper (Table III)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "fig7_comparison_with_paper.png")

# -- save the metrics table
rows = []
for name, r in RESULTS.items():
    rows.append({k: v for k, v in r.items() if k not in ("score", "y_true")} | {"model": name})
pd.DataFrame(rows).set_index("model").to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"))

# ==============================================================================
# DONE
# ==============================================================================
notes = [
    "[bold]How to read the numbers[/bold]",
    "  Precision = of the asteroids flagged hazardous, how many really were.",
    "  Recall    = of the truly hazardous asteroids, how many were caught.",
    "  F1        = one number combining precision and recall (use this, not accuracy).",
    "  Accuracy is ~0.99 for almost any model here because 99.8% of asteroids are harmless.",
    "",
    "[bold]Why results will not match the paper exactly[/bold]",
    "  * The paper's code details (loss weights, dropout, learning rate, exact features) are not all published.",
    f"  * This run used {len(df):,} rows instead of ~938,600, so the test set has only {int(y_te.sum())} "
    "hazardous asteroids (paper: 413) and F1 can swing by several points between runs.",
]
if not REMOVE_LABEL_DEFINING_FEATURES:
    notes.append("  * moid and H are in the features (as in the paper). They define the hazardous label, so tree "
                 "models (marked *) can score near-perfectly: that is leakage, not skill.")
else:
    notes.append("  * moid and H were REMOVED, so scores are lower but honest.")
console.print(Panel("\n".join(notes), border_style="cyan"))
console.print(f"[bold green]Finished in {(time.time() - T_START) / 60:.1f} min.[/bold green]  "
              f"Everything is in the '{OUTPUT_DIR}' folder.")

if SHOW_PLOTS:
    plt.show()