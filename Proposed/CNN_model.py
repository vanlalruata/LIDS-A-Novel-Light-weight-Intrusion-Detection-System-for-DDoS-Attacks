#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

# --------------------------- STEP 0: imports ---------------------------
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

# plotting (headless-safe)
import matplotlib
matplotlib.use("Agg")  # comment out if running interactively
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# tqdm (optional)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

plt.rcParams["figure.figsize"] = (7.5, 4.5)

# ---- DEVICE PICKER (MPS -> CUDA -> CPU) ----
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

PIN_MEMORY = (DEVICE.type == "cuda")  # pin_memory is CUDA-only
NUM_WORKERS = 0                       # safe default on macOS; raise if you want

def _sync_if_accel():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

# --------------------------- STEP 1: load CSVs & filter ---------------------------

# file paths (UPDATE THESE)
TRAIN_CSV = "/Users/drashfaqahmadnajar/Downloads/LIDS-main_multi/Datasets/train_PCA_Dataset.csv"
TEST_CSV  = "/Users/drashfaqahmadnajar/Downloads/LIDS-main_multi/Datasets/test_PCA_Dataset.csv"

# optional row caps (None => all rows)
TRAIN_NROWS = None
TEST_NROWS  = None

# features and labels
N_FEATURES = 10
FEATURES = [f"PC {i+1}" for i in range(N_FEATURES)]
LABEL_COLS = [" Label", "label", "Label"]  # handle variants

# Original raw-id → readable name (for traceability)
ORIG_LABEL_NAME_MAP = {
    0:  "BENIGN",
    1:  "DrDoS_DNS",
    2:  "DrDoS_LDAP",
    3:  "DrDoS_MSSQL",
    4:  "DrDoS_NTP",
    5:  "DrDoS_NetBIOS",
    6:  "DrDoS_SNMP",
    7:  "UDP-lag",
    8:  "DrDoS_UDP",
    9:  "Syn",
    10: "TFTP",
    11: "→ mapped to 7 (UDP-lag) per CIC-clean",
    12: "removed by CIC-clean",
}

# Remove these raw classes entirely
REMOVE_CLASSES = {1, 2, 5}  # DrDoS_DNS, DrDoS_LDAP, DrDoS_NetBIOS

def _find_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_COLS:
        if c in df.columns:
            return c
    raise KeyError("Label column not found. Tried: ' Label', 'label', 'Label'.")

def _load_and_filter(path: str, n_features: int = 10, nrows=None):
    """
    Load a CSV and apply CIC-like cleaning (remove 12, remove 7, 11->7),
    then drop REMOVE_CLASSES. Returns (X_df, y_series_raw, label_col).
    Raw labels remain 'original ids' after CIC steps, NOT renumbered.
    """
    df = pd.read_csv(path, nrows=nrows)
    label_col = _find_label_col(df)

    print(f"\nLoaded: {os.path.basename(path)}  shape={df.shape}")
    print("Label value_counts (raw, top 15):")
    print(df[label_col].value_counts().head(15))

    # CIC2019Multi-like filtering
    df = df[df[label_col] != 12]          # drop 12
    df = df[df[label_col] != 7]           # drop 7
    df[label_col] = df[label_col].replace(11, 7)  # 11 -> 7

    # Drop unwanted classes
    df = df[~df[label_col].isin(REMOVE_CLASSES)]

    df = df.reset_index(drop=True)

    print("\nAfter filter (remove 12, remove 7, replace 11→7) AND drop 1/2/5:")
    vc = df[label_col].value_counts().sort_index()
    print(vc.to_string())

    # Drop helper columns and keep PCs
    drop_cols = [c for c in df.columns if ("Unnamed" in c) or (c == label_col)]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X[FEATURES].astype(np.float32)
    y_raw = df[label_col].astype(int)     # still 'original ids' (0,3,4,6,7,8,9,10)

    return X, y_raw, label_col

# Load both splits (raw ids after filtering)
X_train_raw, y_train_raw, _ = _load_and_filter(TRAIN_CSV, n_features=N_FEATURES, nrows=TRAIN_NROWS)
X_test_raw,  y_test_raw,  _ = _load_and_filter(TEST_CSV,  n_features=N_FEATURES, nrows=TEST_NROWS)

print("\nShapes (raw):")
print("Train:", X_train_raw.shape, y_train_raw.shape)
print("Test :", X_test_raw.shape,  y_test_raw.shape)

# --------------------------- STEP 1B: build a GLOBAL contiguous mapping ---------------------------

# Remaining original classes across BOTH splits
orig_classes = sorted(set(y_train_raw.unique()).union(set(y_test_raw.unique())))
# Build new contiguous ids: 0..K-1
raw_to_new = {orig: new for new, orig in enumerate(orig_classes)}
new_to_raw = {new: orig for orig, new in raw_to_new.items()}
# Human-readable names for new contiguous ids
class_name_list = [ORIG_LABEL_NAME_MAP.get(new_to_raw[i], str(new_to_raw[i])) for i in range(len(orig_classes))]

print("\nNew contiguous mapping (after dropping 1,2,5):")
for new_id, orig_id in new_to_raw.items():
    print(f"  new {new_id} ← orig {orig_id}  ({class_name_list[new_id]})")

# Map labels to contiguous ids for BOTH splits
y_train = y_train_raw.map(raw_to_new).astype(int)
y_test  = y_test_raw.map(raw_to_new).astype(int)
NUM_CLASSES = len(class_name_list)

# Save legend for reproducibility
os.makedirs("../Images", exist_ok=True)
legend_df = pd.DataFrame({
    "new_id": list(range(NUM_CLASSES)),
    "class_name": class_name_list,
    "original_id": [new_to_raw[i] for i in range(NUM_CLASSES)],
})
legend_df.to_csv("Images/class_id_legend.csv", index=False)
print("[SAVE] Images/class_id_legend.csv")

# --------------------------- STEP 2: visualize class balance ---------------------------

def plot_counts(y_contig: pd.Series, title, out_path=None):
    vc = pd.Series(y_contig).value_counts().sort_index()  # indices are 0..K-1
    ax = vc.plot(kind="bar")
    try:
        xtlbls = [class_name_list[int(i)] for i in vc.index.tolist()]
        ax.set_xticklabels(xtlbls, rotation=45, ha='right')
    except Exception:
        pass
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"[PLOT] Saved → {out_path}")
    plt.close()

plot_counts(y_train, "Train class distribution (after filter + drop + renumber)",
            out_path="Images/train_class_distribution.png")
plot_counts(y_test,  "Test class distribution  (after filter + drop + renumber)",
            out_path="Images/test_class_distribution.png")

# --------------------------- STEP 3: Dataset & DataLoaders (contiguous ids) ---------------------------

class PCADataset(Dataset):
    def __init__(self, X_df: pd.DataFrame, y_series_contig: pd.Series):
        self.X = torch.from_numpy(X_df.to_numpy().astype(np.float32))
        self.y = torch.from_numpy(y_series_contig.to_numpy().astype(np.int64))
        uniq = np.sort(np.unique(self.y.numpy()))
        self.num_classes = len(uniq)
        # names by contiguous ids
        self.class_names = [class_name_list[i] for i in uniq]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds_full = PCADataset(X_train_raw, y_train)
test_ds_full  = PCADataset(X_test_raw,  y_test)

# split test -> (test, val)  (10% to val)
VAL_FRACTION = 0.10
val_size  = int(VAL_FRACTION * len(test_ds_full))
test_size = len(test_ds_full) - val_size
test_ds, val_ds = random_split(test_ds_full, [test_size, val_size])

# DataLoaders (mac-safe)
BATCH_SIZE = 1024
train_loader = DataLoader(train_ds_full, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_ds,      batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_ds,     batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

print("Contiguous classes seen in TRAIN:", list(range(NUM_CLASSES)))
print("Class names:", class_name_list)
print("num_classes:", NUM_CLASSES)

# sanity-check one batch
xb, yb = next(iter(train_loader))
print("One batch shapes:", xb.shape, yb.shape, yb.dtype)
print("Unique contiguous labels in batch:", torch.unique(yb).tolist())

# --------------------------- STEP 3A: balance helpers ---------------------------

def summarize_class_balance(loader, name="split"):
    counts = {i: 0 for i in range(NUM_CLASSES)}
    total = 0
    for _, yb in loader:
        vals, cts = torch.unique(yb.cpu(), return_counts=True)
        for v, c in zip(vals.tolist(), cts.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
        total += len(yb)

    pct = {k: (counts[k] / total if total else 0.0) for k in counts}
    for k in range(NUM_CLASSES):
        nm = class_name_list[k]
        print(f"[BALANCE] {name}  class {k:>2} ({nm}): count={counts[k]:>8}  pct={pct[k]*100:6.2f}%")
    return counts, pct

def summarize_loader_balance(loader, name, save_prefix="Images"):
    counts = {i: 0 for i in range(NUM_CLASSES)}
    total = 0
    for _, y in loader:
        vals, cts = torch.unique(y.cpu(), return_counts=True)
        for v, c in zip(vals.tolist(), cts.tolist()):
            counts[int(v)] = counts.get(int(v), 0) + int(c)
        total += len(y)

    rows = [{"class_id": i, "class_name": class_name_list[i],
             "count": counts[i], "pct": counts[i]/total if total else 0.0}
            for i in range(NUM_CLASSES)]
    df = pd.DataFrame(rows)
    print(f"[BALANCE] {name}: total={total}\n{df}")

    # bar plot with readable names
    ax = df.plot(kind="bar", x="class_name", y="count", legend=False, title=f"{name} class counts")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    plt.tight_layout()
    os.makedirs(save_prefix, exist_ok=True)
    outp = os.path.join(save_prefix, f"{name.lower()}_class_counts.png")
    plt.savefig(outp, dpi=300)
    print(f"[PLOT] Saved → {outp}")
    plt.close()
    return df

# quick balance summaries to stdout
counts_train, pct_train = summarize_class_balance(train_loader, "train")
counts_val,   pct_val   = summarize_class_balance(val_loader,   "val")
counts_test,  pct_test  = summarize_class_balance(test_loader,  "test")

df_train_bal = summarize_loader_balance(train_loader, "TRAIN")
df_val_bal   = summarize_loader_balance(val_loader,   "VAL")
df_test_bal  = summarize_loader_balance(test_loader,  "TEST")

# --------------------------- STEP 5: model + class weights + optimizer ---------------------------

# compute class weights from TRAIN split
cls_counts = dict(zip(df_train_bal["class_id"].tolist(), df_train_bal["count"].tolist()))
uniq_sorted = np.array(sorted(cls_counts.keys()))  # should be 0..K-1
counts_vec  = np.array([cls_counts[k] for k in uniq_sorted], dtype=float)
inv         = 1.0 / np.clip(counts_vec, 1, None)
weights     = inv / inv.sum() * len(inv)

print("Class IDs (contiguous):", uniq_sorted.tolist())
print("Counts                :", counts_vec.astype(int).tolist())
print("Weights               :", np.round(weights, 6).tolist())
print("ID → Name mapping:")
for cid in uniq_sorted:
    print(f"  {int(cid)} → {class_name_list[int(cid)]} (orig {new_to_raw[int(cid)]})")

class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

class LCNNModelMulti(nn.Module):
    """
    1D CNN over PCA features with better calibration:
      Conv1d → BatchNorm → ReLU → MaxPool → FC → ReLU → Dropout → FC
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=1, padding=0)
        self.bn   = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 10 -> 5
        self.fc1  = nn.Linear(16 * (N_FEATURES // 2), 64)
        self.drop = nn.Dropout(p=0.4)            # a bit less than 0.6
        self.fc2  = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)             # (N, 1, D)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)               # (N, 16, D//2)
        x = x.flatten(1)               # (N, 16*(D//2))
        x = self.fc1(x)
        x = self.relu(x)               # <— add nonlinearity you were missing
        x = self.drop(x)
        return self.fc2(x)

# ----------------------------------------------------------------------

model = LCNNModelMulti(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)   # weighted CE for imbalance
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

# quick forward test
with torch.no_grad():
    logits = model(xb.to(DEVICE).float())
print("Logits shape:", logits.shape)  # (batch, C)

# --------------------------- STEP 6: training loop + Early Stopping + curves ---------------------------

class EarlyStopping:
    """
    Monitors validation loss (default) with patience; restores best weights.
    """
    def __init__(self, patience=3, min_delta=1e-4, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.best_state = None

    def step(self, metric, model):
        improve = False
        if self.best is None:
            improve = True
        else:
            if self.mode == "min":
                improve = (self.best - metric) > self.min_delta
            else:
                improve = (metric - self.best) > self.min_delta

        if improve:
            self.best = metric
            self.counter = 0
            # keep a CPU copy of best weights
            self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        stop = self.counter >= self.patience
        return stop

train_hist, val_hist = {"loss": [], "acc": []}, {"loss": [], "acc": []}

def run_epoch(loader, train=False):
    model.train() if train else model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, y in tqdm(loader, disable=False):
        X = X.to(DEVICE).float()
        y = y.to(DEVICE).long()
        with torch.set_grad_enabled(train):
            logits = model(X)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        pred = torch.argmax(logits, dim=1)
        total_loss += loss.item() * y.size(0)
        total_acc  += (pred == y).float().sum().item()
        n += y.size(0)
    return total_loss / max(1, n), total_acc / max(1, n)

EPOCHS = 50
early = EarlyStopping(patience=8, min_delta=1e-4, mode="min")  # monitor val loss

_sync_if_accel()
t_train0 = time.perf_counter()

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader,   train=False)
    train_hist["loss"].append(tr_loss); train_hist["acc"].append(tr_acc)
    val_hist["loss"].append(va_loss);   val_hist["acc"].append(va_acc)

    print(f"Epoch {ep:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

    # early stopping on val loss
    stop = early.step(va_loss, model)
    if stop:
        print(f"[EarlyStopping] Patience reached at epoch {ep}. Restoring best weights.")
        break

# restore best weights if available
if early.best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in early.best_state.items()})
    print(f"Best val loss: {early.best:.6f}")

_sync_if_accel()
t_train1 = time.perf_counter()
train_time_s = t_train1 - t_train0
print(f"[TIME] Total training time: {train_time_s:.2f} s")


# Convert the history dictionaries into a pandas DataFrame for saving
history_df = pd.DataFrame({
    'epoch': np.arange(1, len(train_hist["loss"]) + 1),
    'train_loss': train_hist["loss"],
    'val_loss': val_hist["loss"],
    'train_acc': train_hist["acc"],
    'val_acc': val_hist["acc"]
})

# Ensure the directory exists and save the DataFrame
os.makedirs("History", exist_ok=True)
history_filename = "History/training_history.csv"
history_df.to_csv(history_filename, index=False)
print(f"[SAVE] Training history saved → {history_filename}")

# curves
xs = np.arange(1, len(train_hist["loss"]) + 1)
plt.figure(figsize=(8, 3.8))
plt.subplot(1, 2, 1); plt.plot(xs, train_hist["acc"]); plt.plot(xs, val_hist["acc"]); plt.title("Accuracy"); plt.xlabel("epoch"); plt.legend(["train","val"])
plt.subplot(1, 2, 2); plt.plot(xs, train_hist["loss"]); plt.plot(xs, val_hist["loss"]); plt.title("Loss"); plt.xlabel("epoch"); plt.legend(["train","val"])
plt.tight_layout()
os.makedirs("../Images", exist_ok=True)
plt.savefig("Images/training_curves.png", dpi=300)
print("[PLOT] Saved → Images/training_curves.png")
plt.close()

# --------------------------- STEP 7: evaluation on TEST ---------------------------

model.eval()
y_true, y_pred = [], []

_sync_if_accel()
t0 = time.perf_counter()
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEVICE).float()
        logits = model(X)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())
_sync_if_accel()
t1 = time.perf_counter()

test_time_s = t1 - t0
n_samples = len(y_true)
avg_ms_per_sample = (test_time_s / max(1, n_samples)) * 1000.0
throughput = n_samples / max(test_time_s, 1e-9)
print(f"[TIME] Test prediction time: {test_time_s:.2f} s  |  "
      f"{avg_ms_per_sample:.3f} ms/sample  |  {throughput:.1f} samples/s")

# --- Metrics ---
target_names = class_name_list
labels_contig = list(range(NUM_CLASSES))
print("\n=== TEST CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=target_names, labels=labels_contig, digits=3))

cm = confusion_matrix(y_true, y_pred, labels=labels_contig)
print("Confusion matrix shape:", cm.shape)

print("\n=== CONFUSION MATRIX (raw counts) ===")
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=[f"T_{n}" for n in class_name_list],
    columns=[f"P_{n}" for n in class_name_list],
)
print("\n=== CONFUSION MATRIX (labeled) ===")
print(cm_df)

cm_norm = (cm.T / np.clip(cm.sum(axis=1, keepdims=True), 1, None)).T
cmn_df = pd.DataFrame(
    np.round(cm_norm, 4),
    index=[f"T_{n}" for n in class_name_list],
    columns=[f"P_{n}" for n in class_name_list],
)
print("\n=== CONFUSION MATRIX (row-normalized) ===")
print(cmn_df)

os.makedirs("../Images", exist_ok=True)
cm_df.to_csv("Images/confusion_matrix_counts.csv")
cmn_df.to_csv("Images/confusion_matrix_rownorm.csv")
print("[SAVE] CSVs -> Images/confusion_matrix_counts.csv, Images/confusion_matrix_rownorm.ouyrxfcsv")

# --- Plot heatmap (raw counts) with annotations ---
plt.figure(figsize=(7.5, 6.5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(NUM_CLASSES), class_name_list, rotation=45, ha='right')
plt.yticks(range(NUM_CLASSES), class_name_list)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix (Counts)")

thresh = cm.max() / 2.0 if cm.size else 0.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=9)

plt.tight_layout()
plt.savefig("Images/confusion_matrix_counts.png", dpi=300)
print("[PLOT] Saved → Images/confusion_matrix_countskhfd.png")
plt.close()

# --- Plot heatmap (row-normalized %) with annotations ---
plt.figure(figsize=(7.5, 6.5))
plt.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xticks(range(NUM_CLASSES), class_name_list, rotation=45, ha='right')
plt.yticks(range(NUM_CLASSES), class_name_list)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix (Row-normalized %)")

thresh_norm = 0.5
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%",
                 ha="center", va="center",
                 color="white" if cm_norm[i, j] > thresh_norm else "black",
                 fontsize=9)

plt.tight_layout()
plt.savefig("Images/confusion_matrix_percent.png", dpi=300)
print("[PLOT] Saved → Images/confusion_matrix_percentkfdh.png")
plt.close()



# ---------------- Overall confusion matrices ----------------
cm = confusion_matrix(y_true,y_pred,labels=labels_contig)
print("Confusion matrix shape:", cm.shape)
print("\n=== CONFUSION MATRIX (raw counts) ===")
print(cm)

cm_df = pd.DataFrame(cm, index=[f"T_{n}" for n in class_name_list],
                         columns=[f"P_{n}" for n in class_name_list])
print("\n=== CONFUSION MATRIX (labeled) ===")
print(cm_df)

cm_norm = (cm.T / np.clip(cm.sum(axis=1,keepdims=True),1,None)).T
cmn_df = pd.DataFrame(np.round(cm_norm,4),
                      index=[f"T_{n}" for n in class_name_list],
                      columns=[f"P_{n}" for n in class_name_list])
print("\n=== CONFUSION MATRIX (row-normalized) ===")
print(cmn_df)

cm_df.to_csv("Images/confusion_matrix_counts.csv")
cmn_df.to_csv("Images/confusion_matrix_rownorm.csv")
print("[SAVE] CSVs -> Images/confusion_matrix_counts.csv, Images/confusion_matrix_rownormffd.csv")

# Heatmaps (overall)
plt.figure(figsize=(7.5,6.5))
plt.imshow(cm,cmap="Blues"); plt.colorbar()
plt.xticks(range(NUM_CLASSES),class_name_list,rotation=45,ha='right')
plt.yticks(range(NUM_CLASSES),class_name_list)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Counts)")
thresh = cm.max()/2.0 if cm.size else 0.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,str(cm[i,j]),ha="center",va="center",
                 color="white" if cm[i,j]>thresh else "black", fontsize=9)
plt.tight_layout(); plt.savefig("Images/confusion_matrix_countsfd.png",dpi=300); plt.close()
print("[PLOT] Saved → Images/confusion_matrix_countsgfdd.png")

plt.figure(figsize=(7.5,6.5))
plt.imshow(cm_norm,cmap="Blues",vmin=0.0,vmax=1.0); plt.colorbar()
plt.xticks(range(NUM_CLASSES),class_name_list,rotation=45,ha='right')
plt.yticks(range(NUM_CLASSES),class_name_list)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrixff (Row-normalized %)")
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j,i,f"{cm_norm[i,j]*100:.1f}%",ha="center",va="center",
                 color="white" if cm_norm[i,j]>0.5 else "black", fontsize=9)
plt.tight_layout(); plt.savefig("Images/confusion_matrix_percentffx.png",dpi=300); plt.close()
print("[PLOT] Saved → Images/confusion_matrix_percent.png")

# ---------------- NEW: Per-class (one-vs-rest) confusion matrices ----------------
def per_class_confusions(y_true_list, y_pred_list, class_names):
    """
    For each class c, compute binary confusion matrix:
        [[TN, FP],
         [FN, TP]]
    Saves CSV + count heatmap + percent heatmap.
    """
    y_true_arr = np.array(y_true_list, dtype=int)
    y_pred_arr = np.array(y_pred_list, dtype=int)
    out_dir = "Images/per_class"
    os.makedirs(out_dir, exist_ok=True)

    for c in range(len(class_names)):
        cname = class_names[c]
        # one-vs-rest binarization
        yt = (y_true_arr == c).astype(int)
        yp = (y_pred_arr == c).astype(int)

        cm_bin = confusion_matrix(yt, yp, labels=[0, 1])  # [[TN, FP],[FN, TP]]
        TN, FP, FN, TP = cm_bin[0,0], cm_bin[0,1], cm_bin[1,0], cm_bin[1,1]

        # Save CSV
        csv_path = os.path.join(out_dir, f"cls_{c}_{cname}_confusion_counts22.csv")
        pd.DataFrame(
            [["TN", TN], ["FP", FP], ["FN", FN], ["TP", TP]],
            columns=["metric", "count"]
        ).to_csv(csv_path, index=False)

        # Counts heatmap
        plt.figure(figsize=(4.2, 3.8))
        plt.imshow(cm_bin, cmap="Blues")
        plt.colorbar()
        plt.xticks([0,1], ["Pred:Rest", "Pred:Class"])
        plt.yticks([0,1], ["True:Rest", "True:Class"])
        plt.title(f"{c} — {cname}\nConfusion (Counts)")
        thresh_local = cm_bin.max()/2.0 if cm_bin.size else 0.0
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm_bin[i, j]),
                         ha="center", va="center",
                         color="white" if cm_bin[i, j] > thresh_local else "black",
                         fontsize=11)
        plt.tight_layout()
        png_counts = os.path.join(out_dir, f"cls_{c}_{cname}_confusion_counts22.png")
        plt.savefig(png_counts, dpi=300); plt.close()

        # Percent (row-normalized)
        row_sum = cm_bin.sum(axis=1, keepdims=True).clip(min=1)
        cm_bin_norm = (cm_bin / row_sum)
        plt.figure(figsize=(4.2, 3.8))
        plt.imshow(cm_bin_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xticks([0,1], ["Pred:Rest", "Pred:Class"])
        plt.yticks([0,1], ["True:Rest", "True:Class"])
        plt.title(f"{c} — {cname}\nConfusion (Row-normalized %)")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{cm_bin_norm[i, j]*100:.1f}%",
                         ha="center", va="center",
                         color="white" if cm_bin_norm[i, j] > 0.5 else "black",
                         fontsize=11)
        plt.tight_layout()
        png_pct = os.path.join(out_dir, f"cls_{c}_{cname}_confusion_percent33.png")
        plt.savefig(png_pct, dpi=300); plt.close()

        print(f"[PCLASS] Saved per-class confusion for {c} ({cname}) →")
        print(f"         CSV: {csv_path}")
        print(f"         PNG: {png_counts}")
        print(f"         PNG: {png_pct}")

# generate per-class confusions
per_class_confusions(y_true, y_pred, class_name_list)

# --------------------------- STEP 7B: robustness evaluators ---------------------------

def eval_with_noise(model, loader, sigma=0.01, device=None, class_names=None):
    """
    Evaluate robustness to small Gaussian noise added to inputs (preprocessed PCA space).
    """
    import numpy as _np
    import torch as _torch
    from sklearn.metrics import classification_report as _cr

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    yp, yt = [], []
    with _torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            noise = sigma * _torch.randn_like(xb)
            logits = model(xb + noise)
            yp.append(_torch.argmax(logits, 1).cpu().numpy())
            yt.append(yb.cpu().numpy())
    yp = _np.concatenate(yp); yt = _np.concatenate(yt)

    if class_names is None:
        class_names = [class_name_list[i] for i in range(NUM_CLASSES)]

    print(f"\n[ROBUST] Gaussian noise σ={sigma}")
    print(_cr(yt, yp, target_names=class_names, labels=list(range(NUM_CLASSES)), digits=3))

def eval_fgsm(model, loader, epsilon=0.01, device=None, class_names=None):
    """
    One-step FGSM adversarial evaluation with small epsilon on PCA inputs.
    """
    import numpy as _np
    import torch as _torch
    import torch.nn.functional as F
    from sklearn.metrics import classification_report as _cr

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    yp, yt = [], []
    for xb, yb in loader:
        xb = xb.to(device).float().clone().detach().requires_grad_(True)
        yb = yb.to(device).long()

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        x_adv = xb + epsilon * xb.grad.sign()

        with _torch.no_grad():
            logits_adv = model(x_adv)
            yp.append(_torch.argmax(logits_adv, 1).cpu().numpy())
            yt.append(yb.cpu().numpy())

    yp = _np.concatenate(yp); yt = _np.concatenate(yt)

    if class_names is None:
        class_names = [class_name_list[i] for i in range(NUM_CLASSES)]

    print(f"\n[ROBUST] FGSM ε={epsilon}")
    print(_cr(yt, yp, target_names=class_names, labels=list(range(NUM_CLASSES)), digits=3))

# Run robustness checks (comment out if not needed)
names = class_name_list
eval_with_noise(model, test_loader, sigma=0.01, device=DEVICE, class_names=names)
eval_with_noise(model, test_loader, sigma=0.02, device=DEVICE, class_names=names)
eval_fgsm(model, test_loader, epsilon=0.01, device=DEVICE, class_names=names)

# --------------------------- STEP 8: SHAP (robust per-class) ---------------------------
def run_shap_kernel_explainer_per_class(
    model,
    test_loader,
    device=DEVICE,
    out_dir="Images/shap",
    nsamples=200,
    cap_background=200,
    cap_explain=800,
):
    """
    Robust SHAP for multiclass: explain each class c separately by wrapping the model
    to return p(y=c|x) as a scalar. Saves per-class bar charts and a CSV.
    """
    try:
        import shap
    except Exception as e:
        print(f"[SHAP] shap not available: {e}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # ---- collect a moderate test matrix X_test_np (N, D) ----
    Xtest_list = []
    with torch.no_grad():
        for X, _ in test_loader:
            Xtest_list.append(X.numpy())
            if sum(arr.shape[0] for arr in Xtest_list) >= max(cap_explain, cap_background):
                break
    if not Xtest_list:
        print("[SHAP] No test samples collected.")
        return
    X_test_np = np.concatenate(Xtest_list, axis=0)
    N, D = X_test_np.shape
    print("[SHAP] test subset:", X_test_np.shape)

    # ---- infer number of classes C ----
    with torch.no_grad():
        tmp_logits = model(torch.tensor(X_test_np[:4], dtype=torch.float32, device=device))
        C = tmp_logits.shape[1]
    feature_names = [f"PC {i+1}" for i in range(D)]
    topk = min(D, 10)

    # ---- choose background & explain subsets ----
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(N, size=min(cap_background, N), replace=False)
    ex_idx = rng.choice(N, size=min(cap_explain, N), replace=False)
    X_bg = X_test_np[bg_idx]
    X_ex = X_test_np[ex_idx]

    # accumulate rows for a CSV and for a global summary
    csv_rows = []

    # ---- loop over contiguous classes; explain p(y=c|x) ----
    for c in range(C):
        def prob_c(Xnp):
            Xt = torch.tensor(Xnp, dtype=torch.float32, device=device)
            with torch.no_grad():
                probs = torch.softmax(model(Xt), dim=1)[:, c]
            return probs.detach().cpu().numpy()  # (N,)

        # KernelExplainer for this scalar output
        explainer_c = shap.KernelExplainer(prob_c, X_bg)
        sv_c = explainer_c.shap_values(X_ex, nsamples=nsamples)  # (N_explain, D)

        if isinstance(sv_c, list):  # guard
            sv_c = sv_c[0]
        if not isinstance(sv_c, np.ndarray) or sv_c.ndim != 2 or sv_c.shape[1] != D:
            print(f"[SHAP] Unexpected sv shape for class {c}: {getattr(sv_c,'shape',None)}")
            continue

        mean_abs = np.abs(sv_c).mean(axis=0)  # (D,)

        # ---- save per-class bar plot ----
        order = np.argsort(mean_abs)[::-1][:topk]
        plt.figure(figsize=(6.2, 3.6))
        plt.barh(range(topk), mean_abs[order])
        plt.gca().invert_yaxis()
        plt.yticks(range(topk), [feature_names[i] for i in order])
        plt.xlabel("Mean |SHAP|")
        plt.title(f"Top features for class {c} ({class_name_list[c]})")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"shap_top_features_class_{c}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[SHAP] Saved → {out_png}")

        # ---- accumulate rows for CSV ----
        for rank, fidx in enumerate(order, start=1):
            csv_rows.append({
                "class_new_id": c,
                "class_name": class_name_list[c],
                "class_orig_id": new_to_raw[c],
                "rank": rank,
                "feature": feature_names[fidx],
                "mean_abs_shap": float(mean_abs[fidx]),
            })

    # ---- save CSV ----
    if csv_rows:
        df_top = pd.DataFrame(csv_rows)
        out_csv = os.path.join(out_dir, "shap_top_features_per_class.csv")
        df_top.to_csv(out_csv, index=False)
        print(f"[SHAP] Saved → {out_csv}")
    else:
        print("[SHAP] No rows to save to CSV (possibly all classes skipped).")

# >>> ENABLE SHAP CALL (saves into Images/shap/)
run_shap_kernel_explainer_per_class(
    model,
    test_loader,
    device=DEVICE,
    out_dir="Images/shap",
    nsamples=200,         # increase for more precise SHAP (slower)
    cap_background=200,
    cap_explain=800,
)
