from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIRECTORY = "cicids2017"

# reduce leakage
IDENTIFIER_COLUMNS_TO_DROP = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",
    "Timestamp"
]

LABEL_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Bot": "Bot",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
}

LABELS_TO_DROP = {
    "Heartbleed",
    "Infiltration",
    "DoS Hulk - Attempted",
    "DoS GoldenEye - Attempted",
    "DoS slowloris - Attempted",
    "DoS Slowhttptest - Attempted",
    "FTP-Patator - Attempted",
    "SSH-Patator - Attempted",
    "Web Attack - Brute Force - Attempted",
    "Web Attack - XSS - Attempted",
    "Bot - Attempted",
    "Infiltration - Attempted"
}

def load_raw():
    # load all csvs
    csv_paths = sorted(Path(DATA_DIRECTORY).glob("*.csv"))
    frames = []
    for path in csv_paths:
        frames.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
 
    # kill whitespace
    df.columns = df.columns.str.strip()
 
    # (rosay et al. 2022) duplicate column bug
    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])
 
    # reduce leakage
    columns_to_drop = []
    for column in IDENTIFIER_COLUMNS_TO_DROP:
        if column in df.columns:
            columns_to_drop.append(column)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
 
    # inf -> NaN, then drop NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
 
    # check for any weirdness i've missed
    df["Label"] = df["Label"].str.strip()
    print(f"Labels:\n{df['Label'].value_counts()}")
 
    expected_labels = LABELS_TO_DROP.union(set(LABEL_MAP))
    unexpected_labels = set(df["Label"].unique()) - expected_labels
 
    if unexpected_labels:
        print(f"UNEXPECTED LABELS: {unexpected_labels}\n")
 
    df = df.loc[~df["Label"].isin(LABELS_TO_DROP)].copy()
    df["Label"] = df["Label"].map(LABEL_MAP)
 
    y = df["Label"]
    x = df.drop(columns=["Label"])
 
    return x, y

def split_and_preprocess(x, y, seed):
    """Load and pre-process CICIDS2017 train/test data.
    
    Returns:
        x_train_transformed, x_test_transformed,
        y_train_labels, y_test_labels,
        label_encoder"""
    # 70/30 stratified split - seeded PER EXPERIMENT RUN now, not hardcoded
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=seed
    )
 
    # VarianceThreshold - FIT TRAIN ONLY, transform both
    variance_filter = VarianceThreshold(threshold=0)
    x_train_variance = variance_filter.fit_transform(x_train_raw)
    x_test_variance = variance_filter.transform(x_test_raw)
    n_dropped = x_train_raw.shape[1] - x_train_variance.shape[1]
 
    print(f"[seed={seed}] VarianceThreshold: {x_train_raw.shape[1]} -> {x_train_variance.shape[1]} features")
 
    if n_dropped:
        dropped_features = x_train_raw.columns[~variance_filter.get_support()].tolist()
        print(f"[seed={seed}] VarianceThreshold dropped {n_dropped}: {dropped_features}")
 
    # StandardScaler - same deal
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_variance)
    x_test_scaled = scaler.transform(x_test_variance)
 
    # float32 for speed
    x_train = x_train_scaled.astype(np.float32)
    x_test = x_test_scaled.astype(np.float32)
 
    # for xgboost (and now MLP with early_stopping=True)
    label_encoder = LabelEncoder().fit(y_train)
 
    print(f"[seed={seed}] Training shape: {x_train.shape}, Test shape: {x_test.shape}")
 
    return (x_train, x_test, y_train, y_test, label_encoder)

def load(seed):
    x, y = load_raw()
    return split_and_preprocess(x, y, seed)