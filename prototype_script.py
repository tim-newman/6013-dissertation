import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import time

# column names from the kdd docs
columns = [
'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
'label',      # attack type (e.g., normal, neptune, etc.)
'difficulty'  # difficulty level (see 10.1109/CISDA.2009.5356528)
]

train_path = "./nsl-kdd/KDDTrain+.txt"
test_path = "./nsl-kdd/KDDTest+.txt"

train_df = pd.read_csv(train_path, names=columns)
test_df = pd.read_csv(test_path, names=columns)

# drop difficulty and label
train_df = train_df.drop("difficulty", axis=1)
test_df = test_df.drop("difficulty", axis=1)
x_train = train_df.drop("label", axis=1)
y_train = train_df["label"]
x_test = test_df.drop("label", axis=1)
y_test = test_df["label"]

# reduced these - turns out nobody trains on full labels!
# official repo gives 4 attack "categories"
# 10.1038/s41598-026-38317-w, 10.1002/spy2.56 suggest this is normal
label_map = {
'normal': 'Normal',
'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'mailbomb': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS',
'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'saint': 'Probe', 'mscan': 'Probe',
'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'xlock': 'R2L',
'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 'xsnoop': 'R2L',
'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L',
'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}
y_train = y_train.map(label_map)
y_test = y_test.map(label_map)

# pre-process (ColumnTransformer)
# Identify column types (for NSL-KDD, they are object dtype)
cat_cols = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
]
)

# fit on training only, then transform both
x_train_transformed = preprocessor.fit_transform(x_train)
x_test_transformed = preprocessor.transform(x_test)

print(f"Training data shape after preprocessing: {x_train_transformed.shape}")
print(f"Test data shape after preprocessing: {x_test_transformed.shape}")

print("Unique labels in y_train:", y_train.unique())
print("Count of 'normal' in y_train:", (y_train == 'Normal').sum())

# train on full dataset (baseline)
print("Training on full dataset!")
start = time.time()
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(x_train_transformed, y_train)
train_acc = rf_full.score(x_train_transformed, y_train)
train_time_full = time.time() - start

y_pred_full = rf_full.predict(x_test_transformed)
print(classification_report(y_test, y_pred_full, zero_division=0))
f1_macro_full = f1_score(y_test, y_pred_full, average="macro")
print(f"Full data - Train time: {train_time_full:.2f}s, Macro F1: {f1_macro_full:.4f}")

# random undersampling at 50% reduction
print("\nApplying random undersampling (50% reduction)!")
'''Create a random undersampler that reduced majority class to match minority size
There has to be a better way to do this - we're currently culling a % of majority to take down overall set size'''

# identify majority and minority classes (here, 'normal' is majority)
majority_class = "Normal"
minority_mask = (y_train != majority_class)
x_minority = x_train_transformed[minority_mask]
y_minority = y_train[minority_mask]

x_majority = x_train_transformed[~minority_mask]
y_majority = y_train[~minority_mask]

# Sample 50% (or whatever it turns out to be) of majority
np.random.default_rng(42)
indices = np.random.choice(len(x_majority), size=int(0.1 * len(x_majority)), replace=False)
x_majority_sampled = x_majority[indices]
y_majority_sampled = y_majority.iloc[indices]

# Combine
x_train_sampled = np.concatenate([x_minority, x_majority_sampled])
y_train_sampled = pd.concat([y_minority, y_majority_sampled])

print(f"Original size: {len(x_train)}, after sampling: {len(x_train_sampled)} (reduction {1 - len(x_train_sampled)/len(x_train):.1%})")

start = time.time()
rf_sampled = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sampled.fit(x_train_sampled, y_train_sampled)
train_time_sampled = time.time() - start

y_pred_sampled = rf_sampled.predict(x_test_transformed)
f1_macro_sampled = f1_score(y_test, y_pred_sampled, average="macro")
print(f"Sampled data - Train time: {train_time_sampled:.2f}s, Macro F1: {f1_macro_sampled:.4f}")

# quick comparison
print("\nComparison:")
print(f"  Full data:      Time = {train_time_full:.2f}s, Macro F1 = {f1_macro_full:.4f}")
print(f"  Undersampled:   Time = {train_time_sampled:.2f}s, Macro F1 = {f1_macro_sampled:.4f}")