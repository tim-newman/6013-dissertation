# column names from the kdd docs
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

TRAIN_PATH = "./nsl-kdd/KDDTrain+.txt"
TEST_PATH = "./nsl-kdd/KDDTest+.txt"

MAJORITY_CLASS = "Normal"   # this is used for duck typing in experiment.py

COLUMNS = [
'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
'label',      # attack type (e.g., normal, neptune, etc.)
'difficulty'  # difficulty level (see 10.1109/CISDA.2009.5356528)
]

# reduced these - turns out nobody trains on full labels!
# official repo gives 4 attack "categories"
# 10.1038/s41598-026-38317-w, 10.1002/spy2.56 suggest this is normal
LABEL_MAP = {
'normal': 'Normal',
'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'mailbomb': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS',
'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'saint': 'Probe', 'mscan': 'Probe',
'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'xlock': 'R2L',
'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 'xsnoop': 'R2L',
'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L',
'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}

def load():
    """Load and pre-process NSL-KDD train/test data.
    
    Returns:
        x_train_transformed, x_test_transformed,
        y_train_labels, y_test_labels,
        label_encoder"""
    # drop difficulty and label
    train_raw = pd.read_csv(TRAIN_PATH, names=COLUMNS).drop("difficulty", axis=1)
    test_raw = pd.read_csv(TEST_PATH, names=COLUMNS).drop("difficulty", axis=1)

    x_train_raw, y_train_labels = train_raw.drop("label", axis=1), train_raw["label"]
    x_test_raw, y_test_labels = test_raw.drop("label", axis=1), test_raw["label"]

    y_train_labels = y_train_labels.map(LABEL_MAP)
    y_test_labels = y_test_labels.map(LABEL_MAP)
 
    cat_cols = x_train_raw.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = x_train_raw.select_dtypes(include=[np.number]).columns.tolist()
 
    # fit on training only, then transform both
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )
    x_train_transformed = preprocessor.fit_transform(x_train_raw)
    x_test_transformed = preprocessor.transform(x_test_raw)
 
    # fitting LabelEncoder here for XGBoost
    label_encoder = LabelEncoder().fit(y_train_labels)
 
    print(f"Training data shape after preprocessing: {x_train_transformed.shape}")
    print(f"Test data shape after preprocessing: {x_test_transformed.shape}")
    print("Unique labels in y_train:", y_train_labels.unique())
    print("Count of 'normal' in y_train:", (y_train_labels == 'Normal').sum())

    return (x_train_transformed, x_test_transformed, y_train_labels, y_test_labels, label_encoder)