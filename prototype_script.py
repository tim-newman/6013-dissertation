import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import time
from xgboost import XGBClassifier
import argparse

def calculate_majority_target(y, majority_class, reduction_level):
    """Calculate how many majority samples we need to keep to hit a target dataset size.
    
    Args:
        y: The target labels.
        majority_class: The label of the majority class (to be undersampled).
        reduction_level: Fraction of original dataset size as a decimal - the "target"
            (e.g., 0.5 = 50%)
            
    Returns:
        The number of majority samples to retain, or None if minority ALONE is bigger than the target size.
    """
    n_total = len(y)
    n_minority = (y != majority_class).sum()
    n_target = int(n_total * reduction_level)
    n_majority_target = n_target - n_minority

    if n_majority_target < 0:
        return None # it's impossible!
    
    return n_majority_target

def random_undersample(x_transformed, y, majority_class, reduction_level, random_state):
    """Randomly undersamples the majority class to hit a target dataset size.
    
    Args:
        x_transformed: Input features as map
        y: The target labels.
        majority_class: The label of the majority class (to be undersampled).
        reduction_level: Fraction of original dataset size as a decimal - the "target"
            (e.g., 0.5 = 50%)
        random_state: Seed for the random number generator.
        
    Returns:
        A tuple containing (x_out, y_out, actual_reduction).
        actual_reduction is the final size as a fraction of the original dataset.
        Returns (None, None, None) if the reduction level is impossible.
    """
    minority_mask = (y != majority_class)
    x_min = x_transformed[minority_mask]
    y_min = y[minority_mask]
    x_maj = x_transformed[~minority_mask]
    y_maj = y[~minority_mask]

    n_majority_target = calculate_majority_target(y, majority_class, reduction_level)

    if n_majority_target == None:
        return (None, None, None)

    seed = np.random.default_rng(random_state) # FIXME really better to pass seed or generate in every sampling function???
    majority_indices = seed.choice(len(y_maj), size=n_majority_target, replace=False)

    x_out = np.concatenate([x_min, x_maj[majority_indices]])
    y_out = pd.concat([y_min, y_maj.iloc[majority_indices]])
    actual_reduction = len(y_out) / len(y)

    return x_out, y_out, actual_reduction

# i've always wanted to do this in python. handle CLI:
parser = argparse.ArgumentParser()
parser.add_argument("--classifier", "-c",
                    choices=["RandomForest", "LogisticRegression", "XGBoost", "SGD", "MLP"],
                    nargs="+",
                    default=None)
args = parser.parse_args()

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

# pre-process
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

# define classifiers
# note that class_weight="balanced" is NOT picked as a parameter.
# weighting & undersampling both tackle imbalance so to keep a clean comparison, we leave it out for this experiment
classifiers_list = [
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
    ("XGBoost", XGBClassifier(random_state=42)),
    ("SGD", SGDClassifier(random_state=42)),
    ("MLP", MLPClassifier(random_state=42))
]

if args.classifier:
    classifiers = []
    for name, classifier in classifiers_list:
        if name in args.classifier:
            classifiers.append((name, classifier))
else:
    classifiers = classifiers_list


majority_class = "Normal"
reduction_levels = [1.00, 0.75, 0.50, 0.25, 0.10] # as per methodology. skipping 25% and 10% for now as it leads to reducing majority class to smaller than the minority
results = []

for classifier_name, classifier in classifiers:
    print(f"\nCLASSIFIER: {classifier_name}")

    for level in reduction_levels:  # TODO in real experiment, this may make it too difficult to utilise more than one machine
        if level == 1.00:
            print("  Training on full dataset!")
            x_sampled, y_sampled = x_train_transformed, y_train
            actual_size = level
        else:
            print(f"  Applying random undersampling (target: {level * 100}% reduction)!")
            x_sampled, y_sampled, actual_size = random_undersample(x_train_transformed, y_train, majority_class, level, random_state=42) # FIXME THIS DOES NOT BELONG HERE. IT'S A GLOBAL SEED CONSISTENT ACROSS THE ENTIRE PROJECT.

            if x_sampled is None:   # TODO spidey senses are tingling with this level of indentation. some would lament the use of "continue", too!
                print("  SKIPPING REDUCTION LEVEL. Minority classes alone are bigger than the desired set size.")
                results.append({
                    "classifier": classifier_name,
                    "reduction_level": level,
                    "actual_size": None,
                    "dataset_size": None,
                    "train_time": None,
                    "inference_time": None,
                    "macro_f1": None
                })
                continue

            print(f"    Target: {level * 100}%, Actual: {actual_size * 100}%, Literal size: {len(y_sampled)}")

        # XGBoost breaks with string labels. thus, numeric encoding:
        if classifier_name == "XGBoost":
            label_encoder = LabelEncoder().fit(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            y_fit = label_encoder.transform(y_sampled)
            y_eval = y_test_encoded
        else:
            y_fit = y_sampled
            y_eval = y_test

        start = time.time()
        classifier.fit(x_sampled, y_fit) # duck typing is fantastic
        train_time = time.time() - start

        # might as well test inference timing
        # TODO are there more accurate libraries for this?
        start = time.time()
        y_pred = classifier.predict(x_test_transformed)

        if classifier_name == "XGBoost":
            y_pred = label_encoder.inverse_transform(y_pred) # TODO icl this feels stupid to have to do
        inference_time = time.time() - start

        f1 = f1_score(y_test, y_pred, average="macro", zero_division="warn")  #better for debugging than silent fails at this stage
        results.append({
            "classifier": classifier_name,
            "reduction_level": level,
            "actual_size": actual_size,
            "dataset_size": len(y_sampled),
            "train_time": train_time,
            "inference_time": inference_time,
            "macro_f1": f1
        })
        print(f"  Train = {train_time:.2f}s, Infer = {inference_time:.4f}s, Macro F1 = {f1:.4f}")

results_df = pd.DataFrame(results)
print("\nResults:")
print(results_df.to_string(index=False))