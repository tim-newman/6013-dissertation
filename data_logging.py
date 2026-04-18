import csv
import json
import os
import socket
from pathlib import Path
from datetime import datetime
import numpy as np

FIELDNAMES = [
    # config
    "timestamp",
    "machine_id",
    "dataset",
    "sampler",
    "reduction_level",
    "seed",
    "classifier",
    # sampling results
    "actual_reduction",
    "dataset_size",
    "per_class_counts",
    "sample_time",
    # training results
    "train_time",
    "inference_time",
    # metrics
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "balanced_accuracy",
    "per_class_f1",
    "per_class_precision",
    "per_class_recall",
    "confusion_matrix",
    "class_labels",
    # status
    "status",  # "ok", "skipped_infeasible", "error"
    "error_message",
]

def append_result(row, path):
    """Add one row to the CSV. Creates the file with headers if it doesn't exist yet."""
    file_already_exists = Path(path).exists()

    # fill in any missing columns with None so the writer doesn't complain
    full_row = {}
    for column in FIELDNAMES:
        full_row[column] = row.get(column)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_already_exists:
            writer.writeheader()
        writer.writerow(full_row)
        f.flush()           # push to OS buffer
        os.fsync(f.fileno())  # push OS buffer to actual disk

def load_completed_runs(path):
    """Look at an existing CSV and return the set of runs already done successfully.
    Used to skip work we've already completed."""
    if not Path(path).exists():
        return set()

    completed = set()
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok":
                continue   # don't skip failed or skipped runs. they should retry
            key = (
                row["dataset"],
                row["sampler"],
                float(row["reduction_level"]),
                int(row["seed"]),
                row["classifier"],
            )
            completed.add(key)
    return completed

def get_machine_id():
    return socket.gethostname()

def get_timestamp():
    return datetime.now().isoformat()

def convert(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    print(f"Error on {x}")

def dump_json(obj):
    return json.dumps(obj, default=convert)