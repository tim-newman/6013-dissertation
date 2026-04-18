from collections import Counter

from sklearnex import patch_sklearn
patch_sklearn()

import argparse
import time
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from classifiers import build_classifiers
from samplers import SAMPLER_LIST
import data_logging

# i've always wanted to do this in python. handle CLI:
parser = argparse.ArgumentParser()
parser.add_argument("--classifier", "-c",
                    choices=["RandomForest", "LogisticRegression", "XGBoost", "SGD", "MLP"],
                    nargs="+",
                    default=None)
parser.add_argument("--sampler", "-s",
                    choices=["Random", "ClusterCentroids", "NearMiss", "Density"],
                    nargs="+",
                    default=None)
parser.add_argument("--dataset", "-d",
                    choices=["NSL-KDD", "CICIDS2017"],
                    default="NSL-KDD")
parser.add_argument("--reduction", "-r", type=float, nargs="+", default=None)
parser.add_argument("--seed", type=int, nargs="+", default=None)
parser.add_argument("--output", "-o", default=None)
args = parser.parse_args()

# defaults
DEFAULT_REDUCTION_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]

# pick dataset
if args.dataset == "NSL-KDD":
    import preprocessing.nsl_kdd as dataset_module
elif args.dataset == "CICIDS2017":
    import preprocessing.cicids2017 as dataset_module

(x_train_transformed, x_test_transformed, y_train_labels, y_test_labels, label_encoder) = dataset_module.load()

# pick classifiers
classifier_list = build_classifiers(DEFAULT_SEEDS[0]) # TODO awkward artefact in need of refactor. don't actually want to build them right now
if args.classifier:
    classifier_names = []
    for name, classifier in classifier_list:
        if name in args.classifier:
            classifier_names.append(name)
else:
    classifier_names = []
    for name, classifier in classifier_list:
        classifier_names.append(name)

# pick samplers
if args.sampler:
    samplers = []
    for name, sampler in SAMPLER_LIST:
        if name in args.sampler:
            samplers.append((name, sampler))
else:
    samplers = SAMPLER_LIST

# pick reduction levels
if args.reduction:
    reduction_levels = args.reduction
else:
    reduction_levels = DEFAULT_REDUCTION_LEVELS

# pick seeds
if args.seed:
    seeds = args.seed
else:
    seeds = DEFAULT_SEEDS

# pick output path
if args.output:
    output_path = args.output
else:
    output_path = "output.csv"

#----------Experiment----------
results = []
class_labels = sorted(set(y_train_labels) | set(y_test_labels))

for sampler_name, sampler_function in samplers:
    print(f"\nSAMPLER: {sampler_name}")

    for reduction_level in reduction_levels:
        for seed in seeds:
            #sample only once (per sampler, reduction, seed)
            if reduction_level == 1.00:
                print(f"Training on full dataset ({args.dataset})!\n")
                x_train_sampled, y_train_sampled = x_train_transformed, y_train_labels
                actual_reduction = reduction_level
                sample_time = 0
            else:
                print(f"\nAPPLYING {sampler_name} (target: {reduction_level * 100}% reduction, seed: {seed})!")
                start = time.time()
                x_train_sampled, y_train_sampled, actual_reduction = sampler_function(x_train_transformed, y_train_labels, reduction_level, seed)
                sample_time = time.time() - start

                if x_train_sampled is None:   # TODO spidey senses are tingling with this level of indentation. some would lament the use of "continue", too!
                    print("SKIPPING REDUCTION LEVEL. Minority classes alone are bigger than the desired set size.")
                    for classifier_name in classifier_names:
                        skipped_row_info = {
                            "timestamp": data_logging.get_timestamp(),
                            "machine_id": data_logging.get_machine_id(),
                            "dataset": args.dataset,
                            "sampler": sampler_name,
                            "reduction_level": reduction_level,
                            "seed": seed,
                            "classifier": classifier_name,
                            "status": "Skipped - impossible reduction level."
                        }
                        data_logging.append_result(skipped_row_info, output_path)
                        results.append(skipped_row_info)
                    continue

                print(f"    Target: {reduction_level * 100}%, Actual: {actual_reduction * 100}%, Literal size: {len(y_train_sampled)}")

            per_class_counts = dict(Counter(y_train_sampled))
            
            for classifier_name in classifier_names:
                print(f"Classifier: {classifier_name}")

                try:    # crashes on long runs would irk me
                    # TODO unfortunately wrote classifiers.py to be stupid, and now have to do this:
                    classifier = None
                    for name, classy in build_classifiers(seed):
                        if name == classifier_name:
                            classifier = classy
                            break
                    
                    # XGBoost breaks with string labels. thus, numeric encoding:
                    if classifier_name == "XGBoost":
                        y_train_fit = label_encoder.transform(y_train_sampled)
                    else:
                        y_train_fit = y_train_sampled

                    start = time.time()
                    classifier.fit(x_train_sampled, y_train_fit) # duck typing is fantastic
                    train_time = time.time() - start

                    # might as well test inference timing
                    start = time.time()
                    y_test_predictions = classifier.predict(x_test_transformed)

                    inference_time = time.time() - start

                    if classifier_name == "XGBoost":
                        y_test_predictions = label_encoder.inverse_transform(y_test_predictions)

                    # give me all the metrics
                    f1 = f1_score(y_test_labels, y_test_predictions, average="macro", zero_division="warn")  #better for debugging than silent fails at this stage
                    weighted_f1 = f1_score(y_test_labels, y_test_predictions, average="weighted", zero_division=0)
                    accuracy = accuracy_score(y_test_labels, y_test_predictions)
                    balanced_accuracy = balanced_accuracy_score(y_test_labels, y_test_predictions)
                    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                        y_test_labels, y_test_predictions, labels=class_labels, zero_division=0
                    )
                    confusion_mat = confusion_matrix(y_test_labels, y_test_predictions, labels=class_labels)

                    row = {
                        "timestamp": data_logging.get_timestamp(),
                            "machine_id": data_logging.get_machine_id(),
                            "dataset": args.dataset,
                            "sampler": sampler_name,
                            "reduction_level": reduction_level,
                            "seed": seed,
                            "classifier": classifier_name,
                            "actual_reduction": actual_reduction,
                            "dataset_size": len(y_train_sampled),
                            "per_class_counts": data_logging.to_json(per_class_counts),
                            "sample_time": sample_time,
                            "train_time": train_time,
                            "inference_time": inference_time,
                            "macro_f1": f1,
                            "weighted_f1": weighted_f1,
                            "accuracy": accuracy,
                            "balanced_accuracy": balanced_accuracy,
                            "per_class_f1": data_logging.to_json(dict(zip(class_labels, f1_per_class))),
                            "per_class_precision": data_logging.to_json(dict(zip(class_labels, precision_per_class))),
                            "per_class_recall": data_logging.to_json(dict(zip(class_labels, recall_per_class))),
                            "confusion_matrix": data_logging.to_json(confusion_mat),
                            "class_labels": data_logging.to_json(class_labels),
                            "status": "ok"
                    }
                    data_logging.append_result(row, output_path)
                    results.append(row)

                    print(f"  Sample = {sample_time:.2f}s, Train = {train_time:.2f}s, Infer = {inference_time:.4f}s, Macro F1 = {f1:.4f}")

                except Exception as e:
                    print(e)
                    error_row_info = {
                        "timestamp": data_logging.get_timestamp(),
                        "machine_id": data_logging.get_machine_id(),
                        "dataset": args.dataset,
                        "sampler": sampler_name,
                        "reduction_level": reduction_level,
                        "seed": seed,
                        "classifier": classifier_name,
                        "status": f"ERROR {e}",
                    }
                    data_logging.append_result(error_row_info, output_path)
                    results.append(error_row_info)

# go summary
if results:
    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df.to_string(index=False))