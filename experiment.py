from sklearnex import patch_sklearn
patch_sklearn()

import argparse
import time
import pandas as pd
from sklearn.metrics import f1_score
from classifiers import build_classifiers
from samplers import SAMPLER_LIST

RANDOM_STATE = 42   # used throughout the experiment

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
args = parser.parse_args()

# pick dataset
if args.dataset == "NSL-KDD":
    import preprocessing.nsl_kdd as dataset_module
elif args.dataset == "CICIDS2017":
    import preprocessing.cicids2017 as dataset_module

(x_train_transformed, x_test_transformed, y_train_labels, y_test_labels, label_encoder) = dataset_module.load()

# pick classifiers
classifier_list = build_classifiers(RANDOM_STATE)
if args.classifier:
    classifiers = []
    for name, classifier in classifier_list:
        if name in args.classifier:
            classifiers.append((name, classifier))
else:
    classifiers = classifier_list

# pick samplers
if args.sampler:
    samplers = []
    for name, sampler in SAMPLER_LIST:
        if name in args.sampler:
            samplers.append((name, sampler))
else:
    samplers = SAMPLER_LIST


#----------Experiment----------
reduction_levels = [1.00, 0.75, 0.50, 0.25, 0.10] # as per methodology. skipping 25% and 10% for now as it leads to reducing majority class to smaller than the minority
results = []

for sampler_name, sampler_function in samplers:
    print(f"\nSAMPLER: {sampler_name}")

    for reduction_level in reduction_levels:
        #sample only once
        if reduction_level == 1.00:
            print(f"Training on full dataset ({args.dataset})!\n")
            x_train_sampled, y_train_sampled = x_train_transformed, y_train_labels
            actual_reduction = reduction_level
            sample_time = 0
        else:
            print(f"\nAPPLYING {sampler_name} (target: {reduction_level * 100}% reduction)!")
            # TODO JUST FOR TESTING:
            start = time.time()
            x_train_sampled, y_train_sampled, actual_reduction = sampler_function(x_train_transformed, y_train_labels, reduction_level, RANDOM_STATE)
            sample_time = time.time() - start

            if x_train_sampled is None:   # TODO spidey senses are tingling with this level of indentation. some would lament the use of "continue", too!
                print("SKIPPING REDUCTION LEVEL. Minority classes alone are bigger than the desired set size.")
                results.append({
                    "sampler": sampler_name,
                    "reduction_level": reduction_level,
                    "classifier": classifier_name,
                    "actual_reduction": None,
                    "dataset_size": None,
                    "train_time": None,
                    "inference_time": None,
                    "macro_f1": None
                })
                continue

            print(f"    Target: {reduction_level * 100}%, Actual: {actual_reduction * 100}%, Literal size: {len(y_train_sampled)}")
        
        for classifier_name, classifier in classifiers:
            print(f"Classifier: {classifier_name}")

            # XGBoost breaks with string labels. thus, numeric encoding:
            if classifier_name == "XGBoost":
                y_train_fit = label_encoder.transform(y_train_sampled)
            else:
                y_train_fit = y_train_sampled

            start = time.time()
            classifier.fit(x_train_sampled, y_train_fit) # duck typing is fantastic
            train_time = time.time() - start

            # might as well test inference timing
            # TODO are there more accurate libraries for this?
            start = time.time()
            y_test_predictions = classifier.predict(x_test_transformed)

            inference_time = time.time() - start

            if classifier_name == "XGBoost":
                y_test_predictions = label_encoder.inverse_transform(y_test_predictions)

            f1 = f1_score(y_test_labels, y_test_predictions, average="macro", zero_division="warn")  #better for debugging than silent fails at this stage
            results.append({
                "sampler": sampler_name,
                "reduction_level": reduction_level,
                "classifier": classifier_name,
                "actual_reduction": actual_reduction,
                "dataset_size": len(y_train_sampled),
                "train_time": train_time,
                "inference_time": inference_time,
                "macro_f1": f1
            })
            print(f"  Sample = {sample_time:.2f}s, Train = {train_time:.2f}s, Infer = {inference_time:.4f}s, Macro F1 = {f1:.4f}")

    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df.to_string(index=False))