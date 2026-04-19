import pandas as pd
import numpy as np
from imblearn.under_sampling import ClusterCentroids, NearMiss, RandomUnderSampler
from sklearn.cluster import MiniBatchKMeans

def calculate_undersampling_targets(y, reduction_level):
    """Calculate how many per-class samples we need to keep to hit a target dataset size.

    This approach attempts to cap the top 'k' classes at a single common value 'c', and leave every other class alone.
    
    'Stream of consciousness' solution approach:
    - Calculate target dataset size 't': With reduction level 'r' (as a fraction) and total dataset samples 's', t=rs.
    - Iterate over k values, starting at 1:
        - Calculate 'c': (t - sum(untouched_classes_samples)) / k
        - Check c: 
            In cases where c is small, like needing heavy reduction with only k=1,
            the untouched clasess end up bigger than the capped ones. This wouldn't make sense.
            Thus, check c>=size[k].
    - As soon as we find a valid c value, we can return.

    Args:
        y: The target labels.
        reduction_level: Fraction of original dataset size as a decimal.
            (e.g., 0.7 = reduce dataset to 70% of its size)

    Returns:
        A dictionary mapping capped classes to the calculated c value.
        None if no valid k is possible.
    """
    n_total = len(y)
    t = int(reduction_level * n_total)
    unique_labels, label_counts = np.unique(y, return_counts=True)
    counts = {}
    for label, count in zip(unique_labels, label_counts):
        counts[label] = int(count)
    n_classes = len(counts)

    # could be impossible from the outset
    if t < n_classes:
        return None

    sorted_classes = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    labels = []
    sizes = []
    for label, size in sorted_classes:
        labels.append(label)
        sizes.append(size)

    for k in range(1, n_classes + 1):
        # sum sizes we AREN'T capping
        remaining_sum = 0
        for size in sizes[k:]:
            remaining_sum += size

        samples_left_to_distribute = t - remaining_sum

        if samples_left_to_distribute <= 0:
            continue
            
        c = samples_left_to_distribute / k

        size_of_kth_class = sizes[k-1]

        if k < n_classes:
            size_of_next_class = sizes[k]
        else:
            size_of_next_class = 0
        
        # c check
        c_is_valid = size_of_next_class <= c <= size_of_kth_class

        if c_is_valid:
            cap = int(c)
            result = {}
            for i in range(k):
                result[labels[i]] = cap
            return result
    
    return None # no k possible

    
def sample_random(x_transformed, y, reduction_level, random_state): # TODO lot of repetition in the sampler functions that could be extracted
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
    targets = calculate_undersampling_targets(y, reduction_level)
    if targets is None:
        return None, None, None

    undersampler = RandomUnderSampler(
        sampling_strategy=targets,
        random_state=random_state
    )
    x_undersampled, y_undersampled_array = undersampler.fit_resample(x_transformed, y)

    y_undersampled = pd.Series(y_undersampled_array)
    actual_reduction = len(y_undersampled) / len(y)

    return x_undersampled, y_undersampled, actual_reduction


def sample_cluster_centroids(x_train_transformed, y_train_labels, reduction_level, random_state):
    """imbalanced-learn ClusterCentroids"""
    targets = calculate_undersampling_targets(y_train_labels, reduction_level)
    if targets is None:
        return None, None, None

    km_estimator = MiniBatchKMeans(n_init=1, random_state=random_state, batch_size=4096)
    undersampler = ClusterCentroids(sampling_strategy=targets, estimator=km_estimator, random_state=random_state)
    x_undersampled, y_undersampled_array = undersampler.fit_resample(x_train_transformed, y_train_labels)

    y_undersampled = pd.Series(y_undersampled_array)
    actual_reduction = len(y_undersampled) / len(y_train_labels)

    return x_undersampled, y_undersampled, actual_reduction


def sample_nearmiss1(x_train_transformed, y_train_labels, reduction_level, random_state):
    """imbalanced-learn NearMiss-1"""
    targets = calculate_undersampling_targets(y_train_labels, reduction_level)
    if targets is None:
        return None, None, None

    undersampler = NearMiss(sampling_strategy=targets, n_jobs=-1)
    x_undersampled, y_undersampled_array = undersampler.fit_resample(x_train_transformed, y_train_labels)

    y_undersampled = pd.Series(y_undersampled_array)
    actual_reduction = len(y_undersampled) / len(y_train_labels)

    return x_undersampled, y_undersampled, actual_reduction

def sample_nearmiss2(x_train_transformed, y_train_labels, reduction_level, random_state):
    """imbalanced-learn NearMiss-2"""
    targets = calculate_undersampling_targets(y_train_labels, reduction_level)
    if targets is None:
        return None, None, None

    undersampler = NearMiss(sampling_strategy=targets, n_jobs=-1, version=2)
    x_undersampled, y_undersampled_array = undersampler.fit_resample(x_train_transformed, y_train_labels)

    y_undersampled = pd.Series(y_undersampled_array)
    actual_reduction = len(y_undersampled) / len(y_train_labels)

    return x_undersampled, y_undersampled, actual_reduction

def sample_nearmiss3(x_train_transformed, y_train_labels, reduction_level, random_state):
    """imbalanced-learn NearMiss-3"""
    targets = calculate_undersampling_targets(y_train_labels, reduction_level)
    if targets is None:
        return None, None, None

    undersampler = NearMiss(sampling_strategy=targets, n_jobs=-1, version=3)
    x_undersampled, y_undersampled_array = undersampler.fit_resample(x_train_transformed, y_train_labels)

    y_undersampled = pd.Series(y_undersampled_array)

    # bug for imblearn nearmiss-3 is reported that suggests it under-delivers sometimes when given sample_strategy dict?
    # check for it:
    delivered = dict(y_undersampled.value_counts())
    for label, target_count in targets.items():
        delivered_label = delivered.get(label, 0)
        if delivered_label < target_count:
            print(f"    NEARMISS UNDERDELIVERED {label}; TARGET {target_count} BUT DELIVERED {delivered_label}")

    actual_reduction = len(y_undersampled) / len(y_train_labels)

    return x_undersampled, y_undersampled, actual_reduction

# define samplers
SAMPLER_LIST = [
    ("Random", sample_random),
    ("ClusterCentroids", sample_cluster_centroids),
    ("NearMiss1", sample_nearmiss1),
    ("NearMiss2", sample_nearmiss2),
    ("NearMiss3", sample_nearmiss3)
]