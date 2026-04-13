import pandas as pd
import numpy as np
from imblearn.under_sampling import ClusterCentroids, NearMiss

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

def sample_random(x_transformed, y, majority_class, reduction_level, random_state):
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
        return None, None, None

    seed = np.random.default_rng(random_state) # FIXME really better to pass seed or generate in every sampling function???
    majority_indices = seed.choice(len(y_maj), size=n_majority_target, replace=False)

    x_out = np.concatenate([x_min, x_maj[majority_indices]])
    y_out = pd.concat([y_min, y_maj.iloc[majority_indices]])
    actual_reduction = len(y_out) / len(y)

    return x_out, y_out, actual_reduction

def sample_cluster_centroids(x_train_transformed, y_train_labels, majority_class, reduction_level, random_state):
    """imbalanced-learn ClusterCentroids"""
    n_majority_target = calculate_majority_target(y_train_labels, majority_class, reduction_level)
    if n_majority_target is None:
        return None, None, None
    
    undersampler = ClusterCentroids(sampling_strategy={majority_class: n_majority_target}, random_state=random_state)
    x_undersampled, y_undersampled_array = undersampler.fit_resample(x_train_transformed, y_train_labels)

    y_undersampled = pd.Series(y_undersampled_array)
    actual_reduction = len(y_undersampled) / len(y_train_labels)

    return x_undersampled, y_undersampled, actual_reduction

def sample_nearmiss(x_train_transformed, y_train_labels, majority_class, reduction_level, random_state):
    """imbalanced-learn NearMiss-1"""
    n_majority_target = calculate_majority_target(y_train_labels, majority_class, reduction_level)
    if n_majority_target is None:
        return None, None, None
    
    undersampler = NearMiss(sampling_strategy={majority_class: n_majority_target}, n_jobs=-1) # TODO does this allow for a fair comparison?
    x_undersampled, y_undersampled_array = undersampler.fit_resample(x_train_transformed, y_train_labels)

    y_undersampled = pd.Series(y_undersampled_array)
    actual_reduction = len(y_undersampled) / len(y_train_labels)

    return x_undersampled, y_undersampled, actual_reduction

# define samplers
SAMPLER_LIST = [
    ("Random", sample_random),
    ("ClusterCentroids", sample_cluster_centroids),
    ("NearMiss", sample_nearmiss)
]