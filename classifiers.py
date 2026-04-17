from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def build_classifiers(random_state):
    """Generate classifiers. Done here to keep a single source of truth for "random state".
    
    Returns: Full list of (name, classifier_object) tuples."""
    # note that class_weight="balanced" is NOT picked as a parameter.
    # weighting & undersampling both tackle imbalance so to keep a clean comparison, we leave it out for this experiment
    # TODO literature standard parameters for chosen models on CICIDS2017?
    return [
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
    ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=random_state)),
    ("XGBoost", XGBClassifier(random_state=random_state, n_jobs=-1)),
    ("SGD", SGDClassifier(random_state=random_state, n_jobs=-1)),
    ("MLP", MLPClassifier(random_state=random_state))
]