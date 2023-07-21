from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

# AutoML for feature selection
def permutation_importance_feature_selection(X, y, model=RandomForestClassifier(), num_features=None):
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=10)
    sorted_idx = result.importances_mean.argsort()
    
    if num_features is not None:
        sorted_idx = sorted_idx[-num_features:]
    
    return X[:, sorted_idx]