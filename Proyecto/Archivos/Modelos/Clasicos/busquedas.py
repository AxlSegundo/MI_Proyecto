from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _pick_search(estimator, param_grid, scoring="f1", cv=5, n_jobs=-1):

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    if n_combos <= 25:
        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            refit=True,
        )
    else:
        return HalvingGridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            factor=2,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )


def search_svm_linear(X, y):
    params = {
        "C": [0.01, 0.1, 1, 10, 100]
    }
    est = LinearSVC()
    search = _pick_search(est, params, scoring="f1", cv=5)
    search.fit(X, y)
    return search


def search_logreg(X, y):
    params = {
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"],
        "max_iter": [200]
    }
    est = LogisticRegression(solver="lbfgs", n_jobs=-1)
    search = _pick_search(est, params, scoring="f1", cv=5)
    search.fit(X, y)
    return search


def search_rf(X, y):
    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    est = RandomForestClassifier()
    search = _pick_search(est, params, scoring="f1", cv=5)
    search.fit(X, y)
    return search
