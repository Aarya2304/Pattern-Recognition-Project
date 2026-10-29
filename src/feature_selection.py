"""Training-only feature rankings and group-aware selector experiments."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from .evaluation import group_splits


def training_mutual_information(X, y, feature_names, random_state=42):
    scores = mutual_info_classif(X, y, random_state=random_state, n_neighbors=3)
    table = pd.DataFrame({"feature": feature_names, "mutual_information": scores}).sort_values("mutual_information", ascending=False)
    return scores, table


def selector_cv_experiments(X, y, subjects, random_state=42, smoke=False, model_names=None):
    """Compare ANOVA k-best, MI-k-best, PCA and all features inside group CV folds."""
    n_splits = 2 if smoke else 3
    folds = list(group_splits(y, subjects, n_splits=n_splits, random_state=random_state))
    max_k = X.shape[1]
    ks = sorted(set(k for k in ([25, 100, max_k] if smoke else [25, 50, 100, 200, max_k]) if k <= max_k))
    pca_dims = [min(k, max_k, len(y) - 1) for k in ([50, 100] if smoke else [50, 100, 200])]
    configs = [("all", "all")]
    configs += [("anova", k) for k in ks if k != max_k]
    configs += [("mutual_information", k) for k in (ks if smoke else [100]) if k < max_k]
    configs += [("pca", k) for k in sorted(set(pca_dims))]
    rows = []
    selected_models = model_names or ("SVM-RBF", "Random Forest")
    for method, dimension in configs:
        for model_name in selected_models:
            scores, times = [], []
            for train_idx, valid_idx in folds:
                start = time.perf_counter()
                if method == "pca":
                    prep = [("scale", StandardScaler()), ("pca", PCA(n_components=dimension, svd_solver="randomized", random_state=random_state))]
                elif method == "mutual_information":
                    # Selector fit stays inside each training fold; MI is computed only on fold training rows.
                    from sklearn.feature_selection import mutual_info_classif
                    prep = [("scale", StandardScaler()), ("select", SelectKBest(lambda a, b: mutual_info_classif(a, b, random_state=random_state), k=dimension))]
                elif method == "anova":
                    prep = [("scale", StandardScaler()), ("select", SelectKBest(f_classif, k=dimension))]
                else:
                    prep = [("scale", StandardScaler())]
                estimator = SVC(C=10, kernel="rbf", gamma="scale", cache_size=512) if model_name == "SVM-RBF" else RandomForestClassifier(n_estimators=50 if smoke else 80, max_features="sqrt", min_samples_leaf=1, n_jobs=-1, random_state=random_state, class_weight="balanced_subsample")
                pipe = Pipeline(prep + [("classifier", estimator)])
                pipe.fit(X[train_idx], y[train_idx])
                pred = pipe.predict(X[valid_idx])
                scores.append(f1_score(y[valid_idx], pred, average="macro"))
                times.append(time.perf_counter() - start)
            rows.append({"method": method, "model": model_name, "retained_features": int(dimension) if dimension != "all" else X.shape[1], "macro_f1": float(np.mean(scores)), "macro_f1_std": float(np.std(scores)), "cv_fit_seconds_mean": float(np.mean(times))})
    return rows
