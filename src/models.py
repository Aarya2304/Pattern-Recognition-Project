"""Classical estimator definitions and the compact raw-signal CNN."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_svm(kernel: str = "rbf", c: float = 10.0, gamma: str | float = "scale", k: int | str = "all") -> Pipeline:
    steps = [("scale", StandardScaler())]
    if k != "all":
        steps.append(("select", SelectKBest(score_func=f_classif, k=k)))
    steps.append(("classifier", SVC(kernel=kernel, C=c, gamma=gamma, class_weight=None, cache_size=1000)))
    return Pipeline(steps)


def make_random_forest(n_estimators: int = 400, max_depth: int | None = None,
                       min_samples_leaf: int = 1, max_features: str | float = "sqrt",
                       k: int | str = "all", random_state: int = 42) -> Pipeline:
    steps = []
    if k != "all":
        steps.extend([("scale", StandardScaler()), ("select", SelectKBest(score_func=f_classif, k=k))])
    steps.append(("classifier", RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        max_features=max_features, n_jobs=-1, random_state=random_state,
        class_weight="balanced_subsample", oob_score=False)))
    return Pipeline(steps)


def build_cnn(n_channels: int, n_classes: int = 6):
    """Create a small PyTorch 1D CNN; model input is (batch, time, channels)."""
    import torch
    from torch import nn
    # Disable oneDNN CPU kernels: stable behavior across minimal CPU-only environments.
    torch.backends.mkldnn.enabled = False

    class HARConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(96), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(96, n_classes))

        def forward(self, x):
            return self.classifier(self.features(x.transpose(1, 2)))

    return HARConvNet()
