"""Leakage-aware splits and consistent model evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from .data_loader import ACTIVITIES

LABEL_IDS = np.array(list(ACTIVITIES))
LABEL_NAMES = [ACTIVITIES[i] for i in LABEL_IDS]


def group_splits(y, groups, n_splits: int = 5, random_state: int = 42):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train, valid in splitter.split(np.zeros(len(y)), y, groups):
        if set(np.asarray(groups)[train]) & set(np.asarray(groups)[valid]):
            raise RuntimeError("Group leakage detected")
        yield train, valid


def evaluate_predictions(y_true, y_pred, model_name: str, train_seconds: float | None = None,
                         inference_seconds: float | None = None) -> tuple[dict, dict, np.ndarray]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    report = classification_report(y_true, y_pred, labels=LABEL_IDS,
                                   target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    metrics = {
        "model": model_name, "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0)),
        "training_seconds": None if train_seconds is None else float(train_seconds),
        "inference_seconds": None if inference_seconds is None else float(inference_seconds),
        "inference_ms_per_window": None if inference_seconds is None or len(y_true) == 0 else float(inference_seconds * 1000 / len(y_true)),
    }
    for activity in LABEL_NAMES:
        metrics[f"{activity.lower()}_precision"] = float(report[activity]["precision"])
        metrics[f"{activity.lower()}_recall"] = float(report[activity]["recall"])
        metrics[f"{activity.lower()}_f1"] = float(report[activity]["f1-score"])
    return metrics, report, cm


def save_evaluation(metrics: dict, report: dict, y_true, y_pred,
                    prefix: str, out_dir: str | Path) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / f"{prefix}_metrics.csv", index=False)
    (out / f"{prefix}_classification_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out / f"{prefix}_predictions.csv", index=False)
