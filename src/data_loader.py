"""Acquisition and loading for the official UCI HAR archive."""
from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ACTIVITIES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}
SIGNAL_CHANNELS = (
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
)


class DatasetValidationError(ValueError):
    """Raised when local dataset contents do not match the UCI HAR layout."""


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise DatasetValidationError(f"Unsafe archive path: {member.filename}")
    archive.extractall(destination)


def find_dataset_root(data_dir: str | Path) -> Path | None:
    base = Path(data_dir)
    candidates = [base / "UCI HAR Dataset", base / "raw" / "UCI HAR Dataset"]
    candidates.extend(base.glob("**/UCI HAR Dataset"))
    for candidate in candidates:
        if all((candidate / p).is_file() for p in ("features.txt", "train/X_train.txt", "test/X_test.txt")):
            return candidate
    return None


def ensure_dataset(data_dir: str | Path, download: bool = True) -> Path:
    """Find a manually placed dataset or download/extract UCI's nested ZIP."""
    base = Path(data_dir)
    root = find_dataset_root(base)
    if root:
        return root
    raw = base / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    outer = raw / "UCI_HAR_Dataset.zip"
    if not outer.exists():
        local_zips = list(raw.glob("*.zip")) + list(base.glob("*.zip"))
        if local_zips:
            outer = local_zips[0]
        elif download:
            request = urllib.request.Request(UCI_URL, headers={"User-Agent": "har-research/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response, outer.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except Exception as exc:
                raise FileNotFoundError(
                    f"UCI download failed ({exc}). Place the official UCI HAR Dataset directory "
                    f"or its ZIP archive under {raw}. Source: {UCI_URL}"
                ) from exc
        else:
            raise FileNotFoundError(
                f"Official UCI HAR dataset not found. Place extracted 'UCI HAR Dataset' "
                f"or the official ZIP under {raw}; source: {UCI_URL}"
            )
    extracted = raw / "archive_extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(outer) as archive:
            _safe_extract(archive, extracted)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetValidationError(f"Cannot read official archive {outer}: {exc}") from exc
    root = find_dataset_root(extracted)
    if root:
        return root
    nested = list(extracted.rglob("UCI HAR Dataset.zip"))
    for inner in nested:
        target_dir = inner.parent / "dataset_contents"
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(inner) as archive:
                _safe_extract(archive, target_dir)
        except (OSError, zipfile.BadZipFile) as exc:
            raise DatasetValidationError(f"Cannot read nested archive {inner}: {exc}") from exc
        root = find_dataset_root(target_dir)
        if root:
            return root
    raise DatasetValidationError(f"Official archive extracted, but expected UCI HAR files were not found under {extracted}")


def load_split(root: str | Path, split: str) -> dict:
    """Load one official split; signals have shape (windows, 128, channels)."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(root)
    folder = root / split
    try:
        x = np.loadtxt(folder / f"X_{split}.txt", dtype=np.float32)
        labels = np.loadtxt(folder / f"y_{split}.txt", dtype=np.int64).reshape(-1)
        subjects = np.loadtxt(folder / f"subject_{split}.txt", dtype=np.int64).reshape(-1)
        signals = np.stack([
            np.loadtxt(folder / "Inertial Signals" / f"{channel}_{split}.txt", dtype=np.float32)
            for channel in SIGNAL_CHANNELS
        ], axis=-1)
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(f"Unable to load official {split} partition: {exc}") from exc
    validate_split(x, labels, subjects, signals, split)
    return {"X": x, "y": labels, "subjects": subjects, "signals": signals}


def validate_split(x: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                   signals: np.ndarray, split: str = "data") -> None:
    if x.ndim != 2 or x.shape[1] != 561:
        raise DatasetValidationError(f"{split}: expected engineered feature shape (n, 561), got {x.shape}")
    n = len(x)
    if y.shape != (n,) or subjects.shape != (n,) or signals.shape != (n, 128, 9):
        raise DatasetValidationError(
            f"{split}: misaligned dimensions features={x.shape}, labels={y.shape}, "
            f"subjects={subjects.shape}, signals={signals.shape}"
        )
    if not np.isfinite(x).all() or not np.isfinite(signals).all():
        raise DatasetValidationError(f"{split}: non-finite feature or signal values detected")
    if not set(np.unique(y)).issubset(ACTIVITIES):
        raise DatasetValidationError(f"{split}: labels outside official activity IDs 1..6")
    if np.any(subjects < 1) or np.any(subjects > 30):
        raise DatasetValidationError(f"{split}: subject identifiers must be within 1..30")


def load_dataset(data_dir: str | Path = "data", download: bool = True) -> dict:
    root = ensure_dataset(data_dir, download=download)
    features = pd.read_csv(root / "features.txt", sep=r"\s+", header=None, names=["index", "name"])
    if len(features) != 561 or not np.array_equal(features["index"].to_numpy(), np.arange(1, 562)):
        raise DatasetValidationError(f"Expected 561 consecutively indexed feature names in features.txt; found {len(features)}")
    train, test = load_split(root, "train"), load_split(root, "test")
    overlap = sorted(set(train["subjects"]) & set(test["subjects"]))
    if overlap:
        raise DatasetValidationError(f"Official train/test participant sets overlap: {overlap}")
    names = features["name"].astype(str).to_numpy()
    result = {"root": root, "feature_names": names, "train": train, "test": test}
    summary = dataset_summary(result)
    (Path(data_dir) / "processed").mkdir(parents=True, exist_ok=True)
    (Path(data_dir) / "processed" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return result


def dataset_summary(dataset: dict) -> dict:
    train, test = dataset["train"], dataset["test"]
    return {
        "source": "UCI Machine Learning Repository, dataset 240",
        "source_url": "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones",
        "root": str(dataset["root"]), "feature_count": int(train["X"].shape[1]),
        "signal_channels": list(SIGNAL_CHANNELS), "window_shape": list(train["signals"].shape[1:]),
        "sampling_hz": 50, "activity_names": ACTIVITIES,
        "train_windows": int(len(train["y"])), "test_windows": int(len(test["y"])),
        "train_subjects": sorted(map(int, np.unique(train["subjects"]))),
        "test_subjects": sorted(map(int, np.unique(test["subjects"]))),
        "class_counts_train": {ACTIVITIES[i]: int((train["y"] == i).sum()) for i in ACTIVITIES},
        "class_counts_test": {ACTIVITIES[i]: int((test["y"] == i).sum()) for i in ACTIVITIES},
    }
