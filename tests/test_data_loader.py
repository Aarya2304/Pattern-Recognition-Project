import numpy as np
import pytest

from src.data_loader import SIGNAL_CHANNELS, find_dataset_root, load_dataset


def test_official_dataset_loads_and_partitions_are_disjoint():
    if find_dataset_root("data") is None:
        pytest.skip("Official UCI dataset not downloaded/placed")
    dataset = load_dataset("data", download=False)
    assert dataset["train"]["X"].shape == (7352, 561)
    assert dataset["test"]["X"].shape == (2947, 561)
    assert dataset["train"]["signals"].shape == (7352, 128, 9)
    assert dataset["test"]["signals"].shape == (2947, 128, 9)
    assert len(SIGNAL_CHANNELS) == 9
    assert not set(dataset["train"]["subjects"]) & set(dataset["test"]["subjects"])
    # The official features.txt repeats some names (477 unique of 561); columns stay positional.
    assert len(dataset["feature_names"]) == 561
    for split in ("train", "test"):
        assert np.isfinite(dataset[split]["X"]).all()
        assert np.isfinite(dataset[split]["signals"]).all()
        assert set(np.unique(dataset[split]["y"])) == set(range(1, 7))
