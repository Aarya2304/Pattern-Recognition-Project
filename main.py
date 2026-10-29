"""Run reproducible UCI HAR experiments."""
from __future__ import annotations
import argparse
import json
from src.experiments import run_experiments
from src.data_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["full", "smoke", "validate"], default="full")
    parser.add_argument("--stage", choices=["all", "baselines", "classical", "features", "representations", "cnn", "finish"], default="all", help="Run one resumable stage; full mode by default runs the complete workflow.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    if args.mode == "validate":
        dataset = load_dataset(args.data_dir, download=True)
        print(json.dumps({"root": str(dataset["root"]), "train": dataset["train"]["X"].shape, "test": dataset["test"]["X"].shape, "signal_train": dataset["train"]["signals"].shape, "signal_test": dataset["test"]["signals"].shape, "train_subjects": sorted(map(int, set(dataset["train"]["subjects"]))), "test_subjects": sorted(map(int, set(dataset["test"]["subjects"])))}, indent=2))
        return
    stages = None if args.stage == "all" else [args.stage]
    run_experiments(args.data_dir, args.results_dir, args.mode, stages=stages)

if __name__ == "__main__":
    main()
