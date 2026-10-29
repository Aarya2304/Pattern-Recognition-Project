# Advanced Smartphone-Sensor Human Activity Recognition

A reproducible pattern-recognition study comparing engineered-feature SVM and Random Forest models with a raw-signal 1D CNN on the official UCI HAR dataset. The project includes validated data acquisition/loading, signal exploration, statistical/spectral feature extraction, subject-aware cross-validation, feature selection/PCA, final held-out test evaluation, interpretability, figures, tests and an academic report.

## Problem and dataset

Classify each sensor window as WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING or LAYING. The source is the [UCI Machine Learning Repository dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Its original partition supplies 7,352 training and 2,947 test windows, 561 engineered variables and nine 128-sample inertial channels per window. Sampling frequency is 50 Hz. The full official dataset is downloaded automatically when possible; see [data/README.md](data/README.md) for offline placement. Dataset files are excluded from version control.

## Methods

- **SVM:** standardized linear/RBF SVC pipelines; grouped macro-F1 cross-validation tunes C and kernel.
- **Random Forest:** participant-group-selected estimators varying tree count, depth, leaf size and max features; exports impurity and group-held-out permutation importance.
- **1D CNN:** PyTorch temporal convolution network over nine axes with train-only normalization, participant-held-out validation, early stopping and saved best checkpoint.
- **Feature analysis:** 228 per-window statistics and spectral/correlation descriptors; ANOVA and mutual-information selection; scaled PCA; group-aware fold-local experiments.
- **Evaluation:** official held-out test metrics, class reports, confusion matrices, fit/predict timings, predictions and error analysis.

The executed results and report are in [docs/project_report.md](docs/project_report.md) and [docs/experimental_results.md](docs/experimental_results.md). Results reflect the actual experiment outputs; no synthetic examples are substituted.

## Installation and execution

Python 3.10 or later is required.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python main.py --mode validate
python main.py                 # complete full workflow
python main.py --mode smoke    # reduced smoke run; not final research results
python -m pytest -q
```

The archive download can be skipped by placing the extracted official `UCI HAR Dataset/` directory or UCI ZIP in `data/raw/`. The main CLI supports `--data-dir` and `--results-dir`. Resumable full-mode commands support `--stage baselines`, `--stage features`, `--stage representations`, `--stage cnn`, and `--stage finish`; the default full workflow executes all stages. `--stage finish` consolidates saved outputs; use it only after the model/representation/CNN stages have completed. Stage commands overwrite the corresponding stage's outputs.

## Project organization

```text
configs/       Reproducible experiment defaults
src/           Dataset, signal, feature, models, training, metrics and plots
main.py        Dataset validation and experiment CLI
tests/         Lightweight automated unit tests
data/          Dataset attribution and local raw/processed data (ignored)
results/       Generated figures, metrics, model artifacts and predictions
docs/          Formal report, methodology, theory, results and limitations
```

## Outputs

`results/metrics/` includes dataset validation, group-CV hyperparameter search, selector/PCA scores, classification reports and comparisons. `results/figures/` contains measured signal/model graphics. `results/models/` contains the fitted classical pipelines and PyTorch checkpoint. `results/predictions/` contains per-window official-test predictions. All generated artifacts can be recreated with the commands above.

## Validation strategy and limitations

The official test participants are disjoint from training. Hyperparameter selection, selectors, PCA and importance analyses use training data only. `StratifiedGroupKFold` prevents participant overlap across classical CV folds; CNN validation also holds out whole training participants. The fixed official test partition is never used for tuning. The study remains limited by the small, scripted 30-volunteer benchmark and one official participant-held-out test. The full selector grid is constrained to SVM for runtime; see [limitations](docs/limitations.md).

## References

- Reyes-Ortiz et al. (2013), UCI HAR dataset record, https://doi.org/10.24432/C54S4K.
- Anguita et al. (2013), A public domain dataset for human activity recognition using smartphones, ESANN 2013.
- Cortes & Vapnik (1995), Support-vector networks, https://doi.org/10.1007/BF00994018.
- Breiman (2001), Random Forests, https://doi.org/10.1023/A:1010933404324.
- Full references and methods are listed in [the project report](docs/project_report.md).
