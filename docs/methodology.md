# Experimental methodology

## Dataset and alignment

The pipeline loads the official UCI HAR Version 1.0 train/test partitions, including 561 named engineered features, integer activity labels, subject identifiers, and nine inertial axes. Each partition is checked for dimensions, finite values, valid labels and participant IDs, and row alignment. The official archive README and `features.txt` are inspected by the loader. The official train participants (21 people) and test participants (nine people) are checked to be disjoint. The source recording is sampled at 50 Hz; the supplied signal files comprise overlapping 128-reading windows.

## Leakage control and validation

The official test partition is reserved for final performance estimates and is not used for tuning, feature selection, PCA fitting, importance ranking or CNN early stopping. Classical hyperparameter candidates are ranked by mean macro F1 over three stratified group folds (two in smoke mode), with participant ID as the group. Every fold independently fits its scaler, selector and PCA before its validation scores are computed. The CNN reserves whole participants from the official training partition for validation; per-channel mean and standard deviation are fitted on its training participants only. Early stopping uses validation macro F1, and restores the best checkpoint. The reported CNN test result is from the final official held-out test partition.

## Feature representations and signal processing

Three representations are evaluated. First, the supplied 561-dimensional official feature vector. Second, 228 descriptors calculated directly for each nine-channel window, including per-axis distributional and magnitude statistics, zero crossings, correlations, FFT peak, spectral power, entropy, centroid and frequency-band power. Third, the 128-by-9 time series is supplied to the 1D CNN, which learns temporal filters directly. Exploratory waveforms and single-sided FFT magnitude spectra are generated from the actual training partition.

## Classical model selection

The SVM uses a `StandardScaler` and SVC in one pipeline. Linear and RBF kernels, with C=1 and C=10, are assessed under group CV. The random forest explores tree count, depth, minimum leaf size and split feature count. The selected configuration for each family maximizes mean group-CV macro F1. The reported final pipelines are then refit on the full official training partition and evaluated once on the official test partition. Feature selectors (`SelectKBest` ANOVA and mutual information) and scaled PCA are also included inside fold-local pipelines for group-CV comparisons.

## CNN

The compact PyTorch network applies three 1D convolutional blocks with batch normalization and ReLU; the first two blocks use max pooling, the final block uses global average pooling, followed by dropout and a six-logit dense layer. Cross entropy is optimized with Adam and weight decay. The output classes are the six official IDs (internal zero-based labels are used for the loss). The CPU execution disables oneDNN because that backend terminated abnormally in the execution environment; native PyTorch convolution passed shape validation and training.

## Measurements and interpretability

Accuracy, macro precision/recall/F1, weighted F1, class-level metrics, confusion matrices, model fit wall-clock time and full-test prediction wall-clock time are exported. SVM and forest timing is measured around fit and predict. CNN training includes the train/validation loop; inference timing is a separate full test-batch pass. Forest impurity importance is reported; permutation importance is computed on a group-held-out training fold for the 40 highest mutual-information features, limiting runtime and avoiding test leakage. CNN input occlusion measures the change in mean true-class log probability when each sensor/time segment is zeroed on held-out validation subjects; this is a model-dependence diagnostic, not a causal explanation.

## Reproducibility

The seed is 42 for supported estimators, splitters and PyTorch. Configuration defaults are in `configs/experiment_config.yaml`. The default `python main.py` runs the full workflow. For resource-limited runs, `--mode smoke` is explicitly a reduced search/training path. `--stage baselines`, `--stage features`, `--stage representations`, `--stage cnn`, and `--stage finish` allow experiments to persist independently. Generated metrics, model artifacts, predictions and figures reside in `results/` and are excluded from version control by default; regenerate them from the official data and code.
