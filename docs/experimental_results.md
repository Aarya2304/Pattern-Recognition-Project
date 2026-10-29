# Experimental results

All values below are computed from the actual UCI HAR archive and saved output files under `results/metrics/`. The official test partition was not used for fitting, feature selection, PCA, early stopping, or hyperparameter selection.

## Held-out official test performance

| Representation/model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Fit time (s) | Test inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original features — linear SVM | 0.9610 | 0.9629 | 0.9600 | 0.9608 | 0.9609 | 2.770 | 0.930 |
| Original features — Random Forest | 0.9291 | 0.9302 | 0.9259 | 0.9271 | 0.9288 | 13.484 | 0.160 |
| Raw signals — 1D CNN | 0.8904 | 0.8929 | 0.8935 | 0.8912 | 0.8910 | 59.359 | 0.396 |
| Custom statistics/spectra — SVM | 0.9332 | 0.9338 | 0.9339 | 0.9337 | 0.9330 | 1.969 | 2.916 |
| Custom statistics/spectra — Random Forest | 0.9284 | 0.9299 | 0.9281 | 0.9287 | 0.9283 | 9.579 | 0.201 |

The linear SVM with the original 561-dimensional representation achieved the highest test macro F1 (0.9608). The next highest was the SVM over 228 custom descriptors (0.9337). The tested CNN did not outperform the classical engineered-feature systems. The forest had the shortest full-test prediction time of the three principal classifiers in this environment. Figure comparisons and complete class metrics are in `results/figures/14_classifier_comparison.png`, `results/figures/15_representation_comparison.png`, and `results/metrics/model_comparison.csv`.

## Subject-independent model selection

Three-fold `StratifiedGroupKFold`, grouping by participant among 21 official training participants, was used for selection. The selected linear SVM (C=1) had mean group-CV macro F1 0.9172 (SD 0.0143); the selected 160-tree Random Forest (unrestricted depth, minimum leaf 2, `max_features="sqrt"`) had 0.9057 (SD 0.0141). The SVM candidates were linear/RBF with C=1/10 (0.9172, 0.9169, 0.9128, 0.9168). Forest candidates were 120 trees/unrestricted/leaf 1/sqrt (0.9036), 200 trees/depth 20/leaf 1/max_features 0.5 (0.8879), and 160 trees/unrestricted/leaf 2/sqrt (0.9057). The full search is in `results/metrics/group_cv_hyperparameter_search.csv`, with its graphic in `results/figures/22_group_cv_comparison.png`. A single participant-held-out CNN validation split achieved macro F1 0.9631 at the selected first epoch, distinct from the test result.

## Feature selection and PCA

The following results are mean macro-F1 across three participant-group folds on official training data. Scalers, selectors and PCA fit independently within each fold.

| Method | Dimensions | RBF SVM F1 | SVM SD | SVM mean fit (s) | Random Forest F1 | RF SD | RF mean fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 561 | 0.9168 | 0.0101 | 5.894 | 0.9044 | 0.0157 | 4.759 |
| ANOVA | 25 | 0.8403 | 0.0364 | 1.715 | 0.8006 | 0.0277 | 1.394 |
| ANOVA | 50 | 0.8518 | 0.0557 | 1.402 | 0.8243 | 0.0354 | 1.730 |
| ANOVA | 100 | 0.8954 | 0.0225 | 1.204 | 0.8594 | 0.0385 | 2.227 |
| ANOVA | 200 | **0.9223** | 0.0152 | 1.268 | 0.8948 | 0.0192 | 2.836 |
| Mutual information | 100 | 0.8933 | 0.0139 | 26.068 | 0.8830 | 0.0256 | 27.172 |
| PCA | 50 | 0.8846 | 0.0145 | 1.733 | 0.8331 | 0.0290 | 2.406 |
| PCA | 100 | 0.9051 | 0.0163 | 2.470 | 0.8379 | 0.0338 | 3.641 |
| PCA | 200 | 0.9153 | 0.0092 | 3.805 | 0.8425 | 0.0336 | 5.663 |

ANOVA with 200 features produced the strongest evaluated selector CV result for the SVM; its 0.9223 macro F1 was 0.0055 above the all-feature RBF score (0.9168) with a lower mean fold fit time (1.268 s versus 5.894 s). No tested reduced representation exceeded the all-feature Random Forest score (0.9044). PCA-200 came within 0.0015 of the all-feature SVM score with fewer components but reduced forest performance (0.8425). At 25 or 50 variables, ANOVA scores declined markedly. MI-100 scored 0.8933, about 0.0236 below the all-feature SVM, and had by far the largest fit time because mutual information is recomputed inside every fold. Figure 10 plots all classifier/method combinations; Figure 7 shows PCA variance, Figure 8 a two-dimensional training projection, and Figure 9 training MI rankings.

## Class errors and interpretation

SVM test per-class F1 was 0.9773 WALKING, 0.9628 WALKING_UPSTAIRS, 0.9708 WALKING_DOWNSTAIRS, 0.9205 SITTING, 0.9337 STANDING and 1.0000 LAYING. Forest per-class F1 was 0.9377, 0.9049, 0.9024, 0.9036, 0.9140 and 1.0000 in the same order. CNN F1 was lower for SITTING (0.7964) and STANDING (0.8111) than for the dynamic activities (0.9404, 0.9223, 0.9030) and LAYING (0.9742). Confusion matrices are provided for SVM, forest and CNN; Figure 21 aggregates common off-diagonal errors. These scores describe the official test predictions only and do not establish a cause. Forest impurity importance and top-40 MI-restricted permutation importance (calculated on training-held-out participants) are in `feature_importance.csv`. CNN fixed-window occlusion is computed on validation participants only in `cnn_occlusion_importance.csv`; neither importance method is causal.

## Representation and computational trade-offs

The custom descriptor SVM macro F1 was 0.9337, 0.0272 below the original-feature SVM. The custom descriptor forest reached 0.9287, slightly above the original-feature forest (0.9271). The raw nine-channel CNN reached 0.8912. On the recorded CPU run, principal SVM fitting took 2.770 s and full test inference 0.930 s; forest fitting was 13.484 s with 0.160 s full test prediction. CNN training took 59.359 s and measured full-test inference 0.396 s. These timings depend on CPU, framework and batch size and are not mobile latency measurements. Figures 3–6 show representative signal/feature analysis, and Figures 16–17 the CNN's training/validation curves.

## Provenance and reproducibility

The official source, validated partitions, feature/signal dimensions and disjoint participant lists are recorded in `results/metrics/dataset_summary.json`. Seed 42 was used where supported. Principal metric CSVs, class reports, predictions and model artifacts are saved in results. Run `python main.py` to recreate all full-mode stages; the resumable stage commands reproduce the same metrics for their stage. Full experiment results are distinct from reduced `--mode smoke` runs.