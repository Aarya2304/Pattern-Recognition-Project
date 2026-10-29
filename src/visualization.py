"""Publication-quality plotting for UCI HAR experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_loader import ACTIVITIES, SIGNAL_CHANNELS
from .evaluation import LABEL_NAMES

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, directory, name):
    path = Path(directory); path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path / name, dpi=220, bbox_inches="tight"); plt.close(fig)


def exploration_figures(train, feature_names, figure_dir):
    fig, ax = plt.subplots(figsize=(8, 4)); counts = pd.Series(train["y"]).map(ACTIVITIES).value_counts().reindex(LABEL_NAMES)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#35618f"); ax.set(title="Training observations by activity", xlabel="Activity", ylabel="Windows"); ax.tick_params(axis="x", rotation=25); _save(fig, figure_dir, "01_activity_distribution.png")
    subject_counts = pd.Series(train["subjects"]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(subject_counts.index, subject_counts.values, color="#4c956c"); ax.set(title="Training windows by participant", xlabel="Participant ID", ylabel="Windows"); _save(fig, figure_dir, "02_participant_distribution.png")
    selected = ["total_acc_x", "body_acc_x", "body_gyro_x"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, name in zip(axes, selected):
        channel = SIGNAL_CHANNELS.index(name)
        for activity_id in (1, 2, 4, 6):
            idx = np.flatnonzero(train["y"] == activity_id)[0]
            ax.plot(np.arange(128) / 50, train["signals"][idx, :, channel], label=ACTIVITIES[activity_id])
        ax.set_ylabel(name); ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Time (s)"); fig.suptitle("Representative sensor windows by activity"); _save(fig, figure_dir, "03_representative_signals.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    for activity_id in (1, 2, 4, 6):
        idx = np.flatnonzero(train["y"] == activity_id)[0]
        sig = train["signals"][idx, :, SIGNAL_CHANNELS.index("body_acc_x")]
        freq = np.fft.rfftfreq(len(sig), d=1 / 50); mag = np.abs(np.fft.rfft(sig - sig.mean()))
        ax.plot(freq, mag, label=ACTIVITIES[activity_id])
    ax.set(xlim=(0, 10), xlabel="Frequency (Hz)", ylabel="FFT magnitude", title="Body-acceleration spectrum (representative windows)"); ax.legend(); _save(fig, figure_dir, "04_frequency_spectra.png")
    numeric = pd.DataFrame(train["X"][:2000], columns=feature_names)
    sample_features = list(feature_names[:6])
    fig, ax = plt.subplots(figsize=(11, 5)); long = numeric[sample_features].melt(var_name="Feature", value_name="Value"); sns.boxplot(data=long, x="Feature", y="Value", ax=ax, showfliers=False); ax.tick_params(axis="x", rotation=35); ax.set_title("Distributions of selected engineered features"); _save(fig, figure_dir, "05_feature_distributions.png")
    corr = numeric.iloc[:, :30].corr()
    fig, ax = plt.subplots(figsize=(11, 9)); sns.heatmap(corr, cmap="vlag", center=0, ax=ax, xticklabels=False, yticklabels=False); ax.set_title("Correlation heatmap: first 30 official features"); _save(fig, figure_dir, "06_feature_correlation.png")
    sample = train["X"][::max(1, len(train["X"]) // 1500)]
    sample_y = train["y"][::max(1, len(train["X"]) // 1500)]
    scaled = StandardScaler().fit_transform(sample)
    pca = PCA(n_components=2, random_state=42).fit(scaled)
    projection = pca.transform(scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    for activity_id in ACTIVITIES:
        mask = sample_y == activity_id; ax.scatter(projection[mask, 0], projection[mask, 1], s=9, alpha=.45, label=ACTIVITIES[activity_id])
    ax.set(title="Two-component PCA projection (training sample)", xlabel="PC1", ylabel="PC2"); ax.legend(fontsize=8); _save(fig, figure_dir, "08_pca_projection.png")


def model_figures(metrics_rows, reports, matrices, feature_names, importance, figure_dir):
    metrics = pd.DataFrame(metrics_rows)
    slug = lambda name: {"1D CNN": "cnn"}.get(name, name.lower().replace(" ", "_"))
    for name, cm in matrices.items():
        fig, ax = plt.subplots(figsize=(7, 6)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax); ax.set(title=f"{name}: official test confusion matrix", xlabel="Predicted activity", ylabel="True activity"); ax.tick_params(axis="x", rotation=35); _save(fig, figure_dir, f"{slug(name)}_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=metrics, x="model", y="macro_f1", ax=ax, color="#35618f"); ax.set(ylim=(0, 1), title="Held-out test macro-F1 comparison", xlabel="Model", ylabel="Macro F1"); ax.tick_params(axis="x", rotation=20); _save(fig, figure_dir, "14_classifier_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 5)); plot = metrics.melt(id_vars="model", value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1"], var_name="Metric", value_name="Score"); sns.barplot(data=plot, x="model", y="Score", hue="Metric", ax=ax); ax.set(ylim=(0,1), title="Test-set metric comparison"); ax.tick_params(axis="x", rotation=20); _save(fig, figure_dir, "15_representation_comparison.png")
    per_class = []
    for model, report in reports.items():
        for label in LABEL_NAMES: per_class.append({"model": model, "activity": label, "f1": report[label]["f1-score"]})
    fig, ax = plt.subplots(figsize=(11, 5)); sns.barplot(data=pd.DataFrame(per_class), x="activity", y="f1", hue="model", ax=ax); ax.set(ylim=(0,1), title="Per-class F1 on the official test set", xlabel="Activity", ylabel="F1 score"); ax.tick_params(axis="x", rotation=25); _save(fig, figure_dir, "18_per_class_f1.png")
    time_cols = [c for c in ("training_seconds", "inference_seconds") if c in metrics and metrics[c].notna().all()]
    if time_cols:
        fig, ax = plt.subplots(figsize=(9,5)); plot=metrics.melt(id_vars="model",value_vars=time_cols,var_name="Phase",value_name="Seconds"); sns.barplot(data=plot,x="model",y="Seconds",hue="Phase",ax=ax); ax.set(title="Measured fit and test inference time",ylabel="Seconds"); ax.tick_params(axis="x",rotation=20); _save(fig,figure_dir,"19_model_times.png")
    if importance is not None:
        top = np.argsort(importance)[-20:][::-1]
        fig, ax = plt.subplots(figsize=(9, 7)); ax.barh(np.asarray(feature_names)[top][::-1], np.asarray(importance)[top][::-1], color="#4c956c"); ax.set(title="Random Forest impurity importance (top 20)", xlabel="Mean decrease in impurity"); _save(fig, figure_dir, "20_rf_feature_importance.png")
    # Most common off-diagonal activity errors across available principal models.
    pairs = []
    for model, cm in matrices.items():
        normalized = cm.copy(); np.fill_diagonal(normalized, 0)
        for i, j in zip(*np.unravel_index(np.argsort(normalized.ravel())[-8:], normalized.shape)):
            if normalized[i, j]: pairs.append({"model": model, "true": LABEL_NAMES[i], "predicted": LABEL_NAMES[j], "errors": normalized[i,j]})
    if pairs:
        frame = pd.DataFrame(pairs).groupby(["true","predicted"],as_index=False)["errors"].sum().nlargest(10,"errors")
        fig, ax = plt.subplots(figsize=(9,5)); frame["pair"] = frame["true"] + " → " + frame["predicted"]; ax.barh(frame["pair"][::-1],frame["errors"][::-1],color="#c55a4a"); ax.set(title="Most frequent off-diagonal errors (summed across models)",xlabel="Misclassified windows"); _save(fig,figure_dir,"21_common_errors.png")


def feature_experiment_figures(rows, mi_scores, feature_names, pca_variance, figure_dir):
    if mi_scores is not None:
        top = np.argsort(mi_scores)[-25:][::-1]
        fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(np.asarray(feature_names)[top][::-1], np.asarray(mi_scores)[top][::-1]); ax.set(title="Top 25 mutual-information scores (training data)", xlabel="Mutual information (nats)"); _save(fig,figure_dir,"09_mutual_information.png")
    if rows:
        frame=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(10,5)); frame["configuration"]=frame["method"]+" / "+frame["model"]
        sns.barplot(data=frame,x="retained_features",y="macro_f1",hue="configuration",ax=ax); ax.set(title="Feature reduction: group-CV macro-F1",xlabel="Retained features/components",ylabel="Macro F1"); ax.legend(title="Method / classifier",fontsize=7,ncol=2); _save(fig,figure_dir,"10_feature_selection_performance.png")
    if pca_variance is not None:
        cumulative=np.cumsum(pca_variance); fig,ax=plt.subplots(figsize=(8,4)); ax.plot(np.arange(1,len(cumulative)+1),cumulative); ax.axhline(.95,color="red",linestyle="--",label="95% variance"); ax.set(title="PCA cumulative explained variance (training fit)",xlabel="Principal components",ylabel="Cumulative explained variance",ylim=(0,1.02)); ax.legend(); _save(fig,figure_dir,"07_pca_explained_variance.png")
