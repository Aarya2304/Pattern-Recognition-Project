"""End-to-end, reproducible UCI HAR experiment orchestration."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_loader import ACTIVITIES, load_dataset
from .evaluation import evaluate_predictions, group_splits, save_evaluation
from .feature_engineering import extract_window_features
from .feature_selection import training_mutual_information, selector_cv_experiments
from .models import make_random_forest, make_svm
from .training import train_cnn, perturbation_importance
from .visualization import exploration_figures, feature_experiment_figures, model_figures

SEED = 42


def _score_cv(factory, X, y, folds):
    scores = []
    for train_idx, valid_idx in folds:
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[valid_idx])
        scores.append(float(f1_score(y[valid_idx], pred, average="macro")))
    return float(np.mean(scores)), float(np.std(scores))


def _tune_main(X, y, groups, smoke=False):
    folds = list(group_splits(y, groups, n_splits=2 if smoke else 3, random_state=SEED))
    svm_grid = [("linear", 1.0, "scale"), ("linear", 10.0, "scale"), ("rbf", 1.0, "scale"), ("rbf", 10.0, "scale")]
    forest_grid = [(120, None, 1, "sqrt"), (200, 20, 1, 0.5), (160, None, 2, "sqrt")]
    if smoke:
        svm_grid = [("linear", 1.0, "scale"), ("rbf", 1.0, "scale")]
        forest_grid = [(60, None, 1, "sqrt"), (80, 16, 2, 0.5)]
    rows = []
    for kernel, c, gamma in svm_grid:
        score, spread = _score_cv(lambda: make_svm(kernel, c, gamma), X, y, folds)
        rows.append({"model": "SVM", "kernel": kernel, "C": c, "gamma": gamma, "cv_macro_f1": score, "cv_macro_f1_std": spread})
        print(f"Group-CV SVM {kernel} C={c:g}: macro-F1={score:.4f}", flush=True)
    best_svm = max((r for r in rows if r["model"] == "SVM"), key=lambda r: r["cv_macro_f1"])
    for trees, depth, leaf, features in forest_grid:
        score, spread = _score_cv(lambda: make_random_forest(trees, depth, leaf, features, random_state=SEED), X, y, folds)
        rows.append({"model": "Random Forest", "n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features, "cv_macro_f1": score, "cv_macro_f1_std": spread})
        print(f"Group-CV RF trees={trees} depth={depth} leaf={leaf} max_features={features}: macro-F1={score:.4f}", flush=True)
    best_rf = max((r for r in rows if r["model"] == "Random Forest"), key=lambda r: r["cv_macro_f1"])
    return best_svm, best_rf, rows


def _fit_evaluate(model_name, model, X_train, y_train, X_test, y_test, out, rows, reports, matrices):
    start = time.perf_counter(); model.fit(X_train, y_train); training_seconds = time.perf_counter() - start
    start = time.perf_counter(); predictions = model.predict(X_test); inference_seconds = time.perf_counter() - start
    metrics, report, matrix = evaluate_predictions(y_test, predictions, model_name, training_seconds, inference_seconds)
    save_evaluation(metrics, report, y_test, predictions, model_name.lower().replace(" ", "_"), out / "metrics")
    rows.append(metrics); reports[model_name] = report; matrices[model_name] = matrix
    print(f"{model_name}: test accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['macro_f1']:.4f}", flush=True)
    return model, predictions, metrics


def _save_feature_importance(train, names, rf, out, smoke=False, mi_scores=None):
    """Impurity ranking plus training-only permutation importance on top-MI variables."""
    y, groups = train["y"], train["subjects"]
    if mi_scores is None:
        mi_scores, mi_table = training_mutual_information(train["X"], y, names, SEED)
        mi_table.to_csv(out / "metrics" / "mutual_information_ranking.csv", index=False)
    fit_idx, valid_idx = next(group_splits(y, groups, n_splits=5, random_state=SEED))
    explain_indices = np.argsort(mi_scores)[-40:]
    explainer = RandomForestClassifier(n_estimators=80 if smoke else 150, max_features="sqrt", n_jobs=-1, random_state=SEED, class_weight="balanced_subsample")
    explainer.fit(train["X"][fit_idx][:, explain_indices], y[fit_idx])
    perm = permutation_importance(explainer, train["X"][valid_idx][:, explain_indices], y[valid_idx], scoring="f1_macro", n_repeats=2 if smoke else 3, random_state=SEED, n_jobs=-1)
    perm_mean = np.full(len(names), np.nan); perm_std = np.full(len(names), np.nan)
    perm_mean[explain_indices] = perm.importances_mean; perm_std[explain_indices] = perm.importances_std
    pd.DataFrame({"feature": names, "impurity_importance": rf.named_steps["classifier"].feature_importances_, "permutation_importance_mean_top40_mi": perm_mean, "permutation_importance_std_top40_mi": perm_std, "included_in_permutation_analysis": np.isin(np.arange(len(names)), explain_indices)}).sort_values("permutation_importance_mean_top40_mi", ascending=False, na_position="last").to_csv(out / "metrics" / "feature_importance.csv", index=False)


def run_stage(stages, dataset, out, mode, summary):
    """Run selected stages, or resume from complete primary model artifacts."""
    train, test, names = dataset["train"], dataset["test"], dataset["feature_names"]
    smoke = mode == "smoke"
    if stages == ["finish"]:
        # Resume final merge/plots using stored principal and custom-feature evaluations.
        import seaborn as sns
        metrics_frames=[]; reports={}; matrices={}; predictions={}; rows=[]
        for model_key, prefix in [("SVM","svm"),("Random Forest","random_forest"),("1D CNN","cnn"),("Custom features SVM","custom_features_svm"),("Custom features Random Forest","custom_features_random_forest")]:
            metrics_path=out/"metrics"/(prefix+"_metrics.csv")
            if not metrics_path.exists(): raise FileNotFoundError(f"Required saved metrics missing: {metrics_path}")
            metric=pd.read_csv(metrics_path).iloc[0].to_dict(); metric["model"]=model_key; rows.append(metric)
            reports[model_key]=json.loads((out/"metrics"/(prefix+"_classification_report.json")).read_text())
            pred_path=out/"metrics"/(prefix+"_predictions.csv")
            pred_frame=pd.read_csv(pred_path); pred=pred_frame["y_pred"].to_numpy()
            predictions[model_key]=pred
            _,_,matrices[model_key]=evaluate_predictions(test["y"],pred,model_key)
        cnn_report=reports["1D CNN"]
        cnn_history_path=out/"metrics"/"cnn_history.npz"
        if cnn_history_path.exists():
            import matplotlib.pyplot as plt
            history=np.load(cnn_history_path)
            for metric,title,ylabel,filename in [("loss","CNN training and validation loss","Cross-entropy loss","16_cnn_loss.png"),("accuracy","CNN training and validation accuracy","Accuracy","17_cnn_accuracy.png")]:
                fig,ax=plt.subplots(figsize=(8,4));ax.plot(history[metric],label="Train");ax.plot(history["val_"+metric],label="Validation");ax.set(title=title,xlabel="Epoch",ylabel=ylabel);ax.legend();fig.tight_layout();fig.savefig(out/"figures"/filename,dpi=220);plt.close(fig)
        model_figures(rows,reports,matrices,names,None,out/"figures")
        importance_path=out/"metrics"/"feature_importance.csv"
        if importance_path.exists():
            import matplotlib.pyplot as plt
            importance_frame=pd.read_csv(importance_path).sort_values("impurity_importance",ascending=False).head(20).sort_values("impurity_importance")
            fig,ax=plt.subplots(figsize=(9,7));ax.barh(importance_frame["feature"],importance_frame["impurity_importance"],color="#4c956c");ax.set(title="Random Forest impurity importance (top 20)",xlabel="Mean decrease in impurity");fig.tight_layout();fig.savefig(out/"figures"/"20_rf_feature_importance.png",dpi=220);plt.close(fig)
        pd.DataFrame(rows).to_csv(out/"metrics"/"model_comparison.csv",index=False)
        (out / "logs" / "run_summary.json").write_text(json.dumps({"mode": mode, "seed": SEED, "selection_metric": "macro F1", "group_cv_folds": 2 if smoke else 3, "stages_completed": ["baselines", "features", "representations", "cnn", "finish"], "figures_generated": len(list((out / "figures").glob("*.png"))), "execution_times": "results/metrics/*.csv (training_seconds, inference_seconds, cv_fit_seconds_mean)"}, indent=2))
        for model_key,pred in predictions.items():
            pd.DataFrame({"subject":test["subjects"],"true_activity":test["y"],"predicted_activity":pred,"true_name":pd.Series(test["y"]).map(ACTIVITIES),"predicted_name":pd.Series(pred).map(ACTIVITIES)}).to_csv(out/"predictions"/(model_key.lower().replace(" ","_")+".csv"),index=False)
        cv_rows=pd.read_csv(out/"metrics"/"group_cv_hyperparameter_search.csv"); cv_summary=cv_rows.groupby("model")["cv_macro_f1"].agg(["mean","std","max"]).reset_index(); cv_summary.to_csv(out/"metrics"/"subject_group_cv_summary.csv",index=False)
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));ax.bar(cv_summary["model"],cv_summary["max"],yerr=cv_summary["std"],capsize=4);ax.set(ylim=(0,1),title="Best group-CV macro-F1 by model family",ylabel="Macro F1");fig.tight_layout();fig.savefig(out/"figures"/"22_group_cv_comparison.png",dpi=220);plt.close(fig)
        return rows
    if stages == ["baselines"] or stages == ["classical"]:
        best_svm, best_rf, search_rows = _tune_main(train["X"], train["y"], train["subjects"], smoke)
        pd.DataFrame(search_rows).to_csv(out / "metrics" / "group_cv_hyperparameter_search.csv", index=False)
        (out / "metrics" / "selected_hyperparameters.json").write_text(json.dumps({"svm": best_svm, "random_forest": best_rf, "selection_metric": "mean subject-group cross-validation macro-F1", "seed": SEED}, indent=2))
        rows, reports, matrices = [], {}, {}
        svm, _, _ = _fit_evaluate("SVM", make_svm(best_svm["kernel"], best_svm["C"], best_svm["gamma"]), train["X"], train["y"], test["X"], test["y"], out, rows, reports, matrices)
        rf, _, _ = _fit_evaluate("Random Forest", make_random_forest(best_rf["n_estimators"], best_rf["max_depth"], best_rf["min_samples_leaf"], best_rf["max_features"], random_state=SEED), train["X"], train["y"], test["X"], test["y"], out, rows, reports, matrices)
        joblib.dump(svm, out / "models" / "svm_pipeline.joblib"); joblib.dump(rf, out / "models" / "random_forest_pipeline.joblib")
        pd.DataFrame(rows).to_csv(out / "metrics" / "model_comparison.csv", index=False)
        _save_feature_importance(train, names, rf, out, smoke)
        return rows
    if stages == ["features"]:
        mi_scores, mi_table = training_mutual_information(train["X"], train["y"], names, SEED)
        mi_table.to_csv(out / "metrics" / "mutual_information_ranking.csv", index=False)
        selection_rows = selector_cv_experiments(train["X"], train["y"], train["subjects"], SEED, smoke, model_names=("SVM-RBF", "Random Forest"))
        pd.DataFrame(selection_rows).to_csv(out / "metrics" / "feature_selection_group_cv.csv", index=False)
        pca_pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=min(250, len(train["y"]) - 1), svd_solver="randomized", random_state=SEED))])
        pca_pipe.fit(train["X"]); variance = pca_pipe.named_steps["pca"].explained_variance_ratio_
        pd.DataFrame({"component": np.arange(1, len(variance)+1), "explained_variance_ratio": variance, "cumulative_explained_variance": np.cumsum(variance)}).to_csv(out / "metrics" / "pca_explained_variance.csv", index=False)
        feature_experiment_figures(selection_rows, mi_scores, names, variance, out / "figures")
        return selection_rows
    if stages == ["representations"]:
        custom_train, feature_names = extract_window_features(train["signals"])
        custom_test, test_names = extract_window_features(test["signals"])
        if feature_names != test_names: raise RuntimeError("Custom feature columns are misaligned")
        pd.DataFrame({"feature": feature_names}).to_csv(out / "metrics" / "custom_feature_names.csv", index=False)
        rows=[]; reports={}; matrices={}; predictions={}
        for name, model in [("Custom features SVM", make_svm("rbf", 10.0)), ("Custom features Random Forest", make_random_forest(100 if smoke else 200, random_state=SEED))]:
            fitted, pred, _ = _fit_evaluate(name, model, custom_train, train["y"], custom_test, test["y"], out, rows, reports, matrices)
            joblib.dump(fitted, out / "models" / (name.lower().replace(" ", "_") + ".joblib")); predictions[name]=pred
        for name in predictions:
            pd.DataFrame({"subject":test["subjects"],"true_activity":test["y"],"predicted_activity":predictions[name]}).to_csv(out / "predictions" / (name.lower().replace(" ","_")+".csv"),index=False)
        pd.DataFrame(rows).to_csv(out / "metrics" / "custom_representation_comparison.csv",index=False)
        return rows
    if stages == ["cnn"]:
        predictions, seconds, history, _, (x_test, model, x_valid, y_valid) = train_cnn(train, test, out, SEED, smoke)
        metrics, report, matrix = evaluate_predictions(test["y"], predictions, "1D CNN", seconds, None)
        import torch
        import seaborn as sns
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=model.to(device).eval(); xb=torch.from_numpy(x_test).to(device)
        with torch.no_grad():
            start=time.perf_counter(); _=model(xb).argmax(1); metrics["inference_seconds"]=time.perf_counter()-start
        metrics["inference_ms_per_window"]=metrics["inference_seconds"]*1000/len(test["y"])
        save_evaluation(metrics,report,test["y"],predictions,"cnn",out/"metrics")
        pd.DataFrame({"subject":test["subjects"],"true_activity":test["y"],"predicted_activity":predictions}).to_csv(out/"predictions"/"cnn.csv",index=False)
        pd.DataFrame(perturbation_importance(model.cpu(),x_valid,y_valid),index=["total_acc_x","total_acc_y","total_acc_z","body_acc_x","body_acc_y","body_acc_z","body_gyro_x","body_gyro_y","body_gyro_z"],columns=[f"time_bin_{i}" for i in range(8)]).to_csv(out/"metrics"/"cnn_occlusion_importance.csv")
        import matplotlib.pyplot as plt
        hist=np.load(out/"metrics"/"cnn_history.npz")
        for metric,title,ylabel,filename in [("loss","CNN training and validation loss","Cross-entropy loss","16_cnn_loss.png"),("accuracy","CNN training and validation accuracy","Accuracy","17_cnn_accuracy.png")]:
            fig,ax=plt.subplots(figsize=(8,4)); ax.plot(hist[metric],label="Train");ax.plot(hist["val_"+metric],label="Validation");ax.set(title=title,xlabel="Epoch",ylabel=ylabel);ax.legend();fig.tight_layout();fig.savefig(out/"figures"/filename,dpi=220);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,6));sns.heatmap(matrix,annot=True,fmt="d",cmap="Blues",xticklabels=list(ACTIVITIES.values()),yticklabels=list(ACTIVITIES.values()),ax=ax);ax.set(title="1D CNN: official test confusion matrix",xlabel="Predicted",ylabel="True");fig.tight_layout();fig.savefig(out/"figures"/"cnn_confusion_matrix.png",dpi=220);plt.close(fig)
        pd.DataFrame([metrics]).to_csv(out/"metrics"/"cnn_metrics.csv",index=False)
        return metrics
    return None


def run_experiments(data_dir="data", results_dir="results", mode="full", stages=None):
    pipeline_started = time.perf_counter()
    smoke = mode == "smoke"
    out = Path(results_dir); figdir = out / "figures"
    for sub in ("metrics", "figures", "models", "predictions", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(data_dir, download=True)
    train, test, names = dataset["train"], dataset["test"], dataset["feature_names"]
    summary = {
        "source_archive_readme": str(dataset["root"] / "README.txt"),
        "train_windows": len(train["y"]), "test_windows": len(test["y"]),
        "features": len(names), "signal_shape": list(train["signals"].shape[1:]),
        "train_subjects": sorted(map(int, np.unique(train["subjects"]))),
        "test_subjects": sorted(map(int, np.unique(test["subjects"]))),
        "participants_disjoint": not bool(set(train["subjects"]) & set(test["subjects"])), "mode": mode,
    }
    (out / "metrics" / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    exploration_figures(train, names, figdir)
    if stages:
        return run_stage(stages, dataset, out, mode, summary)

    best_svm, best_rf, search_rows = _tune_main(train["X"], train["y"], train["subjects"], smoke)
    pd.DataFrame(search_rows).to_csv(out / "metrics" / "group_cv_hyperparameter_search.csv", index=False)
    (out / "metrics" / "selected_hyperparameters.json").write_text(json.dumps({"svm": best_svm, "random_forest": best_rf, "selection_metric": "mean subject-group cross-validation macro-F1", "seed": SEED}, indent=2))
    rows, reports, matrices, fitted_models, predictions_by_model = [], {}, {}, {}, {}

    svm, pred, _ = _fit_evaluate("SVM", make_svm(best_svm["kernel"], best_svm["C"], best_svm["gamma"]), train["X"], train["y"], test["X"], test["y"], out, rows, reports, matrices)
    fitted_models["SVM"], predictions_by_model["SVM"] = svm, pred
    rf, pred, _ = _fit_evaluate("Random Forest", make_random_forest(best_rf["n_estimators"], best_rf["max_depth"], best_rf["min_samples_leaf"], best_rf["max_features"], random_state=SEED), train["X"], train["y"], test["X"], test["y"], out, rows, reports, matrices)
    fitted_models["Random Forest"], predictions_by_model["Random Forest"] = rf, pred
    joblib.dump(svm, out / "models" / "svm_pipeline.joblib")
    joblib.dump(rf, out / "models" / "random_forest_pipeline.joblib")

    # Feature rankings and explanations use training data only.
    mi_scores, mi_table = training_mutual_information(train["X"], train["y"], names, SEED)
    mi_table.to_csv(out / "metrics" / "mutual_information_ranking.csv", index=False)
    _save_feature_importance(train, names, rf, out, smoke, mi_scores=mi_scores)
    impurity = rf.named_steps["classifier"].feature_importances_
    selection_rows = selector_cv_experiments(train["X"], train["y"], train["subjects"], SEED, smoke, model_names=("SVM-RBF", "Random Forest"))
    pd.DataFrame(selection_rows).to_csv(out / "metrics" / "feature_selection_group_cv.csv", index=False)
    pca_pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=min(250, len(train["y"]) - 1), svd_solver="randomized", random_state=SEED))])
    pca_pipe.fit(train["X"])
    pca_variance = pca_pipe.named_steps["pca"].explained_variance_ratio_
    pd.DataFrame({"component": np.arange(1, len(pca_variance) + 1), "explained_variance_ratio": pca_variance, "cumulative_explained_variance": np.cumsum(pca_variance)}).to_csv(out / "metrics" / "pca_explained_variance.csv", index=False)
    feature_experiment_figures(selection_rows, mi_scores, names, pca_variance, figdir)

    custom_train, custom_names = extract_window_features(train["signals"])
    custom_test, test_names = extract_window_features(test["signals"])
    if custom_names != test_names:
        raise RuntimeError("Custom feature train/test columns are not aligned")
    pd.DataFrame({"feature": custom_names}).to_csv(out / "metrics" / "custom_feature_names.csv", index=False)
    for name, model in [("Custom features SVM", make_svm("rbf", 10.0)), ("Custom features Random Forest", make_random_forest(100 if smoke else 200, random_state=SEED))]:
        fitted, pred, _ = _fit_evaluate(name, model, custom_train, train["y"], custom_test, test["y"], out, rows, reports, matrices)
        fitted_models[name], predictions_by_model[name] = fitted, pred
        joblib.dump(fitted, out / "models" / (name.lower().replace(" ", "_") + ".joblib"))

    cnn_pred, cnn_seconds, history, _, (x_test, cnn_model, x_valid, y_valid) = train_cnn(train, test, out, SEED, smoke)
    cnn_metrics, cnn_report, cnn_cm = evaluate_predictions(test["y"], cnn_pred, "1D CNN", cnn_seconds, None)
    save_evaluation(cnn_metrics, cnn_report, test["y"], cnn_pred, "cnn", out / "metrics")
    rows.append(cnn_metrics); reports["1D CNN"] = cnn_report; matrices["1D CNN"] = cnn_cm; predictions_by_model["1D CNN"] = cnn_pred
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = cnn_model.to(device).eval(); tensor_test = torch.from_numpy(x_test).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    with torch.no_grad():
        start = time.perf_counter(); _ = cnn_model(tensor_test).argmax(1)
        if device.type == "cuda": torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
    rows[-1]["inference_seconds"] = inference_seconds
    rows[-1]["inference_ms_per_window"] = inference_seconds * 1000 / len(test["y"])
    save_evaluation(rows[-1], cnn_report, test["y"], cnn_pred, "cnn", out / "metrics")

    # CNN occlusion importance uses participant-held-out validation signals only.
    saliency = perturbation_importance(cnn_model.cpu(), x_valid, y_valid)
    pd.DataFrame(saliency, index=["total_acc_x", "total_acc_y", "total_acc_z", "body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"], columns=[f"time_bin_{i}" for i in range(8)]).to_csv(out / "metrics" / "cnn_occlusion_importance.csv")
    hist = np.load(out / "metrics" / "cnn_history.npz")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(hist["loss"], label="Train"); ax.plot(hist["val_loss"], label="Validation"); ax.set(title="CNN cross-entropy loss", xlabel="Epoch", ylabel="Loss"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "16_cnn_loss.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(hist["accuracy"], label="Train"); ax.plot(hist["val_accuracy"], label="Validation"); ax.set(title="CNN accuracy", xlabel="Epoch", ylabel="Accuracy"); ax.legend(); fig.tight_layout(); fig.savefig(figdir / "17_cnn_accuracy.png", dpi=220); plt.close(fig)
    model_figures(rows, reports, matrices, names, impurity, figdir)
    pd.DataFrame(rows).to_csv(out / "metrics" / "model_comparison.csv", index=False)
    for name, pred in predictions_by_model.items():
        pd.DataFrame({"subject": test["subjects"], "true_activity": test["y"], "predicted_activity": pred, "true_name": pd.Series(test["y"]).map(ACTIVITIES), "predicted_name": pd.Series(pred).map(ACTIVITIES)}).to_csv(out / "predictions" / (name.lower().replace(" ", "_") + ".csv"), index=False)
    cv_summary = pd.DataFrame(search_rows).groupby("model")["cv_macro_f1"].agg(["mean", "std", "max"]).reset_index()
    cv_summary.to_csv(out / "metrics" / "subject_group_cv_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(cv_summary["model"], cv_summary["max"], yerr=cv_summary["std"], capsize=4); ax.set(ylim=(0, 1), title="Best group-CV macro-F1 by model family", ylabel="Macro F1"); fig.tight_layout(); fig.savefig(figdir / "22_group_cv_comparison.png", dpi=220); plt.close(fig)
    runtime = {"pipeline_seconds": time.perf_counter() - pipeline_started, "mode": mode, "seed": SEED, "selection_metric": "macro F1", "group_cv_folds": 2 if smoke else 3}
    (out / "logs" / "run_summary.json").write_text(json.dumps(runtime, indent=2))
    summary["models"] = {row["model"]: {key: row[key] for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1", "training_seconds", "inference_seconds")} for row in rows}
    (out / "metrics" / "run_results_summary.json").write_text(json.dumps(summary, indent=2))
    return {"dataset": dataset, "summary": summary, "model_rows": rows, "cv_rows": search_rows, "selection_rows": selection_rows, "reports": reports}
