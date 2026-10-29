from pathlib import Path
import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

from src.data_loader import validate_split, DatasetValidationError
from src.evaluation import evaluate_predictions, group_splits
from src.feature_engineering import extract_window_features
from src.feature_selection import training_mutual_information
from src.models import build_cnn, make_random_forest, make_svm
from src.visualization import _save


def test_validate_split_alignment_and_finite_values():
    x=np.zeros((3,561),np.float32); y=np.array([1,2,3]); subjects=np.array([1,2,3]); signals=np.zeros((3,128,9),np.float32)
    validate_split(x,y,subjects,signals)
    with pytest.raises(DatasetValidationError): validate_split(x,y[:2],subjects,signals)
    x[0,0]=np.nan
    with pytest.raises(DatasetValidationError): validate_split(x,y,subjects,signals)


def test_feature_extraction_shape_finite_and_named():
    rng=np.random.default_rng(8); signal=rng.normal(size=(5,128,9))
    features,names=extract_window_features(signal)
    assert features.shape==(5,len(names)) and len(names)>200
    assert len(names)==len(set(names)) and np.isfinite(features).all()
    assert any("spectral_entropy" in name for name in names)
    assert any("corr_ch" in name for name in names)
    assert np.allclose(extract_window_features(signal)[0],features)


def test_group_folds_never_overlap():
    y=np.tile(np.arange(1,7),10); groups=np.repeat(np.arange(10),6)
    for tr,va in group_splits(y,groups,n_splits=5): assert not set(groups[tr])&set(groups[va])


def test_feature_selector_fits_and_reduces():
    rng=np.random.default_rng(4); x=rng.normal(size=(72,20)); y=np.tile(np.array([1,2,3,4,5,6]),12)
    scores,table=training_mutual_information(x,y,[f"x{i}" for i in range(20)])
    assert scores.shape==(20,) and table.iloc[0]["mutual_information"]>=table.iloc[-1]["mutual_information"]
    pipe=Pipeline([("scale",StandardScaler()),("pca",PCA(n_components=4)),("svm",SVC())]); pipe.fit(x,y); assert pipe.predict(x).shape==y.shape


def test_models_eval_and_serialization(tmp_path):
    rng=np.random.default_rng(3); x=rng.normal(size=(120,14)); y=np.tile(np.arange(1,7),20)
    for model in (make_svm("linear",1),make_random_forest(n_estimators=12)):
        model.fit(x,y); pred=model.predict(x); assert set(pred)<=set(range(1,7))
        path=tmp_path/(model.steps[-1][0]+".joblib");joblib.dump(model,path);assert np.array_equal(joblib.load(path).predict(x),pred)
    metrics,report,cm=evaluate_predictions(y,pred,"test",0.1,0.01)
    assert cm.shape==(6,6) and 0<=metrics["macro_f1"]<=1 and "WALKING" in report


def test_figure_generation(tmp_path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots();ax.plot([0,1],[0,1]);_save(fig,tmp_path,"test.png")
    assert (tmp_path/"test.png").stat().st_size>1000


def test_cnn_input_output_shape():
    torch=pytest.importorskip("torch")
    model=build_cnn(9,6); output=model(torch.zeros(4,128,9)); assert tuple(output.shape)==(4,6)
