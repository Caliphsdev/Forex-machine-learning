"""Tests for the feature scaler (fit on train only, apply everywhere)."""
import pandas as pd
import pytest

from fxml.datasets.scaling import FeatureScaler


def frame(**columns):
    return pd.DataFrame(columns)


def test_transform_centres_on_train_median_and_divides_by_train_iqr():
    train = frame(a=[1.0, 2.0, 3.0, 4.0, 5.0])  # median 3, q25 2, q75 4, IQR 2
    scaler = FeatureScaler().fit(train)
    out = scaler.transform(frame(a=[3.0, 5.0]))
    assert out["a"].tolist() == [0.0, 1.0]


def test_transform_never_changes_what_fit_learned():
    scaler = FeatureScaler().fit(frame(a=[1.0, 2.0, 3.0, 4.0, 5.0]))
    before = scaler.center_.copy()
    scaler.transform(frame(a=[1000.0, 2000.0, 3000.0]))  # wild "future" data
    pd.testing.assert_series_equal(scaler.center_, before)
    assert scaler.center_["a"] == 3.0


def test_constant_column_does_not_divide_by_zero():
    scaler = FeatureScaler().fit(frame(b=[7.0, 7.0, 7.0]))
    out = scaler.transform(frame(b=[7.0, 8.0]))
    assert out["b"].tolist() == [0.0, 1.0]


def test_nan_values_are_ignored_when_fitting():
    scaler = FeatureScaler().fit(frame(a=[1.0, 2.0, 3.0, 4.0, 5.0, float("nan")]))
    assert scaler.center_["a"] == 3.0
    assert scaler.scale_["a"] == 2.0


def test_output_keeps_index_and_fitted_column_order():
    scaler = FeatureScaler().fit(frame(a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0]))
    data = pd.DataFrame({"b": [5.0], "a": [2.0]}, index=pd.Index([42], name="time"))
    out = scaler.transform(data)
    assert list(out.columns) == ["a", "b"]
    assert out.index.equals(data.index)


def test_save_and_load_roundtrip(tmp_path):
    train = frame(a=[1.0, 2.0, 3.0, 4.0, 5.0], b=[10.0, 20.0, 30.0, 40.0, 50.0])
    scaler = FeatureScaler().fit(train)
    path = tmp_path / "scaler.json"
    scaler.save(path)
    loaded = FeatureScaler.load(path)
    pd.testing.assert_frame_equal(scaler.transform(train), loaded.transform(train))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeatureScaler().transform(frame(a=[1.0]))


def test_transform_with_missing_column_raises():
    scaler = FeatureScaler().fit(frame(a=[1.0, 2.0], b=[3.0, 4.0]))
    with pytest.raises(ValueError):
        scaler.transform(frame(a=[1.0]))


def test_passthrough_columns_are_left_unchanged():
    train = frame(a=[1.0, 2.0, 3.0, 4.0, 5.0], angle=[0.1, 0.9, -0.5, 0.3, -0.7])
    scaler = FeatureScaler(passthrough=["angle"]).fit(train)
    out = scaler.transform(train)
    assert out["angle"].tolist() == train["angle"].tolist()
    assert out["a"].iloc[2] == 0.0  # the median row of "a"
