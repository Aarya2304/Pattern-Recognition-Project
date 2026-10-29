"""Statistical and spectral descriptors for windowed multichannel signals."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def extract_window_features(windows: np.ndarray, sampling_hz: float = 50.0) -> tuple[np.ndarray, list[str]]:
    """Return a finite feature matrix from (windows, time, channels) input."""
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 2 or x.shape[2] < 1:
        raise ValueError(f"Expected shape (n_windows, n_time>=2, n_channels), got {x.shape}")
    if not np.isfinite(x).all() or sampling_hz <= 0:
        raise ValueError("Signal values and sampling_hz must be finite; sampling_hz must be positive")
    n, time, channels = x.shape
    names: list[str] = []
    cols: list[np.ndarray] = []
    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        cols.append(np.asarray(value).reshape(n))
    axis = 1
    mean = x.mean(axis=axis)
    std = x.std(axis=axis)
    variance = x.var(axis=axis)
    abs_mean = np.abs(x).mean(axis=axis)
    rms = np.sqrt(np.mean(x ** 2, axis=axis))
    for label, values in (
        ("mean", mean), ("std", std), ("variance", variance),
        ("minimum", x.min(axis=axis)), ("maximum", x.max(axis=axis)),
        ("median", np.median(x, axis=axis)),
        ("iqr", np.percentile(x, 75, axis=axis) - np.percentile(x, 25, axis=axis)),
        ("rms", rms), ("mean_abs", abs_mean),
        ("energy", np.sum(x ** 2, axis=axis) / time),
        ("skewness", skew(x, axis=axis, bias=False, nan_policy="omit")),
        ("kurtosis", kurtosis(x, axis=axis, fisher=True, bias=False, nan_policy="omit")),
        ("zero_crossing_rate", np.mean((x[:, 1:, :] * x[:, :-1, :]) < 0, axis=axis)),
    ):
        for channel in range(channels):
            add(f"ch{channel}_{label}", values[:, channel])
    magnitude = np.sqrt(np.sum(x ** 2, axis=2))
    for label, values in (("signal_magnitude_mean", magnitude.mean(axis=1)),
                          ("signal_magnitude_rms", np.sqrt(np.mean(magnitude ** 2, axis=1))),
                          ("signal_magnitude_std", magnitude.std(axis=1))):
        add(label, values)
    for i in range(channels):
        for j in range(i + 1, channels):
            a, b = x[:, :, i], x[:, :, j]
            aa, bb = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
            denominator = np.sqrt(np.sum(aa * aa, axis=1) * np.sum(bb * bb, axis=1))
            corr = np.divide(np.sum(aa * bb, axis=1), denominator,
                             out=np.zeros(n), where=denominator > 1e-12)
            add(f"corr_ch{i}_ch{j}", corr)
    spectrum = np.abs(np.fft.rfft(x, axis=1)) / time
    frequencies = np.fft.rfftfreq(time, d=1.0 / sampling_hz)
    power = spectrum ** 2
    total_power = power.sum(axis=1)
    peak_idx = np.argmax(power[:, 1:, :], axis=1) + 1 if power.shape[1] > 1 else np.zeros((n, channels), int)
    for channel in range(channels):
        p = power[:, :, channel]
        p_sum = total_power[:, channel]
        prob = np.divide(p, p_sum[:, None], out=np.zeros_like(p), where=p_sum[:, None] > 0)
        entropy = -np.sum(np.where(prob > 0, prob * np.log2(prob + 1e-15), 0.0), axis=1)
        for label, values in (
            ("dominant_frequency", frequencies[peak_idx[:, channel]]),
            ("spectral_energy", p_sum),
            ("spectral_entropy", entropy),
            ("spectral_centroid", np.divide((p * frequencies).sum(axis=1), p_sum,
                out=np.zeros(n), where=p_sum > 0)),
        ):
            add(f"ch{channel}_{label}", values)
        for lo, hi in ((0, 0.5), (0.5, 1.0), (1.0, 3.0), (3.0, sampling_hz / 2)):
            mask = (frequencies >= lo) & ((frequencies < hi) if hi < sampling_hz / 2 else (frequencies <= hi))
            add(f"ch{channel}_band_energy_{lo:g}_{hi:g}hz", p[:, mask].sum(axis=1))
    matrix = np.column_stack(cols).astype(np.float32)
    matrix[~np.isfinite(matrix)] = 0.0
    return matrix, names
