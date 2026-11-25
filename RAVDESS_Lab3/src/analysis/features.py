"""Acoustic feature extraction, using a reference and vectorized implementation."""

from __future__ import annotations

import librosa
import numpy as np
from scipy.signal.windows import hann


def window_features_vectorized(audio_matrix: np.ndarray, fs: int, window_ms: int = 25, hop_ms: int = 10, include_pitch: bool = True) -> dict[str, np.ndarray]:
    """Compute 25 ms time/spectral features and framewise pYIN F0 for each actor.

    Inputs: 2D matrix (actors, samples). Outputs: (actors, frames), with
    F0 computed using a longer analysis window but the same hop length.
    """
    if audio_matrix.ndim != 2:
        raise ValueError("Expected a 2-D array (actors, samples)")
    k = round(fs * window_ms / 1000)
    h = round(fs * hop_ms / 1000)
    if min(k, h) < 1 or audio_matrix.shape[1] < k:
        raise ValueError("Invalid window or recording length")

    # librosa framing along the last dimension gives (actors, K, W).
    frames = librosa.util.frame(audio_matrix, frame_length=k, hop_length=h, axis=-1)
    frames = np.swapaxes(frames, -1, -2)  # (actors, W, K)
    rms = np.sqrt(np.mean(frames**2, axis=-1))
    peak = np.max(np.abs(frames), axis=-1)
    zcr = np.mean(np.diff(np.signbit(frames), axis=-1), axis=-1)

    weighted = frames * hann(k, sym=False)
    spectrum = np.abs(np.fft.rfft(weighted, axis=-1))
    frequencies = np.fft.rfftfreq(k, d=1 / fs)
    centroid = np.sum(spectrum * frequencies, axis=-1) / (np.sum(spectrum, axis=-1) + 1e-10)

    result = {"RMS": rms, "Peak": peak, "ZCR": zcr, "Spectral Centroid": centroid}
    if include_pitch:
        p = 2 ** int(np.ceil(np.log2(2 * fs / 65)))
        if audio_matrix.shape[1] < p:
            raise ValueError("Recording too short for the minimum F0 (65 Hz)")
        f0, _, _ = librosa.pyin(audio_matrix, sr=fs, fmin=65, fmax=500,
                               frame_length=p, hop_length=h, center=False)
        result["F0"] = f0
    return result


def window_features_reference(audio_matrix: np.ndarray, fs: int, window_ms: int = 25, hop_ms: int = 10, include_pitch: bool = True) -> dict[str, np.ndarray]:
    """Loop-based baseline for four features; same pYIN estimator as optimized version.

    Retaining identical mathematical definitions supports correctness checks.
    """
    k = round(fs * window_ms / 1000)
    h = round(fs * hop_ms / 1000)
    n_frames = 1 + (audio_matrix.shape[1] - k) // h
    if n_frames < 1:
        raise ValueError("Recording too short for framing")
    weights = hann(k, sym=False)
    frequencies = np.fft.rfftfreq(k, d=1 / fs)
    values = {name: np.empty((len(audio_matrix), n_frames), dtype=float) for name in ("RMS", "Peak", "ZCR", "Spectral Centroid")}
    for actor, signal in enumerate(audio_matrix):
        for window in range(n_frames):
            frame = signal[window*h:window*h+k]
            values["RMS"][actor, window] = np.sqrt(np.mean(frame**2))
            values["Peak"][actor, window] = np.max(np.abs(frame))
            values["ZCR"][actor, window] = np.mean(np.diff(np.signbit(frame)))
            spectrum = np.abs(np.fft.rfft(frame * weights))
            values["Spectral Centroid"][actor, window] = (spectrum * frequencies).sum() / (spectrum.sum() + 1e-10)
    if include_pitch:
        p = 2 ** int(np.ceil(np.log2(2 * fs / 65)))
        values["F0"], _, _ = librosa.pyin(audio_matrix, sr=fs, fmin=65, fmax=500,
                                          frame_length=p, hop_length=h, center=False)
    return values
