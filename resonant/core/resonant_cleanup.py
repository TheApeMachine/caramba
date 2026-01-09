"""Resonant-only tonal artifact detection and cancellation.

This is the "missing audio" piece that pairs naturally with resonant-phase
phasor substrate: a bank of lock-in style resonators (complex demodulators with
exponential decay) used to:
- detect persistent, phase-coherent tones (e.g. mains hum, whine)
- synthesize an estimate of those tones and subtract them

Design goals
------------
- NumPy-only (no SciPy/librosa/audiofile dependencies)
- Works on raw float arrays already loaded by the caller
- Deterministic and reasonably fast for short/medium clips
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np

from resonant.core.artifacts import ArtifactProfile, ArtifactType
from resonant.core.cleanup_report import CleanupReport, rms


def _as_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float64, copy=False)
    return arr.astype(np.float64, copy=True)


def _mono_view(audio: np.ndarray) -> np.ndarray:
    """Return a mono float64 view of audio.

    Supported shapes:
    - (samples,)
    - (samples, channels)
    - (channels, samples)  (heuristic: channels <= 16)
    """

    x = _as_float_array(audio)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"audio must be 1D or 2D array, got shape={x.shape}")

    # Heuristic: if one dimension looks like channels, average over it.
    if x.shape[1] <= 16 and x.shape[0] > x.shape[1]:
        return np.mean(x, axis=1)
    if x.shape[0] <= 16 and x.shape[1] > x.shape[0]:
        return np.mean(x, axis=0)
    # Fall back: treat last axis as channels.
    return np.mean(x, axis=-1)


def _db_to_mag_ratio(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


@dataclass(frozen=True, slots=True)
class ResonantCleanupConfig:
    """Configuration for resonant-only artifact detection/cleanup."""

    min_frequency_hz: float = 40.0
    max_frequency_hz: float = 8000.0
    num_detectors: int = 96
    detector_spacing: Literal["log", "linear"] = "log"
    detector_bandwidth_hz: float = 30.0

    block_size: int = 2048
    hop_size: int | None = None

    # Detection heuristics
    presence_db: float = 6.0
    min_persistence: float = 0.20
    min_phase_coherence: float = 0.55
    detection_threshold: float = 0.55
    lateral_inhibition_hz: float = 25.0
    max_artifacts: int = 6

    # Cancellation
    cancellation_strength: float = 0.9
    include_harmonics: bool = True
    max_harmonics: int = 6

    def __post_init__(self) -> None:
        if float(self.min_frequency_hz) < 0.0:
            raise ValueError("min_frequency_hz must be >= 0")
        if float(self.max_frequency_hz) <= float(self.min_frequency_hz):
            raise ValueError("max_frequency_hz must be > min_frequency_hz")
        if int(self.num_detectors) <= 0:
            raise ValueError("num_detectors must be > 0")
        if float(self.detector_bandwidth_hz) <= 0.0:
            raise ValueError("detector_bandwidth_hz must be > 0")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be > 0")
        if self.hop_size is not None and int(self.hop_size) <= 0:
            raise ValueError("hop_size must be > 0 when provided")
        if not (0.0 <= float(self.cancellation_strength) <= 2.0):
            raise ValueError("cancellation_strength must be in [0,2]")
        if int(self.max_artifacts) < 0:
            raise ValueError("max_artifacts must be >= 0")
        if float(self.lateral_inhibition_hz) < 0.0:
            raise ValueError("lateral_inhibition_hz must be >= 0")
        if int(self.max_harmonics) < 0:
            raise ValueError("max_harmonics must be >= 0")


class _LockInBank:
    """Complex lock-in resonator bank.

    The internal update is an exponential-decay integrator on the demodulated
    signal: z[n+1] = d*z[n] + d*(x[n] * e^{-j 2π f t_n}).
    """

    def __init__(self, *, freqs_hz: np.ndarray, sample_rate_hz: int, bandwidth_hz: float) -> None:
        self.freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        if self.freqs_hz.ndim != 1 or self.freqs_hz.size == 0:
            raise ValueError("freqs_hz must be a non-empty 1D array")
        self.sample_rate_hz = int(sample_rate_hz)
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        self.bandwidth_hz = float(bandwidth_hz)
        if self.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be > 0")

        self._dt = 1.0 / float(self.sample_rate_hz)
        # Convert bandwidth to an exponential decay rate. (2π·bw) is a practical
        # mapping for lock-in style lowpass behavior.
        self._decay_rate = 2.0 * math.pi * float(self.bandwidth_hz)
        self._d = float(math.exp(-self._decay_rate * self._dt))
        # Cache per-block weight vectors for efficient block updates.
        self._w_cache: dict[int, np.ndarray] = {}
        self._d_pow_cache: dict[int, float] = {}
        self.z = np.zeros(self.freqs_hz.shape[0], dtype=np.complex128)

    @property
    def decay_step(self) -> float:
        """Per-sample decay multiplier d in (0,1]."""

        return float(self._d)

    def _weights(self, n: int) -> tuple[np.ndarray, float]:
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")
        w = self._w_cache.get(n)
        d_n = self._d_pow_cache.get(n)
        if w is None or d_n is None:
            # w[k] = d^{(n-1-k)} so that:
            # z_n = z_0*d^n + d*sum(m_k*w_k)
            w = (self._d ** np.arange(n - 1, -1, -1, dtype=np.float64)).astype(np.float64, copy=False)
            d_n = float(self._d ** n)
            self._w_cache[n] = w
            self._d_pow_cache[n] = d_n
        return w, float(d_n)

    def process_block(self, *, x_block: np.ndarray, start_sample: int) -> np.ndarray:
        x_block = np.asarray(x_block, dtype=np.float64)
        n = int(x_block.shape[0])
        if n == 0:
            return self.z

        w, d_n = self._weights(n)
        sr = float(self.sample_rate_hz)
        t = (float(start_sample) + np.arange(n, dtype=np.float64)) / sr
        # phase has shape (F, N)
        phase = (2.0 * math.pi) * (self.freqs_hz[:, None] * t[None, :])
        ref = np.exp(-1j * phase)
        mixed = x_block[None, :] * ref  # (F, N)
        incr = mixed @ w  # (F,)

        self.z = self.z * complex(d_n) + complex(self._d) * incr
        return self.z


def detector_frequencies(*, cfg: ResonantCleanupConfig, sample_rate_hz: int) -> np.ndarray:
    sr = float(int(sample_rate_hz))
    nyq = 0.5 * sr
    min_f = float(max(0.0, float(cfg.min_frequency_hz)))
    max_f = float(min(float(cfg.max_frequency_hz), nyq - 1.0))
    if not (max_f > min_f):
        raise ValueError(f"Invalid frequency range after clamping to Nyquist: [{min_f}, {max_f}] Hz")

    n = int(cfg.num_detectors)
    if cfg.detector_spacing == "linear":
        freqs = np.linspace(min_f, max_f, n, dtype=np.float64)
    else:
        # geomspace requires strictly positive endpoints.
        min_pos = max(min_f, 1e-3)
        freqs = np.geomspace(min_pos, max_f, n, dtype=np.float64)
    return freqs.astype(np.float64, copy=False)


def _classify(freq_hz: float) -> ArtifactType:
    f = float(freq_hz)
    if f <= 0.0:
        return ArtifactType.UNKNOWN
    if f < 220.0:
        return ArtifactType.HUM
    if f < 1200.0:
        return ArtifactType.BUZZ
    if f >= 1200.0:
        return ArtifactType.WHINE
    return ArtifactType.TONAL_NOISE


def _harmonics(freq_hz: float, *, max_f: float, max_harmonics: int) -> list[float]:
    f0 = float(freq_hz)
    if f0 <= 0.0:
        return []
    hs: list[float] = []
    for k in range(2, int(max_harmonics) + 1):
        fk = f0 * float(k)
        if fk > float(max_f):
            break
        hs.append(float(fk))
    return hs


def detect_artifacts_resonant(
    audio: np.ndarray,
    *,
    sample_rate_hz: int,
    cfg: ResonantCleanupConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[ArtifactProfile]:
    """Detect persistent tonal artifacts using a resonant lock-in bank."""

    cfg = ResonantCleanupConfig() if cfg is None else cfg
    x_mono = _mono_view(audio)
    x_mono = x_mono - float(np.mean(x_mono))  # remove DC bias

    sr = int(sample_rate_hz)
    freqs = detector_frequencies(cfg=cfg, sample_rate_hz=sr)
    bank = _LockInBank(freqs_hz=freqs, sample_rate_hz=sr, bandwidth_hz=float(cfg.detector_bandwidth_hz))

    block = int(cfg.block_size)
    hop = int(cfg.hop_size) if cfg.hop_size is not None else block
    n = int(x_mono.shape[0])
    if n == 0 or int(cfg.max_artifacts) == 0:
        return []

    if n <= block:
        frames = 1
    else:
        frames = 1 + int((n - block) // hop)

    z_hist = np.zeros((frames, freqs.shape[0]), dtype=np.complex128)
    baseline = np.zeros(frames, dtype=np.float64)
    eps = 1e-12

    for i in range(frames):
        start = i * hop
        end = min(n, start + block)
        x_block = x_mono[start:end]
        z = bank.process_block(x_block=x_block, start_sample=start)
        z_hist[i] = z
        amp = np.abs(z)
        baseline[i] = float(np.median(amp) + eps)
        if progress is not None:
            progress(i + 1, frames)

    amps = np.abs(z_hist)
    strengths = np.median(amps, axis=0)
    global_med = float(np.median(strengths) + eps)

    presence_ratio = _db_to_mag_ratio(float(cfg.presence_db))
    present = amps > (baseline[:, None] * presence_ratio)
    persistence = np.mean(present, axis=0).astype(np.float64, copy=False)

    coherence = np.zeros(freqs.shape[0], dtype=np.float64)
    if frames >= 2:
        delta = z_hist[1:] * np.conj(z_hist[:-1])
        valid = present[1:] & present[:-1]
        mag = np.abs(delta)
        unit = np.zeros_like(delta, dtype=np.complex128)
        ok = valid & (mag > eps)
        unit[ok] = delta[ok] / mag[ok]
        count = np.sum(ok, axis=0).astype(np.float64, copy=False)
        sum_vec = np.sum(unit, axis=0)
        nonzero = count > 0.0
        coherence[nonzero] = (np.abs(sum_vec[nonzero]) / count[nonzero]).astype(np.float64, copy=False)

    rel_strength = (strengths / global_med).astype(np.float64, copy=False)
    strength_score = rel_strength / (1.0 + rel_strength)
    score = strength_score
    score *= (0.25 + 0.75 * coherence)
    score *= (0.25 + 0.75 * persistence)

    # Candidate selection.
    cand_mask = (persistence >= float(cfg.min_persistence)) & (coherence >= float(cfg.min_phase_coherence))
    cand_mask &= score >= float(cfg.detection_threshold)
    idxs = np.argsort(-score)  # descending

    chosen: list[int] = []
    inh = float(cfg.lateral_inhibition_hz)
    for idx in idxs.tolist():
        if not bool(cand_mask[idx]):
            continue
        if inh > 0.0 and any(abs(float(freqs[idx]) - float(freqs[j])) <= inh for j in chosen):
            continue
        chosen.append(int(idx))
        if len(chosen) >= int(cfg.max_artifacts):
            break

    max_f = float(min(float(cfg.max_frequency_hz), 0.5 * float(sr)))
    profiles: list[ArtifactProfile] = []
    for idx in chosen:
        f = float(freqs[idx])
        typ = _classify(f)
        harm = _harmonics(f, max_f=max_f, max_harmonics=int(cfg.max_harmonics)) if bool(cfg.include_harmonics) else []
        profiles.append(
            ArtifactProfile(
                artifact_type=typ,
                frequency_hz=f,
                bandwidth_hz=float(cfg.detector_bandwidth_hz),
                strength=float(strengths[idx]),
                phase_coherence=float(coherence[idx]),
                persistence=float(persistence[idx]),
                harmonics_hz=harm,
                metadata={
                    "score": float(score[idx]),
                    "rel_strength": float(rel_strength[idx]),
                },
            )
        )
    return profiles


def cleanup_resonant_only(
    audio: np.ndarray,
    *,
    sample_rate_hz: int,
    cfg: ResonantCleanupConfig | None = None,
    artifacts: list[ArtifactProfile] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, CleanupReport]:
    """Clean audio by subtracting detected resonant artifacts.

    Returns
    -------
    cleaned, report
    """

    cfg = ResonantCleanupConfig() if cfg is None else cfg
    x = _as_float_array(audio)
    x_mono = _mono_view(x)
    sr = int(sample_rate_hz)

    if artifacts is None:
        artifacts = detect_artifacts_resonant(x_mono, sample_rate_hz=sr, cfg=cfg, progress=progress)

    if len(artifacts) == 0:
        report = CleanupReport(
            sample_rate_hz=sr,
            num_samples=int(x_mono.shape[0]),
            artifacts=[],
            rms_before=rms(x_mono),
            rms_after=rms(x_mono),
            notes={"status": "no_artifacts_detected"},
        )
        return x.copy(), report

    # Build a unique list of cancellation frequencies + gains.
    freqs: list[float] = []
    gains: list[float] = []
    for art in artifacts:
        base_gain = float(cfg.cancellation_strength)
        base_gain *= float(0.25 + 0.75 * float(art.phase_coherence))
        base_gain *= float(0.25 + 0.75 * float(art.persistence))
        base_gain = float(np.clip(base_gain, 0.0, 2.0))

        freqs.append(float(art.frequency_hz))
        gains.append(base_gain)

        if bool(cfg.include_harmonics):
            for k, fh in enumerate(art.harmonics_hz, start=2):
                freqs.append(float(fh))
                gains.append(float(base_gain / float(k)))

    # Deduplicate nearby frequencies (within 2 Hz) by keeping the one closest to a detected artifact frequency.
    # This prevents interference from multiple detectors picking up the same tone.
    freq_gain: dict[float, float] = {}
    artifact_freqs = {float(art.frequency_hz) for art in artifacts}

    for f, g in zip(freqs, gains, strict=True):
        # Find the closest artifact frequency within 2 Hz
        closest_artifact = None
        min_dist = float('inf')
        for af in artifact_freqs:
            dist = abs(float(f) - af)
            if dist < 2.0 and dist < min_dist:
                min_dist = dist
                closest_artifact = af

        if closest_artifact is not None:
            # Round to the artifact frequency to merge nearby detections
            key = float(round(closest_artifact, 3))
        else:
            # Fallback to original rounding if no nearby artifact found
            key = float(round(float(f), 3))

        freq_gain[key] = max(float(freq_gain.get(key, 0.0)), float(g))

    cancel_freqs = np.array(sorted(freq_gain.keys()), dtype=np.float64)
    cancel_gains = np.array([freq_gain[f] for f in cancel_freqs], dtype=np.float64)

    bank = _LockInBank(freqs_hz=cancel_freqs, sample_rate_hz=sr, bandwidth_hz=float(cfg.detector_bandwidth_hz))
    d = float(bank.decay_step)
    if d <= 0.0 or d > 1.0:
        # d should always be in (0,1] with positive decay rate, but keep this
        # guard in case of numerical issues.
        d = float(np.clip(d, 1e-9, 1.0))
    gain_scale = float((1.0 - d) / d)

    block = int(cfg.block_size)
    hop = int(cfg.hop_size) if cfg.hop_size is not None else block
    n = int(x_mono.shape[0])
    if n == 0:
        report = CleanupReport(
            sample_rate_hz=sr,
            num_samples=0,
            artifacts=list(artifacts),
            rms_before=0.0,
            rms_after=0.0,
            notes={"status": "empty_audio"},
        )
        return x.copy(), report

    if n <= block:
        frames = 1
    else:
        frames = 1 + int((n - block) // hop)

    # Output shape: preserve original input shape.
    y = x.copy()
    y_mono = _mono_view(y)

    for i in range(frames):
        start = i * hop
        end = min(n, start + block)
        x_block_orig = x_mono[start:end]
        z = bank.process_block(x_block=x_block_orig, start_sample=start)

        # Estimate the complex amplitude c ≈ (1-d)/d * z for each cancel frequency.
        # Use the accumulated state z which has converged over previous blocks
        c = gain_scale * z  # (F,) - z is already complex128, gain_scale is float
        t = (float(start) + np.arange(end - start, dtype=np.float64)) / float(sr)
        phase = (2.0 * math.pi) * (cancel_freqs[:, None] * t[None, :])
        osc = np.exp(1j * phase)
        # Reconstruct the tone: c is complex amplitude, osc is the complex exponential
        # The real tone is 2*real(c * osc) because c*osc + conj(c*osc) = 2*real(c*osc)
        tone = 2.0 * np.real(c[:, None] * osc)  # (F, N)
        cancel = np.sum(cancel_gains[:, None] * tone, axis=0)

        # Apply cancellation to the current block
        y_mono[start:end] = x_block_orig - cancel.astype(np.float64, copy=False)

        if progress is not None:
            progress(i + 1, frames)

    # If the original audio had multiple channels, apply the mono cancellation
    # uniformly across channels (keeps phase alignment for shared hum/whine).
    if y.ndim == 2 and y.shape != y_mono.shape:
        cancel_track = y_mono
        if y.shape[1] <= 16 and y.shape[0] > y.shape[1]:
            # (samples, channels)
            y = y.copy()
            y[:, :] = y[:, :] - (x_mono - cancel_track)[:, None]
        elif y.shape[0] <= 16 and y.shape[1] > y.shape[0]:
            # (channels, samples)
            y = y.copy()
            y[:, :] = y[:, :] - (x_mono - cancel_track)[None, :]

    report = CleanupReport(
        sample_rate_hz=sr,
        num_samples=int(x_mono.shape[0]),
        artifacts=list(artifacts),
        rms_before=rms(x_mono),
        rms_after=rms(_mono_view(y)),
        notes={
            "status": "ok",
            "cancel_frequencies_hz": cancel_freqs.tolist(),
        },
    )
    return y, report

