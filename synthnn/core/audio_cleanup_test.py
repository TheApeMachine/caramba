from __future__ import annotations

import numpy as np

from synthnn.core.resonant_cleanup import ResonantCleanupConfig, cleanup_resonant_only, detect_artifacts_resonant


class TestResonantAudioCleanup:
    def test_detects_and_reduces_known_tones(self) -> None:
        sr = 8000
        n = sr  # 1s, 1 Hz FFT resolution
        t = np.arange(n, dtype=np.float64) / float(sr)

        # "Content" signal outside detection band.
        content = 0.08 * np.sin(2.0 * np.pi * 220.0 * t)
        hum60 = 0.30 * np.sin(2.0 * np.pi * 60.0 * t)
        hum180 = 0.20 * np.sin(2.0 * np.pi * 180.0 * t)
        audio = (content + hum60 + hum180).astype(np.float64)

        cfg = ResonantCleanupConfig(
            min_frequency_hz=50.0,
            max_frequency_hz=200.0,
            num_detectors=151,  # 1 Hz bins: includes 60/180 exactly
            detector_spacing="linear",
            detector_bandwidth_hz=5.0,
            block_size=1000,
            hop_size=1000,
            presence_db=3.0,
            min_persistence=0.25,
            min_phase_coherence=0.25,
            detection_threshold=0.15,
            lateral_inhibition_hz=5.0,
            max_artifacts=6,
            cancellation_strength=0.9,
            include_harmonics=False,
        )

        arts = detect_artifacts_resonant(audio, sample_rate_hz=sr, cfg=cfg)
        assert len(arts) >= 1
        freqs = np.array([a.frequency_hz for a in arts], dtype=np.float64)
        assert np.any(np.abs(freqs - 60.0) <= 1.0)
        assert np.any(np.abs(freqs - 180.0) <= 1.0)

        cleaned, report = cleanup_resonant_only(audio, sample_rate_hz=sr, cfg=cfg, artifacts=arts)
        assert cleaned.shape == audio.shape
        assert report.sample_rate_hz == sr

        # Verify energy at hum bins drops substantially.
        X0 = np.fft.rfft(audio)
        X1 = np.fft.rfft(cleaned)
        bin60 = int(round(60.0 * n / sr))
        bin180 = int(round(180.0 * n / sr))
        mag0_60 = float(np.abs(X0[bin60]))
        mag1_60 = float(np.abs(X1[bin60]))
        mag0_180 = float(np.abs(X0[bin180]))
        mag1_180 = float(np.abs(X1[bin180]))
        assert mag1_60 < 0.4 * mag0_60
        assert mag1_180 < 0.4 * mag0_180

