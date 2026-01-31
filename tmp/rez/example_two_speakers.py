"""Example: run the resonant engine on a two-speaker wav file."""

import torch
import torchaudio

from tmp.rez.manifold import Manifold
from tmp.rez.paper.main import AudioTokenizer, ResonantEngine


def run_two_speakers(path: str) -> None:
    engine = ResonantEngine(seed=0)
    manifold = Manifold()
    tokenizer = AudioTokenizer(frame_ms=25.0, hop_ms=10.0, top_k=6, min_energy=0.08)

    audio, sr = torchaudio.load(path)
    frames = tokenizer.tokenize(audio, sr)

    for i, frame in enumerate(frames):
        manifold.clear_signals()
        for sig in frame:
            manifold.add_signal(
                {
                    "frequency": float(sig.freq_hz),
                    "amplitude": float(sig.amplitude),
                    "phase": float(sig.phase),
                    "duration": float(sig.duration_s),
                }
            )
        manifold.step_engine(engine)

        if i % 50 == 0:
            obs = engine.observe()
            print(
                f"frame={i:4d} | N={obs['n_oscillators']:3d} | M={obs['n_carriers']:2d} | "
                f"R={obs['global_sync_R']:.3f} | T={obs['medium_temperature']:.2f} | "
                f"E={float(obs['osc_amplitude'].sum().item()) if obs['n_oscillators'] > 0 else 0.0:.2f}"
            )


if __name__ == "__main__":
    run_two_speakers("tmp/rez/two_speakers.wav")
