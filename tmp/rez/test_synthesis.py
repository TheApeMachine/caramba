#!/usr/bin/env python3
"""
Test script for Manifold synthesis functionality (diffusion generation and next-token prediction).
"""

import torch
import numpy as np
from manifold import Manifold, ManifoldConfig, DTYPE_REAL, TAU

def test_diffusion_chord_generation():
    """Test 1: Generate a C-Major chord from noise using diffusion."""
    print("=" * 60)
    print("Test 1: Diffusion Chord Generation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configure for synthesis
    cfg = ManifoldConfig(
        dt=0.01,
        synthesis_mode=True,
        restoring_force_strength=2.0,
        hold_cost_scale=1.0,  # Lower cost for generation
    )
    
    manifold = Manifold(cfg, device=device)
    
    # C-Major chord frequencies (C4, E4, G4)
    carrier_freqs = [261.63, 329.63, 392.00]
    
    print(f"\nGenerating chord from noise...")
    print(f"Target frequencies: {carrier_freqs} Hz")
    print(f"Carriers: C4, E4, G4")
    
    # Run diffusion
    manifold.generate_diffusion(
        n_oscillators=50,
        carrier_frequencies=carrier_freqs,
        n_steps=100,
        initial_temperature=2.0,
        final_temperature=0.01,
        cooling_schedule="exponential",
    )
    
    # Check results
    osc = manifold.state.get("oscillators")
    carriers = manifold.state.get("carriers")
    bonds = manifold.state.get("bonds")
    
    print(f"\nResults:")
    print(f"  Generated oscillators: {osc.shape[0]}")
    print(f"  Active carriers: {carriers.shape[0]}")
    
    # Show frequency distribution
    generated_freqs = osc.get("frequency").cpu().numpy()
    print(f"\nGenerated frequency distribution:")
    print(f"  Min: {generated_freqs.min():.2f} Hz")
    print(f"  Max: {generated_freqs.max():.2f} Hz")
    print(f"  Mean: {generated_freqs.mean():.2f} Hz")
    print(f"  Std: {generated_freqs.std():.2f} Hz")
    
    # Check bonding
    P = bonds.get("presence")
    if P.numel() > 0:
        n_bonds = (P > 0.0).sum().item()
        print(f"\nBonding:")
        print(f"  Total bonds: {n_bonds}")
        print(f"  Bonds per oscillator: {n_bonds / max(osc.shape[0], 1):.2f}")
    
    # Check how close generated frequencies are to targets
    print(f"\nFrequency alignment (distance to nearest carrier):")
    for i, target_freq in enumerate(carrier_freqs):
        distances = np.abs(generated_freqs - target_freq)
        min_dist = distances.min()
        mean_dist = distances.mean()
        print(f"  Carrier {i} ({target_freq:.2f} Hz): min_dist={min_dist:.2f} Hz, mean_dist={mean_dist:.2f} Hz")
    
    print("\n✓ Test 1 Complete\n")
    return manifold


def test_diffusion_melody():
    """Test 2: Generate a simple melody (C-D-E-F-G) from noise."""
    print("=" * 60)
    print("Test 2: Diffusion Melody Generation")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ManifoldConfig(
        dt=0.01,
        synthesis_mode=True,
        restoring_force_strength=1.5,
    )
    
    manifold = Manifold(cfg, device=device)
    
    # C Major scale (C4, D4, E4, F4, G4)
    melody_freqs = [261.63, 293.66, 329.63, 349.23, 392.00]
    note_names = ["C4", "D4", "E4", "F4", "G4"]
    
    print(f"\nGenerating melody from noise...")
    print(f"Target notes: {', '.join(note_names)}")
    
    manifold.generate_diffusion(
        n_oscillators=100,
        carrier_frequencies=melody_freqs,
        n_steps=150,
        initial_temperature=3.0,
        final_temperature=0.005,
        cooling_schedule="linear",
    )
    
    osc = manifold.state.get("oscillators")
    generated_freqs = osc.get("frequency").cpu().numpy()
    
    print(f"\nResults:")
    print(f"  Generated {osc.shape[0]} oscillators")
    print(f"  Frequency range: {generated_freqs.min():.2f} - {generated_freqs.max():.2f} Hz")
    
    # Count oscillators near each target
    print(f"\nOscillators per note (within 10 Hz):")
    for freq, name in zip(melody_freqs, note_names):
        count = np.sum(np.abs(generated_freqs - freq) < 10.0)
        print(f"  {name} ({freq:.2f} Hz): {count} oscillators")
    
    print("\n✓ Test 2 Complete\n")
    return manifold


def test_next_token_prediction():
    """Test 3: Next token prediction using carrier energy."""
    print("=" * 60)
    print("Test 3: Next Token Prediction")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = ManifoldConfig(dt=0.01)
    manifold = Manifold(cfg, device=device)
    
    # Simulate some context: create oscillators representing tokens
    # In real usage, these would be token embeddings
    n_tokens = 10
    token_freqs = torch.linspace(100, 1000, n_tokens, device=device)
    token_phases = torch.rand(n_tokens, device=device) * TAU
    token_amps = torch.ones(n_tokens, device=device) * 0.5
    
    # Add as oscillators
    from tensordict import TensorDict
    signals = TensorDict(
        {
            "frequency": token_freqs,
            "amplitude": token_amps,
            "phase": token_phases,
            "duration": torch.full((n_tokens,), 1.0, dtype=DTYPE_REAL, device=device),
        },
        batch_size=[n_tokens],
    )
    
    # Process a few steps to let carriers form
    for _ in range(5):
        manifold.step(signals)
    
    # Create dummy vocabulary embeddings
    vocab_size = 1000
    embedding_dim = 128
    vocab_embeddings = torch.randn(vocab_size, embedding_dim, device=device)
    vocab_embeddings = vocab_embeddings / (vocab_embeddings.norm(dim=1, keepdim=True) + 1e-8)
    
    # Predict next token
    print(f"\nPredicting next token...")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    
    logits = manifold.predict_next_token(vocab_embeddings)
    
    print(f"\nResults:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    
    # Get top predictions
    top_k = 5
    top_values, top_indices = torch.topk(logits, top_k)
    
    print(f"\nTop {top_k} predicted tokens:")
    for i, (val, idx) in enumerate(zip(top_values, top_indices)):
        print(f"  {i+1}. Token {idx.item()}: logit={val.item():.4f}")
    
    # Check carrier state
    carriers = manifold.state.get("carriers")
    if carriers.shape[0] > 0:
        carrier_energies = carriers.get("energy").cpu().numpy()
        print(f"\nCarrier energies (active concepts):")
        for i, energy in enumerate(carrier_energies):
            print(f"  Carrier {i}: energy={energy:.4f}")
    
    print("\n✓ Test 3 Complete\n")
    return manifold


def test_audio_separation_existing():
    """Test 4: Test existing audio separation functionality."""
    print("=" * 60)
    print("Test 4: Audio Separation (Existing Functionality)")
    print("=" * 60)
    
    import os
    
    # Check if test audio files exist
    test_files = [
        "two_speakers.wav",
        "four_speakers.wav",
        "source_00_carrier_0.wav",  # From previous runs
    ]
    
    found_file = None
    for f in test_files:
        if os.path.exists(f):
            found_file = f
            break
    
    if not found_file:
        print("\n⚠ No test audio file found. Skipping audio separation test.")
        print("  To test audio separation, provide a WAV file:")
        print("  python manifold.py <mix.wav> <out_dir>")
        return None
    
    print(f"\nFound test file: {found_file}")
    print("  (Run 'python manifold.py <mix.wav> <out_dir>' to test separation)")
    
    print("\n✓ Test 4 Skipped (no audio file provided)\n")
    return None


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Manifold Synthesis Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Chord generation
        test_diffusion_chord_generation()
        
        # Test 2: Melody generation
        test_diffusion_melody()
        
        # Test 3: Next token prediction
        test_next_token_prediction()
        
        # Test 4: Audio separation (info only)
        test_audio_separation_existing()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
