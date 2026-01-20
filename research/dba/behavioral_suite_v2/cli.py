#!/usr/bin/env python3
"""
Command-line interface for the DBA Behavioral Test Suite.

Usage:
    # Run with caramba manifest (recommended)
    python -m behavioral_suite_v2.cli --manifest research/dba/benchmark.yml

    # Or directly from run_eval.py
    python research/dba/run_eval.py --manifest research/dba/benchmark.yml
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DBA Behavioral Test Suite - Multi-model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run evaluation using manifest (loads models via caramba)
    python -m behavioral_suite_v2.cli --manifest research/dba/benchmark.yml

    # Customize test count and output directory
    python -m behavioral_suite_v2.cli --manifest benchmark.yml -o ./results -n 50

    # Skip attention capture for faster runs
    python -m behavioral_suite_v2.cli --manifest benchmark.yml --no-attention

    # Use mock models for testing the pipeline
    python -m behavioral_suite_v2.cli --use-mock
        """,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to caramba manifest (benchmark.yml) for model configs. "
             "This is the recommended way to load models.",
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./results/<timestamp>)",
    )

    parser.add_argument(
        "-n", "--tests-per-category",
        type=int,
        default=30,
        help="Number of tests per category (default: 30)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Skip attention capture (faster but no attention visualizations)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (default: from manifest or cuda if available)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type (default: from manifest or float16)",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate (default: 50)",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser with results",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tiktoken:gpt2",
        help="Tokenizer to use (default: tiktoken:gpt2)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--perplexities",
        type=str,
        default=None,
        help="JSON file or inline JSON with perplexity values per model for Pareto curve. "
             "Format: '{\"model_name\": 15.2, ...}' or path to JSON file",
    )

    parser.add_argument(
        "--perplexity-csv",
        type=str,
        default=None,
        help="CSV file with model perplexities (columns: model,perplexity)",
    )

    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock models for testing (no actual model loading)",
    )

    parser.add_argument(
        "--unsafe-pickle",
        action="store_true",
        help="Allow unsafe pickle loading for checkpoints",
    )

    return parser.parse_args()


def load_perplexities(args: argparse.Namespace, output_dir: Path) -> dict[str, float]:
    """
    Load perplexity values from various sources.
    """
    perplexities = {}

    # Try inline JSON or JSON file
    if args.perplexities:
        try:
            perplexities = json.loads(args.perplexities)
        except json.JSONDecodeError:
            ppl_path = Path(args.perplexities)
            if ppl_path.exists():
                with open(ppl_path) as f:
                    perplexities = json.load(f)

    # Try CSV file
    elif args.perplexity_csv:
        csv_path = Path(args.perplexity_csv)
        if csv_path.exists():
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get('model') or row.get('name') or row.get('model_name')
                    ppl = row.get('perplexity') or row.get('ppl')
                    if model and ppl:
                        try:
                            perplexities[model] = float(ppl)
                        except ValueError:
                            pass

    return perplexities


def open_in_browser(filepath: Path) -> bool:
    """Open a file in the default web browser."""
    filepath = filepath.resolve()
    url = f"file://{filepath}"

    print(f"\nOpening results in browser: {url}")

    try:
        system = platform.system().lower()

        if system == "darwin":
            subprocess.run(["open", str(filepath)], check=True)
        elif system == "linux":
            try:
                subprocess.run(["xdg-open", str(filepath)], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                webbrowser.open(url)
        elif system == "windows":
            os.startfile(str(filepath))
        else:
            webbrowser.open(url)

        return True

    except Exception as e:
        print(f"Could not open browser: {e}")
        print(f"Please open manually: {filepath}")
        return False


def load_models_from_manifest(
    manifest_path: str,
    device: str | None = None,
    dtype: str | None = None,
    unsafe_pickle: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load teacher and student models using the caramba CheckpointCompareTrainer.

    Returns:
        Tuple of (models dict, metadata dict with configs)
    """
    import yaml
    from pathlib import Path

    # Import caramba components
    from trainer.checkpoint_compare import CheckpointCompareTrainer

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load and parse manifest
    with open(manifest_path) as f:
        manifest_raw = yaml.safe_load(f)

    # Resolve variables in manifest
    vars_dict = manifest_raw.get("vars", {})

    def resolve_vars(obj: Any) -> Any:
        if isinstance(obj, str):
            # Replace ${var} with actual values
            for var, val in vars_dict.items():
                obj = obj.replace(f"${{{var}}}", str(val))
            return obj
        elif isinstance(obj, dict):
            return {k: resolve_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_vars(v) for v in obj]
        return obj

    manifest = resolve_vars(manifest_raw)

    # Find the checkpoint_compare target
    targets = manifest.get("targets", [])
    compare_target = None

    for target in targets:
        trainer_cfg = target.get("trainer", {})
        if trainer_cfg.get("ref") == "trainer.checkpoint_compare":
            compare_target = target
            break

    if compare_target is None:
        raise ValueError("No trainer.checkpoint_compare target found in manifest")

    trainer_cfg = compare_target["trainer"]["config"]

    # Override device/dtype if provided
    actual_device = device or trainer_cfg.get("device", "cuda")
    actual_dtype = dtype or trainer_cfg.get("dtype", "float16")

    # Get model configs
    teacher_model_cfg = trainer_cfg["teacher_model"]
    student_model_cfg = trainer_cfg.get("student_model")

    # If student_model not in trainer config, get from system config
    if student_model_cfg is None:
        system_cfg = compare_target.get("system", {}).get("config", {})
        student_model_cfg = system_cfg.get("model")

    if student_model_cfg is None:
        raise ValueError("Could not find student model config in manifest")

    # Create trainer and run to get models
    trainer = CheckpointCompareTrainer(
        teacher_ckpt=trainer_cfg["teacher_ckpt"],
        student_ckpt=trainer_cfg["student_ckpt"],
        teacher_model=teacher_model_cfg,
        student_model=student_model_cfg,
        device=actual_device,
        dtype=actual_dtype,
        strict=trainer_cfg.get("strict", False),
        unsafe_pickle_load=unsafe_pickle or trainer_cfg.get("unsafe_pickle_load", False),
    )

    # Create a minimal manifest/target object for the trainer
    class MinimalManifest:
        def __init__(self, artifacts_dir: str):
            self.artifacts_dir = artifacts_dir

    class MinimalTarget:
        def __init__(self, system_cfg: dict):
            self.system = type('System', (), {
                'ref': 'system.language_model',
                'config': system_cfg
            })()

    mini_manifest = MinimalManifest(manifest.get("artifacts_dir", "artifacts"))
    mini_target = MinimalTarget(compare_target.get("system", {}).get("config", {}))

    # Run trainer to load models
    result = trainer.run(
        manifest=mini_manifest,
        target=mini_target,
        engine=None,
        dry_run=False,
    )

    if result is None:
        raise RuntimeError("Trainer returned None")

    models = {
        "teacher": result["teacher"],
        "student": result["student"],
    }

    metadata = {
        "device": str(result["device"]),
        "teacher_ckpt": trainer_cfg["teacher_ckpt"],
        "student_ckpt": trainer_cfg["student_ckpt"],
    }

    return models, metadata


class CarambaModelWrapper:
    """
    Wraps a caramba Model to work with the behavioral test suite.
    Implements the EvaluableModel protocol.
    """

    def __init__(
        self,
        model: Any,  # caramba.model.Model
        tokenizer: Any,
        device: str | torch.device = "cuda",
        max_length: int = 2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_length = max_length
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """Generate text continuation."""
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

        input_ids = input_ids[-(self.max_length - max_new_tokens):]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        generated_ids = []

        # Mask logits beyond tokenizer vocab (caramba vocab is often padded, e.g., 50304 vs 50257).
        valid_vocab_size = None
        if hasattr(self.tokenizer, "n_vocab"):
            valid_vocab_size = int(self.tokenizer.n_vocab)
        elif hasattr(self.tokenizer, "vocab_size"):
            valid_vocab_size = int(self.tokenizer.vocab_size)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(input_tensor)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                next_logits = logits[0, -1, :].clone()
                if valid_vocab_size is not None and next_logits.shape[0] > valid_vocab_size:
                    next_logits[valid_vocab_size:] = float("-inf")

                if temperature == 0.0:
                    next_token = next_logits.argmax().item()
                else:
                    probs = torch.softmax(next_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                generated_ids.append(next_token)

                eos_token = getattr(self.tokenizer, 'eos_token_id', None)
                if eos_token is not None and next_token == eos_token:
                    break

                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)

                if input_tensor.shape[1] > self.max_length:
                    input_tensor = input_tensor[:, -self.max_length:]

        # Decode
        if hasattr(self.tokenizer, 'decode'):
            return self.tokenizer.decode(generated_ids)
        else:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def get_choice_logprobs(
        self,
        prompt: str,
        choices: list[str],
    ) -> dict[str, float]:
        """
        Get log probabilities for each choice.

        Computes the conditional log probability of the full choice string
        by teacher-forcing each token and summing the per-token logprobs.
        This correctly handles multi-token answers.
        """
        # Tokenize the prompt
        if hasattr(self.tokenizer, 'encode'):
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

        prompt_len = len(prompt_ids)

        valid_vocab_size = None
        if hasattr(self.tokenizer, "n_vocab"):
            valid_vocab_size = int(self.tokenizer.n_vocab)
        elif hasattr(self.tokenizer, "vocab_size"):
            valid_vocab_size = int(self.tokenizer.vocab_size)

        result = {}
        for choice in choices:
            # Tokenize the choice
            if hasattr(self.tokenizer, 'encode'):
                choice_ids = self.tokenizer.encode(choice)
            else:
                choice_ids = self.tokenizer(choice, add_special_tokens=False)["input_ids"]

            if not choice_ids:
                continue

            # Create full sequence: prompt + choice tokens
            full_ids = prompt_ids + list(choice_ids)

            # Truncate from the front if needed
            if len(full_ids) > self.max_length:
                excess = len(full_ids) - self.max_length
                full_ids = full_ids[excess:]
                effective_prompt_len = max(0, prompt_len - excess)
            else:
                effective_prompt_len = prompt_len

            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                # logits shape: [batch, seq_len, vocab_size]
                all_logits = logits[0]  # [seq_len, vocab_size]
                if valid_vocab_size is not None and all_logits.shape[-1] > valid_vocab_size:
                    all_logits = all_logits.clone()
                    all_logits[..., valid_vocab_size:] = float("-inf")
                all_log_probs = torch.log_softmax(all_logits, dim=-1)

                # Sum logprobs for each choice token
                # Position i's logits predict token at position i+1
                total_logprob = 0.0
                for i, token_id in enumerate(choice_ids):
                    pred_pos = effective_prompt_len - 1 + i
                    if pred_pos >= 0 and pred_pos < all_log_probs.shape[0]:
                        total_logprob += all_log_probs[pred_pos, token_id].item()

                result[choice] = total_logprob

        return result


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("DBA Behavioral Test Suite v2")
    print("=" * 60)

    if args.manifest:
        print(f"Manifest: {args.manifest}")
    elif args.use_mock:
        print("Using mock models (--use-mock)")
    else:
        print("WARNING: No manifest specified. Use --manifest or --use-mock")
        return 1

    print(f"Tests per category: {args.tests_per_category}")
    print(f"Seed: {args.seed}")
    print(f"Capture attention: {not args.no_attention}")
    print("=" * 60)

    # Import suite components
    from .generator import generate_suite
    from .runner import EvalRunner, EvalConfig
    from .visualizer import (
        ResultsVisualizer,
        generate_html_report,
    )

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./results") / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load tokenizer
    tokenizer = None
    tokenizer_spec = args.tokenizer
    try:
        if tokenizer_spec.startswith("tiktoken:"):
            import tiktoken
            encoding = tokenizer_spec.split(":", 1)[1]
            tokenizer = tiktoken.get_encoding(encoding)
            print(f"\nLoaded tiktoken tokenizer: {encoding}")
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
            print(f"\nLoaded HuggingFace tokenizer: {tokenizer_spec}")
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        return 1

    # Load models
    print("\n--- Loading models ---")
    models = {}

    if args.use_mock:
        print("Using mock models (--use-mock specified)")
        from .runner import MockModel
        models = {
            "teacher": MockModel("teacher", accuracy=0.75),
            "student": MockModel("student", accuracy=0.80),
        }
    else:
        try:
            raw_models, metadata = load_models_from_manifest(
                manifest_path=args.manifest,
                device=args.device,
                dtype=args.dtype,
                unsafe_pickle=args.unsafe_pickle,
            )

            # Wrap models with our wrapper
            device = metadata["device"]
            for name, model in raw_models.items():
                models[name] = CarambaModelWrapper(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )

            print(f"Loaded from manifest:")
            print(f"  Teacher: {metadata['teacher_ckpt']}")
            print(f"  Student: {metadata['student_ckpt']}")
            print(f"  Device: {device}")

        except Exception as e:
            print(f"ERROR: Failed to load models from manifest: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"Loaded {len(models)} model(s): {list(models.keys())}")

    # Generate test suite
    print("\n--- Generating test suite ---")
    suite = generate_suite(
        seed=args.seed,
        tests_per_category=args.tests_per_category,
    )
    print(f"Generated {len(suite.tests)} tests across {len(suite.category_counts)} categories")

    if args.verbose:
        for cat, count in suite.category_counts.items():
            print(f"  - {cat}: {count}")

    # Save test suite
    suite_path = output_dir / "test_suite.json"
    with open(suite_path, 'w') as f:
        json.dump(suite.to_dict(), f, indent=2, default=str)
    print(f"Saved test suite to: {suite_path}")

    # Run evaluation
    print("\n--- Running evaluation ---")
    config = EvalConfig(
        max_new_tokens=args.max_new_tokens,
        capture_attention=not args.no_attention,
        show_progress=True,
    )

    runner = EvalRunner(models, config)
    results = runner.run(suite.tests)

    # Save results
    print("\n--- Saving results ---")
    results.save(output_dir)

    # Generate visualizations
    print("\n--- Generating visualizations ---")
    viz = ResultsVisualizer()

    # Per-category comparison
    viz.plot_category_comparison(
        results.category_results,
        metric="exact_match_rate",
        output_path=output_dir / "category_exact_match.png",
    )

    viz.plot_category_comparison(
        results.category_results,
        metric="soft_score_avg",
        output_path=output_dir / "category_soft_score.png",
    )

    # Head-to-head matrix
    if len(models) > 1:
        viz.plot_head_to_head_matrix(
            results.comparisons,
            output_path=output_dir / "head_to_head.png",
        )

    # Score distribution
    viz.plot_soft_score_distribution(
        results.scores,
        output_path=output_dir / "score_distribution.png",
    )

    # Failure modes
    viz.plot_failure_modes(
        results.summaries,
        output_path=output_dir / "failure_modes.png",
    )

    # Pareto curve (if perplexity data available)
    perplexities = load_perplexities(args, output_dir)
    if perplexities:
        print(f"\nLoaded perplexity data for {len(perplexities)} model(s)")
        for model, ppl in perplexities.items():
            print(f"  - {model}: {ppl:.2f}")

        matched_perplexities = {}
        for model_id in results.model_ids:
            if model_id in perplexities:
                matched_perplexities[model_id] = perplexities[model_id]
            else:
                for ppl_name, ppl_val in perplexities.items():
                    if ppl_name in model_id or model_id in ppl_name:
                        matched_perplexities[model_id] = ppl_val
                        break

        if matched_perplexities:
            print(f"\nGenerating Pareto curve with {len(matched_perplexities)} models...")
            viz.plot_pareto_from_results(
                summaries=results.summaries,
                perplexities=matched_perplexities,
                accuracy_metric="exact_match_rate",
                output_path=output_dir / "pareto_curve.png",
            )

    # Generate HTML report
    html_path = output_dir / "report.html"
    generate_html_report(results, html_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    for model_id, summary in results.summaries.items():
        print(f"\n{model_id}:")
        print(f"  Exact match rate: {summary['exact_match_rate']:.1%}")
        print(f"  Content match rate: {summary['content_match_rate']:.1%}")
        print(f"  Average soft score: {summary['soft_score_avg']:.2f}")
        print(f"  Repetition loops: {summary['repetition_loops']}")
        print(f"  Distractor contamination: {summary['distractor_contamination']}")

    if len(models) > 1:
        print("\nHead-to-head comparisons:")
        for comp in results.comparisons:
            print(f"  {comp['model_a']} vs {comp['model_b']}:")
            print(f"    {comp['model_a']} wins: {comp['wins_a']}")
            print(f"    {comp['model_b']} wins: {comp['wins_b']}")
            print(f"    Ties: {comp['ties']}")

    print(f"\nResults saved to: {output_dir}")
    print(f"HTML report: {html_path}")

    # Open in browser
    if not args.no_browser:
        open_in_browser(html_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
