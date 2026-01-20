import argparse
import sys
import time
import json
import logging
import yaml
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Any, Dict

# lm_eval integration
try:
    import lm_eval
    from lm_eval.api.model import LM
    from lm_eval import simple_evaluate
except ImportError:
    print("Please install lm-eval to run downstream benchmarks: pip install lm-eval")
    sys.exit(1)

# Local imports
from layer.mlx.transformer import DBATransformer
from scripts.infer_mlx import load_model, load_tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# WRAPPERS
# =============================================================================


class DBALMWrapper(LM):
    """Wrapper for DBA student model."""

    model: nn.Module  # Type hint to allow subclass override

    def __init__(
        self,
        checkpoint: str,
        teacher_weights: Optional[str] = None,
        batch_size: int = 1,
        sem_head_dim: int = 8,
        geo_head_dim: int = 32,
        v_head_dim: int | None = None,
        init_mode: str = "fresh",
        **kwargs,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.teacher_weights = teacher_weights
        self._batch_size = int(batch_size)
        self.sem_head_dim = sem_head_dim
        self.geo_head_dim = int(geo_head_dim)
        self.v_head_dim = v_head_dim
        self.init_mode = str(init_mode)

        # Load model and tokenizer
        logger.info(f"Loading DBA model from {checkpoint}...")
        self.tokenizer, self.tok_type = load_tokenizer(teacher_weights)
        self.model = load_model(
            checkpoint,
            teacher_weights,
            sem_head_dim=sem_head_dim,
            geo_head_dim=self.geo_head_dim,
            v_head_dim=self.v_head_dim,
            init_mode=self.init_mode,
        )

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "mps"

    @property
    def max_length(self) -> int:
        return 2048

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        # Determine strictness for special tokens based on tokenizer type
        kwargs = {}
        if add_special_tokens is not None:
            # Check if tiktoken-like or hf-like
            if hasattr(self.tokenizer, "encode_ordinary"):  # tiktoken
                pass
            else:
                kwargs["add_special_tokens"] = add_special_tokens

        tokens = self.tokenizer.encode(string, **kwargs)
        if left_truncate_len:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        results = []

        # TODO: Implement batching for speed
        pbar = tqdm(requests, desc="Evaluating (loglikelihood)")

        for req in pbar:
            # lm_eval 0.4+ passes Instance objects, older versions pass tuples
            if hasattr(req, 'args'):
                context, continuation = req.args
            else:
                context, continuation = req

            ctx_tokens = self.tok_encode(context, add_special_tokens=False)
            cont_tokens = self.tok_encode(continuation, add_special_tokens=False)

            full_tokens = ctx_tokens + cont_tokens
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[-self.max_length :]

            input_ids = mx.array([full_tokens])  # (1, T)

            logits, _, _ = self.model(input_ids)  # (1, T-1, V)

            targets = input_ids[:, 1:]  # (1, T-1)

            relevant_targets = targets[:, len(ctx_tokens) - 1 :]
            relevant_logits = logits[:, len(ctx_tokens) - 1 : -1, :]

            log_probs = nn.losses.cross_entropy(
                relevant_logits, relevant_targets, reduction="none"
            )
            log_probs = -log_probs

            sum_logprob = float(mx.sum(log_probs))

            greedy_tokens = mx.argmax(relevant_logits, axis=-1)
            comparison = mx.array(greedy_tokens == relevant_targets)
            is_greedy = bool(mx.all(comparison))

            results.append((sum_logprob, is_greedy))

        return results

    def generate_until(self, requests) -> List[str]:
        # Simple generation stub for completeness
        return []

    def loglikelihood_rolling(self, requests) -> List[float]:
        # Stub to satisfy ABC
        return []


class TeacherWrapper(DBALMWrapper):
    """Wrapper for base Llama model using mlx_lm."""

    def __init__(self, model_path: str, batch_size: int = 1, **kwargs):
        LM.__init__(self)
        self.model_path = model_path
        self._batch_size = int(batch_size)

        # Load using mlx_lm
        logger.info(f"Loading Teacher model from {model_path} via mlx_lm...")
        try:
            from mlx_lm import load

            result = load(model_path)
            # mlx_lm.load can return (model, tokenizer) or (model, tokenizer, config)
            self.model = result[0]
            _mlx_tokenizer = result[1]
        except ImportError:
            raise ImportError("mlx_lm not installed.")

        # IMPORTANT: Use the same tokenizer implementation as the DBA student
        # to avoid subtle special-token / byte-level differences.
        self.tokenizer, self.tok_type = load_tokenizer(model_path)

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        kwargs = {}
        if add_special_tokens is not None:
            kwargs["add_special_tokens"] = add_special_tokens

        tokens = self.tokenizer.encode(string, **kwargs)
        if left_truncate_len:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        # Override for mlx_lm model structure
        # mlx_lm models usually have __call__(inputs, cache=None) -> logits
        # Re-implementing explicitly to be safe
        results = []
        pbar = tqdm(requests, desc="Evaluating (loglikelihood)")

        for req in pbar:
            # lm_eval 0.4+ passes Instance objects, older versions pass tuples
            if hasattr(req, 'args'):
                context, continuation = req.args
            else:
                context, continuation = req
            ctx_tokens = self.tok_encode(context, add_special_tokens=False)
            cont_tokens = self.tok_encode(continuation, add_special_tokens=False)
            full_tokens = ctx_tokens + cont_tokens
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[-self.max_length :]

            input_ids = mx.array([full_tokens])

            # mlx_lm forward
            # Depending on version: model(x) -> logits
            logits = self.model(input_ids)

            targets = input_ids[:, 1:]
            relevant_targets = targets[:, len(ctx_tokens) - 1 :]
            relevant_logits = logits[:, len(ctx_tokens) - 1 : -1, :]

            # Sync shapes
            if relevant_logits.shape[1] != relevant_targets.shape[1]:
                # Shift adjustment might be off slightly depending on how mlx_lm handles caching
                # but for fresh input it should be standard causal.
                pass

            log_probs = nn.losses.cross_entropy(
                relevant_logits, relevant_targets, reduction="none"
            )
            log_probs = -log_probs
            sum_logprob = float(mx.sum(log_probs))

            greedy_tokens = mx.argmax(relevant_logits, axis=-1)
            comparison = mx.array(greedy_tokens == relevant_targets)
            is_greedy = bool(mx.all(comparison))
            results.append((sum_logprob, is_greedy))

        return results


# =============================================================================
# BENCHMARKS
# =============================================================================


class BenchmarkSuite:
    def __init__(self, model: nn.Module, tokenizer: Any, output_dir: str):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_behavior(self, cases_file: str, visualize: bool = False):
        """Run behavior benchmark (generation + optional viz)."""
        logger.info(f"Running Behavior Benchmark using {cases_file}")

        with open(cases_file, "r") as f:
            cases = yaml.safe_load(f)

        results = []
        viz_dir = self.output_dir / "viz"
        if visualize:
            viz_dir.mkdir(parents=True, exist_ok=True)

        for case in tqdm(cases, desc="Behavior Cases"):
            cid = case.get("id")
            prompt = case.get("prompt")
            expected = str(case.get("answer")).strip()
            kind = case.get("kind", "exact_match_greedy")

            # 1. Encode
            if hasattr(self.tokenizer, "encode_ordinary"):
                input_ids = mx.array([self.tokenizer.encode_ordinary(prompt)])
            else:
                # Keep special tokens consistent across models/benches
                input_ids = mx.array(
                    [self.tokenizer.encode(prompt, add_special_tokens=False)]
                )

            # 2. Generate (Greedy)
            # Basic generation loop
            tokens = input_ids
            max_new = 32
            generated_tokens = []

            # Llama 3 stop tokens (consistent with infer_mlx.py)
            LLAMA3_STOP_TOKENS = {128001, 128009}  # EOT variants
            eos_id: int | None = getattr(self.tokenizer, "eos_token_id", None)

            # --- Visualization Hook ---
            if visualize:
                # Run a single forward pass with return_attention=True on the PROMPT
                layers = getattr(self.model, "layers", None)
                if layers is not None and hasattr(layers[0], "attention"):
                    # It's DBATransformer
                    try:
                        logits, _, attn_weights = self.model(
                            input_ids, return_attention=True
                        )
                        self._plot_attention(
                            cid, attn_weights, input_ids[0].tolist(), viz_dir
                        )
                    except Exception as e:
                        logger.error(f"Viz failed for {cid}: {e}")
                else:
                    # Teacher / Generic MLX model might not support this easily without surgery
                    pass

            # --- Generation Loop ---
            # Use cached generation for DBATransformer (faster + tests cache correctness)
            is_dba = isinstance(self.model, DBATransformer)
            cache = [] if is_dba else None

            if is_dba:
                # Prefill with cache
                out = self.model(tokens, cache=cache)
                if isinstance(out, tuple):
                    logits, cache, _ = out
                else:
                    logits = out
            else:
                # Full recompute for teacher/generic models
                out = self.model(tokens)
                logits = out[0] if isinstance(out, tuple) else out

            for _ in range(max_new):
                next_tok = mx.argmax(logits[:, -1, :], axis=-1)
                next_tok_id = next_tok.item()
                generated_tokens.append(next_tok_id)

                # Check stop conditions (Llama 3 compatible)
                if eos_id is not None and next_tok_id == eos_id:
                    break
                if next_tok_id in LLAMA3_STOP_TOKENS:
                    break

                # Next step
                if is_dba:
                    # Cached decode: feed only the new token
                    token_arr = mx.array([[next_tok_id]])
                    out = self.model(token_arr, cache=cache)
                    if isinstance(out, tuple):
                        logits, cache, _ = out
                    else:
                        logits = out
                    tokens = mx.concatenate([tokens, token_arr], axis=1)
                else:
                    # Full recompute for teacher
                    tokens = mx.concatenate([tokens, next_tok[None]], axis=1)
                    out = self.model(tokens)
                    logits = out[0] if isinstance(out, tuple) else out

            gen_text = self.tokenizer.decode(generated_tokens).strip()

            # 3. Score - explicit modes
            match = False
            if kind == "int_greedy":
                # Extract first integer and compare
                try:
                    import re

                    nums = re.findall(r"-?\d+", gen_text)
                    if nums and nums[0] == expected:
                        match = True
                except Exception:
                    pass
            elif kind == "exact":
                # Strict exact match (whitespace-normalized)
                match = gen_text.strip() == expected.strip()
            elif kind == "contains":
                # Substring containment (soft match)
                match = expected in gen_text
            else:
                # Default: "exact_match_greedy" treated as contains for backward compat
                # but log warning if not explicitly contains
                if kind != "exact_match_greedy":
                    logger.warning(f"Unknown scoring kind '{kind}', using contains")
                match = expected in gen_text

            results.append(
                {
                    "id": cid,
                    "prompt": prompt,
                    "expected": expected,
                    "generated": gen_text,
                    "match": match,
                    "kind": kind,
                }
            )

        # Summary
        correct = sum(1 for r in results if r["match"])
        logger.info(
            f"Behavior Accuracy: {correct}/{len(results)} ({correct / len(results) * 100:.2f}%)"
        )

        # Save results
        with open(self.output_dir / "behavior_results.json", "w") as f:
            json.dump(results, f, indent=2)

    def _plot_attention(self, case_id, attn_maps, token_ids, viz_dir):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        tokens = [self.tokenizer.decode([t]) for t in token_ids]

        # Plot Layer 0, Middle, Last
        layer_indices = [0, len(attn_maps) // 2, len(attn_maps) - 1]

        for l_idx in layer_indices:
            if l_idx >= len(attn_maps):
                continue

            # Get map: (1, H, S, S)
            amap = attn_maps[l_idx]

            # Handle DBATransformer return signature variations
            # If it returns (attn_out, cache, weights), grab weights
            if isinstance(amap, tuple):
                # Search for weights tensor in tuple
                for item in amap:
                    if isinstance(item, mx.array) and item.ndim == 4:
                        amap = item
                        break

            if isinstance(amap, mx.array) and amap.ndim == 4:
                # Shape (B, H, S, S) -> Take 0th batch -> Mean over heads
                mean_attn = mx.mean(amap[0], axis=0)  # (S, S)

                plt.figure(figsize=(10, 8))
                plt.imshow(np.array(mean_attn), cmap="viridis")

                if len(tokens) < 40:
                    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
                    plt.yticks(range(len(tokens)), tokens, fontsize=8)

                plt.title(f"{case_id} L{l_idx} Mean Head Attn")
                plt.tight_layout()
                plt.savefig(viz_dir / f"{case_id}_L{l_idx}.png")
                plt.close()

    def run_perplexity(self):
        """Run Wikitext-2 perplexity."""
        logger.info("Running Perplexity Benchmark (wikitext-2)...")
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        except ImportError:
            logger.warning("datasets library missing.")
            return

        text = "\n".join(ds["text"])
        try:
            enc = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            enc = self.tokenizer.encode(text)
        if hasattr(enc, "ids"):
            enc = enc.ids  # Handle specialized tokenizer objects

        enc = mx.array(enc)

        # Sliding window
        block_size = 2048
        stride = 2048

        nlls: list[float] = []
        # Eval only first 10 chunks to save time for this demo
        for i in tqdm(
            range(0, min(len(enc) - block_size + 1, 10 * stride), stride),
            desc="PPL Blocks",
        ):
            chunk = enc[i : i + block_size]
            input_ids = chunk[None, :]  # (1, T)

            # Wrapper agnostic forward
            out = self.model(input_ids)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            # Shift targets
            preds = logits[:, :-1, :]
            targets = input_ids[:, 1:]

            loss = nn.losses.cross_entropy(preds, targets)
            nlls.append(mx.mean(loss).item())

        ppl = np.exp(np.mean(nlls))
        logger.info(f"Perplexity: {ppl:.4f}")

    def run_latency(self):
        """Run standard latency sweep."""
        logger.info("Running Latency Sweep...")
        import time

        contexts = [128, 512, 1024, 2048, 4096]

        print(f"{'Context':<10} | {'Prefill TPS':<15}")
        print("-" * 30)

        for ctx in contexts:
            tokens = mx.random.randint(0, 1000, (1, ctx))

            # Warmup
            mx.eval(self.model(tokens))

            start = time.perf_counter()
            mx.eval(self.model(tokens))
            end = time.perf_counter()

            dur = end - start
            tps = ctx / dur
            print(f"{ctx:<10} | {tps:<15.2f}")

    def run_memory(self):
        """Check peak memory."""
        logger.info("Running Memory Benchmark...")
        mx.metal.reset_peak_memory()

        tokens = mx.random.randint(0, 1000, (1, 2048))
        mx.eval(self.model(tokens))

        peak = mx.metal.get_peak_memory() / 1024**3
        logger.info(f"Peak Memory (2048 ctx): {peak:.2f} GB")


def generate_comparison_report(base_dir: str = "artifacts"):
    """Generate a side-by-side comparison log for Teacher vs Student."""
    base_path = Path(base_dir)
    teacher_path = base_path / "bench_Teacher" / "behavior_results.json"
    student_path = base_path / "bench_Student" / "behavior_results.json"

    if not teacher_path.exists() or not student_path.exists():
        return

    logger.info("Generating comparison report...")

    with open(teacher_path, "r") as f:
        t_results = json.load(f)
    with open(student_path, "r") as f:
        s_results = json.load(f)

    # Assume aligned (same YAML source)
    with open(base_path / "behavior_comparison.log", "w") as f:
        f.write("=== BEHAVIOR BENCHMARK COMPARISON ===\n")
        f.write(f"Teacher: {teacher_path}\n")
        f.write(f"Student: {student_path}\n")
        f.write("=" * 60 + "\n\n")

        for t_res, s_res in zip(t_results, s_results):
            # Sanity check alignment
            if t_res["id"] != s_res["id"]:
                f.write(f"WARNING: ID Mismatch {t_res['id']} vs {s_res['id']}\n")
                continue

            f.write(f"CASE: {t_res['id']}\n")
            f.write("-" * 20 + "\n")
            f.write(f"PROMPT:\n{t_res.get('prompt', '').strip()}\n")
            f.write("-" * 20 + "\n")
            f.write(f"EXPECTED: {t_res['expected']}\n")
            f.write(
                f"TEACHER:  {t_res['generated']}  [{'MATCH' if t_res['match'] else 'FAIL'}]\n"
            )
            f.write(
                f"STUDENT:  {s_res['generated']}  [{'MATCH' if s_res['match'] else 'FAIL'}]\n"
            )
            f.write("\n" + "=" * 60 + "\n\n")

    logger.info(f"Comparison log saved to {base_path / 'behavior_comparison.log'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--teacher-weights", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="winogrande,arc_easy")
    parser.add_argument("--include-teacher", action="store_true")

    # DBA model config (student)
    parser.add_argument("--sem-head-dim", type=int, default=8)
    parser.add_argument("--geo-head-dim", type=int, default=32)
    parser.add_argument("--v-head-dim", type=int, default=None)
    parser.add_argument(
        "--init-mode",
        type=str,
        default="fresh",
        choices=["fresh", "copy_vo", "copy_vo_compress_qk", "copy_qkvo"],
    )

    # Modes
    parser.add_argument("--behavior", action="store_true")
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    tasks = args.tasks.split(",") if args.tasks else []

    wrappers: list[tuple[str, DBALMWrapper | TeacherWrapper]] = []
    if args.include_teacher:
        wrappers.append(("Teacher", TeacherWrapper(args.teacher_weights)))

    if args.checkpoint:
        wrappers.append(
            (
                "Student",
                DBALMWrapper(
                    args.checkpoint,
                    args.teacher_weights,
                    sem_head_dim=args.sem_head_dim,
                    geo_head_dim=args.geo_head_dim,
                    v_head_dim=args.v_head_dim,
                    init_mode=args.init_mode,
                ),
            )
        )

    # 1. Standard Eval Tasks
    if tasks and tasks != [""]:
        print("\n=== Standard Benchmarks (accuracy) ===")
        for name, wrapper in wrappers:
            res = simple_evaluate(wrapper, tasks=tasks, num_fewshot=0, device="mps")
            print(f"Results for {name}:")
            for t, mets in res["results"].items():
                print(f"  {t}: {mets}")

    # 2. Extended Suite
    for name, wrapper in wrappers:
        if any([args.behavior, args.perplexity, args.latency, args.memory]):
            print(f"\n=== Extended Suite for {name} ===")
            suite = BenchmarkSuite(
                wrapper.model, wrapper.tokenizer, f"artifacts/bench_{name}"
            )

            if args.memory:
                suite.run_memory()
            if args.latency:
                suite.run_latency()
            if args.perplexity:
                suite.run_perplexity()
            if args.behavior:
                suite.run_behavior(
                    "research/dba/behavior_cases.yml", visualize=args.visualize
                )

    # 3. Generate Comparison Report
    if args.behavior and args.include_teacher and args.checkpoint:
        generate_comparison_report()


if __name__ == "__main__":
    main()
