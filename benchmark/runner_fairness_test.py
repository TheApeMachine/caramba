from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

import benchmark.runner as runner_mod
from benchmark.artifacts import ExperimentMetadata
from benchmark.latency import LatencyMeasurement, LatencyResult
from benchmark.perplexity import PerplexityResult
from config.benchmark import (
    BenchmarkSpec,
    BenchmarkSuite,
    LatencyBenchmarkConfig,
    PerplexityBenchmarkConfig,
)


class _CaptureArtifacts:
    """Fake ArtifactGenerator that captures the inputs it was given."""

    last_kwargs: dict | None = None

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def generate_all(self, **kwargs):
        _CaptureArtifacts.last_kwargs = dict(kwargs)
        return {}

class TestBenchmarkRunnerFairness(unittest.TestCase):
    def test_runner_aggregates_perplexity_over_repeats_weighted_by_tokens(self) -> None:
        class FakePerplexityBenchmark:
            calls: dict[str, int] = {"teacher": 0, "student": 0}

            def __init__(self, cfg, device) -> None:
                self.cfg = cfg
                self.device = device

            def run(self, model: nn.Module, model_name: str) -> PerplexityResult:
                FakePerplexityBenchmark.calls[model_name] += 1
                i = FakePerplexityBenchmark.calls[model_name]

                # Give each repeat a different token count so we can validate weighted averaging.
                num_tokens = 10 * i
                loss = float(i) if model_name == "teacher" else float(2 * i)
                return PerplexityResult(
                    model_name=model_name,
                    perplexity=math.exp(loss),
                    loss=loss,
                    num_tokens=num_tokens,
                    num_batches=1,
                )

        device = torch.device("cpu")
        meta = ExperimentMetadata(
            name="test",
            timestamp="2024-12-26T12:00:00",
            manifest_path="/test/manifest.yml",
            teacher_checkpoint="t",
            student_config="s",
            device="cpu",
        )
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="ppl",
                    config=PerplexityBenchmarkConfig(
                        dataset="/dev/null", block_size=8, batch_size=1, num_batches=1
                    ),
                    models=["teacher", "student"],
                    repeats=3,
                )
            ],
            output_dir=tempfile.mkdtemp(),
            formats=["json"],
        )

        with (
            patch.object(runner_mod, "PerplexityBenchmark", FakePerplexityBenchmark),
            patch.object(runner_mod, "ArtifactGenerator", _CaptureArtifacts),
        ):
            runner = runner_mod.BenchmarkRunner(suite, device, meta)
            runner.run(nn.Linear(1, 1), nn.Linear(1, 1))

        self.assertEqual(FakePerplexityBenchmark.calls["teacher"], 3)
        self.assertEqual(FakePerplexityBenchmark.calls["student"], 3)
        self.assertIsNotNone(_CaptureArtifacts.last_kwargs)
        assert _CaptureArtifacts.last_kwargs is not None  # help type checker

        t = _CaptureArtifacts.last_kwargs["teacher_perplexity"]
        s = _CaptureArtifacts.last_kwargs["student_perplexity"]

        # Weighted mean loss:
        # teacher: losses [1,2,3], tokens [10,20,30] => (1*10 + 2*20 + 3*30)/60 = 140/60 = 7/3
        expected_t_loss = (1 * 10 + 2 * 20 + 3 * 30) / 60
        self.assertEqual(t.num_tokens, 60)
        self.assertEqual(t.num_batches, 3)
        self.assertAlmostEqual(t.loss, expected_t_loss, places=12)
        self.assertAlmostEqual(t.perplexity, math.exp(expected_t_loss), places=12)

        # student: losses [2,4,6], tokens [10,20,30] => (2*10 + 4*20 + 6*30)/60 = 280/60 = 14/3
        expected_s_loss = (2 * 10 + 4 * 20 + 6 * 30) / 60
        self.assertEqual(s.num_tokens, 60)
        self.assertEqual(s.num_batches, 3)
        self.assertAlmostEqual(s.loss, expected_s_loss, places=12)
        self.assertAlmostEqual(s.perplexity, math.exp(expected_s_loss), places=12)

    def test_runner_aggregates_latency_measurements_over_repeats(self) -> None:
        class FakeLatencyBenchmark:
            calls: dict[str, int] = {"teacher": 0, "student": 0}

            def __init__(self, cfg: LatencyBenchmarkConfig, device) -> None:
                self.cfg = cfg
                self.device = device

            def run(self, model: nn.Module, model_name: str) -> LatencyResult:
                FakeLatencyBenchmark.calls[model_name] += 1
                i = FakeLatencyBenchmark.calls[model_name]
                m = LatencyMeasurement(
                    prompt_len=1,
                    gen_len=1,
                    batch_size=1,
                    prefill_time_ms=1.0,
                    decode_time_ms=1.0,
                    total_time_ms=2.0,
                    tokens_per_second=float(i),
                    time_to_first_token_ms=1.0,
                    use_cache=False,
                )
                return LatencyResult(model_name=model_name, measurements=[m])

        device = torch.device("cpu")
        meta = ExperimentMetadata(
            name="test",
            timestamp="2024-12-26T12:00:00",
            manifest_path="/test/manifest.yml",
            teacher_checkpoint="t",
            student_config="s",
            device="cpu",
        )
        suite = BenchmarkSuite(
            benchmarks=[
                BenchmarkSpec(
                    id="lat",
                    config=LatencyBenchmarkConfig(
                        prompt_lengths=[1],
                        generation_lengths=[1],
                        batch_sizes=[1],
                        warmup_runs=1,
                        timed_runs=1,
                        use_cache=False,
                    ),
                    models=["teacher", "student"],
                    repeats=4,
                )
            ],
            output_dir=tempfile.mkdtemp(),
            formats=["json"],
        )

        with (
            patch.object(runner_mod, "LatencyBenchmark", FakeLatencyBenchmark),
            patch.object(runner_mod, "ArtifactGenerator", _CaptureArtifacts),
        ):
            runner = runner_mod.BenchmarkRunner(suite, device, meta)
            runner.run(nn.Linear(1, 1), nn.Linear(1, 1))

        self.assertEqual(FakeLatencyBenchmark.calls["teacher"], 4)
        self.assertEqual(FakeLatencyBenchmark.calls["student"], 4)
        self.assertIsNotNone(_CaptureArtifacts.last_kwargs)
        assert _CaptureArtifacts.last_kwargs is not None  # help type checker

        t = _CaptureArtifacts.last_kwargs["teacher_latency"]
        s = _CaptureArtifacts.last_kwargs["student_latency"]

        self.assertEqual(len(t.measurements), 4)
        self.assertEqual(len(s.measurements), 4)
        # avg_tokens_per_second is mean of [1,2,3,4] = 2.5 for each model
        self.assertAlmostEqual(t.avg_tokens_per_second, 2.5, places=12)
        self.assertAlmostEqual(s.avg_tokens_per_second, 2.5, places=12)


if __name__ == "__main__":
    unittest.main()

