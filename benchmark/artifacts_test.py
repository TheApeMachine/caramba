"""
Unit tests for the artifacts generation module.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from .artifacts import (
    ArtifactGenerator,
    ComparisonSummary,
    ExperimentMetadata,
)
from .latency import LatencyMeasurement, LatencyResult
from .memory import KVCacheAnalysis, MemoryMeasurement, MemoryResult
from .perplexity import PerplexityResult
from .behavior import BehaviorResult, BehaviorMeasurement
from research.dba.behavioral_suite_v2.weighted_scoring import MatchType
from collector.measurement.context.result import ContextResult
from collector.measurement.context.sweep import ContextSweepMeasurement


class TestExperimentMetadata(unittest.TestCase):
    """Tests for ExperimentMetadata dataclass."""

    def test_metadata_fields(self) -> None:
        """Metadata stores all required fields."""
        meta = ExperimentMetadata(
            name="test_experiment",
            timestamp="2024-12-26T12:00:00",
            manifest_path="/path/to/manifest.yml",
            teacher_checkpoint="hf://meta-llama/Llama-3.2-1B",
            student_config="DBA",
            device="mps",
            notes="Test experiment",
        )
        self.assertEqual(meta.name, "test_experiment")
        self.assertEqual(meta.timestamp, "2024-12-26T12:00:00")
        self.assertEqual(meta.manifest_path, "/path/to/manifest.yml")
        self.assertEqual(meta.teacher_checkpoint, "hf://meta-llama/Llama-3.2-1B")
        self.assertEqual(meta.student_config, "DBA")
        self.assertEqual(meta.device, "mps")
        self.assertEqual(meta.notes, "Test experiment")


class TestComparisonSummary(unittest.TestCase):
    """Tests for ComparisonSummary dataclass."""

    def test_summary_fields(self) -> None:
        """Summary stores all required fields."""
        summary = ComparisonSummary(
            teacher_perplexity=8.5,
            student_perplexity=8.7,
            perplexity_ratio=1.02,
            teacher_tokens_per_sec=150.0,
            student_tokens_per_sec=225.0,
            speedup=1.5,
            # Bytes per token: 2.0 MB = 2.0 * 1024 * 1024 bytes
            teacher_kvcache_bytes_per_token=2.0 * 1024 * 1024,
            student_kvcache_bytes_per_token=0.4 * 1024 * 1024,
            memory_reduction=5.0,
        )
        self.assertAlmostEqual(summary.teacher_perplexity, 8.5)
        self.assertAlmostEqual(summary.student_perplexity, 8.7)
        self.assertAlmostEqual(summary.perplexity_ratio, 1.02)
        self.assertAlmostEqual(summary.speedup, 1.5)
        self.assertAlmostEqual(summary.memory_reduction, 5.0)


class TestArtifactGenerator(unittest.TestCase):
    """Tests for ArtifactGenerator."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ArtifactGenerator(self.temp_dir)

        self.metadata = ExperimentMetadata(
            name="test",
            timestamp="2024-12-26T12:00:00",
            manifest_path="/test/manifest.yml",
            teacher_checkpoint="test_ckpt",
            student_config="DBA",
            device="cpu",
        )

        self.teacher_ppl = PerplexityResult(
            model_name="teacher",
            perplexity=8.5,
            loss=2.14,
            num_tokens=10000,
            num_batches=100,
        )

        self.student_ppl = PerplexityResult(
            model_name="student",
            perplexity=8.7,
            loss=2.16,
            num_tokens=10000,
            num_batches=100,
        )

        self.teacher_latency = LatencyResult(
            model_name="teacher",
            measurements=[
                LatencyMeasurement(
                    prompt_len=128, gen_len=64, batch_size=1,
                    prefill_time_ms=10.0, decode_time_ms=40.0,
                    total_time_ms=50.0, tokens_per_second=150.0,
                    time_to_first_token_ms=10.0,
                ),
            ],
        )

        self.student_latency = LatencyResult(
            model_name="student",
            measurements=[
                LatencyMeasurement(
                    prompt_len=128, gen_len=64, batch_size=1,
                    prefill_time_ms=8.0, decode_time_ms=30.0,
                    total_time_ms=38.0, tokens_per_second=225.0,
                    time_to_first_token_ms=8.0,
                ),
            ],
        )

        self.teacher_memory = MemoryResult(
            model_name="teacher",
            measurements=[
                MemoryMeasurement(
                    seq_len=512, batch_size=1,
                    peak_memory_mb=1000.0, kvcache_memory_mb=200.0,
                    model_memory_mb=800.0, quantization="fp16",
                ),
            ],
            kvcache_analysis=KVCacheAnalysis(
                model_name="teacher",
                n_layers=16,
                n_kv_heads=8,
                head_dim=64,
                attention_mode="standard",
                bytes_per_token_fp16=2048.0,
                bytes_per_token_q8=1024.0,
                bytes_per_token_q4=512.0,
            ),
        )

        self.student_memory = MemoryResult(
            model_name="student",
            measurements=[
                MemoryMeasurement(
                    seq_len=512, batch_size=1,
                    peak_memory_mb=900.0, kvcache_memory_mb=50.0,
                    model_memory_mb=800.0, quantization="fp16",
                ),
            ],
            kvcache_analysis=KVCacheAnalysis(
                model_name="student",
                n_layers=16,
                n_kv_heads=8,
                head_dim=64,
                attention_mode="decoupled",
                bytes_per_token_fp16=2048.0,
                bytes_per_token_q8=1024.0,
                bytes_per_token_q4=512.0,
                sem_dim=128,
                geo_dim=256,
                bytes_per_token_dba_fp16=384.0,
            ),
        )

    def tearDown(self) -> None:
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_output_dir_created(self) -> None:
        """Output directory is created."""
        self.assertTrue(Path(self.temp_dir).exists())

    def test_generate_json_report(self) -> None:
        """JSON report is generated correctly."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_perplexity=self.teacher_ppl,
            student_perplexity=self.student_ppl,
            formats=["json"],
        )

        self.assertIn("report.json", paths)
        report_path = paths["report.json"]
        self.assertTrue(report_path.exists())

        with open(report_path) as f:
            report = json.load(f)

        self.assertIn("metadata", report)
        self.assertIn("summary", report)
        self.assertEqual(report["metadata"]["name"], "test")

    def test_generate_csv_perplexity(self) -> None:
        """Perplexity CSV is generated correctly."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_perplexity=self.teacher_ppl,
            student_perplexity=self.student_ppl,
            formats=["csv"],
        )

        self.assertIn("perplexity.csv", paths)
        csv_path = paths["perplexity.csv"]
        self.assertTrue(csv_path.exists())

        content = csv_path.read_text()
        self.assertIn("model,perplexity,loss,num_tokens", content)
        self.assertIn("teacher", content)
        self.assertIn("student", content)

    def test_generate_csv_latency(self) -> None:
        """Latency CSV is generated correctly."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_latency=self.teacher_latency,
            student_latency=self.student_latency,
            formats=["csv"],
        )

        self.assertIn("latency.csv", paths)
        csv_path = paths["latency.csv"]
        self.assertTrue(csv_path.exists())

        content = csv_path.read_text()
        self.assertIn("model,prompt_len,gen_len", content)
        self.assertIn("teacher", content)
        self.assertIn("student", content)
        # Check for use_cache column
        self.assertIn("use_cache", content)

    def test_generate_csv_memory(self) -> None:
        """Memory CSV is generated correctly."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_memory=self.teacher_memory,
            student_memory=self.student_memory,
            formats=["csv"],
        )

        self.assertIn("memory.csv", paths)
        csv_path = paths["memory.csv"]
        self.assertTrue(csv_path.exists())

        content = csv_path.read_text()
        self.assertIn("model,seq_len,batch_size", content)
        self.assertIn("teacher", content)
        self.assertIn("student", content)

    def test_generate_latex_tables(self) -> None:
        """LaTeX tables are generated correctly."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_perplexity=self.teacher_ppl,
            student_perplexity=self.student_ppl,
            formats=["latex"],
        )

        self.assertIn("tables.tex", paths)
        tex_path = paths["tables.tex"]
        self.assertTrue(tex_path.exists())

        content = tex_path.read_text()
        self.assertIn("\\begin{table}", content)
        self.assertIn("\\end{table}", content)
        self.assertIn("Perplexity", content)

    def test_generate_behavior_summary_latex_table(self) -> None:
        """Behavior summary LaTeX is generated when behavior is present."""
        behavior = BehaviorResult(
            benchmark_id="b",
            measurements=[
                BehaviorMeasurement(
                    case_id="math_add_single",
                    teacher_ok=False,
                    student_ok=True,
                    teacher_answer="1",
                    student_answer="2",
                    teacher_match_type=MatchType.NONE,
                    student_match_type=MatchType.EXACT,
                    teacher_raw_score=0.0,
                    student_raw_score=1.0,
                    difficulty_weight=3.0,
                    student_weighted_score=3.0,
                ),
                BehaviorMeasurement(
                    case_id="copy_simple_3",
                    teacher_ok=True,
                    student_ok=True,
                    teacher_answer="MNO",
                    student_answer="MNO",
                    teacher_match_type=MatchType.EXACT,
                    student_match_type=MatchType.CONTAINED,
                    teacher_raw_score=1.0,
                    student_raw_score=0.5,
                    difficulty_weight=1.0,
                    student_weighted_score=0.5,
                ),
            ],
        )

        paths = self.generator.generate_all(
            metadata=self.metadata,
            behavior=behavior,
            formats=["latex"],
        )

        self.assertIn("behavior_results_100k_generated.tex", paths)
        tex_path = paths["behavior_results_100k_generated.tex"]
        self.assertTrue(tex_path.exists())
        content = tex_path.read_text()
        self.assertIn("\\label{tab:behavior_results_100k}", content)

    def test_generate_all_formats(self) -> None:
        """All formats are generated when requested."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_perplexity=self.teacher_ppl,
            student_perplexity=self.student_ppl,
            teacher_latency=self.teacher_latency,
            student_latency=self.student_latency,
            teacher_memory=self.teacher_memory,
            student_memory=self.student_memory,
            formats=["csv", "json", "latex"],
        )

        self.assertIn("report.json", paths)
        self.assertIn("perplexity.csv", paths)
        self.assertIn("latency.csv", paths)
        self.assertIn("memory.csv", paths)
        self.assertIn("tables.tex", paths)

    def test_generate_context_sweep_csv_includes_telemetry_fields(self) -> None:
        """Context CSV headers include optional telemetry when present."""
        ctx = ContextResult(
            model_name="teacher",
            sweep=[
                ContextSweepMeasurement(
                    context_len=128,
                    chunk_size_used=32,
                    batch_size=1,
                    prefill_total_s=0.1,
                    prefill_last_chunk_ms=1.0,
                    decode_one_ms=2.0,
                    decode_one_tok_per_s=500.0,
                    loss=1.0,
                    ppl=2.0,
                    loss_last_chunk=1.0,
                    ppl_last_chunk=2.0,
                    ok=True,
                    rss_mb_before=123.0,
                    rss_mb_after=124.0,
                    mps_allocated_mb_before=10.0,
                    mps_allocated_mb_after=11.0,
                    mps_driver_allocated_mb_before=20.0,
                    mps_driver_allocated_mb_after=21.0,
                    mps_recommended_max_mb=64_000.0,
                )
            ],
            decode=[],
        )
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_context=ctx,
            formats=["csv"],
        )
        self.assertIn("context_sweep_teacher.csv", paths)
        content = paths["context_sweep_teacher.csv"].read_text()
        self.assertIn("rss_mb_before", content)
        self.assertIn("mps_allocated_mb_before", content)

    def test_generate_context_diagnostics_csv(self) -> None:
        """Consolidated context diagnostics CSV is generated."""
        ctx = ContextResult(
            model_name="teacher",
            sweep=[
                ContextSweepMeasurement(
                    context_len=128,
                    chunk_size_used=32,
                    batch_size=1,
                    prefill_total_s=0.1,
                    prefill_last_chunk_ms=1.0,
                    decode_one_ms=2.0,
                    decode_one_tok_per_s=500.0,
                    loss=1.0,
                    ppl=2.0,
                    loss_last_chunk=1.0,
                    ppl_last_chunk=2.0,
                    ok=True,
                )
            ],
            decode=[],
        )
        # Add decode measurement with telemetry (fields are optional but should carry through).
        from collector.measurement.context.decode import ContextDecodeMeasurement

        ctx.decode.append(
            ContextDecodeMeasurement(
                context_len=128,
                chunk_size_used=32,
                batch_size=1,
                decode_len=8,
                decode_warmup=1,
                prefill_total_s=0.1,
                decode_total_ms=10.0,
                decode_tok_per_s=800.0,
                ok=True,
                rss_mb_before=123.0,
                rss_mb_after=124.0,
                mps_driver_allocated_mb_after=42.0,
                mps_recommended_max_mb=1024.0,
            )
        )

        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_context=ctx,
            formats=["csv"],
        )
        self.assertIn("context_diagnostics.csv", paths)
        content = paths["context_diagnostics.csv"].read_text()
        self.assertIn("decode_tok_per_s", content)
        self.assertIn("mps_driver_allocated_mb_after", content)

    def test_summary_computation(self) -> None:
        """Summary is computed correctly from results."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            teacher_perplexity=self.teacher_ppl,
            student_perplexity=self.student_ppl,
            teacher_latency=self.teacher_latency,
            student_latency=self.student_latency,
            formats=["json"],
        )

        with open(paths["report.json"]) as f:
            report = json.load(f)

        summary = report["summary"]
        self.assertAlmostEqual(summary["teacher_perplexity"], 8.5)
        self.assertAlmostEqual(summary["student_perplexity"], 8.7)
        self.assertAlmostEqual(summary["perplexity_ratio"], 8.7 / 8.5, places=2)

    def test_empty_results_handled(self) -> None:
        """Empty results don't cause errors."""
        paths = self.generator.generate_all(
            metadata=self.metadata,
            formats=["json"],
        )

        self.assertIn("report.json", paths)
        with open(paths["report.json"]) as f:
            report = json.load(f)

        summary = report["summary"]
        self.assertAlmostEqual(summary["teacher_perplexity"], 0.0)


class TestComparisonSummaryMBConversion(unittest.TestCase):
    """Tests for ComparisonSummary MB per token conversion."""

    def test_mb_per_token_conversion(self) -> None:
        """MB per token is correctly derived from bytes per token."""
        # 2 MB = 2 * 1024 * 1024 bytes = 2097152 bytes
        bytes_per_token = 2 * 1024 * 1024
        summary = ComparisonSummary(
            teacher_perplexity=8.5,
            student_perplexity=8.7,
            perplexity_ratio=1.02,
            teacher_tokens_per_sec=150.0,
            student_tokens_per_sec=225.0,
            speedup=1.5,
            teacher_kvcache_bytes_per_token=bytes_per_token,
            student_kvcache_bytes_per_token=bytes_per_token / 5,
            memory_reduction=5.0,
        )
        self.assertAlmostEqual(summary.teacher_kvcache_mb_per_token, 2.0)
        self.assertAlmostEqual(summary.student_kvcache_mb_per_token, 0.4)

    def test_raw_bytes_preserved(self) -> None:
        """Raw bytes per token is preserved exactly."""
        summary = ComparisonSummary(
            teacher_perplexity=8.5,
            student_perplexity=8.7,
            perplexity_ratio=1.02,
            teacher_tokens_per_sec=150.0,
            student_tokens_per_sec=225.0,
            speedup=1.5,
            teacher_kvcache_bytes_per_token=16384.0,
            student_kvcache_bytes_per_token=3276.8,
            memory_reduction=5.0,
        )
        self.assertAlmostEqual(summary.teacher_kvcache_bytes_per_token, 16384.0)
        self.assertAlmostEqual(summary.student_kvcache_bytes_per_token, 3276.8)


class TestLatencyMeasurementUseCacheField(unittest.TestCase):
    """Tests for use_cache field in LatencyMeasurement."""

    def test_latency_measurement_with_use_cache(self) -> None:
        """LatencyMeasurement includes use_cache field."""
        m = LatencyMeasurement(
            prompt_len=128, gen_len=64, batch_size=1,
            prefill_time_ms=10.0, decode_time_ms=40.0,
            total_time_ms=50.0, tokens_per_second=150.0,
            time_to_first_token_ms=12.0,
            use_cache=True,
        )
        self.assertTrue(m.use_cache)

    def test_latency_measurement_use_cache_default(self) -> None:
        """LatencyMeasurement use_cache defaults to False."""
        m = LatencyMeasurement(
            prompt_len=128, gen_len=64, batch_size=1,
            prefill_time_ms=10.0, decode_time_ms=40.0,
            total_time_ms=50.0, tokens_per_second=150.0,
            time_to_first_token_ms=10.0,
        )
        self.assertFalse(m.use_cache)


if __name__ == "__main__":
    unittest.main()
