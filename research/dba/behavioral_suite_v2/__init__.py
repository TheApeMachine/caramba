"""
DBA Behavioral Test Suite v2

Comprehensive behavioral evaluation framework for comparing attention architectures.

Features:
- 500+ parameterized test cases across 18 categories
- Template-based generation with randomization
- Adversarial prompt attack suite
- Multi-model comparison (arbitrary number of models)
- Rich scoring: binary, soft, and attention diagnostics
- Visualization dashboard with normalized attention heatmaps

Quick Start (using manifest):
    from behavioral_suite_v2 import generate_suite, EvalRunner, EvalConfig
    from behavioral_suite_v2.cli import load_models_from_manifest, CarambaModelWrapper
    import tiktoken

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load models from caramba manifest
    raw_models, metadata = load_models_from_manifest("benchmark.yml")
    models = {
        name: CarambaModelWrapper(model, tokenizer, metadata["device"])
        for name, model in raw_models.items()
    }

    # Generate test suite
    suite = generate_suite(seed=42, tests_per_category=30)

    # Run evaluation
    runner = EvalRunner(models, EvalConfig(capture_attention=True))
    results = runner.run(suite.tests)

    # Save results
    results.save(Path('./results'))
"""

# Import only template types and generator (no torch dependency)
from .templates import (
    TestCase,
    TestTemplate,
    Difficulty,
    TargetPosition,
    EvalKind,
    ALL_TEMPLATES,
)

from .generator import (
    generate_suite,
    GeneratedSuite,
    GenerationConfig,
    SuiteGenerator,
)

__version__ = "2.0.0"

# Lazy imports for torch-dependent modules
def __getattr__(name):
    """Lazy import for torch-dependent modules."""
    if name in ('SoftScore', 'DiagnosticFlags', 'AttentionMetrics', 'ModelOutput',
                'TestScore', 'ComparisonResult', 'Scorer', 'MultiModelScorer'):
        from . import scoring
        return getattr(scoring, name)

    if name in ('EvalConfig', 'EvalResults', 'EvalRunner'):
        from . import runner
        return getattr(runner, name)

    if name in ('AttentionVisualizer', 'ResultsVisualizer', 'generate_html_report'):
        from . import visualizer
        return getattr(visualizer, name)

    if name in ('load_models_from_manifest', 'CarambaModelWrapper'):
        from . import cli
        return getattr(cli, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Generator (no torch)
    "generate_suite",
    "GeneratedSuite",
    "GenerationConfig",
    "SuiteGenerator",
    # Templates (no torch)
    "TestCase",
    "TestTemplate",
    "Difficulty",
    "TargetPosition",
    "EvalKind",
    "ALL_TEMPLATES",
    # Scoring (lazy - requires torch)
    "SoftScore",
    "DiagnosticFlags",
    "AttentionMetrics",
    "ModelOutput",
    "TestScore",
    "ComparisonResult",
    "Scorer",
    "MultiModelScorer",
    # Runner (lazy - requires torch)
    "EvalConfig",
    "EvalResults",
    "EvalRunner",
    # Visualizer (lazy - requires torch for attention)
    "AttentionVisualizer",
    "ResultsVisualizer",
    "generate_html_report",
    # Model loading (lazy - requires torch)
    "load_models_from_manifest",
    "CarambaModelWrapper",
]
