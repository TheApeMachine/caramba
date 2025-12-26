"""Optimizer Orchestration Layer.

This module implements an online algorithm selection system that can dynamically
switch between training strategies during a single run. The key insight is that
the optimal optimizer/scheduler/clipping configuration is non-stationary—what
works best early in training may not be optimal later.

Architecture Overview:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                        Orchestrator                                  │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
    │  │ Telemetry  │→ │   Bandit   │→ │ Evaluator  │→ │ Checkpoint │     │
    │  │  Stream    │  │  Selector  │  │  (A/B)     │  │  Manager   │     │
    │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
    │         ↓              ↓              ↓              ↓               │
    │  ┌─────────────────────────────────────────────────────────────┐    │
    │  │                    Strategy Portfolio                        │    │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │    │
    │  │  │ AdamW   │ │  SWATS  │ │ PIDAO   │ │ AdaGC   │ │ Custom │ │    │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ │    │
    │  └─────────────────────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────────────────────┘

Key Features:

1. **Strategy Bundles**: Each strategy is a composable bundle with:
   - Base optimizer (AdamW, SGD, PIDAO, etc.)
   - LR scheduler/controller
   - Stability wrappers (gradient clipping, AdaGC)
   - Optional acceleration modules (weight nowcasting)

2. **Telemetry Stream**: Continuous monitoring of:
   - Smoothed loss, loss slope, loss variance
   - Gradient norm statistics (global and per-block)
   - Spike detection (count, magnitude)
   - Curvature proxies (optional)

3. **Bandit-based Selection**: Thompson Sampling or UCB for:
   - Online strategy selection under non-stationarity
   - Balancing exploration vs exploitation
   - Adapting to training phase (early/mid/late)

4. **Speculative Branching**: For single-GPU training:
   - Checkpoint at decision boundaries
   - Fork K candidates, run short horizons
   - Score on probe set, pick winner
   - Rollback if all degrade

5. **Safety + Rollback**:
   - Loss explosion detection
   - NaN/Inf guards
   - Automatic fallback to stable strategy

References:
- PBT: https://arxiv.org/abs/1711.09846
- SWATS: https://arxiv.org/abs/1712.07628
- PIDAO: https://www.nature.com/articles/s41467-024-54451-3
- AdaGC: https://arxiv.org/abs/2502.11034
- CDAT: https://arxiv.org/abs/2407.06183
"""
from orchestrator.strategy import (
    Strategy,
    StrategyBundle,
    StrategyState,
    OptimizerFamily,
    create_strategy,
    DEFAULT_PORTFOLIO,
)
from orchestrator.telemetry import (
    TelemetryStream,
    TelemetrySnapshot,
    SpikeDetector,
    TrainingPhase,
)
from orchestrator.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    DecisionBoundary,
)
from orchestrator.wrappers import (
    AdaGC,
    GradientSmoother,
    GradientNoiseInjector,
)
from orchestrator.swats import (
    SWATS,
    SWATSConfig,
    SWATSStrategy,
)
from orchestrator.pidao import (
    PIDAO,
    PIDAOConfig,
    PIDAOStrategy,
    AdaptivePIDAO,
)
from orchestrator.nowcast import (
    WeightNowcaster,
    NowcastConfig,
)

__all__ = [
    # Core
    "Orchestrator",
    "OrchestratorConfig",
    "DecisionBoundary",
    # Strategy
    "Strategy",
    "StrategyBundle",
    "StrategyState",
    "OptimizerFamily",
    "create_strategy",
    "DEFAULT_PORTFOLIO",
    # Telemetry
    "TelemetryStream",
    "TelemetrySnapshot",
    "SpikeDetector",
    "TrainingPhase",
    # Wrappers
    "AdaGC",
    "GradientSmoother",
    "GradientNoiseInjector",
    # SWATS
    "SWATS",
    "SWATSConfig",
    "SWATSStrategy",
    # PIDAO
    "PIDAO",
    "PIDAOConfig",
    "PIDAOStrategy",
    "AdaptivePIDAO",
    # Nowcasting
    "WeightNowcaster",
    "NowcastConfig",
]
