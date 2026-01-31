"""
Experiment Registry v1 (paper-aligned)
======================================

This module defines a stable experiment registry for the Resonant Compression
Systems paper. The registry is the single source of truth for:

- stable experiment IDs
- configuration schemas (dataclasses)
- metrics to log
- acceptance checks (reported, never used to manipulate results)
- which figure/table each experiment populates

CRITICAL INTEGRITY RULES
------------------------
- No cherry-picking: fixed seed lists per experiment
- No "manager" logic: checks are reporting only (PASS/FAIL), never used to
  re-run, filter, or alter outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ArtifactKind = Literal["figure", "table", "tex"]


@dataclass(frozen=True)
class ArtifactSpec:
    """Declares an output artifact produced by an experiment."""

    kind: ArtifactKind
    # Filename relative to tmp/rez/artifacts/
    path: str
    # Human label to reference from paper text
    label: str


@dataclass(frozen=True)
class ExperimentMeta:
    """Registry metadata (paper-facing)."""

    exp_id: str  # e.g. "EXP-A"
    name: str
    goal: str
    paper_outputs: tuple[ArtifactSpec, ...]


@dataclass(frozen=True)
class MetricSpec:
    """Declares a metric key the experiment must report."""

    key: str
    description: str


@dataclass(frozen=True)
class AcceptanceCheck:
    """Declares a PASS/FAIL predicate (reporting only)."""

    check_id: str
    predicate: str


@dataclass(frozen=True)
class ExperimentSpec:
    """Full registry entry (implementation + paper alignment)."""

    meta: ExperimentMeta
    config_cls: type["CommonInstrumentationConfig"]
    metrics: tuple[MetricSpec, ...]
    acceptance: tuple[AcceptanceCheck, ...]


# -----------------------
# Config schemas (v1)
# -----------------------


@dataclass(frozen=True)
class CommonInstrumentationConfig:
    sim_s: float = 20.0
    log_every_steps: int = 20
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class ExpATuningConfig(CommonInstrumentationConfig):
    # Sweep Δφ in [-π, π]
    n_points: int = 401
    two_carrier_centers: tuple[float, float] = (0.0, 0.8)


@dataclass(frozen=True)
class ExpBGateConfig(CommonInstrumentationConfig):
    n_cycles: int = 3
    samples_per_cycle: int = 200


@dataclass(frozen=True)
class ExpCGenesisConfig(CommonInstrumentationConfig):
    horizon_s: float = 8.0
    n_signals: int = 8
    amplitude: float = 1.0
    freq_hz: float = 1.5
    clustered_phase_sigma: float = 0.08


@dataclass(frozen=True)
class ExpDBondConfig(CommonInstrumentationConfig):
    warmup_s: float = 12.0
    cool_s: float = 12.0


@dataclass(frozen=True)
class ExpEMitosisPurityConfig(CommonInstrumentationConfig):
    horizon_s: float = 30.0
    n_low: int = 6
    n_high: int = 6
    low_freq_hz: float = 1.2
    high_freq_hz: float = 3.2
    amp: float = 1.0
    dur_s: float = 6.0


@dataclass(frozen=True)
class ExpFCoherenceMetabolismConfig(CommonInstrumentationConfig):
    horizon_s: float = 25.0
    n_signals: int = 10
    freq_hz: float = 2.0
    amp: float = 1.0
    dur_s: float = 6.0


@dataclass(frozen=True)
class ExpGCompressionScalingConfig(CommonInstrumentationConfig):
    horizon_s: float = 35.0
    target_N: tuple[int, ...] = (10, 50, 200)


@dataclass(frozen=True)
class ExpHStabilityMapConfig(CommonInstrumentationConfig):
    horizon_s: float = 20.0
    # Sweep a small set of knobs that exist today (gate adaptation rates)
    gate_narrow_rates: tuple[float, ...] = (0.05, 0.1, 0.2)
    gate_widen_rates: tuple[float, ...] = (0.02, 0.05, 0.1)


@dataclass(frozen=True)
class ExpITriggerAuditConfig(CommonInstrumentationConfig):
    horizon_s: float = 30.0


# -----------------------
# Registry (paper-aligned)
# -----------------------


REGISTRY: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-A",
            name="Radio Dial Tuning Curve",
            goal="Validate graded coupling vs alignment and multi-carrier coupling.",
            paper_outputs=(
                ArtifactSpec(
                    "figure", "exp_a_tuning_curve.png", "tuning curve (T vs |u|)"
                ),
            ),
        ),
        config_cls=ExpATuningConfig,
        metrics=(
            MetricSpec("midband_present", "Whether 0.2 < |u|/|u_max| < 0.8 occurs."),
            MetricSpec("n_points", "Number of points in phase sweep."),
        ),
        acceptance=(
            AcceptanceCheck("A1", "|u| decreases with |Δφ| (monotone up to noise)."),
            AcceptanceCheck(
                "A2", "Non-trivial mid-band exists (0.2 < |u|/|u_max| < 0.8)."
            ),
            AcceptanceCheck(
                "A3",
                "Two-carrier case shows simultaneous nonzero coupling near centers.",
            ),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-B",
            name="Gate Capture Windows",
            goal="Verify gating controls when capture happens independent of tuning.",
            paper_outputs=(
                ArtifactSpec(
                    "figure", "exp_b_gate_windows.png", "gate windows vs captured drive"
                ),
            ),
        ),
        config_cls=ExpBGateConfig,
        metrics=(
            MetricSpec("gate_closed_max_u", "Max |u| when gate==0."),
            MetricSpec("gate_open_mean_u", "Mean |u| when gate==1."),
        ),
        acceptance=(
            AcceptanceCheck("B1", "When gate=0 then |u|≈0."),
            AcceptanceCheck(
                "B2", "When gate=1 then |u| follows tuning/phasor magnitude."
            ),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-C",
            name="Genesis Coherence Gating",
            goal="Show coherent unbound energy nucleates carriers more readily than incoherent energy at matched amplitude.",
            paper_outputs=(
                ArtifactSpec(
                    "figure",
                    "exp_c_genesis_coherence.png",
                    "time-to-birth + R_U traces",
                ),
            ),
        ),
        config_cls=ExpCGenesisConfig,
        metrics=(
            MetricSpec(
                "p_birth_clustered",
                "Probability of a birth within horizon (clustered).",
            ),
            MetricSpec(
                "p_birth_random", "Probability of a birth within horizon (random)."
            ),
        ),
        acceptance=(
            AcceptanceCheck(
                "C1",
                "Clustered condition has higher birth probability and/or lower median time-to-birth than random.",
            ),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-D",
            name="Bond Sustainment and Snap",
            goal="Bonds are active memory: decay and snap when not sustained.",
            paper_outputs=(
                ArtifactSpec(
                    "figure", "exp_d_bond_decay.png", "bond decay vs snap threshold"
                ),
            ),
        ),
        config_cls=ExpDBondConfig,
        metrics=(
            MetricSpec("snap_threshold", "Configured snap threshold."),
            MetricSpec("p_final", "Final tracked bond strength."),
        ),
        acceptance=(
            AcceptanceCheck(
                "D1",
                "After detune/remove drive: tracked P_ik decays toward 0 and snaps below threshold.",
            ),
            AcceptanceCheck("D2", "Under sustained alignment: P_ik remains elevated."),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-E",
            name="Spectral Mitosis Purity",
            goal="Mitosis partitions incompatible wavelength clusters into distinct carriers.",
            paper_outputs=(
                ArtifactSpec(
                    "figure",
                    "exp_e_mitosis_purity.png",
                    "before/after bonds + spectral centers",
                ),
                ArtifactSpec(
                    "table", "exp_e_mitosis_table.tex", "mitosis event purity summary"
                ),
            ),
        ),
        config_cls=ExpEMitosisPurityConfig,
        metrics=(MetricSpec("n_events", "Number of mitosis events observed."),),
        acceptance=(
            AcceptanceCheck("E1", "Partition purities exceed threshold across seeds."),
            AcceptanceCheck(
                "E2",
                "Post-split coherence improves (ΔD > 0) and/or intake_eff improves.",
            ),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-F",
            name="Coherence-weighted Metabolism",
            goal="Coherent energy sustains carriers better than incoherent energy at matched raw intake.",
            paper_outputs=(
                ArtifactSpec("figure", "exp_f_survival.png", "survival curves"),
            ),
        ),
        config_cls=ExpFCoherenceMetabolismConfig,
        metrics=(
            MetricSpec("coherent.eff_intake_mean", "Mean effective intake (coherent)."),
            MetricSpec(
                "incoherent.eff_intake_mean", "Mean effective intake (incoherent)."
            ),
            MetricSpec("coherent.deaths_mean", "Mean deaths (coherent)."),
            MetricSpec("incoherent.deaths_mean", "Mean deaths (incoherent)."),
        ),
        acceptance=(
            AcceptanceCheck(
                "F1",
                "At matched raw intake, incoherent condition shows lower effective intake.",
            ),
            AcceptanceCheck(
                "F2",
                "At matched raw intake, incoherent condition shows shorter lifetimes / more dissolution.",
            ),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-G",
            name="Compression vs Structure (Scaling)",
            goal="Structured environments yield better compression than random streams; scaling behavior remains bounded.",
            paper_outputs=(
                ArtifactSpec(
                    "figure", "exp_g_compression_scaling.png", "compression vs N"
                ),
            ),
        ),
        config_cls=ExpGCompressionScalingConfig,
        metrics=(
            MetricSpec("rows", "Per-(N,seed) structured vs random L_comp/N ratios."),
        ),
        acceptance=(
            AcceptanceCheck(
                "G1", "Structured environment yields lower L_comp/N than random."
            ),
            AcceptanceCheck(
                "G2", "Stable regime exists where L_comp does not explode over time."
            ),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-H",
            name="Synchrony Collapse Boundary (Stability Map)",
            goal="Identify operating region avoiding global sync collapse while compression remains non-trivial.",
            paper_outputs=(
                ArtifactSpec(
                    "figure", "exp_h_stability_map.png", "collapse probability heatmap"
                ),
            ),
        ),
        config_cls=ExpHStabilityMapConfig,
        metrics=(MetricSpec("rows", "Grid sweep rows with collapse probabilities."),),
        acceptance=(
            AcceptanceCheck(
                "H1",
                "A region exists where collapse probability is low and compression remains non-trivial.",
            ),
        ),
    ),
    ExperimentSpec(
        meta=ExperimentMeta(
            exp_id="EXP-I",
            name="Mitosis Trigger Audit",
            goal="Ensure mitosis events satisfy the documented predicate (audit, not optimization).",
            paper_outputs=(
                ArtifactSpec(
                    "table",
                    "exp_i_trigger_audit.tex",
                    "mitosis trigger compliance table",
                ),
            ),
        ),
        config_cls=ExpITriggerAuditConfig,
        metrics=(
            MetricSpec(
                "compliance",
                "Fraction of mitosis events satisfying documented trigger predicate.",
            ),
        ),
        acceptance=(
            AcceptanceCheck(
                "I1",
                "100% of mitoses satisfy the documented predicate (or predicate is revised).",
            ),
        ),
    ),
)


REGISTRY_BY_ID: dict[str, ExperimentSpec] = {
    spec.meta.exp_id: spec for spec in REGISTRY
}
