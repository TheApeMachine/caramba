from __future__ import annotations

from caramba.config.component import ComponentSpec
from caramba.config.defaults import Defaults
from caramba.config.manifest import Manifest
from caramba.config.mode import Mode
from caramba.config.run import Run
from caramba.config.target import ExperimentTargetConfig
from caramba.config.train import TrainConfig, TrainPhase
from caramba.compiler.lower import Lowerer


def test_lower_manifest_lowers_language_model_topology_repeat() -> None:
    train = TrainConfig(phase=TrainPhase.STANDARD, batch_size=2, block_size=4, lr=1e-3, device="cpu")
    run = Run(id="r", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=train)
    t = ExperimentTargetConfig(
        name="exp",
        backend="torch",
        task=ComponentSpec(ref="task.language_modeling"),
        data=ComponentSpec(ref="dataset.tokens", config={"path": "x.tokens", "block_size": 4}),
        system=ComponentSpec(
            ref="system.language_model",
            config={
                "model": {
                    "type": "TransformerModel",
                    "topology": {
                        "type": "StackedTopology",
                        "repeat": 2,
                        "layers": [{"type": "LinearLayer", "d_in": 8, "d_out": 8, "bias": True}],
                    },
                }
            },
        ),
        objective=ComponentSpec(ref="objective.next_token_ce"),
        trainer=ComponentSpec(ref="trainer.standard"),
        runs=[run],
    )
    m = Manifest(version=2, defaults=Defaults(), targets=[t])

    lowered = Lowerer().lower_manifest(m)
    t2 = lowered.targets[0]
    assert isinstance(t2, ExperimentTargetConfig)
    topo = t2.system.config["model"]["topology"]
    assert topo["repeat"] == 1
    assert len(topo["layers"]) == 2


def test_lower_manifest_lowers_graph_topology_payload() -> None:
    train = TrainConfig(phase=TrainPhase.STANDARD, batch_size=2, block_size=4, lr=1e-3, device="cpu")
    run = Run(id="r", mode=Mode.TRAIN, exp="e", seed=0, steps=1, expected={}, train=train)
    t = ExperimentTargetConfig(
        name="g",
        backend="torch",
        task=ComponentSpec(ref="task.graph"),
        data=ComponentSpec(ref="dataset.graph_npy", config={"path": "x.npy"}),
        system=ComponentSpec(
            ref="system.generic",
            config={
                "model": {
                    "type": "MLPModel",
                    "topology": {
                        "type": "GraphTopology",
                        "nodes": [{"id": "n", "op": "mlp", "in": "x", "out": "y", "repeat": 1, "config": {}}],
                    },
                },
            },
        ),
        objective=ComponentSpec(ref="objective.mse"),
        trainer=ComponentSpec(ref="trainer.standard"),
        runs=[run],
    )
    m = Manifest(version=2, defaults=Defaults(), targets=[t])
    lowered = Lowerer().lower_manifest(m)
    t2 = lowered.targets[0]
    assert isinstance(t2, ExperimentTargetConfig)
    topo = t2.system.config["model"]["topology"]
    # Ensure aliasing kept keys as "in"/"out" in the dumped payload.
    assert topo["nodes"][0]["in"] == "x"
    assert topo["nodes"][0]["out"] == "y"

