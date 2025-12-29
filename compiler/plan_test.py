from __future__ import annotations

from pathlib import Path

from compiler.plan import Planner
from config.manifest import Manifest


def test_planner_formats_experiment_target_with_topology(tmp_path: Path) -> None:
    path = tmp_path / "m.yml"
    path.write_text(
        "\n".join(
            [
                "version: 2",
                "name: test",
                "defaults: {}",
                "targets:",
                "  - type: experiment",
                "    name: exp",
                "    backend: torch",
                "    task: task.language_modeling",
                "    data: { ref: dataset.tokens, config: { path: 'x.tokens', block_size: 4 } }",
                "    system:",
                "      ref: system.language_model",
                "      config:",
                "        model:",
                "          type: TransformerModel",
                "          topology:",
                "            type: StackedTopology",
                "            layers:",
                "              - type: LinearLayer",
                "                d_in: 8",
                "                d_out: 8",
                "                bias: true",
                "    objective: objective.next_token_ce",
                "    trainer: trainer.standard",
                "    runs: []",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    m = Manifest.from_path(path)
    txt = Planner().format(m)
    assert "manifest.version=2" in txt
    assert "target.name=exp type=experiment" in txt
    assert "model.type=TransformerModel" in txt
    assert "- topology=StackedTopology" in txt
    assert "- layer=LinearLayer" in txt


def test_planner_formats_process_target(tmp_path: Path) -> None:
    path = tmp_path / "m.yml"
    path.write_text(
        "\n".join(
            [
                "version: 2",
                "name: test",
                "defaults: {}",
                "targets:",
                "  - type: process",
                "    name: p",
                "    team:",
                "      leader: research_lead",
                "      developer: developer",
                "    process:",
                "      type: discussion",
                "      name: discuss",
                "      leader: leader",
                "      topic: hello",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    m = Manifest.from_path(path)
    txt = Planner().format(m)
    assert "target.name=p type=process" in txt
    assert "process.type=discussion name=discuss" in txt
    assert "team:" in txt
    assert "- developer: developer" in txt


def test_planner_formats_graph_target(tmp_path: Path) -> None:
    path = tmp_path / "m.yml"
    path.write_text(
        "\n".join(
            [
                "version: 2",
                "name: test",
                "defaults: {}",
                "targets:",
                "  - type: experiment",
                "    name: g",
                "    backend: torch",
                "    task: task.graph",
                "    data: { ref: dataset.graph_npy, config: { path: 'x.npy' } }",
                "    system:",
                "      ref: system.graph",
                "      config:",
                "        topology:",
                "          type: GraphTopology",
                "          layers:",
                "            - id: proj",
                "              in: inputs",
                "              out: h",
                "              type: LinearLayer",
                "              d_in: 4",
                "              d_out: 8",
                "              bias: true",
                "            - id: act",
                "              in: h",
                "              out: h2",
                "              op: ReLU",
                "              config: {}",
                "    objective: objective.mse",
                "    trainer: trainer.standard",
                "    runs: []",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    m = Manifest.from_path(path)
    txt = Planner().format(m)
    assert "target.name=g type=experiment" in txt
    assert "graph.type=GraphTopology" in txt
    assert "graph.layers:" in txt
    assert "layer[0].id=proj layer=LinearLayer" in txt
    assert "layer[1].id=act op=ReLU" in txt
