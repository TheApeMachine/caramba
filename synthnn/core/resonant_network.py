"""Resonant network

Manages a collection of ResonantNodes connected by weighted complex couplings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synthnn.core.resonant_node import ResonantNode


@dataclass
class Connection:
    """Connection between nodes

    A connection propagates a complex signal through a weight and optional delay.
    """

    weight: complex = 1.0 + 0.0j
    delay: float = 0.0
    buffer: list[tuple[complex, float]] = field(default_factory=list)

    def propagate(self, *, signal: complex, dt: float) -> complex:
        """Propagate signal with optional delay."""

        if float(self.delay) <= 0.0:
            return complex(self.weight) * complex(signal)

        self.buffer.append((complex(signal), float(self.delay)))
        output = 0.0 + 0.0j
        remaining: list[tuple[complex, float]] = []
        for sig, remaining_delay in self.buffer:
            new_delay = float(remaining_delay) - float(dt)
            if new_delay <= 0.0:
                output += complex(self.weight) * complex(sig)
            else:
                remaining.append((sig, new_delay))
        self.buffer = remaining
        return output


@dataclass
class ResonantNetwork:
    """Resonant network

    Provides a simple stepping simulation for emergent phase dynamics.
    """

    name: str = "resonant_network"
    global_damping: float = 0.01
    coupling_strength: float = 0.1

    nodes: dict[str, ResonantNode] = field(default_factory=dict)
    connections: dict[tuple[str, str], Connection] = field(default_factory=dict)
    time: float = 0.0

    def addNode(self, node: ResonantNode) -> None:
        """Add a node."""

        if node.node_id in self.nodes:
            raise ValueError(f"Node already exists: {node.node_id}")
        self.nodes[node.node_id] = node

    def connect(self, *, source_id: str, target_id: str, weight: complex, delay: float = 0.0) -> None:
        """Create a directed connection source->target."""

        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError("Both nodes must exist to connect.")
        self.connections[(str(source_id), str(target_id))] = Connection(weight=complex(weight), delay=float(delay))

    def inputs(self, *, node_id: str) -> list[str]:
        """List incoming source node ids."""

        return [src for (src, tgt) in self.connections.keys() if tgt == str(node_id)]

    def computeCoupling(self, *, node_id: str, dt: float) -> complex:
        """Compute total coupling influence on a node."""

        total = 0.0 + 0.0j
        for src in self.inputs(node_id=str(node_id)):
            conn = self.connections[(src, str(node_id))]
            total += conn.propagate(signal=self.nodes[src].signal, dt=float(dt))
        return total * complex(float(self.coupling_strength))

    def step(self, *, dt: float, external_inputs: dict[str, complex] | None = None) -> None:
        """Advance the network one step."""

        dt = float(dt)
        if external_inputs:
            for node_id, signal in external_inputs.items():
                if node_id in self.nodes:
                    self.nodes[node_id].signal += complex(signal)

        couplings: dict[str, complex] = {}
        for node_id in self.nodes.keys():
            couplings[node_id] = self.computeCoupling(node_id=node_id, dt=dt)

        for node_id, node in self.nodes.items():
            node.step(dt=dt, coupling=couplings[node_id], damping_override=float(self.global_damping))
        self.time += dt

