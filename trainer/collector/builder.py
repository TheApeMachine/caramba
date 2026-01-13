"""Collector builder

Builds a collector based on the manifest.
"""
from __future__ import annotations

from caramba.manifest import Manifest
from caramba.manifest.target import Target
from caramba.trainer.collector.base import Collector

class CollectorBuilder:
    def __init__(self, manifest: Manifest, target: Target) -> None:
        self.manifest = manifest
        self.target = target
        self.collectors = []

    def build(self) -> Collector:
        for collector in self.target.collectors:
            match collector.type:
                case CollectorType.DEFAULT:
                    self.collectors.append(DefaultCollector(manifest=self.manifest, target=self.target))
                case _:
                    raise ValueError(f"Unsupported collector type: {collector.type}")

        return self.collectors