"""Manifest workflow process.

Enables AI agents to build, run, and collect results from manifests,
supporting the research workflow of iterative experimentation.
"""

from caramba.ai.process.manifest.process import ManifestProcess
from caramba.ai.process.manifest.builder import ManifestBuilder
from caramba.ai.process.manifest.collector import ResultsCollector

__all__ = ["ManifestProcess", "ManifestBuilder", "ResultsCollector"]
