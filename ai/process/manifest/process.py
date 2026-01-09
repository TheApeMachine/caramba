"""Manifest process for AI-driven experiment management.

This process enables AI agents to build, run, and analyze experiments
as part of the research workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from caramba.ai.agent import Agent
from caramba.ai.process import Process
from caramba.ai.process.manifest.builder import ManifestBuilder, ManifestSpec
from caramba.ai.process.manifest.collector import ResultsCollector, ResultsSummary
from caramba.console import logger
from caramba.experiment.runner import run_from_manifest_path


class ExperimentProposal(BaseModel):
    """Structured output for experiment proposals from AI agents."""

    name: str = Field(description="Name for the experiment")
    hypothesis: str = Field(description="What hypothesis this experiment tests")
    model_type: str = Field(description="Type of model to use")
    dataset: str = Field(description="Dataset to use")
    key_parameters: dict[str, Any] = Field(description="Key training/model parameters")
    success_criteria: str = Field(description="How to determine if the experiment succeeded")
    expected_outcome: str = Field(description="What results are expected")


class ExperimentAnalysis(BaseModel):
    """Structured output for experiment analysis from AI agents."""

    summary: str = Field(description="Summary of the experiment results")
    key_findings: list[str] = Field(description="Key findings from the results")
    hypothesis_supported: bool = Field(description="Whether the hypothesis was supported")
    recommendations: list[str] = Field(description="Recommendations for next steps")
    follow_up_experiments: list[str] = Field(description="Suggested follow-up experiments")


class ManifestProcess(Process):
    """Process for managing the manifest-driven research workflow.

    This process allows AI agents to:
    1. Propose and build new experiments based on research goals
    2. Run experiments and monitor their progress
    3. Collect and analyze results
    4. Iterate based on findings
    """

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        presets_dir: str = "config/presets",
        artifacts_dir: str = "artifacts",
    ) -> None:
        super().__init__(agents, name="manifest")
        self.builder = ManifestBuilder(presets_dir)
        self.collector = ResultsCollector(artifacts_dir)

    async def propose_experiment(self, research_goal: str) -> ExperimentProposal:
        """Have the research lead propose an experiment based on a research goal.

        Args:
            research_goal: Description of what the researcher wants to investigate

        Returns:
            ExperimentProposal with the proposed experiment details
        """
        research_lead = self.agents.get("research_lead")
        if not research_lead:
            raise KeyError("Missing required agent: research_lead")

        # Get context about existing experiments
        existing = self.collector.collect_experiment_results()
        existing_summary = self.collector.format_summary_markdown(
            self.collector.summarize_results(existing)
        )

        prompt = (
            "Based on the following research goal, propose an experiment.\n\n"
            f"Research Goal:\n{research_goal}\n\n"
            f"Existing Experiments Summary:\n{existing_summary}\n\n"
            "Consider:\n"
            "1. What hypothesis does this test?\n"
            "2. What model architecture would be best?\n"
            "3. What dataset should be used?\n"
            "4. What are the key parameters to vary?\n"
            "5. How will success be measured?"
        )

        response = await research_lead.run_async(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        return ExperimentProposal.model_validate_json(response)

    async def build_and_run_experiment(
        self,
        proposal: ExperimentProposal,
        *,
        run_immediately: bool = True,
    ) -> dict[str, Any]:
        """Build a manifest from a proposal and optionally run it.

        Args:
            proposal: The experiment proposal
            run_immediately: Whether to run the experiment after building

        Returns:
            Dict with manifest path and optionally results
        """
        # Build the manifest
        spec = ManifestSpec(
            name=proposal.name,
            notes=f"Hypothesis: {proposal.hypothesis}\n\nExpected: {proposal.expected_outcome}",
            experiments=[
                {
                    "name": proposal.name,
                    "description": proposal.hypothesis,
                    "model_type": proposal.model_type,
                    "dataset": proposal.dataset,
                    "training_config": proposal.key_parameters,
                    "metrics": ["loss", "perplexity"],
                }
            ],
        )

        manifest_dict = self.builder.build_from_spec(spec)

        # Validate
        is_valid, error = self.builder.validate_manifest(manifest_dict)
        if not is_valid:
            raise ValueError(f"Invalid manifest: {error}")

        # Save
        manifest_path = self.builder.save_manifest(manifest_dict, proposal.name)

        result = {"manifest_path": str(manifest_path), "manifest": manifest_dict}

        # Run if requested
        if run_immediately:
            logger.header("Manifest", f"Running experiment: {proposal.name}")
            try:
                run_result = run_from_manifest_path(manifest_path)
                result["run_result"] = run_result
                result["status"] = "completed"
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                result["status"] = "failed"
                result["error"] = str(e)

        return result

    async def analyze_results(
        self,
        experiment_name: str | None = None,
    ) -> ExperimentAnalysis:
        """Have the ML expert analyze experiment results.

        Args:
            experiment_name: Optional specific experiment to analyze

        Returns:
            ExperimentAnalysis with findings and recommendations
        """
        ml_expert = self.agents.get("ml_expert")
        if not ml_expert:
            raise KeyError("Missing required agent: ml_expert")

        # Collect results
        results = self.collector.collect_experiment_results(experiment_name)
        summary = self.collector.summarize_results(results)
        summary_md = self.collector.format_summary_markdown(summary)

        prompt = (
            "Analyze these experiment results and provide insights.\n\n"
            f"Results:\n{summary_md}\n\n"
            "Provide:\n"
            "1. A summary of what the results show\n"
            "2. Key findings\n"
            "3. Whether any hypotheses were supported\n"
            "4. Recommendations for next steps\n"
            "5. Suggestions for follow-up experiments"
        )

        response = await ml_expert.run_async(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        return ExperimentAnalysis.model_validate_json(response)

    async def run_research_loop(
        self,
        research_goal: str,
        *,
        max_iterations: int = 3,
    ) -> list[dict[str, Any]]:
        """Run an iterative research loop.

        Proposes experiments, runs them, analyzes results, and iterates
        based on findings.

        Args:
            research_goal: The overall research objective
            max_iterations: Maximum number of experiment iterations

        Returns:
            List of results from each iteration
        """
        iterations: list[dict[str, Any]] = []
        current_goal = research_goal

        for i in range(max_iterations):
            logger.header("Research Loop", f"Iteration {i + 1}/{max_iterations}")

            # Propose experiment
            logger.info("Proposing experiment...")
            proposal = await self.propose_experiment(current_goal)
            logger.info(f"Proposed: {proposal.name} - {proposal.hypothesis}")

            # Build and run
            logger.info("Building and running experiment...")
            run_result = await self.build_and_run_experiment(proposal)

            # Analyze results
            logger.info("Analyzing results...")
            analysis = await self.analyze_results(proposal.name)

            iteration_result = {
                "iteration": i + 1,
                "proposal": proposal.model_dump(),
                "run_result": run_result,
                "analysis": analysis.model_dump(),
            }
            iterations.append(iteration_result)

            # Check if we should continue
            if analysis.hypothesis_supported:
                logger.success("Hypothesis supported! Research goal may be achieved.")
                if not analysis.follow_up_experiments:
                    break

            # Update goal for next iteration based on recommendations
            if analysis.follow_up_experiments:
                current_goal = (
                    f"Original goal: {research_goal}\n\n"
                    f"Previous findings: {analysis.summary}\n\n"
                    f"Suggested follow-up: {analysis.follow_up_experiments[0]}"
                )

        return iterations

    def get_results_summary(self) -> str:
        """Get a markdown summary of all experiment results."""
        results = self.collector.collect_experiment_results()
        summary = self.collector.summarize_results(results)
        return self.collector.format_summary_markdown(summary)

    def list_available_manifests(self) -> list[str]:
        """List all available manifest presets."""
        return self.builder.list_existing_presets()
