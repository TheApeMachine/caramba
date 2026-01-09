"""Development process for implementing features and fixes.

This process handles the workflow when missing features are discovered in the
caramba framework. Development team agents inspect the codebase using available
tools (filesystem, deeplake, codegraph) and implement changes using the
OpenHands SDK for contained development.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from caramba.ai.agent import Agent
from caramba.ai.process import Process
from caramba.ai.process.development.openhands_workspace import OpenHandsWorkspace, DevelopmentResult
from caramba.console import logger


class FeatureAnalysis(BaseModel):
    """Structured output for feature analysis."""

    is_missing_feature: bool = Field(description="Whether this is a missing feature that needs implementation")
    feature_description: str = Field(description="Clear description of the feature to implement")
    affected_files: list[str] = Field(description="List of files likely to be affected")
    implementation_approach: str = Field(description="High-level approach for implementation")
    test_plan: str = Field(description="How to verify the implementation works")


class CodeReviewOutput(BaseModel):
    """Structured output for code review."""

    approved: bool = Field(description="Whether the changes are approved")
    issues: list[str] = Field(description="List of issues found in the code")
    suggestions: list[str] = Field(description="Suggestions for improvement")
    security_concerns: list[str] = Field(description="Any security concerns identified")


@dataclass
class DevelopmentConfig:
    """Configuration for the development process."""

    repo_url: str
    base_branch: str = "main"
    branch_prefix: str = "feature"
    test_command: str = "python -m pytest -q"
    max_iterations: int = 3
    auto_pr: bool = True


class DevelopmentProcess(Process):
    """Process for implementing features and fixes in the caramba codebase.

    This process is triggered when:
    1. A user identifies a missing feature during discussion
    2. The root agent determines development work is needed
    3. An automated process discovers a gap in functionality

    The workflow:
    1. Analyze the feature request with the architect agent
    2. Inspect relevant code using filesystem/codegraph/deeplake tools
    3. Create a contained workspace using OpenHands SDK
    4. Implement changes with the developer agent
    5. Review changes with the reviewer agent
    6. Run tests and create a pull request
    """

    def __init__(
        self,
        *,
        agents: dict[str, Agent],
        config: DevelopmentConfig,
    ) -> None:
        super().__init__(agents, name="development")
        self.config = config
        self.workspace: OpenHandsWorkspace | None = None

    async def run(self, feature_request: str) -> DevelopmentResult:
        """Execute the development workflow for a feature request.

        Args:
            feature_request: Natural language description of the feature to implement

        Returns:
            DevelopmentResult with the outcome of the development process
        """
        try:
            # Step 1: Analyze the feature request
            logger.header("Development", "Analyzing feature request...")
            analysis = await self.analyze_feature(feature_request)

            if not analysis.is_missing_feature:
                logger.info("Analysis determined this is not a missing feature")
                return DevelopmentResult(
                    success=True,
                    error="Not identified as a missing feature requiring implementation",
                )

            # Step 2: Inspect relevant codebase areas
            logger.header("Development", "Inspecting codebase...")
            context = await self.inspect_codebase(analysis)

            # Step 3: Set up OpenHands workspace
            logger.header("Development", "Setting up development workspace...")
            self.workspace = OpenHandsWorkspace(
                repo_url=self.config.repo_url,
                base_branch=self.config.base_branch,
            )
            self.workspace.setup()

            # Step 4: Create feature branch
            branch_name = self.generate_branch_name(analysis.feature_description)
            self.workspace.create_branch(branch_name)

            # Step 5: Implement changes using OpenHands
            logger.header("Development", "Implementing changes...")
            implementation_task = self.build_implementation_task(analysis, context)
            result = self.workspace.implement_changes(implementation_task)

            if not result.success:
                return result

            # Step 6: Run tests
            logger.header("Development", "Running tests...")
            test_success, test_output = self.workspace.run_tests(self.config.test_command)
            result.test_output = test_output

            if not test_success:
                # Try to fix test failures
                for iteration in range(self.config.max_iterations):
                    logger.info(f"Attempting to fix test failures (iteration {iteration + 1})...")
                    fix_result = self.workspace.implement_changes(
                        f"The tests failed with this output:\n{test_output}\n\nPlease fix the issues."
                    )
                    test_success, test_output = self.workspace.run_tests(self.config.test_command)
                    if test_success:
                        break

            # Step 7: Code review
            logger.header("Development", "Reviewing changes...")
            diff = self.workspace.get_diff()
            review = await self.review_changes(diff, analysis)

            if not review.approved:
                logger.warning(f"Code review failed: {review.issues}")
                result.error = f"Code review failed: {'; '.join(review.issues)}"
                return result

            # Step 8: Commit and create PR
            if self.config.auto_pr and test_success:
                logger.header("Development", "Creating pull request...")
                commit_message = f"feat: {analysis.feature_description[:50]}"
                self.workspace.commit_and_push(branch_name, commit_message)

                pr_url = self.workspace.create_pull_request(
                    branch_name=branch_name,
                    title=f"feat: {analysis.feature_description}",
                    body=self.build_pr_body(analysis, review),
                )
                result.pr_url = pr_url
                result.branch_name = branch_name

            return result

        except Exception as e:
            logger.error(f"Development process failed: {e}")
            return DevelopmentResult(success=False, error=str(e))

        finally:
            if self.workspace:
                self.workspace.cleanup()

    async def analyze_feature(self, feature_request: str) -> FeatureAnalysis:
        """Analyze a feature request using the feature analyst agent."""
        analyst = self.agents.get("feature_analyst")
        if not analyst:
            raise KeyError("Missing required agent: feature_analyst")

        prompt = (
            "Analyze this feature request and determine if it requires new development work.\n\n"
            f"Feature Request:\n{feature_request}\n\n"
            "Use your tools to inspect the codebase and determine:\n"
            "1. Is this actually a missing feature or is it already implemented?\n"
            "2. What files would need to be modified?\n"
            "3. What's the best approach to implement it?\n"
            "4. How should it be tested?"
        )
        response = await analyst.run_async(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        return FeatureAnalysis.model_validate_json(response)

    async def inspect_codebase(self, analysis: FeatureAnalysis) -> str:
        """Inspect relevant codebase areas using available tools."""
        developer = self.agents.get("developer")
        if not developer:
            raise KeyError("Missing required agent: developer")

        prompt = (
            "Inspect the following files and provide context for implementation:\n\n"
            f"Files to inspect: {', '.join(analysis.affected_files)}\n\n"
            f"Feature to implement: {analysis.feature_description}\n\n"
            "Use your tools (filesystem, codegraph, deeplake) to gather relevant context."
        )
        return await developer.run_async(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )

    async def review_changes(self, diff: str, analysis: FeatureAnalysis) -> CodeReviewOutput:
        """Review the implemented changes."""
        reviewer = self.agents.get("code_reviewer")
        if not reviewer:
            raise KeyError("Missing required agent: code_reviewer")

        prompt = (
            "Review these code changes:\n\n"
            f"Feature: {analysis.feature_description}\n\n"
            f"Implementation approach: {analysis.implementation_approach}\n\n"
            f"Diff:\n```\n{diff}\n```\n\n"
            "Check for:\n"
            "- Correctness and completeness\n"
            "- Code style consistency\n"
            "- Security issues\n"
            "- Test coverage"
        )
        response = await reviewer.run_async(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )
        return CodeReviewOutput.model_validate_json(response)

    def generate_branch_name(self, feature_description: str) -> str:
        """Generate a branch name from the feature description."""
        # Sanitize and shorten the description
        slug = feature_description.lower()[:40]
        slug = "".join(c if c.isalnum() or c == " " else "" for c in slug)
        slug = slug.strip().replace(" ", "-")
        return f"{self.config.branch_prefix}/{slug}"

    def build_implementation_task(self, analysis: FeatureAnalysis, context: str) -> str:
        """Build the implementation task for OpenHands."""
        return f"""
Implement the following feature:

## Feature Description
{analysis.feature_description}

## Implementation Approach
{analysis.implementation_approach}

## Files to Modify
{', '.join(analysis.affected_files)}

## Codebase Context
{context}

## Test Plan
{analysis.test_plan}

Please implement this feature following the existing code style and patterns.
Ensure all changes are well-tested.
"""

    def build_pr_body(self, analysis: FeatureAnalysis, review: CodeReviewOutput) -> str:
        """Build the pull request body."""
        return f"""## Summary

{analysis.feature_description}

## Implementation

{analysis.implementation_approach}

## Test Plan

{analysis.test_plan}

## Review Notes

{'; '.join(review.suggestions) if review.suggestions else 'No additional suggestions.'}
"""
