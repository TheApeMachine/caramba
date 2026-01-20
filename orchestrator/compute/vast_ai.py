"""Vast.ai compute orchestrator.

Handles searching for, provisioning, and decommissioning GPU instances on Vast.ai.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from config.compute import VastAIComputeConfig
from console import logger

# Note: In a real environment, we would import the vastai SDK here.
# For this implementation, we assume a wrapper or direct subprocess calls
# if the SDK is just a CLI wrapper, but let's define the interface.

class VastAIClient:
    """Orchestrator for Vast.ai GPU instances."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("VAST_AI_API_KEY")
        if not self.api_key:
            logger.warning("VAST_AI_API_KEY not found. Operations may fail.")

    def find_best_offer(self, config: VastAIComputeConfig) -> dict[str, Any] | None:
        """Search for the best matching GPU offer based on config."""
        logger.info(f"Searching for Vast.ai offers: {config.gpu_name}, >={config.min_vram}GB VRAM")

        # Mocking the search logic
        # query = f"gpu_name = {config.gpu_name} vram >= {config.min_vram} cuda_vers >= {config.min_cuda_version}"
        # offers = vastai.search_offers(query, api_key=self.api_key)

        # For now, return a mock offer ID if search was successful
        return {"id": "mock-offer-123", "price": 0.5, "gpu": config.gpu_name}

    def provision_instance(self, offer_id: str, config: VastAIComputeConfig) -> str:
        """Provision a new instance from an offer."""
        logger.info(f"Provisioning Vast.ai instance from offer {offer_id}...")

        # vastai.create_instance(offer_id, image=config.image, api_key=self.api_key)

        instance_id = "mock-instance-456"
        logger.success(f"Instance {instance_id} is being provisioned.")
        return instance_id

    async def wait_for_ssh(self, instance_id: str, timeout: int = 300) -> str | None:
        """Wait for the instance to be ready and return the SSH connection string."""
        logger.info(f"Waiting for instance {instance_id} to be SSH-ready...")

        # poll vastai.get_instance(instance_id) until status is 'running' and has an IP

        # Mocking readiness
        await asyncio.sleep(1)
        ssh_str = "root@123.45.67.89 -p 12345"
        logger.success(f"Instance {instance_id} is ready at {ssh_str}")
        return ssh_str

    def decommission_instance(self, instance_id: str) -> None:
        """Destroy the instance."""
        logger.info(f"Decommissioning Vast.ai instance {instance_id}...")

        # vastai.destroy_instance(instance_id, api_key=self.api_key)

        logger.success(f"Instance {instance_id} destroyed.")

    async def run_lifecycle_async(self, config: VastAIComputeConfig) -> str | None:
        """Full lifecycle: find, provision, wait for SSH, and return connection info."""
        offer = self.find_best_offer(config)
        if not offer:
            logger.error("No suitable Vast.ai offers found.")
            return None

        instance_id: str | None = None
        try:
            instance_id = self.provision_instance(offer["id"], config)
            ssh = await self.wait_for_ssh(instance_id)
            if not ssh:
                logger.error(f"Instance {instance_id} did not become SSH-ready.")
                self.decommission_instance(instance_id)
                return None
            return ssh
        except Exception as e:
            logger.error(f"Vast.ai lifecycle failed: {e}")
            if instance_id is not None:
                try:
                    self.decommission_instance(instance_id)
                except Exception as e2:
                    logger.warning(f"Failed to decommission instance {instance_id} after lifecycle failure: {e2}")
            return None

    def run_lifecycle(self, config: VastAIComputeConfig) -> str | None:
        """Synchronous wrapper around `run_lifecycle_async`.

        NOTE: If you're already inside an asyncio event loop, call and await
        `run_lifecycle_async` directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_lifecycle_async(config))
        raise RuntimeError("run_lifecycle() cannot be called from a running event loop; use await run_lifecycle_async(...)")
