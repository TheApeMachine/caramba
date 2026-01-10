"""PyTorch Lightning execution engine.

Bridges Caramba manifest targets to Lightning Trainer.
"""
from __future__ import annotations

import pytorch_lightning as L
import torch
from typing import Any, cast
from pathlib import Path
from datetime import datetime

from caramba.config.manifest import Manifest
from caramba.config.target import ExperimentTargetConfig
from caramba.runtime.registry import ComponentRegistry
from caramba.console import logger

class CarambaLightningModule(L.LightningModule):
    """Bridges Caramba System and Objective to Lightning."""

    def __init__(
        self,
        system: Any,
        objective: Any,
        train_cfg: Any,
    ):
        super().__init__()
        self.system = system
        self.objective = objective
        self.train_cfg = train_cfg
        # Some Caramba systems wrap the module in self.module
        self.model = getattr(system, "module", system)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.system.forward(batch)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = self.objective.loss(outputs=outputs, batch=batch)
        
        self.log("train_loss", loss, prog_bar=True)
        if hasattr(self.objective, "metrics"):
            metrics = self.objective.metrics(outputs=outputs, batch=batch, loss=loss)
            if metrics:
                self.log_dict(metrics, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        # This is a bit tricky because Caramba has its own optimizer building logic.
        # For now, we'll implement a simple version or try to reuse Caramba's builders.
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg.lr,
            weight_decay=getattr(self.train_cfg, "weight_decay", 0.0)
        )
        return optimizer

class LightningEngine:
    """Execution engine using PyTorch Lightning."""

    def __init__(self, *, registry: ComponentRegistry | None = None) -> None:
        self.registry = registry or ComponentRegistry()

    def run_experiment(
        self,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        *,
        dry_run: bool = False,
    ) -> Any:
        if dry_run:
            logger.info(f"Dry run: LightningEngine for target {target.name}")
            return {"target": target.name}

        # Build components via registry
        dataset_comp = self.registry.build(target.data, backend="torch")
        system = self.registry.build(target.system, backend="torch")
        objective = self.registry.build(target.objective, backend="torch")

        # For Lightning, we typically run one "representative" run or handle multiple?
        # Standard Caramba runner iterates over target.runs.
        results = {}
        for run in target.runs:
            train_cfg = run.train
            if not train_cfg:
                continue

            # Wrap in LightningModule
            lightning_model = CarambaLightningModule(system, objective, train_cfg)

            # Setup DataLoader
            # Lightning expects a standard DataLoader or a LightningDataModule.
            # We'll use the dataset component which should be a torch.utils.data.Dataset.
            train_loader = torch.utils.data.DataLoader(
                dataset_comp,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=getattr(train_cfg, "num_workers", 0)
            )

            # Setup Trainer
            trainer = L.Trainer(
                max_steps=run.steps,
                devices=1 if train_cfg.device != "cpu" else 0,
                accelerator="auto" if train_cfg.device != "cpu" else "cpu",
                precision="16-mixed" if train_cfg.use_amp else "32-true",
                default_root_dir=str(Path("runs") / target.name / str(run.id))
            )

            logger.info(f"Starting Lightning training for run {run.id}...")
            trainer.fit(lightning_model, train_loader)
            
            results[run.id] = {
                "system": system,
                "trainer": trainer
            }

        return results
