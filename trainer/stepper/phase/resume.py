"""Resume-state handling for stepwise training loops."""

from __future__ import annotations

from caramba.console import logger


class ResumeManager:
    """Resume manager

    Training loops can optionally resume from a checkpoint state dict. The
    checkpoint may include optimizer/scheduler state; when absent, training
    resumes with fresh optimizer state but continues at the recorded step index.
    """

    def apply(
        self,
        *,
        resume_state: dict[str, object] | None,
        run_id: str,
        phase: str,
        optimizer: object,
        scheduler: object | None,
    ) -> int:
        """Apply a resume state dict and return the starting step."""
        if resume_state is None:
            return 0

        if str(resume_state.get("phase", "")) != str(phase):
            return 0
        if str(resume_state.get("run_id", "")) != str(run_id):
            return 0

        step_obj = resume_state.get("step", 0)
        try:
            start_step = int(step_obj)  # type: ignore[arg-type]
        except (TypeError, ValueError) as e:
            raise TypeError(f"Resume state step must be int-like, got {step_obj!r}") from e

        if "optimizer_state_dict" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer_state_dict"])  # type: ignore[attr-defined]
            except Exception as e:
                raise RuntimeError("Failed to load optimizer state dict from resume state.") from e
        else:
            logger.warning("Resume state has no optimizer_state_dict; continuing with fresh optimizer state.")

        if scheduler is not None:
            if "scheduler_state_dict" in resume_state:
                try:
                    scheduler.load_state_dict(resume_state["scheduler_state_dict"])  # type: ignore[attr-defined]
                except Exception as e:
                    raise RuntimeError("Failed to load scheduler state dict from resume state.") from e
            else:
                logger.warning("Resume state has no scheduler_state_dict; continuing with fresh scheduler state.")

        return int(start_step)

