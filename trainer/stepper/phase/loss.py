"""Loss computation for the stepwise training loop."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class StudentLoss:
    """Student loss computation

    This helper encapsulates the "loss contract" expected by the stepwise/global
    training loop. It supports:
    - standard next-token cross-entropy
    - optional diffusion auxiliary losses when a student exposes a diffusion head
    """

    def __init__(self, *, student: nn.Module) -> None:
        """Create a loss computer bound to a student module."""
        self.student = student
        self.has_diffusion = self.has_diffusion_head(student=student)
        self.diffusion_weight = self.diffusion_loss_weight(student=student) if self.has_diffusion else 0.0

    def has_diffusion_head(self, *, student: nn.Module) -> bool:
        """Return whether the module exposes a usable diffusion head."""
        return hasattr(student, "diffusion_head") and getattr(student, "diffusion_head", None) is not None

    def diffusion_loss_weight(self, *, student: nn.Module) -> float:
        """Resolve the diffusion auxiliary loss weight from the student config."""
        if not hasattr(student, "config"):
            raise AttributeError("Student has diffusion_head but no config attribute for diffusion loss weight.")
        cfg = getattr(student, "config")
        if not hasattr(cfg, "diffusion_head"):
            raise AttributeError("Student config has no diffusion_head section for diffusion loss weight.")
        dh = getattr(cfg, "diffusion_head")
        if not hasattr(dh, "loss_weight"):
            raise AttributeError("Student diffusion_head config has no loss_weight.")
        return float(getattr(dh, "loss_weight"))

    def tensor_stats(self, *, name: str, t: Tensor) -> str:
        """Return compact numeric stats for debugging non-finite failures."""
        t_f = t.detach().float()
        finite = torch.isfinite(t_f)
        n = int(t_f.numel())
        nf = int((~finite).sum().item())
        if n == 0:
            return f"{name}: empty"
        if nf == n:
            return f"{name}: all_nonfinite n={n} dtype={t.dtype} device={t.device}"
        tf = t_f[finite]
        if int(tf.numel()) == 0:
            return f"{name}: finite_subset_empty n={n} dtype={t.dtype} device={t.device}"
        return (
            f"{name}: dtype={t.dtype} device={t.device} shape={tuple(t.shape)} "
            f"nonfinite={nf}/{n} min={float(tf.min().item()):.5g} max={float(tf.max().item()):.5g} "
            f"mean={float(tf.mean().item()):.5g} std={float(tf.std(unbiased=False).item()):.5g}"
        )

    def compute(self, *, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        """Compute the total loss and its components for a batch."""
        if self.has_diffusion:
            result = self.student.forward(x, return_features=True)  # type: ignore[call-arg]
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise TypeError("Student forward(return_features=True) must return (features, logits).")
            features = result[0]
            logits = result[1]
            if not isinstance(features, Tensor) or not isinstance(logits, Tensor):
                raise TypeError("Student forward(return_features=True) must return tensors (features, logits).")
            if not torch.isfinite(logits.detach()).all():
                raise RuntimeError(
                    "Non-finite logits detected.\n"
                    f"- {self.tensor_stats(name='logits', t=logits)}\n"
                    "This is a hard failure under the kernel policy (no silent fallback paths)."
                )

            logits_f = logits.float()
            ce_loss = F.cross_entropy(logits_f.view(-1, logits_f.shape[-1]), y.reshape(-1))

            diff_loss = self.student.diffusion_loss(features, y)  # type: ignore[attr-defined]
            if not isinstance(diff_loss, Tensor):
                raise TypeError("Student diffusion_loss(features, y) must return a Tensor.")
            loss = ce_loss + float(self.diffusion_weight) * diff_loss

            if not torch.isfinite(loss.detach()).all():
                raise RuntimeError(
                    "Non-finite loss detected.\n"
                    f"- {self.tensor_stats(name='ce_loss', t=ce_loss)}\n"
                    f"- {self.tensor_stats(name='diff_loss', t=diff_loss)}\n"
                    f"- {self.tensor_stats(name='loss', t=loss)}"
                )
            return loss, ce_loss, diff_loss

        logits = self.student.forward(x)
        if not isinstance(logits, Tensor):
            raise TypeError("Student forward(x) must return a Tensor of logits.")
        if not torch.isfinite(logits.detach()).all():
            raise RuntimeError(
                "Non-finite logits detected.\n"
                f"- {self.tensor_stats(name='logits', t=logits)}\n"
                "This is a hard failure under the kernel policy (no silent fallback paths)."
            )
        logits_f = logits.float()
        ce_loss = F.cross_entropy(logits_f.view(-1, logits_f.shape[-1]), y.reshape(-1))
        if not torch.isfinite(ce_loss.detach()).all():
            raise RuntimeError(
                "Non-finite CE loss detected.\n"
                f"- {self.tensor_stats(name='logits_f', t=logits_f)}\n"
                f"- {self.tensor_stats(name='ce_loss', t=ce_loss)}"
            )
        return ce_loss, ce_loss, None

