from __future__ import annotations

"""CCL trainer (manifest-driven, non-gradient).

Implements Constructive Compression Learning as a Caramba trainer component:
- learn a patch VQ codec (minibatch k-means)
- tokenize images into discrete grids
- train class-conditional context-count models
- evaluate by codelength classification

This trainer is intentionally architecture-agnostic: it does not assume SGD,
transformers, or even an nn.Module with parameters.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from collections.abc import Sized

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from ccl.context_counts import ContextTemplate, predict_class, train_class_counts_models
from ccl.patch_vq import PatchKMeansVQ
from ccl.system import CCLSystem
from config.defaults import Defaults
from config.manifest import Manifest
from config.target import ExperimentTargetConfig
from console import logger


class _Engine(Protocol):
    registry: Any


def _as_dict(x: object) -> dict[str, object]:
    if isinstance(x, dict):
        return x
    try:
        return dict(x)  # type: ignore[arg-type]
    except Exception as e:
        raise TypeError(f"Expected dict-like batch, got {type(x).__name__}") from e


def _collect_images_labels(
    ds: Dataset[Any],
    *,
    input_key: str,
    target_key: str,
    max_items: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(cast(Sized, ds)))
    if max_items is not None:
        n = min(n, int(max_items))
    if n <= 0:
        raise ValueError("Dataset is empty")

    # Discover shape from first sample.
    ex0 = _as_dict(ds[0])
    x0 = ex0.get(str(input_key), None)
    y0 = ex0.get(str(target_key), None)
    if x0 is None or y0 is None:
        raise KeyError(f"Dataset items must contain keys {input_key!r} and {target_key!r}")

    def to_np_img(x: object) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.dtype != torch.float32:
                t = t.float()
            return t.numpy()
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        raise TypeError(f"Unsupported image type {type(x).__name__}")

    def to_int(y: Any) -> int:
        if isinstance(y, torch.Tensor):
            return int(y.detach().cpu().item())
        return int(y)

    # Collect images into (N,C,H,W) float32
    x0n = to_np_img(x0)
    if x0n.ndim == 2:
        x0n = x0n[None, :, :]  # 1HW
    if x0n.ndim != 3:
        raise ValueError(f"Expected image as (H,W) or (C,H,W), got {x0n.shape}")
    c, h, w = int(x0n.shape[0]), int(x0n.shape[1]), int(x0n.shape[2])

    images = np.empty((n, c, h, w), dtype=np.float32)
    labels = np.empty((n,), dtype=np.int64)
    images[0] = x0n.astype(np.float32, copy=False)
    labels[0] = np.int64(to_int(y0))

    for i in range(1, n):
        ex = _as_dict(ds[i])
        xi = to_np_img(ex[str(input_key)])
        yi = to_int(ex[str(target_key)])
        if xi.ndim == 2:
            xi = xi[None, :, :]
        if xi.shape != (c, h, w):
            raise ValueError(f"Inconsistent image shapes: expected {(c,h,w)}, got {xi.shape}")
        images[i] = xi.astype(np.float32, copy=False)
        labels[i] = np.int64(yi)

    return images, labels


def _split_train_val(ds: Dataset[Any], *, val_frac: float) -> tuple[Subset[Any], Subset[Any]]:
    n = int(len(cast(Sized, ds)))
    n_val = int(round(float(n) * float(val_frac)))
    n_val = max(1, min(n - 1, n_val)) if n > 1 else 0
    n_train = int(n - n_val)
    train = Subset(ds, range(n_train))
    val = Subset(ds, range(n_train, n))
    return train, val


def _default_templates() -> tuple[list[ContextTemplate], float]:
    # Mirrors research/ccl/ccl_mnist.py defaults:
    # - full (left, up, upleft): 0.55
    # - mid  (left, up):        0.35
    # - unigram:                0.10
    full = ContextTemplate(
        name="full",
        offsets=((0, -1), (-1, 0), (-1, -1)),
        weight=0.55,
    )
    mid = ContextTemplate(
        name="mid",
        offsets=((0, -1), (-1, 0)),
        weight=0.35,
    )
    return [full, mid], 0.10


@dataclass
class CCLTrainer:
    """Trainer entrypoint for CCL."""

    # Data keys expected in dataset items.
    input_key: str = "inputs"
    target_key: str = "targets"

    # Codec hyperparams.
    k: int = 256
    patch: int = 4
    stride: int = 2
    alpha: float = 0.5

    # Training budget knobs.
    max_train: int | None = None
    max_eval: int | None = None
    image_pool: int = 2048
    sample_patches: int = 200_000
    kmeans_max_iter: int = 200
    kmeans_batch_size: int = 4096
    tokenize_batch_size: int = 256

    # Context model knobs (optional; default matches MNIST script).
    templates: list[dict[str, object]] | None = None
    unigram_weight: float | None = None
    num_classes: int | None = None

    # Optional evaluation dataset override (ComponentSpec-like dict).
    eval_data: dict[str, object] | None = None

    # Optional generation.
    generate: bool = True
    n_gen_per_class: int = 8
    seed: int = 0

    # Output directory override (defaults to runs/<target.name>/ccl/<run.id>)
    out_dir: str | None = None

    def run(
        self,
        *,
        manifest: Manifest,
        target: ExperimentTargetConfig,
        engine: _Engine,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        if dry_run:
            logger.info("Dry run requested, skipping CCL training")
            return None

        if not target.runs:
            raise ValueError("trainer.ccl requires at least one run in target.runs")

        # Build training dataset component from target.data.
        dataset_comp = engine.registry.build(target.data, backend=str(target.backend))
        if not hasattr(dataset_comp, "build"):
            raise TypeError("Dataset component does not expose build()")
        ds = dataset_comp.build()  # type: ignore[attr-defined]
        if not isinstance(ds, Dataset):
            # We only need __len__/__getitem__, but Dataset is a useful contract.
            raise TypeError(f"dataset_comp.build() must return torch Dataset, got {type(ds).__name__}")

        defaults: Defaults = manifest.defaults
        train_ds, val_ds = _split_train_val(ds, val_frac=float(defaults.data.val_frac))

        # Optional explicit eval dataset.
        eval_ds: Dataset[Any]
        if isinstance(self.eval_data, dict) and self.eval_data.get("ref"):
            # Build nested ComponentSpec-like dict (ref, impl, config).
            from config.component import ComponentSpec

            eval_spec = ComponentSpec.model_validate(self.eval_data)
            eval_comp = engine.registry.build(eval_spec, backend=str(target.backend))
            if not hasattr(eval_comp, "build"):
                raise TypeError("eval_data component does not expose build()")
            eval_ds = eval_comp.build()  # type: ignore[attr-defined]
            if not isinstance(eval_ds, Dataset):
                raise TypeError("eval_data.build() must return a torch Dataset")
        else:
            eval_ds = val_ds

        # Use the first run as the "fit" run. (CCL is not iterative in steps.)
        run0 = target.runs[0]
        run_id = str(run0.id)

        base_out = Path(self.out_dir) if self.out_dir else (Path("runs") / str(target.name) / "ccl" / run_id)
        base_out.mkdir(parents=True, exist_ok=True)

        # Export manifest-ish config for reproducibility.
        (base_out / "ccl_config.json").write_text(
            json.dumps(
                {
                    "k": int(self.k),
                    "patch": int(self.patch),
                    "stride": int(self.stride),
                    "alpha": float(self.alpha),
                    "input_key": str(self.input_key),
                    "target_key": str(self.target_key),
                    "max_train": self.max_train,
                    "max_eval": self.max_eval,
                    "image_pool": int(self.image_pool),
                    "sample_patches": int(self.sample_patches),
                    "kmeans_max_iter": int(self.kmeans_max_iter),
                    "kmeans_batch_size": int(self.kmeans_batch_size),
                    "tokenize_batch_size": int(self.tokenize_batch_size),
                    "templates": self.templates,
                    "unigram_weight": self.unigram_weight,
                    "num_classes": self.num_classes,
                    "seed": int(self.seed),
                    "generate": bool(self.generate),
                    "n_gen_per_class": int(self.n_gen_per_class),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # -----------------------------
        # 1) Build an image pool to fit the codec.
        # -----------------------------
        rng = np.random.default_rng(int(self.seed))
        pool_n = min(int(self.image_pool), int(len(cast(Sized, train_ds))))
        if self.max_train is not None:
            pool_n = min(pool_n, int(self.max_train))
        if pool_n <= 0:
            raise ValueError("Training set is empty after max_train constraints")

        pool_idx = rng.choice(int(len(cast(Sized, train_ds))), size=int(pool_n), replace=False)
        pool_items = Subset(train_ds, list(map(int, pool_idx)))
        pool_images, _pool_labels = _collect_images_labels(
            cast(Dataset[Any], pool_items),
            input_key=str(self.input_key),
            target_key=str(self.target_key),
            max_items=int(pool_n),
        )

        codec = PatchKMeansVQ(
            k=int(self.k),
            patch=int(self.patch),
            stride=int(self.stride),
            seed=int(self.seed),
            sample_patches=int(self.sample_patches),
            kmeans_batch_size=int(self.kmeans_batch_size),
            kmeans_max_iter=int(self.kmeans_max_iter),
        )

        logger.header("CCL", "Fitting patch codec (minibatch k-means)")
        centers = codec.fit(pool_images)
        np.save(base_out / "centers.npy", centers)

        # -----------------------------
        # 2) Tokenize train/eval and fit context-count models.
        # -----------------------------
        logger.header("CCL", "Tokenizing + fitting context models")

        train_images, train_labels = _collect_images_labels(
            cast(Dataset[Any], train_ds),
            input_key=str(self.input_key),
            target_key=str(self.target_key),
            max_items=self.max_train,
        )
        train_tokens = codec.tokenize(
            train_images, centers=centers, batch_size=int(self.tokenize_batch_size)
        )

        # Templates.
        if self.templates is None:
            templates, unigram_w = _default_templates()
        else:
            templates = []
            for t in list(self.templates):
                name = str(t.get("name", "ctx"))
                weight = float(cast(Any, t.get("weight", 1.0)))
                offsets_raw = t.get("offsets", None)
                if not isinstance(offsets_raw, list) or not offsets_raw:
                    raise ValueError("Each template must have non-empty offsets list")
                offsets = []
                for o in offsets_raw:
                    if (
                        not isinstance(o, (list, tuple))
                        or len(o) != 2
                        or not isinstance(o[0], (int, float))
                        or not isinstance(o[1], (int, float))
                    ):
                        raise ValueError("Offsets must be pairs like [-1,0]")
                    offsets.append((int(o[0]), int(o[1])))
                templates.append(ContextTemplate(name=name, offsets=tuple(offsets), weight=float(weight)))
            unigram_w = float(self.unigram_weight) if self.unigram_weight is not None else 0.1

        models, label_to_class = train_class_counts_models(
            train_tokens,
            train_labels,
            k=int(self.k),
            alpha=float(self.alpha),
            templates=templates,
            unigram_weight=float(unigram_w),
            num_classes=self.num_classes,
        )
        class_to_label = {int(v): int(k) for k, v in label_to_class.items()}

        (base_out / "label_to_class.json").write_text(
            json.dumps({str(k): int(v) for k, v in label_to_class.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        (base_out / "class_to_label.json").write_text(
            json.dumps({str(k): int(v) for k, v in class_to_label.items()}, indent=2) + "\n",
            encoding="utf-8",
        )

        # -----------------------------
        # 3) Evaluate accuracy on eval dataset.
        # -----------------------------
        eval_images, eval_labels = _collect_images_labels(
            cast(Dataset[Any], eval_ds),
            input_key=str(self.input_key),
            target_key=str(self.target_key),
            max_items=self.max_eval,
        )
        eval_tokens = codec.tokenize(eval_images, centers=centers, batch_size=int(self.tokenize_batch_size))

        correct = 0
        total = int(eval_tokens.shape[0])
        for i in range(total):
            pred_c = int(predict_class(models, eval_tokens[i]))
            pred_label = int(class_to_label.get(pred_c, pred_c))
            yi = int(eval_labels[i])
            correct += int(pred_label == yi)
        acc = float(correct) / float(max(1, total))

        metrics = {"eval_accuracy": float(acc), "eval_correct": int(correct), "eval_total": int(total)}
        (base_out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        logger.info(f"CCL eval_accuracy={acc:.4f} (n={total})")

        # -----------------------------
        # 4) Optional generation.
        # -----------------------------
        if bool(self.generate):
            try:
                from ccl.context_counts import sample_grid
                import matplotlib.pyplot as plt  # type: ignore
            except Exception as e:
                logger.warning(f"CCL generation skipped (missing deps): {type(e).__name__}: {e}")
            else:
                ht, wt = int(train_tokens.shape[1]), int(train_tokens.shape[2])
                # Assume channels inferred from data.
                channels = int(train_images.shape[1])
                n_per = int(max(1, self.n_gen_per_class))
                c = int(len(models))
                fig = plt.figure(figsize=(max(6, int(n_per * 1.2)), max(3, int(c * 0.8))), dpi=150)
                plot_idx = 1
                for ci in range(c):
                    for j in range(n_per):
                        g = sample_grid(models[ci], ht=ht, wt=wt, seed=int(self.seed + 1000 + ci * 100 + j))
                        img = codec.decode(g, centers=centers, channels=int(channels))
                        ax = fig.add_subplot(c, n_per, plot_idx)
                        if channels == 1:
                            ax.imshow(img[0], cmap="gray", vmin=0.0, vmax=1.0)
                        else:
                            ax.imshow(np.transpose(img, (1, 2, 0)))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if j == 0:
                            ax.set_ylabel(str(class_to_label.get(ci, ci)), rotation=0, labelpad=10)
                        plot_idx += 1
                plt.tight_layout(pad=0.2)
                out_path = base_out / "generated_grid.png"
                plt.savefig(out_path)
                logger.info(f"Saved CCL samples to {out_path}")

        system = CCLSystem(
            k=int(self.k),
            patch=int(self.patch),
            stride=int(self.stride),
            centers=centers,
            models=models,
            label_to_class=label_to_class,
            class_to_label=class_to_label,
            tokenize_batch_size=int(self.tokenize_batch_size),
            input_key=str(self.input_key),
        )
        return {"system": system, "device": torch.device("cpu"), "checkpoint_dir": base_out}

