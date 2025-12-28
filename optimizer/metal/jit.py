"""JIT build + load the Metal extension.

We intentionally keep this separate from the main import path so `caramba` can
import/type-check on non-mac platforms without requiring Xcode toolchains.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from optimizer.runtime import METAL_BUILD_TOOLS_AVAILABLE, METAL_SUPPORTED


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


_CACHED_MOD: Any | None = None
_CACHED_ERR: Exception | None = None


def _compile_metallib(*, out_dir: Path, verbose: bool) -> Path:
    """Compile Metal shaders -> `caramba_ops.metallib` in `out_dir`."""

    sources = [
        _this_dir() / "dba_decode.metal",
        _this_dir() / "rmsnorm.metal",
        _this_dir() / "rope.metal",
        _this_dir() / "lion.metal",
    ]
    airs = [out_dir / f"{src.stem}.air" for src in sources]
    metallib = out_dir / "caramba_ops.metallib"

    # Rebuild only when missing or any source is newer.
    if metallib.exists():
        mt = metallib.stat().st_mtime
        if all(mt >= src.stat().st_mtime for src in sources):
            return metallib

    out_dir.mkdir(parents=True, exist_ok=True)

    # Compile each .metal source to .air.
    for src, air in zip(sources, airs, strict=True):
        cmd = [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(src),
            "-o",
            str(air),
        ]
        if verbose:
            print("[caramba] compiling Metal shader:", " ".join(cmd))
        subprocess.check_call(cmd)

    # Link all .air files into a single metallib.
    cmd2 = [
        "xcrun",
        "-sdk",
        "macosx",
        "metallib",
        *[str(air) for air in airs],
        "-o",
        str(metallib),
    ]
    if verbose:
        print("[caramba] linking Metal metallib:", " ".join(cmd2))
    subprocess.check_call(cmd2)
    return metallib


def load_caramba_metal_ops(*, verbose: bool = False) -> Any:
    """Build (if needed) and import the `caramba_metal_ops` extension.

    Returns the imported extension module. Raises on build/import failures.
    """
    global _CACHED_MOD, _CACHED_ERR
    if _CACHED_MOD is not None:
        return _CACHED_MOD
    if _CACHED_ERR is not None:
        raise _CACHED_ERR

    if not METAL_SUPPORTED:
        err = RuntimeError("Metal/MPS is not supported on this runtime")
        _CACHED_ERR = err
        raise err
    if not METAL_BUILD_TOOLS_AVAILABLE:
        err = RuntimeError(
            "Metal build tools unavailable (need Xcode command-line tools: `xcrun`)."
        )
        _CACHED_ERR = err
        raise err

    import torch.utils.cpp_extension as ce

    try:
        name = "caramba_metal_ops"
        build_dir = Path(ce._get_build_directory(name, verbose=verbose))

        _compile_metallib(out_dir=build_dir, verbose=verbose)

        # Build/load the ObjC++ extension. We intentionally place the metallib in the
        # same directory as the built .so so the extension can locate it via dladdr.
        src_ops = str(_this_dir() / "ops.mm")
        extra_cflags = [
            "-O3",
            "-std=c++17",
            "-fobjc-arc",
        ]
        extra_ldflags = [
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
        ]
        mod = ce.load(
            name=name,
            sources=[src_ops],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            with_cuda=False,
            is_python_module=True,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as e:
        _CACHED_ERR = e
        raise

    _CACHED_MOD = mod
    return mod

