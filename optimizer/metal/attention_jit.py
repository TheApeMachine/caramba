"""JIT build + load the Metal attention training extension.

This is intentionally separate from `optimizer/metal/jit.py` to keep the
attention training kernels independent from the existing fused ops module.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

from optimizer.runtime import metal_supported, metal_build_tools_available


def _this_dir() -> Path:
    return Path(__file__).resolve().parent


_CACHED_MOD: Any | None = None
_CACHED_ERR: Exception | None = None


def _xcrun_find(tool: str) -> str:
    """Resolve a tool path from the active Xcode toolchain via xcrun."""
    try:
        out = subprocess.check_output(["xcrun", "-sdk", "macosx", "--find", str(tool)], stderr=subprocess.STDOUT)
        p = out.decode("utf-8", errors="replace").strip()
        if not p:
            raise RuntimeError(f"xcrun returned empty path for tool {tool!r}")
        return p
    except Exception as e:
        try:
            devdir = subprocess.check_output(["xcode-select", "-p"], stderr=subprocess.STDOUT).decode(
                "utf-8", errors="replace"
            ).strip()
        except Exception:
            devdir = "<unknown>"
        raise RuntimeError(
            f"Unable to locate required Xcode tool {tool!r} via xcrun.\n"
            f"Active developer dir: {devdir}\n\n"
            "Fix:\n"
            "  - Install Xcode Command Line Tools: `xcode-select --install`\n"
            "  - OR select Xcode.app:\n"
            "      `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`\n"
            "      `sudo xcodebuild -license accept`\n\n"
            "Verify:\n"
            "  `xcrun -sdk macosx --find metal`\n"
            "  `xcrun -sdk macosx --find metallib`\n"
        ) from e


def _compile_attention_metallib(*, out_dir: Path, verbose: bool) -> Path:
    """Compile attention training Metal shaders -> metallib in `out_dir`."""
    src = _this_dir() / "attention_train.metal"
    air = out_dir / "attention_train.air"
    metallib = out_dir / "caramba_attention_ops.metallib"

    metal = _xcrun_find("metal")
    metallib_tool = _xcrun_find("metallib")

    if metallib.exists():
        mt = metallib.stat().st_mtime
        if mt >= src.stat().st_mtime:
            return metallib

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [metal, "-c", str(src), "-o", str(air)]
    # Ensure the Metal compiler targets the correct AIR architecture on Apple Silicon.
    # Allow override via env for debugging / cross-compiling.
    explicit_arch = os.environ.get("CARAMBA_METAL_AIR_ARCH")
    if explicit_arch:
        cmd += ["-arch", explicit_arch]
    elif sys.platform == "darwin" and platform.machine() == "arm64":
        cmd += ["-arch", "air64"]
    if verbose:
        print("[caramba] compiling Metal attention shader:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to compile Metal attention shaders with the active toolchain.\n\n"
            f"Command:\n  {' '.join(cmd)}\n\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}\n"
        )

    cmd2 = [metallib_tool, str(air), "-o", str(metallib)]
    if verbose:
        print("[caramba] linking Metal attention metallib:", " ".join(cmd2))
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(
            "Failed to link Metal attention metallib (`metallib`).\n\n"
            f"Command:\n  {' '.join(cmd2)}\n\n"
            f"stdout:\n{proc2.stdout}\n\n"
            f"stderr:\n{proc2.stderr}\n"
        )
    return metallib


def load_caramba_metal_attention_ops(*, verbose: bool = False) -> Any:
    """Build (if needed) and import the `caramba_metal_attention_ops` extension."""
    global _CACHED_MOD, _CACHED_ERR
    if _CACHED_MOD is not None:
        return _CACHED_MOD
    if _CACHED_ERR is not None:
        raise _CACHED_ERR

    if not metal_supported():
        err = RuntimeError("Metal/MPS is not supported on this runtime")
        _CACHED_ERR = err
        raise err
    if not metal_build_tools_available():
        err = RuntimeError(
            "Metal build tools unavailable.\n\n"
            "caramba's fused Metal attention kernels require Xcode's Metal toolchain (`metal`, `metallib`).\n"
            "Install/select it:\n"
            "  - `xcode-select --install`\n"
            "  - or install Xcode.app then:\n"
            "      `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`\n"
            "      `sudo xcodebuild -license accept`\n\n"
            "Verify:\n"
            "  `xcrun -sdk macosx --find metal`\n"
            "  `xcrun -sdk macosx --find metallib`\n"
        )
        _CACHED_ERR = err
        raise err

    import torch.utils.cpp_extension as ce

    try:
        name = "caramba_metal_attention_ops"
        build_dir = Path(ce._get_build_directory(name, verbose=verbose))

        _compile_attention_metallib(out_dir=build_dir, verbose=verbose)

        src_ops = str(_this_dir() / "attention_ops.mm")
        mod = ce.load(
            name=name,
            sources=[src_ops],
            extra_cflags=["-O3", "-std=c++17", "-fobjc-arc", "-fblocks"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
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

