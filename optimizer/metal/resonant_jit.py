"""JIT build + load the Metal resonant update extension.

This extension provides fused pointwise kernels used by the resonant router.
Separating it from the main ops module keeps the build unit small and focused.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

from caramba.optimizer.runtime import metal_supported, metal_build_tools_available


def this_dir() -> Path:
    """Get directory of this module.

    Used to locate the Metal shader and ObjC++ binding sources.
    """
    return Path(__file__).resolve().parent


cached_module: Any | None = None
cached_error: Exception | None = None


def xcrun_find(tool: str) -> str:
    """Resolve a tool path from the active Xcode toolchain via xcrun.

    Ensures we use the same toolchain as the active developer directory.
    """
    try:
        out = subprocess.check_output(
            ["xcrun", "-sdk", "macosx", "--find", str(tool)],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        msg = (e.output or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "Failed to resolve Xcode toolchain via xcrun.\n\n"
            f"Tool: {tool!r}\n"
            f"xcrun output:\n{msg}\n"
        ) from e
    p = out.decode("utf-8", errors="replace").strip()
    if not p:
        raise RuntimeError(f"xcrun returned empty path for tool {tool!r}")
    return p


def compile_resonant_metallib(*, out_dir: Path, verbose: bool) -> Path:
    """Compile resonant Metal shaders -> metallib in `out_dir`."""
    src = this_dir() / "resonant_update.metal"
    if not src.exists():
        raise FileNotFoundError(f"resonant_update.metal not found at {src}")
    air = out_dir / "resonant_update.air"
    metallib = out_dir / "caramba_resonant_ops.metallib"

    if metallib.exists():
        mt = metallib.stat().st_mtime
        if mt >= src.stat().st_mtime:
            return metallib

    out_dir.mkdir(parents=True, exist_ok=True)

    metal = xcrun_find("metal")
    metallib_tool = xcrun_find("metallib")

    cmd = [metal, "-c", str(src), "-o", str(air)]
    explicit_arch = os.environ.get("CARAMBA_METAL_AIR_ARCH")
    if explicit_arch:
        cmd += ["-arch", explicit_arch]
    elif sys.platform == "darwin" and platform.machine() == "arm64":
        cmd += ["-arch", "air64"]
    if verbose:
        print("[caramba] compiling Metal resonant shader:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to compile Metal resonant shaders with the active toolchain.\n\n"
            f"Command:\n  {' '.join(cmd)}\n\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}\n"
        )

    cmd2 = [metallib_tool, str(air), "-o", str(metallib)]
    if verbose:
        print("[caramba] linking Metal resonant metallib:", " ".join(cmd2))
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(
            "Failed to link Metal resonant metallib (`metallib`).\n\n"
            f"Command:\n  {' '.join(cmd2)}\n\n"
            f"stdout:\n{proc2.stdout}\n\n"
            f"stderr:\n{proc2.stderr}\n"
        )
    return metallib


def load_caramba_metal_resonant_ops(*, verbose: bool = False) -> Any:
    """Build (if needed) and import the `caramba_metal_resonant_ops` extension."""
    global cached_module, cached_error
    if cached_module is not None:
        return cached_module
    if cached_error is not None:
        raise cached_error

    if not metal_supported():
        err = RuntimeError("Metal/MPS is not supported on this runtime")
        cached_error = err
        raise err
    if not metal_build_tools_available():
        err = RuntimeError(
            "Metal build tools unavailable.\n\n"
            "caramba's fused Metal resonant kernels require Xcode's Metal toolchain (`metal`, `metallib`).\n"
            "Install/select it:\n"
            "  - `xcode-select --install`\n"
            "  - or install Xcode.app then:\n"
            "      `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`\n"
            "      `sudo xcodebuild -license accept`\n\n"
            "Verify:\n"
            "  `xcrun -sdk macosx --find metal`\n"
            "  `xcrun -sdk macosx --find metallib`\n"
        )
        cached_error = err
        raise err

    import torch.utils.cpp_extension as ce

    try:
        name = "caramba_metal_resonant_ops"
        base_build_dir = Path(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                str(Path.home() / ".cache" / "torch_extensions"),
            )
        ).expanduser()
        build_dir = base_build_dir / name
        compile_resonant_metallib(out_dir=build_dir, verbose=verbose)
        src_ops = str(this_dir() / "resonant_ops.mm")
        mod = ce.load(
            name=name,
            sources=[src_ops],
            extra_cflags=["-O3", "-std=c++17", "-fobjc-arc"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            with_cuda=False,
            is_python_module=True,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as e:
        cached_error = e
        raise

    cached_module = mod
    return mod

