"""Platform module.

This module provides details about the current platform we are running on and
informs other modules about the capabilities of the platform.
"""
from __future__ import annotations

import os
import platform
from enum import Enum
from typing import Optional
import importlib.util
import torch

class PlatformType(Enum):
    """Platform type.

    A platform type is a type of platform, such as the operating system, architecture, and compiler.
    """
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"

    @classmethod
    def from_os(cls, os: str) -> PlatformType:
        """Convert an operating system to a platform type.

        A platform type is a type of platform, such as the operating system, architecture, and compiler.
        """
        raw = (os or "").lower()
        sys_name = platform.system().lower()

        if raw in {"nt"} or sys_name.startswith("windows"):
            return cls.WINDOWS

        if sys_name.startswith("darwin") or sys_name.startswith("mac"):
            return cls.MACOS

        if raw == "posix" or sys_name.startswith("linux"):
            return cls.LINUX

        return cls.UNKNOWN


class ArchitectureType(Enum):
    """Architecture type.

    An architecture type is a type of architecture, such as the architecture of the processor.
    """
    X86 = "x86"
    ARM = "arm"
    UNKNOWN = "unknown"

    @classmethod
    def from_arch(cls, arch: str) -> ArchitectureType:
        """Convert an architecture to a architecture type.

        An architecture type is a type of architecture, such as the architecture of the processor.
        """
        raw = (arch or "").lower()

        if raw in {"x86_64", "amd64", "x64", "i386", "i686"}:
            return cls.X86

        if raw.startswith(("arm", "aarch64")):
            return cls.ARM

        return cls.UNKNOWN


class DeviceType(Enum):
    """Device type.

    A device type is a type of device, such as the device of the processor.
    """
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    XPU = "xpu"
    TPU = "tpu"
    UNKNOWN = "unknown"

    @classmethod
    def from_device(cls, device: str) -> DeviceType:
        """Convert a device to a device type.

        A device type is a type of device, such as the device of the processor.
        """
        raw = (device or "").strip().lower()

        if not raw:
            return cls.UNKNOWN

        base = raw.split(":", 1)[0]
        aliases = {
            # common generic names
            "gpu": "cuda",
            "nvidia": "cuda",
            "metal": "mps",
            "apple": "mps",
        }
        base = aliases.get(base, base)

        try:
            return cls(base)
        except ValueError:
            return cls.UNKNOWN


class Platform:
    """Platform interface.

    A platform is a source of platform-specific information, such as the
    operating system, architecture, and compiler.
    """
    def __init__(self) -> None:
        self.os: PlatformType = PlatformType.from_os(os.name)
        self.arch: ArchitectureType = ArchitectureType.from_arch(platform.machine())
        self.device: DeviceType = self.get_device()

    def get_device(self) -> DeviceType:
        """Get the device type.

        A device type is a type of device, such as the device of the processor.
        """
        if torch.cuda.is_available():
            return DeviceType.CUDA

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceType.MPS

        if hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
            return DeviceType.XPU

        if importlib.util.find_spec("torch_xla") is not None:
            return DeviceType.TPU

        return DeviceType.CPU

    def torch_version(self) -> tuple[int, int]:
        """Parse torch.__version__ into (major, minor)."""
        version_str = str(torch.__version__).split("+")[0]
        parts = version_str.split(".")

        if len(parts) < 2:
            raise ValueError(f"Unexpected torch.__version__ format: {torch.__version__!r}")

        return int(parts[0]), int(parts[1])