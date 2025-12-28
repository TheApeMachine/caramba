from __future__ import annotations

from pathlib import Path
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent


def compile_metal(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    src = HERE / "dba_decode.metal"
    air = out_dir / "dba_decode.air"
    metallib = out_dir / "dba_decode.metallib"

    subprocess.check_call(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(src), "-o", str(air)]
    )
    subprocess.check_call(
        ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(metallib)]
    )


class CustomBuild(BuildExtension):
    def run(self) -> None:
        # Build the Python extension first, then place the metallib next to the .so.
        super().run()
        compile_metal(Path(self.build_lib))


setup(
    name="caramba_metal_ops",
    ext_modules=[
        CppExtension(
            name="caramba_metal_ops",
            sources=[str(HERE / "ops.mm")],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                    "-fobjc-arc",
                ]
            },
            extra_link_args=[
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
            ],
        )
    ],
    cmdclass={"build_ext": CustomBuild},
)

