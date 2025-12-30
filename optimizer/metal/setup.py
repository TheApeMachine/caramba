from __future__ import annotations

from pathlib import Path
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


HERE = Path(__file__).resolve().parent


def compile_metal(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        HERE / "dba_decode.metal",
        HERE / "rmsnorm.metal",
        HERE / "layernorm.metal",
        HERE / "rope.metal",
        HERE / "lion.metal",
    ]
    airs = [out_dir / f"{src.stem}.air" for src in sources]
    metallib = out_dir / "caramba_ops.metallib"

    for src, air in zip(sources, airs, strict=True):
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(src), "-o", str(air)]
        )
    subprocess.check_call(
        ["xcrun", "-sdk", "macosx", "metallib", *[str(a) for a in airs], "-o", str(metallib)]
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

