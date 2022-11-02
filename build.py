import os
from pathlib import Path
import subprocess

from setuptools.command.build_ext import build_ext
from setuptools import Extension

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension):
        extdir = (Path.cwd() / self.get_ext_fullpath(ext.name)).resolve().parent

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        configuration = "Debug" if debug else "Release"
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = []
        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += ["-GNinja", f"-DCMAKE_MAKE_PROGRAM={ninja_executable}"]
                except ImportError:
                    pass
        else:
            raise NotImplementedError("Windwos is not supported yet")

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(exist_ok=True, parents=True)

        subprocess.run(["cmake", ext.sourcedir]+cmake_args, cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", "."]+build_args, cwd=build_temp, check=True)


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": [CMakeExtension("itesol_core")],
            "cmd_class": {"build_ext": CMakeBuild},
            "zip_safe": False,
        }
    )
