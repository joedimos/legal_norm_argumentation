"""Bundle immutable research inputs and migration scripts in wheel installations."""

from pathlib import Path
from shutil import copytree, rmtree

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithResources(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        for name in ("benchmarks", "schemas", "examples", "migrations"):
            destination = Path(self.build_lib) / "src" / "_data" / name
            if destination.exists():
                rmtree(destination)
            copytree(
                root / name,
                destination,
                dirs_exist_ok=True,
                ignore=lambda directory, names: [n for n in names if n == "__pycache__" or n.endswith(".pyc")],
            )


setup(cmdclass={"build_py": BuildWithResources})
