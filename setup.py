from setuptools import setup
from setuptools.command.build_py import build_py


class PublicBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [
            (Package, Module, FilePath)
            for Package, Module, FilePath in super().find_package_modules(package, package_dir)
            if (Package, Module) != ("koco", "new_functions")
        ]


setup(cmdclass={"build_py": PublicBuildPy})