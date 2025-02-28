from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "elr_lrt.dbpm",
        ["cpp/src/frequency_table.cpp", "cpp/src/patcher.cpp", "cpp/src/bindings.cpp"],
        include_dirs=["cpp/include"],
    ),
]

setup(
    name="elr_lrt",
    version="0.1",
    packages=["elr_lrt"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
