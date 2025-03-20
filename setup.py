from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import os
import sys

# Get the absolute path to the cpp include directory
include_dir = os.path.abspath("cpp/include")

# Configure the extension with more explicit options
ext_modules = [
    Pybind11Extension(
        "elr_lrt.dbpm",
        sources=[
            "cpp/src/frequency_table.cpp", 
            "cpp/src/patcher.cpp", 
            "cpp/src/bindings.cpp"
        ],
        include_dirs=[include_dir],
        cxx_std=17,
        extra_compile_args=["-fvisibility=hidden"],
        define_macros=[("VERSION_INFO", "0.1.2")]
    ),
]

# Version should match the one in pyproject.toml
setup(
    name="elr_lrt",
    version="0.1.3", 
    description="ELR-LRT sequence analysis package",
    author="Prabin Panta",
    author_email="pantaprabin30@gmail.com",
    packages=["elr_lrt"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires='>=3.7',
    install_requires=[
        'pybind11>=2.6.0',
    ],
    entry_points={
        'console_scripts': [
            'elr-lrt=elr_lrt:main',
        ],
    },
)
