from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "elr_lrt.dbpm",
        ["cpp/src/frequency_table.cpp", "cpp/src/patcher.cpp", "cpp/src/bindings.cpp"],
        include_dirs=["cpp/include"],
        cxx_std=17,  # Add C++17 support
    ),
]

# Version should match the one in pyproject.toml
setup(
    name="elr_lrt",
    version="0.1.2",  # Updated to match pyproject.toml
    description="ELR-LRT sequence analysis package",
    author="Prabin Panta",
    author_email="pantaprabin30@gmail.com",
    packages=["elr_lrt"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    # Other configuration like classifiers and dependencies are handled by pyproject.toml
    python_requires='>=3.7',  # Updated to match pyproject.toml
    install_requires=[
        'pybind11>=2.6.0',  # Updated to match dependency in pyproject.toml
    ],
    entry_points={
        'console_scripts': [
            'elr-lrt=elr_lrt:main',
        ],
    },
)
