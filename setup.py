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

setup(
    name="elr_lrt",
    version="0.1",
    description="ELR-LRT sequence analysis package",
    author="Prabin Panta",
    author_email="pantaprabin30@gmail.com",
    packages=["elr_lrt"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pybind11',
    ],
    entry_points={
        'console_scripts': [
            'elr-lrt=elr_lrt:main',
        ],
    },
)
