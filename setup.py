from setuptools import setup
import os
import sys

# Version should match the one in pyproject.toml
VERSION = "0.1.5"

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    
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
            include_dirs=[include_dir, "cpp/include"],
            cxx_std=17,
            extra_compile_args=["-fvisibility=hidden"],
            define_macros=[("VERSION_INFO", VERSION)]
        ),
    ]
    
    cmdclass = {"build_ext": build_ext}
except ImportError:
    # If pybind11 is not available, provide a dummy extension to allow sdist
    ext_modules = []
    cmdclass = {}
    print("WARNING: pybind11 not available - C++ extensions will not be built!")

setup(
    name="elr_lrt",
    version=VERSION,
    description="ELR-LRT sequence analysis package",
    author="Prabin Panta",
    author_email="pantaprabin30@gmail.com",
    packages=["elr_lrt"],
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
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
