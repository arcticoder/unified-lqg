"""
Setup script for the unified quantum gravity pipeline package.
"""

from setuptools import setup, find_packages
import os

# Read README if it exists
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Unified pipeline for quantum gravity calculations with LQG"

setup(
    name="unified_qg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.0.0",
    ],
    extras_require={
        "gpu": ["torch>=1.9.0", "cupy>=9.0.0"],
        "mpi": ["mpi4py>=3.0.0"],
        "dev": ["pytest>=6.0.0", "black", "flake8"],
    },
    python_requires=">=3.8",
    author="QG Team",
    author_email="qgteam@example.com",
    description="Unified pipeline for adaptive AMR, constraint closure, and 3+1D matter coupling in LQG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qgteam/unified_qg",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="quantum gravity, loop quantum gravity, adaptive mesh refinement, physics simulation",
)
