#!/usr/bin/env python
"""
Setup script for CQL-Sepsis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "Conservative Q-Learning for Sepsis Treatment"

setup(
    name="cql-sepsis",
    version="0.1.0",
    author="Anish Mariathasan",
    author_email="anish.mariathasan@imperial.ac.uk",
    description="Conservative Q-Learning for Sepsis Treatment in the ICU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/CQL-sepsis",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gymnasium>=0.28.0",
        "icu-sepsis>=1.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pandas>=1.4.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "logging": [
            "wandb>=0.15.0",
            "tensorboard>=2.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cql-train=scripts.03_train_cql:main",
            "cql-evaluate=scripts.05_evaluate_policies:main",
        ],
    },
)
