"""
Setup configuration for mlflow-tracking package.

This setup.py defines the package metadata, dependencies, and CLI entry points.
Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="mlflow-tracking",
    version="0.1.0",
    description="Experiment tracking framework for ML research",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "mlflow>=3.8.0",
        "pyyaml",
        "pydantic>=2.0",
        "jinja2",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "optuna>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "exp-run=mlflow_tracking.cli:main",
            "exp-run-batch=mlflow_tracking.cli:main_batch",
            "exp-run-optimize=mlflow_tracking.cli:main_optimize",
        ],
    },
    python_requires=">=3.8",
)
