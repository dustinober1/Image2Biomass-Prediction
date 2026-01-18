"""
Setup configuration for mlflow-tracking package.

This setup.py defines the package metadata, dependencies, and CLI entry points.
Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="mlflow-tracking",
    version="0.1.0",
    description="Experiment tracking framework for ML research with advanced analytics",
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
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "analytics": [
            "shap>=0.46.0",
            "eli5>=0.13.0",
        ],
        "reporting": [
            "weasyprint>=60.0",
        ],
        "plots": [
            "plotly>=5.18.0",
        ],
        "all": [
            "shap>=0.46.0",
            "eli5>=0.13.0",
            "weasyprint>=60.0",
            "plotly>=5.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "exp-run=mlflow_tracking.cli:main",
            "exp-run-batch=mlflow_tracking.cli:main_batch",
            "exp-run-optimize=mlflow_tracking.cli:main_optimize",
            "exp-analyze-errors=mlflow_tracking.cli:main_analyze_errors",
            "exp-interpret=mlflow_tracking.cli:main_interpret",
            "exp-insights=mlflow_tracking.cli:main_insights",
        ],
    },
    python_requires=">=3.8",
)
