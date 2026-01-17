"""
Environment capture utilities for experiment reproducibility.

This module provides functions to automatically capture environment metadata
including git commit, system information, and package versions.
"""

import subprocess
import platform
from typing import Dict, List
import sys


def get_git_hash() -> str:
    """Get current git commit hash. Returns 'unknown' if not git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch. Returns 'unknown' if not git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_package_versions(packages: List[str] = None) -> Dict[str, str]:
    """
    Get versions of specified packages.
    If packages is None, capture common ML packages.
    """
    if packages is None:
        packages = [
            "python", "numpy", "pandas", "scikit-learn",
            "torch", "xgboost", "mlflow", "shap"
        ]

    versions = {}

    # Python version
    versions["python"] = sys.version.split()[0]

    # Package versions
    for package in packages:
        if package == "python":
            continue
        try:
            if package == "torch":
                import torch
                versions[package] = torch.__version__
            elif package == "xgboost":
                import xgboost
                versions[package] = xgboost.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                versions[package] = version
        except ImportError:
            versions[package] = "not_installed"

    return versions


def get_system_info() -> Dict[str, str]:
    """Get system information"""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation()
    }


def get_environment(packages: List[str] = None) -> Dict[str, Dict]:
    """
    Capture complete environment metadata for reproducibility.

    Returns dict with:
    - git: {commit_hash, branch}
    - system: {os, os_version, architecture, ...}
    - packages: {package_name: version}
    """
    return {
        "git": {
            "commit_hash": get_git_hash(),
            "branch": get_git_branch()
        },
        "system": get_system_info(),
        "packages": get_package_versions(packages)
    }


def log_environment_to_mlflow(env: Dict = None):
    """
    Log environment metadata to active MLflow run.
    Logs as tags for git/system info, params for package versions.
    """
    if env is None:
        env = get_environment()

    import mlflow

    # Log git and system info as tags (not hyperparameters)
    mlflow.set_tags({
        "git.commit_hash": env["git"]["commit_hash"],
        "git.branch": env["git"]["branch"],
        "system.os": env["system"]["os"],
        "system.architecture": env["system"]["architecture"],
        "python.version": env["system"]["python_version"]
    })

    # Log package versions as params
    mlflow.log_params({f"env.{pkg}": ver for pkg, ver in env["packages"].items()})

    return env
