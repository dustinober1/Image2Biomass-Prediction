"""
Resource manager for GPU/CPU detection and allocation.

This module provides ResourceManager class for detecting available hardware
resources (GPUs, CPUs) and managing allocation to prevent resource conflicts
during parallel experiment execution.
"""

import os
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class ResourceToken:
    """
    Token representing allocated resources.

    Attributes:
        gpu_id: GPU ID if allocated, None for CPU-only
        cpu_cores: Number of CPU cores allocated
        active: Whether resources are currently allocated
    """
    gpu_id: Optional[int]
    cpu_cores: int
    active: bool = True

    def release(self):
        """Mark resources as released."""
        self.active = False


class ResourceManager:
    """
    Detects and manages GPU/CPU resources for parallel experiment execution.

    This class provides hardware detection, resource allocation, and concurrency
    management to prevent resource conflicts when running multiple experiments
    in parallel.

    Attributes:
        _lock: Thread lock for thread-safe resource allocation
        _allocated_gpus: Set of currently allocated GPU IDs
        _allocated_cpus: Count of currently allocated CPU cores
        _total_gpus: Total number of available GPUs
        _total_cpus: Total number of available CPU cores
        _reserve_cores: Number of CPU cores to reserve for system processes

    Example:
        >>> rm = ResourceManager()
        >>> gpus = rm.get_available_gpus()  # [0, 1, 2, 3]
        >>> cpus = rm.get_available_cores()  # 6 (8 total - 2 reserved)
        >>> if rm.can_allocate(gpu_count=1):
        ...     with rm.allocate(gpu_id=0, cpu_cores=4) as token:
        ...         # Run experiment with GPU 0 and 4 CPU cores
        ...         pass
        ...     # Resources automatically released
    """

    _instance: Optional['ResourceManager'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """Implement singleton pattern for ResourceManager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        reserve_cores: int = 2,
        detect_gpus: bool = True
    ):
        """
        Initialize ResourceManager with hardware detection.

        Args:
            reserve_cores: Number of CPU cores to reserve for system processes
            detect_gpus: Whether to detect GPUs (disable for CPU-only systems)
        """
        if self._initialized:
            return

        self._lock = threading.Lock()
        self._allocated_gpus: set = set()
        self._allocated_cpus: int = 0
        self._reserve_cores: int = reserve_cores

        # Detect total CPUs
        self._total_cpus: int = os.cpu_count() or 1

        # Detect GPUs
        if detect_gpus:
            self._total_gpus: int = len(self._detect_gpus())
        else:
            self._total_gpus: int = 0

        self._initialized = True

    def _detect_gpus(self) -> List[int]:
        """
        Detect available GPUs using PyTorch or nvidia-smi.

        Returns:
            List of available GPU IDs (e.g., [0, 1, 2, 3])

        Tries:
            1. PyTorch CUDA detection (if torch available)
            2. nvidia-smi command via subprocess (fallback)
            3. Returns empty list if neither available
        """
        gpu_ids: List[int] = []

        # Try PyTorch CUDA detection
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_ids = list(range(gpu_count))
                return gpu_ids
        except ImportError:
            pass  # PyTorch not available, try nvidia-smi

        # Fallback to nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
                return gpu_ids
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass  # nvidia-smi not available or failed

        return gpu_ids  # Empty list if no GPUs detected

    def get_available_gpus(self) -> List[int]:
        """
        Get list of available GPU IDs.

        Returns:
            List of GPU IDs (e.g., [0, 1, 2, 3] for 4-GPU system)
            Returns empty list if no GPUs available
        """
        return list(range(self._total_gpus))

    def get_available_cores(self, reserve_cores: Optional[int] = None) -> int:
        """
        Get number of available CPU cores for experiments.

        Args:
            reserve_cores: Override default reserve_cores for this call

        Returns:
            Number of available CPU cores (total - reserved)
        """
        reserve = reserve_cores if reserve_cores is not None else self._reserve_cores
        return max(1, self._total_cpus - reserve)

    def can_allocate(self, gpu_count: int = 0, cpu_cores: int = 1) -> bool:
        """
        Check if requested resources are available for allocation.

        Args:
            gpu_count: Number of GPUs required
            cpu_cores: Number of CPU cores required

        Returns:
            True if resources available, False otherwise
        """
        with self._lock:
            # Check GPU availability
            if gpu_count > 0:
                available_gpus = self._total_gpus - len(self._allocated_gpus)
                if available_gpus < gpu_count:
                    return False

            # Check CPU availability
            available_cpus = self.get_available_cores() - self._allocated_cpus
            if available_cpus < cpu_cores:
                return False

            return True

    @contextmanager
    def allocate(self, gpu_id: Optional[int] = None, cpu_cores: int = 1):
        """
        Allocate resources with context manager for automatic cleanup.

        Args:
            gpu_id: Specific GPU ID to allocate (None for CPU-only)
            cpu_cores: Number of CPU cores to allocate

        Yields:
            ResourceToken with allocation details

        Raises:
            ValueError: If requested resources not available
        """
        with self._lock:
            # Validate GPU request
            if gpu_id is not None:
                if gpu_id >= self._total_gpus:
                    raise ValueError(
                        f"GPU {gpu_id} not available. "
                        f"System has {self._total_gpus} GPU(s)."
                    )
                if gpu_id in self._allocated_gpus:
                    raise ValueError(
                        f"GPU {gpu_id} already allocated. "
                        f"Available GPUs: {[g for g in range(self._total_gpus) if g not in self._allocated_gpus]}"
                    )

            # Validate CPU request
            available_cpus = self.get_available_cores() - self._allocated_cpus
            if cpu_cores > available_cpus:
                raise ValueError(
                    f"Cannot allocate {cpu_cores} CPU cores. "
                    f"Available: {available_cpus}, Allocated: {self._allocated_cpus}"
                )

            # Allocate resources
            if gpu_id is not None:
                self._allocated_gpus.add(gpu_id)
            self._allocated_cpus += cpu_cores

            token = ResourceToken(gpu_id=gpu_id, cpu_cores=cpu_cores, active=True)

        try:
            yield token
        finally:
            # Release resources
            with self._lock:
                if token.gpu_id is not None and token.gpu_id in self._allocated_gpus:
                    self._allocated_gpus.remove(token.gpu_id)
                self._allocated_cpus -= token.cpu_cores
                token.active = False

    def suggest_concurrent_experiments(
        self,
        gpus_per_exp: int = 0,
        cpu_per_exp: int = 1
    ) -> int:
        """
        Suggest safe concurrency limit based on available resources.

        Args:
            gpus_per_exp: Number of GPUs required per experiment
            cpu_per_exp: Number of CPU cores required per experiment

        Returns:
            Maximum number of experiments that can run concurrently

        Example:
            >>> rm = ResourceManager()
            >>> rm.suggest_concurrent_experiments(gpus_per_exp=1, cpu_per_exp=4)
            2  # Can run 2 GPU experiments with 4 CPU cores each
        """
        with self._lock:
            if gpus_per_exp > 0:
                # GPU-bound experiments
                max_by_gpu = self._total_gpus // gpus_per_exp if gpus_per_exp > 0 else float('inf')
                max_by_cpu = self.get_available_cores() // cpu_per_exp if cpu_per_exp > 0 else float('inf')
                return int(min(max_by_gpu, max_by_cpu))
            else:
                # CPU-only experiments
                return self.get_available_cores() // cpu_per_exp if cpu_per_exp > 0 else 1

    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get current resource allocation state.

        Returns:
            Dictionary with resource summary:
                - total_gpus: Total number of GPUs
                - allocated_gpus: List of allocated GPU IDs
                - available_gpus: Number of available GPUs
                - total_cpus: Total number of CPU cores
                - allocated_cpus: Number of allocated CPU cores
                - available_cpus: Number of available CPU cores
                - reserved_cpus: Number of reserved CPU cores
                - suggested_concurrent: Suggested max concurrent experiments

        Example:
            >>> rm = ResourceManager()
            >>> summary = rm.get_resource_summary()
            >>> print(f"GPUs: {summary['available_gpus']}/{summary['total_gpus']}")
            >>> print(f"CPUs: {summary['available_cpus']}/{summary['total_cpus']}")
        """
        with self._lock:
            return {
                'total_gpus': self._total_gpus,
                'allocated_gpus': list(self._allocated_gpus),
                'available_gpus': self._total_gpus - len(self._allocated_gpus),
                'total_cpus': self._total_cpus,
                'allocated_cpus': self._allocated_cpus,
                'available_cpus': self.get_available_cores(),
                'reserved_cpus': self._reserve_cores,
                'suggested_concurrent': self.suggest_concurrent_experiments()
            }

    def deallocate(self, token: ResourceToken):
        """
        Manually release allocated resources.

        Args:
            token: ResourceToken from allocate()

        Note:
            Context manager usage preferred for automatic cleanup.
            This method is for manual resource management.
        """
        with self._lock:
            if token.gpu_id is not None and token.gpu_id in self._allocated_gpus:
                self._allocated_gpus.remove(token.gpu_id)
            if token.active:
                self._allocated_cpus -= token.cpu_cores
            token.active = False
