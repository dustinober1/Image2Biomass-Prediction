"""
Random seed management for reproducible experiments.

This module provides the SeedManager class which ensures reproducibility across
Python, NumPy, and PyTorch by setting consistent random seeds.

Reproducibility is critical for:
- REPRO-03: Comparing experiments across different runs
- Debugging: Re-running experiments produces identical results
- Scientific validity: Results can be verified by others

The SeedManager sets seeds for:
- Python's built-in random module
- NumPy's random number generator
- PyTorch's CPU and CUDA random number generators
- PyTorch's deterministic mode (cuDNN)

Example:
    >>> # Set seed for reproducible training
    >>> with SeedManager(42):
    ...     model.train(X, y)  # Results are reproducible

    >>> # Validate seed before use
    >>> seed = SeedManager.validate_seed("42")  # Returns 42
"""

import random
from typing import Optional

# Handle optional PyTorch dependency
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class SeedManager:
    """
    Manage random seeds for reproducibility across Python, NumPy, PyTorch.

    This class provides a context manager interface to set random seeds
    for all commonly used random number generators in ML experiments.

    Example:
        >>> # Basic usage
        >>> with SeedManager(42):
        ...     random.random()  # Always produces same value

        >>> # With validation
        >>> seed = SeedManager.validate_seed("42")
        >>> with SeedManager(seed):
        ...     np.random.rand()  # Reproducible result

    Attributes:
        seed: Random seed value (integer between 0 and 2^32-1)
        _original_states: Saved random states (for potential restoration)

    Note:
        For PyTorch, this class also sets deterministic mode which may
        impact performance. See PyTorch documentation on reproducibility.
    """

    def __init__(self, seed: int):
        """
        Initialize SeedManager with a specific seed value.

        Args:
            seed: Random seed value (will be validated and converted to int)

        Raises:
            ValueError: If seed validation fails (None, invalid type, out of range)

        Example:
            >>> sm = SeedManager(42)
            >>> sm.seed
            42
            >>> sm = SeedManager("42")
            >>> sm.seed
            42
        """
        self.seed = self.validate_seed(seed)
        self._original_states = None

    def set_seed(self):
        """
        Set random seed for all libraries.

        This method configures the random seed for:
        - Python's built-in random module
        - NumPy's random number generator (numpy.random)
        - PyTorch's CPU and CUDA generators (if available)
        - PyTorch's deterministic backend settings (if available)

        Example:
            >>> sm = SeedManager(42)
            >>> sm.set_seed()
            >>> # All random operations now use seed 42
        """
        # Set Python random seed
        random.seed(self.seed)

        # Set NumPy random seed
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            # NumPy not installed, skip
            pass

        # Set PyTorch random seed (if available)
        if HAS_TORCH and torch.is_available():
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            # Ensure deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __enter__(self):
        """
        Set seed on context entry.

        This method is called when entering the context manager and
        sets the random seed for all supported libraries.

        Returns:
            self (allows chaining context managers)

        Example:
            >>> with SeedManager(42):
            ...     # Seed is set for all libraries
            ...     model.train(X, y)
        """
        # Save original states (for potential future restoration)
        self._original_states = {
            'random': random.getstate(),
        }

        try:
            import numpy as np
            self._original_states['numpy'] = np.random.get_state()
        except ImportError:
            pass

        if HAS_TORCH and torch.is_available():
            self._original_states['torch'] = torch.get_rng_state()

        # Set the seed for all libraries
        self.set_seed()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context (optional restoration of original states).

        By default, we don't restore original states because:
        1. Seed persistence is usually desired for reproducibility
        2. Training should continue with the same seed
        3. Restoring might break downstream operations

        If you need to restore original states, you can access
        self._original_states after the context exits.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        # Usually we don't restore, seed persists for reproducibility
        # Original states are preserved in self._original_states if needed
        pass

    @staticmethod
    def validate_seed(seed) -> int:
        """
        Validate and convert seed to integer.

        This method ensures the seed is a valid integer within the
        acceptable range for random number generators.

        Args:
            seed: Seed value to validate (int, str, or other numeric type)

        Returns:
            Validated integer seed

        Raises:
            ValueError: If seed is None or cannot be converted to int
            ValueError: If seed is out of valid range (0 to 2^32-1)

        Example:
            >>> SeedManager.validate_seed(42)
            42
            >>> SeedManager.validate_seed("42")
            42
            >>> SeedManager.validate_seed(None)
            ValueError: Seed cannot be None
            >>> SeedManager.validate_seed(-1)
            ValueError: Seed must be between 0 and 2^32-1

        Note:
            Valid seed range is 0 to 2^32-1 (4,294,967,295) which is
            the standard range for most random number generators.
        """
        if seed is None:
            raise ValueError("Seed cannot be None")

        try:
            seed_int = int(seed)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid seed value: {seed}") from e

        # Validate range (0 to 2^32 - 1)
        if seed_int < 0 or seed_int > 2**32 - 1:
            raise ValueError(
                f"Seed must be between 0 and 2^32-1, got {seed_int}"
            )

        return seed_int

    def get_original_states(self) -> Optional[dict]:
        """
        Get the original random states saved on context entry.

        This can be useful if you need to restore the previous random
        state after exiting the context manager.

        Returns:
            Dictionary of original random states, or None if context
            not yet entered

        Example:
            >>> with SeedManager(42) as sm:
            ...     states = sm.get_original_states()
            ...     # Use states later if needed
        """
        return self._original_states
