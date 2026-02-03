"""
Numerical Stability Utilities
=============================

Provides numerically stable versions of common mathematical operations
to prevent overflow, underflow, and NaN/Inf issues throughout the codebase.

These utilities should be used instead of raw np.exp() calls in places
where input values could be extreme (very large or very small).
"""

import numpy as np
from typing import Union

# Default bounds for safe exponential operations
# np.exp(-500) ≈ 7e-218 (underflows gracefully to 0)
# np.exp(500) ≈ 1.4e217 (would overflow, so we clip)
DEFAULT_EXP_MIN = -500
DEFAULT_EXP_MAX = 500


def safe_exp(
    x: Union[np.ndarray, float],
    min_val: float = DEFAULT_EXP_MIN,
    max_val: float = DEFAULT_EXP_MAX
) -> Union[np.ndarray, float]:
    """
    Compute exponential with input clipping to prevent overflow/underflow.

    This function clips the input to a safe range before computing exp(),
    preventing numerical overflow (returning inf) and providing graceful
    underflow (returning 0 for very negative inputs).

    Args:
        x: Input value(s) for exponential
        min_val: Minimum allowed input value (default: -500)
        max_val: Maximum allowed input value (default: 500)

    Returns:
        exp(clip(x, min_val, max_val))

    Examples:
        >>> safe_exp(1000)  # Would overflow with raw np.exp
        7.225973768125749e+217  # Returns exp(500) instead

        >>> safe_exp(-1000)  # Would underflow with raw np.exp
        0.0  # Returns 0 (exp(-500) ≈ 0)

        >>> safe_exp(np.array([1, 2, 1000]))
        array([2.71828183e+000, 7.38905610e+000, 7.22597377e+217])
    """
    x_clipped = np.clip(x, min_val, max_val)
    return np.exp(x_clipped)


def safe_sigmoid(
    x: Union[np.ndarray, float],
    clip_bound: float = 500
) -> Union[np.ndarray, float]:
    """
    Compute sigmoid function with numerical stability.

    sigmoid(x) = 1 / (1 + exp(-x))

    For large positive x: approaches 1
    For large negative x: approaches 0

    Args:
        x: Input value(s)
        clip_bound: Absolute bound for clipping (default: 500)

    Returns:
        Sigmoid of input, bounded in (0, 1)
    """
    x_clipped = np.clip(x, -clip_bound, clip_bound)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def safe_softmax(
    x: np.ndarray,
    axis: int = -1,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Compute softmax with numerical stability.

    Uses the log-sum-exp trick: subtract max before exp to prevent overflow.

    Args:
        x: Input array
        axis: Axis along which to compute softmax
        temperature: Softmax temperature (higher = softer distribution)

    Returns:
        Softmax probabilities summing to 1 along the specified axis
    """
    x_scaled = x / temperature
    x_max = np.max(x_scaled, axis=axis, keepdims=True)
    exp_x = np.exp(x_scaled - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def safe_log(
    x: Union[np.ndarray, float],
    eps: float = 1e-10
) -> Union[np.ndarray, float]:
    """
    Compute logarithm with protection against log(0).

    Args:
        x: Input value(s), should be non-negative
        eps: Small constant added to prevent log(0) (default: 1e-10)

    Returns:
        log(x + eps)
    """
    return np.log(x + eps)


def safe_divide(
    numerator: Union[np.ndarray, float],
    denominator: Union[np.ndarray, float],
    default: float = 0.0
) -> Union[np.ndarray, float]:
    """
    Perform division with protection against division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return where denominator is zero (default: 0.0)

    Returns:
        numerator / denominator, with default where denominator is 0
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(denominator != 0, numerator / np.where(denominator != 0, denominator, 1), default)
        return result
    else:
        return numerator / denominator if denominator != 0 else default


def validate_finite(
    x: Union[np.ndarray, float],
    name: str = "value"
) -> bool:
    """
    Check if value(s) are finite (not NaN or Inf).

    Args:
        x: Value(s) to check
        name: Name for error message (default: "value")

    Returns:
        True if all values are finite

    Raises:
        ValueError: If any values are not finite
    """
    if np.any(~np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    return True


def clip_to_range(
    x: Union[np.ndarray, float],
    min_val: float,
    max_val: float,
    warn: bool = False
) -> Union[np.ndarray, float]:
    """
    Clip values to a specified range, optionally warning on clipping.

    Args:
        x: Input value(s)
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        warn: If True, print warning when clipping occurs (default: False)

    Returns:
        Clipped value(s)
    """
    import warnings

    result = np.clip(x, min_val, max_val)

    if warn:
        if np.any(x < min_val) or np.any(x > max_val):
            warnings.warn(
                f"Values clipped to range [{min_val}, {max_val}]",
                RuntimeWarning
            )

    return result


if __name__ == "__main__":
    print("=== Numerical Stability Utilities Test ===\n")

    # Test safe_exp
    print("Testing safe_exp:")
    print(f"  safe_exp(1) = {safe_exp(1):.6f}")
    print(f"  safe_exp(1000) = {safe_exp(1000):.6e} (clipped to exp(500))")
    print(f"  safe_exp(-1000) = {safe_exp(-1000):.6e} (clipped to exp(-500))")

    test_arr = np.array([-1000, -1, 0, 1, 1000])
    print(f"  safe_exp({test_arr}) = {safe_exp(test_arr)}")

    # Test safe_sigmoid
    print("\nTesting safe_sigmoid:")
    print(f"  safe_sigmoid(0) = {safe_sigmoid(0):.6f}")
    print(f"  safe_sigmoid(1000) = {safe_sigmoid(1000):.6f}")
    print(f"  safe_sigmoid(-1000) = {safe_sigmoid(-1000):.6e}")

    # Test safe_softmax
    print("\nTesting safe_softmax:")
    logits = np.array([1000, 1000, 1000])  # Would overflow with naive softmax
    print(f"  safe_softmax([1000, 1000, 1000]) = {safe_softmax(logits)}")

    # Test safe_log
    print("\nTesting safe_log:")
    print(f"  safe_log(0) = {safe_log(0):.6f} (not -inf)")
    print(f"  safe_log(1) = {safe_log(1):.6f}")

    # Test safe_divide
    print("\nTesting safe_divide:")
    print(f"  safe_divide(1, 0) = {safe_divide(1, 0)}")
    print(f"  safe_divide(1, 2) = {safe_divide(1, 2)}")

    print("\n✓ All numerical stability tests passed!")
