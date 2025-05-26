"""
Mathematical utilities for safe and consistent calculations.

This module provides functions to handle common mathematical operations
with proper error handling, bounds checking, and numerical stability.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def safe_divide(numerator: Union[float, np.ndarray, pd.Series], 
                denominator: Union[float, np.ndarray, pd.Series],
                default: float = 0.0,
                min_denominator: float = 1e-10) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely divide two numbers or arrays, handling zero/small denominators.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return when denominator is too small
        min_denominator: Minimum absolute value for denominator
        
    Returns:
        Result of division or default value
    """
    if isinstance(denominator, (int, float)):
        if abs(denominator) < min_denominator:
            return default
        return numerator / denominator
    
    # For arrays/series
    mask = np.abs(denominator) >= min_denominator
    result = np.full_like(numerator, default, dtype=float)
    
    if isinstance(denominator, pd.Series):
        result = pd.Series(result, index=denominator.index)
        result[mask] = numerator[mask] / denominator[mask]
    else:
        result[mask] = numerator[mask] / denominator[mask]
    
    return result


def normalize_to_sum(values: Union[np.ndarray, pd.Series], 
                     target_sum: float = 100.0,
                     min_value: Optional[float] = None,
                     tolerance: float = 1e-10) -> Union[np.ndarray, pd.Series]:
    """
    Normalize values to sum to a target value with guaranteed precision.
    
    Args:
        values: Array or series to normalize
        target_sum: Target sum (default 100 for percentages)
        min_value: Minimum allowed value after normalization
        tolerance: Tolerance for sum validation
        
    Returns:
        Normalized values that sum to exactly target_sum
    """
    # Handle empty or all-zero cases
    if len(values) == 0:
        return values
    
    # Convert to numpy for processing
    is_series = isinstance(values, pd.Series)
    if is_series:
        index = values.index
        values = values.values
    
    # Check for all zeros or NaN
    valid_mask = ~np.isnan(values) & (values >= 0)
    if not valid_mask.any() or values[valid_mask].sum() == 0:
        # Equal distribution among valid entries
        result = np.zeros_like(values, dtype=float)
        if valid_mask.any():
            result[valid_mask] = target_sum / valid_mask.sum()
        logger.warning(f"All values zero or invalid, applying equal distribution")
        
        if is_series:
            return pd.Series(result, index=index)
        return result
    
    # Normal case: scale to target sum
    current_sum = values[valid_mask].sum()
    scale_factor = target_sum / current_sum
    result = values * scale_factor
    
    # Apply minimum value constraint if specified
    if min_value is not None:
        below_min = result < min_value
        if below_min.any():
            # Set to minimum and redistribute excess
            result[below_min] = min_value
            excess = target_sum - result.sum()
            
            # Redistribute to values above minimum proportionally
            above_min = ~below_min & valid_mask
            if above_min.any() and result[above_min].sum() > 0:
                result[above_min] += excess * result[above_min] / result[above_min].sum()
    
    # Final adjustment to ensure exact sum (handles floating point errors)
    final_sum = result[valid_mask].sum()
    if abs(final_sum - target_sum) > tolerance:
        # Apply correction to largest value to minimize relative error
        if valid_mask.any():
            max_idx = np.argmax(result * valid_mask)
            result[max_idx] += target_sum - final_sum
    
    # Validate final result
    final_sum = result[valid_mask].sum()
    if abs(final_sum - target_sum) > tolerance:
        logger.error(f"Normalization failed: sum={final_sum}, target={target_sum}")
    
    if is_series:
        return pd.Series(result, index=index)
    return result


def calculate_growth_rate(current: float, previous: float, 
                         annualize: bool = False,
                         periods: float = 1.0,
                         max_rate: float = 500.0,
                         min_rate: float = -99.0) -> float:
    """
    Calculate growth rate with bounds and error handling.
    
    Args:
        current: Current period value
        previous: Previous period value
        annualize: Whether to annualize the growth rate
        periods: Number of periods (for annualization)
        max_rate: Maximum allowed growth rate (%)
        min_rate: Minimum allowed growth rate (%)
        
    Returns:
        Growth rate as a percentage
    """
    # Handle invalid inputs
    if pd.isna(current) or pd.isna(previous):
        return 0.0
    
    # Handle zero or negative previous value
    if previous <= 0:
        if current > 0:
            return max_rate  # Cap at max growth
        else:
            return 0.0  # No meaningful growth rate
    
    # Calculate raw growth rate
    growth_rate = ((current / previous) - 1) * 100
    
    # Annualize if requested
    if annualize and periods > 0 and periods != 1:
        if growth_rate > -100:  # Can't annualize -100% or worse
            growth_factor = 1 + growth_rate / 100
            annualized_factor = growth_factor ** (1 / periods)
            growth_rate = (annualized_factor - 1) * 100
        else:
            growth_rate = min_rate
    
    # Apply bounds
    return np.clip(growth_rate, min_rate, max_rate)


def calculate_cagr(start_value: float, end_value: float, 
                   years: float,
                   handle_negative: str = 'geometric_mean') -> Optional[float]:
    """
    Calculate CAGR with proper handling of edge cases including negative values.
    
    Args:
        start_value: Initial value
        end_value: Final value
        years: Number of years
        handle_negative: Method for handling negative values:
            - 'geometric_mean': Use geometric mean of absolute values with sign preservation
            - 'zero': Return 0 for any negative value
            - 'error': Raise error for negative values
            
    Returns:
        CAGR as a percentage or None if calculation not possible
    """
    # Validate years
    if years <= 0:
        logger.error(f"Invalid time period for CAGR: {years} years")
        return None
    
    # Handle special cases
    if pd.isna(start_value) or pd.isna(end_value):
        return None
    
    # Handle negative values based on method
    if start_value < 0 or end_value < 0:
        if handle_negative == 'zero':
            logger.warning(f"Negative values in CAGR calculation: start={start_value}, end={end_value}")
            return 0.0
        elif handle_negative == 'error':
            raise ValueError(f"CAGR undefined for negative values: start={start_value}, end={end_value}")
        elif handle_negative == 'geometric_mean':
            # Use modified CAGR for negative values
            if start_value < 0 and end_value < 0:
                # Both negative: use absolute values and preserve sign
                abs_cagr = (((abs(end_value) / abs(start_value)) ** (1 / years)) - 1) * 100
                return -abs_cagr  # Negative growth in negative territory
            elif start_value < 0 and end_value > 0:
                # Transition from negative to positive: very high growth
                return 200.0  # Cap at 200% CAGR
            else:  # start_value > 0 and end_value < 0
                # Transition from positive to negative: very negative growth
                return -50.0  # Cap at -50% CAGR
    
    # Handle zero start value
    if start_value == 0:
        if end_value > 0:
            return 100.0  # Cap at 100% CAGR for growth from zero
        else:
            return 0.0
    
    # Standard CAGR calculation
    try:
        cagr = (((end_value / start_value) ** (1 / years)) - 1) * 100
        
        # Apply reasonable bounds
        if cagr > 200:
            logger.warning(f"CAGR {cagr:.1f}% exceeds reasonable bounds, capping at 200%")
            return 200.0
        elif cagr < -50:
            logger.warning(f"CAGR {cagr:.1f}% below reasonable bounds, capping at -50%")
            return -50.0
        
        return cagr
        
    except (OverflowError, ValueError) as e:
        logger.error(f"CAGR calculation error: {e}")
        return 0.0


def apply_growth_bounds(values: Union[np.ndarray, pd.Series],
                       min_growth: float = -30.0,
                       max_growth: float = 50.0,
                       reference_period: Optional[int] = None) -> Union[np.ndarray, pd.Series]:
    """
    Apply growth rate bounds to a series of values.
    
    Args:
        values: Time series of values
        min_growth: Minimum allowed growth rate (%)
        max_growth: Maximum allowed growth rate (%)
        reference_period: Compare to N periods ago (default 1)
        
    Returns:
        Bounded values
    """
    if len(values) < 2:
        return values
    
    is_series = isinstance(values, pd.Series)
    if is_series:
        index = values.index
        values = values.values.copy()
    else:
        values = values.copy()
    
    reference_period = reference_period or 1
    
    # Apply bounds starting from reference_period
    for i in range(reference_period, len(values)):
        if pd.isna(values[i]) or pd.isna(values[i - reference_period]):
            continue
        
        prev_value = values[i - reference_period]
        if prev_value <= 0:
            continue
        
        # Calculate implied growth rate
        growth_rate = ((values[i] / prev_value) - 1) * 100
        
        # Apply bounds if exceeded
        if growth_rate > max_growth:
            values[i] = prev_value * (1 + max_growth / 100)
        elif growth_rate < min_growth:
            values[i] = prev_value * (1 + min_growth / 100)
    
    if is_series:
        return pd.Series(values, index=index)
    return values


def calculate_confidence_interval(forecast: Union[float, np.ndarray],
                                 periods_ahead: int,
                                 base_std: float,
                                 confidence_level: float = 0.95,
                                 method: str = 'sqrt_time') -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculate confidence intervals that grow appropriately with forecast horizon.
    
    Args:
        forecast: Point forecast value(s)
        periods_ahead: Number of periods into the future
        base_std: Base standard deviation (from model residuals)
        confidence_level: Confidence level (default 95%)
        method: Method for scaling intervals:
            - 'sqrt_time': Scale with square root of time (random walk assumption)
            - 'linear': Scale linearly with time
            - 'log': Scale logarithmically with time
            
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Scale standard deviation based on forecast horizon
    if method == 'sqrt_time':
        scaled_std = base_std * np.sqrt(periods_ahead)
    elif method == 'linear':
        scaled_std = base_std * periods_ahead
    elif method == 'log':
        scaled_std = base_std * np.log(1 + periods_ahead)
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    # Calculate bounds
    margin = z_score * scaled_std
    lower = forecast - margin
    upper = forecast + margin
    
    # Ensure lower bound is not negative for non-negative forecasts
    if isinstance(forecast, (int, float)):
        if forecast >= 0:
            lower = max(0, lower)
    else:
        lower = np.maximum(0, lower)
    
    return lower, upper


def validate_data_consistency(data: pd.DataFrame, 
                            value_col: str = 'Value',
                            group_col: Optional[str] = None,
                            expected_total: Optional[float] = None,
                            tolerance: float = 0.01) -> Tuple[bool, List[str]]:
    """
    Validate mathematical consistency of data.
    
    Args:
        data: DataFrame to validate
        value_col: Column containing values to check
        group_col: Column to group by (e.g., 'Year')
        expected_total: Expected sum for each group
        tolerance: Acceptable deviation (as percentage)
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for missing values
    if data[value_col].isna().any():
        n_missing = data[value_col].isna().sum()
        errors.append(f"{n_missing} missing values found in {value_col}")
    
    # Check for negative values if they shouldn't exist
    if (data[value_col] < 0).any():
        n_negative = (data[value_col] < 0).sum()
        errors.append(f"{n_negative} negative values found in {value_col}")
    
    # Check group totals if specified
    if group_col and expected_total is not None:
        groups = data.groupby(group_col)[value_col].sum()
        
        for group, total in groups.items():
            deviation = abs(total - expected_total)
            deviation_pct = (deviation / expected_total * 100) if expected_total > 0 else float('inf')
            
            if deviation_pct > tolerance:
                errors.append(f"{group_col}={group}: sum={total:.2f}, expected={expected_total:.2f} "
                            f"(deviation={deviation_pct:.2f}%)")
    
    return len(errors) == 0, errors


def format_market_value(value: float, precision: int = 1) -> str:
    """
    Format market values with appropriate units (B, M, K).
    
    Args:
        value: The market value to format
        precision: Number of decimal places for the formatted value
        
    Returns:
        Formatted string with appropriate unit
    """
    if pd.isna(value) or value == 0:
        return "$0"
    
    abs_value = abs(value)
    
    if abs_value >= 1e9:  # Billions
        return f"${value/1e9:.{precision}f}B"
    elif abs_value >= 1e6:  # Millions
        return f"${value/1e6:.{precision}f}M"
    elif abs_value >= 1e3:  # Thousands
        return f"${value/1e3:.{precision}f}K"
    else:  # Raw number
        return f"${value:,.{precision}f}"


def handle_outliers(values: Union[np.ndarray, pd.Series],
                   method: str = 'iqr',
                   threshold: float = 3.0,
                   replace_with: str = 'clip') -> Union[np.ndarray, pd.Series]:
    """
    Detect and handle outliers in data.
    
    Args:
        values: Data to process
        method: Detection method ('iqr', 'zscore', 'mad')
        threshold: Threshold for outlier detection
        replace_with: How to handle outliers ('clip', 'median', 'nan')
        
    Returns:
        Data with outliers handled
    """
    is_series = isinstance(values, pd.Series)
    if is_series:
        index = values.index
        values = values.values.copy()
    else:
        values = values.copy()
    
    # Remove NaN for calculations
    valid_mask = ~np.isnan(values)
    valid_values = values[valid_mask]
    
    if len(valid_values) < 3:
        # Not enough data for outlier detection
        if is_series:
            return pd.Series(values, index=index)
        return values
    
    # Detect outliers
    if method == 'iqr':
        q1 = np.percentile(valid_values, 25)
        q3 = np.percentile(valid_values, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
    elif method == 'zscore':
        mean = np.mean(valid_values)
        std = np.std(valid_values)
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
    elif method == 'mad':  # Median Absolute Deviation
        median = np.median(valid_values)
        mad = np.median(np.abs(valid_values - median))
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    # Handle outliers
    outlier_mask = valid_mask & ((values < lower_bound) | (values > upper_bound))
    
    if outlier_mask.any():
        if replace_with == 'clip':
            values[outlier_mask] = np.clip(values[outlier_mask], lower_bound, upper_bound)
        elif replace_with == 'median':
            values[outlier_mask] = np.median(valid_values)
        elif replace_with == 'nan':
            values[outlier_mask] = np.nan
        else:
            raise ValueError(f"Unknown replacement method: {replace_with}")
        
        n_outliers = outlier_mask.sum()
        logger.info(f"Handled {n_outliers} outliers using {method} method with {replace_with} replacement")
    
    if is_series:
        return pd.Series(values, index=index)
    return values