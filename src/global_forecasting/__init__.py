"""
Global Forecasting Module - Universal market forecasting capabilities

This package provides a comprehensive set of forecasting models for generating
global market forecasts. It includes statistical methods, machine learning approaches,
technology-specific methods, and ensemble techniques.
"""

from src.global_forecasting.base_forecaster import BaseForecaster

# Statistical forecasting methods
from src.global_forecasting.statistical import (
    CAGRForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster
)

# Technology-specific forecasting methods
from src.global_forecasting.tech_specific import (
    BassDiffusionForecaster,
    GompertzCurveForecaster,
    TechnologySCurveForecaster
)

# Ensemble forecasting methods
from src.global_forecasting.ensemble import (
    SimpleAverageEnsemble,
    WeightedEnsembleForecaster
)

# Dictionary of available forecasters for easy access
AVAILABLE_FORECASTERS = {
    # Statistical methods
    'cagr': CAGRForecaster,
    'moving_average': MovingAverageForecaster,
    'exponential_smoothing': ExponentialSmoothingForecaster,
    
    # Technology-specific methods
    'bass_diffusion': BassDiffusionForecaster,
    'gompertz_curve': GompertzCurveForecaster,
    'tech_s_curve': TechnologySCurveForecaster,
    
    # Ensemble methods
    'simple_average_ensemble': SimpleAverageEnsemble,
    'weighted_ensemble': WeightedEnsembleForecaster,
}

# Dictionary mapping friendly names to forecaster classes
FORECASTER_NAMES = {
    # Statistical methods
    'CAGR': CAGRForecaster,
    'Moving Average': MovingAverageForecaster,
    'Exponential Smoothing': ExponentialSmoothingForecaster,
    
    # Technology-specific methods
    'Bass Diffusion': BassDiffusionForecaster,
    'Gompertz Curve': GompertzCurveForecaster,
    'Technology S-Curve': TechnologySCurveForecaster,
    
    # Ensemble methods
    'Simple Average Ensemble': SimpleAverageEnsemble,
    'Weighted Ensemble': WeightedEnsembleForecaster,
}

# Categories of forecasters
FORECASTER_CATEGORIES = {
    'Statistical Methods': [
        'CAGR',
        'Moving Average',
        'Exponential Smoothing',
    ],
    'Technology-Specific Methods': [
        'Bass Diffusion',
        'Gompertz Curve',
        'Technology S-Curve',
    ],
    'Ensemble Methods': [
        'Simple Average Ensemble',
        'Weighted Ensemble',
    ],
}

def create_forecaster(name: str, config: dict = None):
    """
    Create a forecaster instance by name.
    
    Args:
        name: Name of the forecaster (either key from AVAILABLE_FORECASTERS or friendly name)
        config: Optional configuration dictionary
        
    Returns:
        Instance of the requested forecaster
    
    Raises:
        ValueError: If the forecaster name is not recognized
    """
    # Try to match by key in AVAILABLE_FORECASTERS
    if name in AVAILABLE_FORECASTERS:
        forecaster_class = AVAILABLE_FORECASTERS[name]
        return forecaster_class(config)
    
    # Try to match by friendly name
    for friendly_name, forecaster_class in FORECASTER_NAMES.items():
        if name.lower() == friendly_name.lower():
            return forecaster_class(config)
    
    # Not found
    raise ValueError(f"Unknown forecaster: {name}")

def get_available_forecasters():
    """
    Get a list of available forecasters.
    
    Returns:
        Dictionary mapping category names to lists of forecaster names
    """
    return FORECASTER_CATEGORIES.copy()

def get_forecaster_description(name: str):
    """
    Get a description of a forecaster.
    
    Args:
        name: Name of the forecaster
        
    Returns:
        Dictionary with forecaster information
    
    Raises:
        ValueError: If the forecaster name is not recognized
    """
    # Try to match by key in AVAILABLE_FORECASTERS
    if name in AVAILABLE_FORECASTERS:
        forecaster_class = AVAILABLE_FORECASTERS[name]
    elif name in FORECASTER_NAMES:
        # Match by friendly name
        forecaster_class = FORECASTER_NAMES[name]
    else:
        # Try case-insensitive match with friendly names
        for friendly_name, cls in FORECASTER_NAMES.items():
            if name.lower() == friendly_name.lower():
                forecaster_class = cls
                break
        else:
            raise ValueError(f"Unknown forecaster: {name}")
    
    # Get the docstring
    docstring = forecaster_class.__doc__ or "No description available."
    
    # Clean up the docstring (remove leading whitespace from lines)
    lines = docstring.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    cleaned_docstring = '\n'.join(line for line in cleaned_lines if line)
    
    # Extract a short description (first paragraph)
    short_description = cleaned_docstring.split('\n\n')[0]
    
    return {
        'name': name,
        'class': forecaster_class.__name__,
        'short_description': short_description,
        'description': cleaned_docstring,
    }