"""
Global Forecasting Module - Universal market forecasting capabilities

This package provides a comprehensive set of forecasting models for generating
global market forecasts. It includes statistical methods, machine learning approaches,
technology-specific methods, ensemble techniques, and time series-specific models.
"""

import warnings

from src.global_forecasting.base_forecaster import BaseForecaster

# Statistical forecasting methods
from src.global_forecasting.statistical import (
    CAGRForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    ARIMAForecaster,
    SARIMAForecaster,
    RegressionForecaster
)

# Technology-specific forecasting methods
from src.global_forecasting.tech_specific import (
    BassDiffusionForecaster,
    GompertzCurveForecaster,
    TechnologySCurveForecaster
)

# Market-specific forecasting methods
from src.global_forecasting.market_models import (
    FisherPryForecaster,
    HarveyLogisticForecaster,
    NortonBassForecaster,
    LotkaVolterraForecaster
)

# Time series-specific forecasting methods - imported conditionally
try:
    from src.global_forecasting.time_series_models import VARForecaster
    VAR_AVAILABLE = True
except ImportError:
    VAR_AVAILABLE = False
    warnings.warn("statsmodels not installed. VARForecaster will not be available.")

try:
    from src.global_forecasting.time_series_models import TemporalFusionTransformerForecaster
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    warnings.warn("PyTorch/PyTorch Forecasting not installed. TemporalFusionTransformerForecaster will not be available.")

try:
    from src.global_forecasting.time_series_models import DeepARForecaster
    DEEPAR_AVAILABLE = True
except ImportError:
    DEEPAR_AVAILABLE = False
    warnings.warn("MXNet/GluonTS not installed. DeepARForecaster will not be available.")

# Hybrid forecasting methods - imported conditionally
try:
    from src.global_forecasting.hybrid_models import TBATSForecaster
    TBATS_AVAILABLE = True
except ImportError:
    TBATS_AVAILABLE = False
    warnings.warn("TBATS not installed. TBATSForecaster will not be available.")

try:
    from src.global_forecasting.hybrid_models import NBEATSForecaster
    NBEATS_AVAILABLE = True
except ImportError:
    NBEATS_AVAILABLE = False
    warnings.warn("TensorFlow not installed. NBEATSForecaster will not be available.")

try:
    from src.global_forecasting.hybrid_models import HybridETSARIMAForecaster
    HYBRID_ETSARIMA_AVAILABLE = True
except ImportError:
    HYBRID_ETSARIMA_AVAILABLE = False
    warnings.warn("Required packages for HybridETSARIMAForecaster not installed.")

# Probabilistic forecasting methods - imported conditionally
try:
    from src.global_forecasting.probabilistic_models import BayesianStructuralTimeSeriesForecaster
    BSTS_AVAILABLE = True
except ImportError:
    BSTS_AVAILABLE = False
    warnings.warn("TensorFlow Probability or Stan not installed. BayesianStructuralTimeSeriesForecaster will not be available.")

try:
    from src.global_forecasting.probabilistic_models import GaussianProcessForecaster
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False
    warnings.warn("scikit-learn or GPyTorch not installed. GaussianProcessForecaster will not be available.")

# Ensemble forecasting methods
from src.global_forecasting.ensemble import (
    SimpleAverageEnsemble,
    WeightedEnsembleForecaster
)

# Machine learning forecasting methods - imported conditionally
try:
    from src.global_forecasting.ml_models import ProphetForecaster
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not installed. ProphetForecaster will not be available.")

try:
    from src.global_forecasting.ml_models import XGBoostForecaster
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. XGBoostForecaster will not be available.")

try:
    from src.global_forecasting.ml_models import LSTMForecaster
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    warnings.warn("TensorFlow not installed. LSTMForecaster will not be available.")

# Initialize forecaster dictionaries
AVAILABLE_FORECASTERS = {
    # Statistical methods
    'cagr': CAGRForecaster,
    'moving_average': MovingAverageForecaster,
    'exponential_smoothing': ExponentialSmoothingForecaster,
    'arima': ARIMAForecaster,
    'sarima': SARIMAForecaster,
    'regression': RegressionForecaster,
    
    # Technology-specific methods
    'bass_diffusion': BassDiffusionForecaster,
    'gompertz_curve': GompertzCurveForecaster,
    'tech_s_curve': TechnologySCurveForecaster,
    
    # Market-specific methods
    'fisher_pry': FisherPryForecaster,
    'harvey_logistic': HarveyLogisticForecaster,
    'norton_bass': NortonBassForecaster,
    'lotka_volterra': LotkaVolterraForecaster,
    
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
    'ARIMA': ARIMAForecaster,
    'SARIMA': SARIMAForecaster,
    'Regression': RegressionForecaster,
    
    # Technology-specific methods
    'Bass Diffusion': BassDiffusionForecaster,
    'Gompertz Curve': GompertzCurveForecaster,
    'Technology S-Curve': TechnologySCurveForecaster,
    
    # Market-specific methods
    'Fisher-Pry': FisherPryForecaster,
    'Harvey Logistic': HarveyLogisticForecaster,
    'Norton-Bass': NortonBassForecaster,
    'Lotka-Volterra': LotkaVolterraForecaster,
    
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
        'ARIMA',
        'SARIMA',
        'Regression',
    ],
    'Technology-Specific Methods': [
        'Bass Diffusion',
        'Gompertz Curve',
        'Technology S-Curve',
    ],
    'Market-Specific Methods': [
        'Fisher-Pry',
        'Harvey Logistic',
        'Norton-Bass',
        'Lotka-Volterra',
    ],
    'Hybrid Methods': [],  # Will be populated conditionally
    'Probabilistic Methods': [],  # Will be populated conditionally
    'Machine Learning Methods': [],  # Will be populated conditionally
    'Ensemble Methods': [
        'Simple Average Ensemble',
        'Weighted Ensemble',
    ],
    'Time Series-Specific Methods': [],  # Will be populated conditionally
}

# Add machine learning models conditionally based on availability
if PROPHET_AVAILABLE:
    AVAILABLE_FORECASTERS['prophet'] = ProphetForecaster
    FORECASTER_NAMES['Prophet'] = ProphetForecaster
    FORECASTER_CATEGORIES['Machine Learning Methods'].append('Prophet')

if XGBOOST_AVAILABLE:
    AVAILABLE_FORECASTERS['xgboost'] = XGBoostForecaster
    FORECASTER_NAMES['XGBoost'] = XGBoostForecaster
    FORECASTER_CATEGORIES['Machine Learning Methods'].append('XGBoost')

if LSTM_AVAILABLE:
    AVAILABLE_FORECASTERS['lstm'] = LSTMForecaster
    FORECASTER_NAMES['LSTM'] = LSTMForecaster
    FORECASTER_CATEGORIES['Machine Learning Methods'].append('LSTM')

# Add hybrid models conditionally based on availability
if TBATS_AVAILABLE:
    AVAILABLE_FORECASTERS['tbats'] = TBATSForecaster
    FORECASTER_NAMES['TBATS'] = TBATSForecaster
    FORECASTER_CATEGORIES['Hybrid Methods'].append('TBATS')

if NBEATS_AVAILABLE:
    AVAILABLE_FORECASTERS['nbeats'] = NBEATSForecaster
    FORECASTER_NAMES['NBEATS'] = NBEATSForecaster
    FORECASTER_CATEGORIES['Hybrid Methods'].append('NBEATS')

if HYBRID_ETSARIMA_AVAILABLE:
    AVAILABLE_FORECASTERS['hybrid_etsarima'] = HybridETSARIMAForecaster
    FORECASTER_NAMES['Hybrid ETS-ARIMA'] = HybridETSARIMAForecaster
    FORECASTER_CATEGORIES['Hybrid Methods'].append('Hybrid ETS-ARIMA')

# If no ML methods are available, remove the category
if len(FORECASTER_CATEGORIES['Machine Learning Methods']) == 0:
    del FORECASTER_CATEGORIES['Machine Learning Methods']

# If no hybrid methods are available, remove the category
if len(FORECASTER_CATEGORIES['Hybrid Methods']) == 0:
    del FORECASTER_CATEGORIES['Hybrid Methods']

# Add probabilistic models conditionally based on availability
if BSTS_AVAILABLE:
    AVAILABLE_FORECASTERS['bsts'] = BayesianStructuralTimeSeriesForecaster
    FORECASTER_NAMES['Bayesian Structural Time Series'] = BayesianStructuralTimeSeriesForecaster
    FORECASTER_CATEGORIES['Probabilistic Methods'].append('Bayesian Structural Time Series')

if GP_AVAILABLE:
    AVAILABLE_FORECASTERS['gaussian_process'] = GaussianProcessForecaster
    FORECASTER_NAMES['Gaussian Process'] = GaussianProcessForecaster
    FORECASTER_CATEGORIES['Probabilistic Methods'].append('Gaussian Process')

# If no probabilistic methods are available, remove the category
if len(FORECASTER_CATEGORIES['Probabilistic Methods']) == 0:
    del FORECASTER_CATEGORIES['Probabilistic Methods']

# Add time series-specific models conditionally based on availability
if VAR_AVAILABLE:
    AVAILABLE_FORECASTERS['var'] = VARForecaster
    FORECASTER_NAMES['VAR'] = VARForecaster
    FORECASTER_CATEGORIES['Time Series-Specific Methods'].append('VAR')

if TFT_AVAILABLE:
    AVAILABLE_FORECASTERS['temporal_fusion_transformer'] = TemporalFusionTransformerForecaster
    FORECASTER_NAMES['Temporal Fusion Transformer'] = TemporalFusionTransformerForecaster
    FORECASTER_CATEGORIES['Time Series-Specific Methods'].append('Temporal Fusion Transformer')

if DEEPAR_AVAILABLE:
    AVAILABLE_FORECASTERS['deepar'] = DeepARForecaster
    FORECASTER_NAMES['DeepAR'] = DeepARForecaster
    FORECASTER_CATEGORIES['Time Series-Specific Methods'].append('DeepAR')

# If no time series-specific methods are available, remove the category
if len(FORECASTER_CATEGORIES['Time Series-Specific Methods']) == 0:
    del FORECASTER_CATEGORIES['Time Series-Specific Methods']

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
    forecaster_class = None
    
    # Try to match by key in AVAILABLE_FORECASTERS
    if name in AVAILABLE_FORECASTERS:
        forecaster_class = AVAILABLE_FORECASTERS[name]
    else:
        # Try to match by friendly name
        for friendly_name, cls in FORECASTER_NAMES.items():
            if name.lower() == friendly_name.lower():
                forecaster_class = cls
                break
    
    # Not found
    if forecaster_class is None:
        raise ValueError(f"Unknown forecaster: {name}")
    
    try:
        # Initialize the forecaster
        forecaster = forecaster_class(config)
        return forecaster
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to initialize forecaster '{name}': {str(e)}")
        raise RuntimeError(f"Failed to initialize forecaster '{name}': {str(e)}") from e

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