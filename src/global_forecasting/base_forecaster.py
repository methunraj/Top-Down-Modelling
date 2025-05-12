"""
Base Forecaster Module - Foundation for all forecasting methods

This module provides the abstract base class that all forecasting methods must implement,
ensuring a consistent interface across different forecasting approaches.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting methods.
    
    This class defines the interface that all forecasting methods must implement,
    ensuring consistency across different approaches.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the forecaster.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.model = None
        self.fitted = False
        self.forecast_result = None
        self.forecast_dates = None
        self.confidence_intervals = None
        self.performance_metrics = {}
        
        # Initialize parameters
        self._initialize_parameters()
        
        logger.info(f"Initialized {self.name} forecaster")
    
    def _initialize_parameters(self) -> None:
        """
        Initialize model parameters from configuration.
        
        This method should be overridden by subclasses to set up specific parameters.
        """
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseForecaster':
        """
        Fit the forecasting model to historical data.
        
        Args:
            data: DataFrame containing historical data with at least 'date' and 'value' columns
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate a forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
            
        Returns:
            DataFrame with forecasted values
        """
        pass
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate forecast performance on test data.
        
        Args:
            test_data: DataFrame containing actual values for comparison
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        # Extract actual values from test data
        actual = test_data['value'].values
        
        # Get predictions for the test period
        pred_dates = test_data['date'].values
        predictions = self._predict_for_dates(pred_dates)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(actual, predictions)
        self.performance_metrics = metrics
        
        return metrics
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard performance metrics.
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            
        Returns:
            Dictionary of performance metrics
        """
        if len(actual) != len(predicted):
            raise ValueError("Length of actual and predicted arrays must match")
            
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Mean Absolute Percentage Error
        non_zero_mask = actual != 0
        mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Theil's U Statistic (modified to handle zeros)
        u_num = np.sqrt(np.mean((actual - predicted) ** 2))
        u_denom = np.sqrt(np.mean(actual ** 2) + np.mean(predicted ** 2))
        theil_u = u_num / u_denom if u_denom != 0 else np.nan
        
        return {
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Theil_U': theil_u
        }
    
    def _predict_for_dates(self, dates: np.ndarray) -> np.ndarray:
        """
        Generate predictions for specific dates.
        
        This method should be implemented by subclasses that support 
        predictions for arbitrary dates.
        
        Args:
            dates: Array of dates to predict for
            
        Returns:
            Array of predictions
        """
        raise NotImplementedError(
            f"{self.name} does not support prediction for arbitrary dates"
        )
    
    def get_confidence_intervals(self, level: float = 0.95) -> Optional[pd.DataFrame]:
        """
        Get confidence intervals for the forecast.
        
        Args:
            level: Confidence level (0-1)
            
        Returns:
            DataFrame with forecast and confidence intervals,
            or None if not available
        """
        if self.confidence_intervals is None:
            return None
            
        return self.confidence_intervals
    
    def reset(self) -> None:
        """
        Reset the forecaster to its initial state.
        
        This clears fitted model and results without changing configuration.
        """
        self.model = None
        self.fitted = False
        self.forecast_result = None
        self.forecast_dates = None
        self.confidence_intervals = None
        self.performance_metrics = {}
        
        # Reinitialize parameters
        self._initialize_parameters()
        
        logger.info(f"Reset {self.name} forecaster")
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the forecaster.
        
        Returns:
            Dictionary containing forecaster information
        """
        return {
            'name': self.name,
            'fitted': self.fitted,
            'parameters': self.config,
            'performance_metrics': self.performance_metrics
        }
    
    def plot(self, include_history: bool = True, 
            include_intervals: bool = True,
            figsize: Tuple[int, int] = (10, 6)) -> Any:
        """
        Plot the forecast results.
        
        Args:
            include_history: Whether to include historical data
            include_intervals: Whether to include confidence intervals
            figsize: Figure size tuple (width, height)
            
        Returns:
            Matplotlib figure object or None if plotting is not possible
        """
        if not self.fitted or self.forecast_result is None:
            logger.warning("Cannot plot: model not fitted or no forecast available")
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot forecast
            forecast_df = self.forecast_result
            ax.plot(forecast_df['date'], forecast_df['value'], 'b-', label='Forecast')
            
            # Plot confidence intervals if available and requested
            if include_intervals and self.confidence_intervals is not None:
                ci = self.confidence_intervals
                ax.fill_between(ci['date'], ci['lower'], ci['upper'], 
                               color='b', alpha=0.1, label='Confidence Interval')
            
            ax.set_title(f"{self.name} Forecast")
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None