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
    
    def _validate_input_data(self, data: pd.DataFrame, required_columns: List[str] = None) -> List[str]:
        """
        Comprehensive validation of input data for forecasting.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Basic structure validation
        if data is None:
            errors.append("Input data cannot be None")
            return errors
            
        if data.empty:
            errors.append("Input data cannot be empty")
            return errors
        
        # Required columns validation
        if required_columns is None:
            required_columns = ['date', 'value']
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return errors
        
        # Data quality validation
        if 'value' in data.columns:
            # Check for numeric values
            if not pd.api.types.is_numeric_dtype(data['value']):
                errors.append("Value column must contain numeric data")
            
            # Check for negative values
            if (data['value'] < 0).any():
                errors.append("Value column contains negative values")
            
            # Check for NaN/infinity
            if data['value'].isna().any():
                errors.append("Value column contains NaN values")
                
            if np.isinf(data['value']).any():
                errors.append("Value column contains infinity values")
            
            # Check for zero variance
            if data['value'].var() == 0:
                errors.append("Value column has zero variance (all values are identical)")
        
        # Date column validation
        if 'date' in data.columns:
            try:
                pd.to_datetime(data['date'])
            except (ValueError, TypeError):
                errors.append("Date column contains invalid date formats")
        
        # Temporal consistency validation
        if 'date' in data.columns and len(data) > 1:
            dates = pd.to_datetime(data['date']).sort_values()
            date_diffs = dates.diff().dropna()
            
            # Check for duplicate dates
            if dates.duplicated().any():
                errors.append("Date column contains duplicate values")
            
            # Check for reasonable time intervals (not too large gaps)
            max_gap = date_diffs.max()
            median_gap = date_diffs.median()
            if max_gap > median_gap * 5:  # Allow up to 5x median gap
                errors.append(f"Irregular time intervals detected (max gap: {max_gap}, median: {median_gap})")
        
        # Data sufficiency validation
        min_data_points = getattr(self, 'min_data_points', 3)
        if len(data) < min_data_points:
            errors.append(f"Insufficient data points: {len(data)} < {min_data_points} required")
        
        return errors
    
    def _validate_forecast_parameters(self, periods: int, frequency: str = 'Y') -> List[str]:
        """
        Validate forecasting parameters.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Periods validation
        if not isinstance(periods, int):
            errors.append("Periods must be an integer")
        elif periods <= 0:
            errors.append("Periods must be positive")
        elif periods > 100:  # Reasonable upper limit
            errors.append("Periods exceeds reasonable limit (100)")
        
        # Frequency validation
        valid_frequencies = ['Y', 'Q', 'M', 'W', 'D']
        if frequency not in valid_frequencies:
            errors.append(f"Invalid frequency '{frequency}'. Must be one of: {valid_frequencies}")
        
        return errors
    
    def _handle_data_quality_issues(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle common data quality issues with appropriate warnings.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()
        
        # Handle missing values
        if 'value' in cleaned_data.columns and cleaned_data['value'].isna().any():
            missing_count = cleaned_data['value'].isna().sum()
            logger.warning(f"Found {missing_count} missing values, using forward fill")
            cleaned_data['value'] = cleaned_data['value'].fillna(method='ffill')
            
            # If still missing (at the beginning), use backward fill
            if cleaned_data['value'].isna().any():
                cleaned_data['value'] = cleaned_data['value'].fillna(method='bfill')
        
        # Handle outliers (values beyond 3 standard deviations)
        if 'value' in cleaned_data.columns:
            mean_val = cleaned_data['value'].mean()
            std_val = cleaned_data['value'].std()
            outlier_threshold = 3 * std_val
            
            outliers = np.abs(cleaned_data['value'] - mean_val) > outlier_threshold
            if outliers.any():
                outlier_count = outliers.sum()
                logger.warning(f"Found {outlier_count} outliers, capping at 3 standard deviations")
                
                # Cap outliers at 3 standard deviations
                upper_bound = mean_val + outlier_threshold
                lower_bound = mean_val - outlier_threshold
                cleaned_data.loc[outliers, 'value'] = np.clip(
                    cleaned_data.loc[outliers, 'value'], 
                    lower_bound, 
                    upper_bound
                )
        
        return cleaned_data
    
    def _create_fallback_model(self, fallback_type: str = 'simple_linear'):
        """
        Create a simple fallback model when advanced models fail.
        
        Args:
            fallback_type: Type of fallback model to create
        """
        if fallback_type == 'simple_linear':
            from sklearn.linear_model import LinearRegression
            self.fallback_model = LinearRegression()
            self.is_fallback = True
            logger.warning(f"{self.name} using simple linear regression fallback")
        elif fallback_type == 'mean':
            self.fallback_model = None  # Will use mean prediction
            self.is_fallback = True
            self.fallback_mean = None
            logger.warning(f"{self.name} using mean value fallback")
        else:
            raise ValueError(f"Unknown fallback type: {fallback_type}")
    
    def _fit_fallback_model(self, data: pd.DataFrame) -> None:
        """
        Fit the fallback model to data.
        
        Args:
            data: Training data
        """
        if not hasattr(self, 'is_fallback') or not self.is_fallback:
            return
        
        if hasattr(self, 'fallback_model') and self.fallback_model is not None:
            # Linear regression fallback
            if 'date' in data.columns:
                # Use ordinal dates for linear regression
                dates = pd.to_datetime(data['date'])
                X = dates.astype(int).values.reshape(-1, 1)
                y = data['value'].values
                self.fallback_model.fit(X, y)
        else:
            # Mean fallback
            self.fallback_mean = data['value'].mean()
        
        self.fitted = True
    
    def _forecast_fallback(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast using fallback model.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency
            
        Returns:
            DataFrame with forecasted values
        """
        if not hasattr(self, 'is_fallback') or not self.is_fallback:
            raise ValueError("No fallback model available")
        
        # Generate future dates
        last_date = getattr(self, 'last_training_date', pd.Timestamp.now())
        if frequency == 'Y':
            future_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), 
                                       periods=periods, freq='YS')
        elif frequency == 'Q':
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), 
                                       periods=periods, freq='QS')
        elif frequency == 'M':
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                       periods=periods, freq='MS')
        else:
            # Default to yearly
            future_dates = pd.date_range(start=last_date + pd.DateOffset(years=1), 
                                       periods=periods, freq='YS')
        
        if hasattr(self, 'fallback_model') and self.fallback_model is not None:
            # Linear regression prediction
            X_future = future_dates.astype(int).values.reshape(-1, 1)
            predictions = self.fallback_model.predict(X_future)
        else:
            # Mean prediction
            predictions = np.full(periods, self.fallback_mean)
        
        return pd.DataFrame({
            'date': future_dates,
            'value': predictions,
            'model': self.name,
            'is_fallback': True
        })
    
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
    
    @staticmethod
    def _calculate_date_difference_days(date1, date2):
        """
        Calculate difference between two dates in days, handling different date types.
        
        Args:
            date1: First date (pandas Timestamp, numpy datetime64, etc.)
            date2: Second date (pandas Timestamp, numpy datetime64, etc.)
            
        Returns:
            Number of days between dates as float
        """
        try:
            diff = date2 - date1
            
            # Handle different date difference types
            if hasattr(diff, 'days'):
                # pandas Timedelta
                return float(diff.days)
            elif isinstance(diff, np.timedelta64):
                # numpy timedelta64
                return float(diff / np.timedelta64(1, 'D'))
            else:
                # Convert to pandas Timestamp and calculate difference
                try:
                    ts1 = pd.Timestamp(date1)
                    ts2 = pd.Timestamp(date2)
                    return float((ts2 - ts1).days)
                except (ValueError, TypeError):
                    # Fallback: assume 1 day difference
                    return 1.0
        except Exception:
            # Final fallback
            return 1.0
    
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