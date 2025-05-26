"""
Ensemble Forecasting Module - Combining multiple forecasting methods

This module provides implementations of ensemble forecasting methods that combine
predictions from multiple forecasting models to produce more robust and accurate forecasts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from src.global_forecasting.base_forecaster import BaseForecaster
from src.utils.math_utils import calculate_confidence_interval

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAverageEnsemble(BaseForecaster):
    """
    Simple Average Ensemble Forecaster.

    This forecaster takes the simple average of predictions from multiple
    forecasting models to produce a final forecast. This is a basic form of
    ensemble that can help reduce prediction error and variance.
    """

    def _initialize_parameters(self) -> None:
        """Initialize ensemble parameters from configuration"""
        # List of forecasters to use in the ensemble
        self.forecasters = []

        # Method to use for averaging: 'mean', 'median', or 'trimmed_mean'
        self.average_method = self.config.get('average_method', 'mean')

        # Trimming percentage (used only if average_method is 'trimmed_mean')
        self.trim_percentage = self.config.get('trim_percentage', 10)

        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)

        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)

        # Store historical data
        self.history = None

        # Store individual forecasts
        self.individual_forecasts = {}

        # Add base forecasters from config if provided
        if self.config and 'base_forecasters' in self.config:
            base_forecasters = self.config.get('base_forecasters', [])
            if isinstance(base_forecasters, list) and base_forecasters:
                for forecaster_config in base_forecasters:
                    if 'name' in forecaster_config and 'params' in forecaster_config:
                        try:
                            from src.global_forecasting import create_forecaster
                            forecaster = create_forecaster(
                                forecaster_config['name'],
                                forecaster_config['params']
                            )
                            self.forecasters.append(forecaster)
                            logger.info(f"Added {forecaster.name} to ensemble from config")
                        except Exception as e:
                            logger.error(f"Error adding forecaster from config: {str(e)}")
    
    def add_forecaster(self, forecaster: BaseForecaster) -> 'SimpleAverageEnsemble':
        """
        Add a forecaster to the ensemble.
        
        Args:
            forecaster: A fitted forecaster to add to the ensemble
            
        Returns:
            Self for method chaining
        """
        if not isinstance(forecaster, BaseForecaster):
            raise TypeError("forecaster must be an instance of BaseForecaster")
        
        # Make sure the forecaster has been fitted
        if not forecaster.fitted:
            raise ValueError("forecaster must be fitted before adding to the ensemble")
            
        # Add the forecaster to the list
        self.forecasters.append(forecaster)
        logger.info(f"Added {forecaster.name} to ensemble")
        
        return self
    
    def clear_forecasters(self) -> None:
        """Remove all forecasters from the ensemble"""
        self.forecasters = []
        self.fitted = False
        logger.info("Cleared all forecasters from ensemble")
    
    def _validate_forecasters(self) -> None:
        """
        Validate that forecasters are properly configured
        
        Raises:
            ValueError: If no forecasters are available or if they are invalid
        """
        if not self.forecasters:
            raise ValueError(
                "No forecasters available in ensemble. "
                "Please add forecasters using add_forecaster() method or "
                "configure base_forecasters in the ensemble configuration."
            )
        
        # Validate each forecaster
        invalid_forecasters = []
        for i, forecaster in enumerate(self.forecasters):
            if not isinstance(forecaster, BaseForecaster):
                invalid_forecasters.append((i, type(forecaster).__name__))
            elif not hasattr(forecaster, 'forecast'):
                invalid_forecasters.append((i, f"{forecaster.name} - missing forecast method"))
        
        if invalid_forecasters:
            error_msg = "Invalid forecasters found:\n"
            for idx, issue in invalid_forecasters:
                error_msg += f"  - Forecaster {idx}: {issue}\n"
            raise TypeError(error_msg)
    
    def fit(self, data: pd.DataFrame) -> 'SimpleAverageEnsemble':
        """
        Fit the ensemble using already fitted forecasters.

        Note: This ensemble doesn't fit its own model, it uses already fitted
        forecasters to generate predictions. This method just validates the
        ensemble is ready to forecast.

        Args:
            data: Historical data with columns 'date' and 'value'

        Returns:
            Self for method chaining
        """
        # Validate data
        self._validate_data(data)
        
        # Store historical data for reference
        self.history = data.copy()
        
        # Mark as fitted
        self.fitted = True
        
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast by averaging predictions from all forecasters.

        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)

        Returns:
            DataFrame with columns 'date' and 'value'
        """
        if not self.fitted:
            logger.warning("Ensemble not fitted. Attempting to forecast anyway.")
            # Set fitted to true to continue
            self.fitted = True

        # Validate forecasters before proceeding
        self._validate_forecasters()
        
        # Generate forecasts from all models
        forecasts = []
        forecast_dates = None
        
        # Clear previous individual forecasts
        self.individual_forecasts = {}
        
        for forecaster in self.forecasters:
            try:
                # Generate forecast
                forecast_df = forecaster.forecast(periods, frequency)
                
                # Store forecast
                self.individual_forecasts[forecaster.name] = forecast_df.copy()
                
                # Check dates consistency
                if forecast_dates is None:
                    forecast_dates = forecast_df['date'].tolist()
                else:
                    # Ensure dates match
                    if forecast_df['date'].tolist() != forecast_dates:
                        logger.warning(f"Date mismatch in {forecaster.name} forecast, will interpolate")
                
                # Store forecast values
                forecasts.append(forecast_df['value'].values)
                
            except Exception as e:
                logger.error(f"Error generating forecast from {forecaster.name}: {str(e)}")
                continue
        
        if not forecasts:
            raise RuntimeError("No forecasts were successfully generated")
        
        # Convert to numpy array for easier manipulation
        forecasts_array = np.array(forecasts)
        
        # Calculate ensemble forecast based on average method
        if self.average_method == 'mean':
            ensemble_values = np.mean(forecasts_array, axis=0)
        elif self.average_method == 'median':
            ensemble_values = np.median(forecasts_array, axis=0)
        elif self.average_method == 'trimmed_mean':
            # Trim the top and bottom percentages
            trim_pct = self.trim_percentage / 100
            trimmed_mean = []
            for i in range(forecasts_array.shape[1]):
                values = forecasts_array[:, i]
                n_values = len(values)
                if n_values == 0:
                    trimmed_mean.append(0.0)
                    continue
                
                # Fix: Ensure we don't create empty slices
                trim_count = int(n_values * trim_pct)
                start_idx = max(0, trim_count)
                end_idx = min(n_values, n_values - trim_count)
                
                if start_idx >= end_idx:
                    # If trimming would remove all values, use all values
                    trimmed = values
                else:
                    trimmed = np.sort(values)[start_idx:end_idx]
                
                trimmed_mean.append(np.mean(trimmed) if len(trimmed) > 0 else 0.0)
            ensemble_values = np.array(trimmed_mean)
        else:
            raise ValueError(f"Unknown average method: {self.average_method}")
        
        # Ensure minimum value if requested
        if self.ensure_minimum:
            ensemble_values = np.maximum(ensemble_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': ensemble_values
        })
        
        # Calculate confidence intervals based on individual forecast spread
        std_dev = np.std(forecasts_array, axis=0)
        lower_bound = ensemble_values - 1.96 * std_dev
        upper_bound = ensemble_values + 1.96 * std_dev
        
        # Ensure bounds respect minimum value
        if self.ensure_minimum:
            lower_bound = np.maximum(lower_bound, self.minimum_value)
        
        # Store confidence intervals
        self.confidence_intervals = pd.DataFrame({
            'date': forecast_dates,
            'lower': lower_bound,
            'value': ensemble_values,
            'upper': upper_bound
        })
        
        # Store forecast result and dates
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_dates
        
        return forecast_df
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'average_method': self.average_method,
            'trim_percentage': self.trim_percentage,
            'ensure_minimum': self.ensure_minimum,
            'minimum_value': self.minimum_value,
            'n_forecasters': len(self.forecasters),
            'forecaster_names': [f.name for f in self.forecasters]
        }
    
    def set_params(self, **params) -> 'SimpleAverageEnsemble':
        """Set model parameters"""
        for key, value in params.items():
            if key == 'average_method':
                if value not in ['mean', 'median', 'trimmed_mean']:
                    raise ValueError(f"Invalid average_method: {value}")
                self.average_method = value
            elif key == 'trim_percentage':
                if not 0 <= value <= 50:
                    raise ValueError("trim_percentage must be between 0 and 50")
                self.trim_percentage = value
            elif key == 'ensure_minimum':
                self.ensure_minimum = bool(value)
            elif key == 'minimum_value':
                self.minimum_value = float(value)
        
        return self


class WeightedEnsembleForecaster(BaseForecaster):
    """
    Weighted Ensemble Forecaster.

    This forecaster creates a weighted average of predictions from multiple
    forecasting models, with weights based on model performance or other criteria.
    """

    def _initialize_parameters(self) -> None:
        """Initialize ensemble parameters from configuration"""
        # List of forecasters to use in the ensemble
        self.forecasters = []

        # Dictionary mapping forecaster name to weight
        self.weights = {}

        # Weight calculation method
        # Options: 'equal', 'performance', 'inverse_error', 'custom'
        self.weight_method = self.config.get('weight_method', 'equal')

        # Performance metric to use for weighting (if weight_method is 'performance')
        # Options: 'MAE', 'MAPE', 'RMSE', 'Theil_U'
        self.performance_metric = self.config.get('performance_metric', 'MAPE')

        # Whether weights should be inversely proportional to error
        # (applies when weight_method is 'performance' or 'inverse_error')
        self.inverse_error = self.config.get('inverse_error', True)

        # Custom weights (used only if weight_method is 'custom')
        self.custom_weights = self.config.get('custom_weights', {})

        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)

        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)

        # Store historical data
        self.history = None

        # Store validation data
        self.validation_data = None

        # Store individual forecasts
        self.individual_forecasts = {}

        # Add base forecasters from config if provided
        if self.config and 'base_forecasters' in self.config:
            base_forecasters = self.config.get('base_forecasters', [])
            if isinstance(base_forecasters, list) and base_forecasters:
                for forecaster_config in base_forecasters:
                    if 'name' in forecaster_config and 'params' in forecaster_config:
                        try:
                            from src.global_forecasting import create_forecaster
                            forecaster = create_forecaster(
                                forecaster_config['name'],
                                forecaster_config['params']
                            )
                            # Add with custom weight if provided
                            weight = forecaster_config.get('weight', None)
                            self.add_forecaster(forecaster, weight)
                            logger.info(f"Added {forecaster.name} to ensemble from config with weight {weight}")
                        except Exception as e:
                            logger.error(f"Error adding forecaster from config: {str(e)}")

                # Mark as fitted if any forecasters were added
                if self.forecasters:
                    self.fitted = True
    
    def add_forecaster(self, forecaster: BaseForecaster, weight: Optional[float] = None) -> 'WeightedEnsembleForecaster':
        """
        Add a forecaster to the ensemble.
        
        Args:
            forecaster: A fitted forecaster to add to the ensemble
            weight: Optional custom weight for this forecaster
                (only used if weight_method is 'custom')
            
        Returns:
            Self for method chaining
        """
        if not isinstance(forecaster, BaseForecaster):
            raise TypeError("forecaster must be an instance of BaseForecaster")
        
        # Make sure the forecaster has been fitted
        if not forecaster.fitted:
            raise ValueError("forecaster must be fitted before adding to the ensemble")
            
        # Add the forecaster to the list
        self.forecasters.append(forecaster)
        
        # Store custom weight if provided
        if weight is not None:
            self.custom_weights[forecaster.name] = weight
            
        logger.info(f"Added {forecaster.name} to ensemble")
        
        return self
    
    def clear_forecasters(self) -> None:
        """Remove all forecasters from the ensemble"""
        self.forecasters = []
        self.weights = {}
        self.fitted = False
        logger.info("Cleared all forecasters from ensemble")
    
    def _validate_forecasters(self) -> None:
        """
        Validate that forecasters are properly configured
        
        Raises:
            ValueError: If no forecasters are available or if they are invalid
        """
        if not self.forecasters:
            raise ValueError(
                "No forecasters available in ensemble. "
                "Please add forecasters using add_forecaster() method or "
                "configure base_forecasters in the ensemble configuration."
            )
        
        # Validate each forecaster
        invalid_forecasters = []
        for i, forecaster in enumerate(self.forecasters):
            if not isinstance(forecaster, BaseForecaster):
                invalid_forecasters.append((i, type(forecaster).__name__))
            elif not hasattr(forecaster, 'forecast'):
                invalid_forecasters.append((i, f"{forecaster.name} - missing forecast method"))
        
        if invalid_forecasters:
            error_msg = "Invalid forecasters found:\n"
            for idx, issue in invalid_forecasters:
                error_msg += f"  - Forecaster {idx}: {issue}\n"
            raise TypeError(error_msg)
    
    def set_validation_data(self, validation_data: pd.DataFrame) -> 'WeightedEnsembleForecaster':
        """
        Set validation data for performance-based weighting.
        
        Args:
            validation_data: DataFrame with 'date' and 'value' columns
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in validation_data.columns or 'value' not in validation_data.columns:
            raise ValueError("Validation data must contain 'date' and 'value' columns")
        
        # Convert dates if they're strings
        if validation_data['date'].dtype == 'object':
            validation_data = validation_data.copy()
            validation_data['date'] = pd.to_datetime(validation_data['date'])
        
        # Sort by date
        validation_data = validation_data.sort_values('date')
        
        # Store validation data
        self.validation_data = validation_data.copy()
        
        return self
    
    def calculate_inverse_error_weights(self, errors: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate weights with proper handling of edge cases
        
        Args:
            errors: Dictionary mapping forecaster names to error values
            
        Returns:
            Dictionary mapping forecaster names to normalized weights
        """
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Calculate inverse errors
        inverse_errors = {}
        for name, error in errors.items():
            # Ensure error is positive
            if error <= 0:
                logger.warning(f"Non-positive error {error} for {name}, using epsilon")
                error = epsilon
            
            inverse_errors[name] = 1.0 / (error + epsilon)
        
        # Calculate total for normalization
        total_inverse = sum(inverse_errors.values())
        
        # Normalize weights
        weights = {}
        if total_inverse > 0:
            for name, inv_error in inverse_errors.items():
                weights[name] = inv_error / total_inverse
        else:
            # Fallback to equal weights
            logger.warning("Total inverse error is zero, using equal weights")
            n_forecasters = len(errors)
            for name in errors:
                weights[name] = 1.0 / n_forecasters if n_forecasters > 0 else 0.0
        
        return weights
    
    def _calculate_weights(self) -> None:
        """
        Calculate weights for each forecaster based on the selected method.
        """
        # Clear any existing weights
        self.weights = {}
        
        # Equal weighting
        if self.weight_method == 'equal':
            weight = 1.0 / len(self.forecasters)
            for forecaster in self.forecasters:
                self.weights[forecaster.name] = weight
                
        # Custom weighting
        elif self.weight_method == 'custom':
            total_weight = sum(self.custom_weights.values())
            
            # Normalize weights to sum to 1
            if total_weight > 0:
                for forecaster in self.forecasters:
                    if forecaster.name in self.custom_weights:
                        self.weights[forecaster.name] = self.custom_weights[forecaster.name] / total_weight
                    else:
                        # Use default weight for forecasters without custom weights
                        self.weights[forecaster.name] = 0.0
                        logger.warning(f"No custom weight for {forecaster.name}, using 0.0")
            else:
                # Fall back to equal weights if total is zero
                logger.warning("Total custom weight is zero, falling back to equal weights")
                weight = 1.0 / len(self.forecasters)
                for forecaster in self.forecasters:
                    self.weights[forecaster.name] = weight
        
        # Performance-based weighting
        elif self.weight_method in ['performance', 'inverse_error']:
            # Validate that we have validation data
            if self.validation_data is None:
                logger.warning("No validation data available, falling back to equal weights")
                weight = 1.0 / len(self.forecasters)
                for forecaster in self.forecasters:
                    self.weights[forecaster.name] = weight
                return
            
            # Calculate performance metrics for each forecaster
            metrics = {}
            
            for forecaster in self.forecasters:
                # Evaluate performance on validation data
                perf_metrics = forecaster.evaluate(self.validation_data)
                
                # Get the selected metric
                if self.performance_metric in perf_metrics:
                    metric_value = perf_metrics[self.performance_metric]
                    metrics[forecaster.name] = metric_value
                else:
                    # Fall back to MAPE if selected metric not available
                    logger.warning(f"{self.performance_metric} not available for {forecaster.name}, using MAPE")
                    metrics[forecaster.name] = perf_metrics.get('MAPE', 100.0)  # Use large but finite default
            
            # Calculate weights based on metrics
            if self.inverse_error:
                # Use the safe inverse error calculation
                self.weights = self.calculate_inverse_error_weights(metrics)
            else:
                # For non-error metrics (higher is better)
                total = sum(metrics.values())
                if total > 0:
                    for name, value in metrics.items():
                        self.weights[name] = value / total
                else:
                    # Fall back to equal weights
                    weight = 1.0 / len(self.forecasters)
                    for forecaster in self.forecasters:
                        self.weights[forecaster.name] = weight
        
        else:
            raise ValueError(f"Unknown weight method: {self.weight_method}")
        
        # Log calculated weights
        logger.info(f"Calculated weights: {self.weights}")
    
    def fit(self, data: pd.DataFrame) -> 'WeightedEnsembleForecaster':
        """
        Fit the ensemble and calculate weights.

        Args:
            data: Historical data with columns 'date' and 'value'

        Returns:
            Self for method chaining
        """
        # Validate data
        self._validate_data(data)
        
        # Store historical data for reference
        self.history = data.copy()
        
        # Calculate weights
        self._calculate_weights()
        
        # Mark as fitted
        self.fitted = True
        
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate weighted forecast from all forecasters.

        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)

        Returns:
            DataFrame with columns 'date' and 'value'
        """
        if not self.fitted:
            logger.warning("Ensemble not fitted. Attempting to forecast anyway.")
            # Calculate weights if not already done
            if not self.weights:
                self._calculate_weights()
            self.fitted = True

        # Validate forecasters before proceeding
        self._validate_forecasters()
        
        # Generate forecasts from all models
        forecasts = {}
        forecast_dates = None
        
        # Clear previous individual forecasts
        self.individual_forecasts = {}
        
        for forecaster in self.forecasters:
            try:
                # Generate forecast
                forecast_df = forecaster.forecast(periods, frequency)
                
                # Store forecast
                self.individual_forecasts[forecaster.name] = forecast_df.copy()
                
                # Check dates consistency
                if forecast_dates is None:
                    forecast_dates = forecast_df['date'].tolist()
                else:
                    # Ensure dates match
                    if forecast_df['date'].tolist() != forecast_dates:
                        logger.warning(f"Date mismatch in {forecaster.name} forecast, will interpolate")
                
                # Store forecast values
                forecasts[forecaster.name] = forecast_df['value'].values
                
            except Exception as e:
                logger.error(f"Error generating forecast from {forecaster.name}: {str(e)}")
                continue
        
        if not forecasts:
            raise RuntimeError("No forecasts were successfully generated")
        
        # Validate weights and calculate weighted average
        total_weight = sum(self.weights.get(name, 0.0) for name in forecasts.keys())
        
        if total_weight == 0:
            logger.warning("No valid weights found, using equal weights")
            # Use equal weights as fallback
            equal_weight = 1.0 / len(forecasts)
            for name in forecasts.keys():
                self.weights[name] = equal_weight
            total_weight = 1.0
        
        weighted_values = np.zeros(periods)
        
        for name, values in forecasts.items():
            weight = self.weights.get(name, 0.0)
            if weight > 0:  # Only apply non-zero weights
                weighted_values += weight * values
        
        # Ensure minimum value if requested
        if self.ensure_minimum:
            weighted_values = np.maximum(weighted_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': weighted_values
        })
        
        # Calculate confidence intervals based on weighted spread
        # Fix: Proper weighted variance calculation
        variance = np.zeros(periods)
        mean_forecast = weighted_values
        # Use the already calculated and normalized total_weight
        
        if total_weight > 0:
            for name, values in forecasts.items():
                weight = self.weights.get(name, 0.0)
                # Proper weighted variance: weight * (value - mean)^2
                variance += weight * (values - mean_forecast) ** 2
            
            # Normalize by total weight for proper weighted variance
            variance = variance / total_weight
        
        # Base standard deviation (from model disagreement)
        base_std = np.sqrt(np.mean(variance))  # Average std across all periods
        
        # Apply growing confidence intervals using sqrt(time) method
        lower_bounds = []
        upper_bounds = []
        
        for i in range(periods):
            periods_ahead = i + 1
            lower, upper = calculate_confidence_interval(
                mean_forecast[i],
                periods_ahead,
                base_std,
                confidence_level=0.95,
                method='sqrt_time'
            )
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        lower_bound = np.array(lower_bounds)
        upper_bound = np.array(upper_bounds)
        
        # Ensure bounds respect minimum value
        if self.ensure_minimum:
            lower_bound = np.maximum(lower_bound, self.minimum_value)
        
        # Store confidence intervals
        self.confidence_intervals = pd.DataFrame({
            'date': forecast_dates,
            'lower': lower_bound,
            'value': weighted_values,
            'upper': upper_bound
        })
        
        # Store forecast result and dates
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_dates
        
        return forecast_df
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'weight_method': self.weight_method,
            'performance_metric': self.performance_metric,
            'inverse_error': self.inverse_error,
            'ensure_minimum': self.ensure_minimum,
            'minimum_value': self.minimum_value,
            'n_forecasters': len(self.forecasters),
            'forecaster_names': [f.name for f in self.forecasters],
            'weights': self.weights
        }
    
    def set_params(self, **params) -> 'WeightedEnsembleForecaster':
        """Set model parameters"""
        for key, value in params.items():
            if key == 'weight_method':
                if value not in ['equal', 'performance', 'inverse_error', 'custom']:
                    raise ValueError(f"Invalid weight_method: {value}")
                self.weight_method = value
            elif key == 'performance_metric':
                if value not in ['MAE', 'MAPE', 'RMSE', 'Theil_U']:
                    raise ValueError(f"Invalid performance_metric: {value}")
                self.performance_metric = value
            elif key == 'inverse_error':
                self.inverse_error = bool(value)
            elif key == 'ensure_minimum':
                self.ensure_minimum = bool(value)
            elif key == 'minimum_value':
                self.minimum_value = float(value)
            elif key == 'custom_weights':
                self.custom_weights = dict(value)
        
        # Recalculate weights if the method has changed
        if 'weight_method' in params or 'performance_metric' in params or 'custom_weights' in params:
            self._calculate_weights()
        
        return self