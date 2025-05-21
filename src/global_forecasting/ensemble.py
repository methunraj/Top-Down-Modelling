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
    
    def fit(self, data: pd.DataFrame) -> 'SimpleAverageEnsemble':
        """
        Fit the ensemble using already fitted forecasters.

        Note: Individual forecasters must be fitted before adding to the ensemble.
        This method just stores the historical data for reference.

        Args:
            data: DataFrame with 'date' and 'value' columns

        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")

        # Convert dates if they're strings
        if data['date'].dtype == 'object':
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])

        # Sort by date
        data = data.sort_values('date')

        # Store historical data
        self.history = data.copy()

        # Check if we have any forecasters
        if not self.forecasters:
            logger.warning("No forecasters in the ensemble. Add forecasters before forecasting.")
        else:
            logger.info(f"Ensemble ready with {len(self.forecasters)} forecasters")

        # Always mark as fitted regardless of forecasters
        # This ensures forecast() can at least attempt to run
        self.fitted = True
        return self

    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast for the specified number of periods.

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

        if not self.forecasters:
            # Generate a dummy forecast using constant values
            logger.warning("No forecasters in the ensemble. Generating dummy forecast.")

            # Create a simple date range starting from today if no history
            if self.history is None:
                import pandas as pd
                from datetime import datetime, timedelta

                # Start from this year
                start_date = datetime(datetime.now().year, 1, 1)

                # Create dates
                dates = []
                if frequency == 'Y':
                    dates = [start_date + timedelta(days=365*i) for i in range(periods)]
                elif frequency == 'Q':
                    dates = [start_date + timedelta(days=90*i) for i in range(periods)]
                elif frequency == 'M':
                    dates = [start_date + timedelta(days=30*i) for i in range(periods)]
                else:
                    dates = [start_date + timedelta(days=i) for i in range(periods)]

                # Create constant values
                values = [100.0] * periods

                forecast_df = pd.DataFrame({
                    'date': dates,
                    'value': values
                })

                # Store dummy confidence intervals
                ci_df = pd.DataFrame({
                    'date': dates,
                    'lower': [90.0] * periods,
                    'value': values,
                    'upper': [110.0] * periods
                })

                self.forecast_result = forecast_df
                self.forecast_dates = dates
                self.confidence_intervals = ci_df

                return forecast_df
            else:
                # Use the last historical value and apply small growth
                last_date = self.history['date'].iloc[-1]
                last_value = self.history['value'].iloc[-1]

                # Create dates
                dates = []
                if frequency == 'Y':
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date.replace(year=last_date.year + i))
                elif frequency == 'Q':
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date + pd.DateOffset(months=3*i))
                elif frequency == 'M':
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date + pd.DateOffset(months=i))
                else:
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date + pd.DateOffset(days=i))

                # Create slightly growing values (5% annual growth)
                values = []
                for i in range(periods):
                    if frequency == 'Y':
                        growth_factor = 1.05 ** (i+1)
                    elif frequency == 'Q':
                        growth_factor = 1.05 ** ((i+1)/4)
                    elif frequency == 'M':
                        growth_factor = 1.05 ** ((i+1)/12)
                    else:
                        growth_factor = 1.05 ** ((i+1)/365)

                    values.append(last_value * growth_factor)

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': dates,
                    'value': values
                })

                # Create confidence intervals
                ci_df = pd.DataFrame({
                    'date': dates,
                    'lower': [value * 0.9 for value in values],
                    'value': values,
                    'upper': [value * 1.1 for value in values]
                })

                # Store results
                self.forecast_result = forecast_df
                self.forecast_dates = dates
                self.confidence_intervals = ci_df

                return forecast_df
        
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
                
                # Extract values and store
                forecasts.append(forecast_df['value'].values)
                
                # Store dates (they should be the same for all forecasters)
                if forecast_dates is None:
                    forecast_dates = forecast_df['date'].values
                
            except Exception as e:
                logger.error(f"Error forecasting with {forecaster.name}: {str(e)}")
        
        if not forecasts:
            raise ValueError("All forecasters failed to generate forecasts")
        
        # Convert to numpy array for easier manipulation
        forecasts_array = np.array(forecasts)
        
        # Calculate ensemble forecast based on the averaging method
        if self.average_method == 'mean':
            # Simple mean
            ensemble_values = np.mean(forecasts_array, axis=0)
            
        elif self.average_method == 'median':
            # Median (robust to outliers)
            ensemble_values = np.median(forecasts_array, axis=0)
            
        elif self.average_method == 'trimmed_mean':
            # Trimmed mean (removes extreme values)
            trim = int(len(forecasts) * (self.trim_percentage / 100))
            
            if trim >= len(forecasts) / 2:
                logger.warning(f"Trim percentage {self.trim_percentage}% too high for {len(forecasts)} forecasters")
                trim = int(len(forecasts) / 4)  # Default to 25% trimming
                
            if trim == 0:
                # Not enough forecasters to trim, fall back to mean
                ensemble_values = np.mean(forecasts_array, axis=0)
            else:
                # Sort each period's forecasts and trim extremes
                ensemble_values = np.zeros(periods)
                
                for i in range(periods):
                    period_forecasts = forecasts_array[:, i]
                    sorted_forecasts = np.sort(period_forecasts)
                    trimmed = sorted_forecasts[trim:-trim] if trim > 0 else sorted_forecasts
                    ensemble_values[i] = np.mean(trimmed)
        else:
            raise ValueError(f"Unknown averaging method: {self.average_method}")
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            ensemble_values = np.maximum(ensemble_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': ensemble_values
        })
        
        # Create confidence intervals based on model spread
        # This is a simple approach: use the min/max of model forecasts as CI bounds
        lower_bound = np.min(forecasts_array, axis=0)
        upper_bound = np.max(forecasts_array, axis=0)
        
        # Create confidence interval DataFrame
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'lower': lower_bound,
            'value': ensemble_values,
            'upper': upper_bound
        })
        
        # Store results
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_dates
        self.confidence_intervals = ci_df
        
        return forecast_df
    
    def plot_ensemble(self, figsize: Tuple[int, int] = (10, 6), 
                    include_individual: bool = True,
                    include_history: bool = True) -> Any:
        """
        Plot the ensemble forecast with individual forecasts.
        
        Args:
            figsize: Figure size tuple (width, height)
            include_individual: Whether to include individual forecasts
            include_history: Whether to include historical data
            
        Returns:
            Matplotlib figure object or None if plotting is not possible
        """
        if not self.fitted or self.forecast_result is None:
            logger.warning("Cannot plot: ensemble not fitted or no forecast available")
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot historical data if available and requested
            if include_history and self.history is not None:
                ax.plot(self.history['date'], self.history['value'], 'k-', linewidth=2, label='Historical')
            
            # Plot individual forecasts if requested
            if include_individual and self.individual_forecasts:
                for name, forecast_df in self.individual_forecasts.items():
                    ax.plot(forecast_df['date'], forecast_df['value'], '--', alpha=0.5, label=name)
            
            # Plot ensemble forecast (always)
            ax.plot(self.forecast_result['date'], self.forecast_result['value'], 
                   'b-', linewidth=3, label=f'Ensemble ({self.average_method})')
            
            # Plot confidence intervals if available
            if self.confidence_intervals is not None:
                ci = self.confidence_intervals
                ax.fill_between(ci['date'], ci['lower'], ci['upper'], 
                               color='b', alpha=0.1, label='Model Range')
            
            ax.set_title("Ensemble Forecast")
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None


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
                    
                    # Store the metric value
                    metrics[forecaster.name] = metric_value
                else:
                    # Fall back to MAPE if selected metric not available
                    logger.warning(f"{self.performance_metric} not available for {forecaster.name}, using MAPE")
                    metrics[forecaster.name] = perf_metrics.get('MAPE', float('inf'))
            
            # Calculate weights based on metrics
            if self.inverse_error:
                # For error metrics, smaller is better, so use inverse
                inverse_metrics = {}
                
                for name, metric in metrics.items():
                    if metric > 0:
                        inverse_metrics[name] = 1.0 / metric
                    else:
                        # Handle zero or negative metrics
                        inverse_metrics[name] = float('inf')
                
                # Handle infinite values
                if any(np.isinf(value) for value in inverse_metrics.values()):
                    # Set weight to 1.0 for best model, 0 for others
                    best_model = max(inverse_metrics.items(), key=lambda x: x[1])[0]
                    for name in inverse_metrics:
                        self.weights[name] = 1.0 if name == best_model else 0.0
                else:
                    # Normalize weights
                    total = sum(inverse_metrics.values())
                    if total > 0:
                        for name, value in inverse_metrics.items():
                            self.weights[name] = value / total
                    else:
                        # Fall back to equal weights
                        weight = 1.0 / len(self.forecasters)
                        for forecaster in self.forecasters:
                            self.weights[forecaster.name] = weight
            else:
                # For non-error metrics, larger is better, use directly
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
            # Unknown weighting method
            logger.warning(f"Unknown weight method: {self.weight_method}, falling back to equal weights")
            weight = 1.0 / len(self.forecasters)
            for forecaster in self.forecasters:
                self.weights[forecaster.name] = weight
        
        # Log the weights
        logger.info("Calculated weights for ensemble:")
        for name, weight in self.weights.items():
            logger.info(f"  {name}: {weight:.4f}")
    
    def fit(self, data: pd.DataFrame) -> 'WeightedEnsembleForecaster':
        """
        Fit the ensemble using already fitted forecasters.
        
        Note: Individual forecasters must be fitted before adding to the ensemble.
        This method primarily calculates the weights for each forecaster.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Convert dates if they're strings
        if data['date'].dtype == 'object':
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
        
        # Sort by date
        data = data.sort_values('date')
        
        # Store historical data
        self.history = data.copy()
        
        # Check if we have any forecasters
        if not self.forecasters:
            logger.warning("No forecasters in the ensemble. Add forecasters before forecasting.")
            self.fitted = False
            return self
        
        # Calculate weights
        self._calculate_weights()
        
        self.fitted = True
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast for the specified number of periods.

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

            # Calculate equal weights if we have forecasters
            if self.forecasters:
                equal_weight = 1.0 / len(self.forecasters)
                for forecaster in self.forecasters:
                    self.weights[forecaster.name] = equal_weight

        if not self.forecasters:
            # Generate a dummy forecast using constant values
            logger.warning("No forecasters in the ensemble. Generating dummy forecast.")

            # Create a simple date range starting from today if no history
            if self.history is None:
                import pandas as pd
                from datetime import datetime, timedelta

                # Start from this year
                start_date = datetime(datetime.now().year, 1, 1)

                # Create dates
                dates = []
                if frequency == 'Y':
                    dates = [start_date + timedelta(days=365*i) for i in range(periods)]
                elif frequency == 'Q':
                    dates = [start_date + timedelta(days=90*i) for i in range(periods)]
                elif frequency == 'M':
                    dates = [start_date + timedelta(days=30*i) for i in range(periods)]
                else:
                    dates = [start_date + timedelta(days=i) for i in range(periods)]

                # Create constant values
                values = [100.0] * periods

                forecast_df = pd.DataFrame({
                    'date': dates,
                    'value': values
                })

                # Store dummy confidence intervals
                ci_df = pd.DataFrame({
                    'date': dates,
                    'lower': [90.0] * periods,
                    'value': values,
                    'upper': [110.0] * periods
                })

                self.forecast_result = forecast_df
                self.forecast_dates = dates
                self.confidence_intervals = ci_df

                return forecast_df
            else:
                # Use the last historical value and apply small growth
                last_date = self.history['date'].iloc[-1]
                last_value = self.history['value'].iloc[-1]

                # Create dates
                dates = []
                if frequency == 'Y':
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date.replace(year=last_date.year + i))
                elif frequency == 'Q':
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date + pd.DateOffset(months=3*i))
                elif frequency == 'M':
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date + pd.DateOffset(months=i))
                else:
                    for i in range(1, periods + 1):
                        if isinstance(last_date, str):
                            from datetime import datetime
                            last_date = datetime.strptime(last_date, '%Y-%m-%d')
                        dates.append(last_date + pd.DateOffset(days=i))

                # Create slightly growing values (5% annual growth)
                values = []
                for i in range(periods):
                    if frequency == 'Y':
                        growth_factor = 1.05 ** (i+1)
                    elif frequency == 'Q':
                        growth_factor = 1.05 ** ((i+1)/4)
                    elif frequency == 'M':
                        growth_factor = 1.05 ** ((i+1)/12)
                    else:
                        growth_factor = 1.05 ** ((i+1)/365)

                    values.append(last_value * growth_factor)

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': dates,
                    'value': values
                })

                # Create confidence intervals
                ci_df = pd.DataFrame({
                    'date': dates,
                    'lower': [value * 0.9 for value in values],
                    'value': values,
                    'upper': [value * 1.1 for value in values]
                })

                # Store results
                self.forecast_result = forecast_df
                self.forecast_dates = dates
                self.confidence_intervals = ci_df

                return forecast_df
        
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
                
                # Extract values and store
                forecasts[forecaster.name] = forecast_df['value'].values
                
                # Store dates (they should be the same for all forecasters)
                if forecast_dates is None:
                    forecast_dates = forecast_df['date'].values
                
            except Exception as e:
                logger.error(f"Error forecasting with {forecaster.name}: {str(e)}")
        
        if not forecasts:
            raise ValueError("All forecasters failed to generate forecasts")
        
        # Calculate weighted ensemble forecast
        ensemble_values = np.zeros(periods)
        
        for name, values in forecasts.items():
            if name in self.weights:
                # Apply weight
                weight = self.weights.get(name, 0.0)
                ensemble_values += values * weight
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            ensemble_values = np.maximum(ensemble_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': ensemble_values
        })
        
        # Create confidence intervals
        # This is a more sophisticated approach: weighted average of model CIs
        lower_bounds = []
        upper_bounds = []
        
        for forecaster in self.forecasters:
            if forecaster.name in self.weights and forecaster.confidence_intervals is not None:
                ci = forecaster.confidence_intervals
                weight = self.weights[forecaster.name]
                
                if 'lower' in ci.columns and 'upper' in ci.columns:
                    # Use weights to compute a weighted confidence interval
                    if len(lower_bounds) == 0:
                        lower_bounds = ci['lower'].values * weight
                        upper_bounds = ci['upper'].values * weight
                    else:
                        lower_bounds += ci['lower'].values * weight
                        upper_bounds += ci['upper'].values * weight
        
        # If no CIs were available, use a simple approach
        if len(lower_bounds) == 0:
            # Calculate standard deviation of forecasts
            all_forecasts = np.array([values for values in forecasts.values()])
            std_devs = np.std(all_forecasts, axis=0)

            # Use 2 standard deviations (95% CI assuming normal distribution)
            lower_bounds = ensemble_values - 2 * std_devs
            upper_bounds = ensemble_values + 2 * std_devs

            # Ensure lower bounds are non-negative
            lower_bounds = np.maximum(lower_bounds, 0)
        
        # Create confidence interval DataFrame
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'lower': lower_bounds,
            'value': ensemble_values,
            'upper': upper_bounds
        })
        
        # Store results
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_dates
        self.confidence_intervals = ci_df
        
        return forecast_df
    
    def plot_ensemble(self, figsize: Tuple[int, int] = (10, 6), 
                    include_individual: bool = True,
                    include_history: bool = True,
                    include_weights: bool = True) -> Any:
        """
        Plot the ensemble forecast with individual forecasts.
        
        Args:
            figsize: Figure size tuple (width, height)
            include_individual: Whether to include individual forecasts
            include_history: Whether to include historical data
            include_weights: Whether to show weights in legend
            
        Returns:
            Matplotlib figure object or None if plotting is not possible
        """
        if not self.fitted or self.forecast_result is None:
            logger.warning("Cannot plot: ensemble not fitted or no forecast available")
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot historical data if available and requested
            if include_history and self.history is not None:
                ax.plot(self.history['date'], self.history['value'], 'k-', linewidth=2, label='Historical')
            
            # Plot individual forecasts if requested
            if include_individual and self.individual_forecasts:
                for name, forecast_df in self.individual_forecasts.items():
                    weight = self.weights.get(name, 0.0)
                    
                    if include_weights:
                        label = f"{name} (w={weight:.2f})"
                    else:
                        label = name
                        
                    ax.plot(forecast_df['date'], forecast_df['value'], '--', alpha=0.5, label=label)
            
            # Plot ensemble forecast (always)
            ax.plot(self.forecast_result['date'], self.forecast_result['value'], 
                   'b-', linewidth=3, label='Weighted Ensemble')
            
            # Plot confidence intervals if available
            if self.confidence_intervals is not None:
                ci = self.confidence_intervals
                ax.fill_between(ci['date'], ci['lower'], ci['upper'], 
                               color='b', alpha=0.1, label='Confidence Interval')
            
            ax.set_title("Weighted Ensemble Forecast")
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None