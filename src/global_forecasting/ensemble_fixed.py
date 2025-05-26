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
                trimmed = np.sort(values)[int(len(values)*trim_pct):int(len(values)*(1-trim_pct))]
                trimmed_mean.append(np.mean(trimmed))
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