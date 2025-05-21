"""
Statistical Forecasting Module - Traditional forecasting methods

This module provides implementations of common statistical forecasting methods,
including CAGR, Moving Average, Exponential Smoothing, and ARIMA/SARIMA.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA

from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CAGRForecaster(BaseForecaster):
    """
    Compound Annual Growth Rate (CAGR) forecasting method.
    
    This method calculates the CAGR from historical data and
    applies it to generate future forecasts. CAGR represents
    the mean annual growth rate over a specified time period.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize CAGR-specific parameters from configuration"""
        # Period to calculate CAGR (years)
        self.cagr_period = self.config.get('cagr_period', None)  # None means use all available data
        
        # Fixed CAGR value (if provided)
        self.fixed_cagr = self.config.get('fixed_cagr', None)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store calculated CAGR
        self.cagr = None
        
        # Historical data
        self.history = None
    
    def fit(self, data: pd.DataFrame) -> 'CAGRForecaster':
        """
        Fit the CAGR model to historical data.
        
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
        
        # If fixed CAGR is provided, use it
        if self.fixed_cagr is not None:
            self.cagr = self.fixed_cagr
            logger.info(f"Using fixed CAGR: {self.cagr:.2f}%")
            self.fitted = True
            return self
        
        # Calculate CAGR from data
        if len(data) < 2:
            raise ValueError("At least two data points required to calculate CAGR")
        
        # Determine period for CAGR calculation
        if self.cagr_period is not None:
            # Use the last N years of data
            cutoff_date = data['date'].iloc[-1] - relativedelta(years=self.cagr_period)
            period_data = data[data['date'] >= cutoff_date]
            
            if len(period_data) < 2:
                logger.warning(f"Not enough data for {self.cagr_period}-year CAGR, using all data")
                period_data = data
        else:
            # Use all data
            period_data = data
        
        # Calculate CAGR
        start_value = period_data['value'].iloc[0]
        end_value = period_data['value'].iloc[-1]
        
        # Calculate years between first and last data point
        start_date = pd.to_datetime(period_data['date'].iloc[0])
        end_date = pd.to_datetime(period_data['date'].iloc[-1])
        years = (end_date - start_date).days / 365.25
        
        # Ensure years is not zero
        if years < 0.1:  # Less than ~36 days
            raise ValueError("Time period too short to calculate meaningful CAGR")
        
        # Calculate CAGR
        if start_value > 0:
            self.cagr = (((end_value / start_value) ** (1 / years)) - 1) * 100
        else:
            logger.warning("Starting value is zero or negative, cannot calculate CAGR")
            self.cagr = 0
        
        logger.info(f"Calculated CAGR: {self.cagr:.2f}%")
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
            raise ValueError("Model must be fitted before forecasting")
        
        if self.history is None or len(self.history) == 0:
            raise ValueError("No historical data available")
        
        # Get last date and value from historical data
        last_date = self.history['date'].iloc[-1]
        last_value = self.history['value'].iloc[-1]
        
        # Generate dates for forecast
        forecast_dates = []
        if frequency == 'Y':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(years=i))
        elif frequency == 'Q':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(months=3*i))
        elif frequency == 'M':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(months=i))
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Calculate growth factor
        if frequency == 'Y':
            growth_factor = 1 + (self.cagr / 100)
        elif frequency == 'Q':
            growth_factor = (1 + (self.cagr / 100)) ** 0.25  # Quarterly
        elif frequency == 'M':
            growth_factor = (1 + (self.cagr / 100)) ** (1/12)  # Monthly
        
        # Generate forecasted values
        forecast_values = []
        current_value = last_value
        
        for _ in range(periods):
            current_value = current_value * growth_factor
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                current_value = max(current_value, self.minimum_value)
                
            forecast_values.append(current_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals (simple implementation)
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'lower': [value * 0.9 for value in forecast_values],
            'value': forecast_values,
            'upper': [value * 1.1 for value in forecast_values]
        })
        
        # Store results
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_dates
        self.confidence_intervals = ci_df
        
        return forecast_df
    
    def _predict_for_dates(self, dates: np.ndarray) -> np.ndarray:
        """
        Generate predictions for specific dates.
        
        Args:
            dates: Array of dates to predict for
            
        Returns:
            Array of predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
            
        # Get last date and value from historical data
        last_date = self.history['date'].iloc[-1]
        last_value = self.history['value'].iloc[-1]
        
        # Convert to datetime if necessary
        if isinstance(dates[0], str):
            dates = pd.to_datetime(dates)
        
        # Calculate growth factor
        growth_factor = 1 + (self.cagr / 100)
        
        # Generate forecasted values
        predictions = []
        
        for date in dates:
            # Calculate years since last historical date
            years = (date - last_date).days / 365.25
            
            # Calculate value using CAGR
            if years >= 0:
                value = last_value * (growth_factor ** years)
                
                # Apply minimum value if configured
                if self.ensure_minimum:
                    value = max(value, self.minimum_value)
            else:
                # For dates before last historical, extrapolate backwards
                # (less common but supported for completeness)
                value = last_value / (growth_factor ** abs(years))
                
            predictions.append(value)
        
        return np.array(predictions)


class MovingAverageForecaster(BaseForecaster):
    """
    Moving Average forecasting method.
    
    This method calculates the moving average from historical data
    and uses it to forecast future values. It's good for smoothing
    short-term fluctuations and highlighting longer-term trends.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Moving Average parameters from configuration"""
        # Window size for moving average
        self.window_size = self.config.get('window_size', 3)
        
        # Whether to use weighted moving average
        self.weighted = self.config.get('weighted', False)
        
        # Weights for weighted moving average (if weighted=True)
        self.weights = self.config.get('weights', None)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store historical data
        self.history = None
        
        # Store moving averages
        self.moving_averages = None
    
    def fit(self, data: pd.DataFrame) -> 'MovingAverageForecaster':
        """
        Fit the Moving Average model to historical data.
        
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
        
        # Validate window size
        if self.window_size > len(data):
            logger.warning(f"Window size {self.window_size} is larger than data length {len(data)}")
            self.window_size = len(data)
            logger.warning(f"Adjusted window size to {self.window_size}")
        
        # Calculate moving averages
        values = data['value'].values
        dates = data['date'].values
        
        if self.weighted:
            # Generate weights if not provided
            if self.weights is None:
                # Linear weights increasing from 1 to window_size
                self.weights = list(range(1, self.window_size + 1))
                
            # Ensure weights match window size
            if len(self.weights) != self.window_size:
                raise ValueError(f"Length of weights ({len(self.weights)}) must match window size ({self.window_size})")
                
            # Normalize weights to sum to 1
            weights_sum = sum(self.weights)
            self.weights = [w / weights_sum for w in self.weights]
            
            # Calculate weighted moving averages
            moving_avgs = []
            
            for i in range(len(values) - self.window_size + 1):
                window = values[i:i+self.window_size]
                weighted_avg = sum(w * v for w, v in zip(self.weights, window))
                moving_avgs.append(weighted_avg)
            
            # Pad with NaN for the first window_size-1 values
            padding = [np.nan] * (self.window_size - 1)
            moving_avgs = padding + moving_avgs
        else:
            # Calculate simple moving averages
            moving_avgs = []
            
            for i in range(len(values)):
                if i < self.window_size - 1:
                    # Not enough data for a full window
                    moving_avgs.append(np.nan)
                else:
                    # Calculate average of the window
                    window = values[i-(self.window_size-1):i+1]
                    moving_avgs.append(np.mean(window))
        
        # Store moving averages
        self.moving_averages = pd.DataFrame({
            'date': dates,
            'original': values,
            'moving_avg': moving_avgs
        })
        
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
            raise ValueError("Model must be fitted before forecasting")
        
        if self.history is None or len(self.history) == 0:
            raise ValueError("No historical data available")
        
        # Get last date from historical data
        last_date = self.history['date'].iloc[-1]
        
        # Get the last window_size values for the initial window
        if len(self.history) < self.window_size:
            initial_window = self.history['value'].values
        else:
            initial_window = self.history['value'].values[-self.window_size:]
        
        # Generate dates for forecast
        forecast_dates = []
        if frequency == 'Y':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(years=i))
        elif frequency == 'Q':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(months=3*i))
        elif frequency == 'M':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(months=i))
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Generate forecasted values
        forecast_values = []
        current_window = list(initial_window)
        
        for _ in range(periods):
            if self.weighted:
                # Weighted moving average
                forecast = sum(w * v for w, v in zip(self.weights, current_window))
            else:
                # Simple moving average
                forecast = np.mean(current_window)
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                forecast = max(forecast, self.minimum_value)
                
            forecast_values.append(forecast)
            
            # Update window for next forecast
            current_window.pop(0)
            current_window.append(forecast)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals (simple implementation)
        # For moving average, use standard deviation of the window
        std_dev = np.std(initial_window)
        ci_range = 1.96 * std_dev  # ~95% confidence interval assuming normal distribution
        
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'lower': [max(value - ci_range, 0) for value in forecast_values],
            'value': forecast_values,
            'upper': [value + ci_range for value in forecast_values]
        })
        
        # Store results
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_dates
        self.confidence_intervals = ci_df
        
        return forecast_df


class ExponentialSmoothingForecaster(BaseForecaster):
    """
    Exponential Smoothing forecasting method.
    
    This method applies exponential smoothing to historical data to
    generate forecasts. Multiple variants are supported, including
    Simple, Double (Holt's), and Triple (Holt-Winters) exponential smoothing.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Exponential Smoothing parameters from configuration"""
        # Type of exponential smoothing
        # Options: 'simple', 'double', 'triple'
        self.smoothing_type = self.config.get('smoothing_type', 'simple')
        
        # Alpha parameter (level smoothing factor, 0-1)
        self.alpha = self.config.get('alpha', 0.3)
        
        # Beta parameter (trend smoothing factor, 0-1, for double/triple)
        self.beta = self.config.get('beta', 0.1)
        
        # Gamma parameter (seasonal smoothing factor, 0-1, for triple)
        self.gamma = self.config.get('gamma', 0.1)
        
        # Seasonal period (for triple exponential smoothing)
        self.seasonal_periods = self.config.get('seasonal_periods', 4)
        
        # Trend type (for double/triple)
        # Options: 'additive', 'multiplicative'
        self.trend_type = self.config.get('trend_type', 'additive')
        
        # Seasonal type (for triple)
        # Options: 'additive', 'multiplicative'
        self.seasonal_type = self.config.get('seasonal_type', 'additive')
        
        # Damping factor for trending (0-1, 1 = no damping)
        self.damping = self.config.get('damping', 1.0)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store the fitted model
        self.model = None
        
        # Store historical data
        self.history = None
        
        # Store fitted values and components
        self.level = None
        self.trend = None
        self.seasonal = None
    
    def fit(self, data: pd.DataFrame) -> 'ExponentialSmoothingForecaster':
        """
        Fit Exponential Smoothing model to historical data.
        
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
        
        # Validate seasonal periods for triple smoothing
        if self.smoothing_type == 'triple' and len(data) < 2 * self.seasonal_periods:
            logger.warning(f"Not enough data for triple smoothing with {self.seasonal_periods} periods")
            logger.warning(f"Need at least {2 * self.seasonal_periods} data points, but only have {len(data)}")
            logger.warning("Falling back to double exponential smoothing")
            self.smoothing_type = 'double'
        
        # Get values
        values = data['value'].values
        
        # Fit the appropriate model
        if self.smoothing_type == 'simple':
            # Simple Exponential Smoothing
            self._fit_simple(values)
        elif self.smoothing_type == 'double':
            # Double Exponential Smoothing (Holt's method)
            self._fit_double(values)
        elif self.smoothing_type == 'triple':
            # Triple Exponential Smoothing (Holt-Winters method)
            self._fit_triple(values)
        else:
            raise ValueError(f"Unknown smoothing type: {self.smoothing_type}")
        
        self.fitted = True
        return self
    
    def _fit_simple(self, values: np.ndarray) -> None:
        """
        Fit Simple Exponential Smoothing model.
        
        Args:
            values: Array of historical values
        """
        if len(values) < 2:
            raise ValueError("At least two data points required for Simple Exponential Smoothing")
        
        # Initialize level with first value
        level = values[0]
        levels = [level]
        
        # Apply smoothing
        for i in range(1, len(values)):
            level = self.alpha * values[i] + (1 - self.alpha) * level
            levels.append(level)
        
        # Store the fitted values
        self.level = np.array(levels)
        self.trend = None
        self.seasonal = None
    
    def _fit_double(self, values: np.ndarray) -> None:
        """
        Fit Double Exponential Smoothing model (Holt's method).
        
        Args:
            values: Array of historical values
        """
        if len(values) < 3:
            raise ValueError("At least three data points required for Double Exponential Smoothing")
        
        # Initialize level and trend
        level = values[0]
        if self.trend_type == 'additive':
            trend = values[1] - values[0]
        else:  # multiplicative
            trend = values[1] / values[0] if values[0] != 0 else 1.0
        
        levels = [level]
        trends = [trend]
        
        # Apply smoothing
        for i in range(1, len(values)):
            old_level = level
            
            # Update level
            level = self.alpha * values[i]
            if self.trend_type == 'additive':
                level += (1 - self.alpha) * (old_level + self.damping * trend)
            else:  # multiplicative
                level += (1 - self.alpha) * (old_level * (self.damping * trend))
            
            # Update trend
            if self.trend_type == 'additive':
                trend = self.beta * (level - old_level) + (1 - self.beta) * self.damping * trend
            else:  # multiplicative
                trend = self.beta * (level / old_level) + (1 - self.beta) * self.damping * trend
            
            levels.append(level)
            trends.append(trend)
        
        # Store the fitted values
        self.level = np.array(levels)
        self.trend = np.array(trends)
        self.seasonal = None
    
    def _fit_triple(self, values: np.ndarray) -> None:
        """
        Fit Triple Exponential Smoothing model (Holt-Winters).
        
        Args:
            values: Array of historical values
        """
        if len(values) < 2 * self.seasonal_periods:
            raise ValueError(f"Triple Exponential Smoothing requires at least {2 * self.seasonal_periods} data points")
        
        seasons = self.seasonal_periods
        
        # Initialize seasonal components
        if self.seasonal_type == 'additive':
            # Average first season
            season_avgs = [np.mean(values[i:i+seasons]) for i in range(seasons)]
            
            # Initialize seasonal components
            seasonals = []
            for i in range(seasons):
                seasonal = values[i] - season_avgs[i % seasons]
                seasonals.append(seasonal)
            
            # Normalize to sum to 0
            seasonals = np.array(seasonals)
            seasonals -= np.mean(seasonals)
        else:  # multiplicative
            # Average first season
            season_avgs = [np.mean(values[i:i+seasons]) for i in range(seasons)]
            
            # Initialize seasonal components
            seasonals = []
            for i in range(seasons):
                if season_avgs[i % seasons] != 0:
                    seasonal = values[i] / season_avgs[i % seasons]
                else:
                    seasonal = 1.0
                seasonals.append(seasonal)
            
            # Normalize to average to 1
            seasonals = np.array(seasonals)
            seasonals = seasonals / np.mean(seasonals)
        
        # Extend seasonal components to match data length
        extended_seasonals = []
        for i in range(len(values)):
            extended_seasonals.append(seasonals[i % seasons])
        
        # Initialize level and trend
        if self.seasonal_type == 'additive':
            level = values[0] - extended_seasonals[0]
        else:  # multiplicative
            level = values[0] / extended_seasonals[0] if extended_seasonals[0] != 0 else values[0]
        
        if self.trend_type == 'additive':
            trend = (values[seasons] - values[0]) / seasons
        else:  # multiplicative
            trend = (values[seasons] / values[0]) ** (1 / seasons) if values[0] != 0 else 1.0
        
        levels = [level]
        trends = [trend]
        final_seasonals = [extended_seasonals[0]]
        
        # Apply smoothing
        for i in range(1, len(values)):
            old_level = level
            
            # Update level, trend, and seasonal components
            if self.seasonal_type == 'additive':
                if self.trend_type == 'additive':
                    level = self.alpha * (values[i] - extended_seasonals[i-seasons]) + (1 - self.alpha) * (old_level + self.damping * trend)
                else:  # multiplicative trend
                    level = self.alpha * (values[i] - extended_seasonals[i-seasons]) + (1 - self.alpha) * (old_level * (self.damping * trend))
                
                if self.trend_type == 'additive':
                    trend = self.beta * (level - old_level) + (1 - self.beta) * self.damping * trend
                else:  # multiplicative trend
                    trend = self.beta * (level / old_level) + (1 - self.beta) * self.damping * trend
                
                final_seasonals.append(self.gamma * (values[i] - level) + (1 - self.gamma) * extended_seasonals[i-seasons])
                
            else:  # multiplicative seasonal
                if self.trend_type == 'additive':
                    level = self.alpha * (values[i] / extended_seasonals[i-seasons]) + (1 - self.alpha) * (old_level + self.damping * trend)
                else:  # multiplicative trend
                    level = self.alpha * (values[i] / extended_seasonals[i-seasons]) + (1 - self.alpha) * (old_level * (self.damping * trend))
                
                if self.trend_type == 'additive':
                    trend = self.beta * (level - old_level) + (1 - self.beta) * self.damping * trend
                else:  # multiplicative trend
                    trend = self.beta * (level / old_level) + (1 - self.beta) * self.damping * trend
                
                if level != 0:
                    final_seasonals.append(self.gamma * (values[i] / level) + (1 - self.gamma) * extended_seasonals[i-seasons])
                else:
                    final_seasonals.append(extended_seasonals[i-seasons])
            
            levels.append(level)
            trends.append(trend)
        
        # Store the fitted values
        self.level = np.array(levels)
        self.trend = np.array(trends)
        self.seasonal = np.array(final_seasonals)
    
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
            raise ValueError("Model must be fitted before forecasting")
        
        if self.history is None or len(self.history) == 0:
            raise ValueError("No historical data available")
        
        # Get last date from historical data
        last_date = self.history['date'].iloc[-1]
        
        # Generate dates for forecast
        forecast_dates = []
        if frequency == 'Y':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(years=i))
        elif frequency == 'Q':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(months=3*i))
        elif frequency == 'M':
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + relativedelta(months=i))
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Generate forecasted values
        if self.smoothing_type == 'simple':
            forecast_values = self._forecast_simple(periods)
        elif self.smoothing_type == 'double':
            forecast_values = self._forecast_double(periods)
        elif self.smoothing_type == 'triple':
            forecast_values = self._forecast_triple(periods)
        else:
            raise ValueError(f"Unknown smoothing type: {self.smoothing_type}")
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals (simple implementation)
        # For exponential smoothing, growing uncertainty over time
        history_std = np.std(self.history['value'].values)
        
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        lower_bound = []
        upper_bound = []
        
        for i in range(periods):
            # Increase uncertainty with time
            ci_width = history_std * np.sqrt(1 + i * 0.1)
            lower = max(forecast_values[i] - 1.96 * ci_width, 0)
            upper = forecast_values[i] + 1.96 * ci_width
            
            lower_bound.append(lower)
            upper_bound.append(upper)
        
        ci_df['lower'] = lower_bound
        ci_df['upper'] = upper_bound
        
        # Store results
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_dates
        self.confidence_intervals = ci_df
        
        return forecast_df
    
    def _forecast_simple(self, periods: int) -> np.ndarray:
        """
        Generate forecast for Simple Exponential Smoothing.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Array of forecasted values
        """
        # Last smoothed level becomes the forecast for all future periods
        last_level = self.level[-1]
        return np.full(periods, last_level)
    
    def _forecast_double(self, periods: int) -> np.ndarray:
        """
        Generate forecast for Double Exponential Smoothing.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Array of forecasted values
        """
        last_level = self.level[-1]
        last_trend = self.trend[-1]
        
        forecasts = []
        
        for h in range(1, periods + 1):
            if self.trend_type == 'additive':
                # Additive trend with damping
                damped_trend = sum(self.damping ** j for j in range(1, h + 1)) * last_trend
                forecast = last_level + damped_trend
            else:  # multiplicative
                # Multiplicative trend with damping
                damped_trend = self.damping ** h * last_trend
                forecast = last_level * damped_trend ** h
            
            forecasts.append(forecast)
        
        return np.array(forecasts)


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA (AutoRegressive Integrated Moving Average) forecasting method.

    This method uses the ARIMA model to capture complex time series dynamics,
    including trends, seasonality, and autoregressive patterns.
    """

    def _initialize_parameters(self) -> None:
        """Initialize ARIMA-specific parameters from configuration"""
        # ARIMA order (p, d, q)
        self.order = self.config.get('order', (1, 1, 1))
        
        # Seasonal order (P, D, Q, s)
        self.seasonal_order = self.config.get('seasonal_order', (0, 0, 0, 0))
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store historical data
        self.history = None
        self._last_date = None
        self._data_freq = None  # Inferred frequency of historical data

    def fit(self, data: pd.DataFrame) -> 'ARIMAForecaster':
        """
        Fit the ARIMA model to historical data.

        Args:
            data: DataFrame with 'date' and 'value' columns

        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")

        if data['date'].dtype == 'object':
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])

        data = data.sort_values('date').set_index('date')
        self.history = data['value']
        self._last_date = data.index[-1]

        # Infer frequency if possible
        if len(data.index) > 1:
            self._data_freq = pd.infer_freq(data.index)
            if self._data_freq is None:
                logger.warning("Could not infer frequency from data index. Forecasting dates might be incorrect.")
        else:
            self._data_freq = None # Cannot infer with single point

        # Fit ARIMA model
        # Note: statsmodels ARIMA automatically handles differencing (d)
        try:
            self.model = ARIMA(
                self.history,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False, # Common practice
                enforce_invertibility=False # Common practice
            ).fit()
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            # Fallback for non-stationary data if auto-differencing isn't enough
            # Or if seasonal_order is causing issues with non-seasonal data
            if 'stationary' in str(e).lower() or 'invertibility' in str(e).lower():
                 logger.info("Attempting fallback with simpler ARIMA order (1,1,0)")
                 try:
                     self.model = ARIMA(self.history, order=(1,1,0)).fit()
                 except Exception as e_fallback:
                     logger.error(f"Fallback ARIMA fitting also failed: {e_fallback}")
                     raise e_fallback # Re-raise original error if fallback fails
            elif self.seasonal_order != (0,0,0,0) and 'seasonal' in str(e).lower():
                logger.info("Attempting fallback without seasonal component")
                try:
                    self.model = ARIMA(self.history, order=self.order, seasonal_order=(0,0,0,0)).fit()
                except Exception as e_fallback_seasonal:
                    logger.error(f"Fallback ARIMA (no seasonal) fitting also failed: {e_fallback_seasonal}")
                    raise e_fallback_seasonal
            else:
                raise e


        logger.info(f"ARIMA model fitted with order={self.order} and seasonal_order={self.seasonal_order}")
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
            raise ValueError("Model must be fitted before forecasting")

        # Generate forecast dates
        # Use inferred frequency if available and matches requested frequency prefix
        # Otherwise, use the provided frequency string directly
        
        base_freq_char = frequency[0].upper() # Y, Q, M
        
        if self._data_freq:
            # Attempt to map inferred frequency to pandas offset alias
            # This handles cases like 'YS' (Year Start), 'AS' (Annual Start), etc.
            # and aligns them with 'Y' for relativedelta.
            # 'MS' for Month Start, 'QS' for Quarter Start
            
            # Try to make a more robust mapping for frequency
            if self._data_freq.startswith('A') or self._data_freq.startswith('Y'):
                inferred_base_freq = 'Y'
            elif self._data_freq.startswith('Q'):
                inferred_base_freq = 'Q'
            elif self._data_freq.startswith('M'):
                inferred_base_freq = 'M'
            else: # Daily, weekly, hourly etc. - may not align well with Y/Q/M forecast
                inferred_base_freq = None
                logger.warning(f"Inferred data frequency '{self._data_freq}' may not align well with requested forecast frequency '{frequency}'.")

            if inferred_base_freq and inferred_base_freq == base_freq_char:
                 # Use a DatetimeIndex for date generation if frequencies match
                forecast_dates_index = pd.date_range(
                    start=self._last_date,
                    periods=periods + 1,  # +1 to include start, then drop it
                    freq=self._data_freq  # Use the inferred frequency
                )[1:]
                forecast_dates = list(forecast_dates_index.to_pydatetime())
            else:
                # Fallback to relativedelta if frequencies don't match well
                forecast_dates = []
                current_date = self._last_date
                for _ in range(periods):
                    if frequency == 'Y':
                        current_date += relativedelta(years=1)
                    elif frequency == 'Q':
                        current_date += relativedelta(months=3)
                    elif frequency == 'M':
                        current_date += relativedelta(months=1)
                    else:
                        raise ValueError(f"Unsupported frequency: {frequency}")
                    forecast_dates.append(current_date)
        else: # No inferred frequency, rely on relativedelta
            forecast_dates = []
            current_date = self._last_date
            for _ in range(periods):
                if frequency == 'Y':
                    current_date += relativedelta(years=1)
                elif frequency == 'Q':
                    current_date += relativedelta(months=3)
                elif frequency == 'M':
                    current_date += relativedelta(months=1)
                else:
                    raise ValueError(f"Unsupported frequency: {frequency}")
                forecast_dates.append(current_date)


        # Get forecast and confidence intervals from ARIMA model
        forecast_obj = self.model.get_forecast(steps=periods)
        forecast_values = forecast_obj.predicted_mean.values
        conf_int = forecast_obj.conf_int()

        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
            conf_int_lower = np.maximum(conf_int.iloc[:, 0].values, self.minimum_value)
            conf_int_upper = np.maximum(conf_int.iloc[:, 1].values, self.minimum_value)
        else:
            conf_int_lower = conf_int.iloc[:, 0].values
            conf_int_upper = conf_int.iloc[:, 1].values


        # Create forecast DataFrame
        self.forecast_result = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })

        # Create confidence intervals DataFrame
        self.confidence_intervals = pd.DataFrame({
            'date': forecast_dates,
            'lower': conf_int_lower,
            'value': forecast_values, # Include the mean forecast for easier plotting
            'upper': conf_int_upper
        })
        
        self.forecast_dates = forecast_dates
        
        return self.forecast_result

    def _predict_for_dates(self, dates: np.ndarray) -> np.ndarray:
        """
        Generate predictions for specific dates.
        This requires re-fitting or extending the model up to the latest date in `dates`.
        For simplicity, this basic implementation will only predict future dates
        relative to the end of the training data.
        A more robust version would handle dates within the training period (by returning
        fitted values) or dates far past the forecast horizon (by re-forecasting).

        Args:
            dates: Array of dates to predict for. Must be after self._last_date.

        Returns:
            Array of predictions.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")

        if dates.size == 0:
            return np.array([])

        # Ensure dates are pd.Timestamp for comparison
        dates = pd.to_datetime(dates)
        
        if np.any(dates <= self._last_date):
            # This is a simplification. A full implementation might return:
            # 1. Fitted values for dates within self.history.index
            # 2. Forecasted values for dates beyond self._last_date
            # This would require careful indexing and merging.
            # For now, we raise an error or warn if dates are not in the future.
            logger.warning("Prediction for past or current dates requested. "
                           "This basic _predict_for_dates returns forecasts "
                           "only for future dates relative to training data.")
            # Filter to only future dates for this simplified example
            future_dates_mask = dates > self._last_date
            if not np.any(future_dates_mask):
                 return np.array([]) # No future dates to predict
            dates = dates[future_dates_mask]
            if len(dates) == 0: # Re-check after filtering
                return np.array([])


        # Determine the number of steps needed from the last historical date
        # This requires knowing the frequency of the data.
        # This is a simplified example assuming the dates align with a predictable frequency.
        
        # Calculate steps for each date based on the most common frequency
        # This is tricky if `dates` are not regularly spaced or don't align with `self._data_freq`
        # For this example, let's assume they are future dates and we want to forecast up to the max date
        
        max_pred_date = max(dates)
        
        # Estimate number of periods to forecast to cover all requested dates
        # This part is tricky if the frequency of `dates` is different from `self._data_freq`
        # or if `self._data_freq` is None.
        
        # Simplification: forecast up to the furthest date and then select
        # This isn't the most efficient for sparse/irregular `dates`
        
        if self._data_freq:
            # Use date_range to count periods if frequency is known
            try:
                # Create a range from end of history to the max prediction date
                future_range = pd.date_range(start=self._last_date, end=max_pred_date, freq=self._data_freq)
                steps_needed = len(future_range) -1 # -1 because range includes the start_date
                if steps_needed <= 0: # All dates are before or at _last_date
                    return np.array([self.history.iloc[-1]] * len(dates)) # Or some other logic for past dates
            except ValueError as e: # If freq doesn't work with date_range end
                logger.warning(f"Could not determine steps with self._data_freq {self._data_freq} for prediction: {e}. Using approximation.")
                # Fallback: approximate steps assuming a 'standard' year/month/quarter length
                # This is a rough approximation
                time_delta_days = (max_pred_date - self._last_date).days
                if self._data_freq and self._data_freq[0] in ['A', 'Y']:
                    steps_needed = time_delta_days // 365 + 1
                elif self._data_freq and self._data_freq[0] == 'Q':
                    steps_needed = time_delta_days // 90 + 1
                elif self._data_freq and self._data_freq[0] == 'M':
                    steps_needed = time_delta_days // 30 + 1
                else: # daily or unknown
                    steps_needed = time_delta_days +1
        else: # No data frequency, rough approximation
            time_delta_days = (max_pred_date - self._last_date).days
            # Assuming daily steps if frequency is unknown, this might be very wrong
            # A better approach might be to require frequency for _predict_for_dates
            steps_needed = time_delta_days + 1 
            logger.warning("Data frequency is unknown, _predict_for_dates might be inaccurate.")

        if steps_needed <= 0: # If max_pred_date is not in the future
            # Handle cases where all requested dates are in the past or are the last known date
            # This part needs more robust logic based on how historical lookups should behave.
            # For now, returning last known value for any date <= self._last_date if it's among `dates`
            # This is not ideal. A proper solution would involve looking up fitted values from self.model.fittedvalues
            # or historical values from self.history.
            
            # Temporary placeholder:
            # For dates that are not in the future, we'd ideally return historical or fitted values.
            # This method is primarily for future unseen dates.
            # If all dates are in the past or present, this will result in an empty forecast or error from get_forecast.
            # Let's return the last known value for any date that is not strictly in the future.
            # This is a placeholder for a more robust handling of historical/fitted values.
            
            predictions = []
            for date_val in dates:
                if date_val <= self._last_date:
                    # Attempt to find exact match in history
                    if date_val in self.history.index:
                         predictions.append(self.history.loc[date_val])
                    else: # If no exact match, use the last known value (simplification)
                         predictions.append(self.history.iloc[-1])
                else: 
                    # This case should not be hit if steps_needed <=0 means all dates are past/present
                    # However, to be safe, predict using 1 step if a future date somehow gets here
                    future_pred = self.model.get_forecast(steps=1).predicted_mean.iloc[0]
                    predictions.append(future_pred)
            return np.array(predictions)


        all_forecasts = self.model.get_forecast(steps=steps_needed)
        all_predicted_means = all_forecasts.predicted_mean
        
        # Now, match the forecasted values to the requested `dates`
        # This requires aligning `dates` with the index of `all_predicted_means`
        
        # Create a temporary series from predictions with their dates
        # The index of all_predicted_means starts right after self._last_date with self._data_freq
        
        if not self._data_freq:
            logger.warning("Cannot accurately map predictions to specific dates without a known data frequency for history.")
            # Fallback: return the sequence of predictions, assuming `dates` align with this sequence. Risky.
            if len(all_predicted_means) < len(dates): # Should not happen if steps_needed was calculated correctly
                 return np.array([self.minimum_value] * len(dates)) # Or some other error indicator
            return all_predicted_means.iloc[:len(dates)].values


        pred_index = pd.date_range(start=self._last_date + pd.Timedelta(days=1) if self._data_freq is None or not self._data_freq.startswith('D') else self._last_date + pd.tseries.frequencies.to_offset(self._data_freq), 
                                   periods=steps_needed, freq=self._data_freq)

        # Create a series with the generated forecast index
        forecast_series = pd.Series(all_predicted_means.values, index=pred_index)
        
        # Reindex to match the exact dates requested, filling missing ones
        # Use 'nearest' or 'ffill'/'bfill' if exact matches are not guaranteed
        # For ARIMA, forecasts are for specific steps into the future, so direct mapping is usually what's needed
        # if the `dates` align with the frequency.
        
        # Filter the series to only include requested dates
        # This ensures that if `dates` are sparse, we only return values for them.
        # We need to handle cases where a date in `dates` might not perfectly align with `pred_index`.
        
        # Use reindex with tolerance if dates might be slightly off, or ensure `dates` are aligned.
        # For simplicity here, assume `dates` will align if they are sensible future dates.
        try:
            # Attempt to directly look up, assuming dates align with the forecast frequency
            # This might fail if dates are arbitrary and don't match the frequency.
            predictions = forecast_series.reindex(dates, method=None) # Exact match or NaN
            
            # Handle NaNs if any date didn't match (e.g. if a date is off-frequency)
            # A more robust solution might interpolate or use method='nearest'
            if predictions.isna().any():
                logger.warning(f"Some requested prediction dates did not align with the model's forecast frequency ({self._data_freq}). Using nearest available forecast.")
                # Fallback to nearest if exact match fails
                predictions = forecast_series.reindex(dates, method='nearest', limit=1) # limit=1 to prevent far matches
                # Ensure the nearest is within a reasonable tolerance, e.g., 1 period
                # This part can get complex; for now, nearest is a simplification.


        except KeyError: # Should not happen with reindex, but as a safeguard
            logger.error(f"Could not map all prediction dates. Ensure they are compatible with data frequency {self._data_freq}")
            # Fallback to returning a sequence of minimum values or raise error
            return np.full(len(dates), self.minimum_value) 

        final_predictions = predictions.values
        if self.ensure_minimum:
            final_predictions = np.maximum(final_predictions, self.minimum_value)
            
        return final_predictions
    
    def _forecast_triple(self, periods: int) -> np.ndarray:
        """
        Generate forecast for Triple Exponential Smoothing.
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            Array of forecasted values
        """
        last_level = self.level[-1]
        last_trend = self.trend[-1]
        
        # Get the last full season of seasonal factors
        season_start = len(self.seasonal) - self.seasonal_periods
        season_factors = self.seasonal[max(0, season_start):]
        
        # If we don't have a full season, repeat what we have
        while len(season_factors) < self.seasonal_periods:
            season_factors = np.concatenate([season_factors, season_factors])
        
        # Trim to exactly one season
        season_factors = season_factors[:self.seasonal_periods]
        
        forecasts = []
        
        for h in range(1, periods + 1):
            # Get seasonal factor for this period
            season_idx = (len(self.seasonal) + h - 1) % self.seasonal_periods
            seasonal = season_factors[season_idx]
            
            if self.trend_type == 'additive':
                # Additive trend with damping
                if h > 1:
                    damped_trend = sum(self.damping ** j for j in range(1, h + 1)) * last_trend
                else:
                    damped_trend = self.damping * last_trend
                
                if self.seasonal_type == 'additive':
                    # Additive seasonality
                    forecast = last_level + damped_trend + seasonal
                else:  # multiplicative
                    # Multiplicative seasonality
                    forecast = (last_level + damped_trend) * seasonal
            else:  # multiplicative trend
                # Multiplicative trend with damping
                damped_trend = self.damping ** h * last_trend
                
                if self.seasonal_type == 'additive':
                    # Additive seasonality
                    forecast = last_level * damped_trend ** h + seasonal
                else:  # multiplicative
                    # Multiplicative seasonality
                    forecast = last_level * damped_trend ** h * seasonal
            
            forecasts.append(forecast)
        
        return np.array(forecasts)