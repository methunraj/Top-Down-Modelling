"""
Statistical Forecasting Module - Traditional forecasting methods

This module provides implementations of common statistical forecasting methods,
including CAGR, Moving Average, Exponential Smoothing, ARIMA, SARIMA, and various
regression models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

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


class ARIMAForecaster(BaseForecaster):
    """
    Auto-Regressive Integrated Moving Average (ARIMA) forecasting method.
    
    This method combines autoregression (AR), differencing for non-stationarity (I),
    and moving average (MA) components to create a powerful and flexible time series
    forecasting model. ARIMA is widely used for financial, economic, and market data
    that exhibits trends but no seasonality.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize ARIMA parameters from configuration"""
        # ARIMA order: (p, d, q)
        # p: AR order (number of lag observations)
        # d: Differencing order (number of times to difference)
        # q: MA order (size of moving average window)
        self.p = self.config.get('p', 1)
        self.d = self.config.get('d', 1)
        self.q = self.config.get('q', 1)
        
        # Auto-determine order using information criteria
        self.auto_order = self.config.get('auto_order', False)
        
        # Maximum values to consider for auto-determination
        self.max_p = self.config.get('max_p', 5)
        self.max_d = self.config.get('max_d', 2)
        self.max_q = self.config.get('max_q', 5)
        
        # Information criterion for model selection (aic or bic)
        self.criterion = self.config.get('criterion', 'aic')
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store fitted model
        self.model = None
        self.fitted_model = None
        self.history = None
        self.time_periods = None
        self.order = None
    
    def _determine_best_order(self, values: np.ndarray) -> Tuple[int, int, int]:
        """
        Determine the best ARIMA order using grid search and information criteria.
        
        Args:
            values: Time series values
            
        Returns:
            Tuple with best (p, d, q) order
        """
        # Check for stationarity and determine d
        # Use Augmented Dickey-Fuller test
        best_d = 0
        adf_result = adfuller(values, autolag='AIC')
        p_value = adf_result[1]
        
        # If p-value > 0.05, series is non-stationary
        if p_value > 0.05:
            # Try first differencing
            diff1 = np.diff(values, n=1)
            adf_result = adfuller(diff1, autolag='AIC')
            p_value = adf_result[1]
            best_d = 1
            
            # If still non-stationary, try second differencing
            if p_value > 0.05 and self.max_d >= 2:
                diff2 = np.diff(values, n=2)
                adf_result = adfuller(diff2, autolag='AIC')
                p_value = adf_result[1]
                best_d = 2 if p_value <= 0.05 else 1
        
        # Determine p and q using ACF and PACF
        diffed_values = np.diff(values, n=best_d) if best_d > 0 else values
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            acf_values = acf(diffed_values, nlags=min(20, len(diffed_values) // 2 - 1))
            pacf_values = pacf(diffed_values, nlags=min(20, len(diffed_values) // 2 - 1))
        
        # Get significant lag for p from PACF
        significant_pacf = np.where(np.abs(pacf_values) > 1.96 / np.sqrt(len(diffed_values)))[0]
        best_p = min(significant_pacf[-1] if len(significant_pacf) > 0 else 0, self.max_p)
        
        # Get significant lag for q from ACF
        significant_acf = np.where(np.abs(acf_values) > 1.96 / np.sqrt(len(diffed_values)))[0]
        best_q = min(significant_acf[-1] if len(significant_acf) > 0 else 0, self.max_q)
        
        # Ensure p and q are at least 1 if they were estimated as 0
        best_p = max(1, best_p)
        best_q = max(1, best_q)
        
        logger.info(f"Determined best ARIMA order: ({best_p}, {best_d}, {best_q})")
        return (best_p, best_d, best_q)
    
    def fit(self, data: pd.DataFrame) -> 'ARIMAForecaster':
        """
        Fit ARIMA model to historical data.
        
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
        
        # Calculate time periods
        self.time_periods = data['date'].values
        
        # Get values
        values = data['value'].values
        
        # Determine ARIMA order
        if self.auto_order:
            try:
                self.order = self._determine_best_order(values)
            except Exception as e:
                logger.warning(f"Error determining best order: {str(e)}")
                logger.warning("Falling back to default order")
                self.order = (self.p, self.d, self.q)
        else:
            self.order = (self.p, self.d, self.q)
        
        # Update instance variables
        self.p, self.d, self.q = self.order
        
        # Create and fit the ARIMA model
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model = ARIMA(values, order=self.order)
                self.fitted_model = self.model.fit()
                
            logger.info(f"Fitted ARIMA model with order {self.order}")
            self.fitted = True
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            logger.warning("Try adjusting p, d, q parameters or enable auto_order")
            raise
        
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
        
        # Generate forecast
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            forecast_result = self.fitted_model.forecast(steps=periods)
        
        # Get standard errors for confidence intervals
        forecast_errors = np.sqrt(self.fitted_model.forecast_variance(periods))
        
        # Create point forecast
        forecast_values = forecast_result
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals
        z_value = 1.96  # 95% confidence interval
        lower_bound = forecast_values - z_value * forecast_errors
        upper_bound = forecast_values + z_value * forecast_errors
        
        if self.ensure_minimum:
            lower_bound = np.maximum(lower_bound, self.minimum_value)
        
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'lower': lower_bound,
            'value': forecast_values,
            'upper': upper_bound
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
        
        # Convert dates to periods relative to last historical date
        last_date = self.history['date'].iloc[-1]
        periods = []
        
        for date in dates:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            # Calculate periods based on frequency
            if (date - last_date).days <= 366:
                # Within a year, use monthly frequency
                periods.append((date.year - last_date.year) * 12 + (date.month - last_date.month))
            else:
                # Beyond a year, use yearly frequency
                periods.append(date.year - last_date.year)
        
        # Filter out dates before the last historical date
        future_indices = [i for i, p in enumerate(periods) if p > 0]
        future_periods = [periods[i] for i in future_indices]
        
        if not future_periods:
            return np.array([self.history['value'].iloc[-1]] * len(dates))
        
        # Generate forecast for maximum period
        max_period = max(future_periods)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            forecast_values = self.fitted_model.forecast(steps=max_period)
        
        # Create predictions array
        predictions = np.zeros(len(dates))
        
        for i, period in enumerate(periods):
            if period <= 0:
                # Use actual value for historical dates
                closest_date = min(self.history['date'], key=lambda x: abs((x - dates[i]).days))
                predictions[i] = self.history.loc[self.history['date'] == closest_date, 'value'].iloc[0]
            else:
                # Use forecast value for future dates
                predictions[i] = forecast_values[period - 1]
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            predictions = np.maximum(predictions, self.minimum_value)
        
        return predictions


class SARIMAForecaster(BaseForecaster):
    """
    Seasonal Auto-Regressive Integrated Moving Average (SARIMA) forecasting method.
    
    This method extends ARIMA to include seasonal components, making it suitable for
    forecasting time series with seasonal patterns. SARIMA is excellent for market data
    with quarterly or yearly seasonality patterns. It's represented as SARIMA(p,d,q)(P,D,Q,s)
    where the second set of parameters handles the seasonal components.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize SARIMA parameters from configuration"""
        # ARIMA order: (p, d, q)
        self.p = self.config.get('p', 1)
        self.d = self.config.get('d', 1)
        self.q = self.config.get('q', 1)
        
        # Seasonal order: (P, D, Q, s)
        # P: Seasonal AR order
        # D: Seasonal differencing order
        # Q: Seasonal MA order
        # s: Seasonal period
        self.P = self.config.get('P', 1)
        self.D = self.config.get('D', 0)
        self.Q = self.config.get('Q', 1)
        self.s = self.config.get('s', 4)  # Default to quarterly seasonality
        
        # Auto-determine order using information criteria
        self.auto_order = self.config.get('auto_order', False)
        
        # Maximum values to consider for auto-determination
        self.max_p = self.config.get('max_p', 3)
        self.max_d = self.config.get('max_d', 2)
        self.max_q = self.config.get('max_q', 3)
        self.max_P = self.config.get('max_P', 2)
        self.max_D = self.config.get('max_D', 1)
        self.max_Q = self.config.get('max_Q', 2)
        
        # Information criterion for model selection (aic or bic)
        self.criterion = self.config.get('criterion', 'aic')
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store fitted model
        self.model = None
        self.fitted_model = None
        self.history = None
        self.time_periods = None
        self.order = None
        self.seasonal_order = None
    
    def _determine_best_order(self, values: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Determine the best SARIMA order using grid search and information criteria.
        This is a simplified approach that's faster than an exhaustive grid search.
        
        Args:
            values: Time series values
            
        Returns:
            Tuple with (p, d, q) order and (P, D, Q, s) seasonal order
        """
        # Check for stationarity and determine d
        best_d = 0
        adf_result = adfuller(values, autolag='AIC')
        p_value = adf_result[1]
        
        # If p-value > 0.05, series is non-stationary
        if p_value > 0.05:
            # Try first differencing
            diff1 = np.diff(values, n=1)
            adf_result = adfuller(diff1, autolag='AIC')
            p_value = adf_result[1]
            best_d = 1
            
            # If still non-stationary, try second differencing
            if p_value > 0.05 and self.max_d >= 2:
                diff2 = np.diff(values, n=2)
                adf_result = adfuller(diff2, autolag='AIC')
                p_value = adf_result[1]
                best_d = 2 if p_value <= 0.05 else 1
        
        # For seasonal differencing, we'll use a simpler approach
        # Determine if there's strong seasonality
        best_D = 0
        if len(values) >= 2 * self.s:
            # Calculate seasonal differences
            seasonal_diff = np.array([values[i] - values[i - self.s] for i in range(self.s, len(values))])
            adf_result = adfuller(seasonal_diff, autolag='AIC')
            p_value = adf_result[1]
            
            # If seasonal difference is stationary, use D=1
            if p_value <= 0.05:
                best_D = 1
        
        # Simplified approach for p, q, P, Q using a minimal grid search
        best_order = (1, best_d, 1)
        best_seasonal_order = (1, best_D, 1, self.s)
        best_criterion = float('inf')
        
        # Try a limited set of combinations
        p_values = [0, 1, 2] if self.max_p >= 2 else [0, 1]
        q_values = [0, 1, 2] if self.max_q >= 2 else [0, 1]
        P_values = [0, 1] if self.max_P >= 1 else [0]
        Q_values = [0, 1] if self.max_Q >= 1 else [0]
        
        # Limit number of combinations to avoid excessive computation
        total_combinations = len(p_values) * len(q_values) * len(P_values) * len(Q_values)
        if total_combinations > 16:  # Arbitrary limit
            logger.warning(f"Too many combinations ({total_combinations}), limiting search space")
            p_values = p_values[:2]
            q_values = q_values[:2]
            P_values = P_values[:1]
            Q_values = Q_values[:1]
        
        for p in p_values:
            for q in q_values:
                for P in P_values:
                    for Q in Q_values:
                        try:
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore")
                                model = SARIMAX(
                                    values,
                                    order=(p, best_d, q),
                                    seasonal_order=(P, best_D, Q, self.s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                fitted_model = model.fit(disp=False, maxiter=50)
                                
                                # Get information criterion
                                criterion_value = fitted_model.aic if self.criterion == 'aic' else fitted_model.bic
                                
                                if criterion_value < best_criterion:
                                    best_criterion = criterion_value
                                    best_order = (p, best_d, q)
                                    best_seasonal_order = (P, best_D, Q, self.s)
                        except Exception as e:
                            # Skip combinations that fail to converge
                            continue
        
        logger.info(f"Determined best SARIMA order: {best_order}{best_seasonal_order}")
        return best_order, best_seasonal_order
    
    def fit(self, data: pd.DataFrame) -> 'SARIMAForecaster':
        """
        Fit SARIMA model to historical data.
        
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
        
        # Calculate time periods
        self.time_periods = data['date'].values
        
        # Get values
        values = data['value'].values
        
        # Set seasonal period based on data frequency
        dates = data['date'].values
        if len(dates) >= 3:
            # Try to infer frequency from dates
            date_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            avg_diff = sum(date_diffs) / len(date_diffs)
            
            if avg_diff < 45:  # Monthly data
                self.s = 12
            elif avg_diff < 100:  # Quarterly data
                self.s = 4
            elif avg_diff > 300:  # Annual data
                self.s = 1  # No seasonality for annual data
        
        # Check if enough data for seasonal modeling
        if len(values) < 2 * self.s:
            logger.warning(f"Not enough data points ({len(values)}) for seasonal modeling with s={self.s}")
            logger.warning("Setting seasonal order to (0,0,0,0) - equivalent to ARIMA")
            self.seasonal_order = (0, 0, 0, 0)
            self.auto_order = True  # Force auto determination of ARIMA parameters
        
        # Determine SARIMA order
        if self.auto_order:
            try:
                # Auto-determine order
                self.order, self.seasonal_order = self._determine_best_order(values)
            except Exception as e:
                logger.warning(f"Error determining best order: {str(e)}")
                logger.warning("Falling back to default order")
                self.order = (self.p, self.d, self.q)
                self.seasonal_order = (self.P, self.D, self.Q, self.s)
        else:
            # Use provided order
            self.order = (self.p, self.d, self.q)
            self.seasonal_order = (self.P, self.D, self.Q, self.s)
        
        # Update instance variables
        self.p, self.d, self.q = self.order
        self.P, self.D, self.Q, self.s = self.seasonal_order
        
        # Create and fit the SARIMA model
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.model = SARIMAX(
                    values,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.fitted_model = self.model.fit(disp=False)
                
            logger.info(f"Fitted SARIMA model with order {self.order}{self.seasonal_order}")
            self.fitted = True
        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            logger.warning("Try adjusting parameters or enable auto_order")
            raise
        
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
        
        # Generate forecast
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            forecast_result = self.fitted_model.get_forecast(steps=periods)
            forecast_values = forecast_result.predicted_mean
            
            # Get confidence intervals
            conf_int = forecast_result.conf_int(alpha=0.05)
            lower_bound = conf_int.iloc[:, 0].values
            upper_bound = conf_int.iloc[:, 1].values
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
            lower_bound = np.maximum(lower_bound, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals DataFrame
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'lower': lower_bound,
            'value': forecast_values,
            'upper': upper_bound
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
        
        # Convert dates to periods relative to last historical date
        last_date = self.history['date'].iloc[-1]
        periods = []
        
        for date in dates:
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            # Calculate periods based on frequency
            if (date - last_date).days <= 366:
                # Within a year, use monthly frequency
                periods.append((date.year - last_date.year) * 12 + (date.month - last_date.month))
            else:
                # Beyond a year, use yearly frequency
                periods.append(date.year - last_date.year)
        
        # Filter out dates before the last historical date
        future_indices = [i for i, p in enumerate(periods) if p > 0]
        future_periods = [periods[i] for i in future_indices]
        
        if not future_periods:
            return np.array([self.history['value'].iloc[-1]] * len(dates))
        
        # Generate forecast for maximum period
        max_period = max(future_periods)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            forecast_result = self.fitted_model.get_forecast(steps=max_period)
            forecast_values = forecast_result.predicted_mean.values
        
        # Create predictions array
        predictions = np.zeros(len(dates))
        
        for i, period in enumerate(periods):
            if period <= 0:
                # Use actual value for historical dates
                closest_date = min(self.history['date'], key=lambda x: abs((x - dates[i]).days))
                predictions[i] = self.history.loc[self.history['date'] == closest_date, 'value'].iloc[0]
            else:
                # Use forecast value for future dates
                predictions[i] = forecast_values[period - 1]
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            predictions = np.maximum(predictions, self.minimum_value)
        
        return predictions


class RegressionForecaster(BaseForecaster):
    """
    Regression-based forecasting method.
    
    This method uses various regression models to predict future values based on
    time features. It supports linear regression, polynomial regression, ridge,
    lasso, elastic net, and random forest regression. Feature engineering is used
    to transform the time dimension into predictive features.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize regression parameters from configuration"""
        # Regression model type
        self.model_type = self.config.get('model_type', 'linear')
        # Valid types: 'linear', 'polynomial', 'ridge', 'lasso', 'elasticnet', 'randomforest'
        
        # Polynomial degree (only used if model_type is 'polynomial')
        self.poly_degree = self.config.get('poly_degree', 2)
        
        # Regularization strength (for Ridge, Lasso, ElasticNet)
        self.alpha = self.config.get('alpha', 1.0)
        
        # L1 ratio for ElasticNet (0 = Ridge, 1 = Lasso)
        self.l1_ratio = self.config.get('l1_ratio', 0.5)
        
        # Random Forest parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', None)
        
        # Feature engineering options
        self.include_trend = self.config.get('include_trend', True)
        self.include_seasonal = self.config.get('include_seasonal', True)
        self.include_lag = self.config.get('include_lag', False)
        self.lag_periods = self.config.get('lag_periods', [1, 2, 3])
        self.seasonal_periods = self.config.get('seasonal_periods', [4, 12])  # Quarters, months
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store fitted model
        self.model = None
        self.history = None
        self.time_periods = None
        self.features = None
        self.feature_names = None
    
    def _create_time_features(self, dates: np.ndarray, values: np.ndarray = None) -> pd.DataFrame:
        """
        Create time-based features for regression.
        
        Args:
            dates: Array of dates
            values: Optional array of values for lag features
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Convert dates to datetime if they're strings
        if isinstance(dates[0], str):
            dates = pd.to_datetime(dates)
        
        # Basic time features
        features['year'] = [d.year for d in dates]
        features['month'] = [d.month for d in dates]
        features['quarter'] = [(d.month - 1) // 3 + 1 for d in dates]
        
        # Trend feature
        if self.include_trend:
            min_year = min(d.year for d in dates)
            features['trend'] = [(d.year - min_year) + (d.month - 1) / 12 for d in dates]
        
        # Seasonal features
        if self.include_seasonal:
            # Monthly seasonality
            for month in range(1, 13):
                features[f'month_{month}'] = [1 if d.month == month else 0 for d in dates]
            
            # Quarterly seasonality
            for quarter in range(1, 5):
                features[f'quarter_{quarter}'] = [1 if (d.month - 1) // 3 + 1 == quarter else 0 for d in dates]
        
        # Lag features (only for training, not for prediction)
        if self.include_lag and values is not None and len(values) > max(self.lag_periods):
            for lag in self.lag_periods:
                lag_values = np.zeros(len(dates))
                lag_values[lag:] = values[:-lag]
                features[f'lag_{lag}'] = lag_values
        
        return features
    
    def fit(self, data: pd.DataFrame) -> 'RegressionForecaster':
        """
        Fit regression model to historical data.
        
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
        
        # Get dates and values
        dates = data['date'].values
        values = data['value'].values
        
        # Create features
        features = self._create_time_features(dates, values)
        self.feature_names = features.columns.tolist()
        X = features.values
        y = values
        
        # Create appropriate regression model
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'polynomial':
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=self.poly_degree)),
                ('linear', LinearRegression())
            ])
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=self.alpha)
        elif self.model_type == 'elasticnet':
            self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        elif self.model_type == 'randomforest':
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Fit the model
        try:
            self.model.fit(X, y)
            self.fitted = True
            logger.info(f"Fitted {self.model_type} regression model")
        except Exception as e:
            logger.error(f"Error fitting regression model: {str(e)}")
            raise
        
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
        
        # Create features for forecast periods
        forecast_features = self._create_time_features(forecast_dates)
        
        # Handle lag features for forecasting
        # For forecasting, we need to iteratively predict and use those predictions for subsequent periods
        if self.include_lag:
            # Get historical values
            historical_values = self.history['value'].values
            
            # Predict each period in sequence
            forecast_values = []
            for i in range(periods):
                # Update lag features with the most recent available values
                for lag in self.lag_periods:
                    lag_idx = i - lag
                    if lag_idx < 0:
                        # Use historical values for initial forecasts
                        if abs(lag_idx) <= len(historical_values):
                            forecast_features.loc[i, f'lag_{lag}'] = historical_values[len(historical_values) + lag_idx]
                        else:
                            forecast_features.loc[i, f'lag_{lag}'] = 0
                    else:
                        # Use previously forecasted values
                        forecast_features.loc[i, f'lag_{lag}'] = forecast_values[lag_idx]
                
                # Predict for this period
                X_forecast = forecast_features.iloc[i:i+1]
                y_pred = self.model.predict(X_forecast)[0]
                forecast_values.append(y_pred)
        else:
            # Predict all periods at once
            X_forecast = forecast_features.values
            forecast_values = self.model.predict(X_forecast)
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals
        # For regression, we'll use a simple prediction error based approach
        if len(self.history) > 1:
            # Calculate RMSE on training data
            X_train = self._create_time_features(self.history['date'].values, self.history['value'].values).values
            y_train = self.history['value'].values
            y_pred_train = self.model.predict(X_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            
            # Use RMSE for confidence intervals
            z_value = 1.96  # 95% confidence
            margin = rmse * z_value
            
            # For forecasts further in the future, increase uncertainty
            margins = [margin * (1 + 0.1 * i) for i in range(periods)]
            
            lower_bound = forecast_values - margins
            upper_bound = forecast_values + margins
            
            if self.ensure_minimum:
                lower_bound = np.maximum(lower_bound, self.minimum_value)
        else:
            # If only one data point, use a percentage-based approach
            lower_bound = forecast_values * 0.8
            upper_bound = forecast_values * 1.2
        
        # Create confidence interval DataFrame
        ci_df = pd.DataFrame({
            'date': forecast_dates,
            'lower': lower_bound,
            'value': forecast_values,
            'upper': upper_bound
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
        
        # Create features for the specified dates
        features = self._create_time_features(dates)
        
        # Handle lag features for prediction
        # This is a simplified approach - for actual forecasts, use the forecast method
        if self.include_lag:
            # Get the latest values from history
            latest_values = self.history['value'].values[-max(self.lag_periods):]
            
            # Fill in lag features with the most recent available values
            for i, date in enumerate(dates):
                for lag in self.lag_periods:
                    features.loc[i, f'lag_{lag}'] = latest_values[-(lag % len(latest_values))]
        
        # Make predictions
        X_pred = features.values
        predictions = self.model.predict(X_pred)
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            predictions = np.maximum(predictions, self.minimum_value)
        
        return predictions


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