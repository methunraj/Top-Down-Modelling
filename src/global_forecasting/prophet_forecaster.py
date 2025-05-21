"""
Prophet Forecasting Module

This module provides an implementation of the Prophet forecasting method,
developed by Facebook, for time series forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from typing import Dict, List, Any, Optional

from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logger = logging.getLogger(__name__)
# Basic logging configuration if not already set by a higher-level module
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ProphetForecaster(BaseForecaster):
    """
    Prophet forecasting method.

    This class wraps the Prophet model from Facebook for time series forecasting,
    integrating it into the common forecasting framework.
    """

    def _initialize_parameters(self) -> None:
        """Initialize Prophet-specific parameters from configuration."""
        self.growth = self.config.get('growth', 'linear')
        self.seasonality_mode = self.config.get('seasonality_mode', 'additive')
        self.daily_seasonality = self.config.get('daily_seasonality', False)
        self.weekly_seasonality = self.config.get('weekly_seasonality', False)
        self.yearly_seasonality = self.config.get('yearly_seasonality', True)
        
        # Holidays should be a DataFrame with 'holiday' and 'ds' columns, and optionally 'lower_window', 'upper_window'
        holidays_df = self.config.get('holidays', None)
        if holidays_df is not None and not isinstance(holidays_df, pd.DataFrame):
            logger.warning("Holidays parameter provided but not a pandas DataFrame. Ignoring holidays.")
            self.holidays: Optional[pd.DataFrame] = None
        elif holidays_df is not None:
             if not all(col in holidays_df.columns for col in ['holiday', 'ds']):
                 logger.warning("Holidays DataFrame must contain 'holiday' and 'ds' columns. Ignoring holidays.")
                 self.holidays: Optional[pd.DataFrame] = None
             else:
                holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
                self.holidays: Optional[pd.DataFrame] = holidays_df
        else:
            self.holidays: Optional[pd.DataFrame] = None

        self.changepoint_prior_scale = self.config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = self.config.get('seasonality_prior_scale', 10.0)
        
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        self.minimum_value = self.config.get('minimum_value', 0.0) # Prophet deals with floats
        
        self.history: Optional[pd.DataFrame] = None
        self._last_date: Optional[pd.Timestamp] = None
        # self.model is initialized in __init__ of BaseForecaster as None

    def fit(self, data: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit the Prophet model to historical data.

        Args:
            data: DataFrame with 'date' and 'value' columns

        Returns:
            Self for method chaining
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns.")

        self.history = data.copy()
        
        if self.history['date'].dtype == 'object':
             self.history['date'] = pd.to_datetime(self.history['date'])
        
        self._last_date = self.history['date'].max()

        # Prophet expects columns 'ds' and 'y'
        prophet_data = self.history[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
        
        # Ensure 'y' is numeric, handle potential issues
        prophet_data['y'] = pd.to_numeric(prophet_data['y'], errors='coerce')
        if prophet_data['y'].isnull().any():
            logger.warning("NaN values found in 'value' column after conversion to numeric. These will be handled by Prophet if possible.")

        self.model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            holidays=self.holidays,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale
        )
        
        try:
            self.model.fit(prophet_data)
            self.fitted = True
            logger.info("Prophet model fitted successfully.")
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            self.fitted = False # Ensure fitted is False if error occurs
            # Re-raise the exception to allow higher-level error handling
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
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted before forecasting. Call fit() first.")

        # Map frequency for Prophet's make_future_dataframe
        # 'AS' (or 'A'), 'QS' (or 'Q'), 'MS' (or 'M') are robust choices.
        prophet_freq_map = {'Y': 'AS', 'Q': 'QS', 'M': 'MS'}
        
        if frequency.upper() not in prophet_freq_map:
            logger.warning(f"Unsupported frequency '{frequency}' for Prophet. Defaulting to Daily ('D').")
            # This might not be ideal if the original data wasn't daily.
            # Prophet can also use the inferred frequency of the historical data if freq is not specified,
            # but make_future_dataframe requires a freq for future periods.
            # A more robust approach might be to infer from self._last_date and historical data frequency.
            # For now, using 'D' as a fallback or raising an error.
            # Let's try to use the most common business frequencies, or raise error.
            raise ValueError(f"Unsupported frequency: {frequency}. Supported: Y, Q, M.")

        prophet_freq = prophet_freq_map[frequency.upper()]

        future_df = self.model.make_future_dataframe(periods=periods, freq=prophet_freq)
        
        # Predict
        forecast_output = self.model.predict(future_df)

        # Extract relevant columns for the forecast horizon
        # Prophet's forecast_output includes historical dates as well if future_df starts from history.
        # make_future_dataframe usually extends from the last date in the training data.
        # We only want the new 'periods' forecasted.
        
        forecast_dates_dt = forecast_output['ds'].dt.to_pydatetime() # Convert to datetime objects
        
        # Ensure we only take the future `periods`
        # This logic assumes `make_future_dataframe` appends `periods` new rows.
        # If the input data to fit() had N rows, future_df has N+periods rows (if it includes history)
        # or just `periods` rows if it only creates future dates.
        # Prophet's predict output matches the rows of the df passed to it.
        # `make_future_dataframe` when given `periods` makes a df of that many future rows PLUS historical dates.
        # So we need to take the tail.
        
        actual_forecast_dates = list(forecast_dates_dt[-periods:])
        forecast_values_np = forecast_output['yhat'][-periods:].values
        lower_ci_np = forecast_output['yhat_lower'][-periods:].values
        upper_ci_np = forecast_output['yhat_upper'][-periods:].values

        if self.ensure_minimum:
            forecast_values_np = np.maximum(forecast_values_np, self.minimum_value)
            lower_ci_np = np.maximum(lower_ci_np, self.minimum_value)
            # Ensure upper bound is also at least the new (capped) forecast value
            upper_ci_np = np.maximum(upper_ci_np, forecast_values_np)

        self.forecast_result = pd.DataFrame({
            'date': actual_forecast_dates,
            'value': forecast_values_np
        })
        
        self.confidence_intervals = pd.DataFrame({
            'date': actual_forecast_dates,
            'lower': lower_ci_np,
            'value': forecast_values_np, # Include mean forecast for convenience
            'upper': upper_ci_np
        })
        
        self.forecast_dates = actual_forecast_dates # Store list of datetime objects

        return self.forecast_result

    def _predict_for_dates(self, dates: np.ndarray) -> np.ndarray:
        """
        Generate predictions for specific dates using the fitted Prophet model.

        Args:
            dates: Array of dates (strings, np.datetime64, or pd.Timestamp) to predict for.

        Returns:
            Array of predicted values.
        """
        if not self.fitted or self.model is None:
            raise ValueError("Model must be fitted before predicting. Call fit() first.")

        if not isinstance(dates, (list, np.ndarray, pd.Series)):
             raise TypeError("Input 'dates' must be a list, numpy array, or pandas Series of dates.")

        # Prophet expects a DataFrame with a 'ds' column
        predict_df = pd.DataFrame({'ds': pd.to_datetime(dates)})

        try:
            forecast_output = self.model.predict(predict_df)
            predictions = forecast_output['yhat'].values
        except Exception as e:
            logger.error(f"Error during Prophet prediction for specific dates: {e}")
            # Return an array of NaNs or minimum_value on error, matching length of dates
            predictions = np.full(len(dates), np.nan if not self.ensure_minimum else self.minimum_value)
            # Or re-raise: raise

        if self.ensure_minimum:
            predictions = np.maximum(predictions, self.minimum_value)
            
        return predictions

    def plot(self, include_history: bool = True, 
            include_intervals: bool = True,
            figsize: tuple = (10, 6)) -> Any:
        """
        Plot the forecast results using Prophet's built-in plotting capabilities
        or a custom plot if BaseForecaster's plot is preferred.

        Args:
            include_history: Whether to include historical data (Prophet's plot often does by default).
            include_intervals: Whether to include confidence intervals.
            figsize: Figure size tuple (width, height).

        Returns:
            Matplotlib figure object or None if plotting is not possible.
        """
        if not self.fitted or self.model is None or self.forecast_result is None:
            logger.warning("Cannot plot: model not fitted or no forecast available.")
            return None

        try:
            # Use Prophet's own plotting for more detailed component plots if desired
            # fig1 = self.model.plot(self.forecast_result.rename(columns={'date':'ds', 'value':'yhat'}))
            # fig2 = self.model.plot_components(self.forecast_result.rename(columns={'date':'ds', 'value':'yhat'}))
            # return [fig1, fig2] # Or handle them differently
            
            # For consistency with other forecasters, use BaseForecaster's plot logic
            # Ensure forecast_result and confidence_intervals are correctly formatted
            # The BaseForecaster.plot() method will be called if this method is not overridden
            # or if super().plot() is called.
            # Here, we can leverage Prophet's plotting if we have the full forecast object from predict()
            
            # Get the forecast object that includes historical data for Prophet's plot
            # This requires `make_future_dataframe` to include historical dates (default)
            # and then `predict` on that. Our current `self.forecast_result` only has future dates.
            
            # Let's get the full forecast from Prophet for plotting
            if self.forecast_dates is not None and len(self.forecast_dates) > 0:
                # Create a DataFrame that Prophet's plot function expects, including future dates
                # The `self.model.history` contains the original training data ('ds', 'y')
                # `self.forecast_result` has 'date', 'value' for future.
                # We need to combine or re-predict for a continuous plot.
                
                # Simplest is to use the forecast object that was used to generate self.forecast_result
                # This means `forecast_output` in the `forecast` method.
                # We need to store it or re-generate it.
                # For now, let's rely on the BaseForecaster plot if this becomes too complex.
                
                # Re-generate future_df and forecast_output for plotting if not stored
                # This is inefficient but ensures Prophet's plot works as intended.
                
                # Determine periods and frequency from stored forecast_dates if possible
                # This is a bit of a reverse engineering from what we have.
                if self.forecast_dates and self._last_date:
                    # This assumes forecast_dates are contiguous from _last_date
                    # and have a consistent frequency that can be mapped back.
                    # This is complex. Let's just use the BaseForecaster plot.
                    pass # Fall through to super().plot() by not returning early
                
            logger.info("Using BaseForecaster's plot method for ProphetForecaster.")
            return super().plot(include_history=include_history, 
                                include_intervals=include_intervals, 
                                figsize=figsize)

        except ImportError:
            logger.warning("Matplotlib not available for plotting.")
            return None
        except Exception as e:
            logger.error(f"Error during Prophet plotting: {e}")
            return None
