"""
Time Series-Specific Forecasting Module - Advanced time series modeling techniques

This module provides implementations of specialized time series forecasting methods,
including Vector Autoregression (VAR), Temporal Fusion Transformer (TFT), and DeepAR.
These models are particularly well-suited for multivariate time series and complex
temporal dependencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings

from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VARForecaster(BaseForecaster):
    """
    Vector Autoregression (VAR) forecasting method.
    
    VAR models the relationships between multiple quantities as they change over time.
    It's especially useful when variables influence each other, allowing for more
    accurate forecasts by incorporating these interdependencies. VAR extends 
    autoregressive models to capture the linear interdependencies among multiple time series.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize VAR parameters from configuration"""
        # Order of the VAR model (number of lags)
        self.order = self.config.get('order', 1)
        
        # Method to determine optimal lag order if auto_order is True
        # Options: 'aic', 'bic', 'fpe', 'hqic'
        self.criterion = self.config.get('criterion', 'aic')
        
        # Maximum lag order to consider if auto_order is True
        self.max_order = self.config.get('max_order', 10)
        
        # Whether to automatically determine the order
        self.auto_order = self.config.get('auto_order', False)
        
        # Whether to include a trend term
        # Options: 'n'=no trend, 'c'=constant, 't'=linear, 'ct'=constant and trend
        self.trend = self.config.get('trend', 'c')
        
        # List of column names for exogenous variables
        self.exog_vars = self.config.get('exog_vars', None)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store fitted model
        self.model = None
        self.fitted_model = None
        self.history = None
        self.endog_cols = None  # Target variables
        self.exog_cols = None   # Exogenous variables
        
    def fit(self, data: pd.DataFrame) -> 'VARForecaster':
        """
        Fit VAR model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns, plus any exogenous variables
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Check if statsmodels is available
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            from statsmodels.tsa.vector_ar.vecm import select_order
        except ImportError:
            logger.error("statsmodels is required for VAR forecasting")
            raise ImportError("statsmodels is required for VAR forecasting")
        
        # Convert dates if they're strings
        if data['date'].dtype == 'object':
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
        
        # Sort by date
        data = data.sort_values('date')
        
        # Store historical data
        self.history = data.copy()
        
        # Set up endogenous variables (targets)
        self.endog_cols = ['value']
        endog_data = data[self.endog_cols].values
        
        # Set up exogenous variables if provided
        self.exog_cols = self.exog_vars
        exog_data = None
        if self.exog_cols is not None:
            # Check if all exog columns exist
            missing_cols = [col for col in self.exog_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Exogenous variables not found in data: {missing_cols}")
            
            exog_data = data[self.exog_cols].values
        
        # Create and fit VAR model
        try:
            # Determine optimal lag order if requested
            if self.auto_order:
                order_results = select_order(endog_data, maxlags=self.max_order, 
                                            trend=self.trend)
                optimal_order = getattr(order_results, self.criterion)
                self.order = optimal_order
                logger.info(f"Selected optimal VAR order: {self.order} using {self.criterion}")
            
            # Create and fit the model
            self.model = VAR(endog_data)
            self.fitted_model = self.model.fit(maxlags=self.order, trend=self.trend, 
                                              exog=exog_data)
            
            logger.info(f"Fitted VAR model with order {self.order}")
            self.fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting VAR model: {str(e)}")
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
        
        # Set up exogenous data for forecast if needed
        forecast_exog = None
        if self.exog_cols is not None:
            # For simplicity, we'll use the last values of exog variables
            # In a real implementation, you would want to forecast these as well
            last_exog = self.history[self.exog_cols].iloc[-1].values
            forecast_exog = np.tile(last_exog, (periods, 1))
        
        # Generate forecast
        try:
            # Use the forecast method from the fitted model
            forecast_result = self.fitted_model.forecast(y=self.model.endog, 
                                                        steps=periods, 
                                                        exog_future=forecast_exog)
            
            # Extract the values for the 'value' column (first column in our setup)
            forecast_values = forecast_result[:, 0]
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                forecast_values = np.maximum(forecast_values, self.minimum_value)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'value': forecast_values
            })
            
            # Create confidence intervals for the forecast
            # This is a simple implementation - VAR provides more sophisticated methods
            forecast_stderr = np.zeros(periods)
            
            # Get increasing standard error based on forecast period
            for i in range(periods):
                # Standard error increases with forecast horizon
                forecast_stderr[i] = np.std(self.history['value']) * np.sqrt(1 + 0.1 * i)
            
            # 95% confidence interval (approximately 2 standard deviations)
            z_value = 1.96
            lower_bound = forecast_values - z_value * forecast_stderr
            upper_bound = forecast_values + z_value * forecast_stderr
            
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
            
        except Exception as e:
            logger.error(f"Error generating VAR forecast: {str(e)}")
            raise
    
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
        
        # Generate forecast for maximum period needed
        max_period = max(future_periods)
        
        # Set up exogenous data for forecast if needed
        forecast_exog = None
        if self.exog_cols is not None:
            # For simplicity, use the last values of exog variables
            last_exog = self.history[self.exog_cols].iloc[-1].values
            forecast_exog = np.tile(last_exog, (max_period, 1))
        
        # Generate forecast
        try:
            forecast_result = self.fitted_model.forecast(y=self.model.endog, 
                                                        steps=max_period, 
                                                        exog_future=forecast_exog)
            
            # Extract the values for the 'value' column
            all_forecast_values = forecast_result[:, 0]
            
            # Create predictions array
            predictions = np.zeros(len(dates))
            
            for i, period in enumerate(periods):
                if period <= 0:
                    # Use actual value for historical dates
                    closest_date = min(self.history['date'], key=lambda x: abs((x - dates[i]).days))
                    predictions[i] = self.history.loc[self.history['date'] == closest_date, 'value'].iloc[0]
                else:
                    # Use forecast value for future dates
                    predictions[i] = all_forecast_values[period - 1]
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                predictions = np.maximum(predictions, self.minimum_value)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for specified dates: {str(e)}")
            raise


class TemporalFusionTransformerForecaster(BaseForecaster):
    """
    Temporal Fusion Transformer (TFT) forecasting method.
    
    TFT is a powerful attention-based deep learning model specifically designed for
    multi-horizon time series forecasting. It combines high-performance sequence modeling,
    interpretable attention layers, and specialized components for time-based data.
    It excels at capturing temporal relationships and variable selection.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize TFT parameters from configuration"""
        # Number of past time steps to use as context
        self.context_length = self.config.get('context_length', 24)
        
        # Learning rate for model training
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # Hidden dimension size
        self.hidden_size = self.config.get('hidden_size', 128)
        
        # Number of attention heads
        self.attention_heads = self.config.get('attention_heads', 4)
        
        # Number of training epochs
        self.epochs = self.config.get('epochs', 100)
        
        # Dropout rate
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        
        # Whether to use GPU if available
        self.use_gpu = self.config.get('use_gpu', True)
        
        # Batch size for training
        self.batch_size = self.config.get('batch_size', 64)
        
        # Number of hidden layers
        self.hidden_layers = self.config.get('hidden_layers', 2)
        
        # List of static categorical features
        self.static_categoricals = self.config.get('static_categoricals', [])
        
        # List of static real features
        self.static_reals = self.config.get('static_reals', [])
        
        # List of time-varying categorical features known for future time steps
        self.future_categoricals = self.config.get('future_categoricals', [])
        
        # List of time-varying real features known for future time steps
        self.future_reals = self.config.get('future_reals', [])
        
        # List of time-varying categorical features only known for history
        self.past_categoricals = self.config.get('past_categoricals', [])
        
        # List of time-varying real features only known for history
        self.past_reals = self.config.get('past_reals', ['value'])
        
        # Early stopping patience
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store fitted model
        self.model = None
        self.history = None
        self.time_index = None
        self.trainer = None
        self.training_data = None
        
    def fit(self, data: pd.DataFrame) -> 'TemporalFusionTransformerForecaster':
        """
        Fit TFT model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns, plus any additional features
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Check if PyTorch/PyTorch Lightning/PyTorch Forecasting are available
        try:
            import torch
            import pytorch_lightning as pl
            from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
            from pytorch_forecasting.metrics import QuantileLoss
            from pytorch_forecasting.data import GroupNormalizer
        except ImportError:
            logger.error("PyTorch, PyTorch Lightning, and PyTorch Forecasting are required for TFT forecasting")
            logger.warning("Using mock implementation for compatibility")
            # We'll set up a mock implementation that returns reasonable forecasts
            self._setup_mock_model(data)
            return self
        
        # Convert dates if they're strings
        if data['date'].dtype == 'object':
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
        
        # Sort by date
        data = data.sort_values('date')
        
        # Store historical data
        self.history = data.copy()
        
        try:
            # Prepare data for TFT
            # Add time index column if needed
            if 'time_idx' not in data.columns:
                data['time_idx'] = range(len(data))
            self.time_index = data['time_idx'].max()
            
            # Add group ID if needed (TFT requires a group_id column)
            if 'group_id' not in data.columns:
                data['group_id'] = 0  # Single time series
            
            # Create dataset
            max_prediction_length = 24  # Maximum forecast horizon
            max_encoder_length = self.context_length
            training_cutoff = data['time_idx'].max() - max_prediction_length
            
            self.training_data = TimeSeriesDataSet(
                data=data[lambda x: x['time_idx'] <= training_cutoff],
                time_idx="time_idx",
                target="value",
                group_ids=["group_id"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                static_categoricals=self.static_categoricals,
                static_reals=self.static_reals,
                time_varying_known_categoricals=self.future_categoricals,
                time_varying_known_reals=self.future_reals,
                time_varying_unknown_categoricals=self.past_categoricals,
                time_varying_unknown_reals=self.past_reals,
                target_normalizer=GroupNormalizer(
                    groups=["group_id"], transformation="softplus"
                ),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
            
            # Create validation dataset
            validation_data = TimeSeriesDataSet.from_dataset(
                self.training_data, data, min_prediction_idx=training_cutoff + 1, stop_randomization=True
            )
            
            # Create data loaders
            train_dataloader = self.training_data.to_dataloader(
                train=True, batch_size=self.batch_size, num_workers=0
            )
            val_dataloader = validation_data.to_dataloader(
                train=False, batch_size=self.batch_size, num_workers=0
            )
            
            # Create model
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_data,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_heads,
                dropout=self.dropout_rate,
                hidden_continuous_size=self.hidden_size,
                loss=QuantileLoss(),
                log_interval=10,
                reduce_on_plateau_patience=3,
            )
            
            # Create trainer
            early_stop_callback = pl.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=self.early_stopping_patience, verbose=False, mode="min"
            )
            
            self.trainer = pl.Trainer(
                max_epochs=self.epochs,
                gpus=1 if torch.cuda.is_available() and self.use_gpu else 0,
                gradient_clip_val=0.1,
                callbacks=[early_stop_callback],
                limit_train_batches=30,  # Comment out for full training
                enable_checkpointing=False,
                logger=False,
            )
            
            # Fit model
            self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            
            logger.info("Fitted Temporal Fusion Transformer model")
            self.fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting TFT model: {str(e)}")
            logger.warning("Using mock implementation for compatibility")
            # If fitting fails, set up a mock implementation
            self._setup_mock_model(data)
            
        return self
    
    def _setup_mock_model(self, data: pd.DataFrame) -> None:
        """
        Set up a mock model for compatibility when PyTorch/TFT isn't available.
        Uses exponential smoothing as a fallback.
        
        Args:
            data: Historical data
        """
        from src.global_forecasting.statistical import ExponentialSmoothingForecaster
        
        logger.warning("Using ExponentialSmoothingForecaster as a fallback for TFT")
        
        # Create and fit a simple exponential smoothing model as fallback
        self.mock_model = ExponentialSmoothingForecaster(self.config)
        self.mock_model.fit(data)
        self.fitted = True
        
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
        
        try:
            # Check if we're using the mock model (fallback)
            if hasattr(self, 'mock_model'):
                mock_forecast = self.mock_model.forecast(periods, frequency)
                self.forecast_result = mock_forecast
                self.forecast_dates = forecast_dates
                self.confidence_intervals = self.mock_model.confidence_intervals
                return mock_forecast
            
            # Real TFT prediction
            import torch
            from pytorch_forecasting import TimeSeriesDataSet
            
            # Prepare prediction data
            encoder_data = self.history.iloc[-self.context_length:].copy()
            encoder_data['time_idx'] = range(self.time_index - self.context_length + 1, self.time_index + 1)
            
            # For simplicity, we're using a naive approach for future known features
            # In a real implementation, you would properly set up these future features
            last_history_row = self.history.iloc[-1:].copy()
            
            # Create prediction data
            prediction_data = pd.DataFrame()
            for i in range(periods):
                new_row = last_history_row.copy()
                new_row['time_idx'] = self.time_index + i + 1
                new_row['date'] = forecast_dates[i]
                prediction_data = pd.concat([prediction_data, new_row])
            
            # Combine encoder and prediction data
            combined_data = pd.concat([encoder_data, prediction_data])
            
            # Predict
            predictions = self.model.predict(combined_data, mode="raw")
            
            # Extract median prediction (50th quantile)
            forecast_values = predictions.output.prediction[:, 0, predictions.output.quantiles.index(0.5)].numpy()
            
            # Get confidence intervals (10th and 90th percentiles)
            lower_bound = predictions.output.prediction[:, 0, predictions.output.quantiles.index(0.1)].numpy()
            upper_bound = predictions.output.prediction[:, 0, predictions.output.quantiles.index(0.9)].numpy()
            
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
            
        except Exception as e:
            logger.error(f"Error generating TFT forecast: {str(e)}")
            
            # If real prediction fails, use the mock model as backup
            if not hasattr(self, 'mock_model'):
                self._setup_mock_model(self.history)
            
            mock_forecast = self.mock_model.forecast(periods, frequency)
            self.forecast_result = mock_forecast
            self.forecast_dates = forecast_dates
            self.confidence_intervals = self.mock_model.confidence_intervals
            
            return mock_forecast
    
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
        
        # If we're using the mock model, delegate to it
        if hasattr(self, 'mock_model'):
            return self.mock_model._predict_for_dates(dates)
        
        # Convert dates to an array of forecast periods
        last_date = self.history['date'].iloc[-1]
        periods = []
        date_indices = []
        
        for i, date in enumerate(dates):
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            if date <= last_date:
                # For dates in the past, use the actual historical value
                periods.append(0)
                date_indices.append(i)
            else:
                # For future dates, calculate the period offset
                # This is a simplification - in reality you'd want to respect the frequency
                delta = (date - last_date).days
                if delta <= 31:  # Roughly a month
                    periods.append(delta // 7 + 1)  # Weekly approximation
                elif delta <= 366:  # Roughly a year
                    periods.append(delta // 30 + 1)  # Monthly approximation
                else:
                    periods.append(delta // 365 + 1)  # Yearly approximation
                date_indices.append(i)
        
        # Get maximum forecast period needed
        max_period = max(periods) if periods else 0
        
        if max_period == 0:
            # All dates are in the past, use historical values
            predictions = np.zeros(len(dates))
            for i, date in enumerate(dates):
                closest_date = min(self.history['date'], key=lambda x: abs((x - date).days))
                predictions[i] = self.history.loc[self.history['date'] == closest_date, 'value'].iloc[0]
            return predictions
        
        # Generate a forecast for the maximum needed periods
        forecast_df = self.forecast(max_period, 'M')  # Use monthly frequency as default
        
        # Map each date to the appropriate forecast value
        predictions = np.zeros(len(dates))
        
        for i, period in zip(date_indices, periods):
            if period == 0:
                # Historical date
                closest_date = min(self.history['date'], key=lambda x: abs((x - dates[i]).days))
                predictions[i] = self.history.loc[self.history['date'] == closest_date, 'value'].iloc[0]
            else:
                # Future date - get the corresponding forecast
                period_idx = min(period - 1, len(forecast_df) - 1)  # Ensure we don't go out of bounds
                predictions[i] = forecast_df['value'].iloc[period_idx]
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            predictions = np.maximum(predictions, self.minimum_value)
        
        return predictions


class DeepARForecaster(BaseForecaster):
    """
    DeepAR forecasting method.
    
    DeepAR is a probabilistic forecasting model using autoregressive recurrent neural
    networks. Developed by Amazon, it captures complex patterns and provides full
    predictive distributions. It excels at forecasting multiple related time series
    simultaneously and handles missing values and variable-length time series.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize DeepAR parameters from configuration"""
        # Number of past time steps to use
        self.context_length = self.config.get('context_length', 30)
        
        # Number of sample paths to draw
        self.num_samples = self.config.get('num_samples', 100)
        
        # Hidden dimension of LSTM/GRU cells
        self.hidden_size = self.config.get('hidden_size', 40)
        
        # Number of LSTM/GRU layers
        self.num_layers = self.config.get('num_layers', 2)
        
        # Dropout rate
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        
        # Learning rate for model training
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # Number of training epochs
        self.epochs = self.config.get('epochs', 100)
        
        # RNN cell type ('lstm' or 'gru')
        self.cell_type = self.config.get('cell_type', 'lstm')
        
        # Whether to use GPU if available
        self.use_gpu = self.config.get('use_gpu', True)
        
        # Batch size for training
        self.batch_size = self.config.get('batch_size', 32)
        
        # Frequency of prediction
        self.freq = self.config.get('freq', 'M')  # Monthly by default
        
        # Likelihood for the model (default: StudentT)
        self.likelihood = self.config.get('likelihood', 'studentt')
        
        # Early stopping patience
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store fitted model
        self.model = None
        self.history = None
        self.predictor = None
        
    def fit(self, data: pd.DataFrame) -> 'DeepARForecaster':
        """
        Fit DeepAR model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Check if GluonTS is available
        try:
            import mxnet as mx
            from gluonts.dataset.common import ListDataset
            from gluonts.model.deepar import DeepAREstimator
            from gluonts.mx.trainer import Trainer
        except ImportError:
            logger.error("MXNet and GluonTS are required for DeepAR forecasting")
            logger.warning("Using mock implementation for compatibility")
            # We'll set up a mock implementation that returns reasonable forecasts
            self._setup_mock_model(data)
            return self
        
        # Convert dates if they're strings
        if data['date'].dtype == 'object':
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
        
        # Sort by date
        data = data.sort_values('date')
        
        # Store historical data
        self.history = data.copy()
        
        try:
            # Determine frequency
            date_diffs = [(data['date'].iloc[i+1] - data['date'].iloc[i]).days 
                        for i in range(len(data)-1)]
            avg_diff = sum(date_diffs) / len(date_diffs) if date_diffs else 30
            
            if avg_diff < 10:  # Roughly weekly or less
                freq = '1D'  # Daily
            elif avg_diff < 45:  # Roughly monthly
                freq = '1M'  # Monthly
            elif avg_diff < 100:  # Roughly quarterly
                freq = '1Q'  # Quarterly
            else:  # Roughly yearly or more
                freq = '1Y'  # Yearly
            
            self.freq = freq
            
            # Prepare data for DeepAR
            start_date = data['date'].iloc[0]
            
            # Convert to GluonTS compatible format
            training_data = [
                {
                    "start": start_date,
                    "target": data['value'].values,
                    "item_id": 0,  # Single time series
                }
            ]
            
            train_ds = ListDataset(training_data, freq=freq)
            
            # Create Trainer
            trainer = Trainer(
                ctx=mx.gpu() if mx.context.num_gpus() > 0 and self.use_gpu else mx.cpu(),
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                patience=self.early_stopping_patience,
            )
            
            # Create DeepAR model
            estimator = DeepAREstimator(
                freq=freq,
                prediction_length=24,  # Maximum forecast horizon
                context_length=self.context_length,
                num_layers=self.num_layers,
                num_cells=self.hidden_size,
                cell_type=self.cell_type,
                dropout_rate=self.dropout_rate,
                use_feat_dynamic_real=False,  # No dynamic features for simplicity
                use_feat_static_cat=False,    # No categorical features for simplicity
                use_feat_static_real=False,   # No static features for simplicity
                cardinality=[1],              # Single time series
                trainer=trainer,
            )
            
            # Train the model
            self.predictor = estimator.train(train_ds)
            
            logger.info("Fitted DeepAR model")
            self.fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting DeepAR model: {str(e)}")
            logger.warning("Using mock implementation for compatibility")
            # If fitting fails, set up a mock implementation
            self._setup_mock_model(data)
            
        return self
    
    def _setup_mock_model(self, data: pd.DataFrame) -> None:
        """
        Set up a mock model for compatibility when GluonTS/DeepAR isn't available.
        Uses SARIMA as a fallback.
        
        Args:
            data: Historical data
        """
        from src.global_forecasting.statistical import SARIMAForecaster
        
        logger.warning("Using SARIMAForecaster as a fallback for DeepAR")
        
        # Create and fit a SARIMA model as fallback
        self.mock_model = SARIMAForecaster(self.config)
        self.mock_model.fit(data)
        self.fitted = True
        
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
        
        try:
            # Check if we're using the mock model (fallback)
            if hasattr(self, 'mock_model'):
                mock_forecast = self.mock_model.forecast(periods, frequency)
                self.forecast_result = mock_forecast
                self.forecast_dates = forecast_dates
                self.confidence_intervals = self.mock_model.confidence_intervals
                return mock_forecast
            
            # Real DeepAR prediction
            from gluonts.dataset.common import ListDataset
            from gluonts.dataset.field_names import FieldName
            
            # Prepare data for prediction
            prediction_input = [
                {
                    "start": self.history['date'].iloc[0],
                    "target": self.history['value'].values,
                    "item_id": 0,
                }
            ]
            
            test_ds = ListDataset(
                prediction_input,
                freq=self.freq,
            )
            
            # Generate predictions
            predictions = list(self.predictor.predict(test_ds))
            prediction = predictions[0]  # We only have one time series
            
            # Extract median prediction (quantile 0.5) and confidence intervals
            samples = prediction.samples
            forecast_values = np.median(samples, axis=0)
            
            # Trim to the requested number of periods
            if len(forecast_values) > periods:
                forecast_values = forecast_values[:periods]
            elif len(forecast_values) < periods:
                # Extend if necessary
                extension = [forecast_values[-1]] * (periods - len(forecast_values))
                forecast_values = np.append(forecast_values, extension)
            
            # Calculate confidence intervals from samples
            lower_bound = np.quantile(samples, 0.1, axis=0)
            upper_bound = np.quantile(samples, 0.9, axis=0)
            
            # Trim intervals to the requested number of periods
            if len(lower_bound) > periods:
                lower_bound = lower_bound[:periods]
                upper_bound = upper_bound[:periods]
            elif len(lower_bound) < periods:
                # Extend if necessary
                lower_extension = [lower_bound[-1]] * (periods - len(lower_bound))
                upper_extension = [upper_bound[-1]] * (periods - len(upper_bound))
                lower_bound = np.append(lower_bound, lower_extension)
                upper_bound = np.append(upper_bound, upper_extension)
            
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
            
        except Exception as e:
            logger.error(f"Error generating DeepAR forecast: {str(e)}")
            
            # If real prediction fails, use the mock model as backup
            if not hasattr(self, 'mock_model'):
                self._setup_mock_model(self.history)
            
            mock_forecast = self.mock_model.forecast(periods, frequency)
            self.forecast_result = mock_forecast
            self.forecast_dates = forecast_dates
            self.confidence_intervals = self.mock_model.confidence_intervals
            
            return mock_forecast
    
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
        
        # If we're using the mock model, delegate to it
        if hasattr(self, 'mock_model'):
            return self.mock_model._predict_for_dates(dates)
        
        # Convert dates to an array of forecast periods
        last_date = self.history['date'].iloc[-1]
        periods = []
        date_indices = []
        
        for i, date in enumerate(dates):
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            if date <= last_date:
                # For dates in the past, use the actual historical value
                periods.append(0)
                date_indices.append(i)
            else:
                # For future dates, calculate the period offset
                # This is a simplification - in reality you'd want to respect the frequency
                delta = (date - last_date).days
                if delta <= 31:  # Roughly a month
                    periods.append(delta // 7 + 1)  # Weekly approximation
                elif delta <= 366:  # Roughly a year
                    periods.append(delta // 30 + 1)  # Monthly approximation
                else:
                    periods.append(delta // 365 + 1)  # Yearly approximation
                date_indices.append(i)
        
        # Get maximum forecast period needed
        max_period = max(periods) if periods else 0
        
        if max_period == 0:
            # All dates are in the past, use historical values
            predictions = np.zeros(len(dates))
            for i, date in enumerate(dates):
                closest_date = min(self.history['date'], key=lambda x: abs((x - date).days))
                predictions[i] = self.history.loc[self.history['date'] == closest_date, 'value'].iloc[0]
            return predictions
        
        # Generate a forecast for the maximum needed periods
        forecast_df = self.forecast(max_period, 'M')  # Use monthly frequency as default
        
        # Map each date to the appropriate forecast value
        predictions = np.zeros(len(dates))
        
        for i, period in zip(date_indices, periods):
            if period == 0:
                # Historical date
                closest_date = min(self.history['date'], key=lambda x: abs((x - dates[i]).days))
                predictions[i] = self.history.loc[self.history['date'] == closest_date, 'value'].iloc[0]
            else:
                # Future date - get the corresponding forecast
                period_idx = min(period - 1, len(forecast_df) - 1)  # Ensure we don't go out of bounds
                predictions[i] = forecast_df['value'].iloc[period_idx]
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            predictions = np.maximum(predictions, self.minimum_value)
        
        return predictions