"""
Hybrid Forecasting Models

This module implements hybrid time series forecasting methods that combine multiple
approaches to achieve more robust and accurate forecasts.

Models implemented:
- TBATS: Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components
- NBEATS: Neural Basis Expansion Analysis for Time Series
- Hybrid ETS-ARIMA: Combines exponential smoothing and ARIMA models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import warnings
from datetime import datetime
from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TBATSForecaster(BaseForecaster):
    """
    TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components) Forecaster.
    
    TBATS is an exponential smoothing state space model with Box-Cox transformation, ARMA
    errors, trend and seasonal components. It uses trigonometric seasonality based on Fourier
    series to handle complex seasonal patterns, multiple seasonal periods, and non-integer
    seasonal periods.
    
    This model is particularly effective for time series with multiple seasonal patterns,
    such as hourly data with both daily and weekly seasonality.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize TBATS model parameters from configuration."""
        # Whether to use Box-Cox transformation (None = auto-detection)
        self.use_box_cox = self.config.get('use_box_cox', None)
        
        # Box-Cox transformation parameter (only used if use_box_cox is True)
        self.box_cox_lambda = self.config.get('box_cox_lambda', None)
        
        # Whether to use trend
        self.use_trend = self.config.get('use_trend', True)
        
        # Whether to use damped trend
        self.use_damped_trend = self.config.get('use_damped_trend', False)
        
        # Seasonal periods (e.g., [12] for annual seasonality with monthly data)
        self.seasonal_periods = self.config.get('seasonal_periods', None)
        
        # Whether to use ARMA errors
        self.use_arma_errors = self.config.get('use_arma_errors', True)
        
        # Order of ARMA errors
        self.arma_order = self.config.get('arma_order', None)  # (p, q) tuple
        
        # Settings for model selection
        self.show_warnings = self.config.get('show_warnings', False)
        self.n_jobs = self.config.get('n_jobs', 1)
        
        # Whether parameters have been automatically estimated
        self.auto_params = False
        
        # Verification
        if self.seasonal_periods is not None and not isinstance(self.seasonal_periods, list):
            self.seasonal_periods = [self.seasonal_periods]
            
        # Store model context
        self.model = None
        self.fitted = False
        
        # Import TBATS
        try:
            from tbats import TBATS
            logger.info("TBATS package successfully imported")
        except ImportError:
            logger.error("Failed to import TBATS. Please install it with: pip install tbats")
            raise ImportError("The TBATS package is required for this forecaster. Please install it with: pip install tbats")

    def fit(self, data: pd.DataFrame) -> 'TBATSForecaster':
        """
        Fit the TBATS model to historical data.
        
        Args:
            data: DataFrame containing historical data with 'date' and 'value' columns
                
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Store historical data
        self.history_df = data.copy()
        
        # Convert to numpy array for TBATS
        self.y = data['value'].values
        self.dates = pd.to_datetime(data['date'])
        
        # Ensure data is sorted by date
        if not self.dates.equals(self.dates.sort_values()):
            logger.warning("Data not sorted by date. Sorting automatically.")
            sort_idx = self.dates.argsort()
            self.y = self.y[sort_idx]
            self.dates = self.dates.iloc[sort_idx]
        
        # Determine data frequency if seasonal_periods not provided
        if self.seasonal_periods is None:
            self.auto_params = True
            
            # Detect interval between dates
            try:
                # Try to infer frequency
                freq = pd.infer_freq(self.dates)
                
                if freq is None:
                    # If can't infer, check if the dates are evenly spaced
                    date_diffs = np.diff(self.dates)
                    if np.all(date_diffs == date_diffs[0]):
                        # Data is evenly spaced, try to determine if it's monthly, quarterly, etc.
                        days_diff = date_diffs[0].days
                        if 28 <= days_diff <= 31:
                            # Monthly data, annual seasonality
                            self.seasonal_periods = [12]
                        elif 90 <= days_diff <= 92:
                            # Quarterly data, annual seasonality
                            self.seasonal_periods = [4]
                        elif 360 <= days_diff <= 366:
                            # Yearly data, no obvious seasonality
                            self.seasonal_periods = None
                    else:
                        # Not evenly spaced, can't determine seasonality
                        self.seasonal_periods = None
                else:
                    # Use inferred frequency to set seasonal periods
                    if freq.startswith('D'):
                        # Daily data
                        self.seasonal_periods = [7, 365.25]  # Weekly and annual
                    elif freq.startswith('W'):
                        # Weekly data
                        self.seasonal_periods = [52]  # Annual
                    elif freq.startswith('M'):
                        # Monthly data
                        self.seasonal_periods = [12]  # Annual
                    elif freq.startswith('Q'):
                        # Quarterly data
                        self.seasonal_periods = [4]  # Annual
                    else:
                        # Unknown frequency
                        self.seasonal_periods = None
            except Exception as e:
                logger.warning(f"Could not determine seasonal periods automatically: {str(e)}")
                self.seasonal_periods = None
        
        # Import TBATS
        try:
            from tbats import TBATS
        except ImportError:
            logger.error("Failed to import TBATS. Please install it with: pip install tbats")
            raise ImportError("The TBATS package is required for this forecaster")
            
        # Build model configuration
        tbats_params = {
            'use_box_cox': self.use_box_cox,
            'box_cox_bounds': (0, 1) if self.use_box_cox is True else None,
            'use_trend': self.use_trend,
            'use_damped_trend': self.use_damped_trend,
            'seasonal_periods': self.seasonal_periods,
            'use_arma_errors': self.use_arma_errors,
            'show_warnings': self.show_warnings,
            'n_jobs': self.n_jobs
        }
        
        # Remove None values to use defaults
        tbats_params = {k: v for k, v in tbats_params.items() if v is not None}
        
        # Specific parameters for box_cox_lambda and arma_order if provided
        if self.use_box_cox and self.box_cox_lambda is not None:
            tbats_params['box_cox_lambda'] = self.box_cox_lambda
            
        if self.use_arma_errors and self.arma_order is not None:
            tbats_params['p'] = self.arma_order[0]
            tbats_params['q'] = self.arma_order[1]
        
        try:
            # Create and fit the TBATS model
            logger.info("Fitting TBATS model...")
            estimator = TBATS(**tbats_params)
            self.model = estimator.fit(self.y)
            
            # Store fitted parameters
            self.fitted_params = {
                'box_cox_lambda': self.model.params.box_cox_lambda,
                'alpha': self.model.params.alpha,
                'beta': self.model.params.beta,
                'phi': self.model.params.phi,
                'seasonal_harmonics': self.model.params.seasonal_harmonics,
                'arma_p': self.model.params.p,
                'arma_q': self.model.params.q,
                'seasonal_periods': self.model.params.seasonal_periods
            }
            
            logger.info("TBATS model fitted successfully")
            logger.info(f"Fitted parameters: {self.fitted_params}")
            
            self.fitted = True
            
        except Exception as e:
            logger.error(f"Error during TBATS model fitting: {str(e)}")
            self.fitted = False
            raise
            
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate a forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
                
        Returns:
            DataFrame with forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
            
        # Generate forecasts
        logger.info(f"Generating TBATS forecast for {periods} periods...")
        y_forecast = self.model.forecast(steps=periods)
        
        # Generate future dates
        last_date = self.dates.iloc[-1]
        
        # Create date sequence based on frequency
        if frequency == 'Y':
            future_dates = pd.date_range(
                start=pd.Timestamp(last_date.year + 1, 1, 1),
                periods=periods,
                freq='YS'  # Year start
            )
        elif frequency == 'Q':
            # Find next quarter start
            last_quarter = pd.Timestamp(last_date).to_period('Q')
            next_quarter_start = (last_quarter + 1).to_timestamp()
            
            future_dates = pd.date_range(
                start=next_quarter_start,
                periods=periods,
                freq='QS'  # Quarter start
            )
        elif frequency == 'M':
            # Find next month start
            last_month = pd.Timestamp(last_date).to_period('M')
            next_month_start = (last_month + 1).to_timestamp()
            
            future_dates = pd.date_range(
                start=next_month_start,
                periods=periods,
                freq='MS'  # Month start
            )
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'value': y_forecast
        })
        
        # Store forecast result
        self.forecast_result = forecast_df
        self.forecast_dates = future_dates
        
        # Generate confidence intervals
        try:
            lower, upper = self.model.forecast_interval(steps=periods, confidence=0.95)
            
            self.confidence_intervals = pd.DataFrame({
                'date': future_dates,
                'value': y_forecast,
                'lower': lower,
                'upper': upper
            })
        except Exception as e:
            logger.warning(f"Could not generate confidence intervals: {str(e)}")
            self.confidence_intervals = None
        
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
            
        # Convert dates to pandas datetime
        dates_pd = pd.to_datetime(dates)
        
        # Sort dates
        dates_sorted = dates_pd.sort_values()
        
        # Check which dates are in the historical data
        historical_mask = dates_sorted.isin(self.dates)
        historical_indices = historical_mask.nonzero()[0]
        
        # Check which dates are in the future
        future_mask = ~historical_mask
        future_indices = future_mask.nonzero()[0]
        
        # Initialize results array
        results = np.zeros(len(dates_sorted))
        
        # For historical dates, use the actual historical values
        for i in historical_indices:
            date = dates_sorted.iloc[i]
            idx = self.dates[self.dates == date].index[0]
            results[i] = self.y[idx]
        
        # For future dates, generate forecasts
        if len(future_indices) > 0:
            # Calculate periods from end of historical data to each future date
            future_dates = dates_sorted.iloc[future_indices]
            
            # Calculate time difference from last historical date to each future date
            time_diff = future_dates - self.dates.iloc[-1]
            
            # Convert to periods based on the model frequency
            if self.seasonal_periods is not None and len(self.seasonal_periods) > 0:
                # Use the first seasonal period to estimate frequency
                seasonal_period = self.seasonal_periods[0]
                
                # Different conversions based on seasonal period
                if seasonal_period == 12:  # Monthly data
                    periods = (time_diff.dt.days / 30.44).astype(int) + 1
                elif seasonal_period == 4:  # Quarterly data
                    periods = (time_diff.dt.days / 91.31).astype(int) + 1
                elif seasonal_period == 52:  # Weekly data
                    periods = (time_diff.dt.days / 7).astype(int) + 1
                else:
                    # Default: assume daily data
                    periods = time_diff.dt.days + 1
            else:
                # If no seasonal periods, assume yearly data
                periods = time_diff.dt.days / 365.25 + 1
                
            # Forecast for the maximum period needed
            max_period = int(np.ceil(max(periods)))
            forecast = self.model.forecast(steps=max_period)
            
            # Assign forecasts to the results array
            for i, period in zip(future_indices, periods):
                period_idx = min(int(period) - 1, len(forecast) - 1)
                if period_idx >= 0:
                    results[i] = forecast[period_idx]
                else:
                    # For periods that are out of forecast range, use the last forecast
                    results[i] = forecast[-1]
        
        # Reorder results to match original dates order
        reorder_idx = np.argsort(np.argsort(dates_pd))
        return results[reorder_idx]


class NBEATSForecaster(BaseForecaster):
    """
    Neural Basis Expansion Analysis for Time Series (N-BEATS) Forecaster.
    
    N-BEATS is a deep learning architecture based on backward and forward residual links and
    a deep stack of fully-connected layers. It uses a unique architecture that employs
    forecast-interpretation principles for decomposing time series into components like trend
    and seasonality.
    
    This model is highly flexible and effective for time series forecasting without the need
    to specify a model structure or feature engineering.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize N-BEATS model parameters from configuration."""
        # Lookback window size (how many time steps to use as input)
        self.lookback = self.config.get('lookback', 10)
        
        # Forecast horizon
        self.forecast_horizon = self.config.get('forecast_horizon', 5)
        
        # Stack parameters
        self.stack_types = self.config.get('stack_types', ['trend_block', 'seasonality_block'])
        self.nb_blocks_per_stack = self.config.get('nb_blocks_per_stack', 3)
        self.thetas_dim = self.config.get('thetas_dim', 4)
        self.hidden_layer_units = self.config.get('hidden_layer_units', 256)
        self.share_weights_in_stack = self.config.get('share_weights_in_stack', False)
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 128)
        self.max_epochs = self.config.get('max_epochs', 50)
        self.patience = self.config.get('patience', 10)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # Preprocessing parameters
        self.normalize = self.config.get('normalize', True)
        self.sequence_stride = self.config.get('sequence_stride', 1)
        
        # Ensure valid parameters
        if self.lookback < 4:
            logger.warning(f"Lookback window too small ({self.lookback}), setting to 4")
            self.lookback = 4
            
        if self.forecast_horizon < 1:
            logger.warning(f"Forecast horizon too small ({self.forecast_horizon}), setting to 1")
            self.forecast_horizon = 1
        
        self.model = None
        self.fitted = False
        
        # Try to import TensorFlow
        try:
            import tensorflow as tf
            from tensorflow.keras import Model
            self.tf = tf
            logger.info("TensorFlow successfully imported")
        except ImportError:
            logger.error("Failed to import TensorFlow. Please install it with: pip install tensorflow")
            raise ImportError("TensorFlow is required for the NBEATS forecaster")

    def _create_sequences(self, data: np.ndarray, lookback: int, forecast_horizon: int, stride: int = 1):
        """
        Create input sequences for the N-BEATS model.
        
        Args:
            data: Time series data as numpy array
            lookback: Number of time steps to use as input
            forecast_horizon: Number of time steps to predict
            stride: Stride between consecutive windows
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target sequences
        """
        X, y = [], []
        for i in range(0, len(data) - lookback - forecast_horizon + 1, stride):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback:i+lookback+forecast_horizon])
        return np.array(X), np.array(y)

    def _build_nbeats_model(self, lookback: int, forecast_horizon: int):
        """
        Build the N-BEATS model architecture.
        
        Args:
            lookback: Input sequence length
            forecast_horizon: Forecast horizon
            
        Returns:
            Compiled TensorFlow model
        """
        # Check if TensorFlow is imported
        if not hasattr(self, 'tf'):
            try:
                import tensorflow as tf
                self.tf = tf
            except ImportError:
                raise ImportError("TensorFlow is required for the NBEATS forecaster")
        
        # Define input layer
        inputs = self.tf.keras.layers.Input(shape=(lookback, 1))
        
        # Initial residual is the input
        backcast = inputs
        forecast = None
        residuals = inputs
        
        # Build stacks based on stack types
        for stack_id, stack_type in enumerate(self.stack_types):
            for block_id in range(self.nb_blocks_per_stack):
                # Check if weights should be shared
                if self.share_weights_in_stack and block_id > 0:
                    # Reuse the layers from the first block in this stack
                    backcast_layer = backcast_layers[stack_id]
                    forecast_layer = forecast_layers[stack_id]
                else:
                    # Create new Dense layers for this block
                    # Create layers for this block
                    if stack_type == 'generic_block':
                        # Generic fully connected block
                        block_layers = []
                        for _ in range(4):  # 4 fully connected layers
                            block_layers.append(self.tf.keras.layers.Dense(
                                self.hidden_layer_units, 
                                activation='relu'
                            ))
                        
                        # Project to backcast and forecast
                        backcast_layer = self.tf.keras.layers.Dense(lookback)
                        forecast_layer = self.tf.keras.layers.Dense(forecast_horizon)
                        
                        # Apply the block
                        x = self.tf.keras.layers.Flatten()(residuals)
                        for layer in block_layers:
                            x = layer(x)
                        
                        backcast_output = backcast_layer(x)
                        backcast_output = self.tf.keras.layers.Reshape((lookback, 1))(backcast_output)
                        
                        forecast_output = forecast_layer(x)
                        forecast_output = self.tf.keras.layers.Reshape((forecast_horizon, 1))(forecast_output)
                        
                    elif stack_type in ['trend_block', 'seasonality_block']:
                        # Specialized block for trend or seasonality
                        # First, apply fully connected layers
                        block_layers = []
                        for _ in range(4):  # 4 fully connected layers
                            block_layers.append(self.tf.keras.layers.Dense(
                                self.hidden_layer_units, 
                                activation='relu'
                            ))
                        
                        # Then project to theta parameters
                        theta_layer = self.tf.keras.layers.Dense(self.thetas_dim)
                        
                        # Apply the block
                        x = self.tf.keras.layers.Flatten()(residuals)
                        for layer in block_layers:
                            x = layer(x)
                        
                        thetas = theta_layer(x)
                        
                        # Create basis functions
                        if stack_type == 'trend_block':
                            # Polynomial basis for trend
                            backcast_basis = self._get_trend_basis(lookback, self.thetas_dim)
                            forecast_basis = self._get_trend_basis(forecast_horizon, self.thetas_dim)
                        else:  # seasonality_block
                            # Fourier basis for seasonality
                            backcast_basis = self._get_seasonality_basis(lookback, self.thetas_dim)
                            forecast_basis = self._get_seasonality_basis(forecast_horizon, self.thetas_dim)
                        
                        # Calculate backcast and forecast using dot product with basis
                        backcast_output = self.tf.keras.layers.Dot(axes=1)([thetas, backcast_basis])
                        backcast_output = self.tf.keras.layers.Reshape((lookback, 1))(backcast_output)
                        
                        forecast_output = self.tf.keras.layers.Dot(axes=1)([thetas, forecast_basis])
                        forecast_output = self.tf.keras.layers.Reshape((forecast_horizon, 1))(forecast_output)
                    
                    else:
                        raise ValueError(f"Unknown stack type: {stack_type}")
                
                # Update residuals
                residuals = self.tf.keras.layers.Subtract()([residuals, backcast_output])
                
                # Update forecast
                if forecast is None:
                    forecast = forecast_output
                else:
                    forecast = self.tf.keras.layers.Add()([forecast, forecast_output])
                
                # Save layers for weight sharing
                if block_id == 0 and self.share_weights_in_stack:
                    if not hasattr(self, 'backcast_layers'):
                        self.backcast_layers = {}
                        self.forecast_layers = {}
                    self.backcast_layers[stack_id] = backcast_layer
                    self.forecast_layers[stack_id] = forecast_layer
        
        # Create model
        model = self.tf.keras.Model(inputs=inputs, outputs=forecast)
        
        # Compile model
        model.compile(
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.tf.keras.losses.MeanSquaredError()
        )
        
        return model
    
    def _get_trend_basis(self, size, thetas_dim):
        """
        Generate polynomial basis for trend component.
        
        Args:
            size: Length of the basis vectors
            thetas_dim: Number of basis vectors
            
        Returns:
            Tensorflow constant with basis vectors
        """
        import tensorflow as tf
        
        # Create polynomial basis (1, x, x^2, ..., x^(thetas_dim-1))
        x = np.arange(size) / size
        basis = np.power(x[:, np.newaxis], np.arange(thetas_dim))
        
        # Normalize
        basis = basis / np.sqrt(np.sum(basis ** 2, axis=0, keepdims=True))
        
        return tf.constant(basis, dtype=tf.float32)
    
    def _get_seasonality_basis(self, size, thetas_dim):
        """
        Generate Fourier basis for seasonality component.
        
        Args:
            size: Length of the basis vectors
            thetas_dim: Number of basis vectors (should be even for sin/cos pairs)
            
        Returns:
            Tensorflow constant with basis vectors
        """
        import tensorflow as tf
        
        # Ensure even number for sin/cos pairs
        thetas_dim = thetas_dim + 1 if thetas_dim % 2 == 1 else thetas_dim
        
        # Create Fourier basis (sin(2πx), cos(2πx), sin(4πx), cos(4πx), ...)
        x = np.arange(size) / size
        harmonics = np.arange(1, thetas_dim // 2 + 1)
        
        basis = np.zeros((size, thetas_dim))
        for i, h in enumerate(harmonics):
            basis[:, 2*i] = np.sin(2 * np.pi * h * x)
            basis[:, 2*i+1] = np.cos(2 * np.pi * h * x)
        
        # Normalize
        basis = basis / np.sqrt(np.sum(basis ** 2, axis=0, keepdims=True))
        
        return tf.constant(basis, dtype=tf.float32)

    def fit(self, data: pd.DataFrame) -> 'NBEATSForecaster':
        """
        Fit the N-BEATS model to historical data.
        
        Args:
            data: DataFrame containing historical data with 'date' and 'value' columns
                
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Store historical data
        self.history_df = data.copy()
        
        # Extract dates and values
        self.dates = pd.to_datetime(data['date'])
        values = data['value'].values
        
        # Ensure data is sorted by date
        if not self.dates.equals(self.dates.sort_values()):
            logger.warning("Data not sorted by date. Sorting automatically.")
            sort_idx = self.dates.argsort()
            values = values[sort_idx]
            self.dates = self.dates.iloc[sort_idx]
        
        # Store original values
        self.original_values = values
        
        # Normalize data if needed
        if self.normalize:
            self.scale = np.max(np.abs(values))
            if self.scale == 0:
                self.scale = 1.0  # Prevent division by zero
            values = values / self.scale
        else:
            self.scale = 1.0
        
        # Store normalized values
        self.normalized_values = values
        
        # Ensure lookback and forecast horizon are appropriate for the data length
        if len(values) < self.lookback + self.forecast_horizon:
            raise ValueError(f"Not enough data points ({len(values)}) for lookback ({self.lookback}) + forecast_horizon ({self.forecast_horizon})")
        
        # Create sequences for training
        X, y = self._create_sequences(
            values, 
            self.lookback, 
            self.forecast_horizon,
            self.sequence_stride
        )
        
        # Reshape for NBEATS input (samples, time steps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        y = y.reshape((y.shape[0], y.shape[1], 1))
        
        # Build and train the model
        try:
            # Create the model
            logger.info("Building N-BEATS model...")
            self.model = self._build_nbeats_model(self.lookback, self.forecast_horizon)
            
            # Early stopping callback
            early_stopping = self.tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True
            )
            
            # Train the model
            logger.info("Training N-BEATS model...")
            history = self.model.fit(
                X, y,
                epochs=self.max_epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Store training history
            self.train_history = history.history
            
            logger.info("N-BEATS model trained successfully")
            self.fitted = True
            
        except Exception as e:
            logger.error(f"Error during N-BEATS model training: {str(e)}")
            self.fitted = False
            raise
            
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate a forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
                
        Returns:
            DataFrame with forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # For N-BEATS, we need to forecast iteratively if periods > forecast_horizon
        num_iterations = (periods + self.forecast_horizon - 1) // self.forecast_horizon
        
        # Initialize forecast array with the last values from the original data
        forecast_input = self.normalized_values[-self.lookback:]
        all_forecasts = []
        
        # Iteratively generate forecasts
        for i in range(num_iterations):
            # Prepare input
            model_input = forecast_input[-self.lookback:].reshape(1, self.lookback, 1)
            
            # Generate forecast
            forecast_chunk = self.model.predict(model_input, verbose=0)
            forecast_chunk = forecast_chunk.reshape(-1)
            
            # Store forecast
            all_forecasts.append(forecast_chunk)
            
            # Update input for next iteration
            forecast_input = np.append(forecast_input, forecast_chunk)
        
        # Concatenate all forecasts and take only the requested number of periods
        y_forecast = np.concatenate(all_forecasts)[:periods]
        
        # Denormalize
        y_forecast = y_forecast * self.scale
        
        # Generate future dates
        last_date = self.dates.iloc[-1]
        
        # Create date sequence based on frequency
        if frequency == 'Y':
            future_dates = pd.date_range(
                start=pd.Timestamp(last_date.year + 1, 1, 1),
                periods=periods,
                freq='YS'  # Year start
            )
        elif frequency == 'Q':
            # Find next quarter start
            last_quarter = pd.Timestamp(last_date).to_period('Q')
            next_quarter_start = (last_quarter + 1).to_timestamp()
            
            future_dates = pd.date_range(
                start=next_quarter_start,
                periods=periods,
                freq='QS'  # Quarter start
            )
        elif frequency == 'M':
            # Find next month start
            last_month = pd.Timestamp(last_date).to_period('M')
            next_month_start = (last_month + 1).to_timestamp()
            
            future_dates = pd.date_range(
                start=next_month_start,
                periods=periods,
                freq='MS'  # Month start
            )
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'value': y_forecast
        })
        
        # Store forecast result
        self.forecast_result = forecast_df
        self.forecast_dates = future_dates
        
        # Generate simple confidence intervals based on training error
        if hasattr(self, 'train_history') and 'val_loss' in self.train_history:
            # Use validation loss (MSE) to estimate prediction standard deviation
            std_dev = np.sqrt(self.train_history['val_loss'][-1]) * self.scale
            
            # Create confidence intervals (assuming normal distribution)
            lower = y_forecast - 1.96 * std_dev  # 95% confidence
            upper = y_forecast + 1.96 * std_dev
            
            self.confidence_intervals = pd.DataFrame({
                'date': future_dates,
                'value': y_forecast,
                'lower': lower,
                'upper': upper
            })
        else:
            # Simple heuristic if no training history available
            lower = y_forecast * 0.9
            upper = y_forecast * 1.1
            
            self.confidence_intervals = pd.DataFrame({
                'date': future_dates,
                'value': y_forecast,
                'lower': lower,
                'upper': upper
            })
        
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
            
        # Convert dates to pandas datetime
        dates_pd = pd.to_datetime(dates)
        
        # Sort dates
        dates_sorted = dates_pd.sort_values()
        
        # Check which dates are in the historical data
        historical_mask = dates_sorted.isin(self.dates)
        historical_indices = historical_mask.nonzero()[0]
        
        # Check which dates are in the future
        future_mask = ~historical_mask
        future_indices = future_mask.nonzero()[0]
        
        # Initialize results array
        results = np.zeros(len(dates_sorted))
        
        # For historical dates, use the actual historical values
        for i in historical_indices:
            date = dates_sorted.iloc[i]
            idx = self.dates[self.dates == date].index[0]
            results[i] = self.original_values[idx]
        
        # For future dates, generate forecasts
        if len(future_indices) > 0:
            # Calculate periods from end of historical data to each future date
            future_dates = dates_sorted.iloc[future_indices]
            
            # Calculate time difference from last historical date to each future date
            time_diff = future_dates - self.dates.iloc[-1]
            
            # Convert to periods based on the data frequency
            # Default: assume yearly data (most common for general forecasting)
            periods = (time_diff.dt.days / 365.25).astype(int) + 1
            
            # Forecast for the maximum number of periods needed
            max_period = int(np.ceil(max(periods)))
            
            # Initialize forecast input with the last values from normalized data
            forecast_input = self.normalized_values[-self.lookback:]
            all_forecasts = []
            
            # Number of iterations needed
            num_iterations = (max_period + self.forecast_horizon - 1) // self.forecast_horizon
            
            # Iteratively generate forecasts
            for i in range(num_iterations):
                # Prepare input
                model_input = forecast_input[-self.lookback:].reshape(1, self.lookback, 1)
                
                # Generate forecast
                forecast_chunk = self.model.predict(model_input, verbose=0)
                forecast_chunk = forecast_chunk.reshape(-1)
                
                # Store forecast
                all_forecasts.append(forecast_chunk)
                
                # Update input for next iteration
                forecast_input = np.append(forecast_input, forecast_chunk)
            
            # Concatenate all forecasts and take only the requested number of periods
            y_forecast = np.concatenate(all_forecasts)[:max_period]
            
            # Denormalize
            y_forecast = y_forecast * self.scale
            
            # Assign forecasts to the results array
            for i, period in zip(future_indices, periods):
                period_idx = min(int(period) - 1, len(y_forecast) - 1)
                if period_idx >= 0:
                    results[i] = y_forecast[period_idx]
                else:
                    # For periods that are out of forecast range, use the last forecast
                    results[i] = y_forecast[-1]
        
        # Reorder results to match original dates order
        reorder_idx = np.argsort(np.argsort(dates_pd))
        return results[reorder_idx]


class HybridETSARIMAForecaster(BaseForecaster):
    """
    Hybrid ETS-ARIMA Forecaster.
    
    This model combines Exponential Smoothing (ETS) and ARIMA to leverage the strengths of
    both approaches. ETS is effective at capturing trend and seasonality, while ARIMA is
    good at capturing complex autocorrelation patterns.
    
    The hybrid approach works by:
    1. Fitting an ETS model to the original time series
    2. Computing residuals (observed - ETS prediction)
    3. Fitting an ARIMA model to the residuals
    4. Final forecast = ETS forecast + ARIMA forecast of residuals
    
    This hybrid approach often outperforms individual models, especially for time series
    with both trend/seasonality and complex autocorrelation structures.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Hybrid ETS-ARIMA model parameters from configuration."""
        # ETS parameters
        self.ets_trend = self.config.get('ets_trend', 'add')  # Options: None, 'add', 'mul'
        self.ets_damped_trend = self.config.get('ets_damped_trend', False)
        self.ets_seasonal = self.config.get('ets_seasonal', 'add')  # Options: None, 'add', 'mul'
        self.ets_seasonal_periods = self.config.get('ets_seasonal_periods', None)
        
        # ARIMA parameters
        self.arima_order = self.config.get('arima_order', None)  # (p, d, q)
        self.arima_seasonal_order = self.config.get('arima_seasonal_order', None)  # (P, D, Q, s)
        
        # Auto parameter selection
        self.auto_arima = self.config.get('auto_arima', True)
        self.auto_ets = self.config.get('auto_ets', True)
        
        # Component weights
        self.ets_weight = self.config.get('ets_weight', 0.5)
        self.arima_weight = self.config.get('arima_weight', 0.5)
        
        # Initialize models
        self.ets_model = None
        self.arima_model = None
        self.fitted = False

    def fit(self, data: pd.DataFrame) -> 'HybridETSARIMAForecaster':
        """
        Fit the Hybrid ETS-ARIMA model to historical data.
        
        Args:
            data: DataFrame containing historical data with 'date' and 'value' columns
                
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Store historical data
        self.history_df = data.copy()
        
        # Extract dates and values
        self.dates = pd.to_datetime(data['date'])
        self.values = data['value'].values
        
        # Ensure data is sorted by date
        if not self.dates.equals(self.dates.sort_values()):
            logger.warning("Data not sorted by date. Sorting automatically.")
            sort_idx = self.dates.argsort()
            self.values = self.values[sort_idx]
            self.dates = self.dates.iloc[sort_idx]
        
        # Detect seasonality if not specified
        if self.ets_seasonal_periods is None:
            # Get frequency information from dates
            try:
                freq = pd.infer_freq(self.dates)
                
                if freq is None:
                    # Try to infer based on date differences
                    date_diffs = np.diff(self.dates)
                    if np.all(date_diffs == date_diffs[0]):
                        # Evenly spaced data
                        # Get average days between observations
                        days_diff = np.mean([d.days for d in date_diffs])
                        
                        if 28 <= days_diff <= 31:
                            # Monthly data
                            self.ets_seasonal_periods = 12
                        elif 90 <= days_diff <= 92:
                            # Quarterly data
                            self.ets_seasonal_periods = 4
                        elif 7 <= days_diff <= 7.5:
                            # Weekly data
                            self.ets_seasonal_periods = 52
                        else:
                            # Default or no seasonality
                            self.ets_seasonal_periods = None
                    else:
                        # Not evenly spaced data
                        self.ets_seasonal_periods = None
                        self.ets_seasonal = None
                else:
                    # Use inferred frequency
                    if freq.startswith('M'):
                        # Monthly data
                        self.ets_seasonal_periods = 12
                    elif freq.startswith('Q'):
                        # Quarterly data
                        self.ets_seasonal_periods = 4
                    elif freq.startswith('W'):
                        # Weekly data
                        self.ets_seasonal_periods = 52
                    elif freq.startswith('D'):
                        # Daily data
                        # Check if we have enough data for yearly seasonality
                        if len(self.dates) >= 2 * 365:
                            self.ets_seasonal_periods = 365
                        else:
                            # Just weekly seasonality
                            self.ets_seasonal_periods = 7
                    else:
                        # Default or unknown frequency
                        self.ets_seasonal_periods = None
                        self.ets_seasonal = None
            except Exception as e:
                logger.warning(f"Could not infer seasonality: {str(e)}")
                self.ets_seasonal_periods = None
                self.ets_seasonal = None
        
        # Import required libraries
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import pmdarima as pm
        except ImportError:
            logger.error("Failed to import required libraries. Please install them with: pip install statsmodels pmdarima")
            raise ImportError("The statsmodels and pmdarima packages are required for this forecaster")
        
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import pmdarima as pm
            
            # Step 1: Fit ETS model
            if self.auto_ets:
                # Try different ETS models and select the best one using AIC
                logger.info("Automatically selecting best ETS model...")
                best_aic = np.inf
                best_model = None
                best_params = {}
                
                # Candidate parameters
                trend_options = [None, 'add', 'mul'] if len(self.values) > 10 else [None, 'add']
                damped_options = [True, False]
                seasonal_options = [None, 'add', 'mul'] if self.ets_seasonal_periods else [None]
                
                # Try different combinations
                for trend in trend_options:
                    for damped in damped_options:
                        # Skip damped if no trend
                        if damped and trend is None:
                            continue
                            
                        for seasonal in seasonal_options:
                            try:
                                # Create model
                                model = ExponentialSmoothing(
                                    self.values,
                                    trend=trend,
                                    damped_trend=damped if trend else False,
                                    seasonal=seasonal,
                                    seasonal_periods=self.ets_seasonal_periods if seasonal else None
                                )
                                
                                # Fit model
                                fitted_model = model.fit(optimized=True)
                                
                                # Check AIC
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_model = fitted_model
                                    best_params = {
                                        'trend': trend,
                                        'damped_trend': damped if trend else False,
                                        'seasonal': seasonal,
                                        'seasonal_periods': self.ets_seasonal_periods if seasonal else None
                                    }
                            except Exception as e:
                                logger.debug(f"ETS model failed with {trend}, {damped}, {seasonal}: {str(e)}")
                                continue
                
                if best_model is None:
                    raise ValueError("Could not find a suitable ETS model")
                    
                self.ets_model = best_model
                logger.info(f"Selected ETS model with parameters: {best_params}")
                
                # Update instance parameters
                self.ets_trend = best_params['trend']
                self.ets_damped_trend = best_params['damped_trend']
                self.ets_seasonal = best_params['seasonal']
                self.ets_seasonal_periods = best_params['seasonal_periods']
                
            else:
                # Use the specified ETS parameters
                logger.info("Fitting ETS model with specified parameters...")
                
                # Create and fit model
                model = ExponentialSmoothing(
                    self.values,
                    trend=self.ets_trend,
                    damped_trend=self.ets_damped_trend if self.ets_trend else False,
                    seasonal=self.ets_seasonal,
                    seasonal_periods=self.ets_seasonal_periods if self.ets_seasonal else None
                )
                
                self.ets_model = model.fit(optimized=True)
                
                logger.info("ETS model fitted successfully")
            
            # Step 2: Compute residuals
            ets_predictions = self.ets_model.fittedvalues
            self.residuals = self.values - ets_predictions
            
            # Step 3: Fit ARIMA model to residuals
            if self.auto_arima:
                # Use pmdarima (auto_arima) to find the best ARIMA model
                logger.info("Automatically selecting best ARIMA model for residuals...")
                
                # Set max order limits to prevent long fitting time
                max_p = min(5, len(self.residuals) // 10)
                max_q = min(5, len(self.residuals) // 10)
                
                # Create auto_arima model
                self.arima_model = pm.auto_arima(
                    self.residuals,
                    start_p=0, max_p=max_p,
                    start_q=0, max_q=max_q,
                    d=None,  # Let the function determine the best value
                    seasonal=True,
                    m=self.ets_seasonal_periods if self.ets_seasonal_periods and self.ets_seasonal_periods <= 12 else 1,
                    information_criterion='aic',
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                
                # Extract order and seasonal_order
                self.arima_order = self.arima_model.order
                self.arima_seasonal_order = self.arima_model.seasonal_order
                
                logger.info(f"Selected ARIMA model for residuals: SARIMA{self.arima_order}{self.arima_seasonal_order}")
                
            else:
                # Use the specified ARIMA parameters
                logger.info("Fitting ARIMA model to residuals with specified parameters...")
                
                # Set default orders if not provided
                if self.arima_order is None:
                    self.arima_order = (1, 0, 1)
                    
                if self.arima_seasonal_order is None and self.ets_seasonal_periods:
                    # Only add seasonal component if frequency is manageable
                    if self.ets_seasonal_periods <= 12:
                        self.arima_seasonal_order = (1, 0, 1, self.ets_seasonal_periods)
                    else:
                        self.arima_seasonal_order = (0, 0, 0, 0)
                
                # Create and fit model
                arima = SARIMAX(
                    self.residuals,
                    order=self.arima_order,
                    seasonal_order=self.arima_seasonal_order if self.arima_seasonal_order else (0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                self.arima_model = arima.fit(disp=False)
                
                logger.info("ARIMA model for residuals fitted successfully")
            
            # Store fitted components
            self.ets_fitted = self.ets_model.fittedvalues
            self.arima_fitted = getattr(self.arima_model, 'fittedvalues', 
                                        self.arima_model.predict_in_sample())
            
            # Pad arima_fitted if lengths don't match (ARIMA can lose some initial values)
            if len(self.arima_fitted) < len(self.values):
                padding = np.zeros(len(self.values) - len(self.arima_fitted))
                self.arima_fitted = np.concatenate([padding, self.arima_fitted])
            
            # Store combined forecast
            self.combined_fitted = (self.ets_weight * self.ets_fitted + 
                                    self.arima_weight * self.arima_fitted)
            
            # Calculate goodness of fit metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            self.mse = mean_squared_error(self.values, self.combined_fitted)
            self.mae = mean_absolute_error(self.values, self.combined_fitted)
            
            logger.info(f"Hybrid model fitted successfully. MSE: {self.mse:.4f}, MAE: {self.mae:.4f}")
            
            self.fitted = True
            
        except Exception as e:
            logger.error(f"Error during Hybrid ETS-ARIMA model fitting: {str(e)}")
            self.fitted = False
            raise
            
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate a forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
                
        Returns:
            DataFrame with forecasted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
            
        try:
            # Step 1: Generate ETS forecast
            ets_forecast = self.ets_model.forecast(periods)
            
            # Step 2: Generate ARIMA forecast
            # For pmdarima vs statsmodels handling
            if hasattr(self.arima_model, 'predict'):
                # statsmodels approach
                arima_forecast = self.arima_model.forecast(periods)
            else:
                # pmdarima approach
                arima_forecast = self.arima_model.predict(n_periods=periods)
            
            # Step 3: Combine forecasts
            combined_forecast = (self.ets_weight * ets_forecast + 
                                self.arima_weight * arima_forecast)
            
            # Generate future dates
            last_date = self.dates.iloc[-1]
            
            # Create date sequence based on frequency
            if frequency == 'Y':
                future_dates = pd.date_range(
                    start=pd.Timestamp(last_date.year + 1, 1, 1),
                    periods=periods,
                    freq='YS'  # Year start
                )
            elif frequency == 'Q':
                # Find next quarter start
                last_quarter = pd.Timestamp(last_date).to_period('Q')
                next_quarter_start = (last_quarter + 1).to_timestamp()
                
                future_dates = pd.date_range(
                    start=next_quarter_start,
                    periods=periods,
                    freq='QS'  # Quarter start
                )
            elif frequency == 'M':
                # Find next month start
                last_month = pd.Timestamp(last_date).to_period('M')
                next_month_start = (last_month + 1).to_timestamp()
                
                future_dates = pd.date_range(
                    start=next_month_start,
                    periods=periods,
                    freq='MS'  # Month start
                )
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'value': combined_forecast
            })
            
            # Store forecast result
            self.forecast_result = forecast_df
            self.forecast_dates = future_dates
            
            # Generate confidence intervals
            # For ETS model
            if hasattr(self.ets_model, 'simulate'):
                # Use simulation to generate confidence intervals
                num_simulations = 1000
                simulations = np.zeros((num_simulations, periods))
                
                for i in range(num_simulations):
                    simulations[i, :] = self.ets_model.simulate(periods, anchor='end')
                
                # Calculate confidence intervals
                lower_ets = np.percentile(simulations, 2.5, axis=0)
                upper_ets = np.percentile(simulations, 97.5, axis=0)
                
                # For ARIMA model
                if hasattr(self.arima_model, 'get_forecast'):
                    # statsmodels approach
                    arima_forecast_obj = self.arima_model.get_forecast(periods)
                    arima_ci = arima_forecast_obj.conf_int(alpha=0.05)
                    lower_arima = arima_ci.iloc[:, 0].values
                    upper_arima = arima_ci.iloc[:, 1].values
                else:
                    # Simplified approach for pmdarima
                    std_dev = np.sqrt(self.mse)
                    lower_arima = arima_forecast - 1.96 * std_dev
                    upper_arima = arima_forecast + 1.96 * std_dev
                
                # Combine intervals using the same weights
                lower = self.ets_weight * lower_ets + self.arima_weight * lower_arima
                upper = self.ets_weight * upper_ets + self.arima_weight * upper_arima
                
            else:
                # Simplified approach
                std_dev = np.sqrt(self.mse)
                lower = combined_forecast - 1.96 * std_dev
                upper = combined_forecast + 1.96 * std_dev
            
            # Create confidence intervals DataFrame
            self.confidence_intervals = pd.DataFrame({
                'date': future_dates,
                'value': combined_forecast,
                'lower': lower,
                'upper': upper
            })
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"Error during Hybrid ETS-ARIMA forecasting: {str(e)}")
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
            
        # Convert dates to pandas datetime
        dates_pd = pd.to_datetime(dates)
        
        # Sort dates
        dates_sorted = dates_pd.sort_values()
        
        # Check which dates are in the historical data
        historical_mask = dates_sorted.isin(self.dates)
        historical_indices = historical_mask.nonzero()[0]
        
        # Check which dates are in the future
        future_mask = ~historical_mask
        future_indices = future_mask.nonzero()[0]
        
        # Initialize results array
        results = np.zeros(len(dates_sorted))
        
        # For historical dates, use the actual historical values
        for i in historical_indices:
            date = dates_sorted.iloc[i]
            idx = self.dates[self.dates == date].index[0]
            results[i] = self.values[idx]
        
        # For future dates, generate forecasts
        if len(future_indices) > 0:
            # Calculate periods from end of historical data to each future date
            future_dates = dates_sorted.iloc[future_indices]
            
            # Calculate time difference from last historical date to each future date
            time_diff = future_dates - self.dates.iloc[-1]
            
            # Convert to periods based on the model's seasonal frequency
            if self.ets_seasonal_periods is not None:
                if self.ets_seasonal_periods == 12:  # Monthly data
                    periods = (time_diff.dt.days / 30.44).astype(int) + 1
                elif self.ets_seasonal_periods == 4:  # Quarterly data
                    periods = (time_diff.dt.days / 91.31).astype(int) + 1
                elif self.ets_seasonal_periods == 52:  # Weekly data
                    periods = (time_diff.dt.days / 7).astype(int) + 1
                else:
                    # Default: assume yearly data
                    periods = (time_diff.dt.days / 365.25).astype(int) + 1
            else:
                # Default: assume yearly data
                periods = (time_diff.dt.days / 365.25).astype(int) + 1
            
            # Forecast for the maximum number of periods needed
            max_period = int(np.ceil(max(periods)))
            
            # Generate forecasts
            ets_forecast = self.ets_model.forecast(max_period)
            
            # For pmdarima vs statsmodels handling
            if hasattr(self.arima_model, 'predict'):
                # statsmodels approach
                arima_forecast = self.arima_model.forecast(max_period)
            else:
                # pmdarima approach
                arima_forecast = self.arima_model.predict(n_periods=max_period)
            
            # Combine forecasts
            combined_forecast = (self.ets_weight * ets_forecast + 
                                self.arima_weight * arima_forecast)
            
            # Assign forecasts to the results array
            for i, period in zip(future_indices, periods):
                period_idx = min(int(period) - 1, len(combined_forecast) - 1)
                if period_idx >= 0:
                    results[i] = combined_forecast[period_idx]
                else:
                    # For periods that are out of forecast range, use the last value
                    results[i] = combined_forecast[-1]
        
        # Reorder results to match original dates order
        reorder_idx = np.argsort(np.argsort(dates_pd))
        return results[reorder_idx]