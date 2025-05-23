"""
Machine Learning Forecasting Module - Advanced ML-based time series forecasting

This module provides implementations of advanced machine learning forecasting methods,
including Prophet, XGBoost, and LSTM neural networks for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import machine learning models
# Prophet requires a separate installation
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet package not found. ProphetForecaster will not be available.")

# XGBoost requires a separate installation
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost package not found. XGBoostForecaster will not be available.")

# TensorFlow/Keras for LSTM requires a separate installation
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM as KerasLSTM
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow package not found. LSTMForecaster will not be available.")

# Scikit-learn for preprocessing and metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """
    Prophet Forecasting Model.
    
    Facebook Prophet is designed for forecasting time series data with strong seasonal effects
    and multiple seasons of historical data. It provides automatic handling of missing data,
    trend shifts, and outliers, making it ideal for business forecasting scenarios.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Prophet-specific parameters from configuration"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet package is required. Install with 'pip install prophet'")
        
        # Growth model (linear or logistic)
        self.growth = self.config.get('growth', 'linear')
        
        # Capacity for logistic growth
        if self.growth == 'logistic':
            self.cap = self.config.get('cap', None)
            self.floor = self.config.get('floor', 0)
        
        # Seasonality parameters
        self.yearly_seasonality = self.config.get('yearly_seasonality', 'auto')
        self.weekly_seasonality = self.config.get('weekly_seasonality', 'auto')
        self.daily_seasonality = self.config.get('daily_seasonality', 'auto')
        self.seasonality_mode = self.config.get('seasonality_mode', 'additive')
        
        # Additional seasonalities
        self.add_seasonalities = self.config.get('add_seasonalities', [])
        
        # Changepoint parameters
        self.changepoint_prior_scale = self.config.get('changepoint_prior_scale', 0.05)
        self.changepoint_range = self.config.get('changepoint_range', 0.8)
        self.changepoints = self.config.get('changepoints', None)
        
        # Holidays
        self.holidays = self.config.get('holidays', None)
        
        # Uncertainty parameters
        self.interval_width = self.config.get('interval_width', 0.8)
        self.mcmc_samples = self.config.get('mcmc_samples', 0)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Internal storage
        self.model = None
        self.history = None
        self.future = None
        self.forecast_result = None
        self.forecast_data = None
    
    def fit(self, data: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit the Prophet model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Store original data
        self.history = data.copy()
        
        # Format data for Prophet
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(data['date']),
            'y': data['value']
        })
        
        # Handle capacity for logistic growth
        if self.growth == 'logistic':
            if self.cap is not None:
                prophet_data['cap'] = self.cap
            else:
                # Estimate cap as 3x the maximum observed value
                max_value = prophet_data['y'].max()
                prophet_data['cap'] = max_value * 3
                
            if self.floor is not None:
                prophet_data['floor'] = self.floor
        
        # Create model
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            changepoint_range=self.changepoint_range,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width
        )
        
        # Add custom seasonalities
        for seasonality in self.add_seasonalities:
            self.model.add_seasonality(**seasonality)
        
        # Add holidays if provided
        if self.holidays is not None:
            self.model.add_country_holidays(country_name=self.holidays)
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model.fit(prophet_data)
        
        logger.info(f"Fitted Prophet model")
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
        
        # Map frequency to Prophet frequency
        freq_map = {
            'Y': 'AS',  # Annual start
            'Q': 'QS',  # Quarterly start
            'M': 'MS',  # Monthly start
            'W': 'W-MON',  # Weekly
            'D': 'D'    # Daily
        }
        prophet_freq = freq_map.get(frequency, 'AS')
        
        # Create future dataframe for prediction
        last_date = self.history['date'].max()
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        # Use Prophet's make_future_dataframe
        self.future = self.model.make_future_dataframe(
            periods=periods,
            freq=prophet_freq,
            include_history=False
        )
        
        # Add capacity for logistic growth
        if self.growth == 'logistic':
            if self.cap is not None:
                self.future['cap'] = self.cap
            else:
                max_value = self.history['value'].max()
                self.future['cap'] = max_value * 3
                
            if self.floor is not None:
                self.future['floor'] = self.floor
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.forecast_data = self.model.predict(self.future)
        
        # Extract forecasted values
        forecast_values = self.forecast_data['yhat'].values
        forecast_dates = self.forecast_data['ds'].values
        
        # Apply minimum value if needed
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        # Create result DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Extract confidence intervals
        lower_bound = self.forecast_data['yhat_lower'].values
        upper_bound = self.forecast_data['yhat_upper'].values
        
        if self.ensure_minimum:
            lower_bound = np.maximum(lower_bound, self.minimum_value)
        
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
        
        # Create dataframe for prediction
        future = pd.DataFrame({
            'ds': pd.to_datetime(dates)
        })
        
        # Add capacity for logistic growth
        if self.growth == 'logistic':
            if self.cap is not None:
                future['cap'] = self.cap
            else:
                max_value = self.history['value'].max()
                future['cap'] = max_value * 3
                
            if self.floor is not None:
                future['floor'] = self.floor
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            forecast = self.model.predict(future)
        
        # Extract forecasted values
        forecast_values = forecast['yhat'].values
        
        # Apply minimum value if needed
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        return forecast_values
    
    def plot_components(self) -> Any:
        """
        Plot the forecast components (trend, seasonality, etc.)
        
        Returns:
            Matplotlib figure with forecast components
        """
        if not self.fitted or self.forecast_data is None:
            raise ValueError("Model must be fitted and forecast generated before plotting components")
        
        try:
            from prophet.plot import plot_components
            fig = plot_components(self.model, self.forecast_data)
            return fig
        except ImportError:
            logger.warning("Could not import prophet.plot. Plotting unavailable.")
            return None


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost-based Forecasting Model.
    
    XGBoost is a powerful gradient boosting framework that can be used for time series 
    forecasting with engineered features. This implementation transforms time series data 
    into a supervised learning problem with lagged features and temporal attributes.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize XGBoost-specific parameters from configuration"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost package is required. Install with 'pip install xgboost'")
        
        # XGBoost parameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 3)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.objective = self.config.get('objective', 'reg:squarederror')
        self.booster = self.config.get('booster', 'gbtree')
        self.subsample = self.config.get('subsample', 0.8)
        self.colsample_bytree = self.config.get('colsample_bytree', 0.8)
        self.random_state = self.config.get('random_state', 42)
        
        # Feature engineering parameters
        self.lag_size = self.config.get('lag_size', 3)  # Number of lag features
        self.include_date_features = self.config.get('include_date_features', True)
        self.validation_split = self.config.get('validation_split', 0.2)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Internal storage
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_names = None
    
    def _create_features(self, data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Create features for XGBoost model.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            is_training: Whether this is for training or prediction
            
        Returns:
            DataFrame with features and target (if training)
        """
        df = data.copy()
        
        # Make sure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create time-based features if requested
        if self.include_date_features:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year_day'] = df['date'].dt.dayofyear
            
            # Create normalized cyclical features for seasonal components
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        # Create trend feature
        min_date = df['date'].min()
        df['trend'] = [(d - min_date).days / 365.25 for d in df['date']]
        
        # Create lag features
        for i in range(1, self.lag_size + 1):
            if is_training:
                df[f'lag_{i}'] = df['value'].shift(i)
            else:
                # For prediction, we need to get lag values from history
                if i <= len(self.history):
                    df[f'lag_{i}'] = np.concatenate([[self.history['value'].iloc[-i]], np.zeros(len(df) - 1)])
                else:
                    df[f'lag_{i}'] = 0
        
        # Drop rows with NAs from lagged features if training
        if is_training:
            df = df.dropna()
            
            # Store feature names for later
            self.feature_names = [col for col in df.columns if col not in ['date', 'value']]
        
        return df
    
    def fit(self, data: pd.DataFrame) -> 'XGBoostForecaster':
        """
        Fit the XGBoost model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Make sure we have enough data
        if len(data) <= self.lag_size:
            raise ValueError(f"Not enough data points. Need more than lag_size ({self.lag_size}).")
        
        # Store historical data
        self.history = data.copy()
        
        # Create features
        features_df = self._create_features(data)
        
        # Create training data
        X = features_df[self.feature_names]
        y = features_df['value']
        
        # Scale features if we have numerical ones
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=self.validation_split, random_state=self.random_state, shuffle=False
        )
        
        # Create evaluation set
        eval_set = [(X_val, y_val)]
        
        # Create and fit XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            booster=self.booster,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric='rmse',
            verbosity=0  # Set verbosity to 0 for no output
        )
        
        self.model.fit(X_train, y_train, eval_set=eval_set)
        
        logger.info(f"Fitted XGBoost model with {self.n_estimators} trees")
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
        
        # Get last date from historical data
        last_date = self.history['date'].iloc[-1]
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        # Generate future dates
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
        
        # Create initial future dataframe
        future_df = pd.DataFrame({
            'date': forecast_dates,
            'value': np.zeros(periods)  # Dummy values, will be replaced
        })
        
        # Generate forecast step by step (because we need previous predictions for lag features)
        forecast_values = []
        current_history = self.history.copy()
        
        for i in range(periods):
            # Create a one-step future dataframe
            one_step = future_df.iloc[i:i+1].copy()
            
            # Create features for this step
            features_df = self._create_features(
                pd.concat([current_history, one_step.iloc[:i]]) if i > 0 else current_history, 
                is_training=False
            ).tail(1)
            
            # Extract features (only the last row is needed)
            X = features_df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                prediction = max(prediction, self.minimum_value)
            
            # Store prediction
            forecast_values.append(prediction)
            
            # Update history for next step's lag features
            if i < periods - 1:  # No need to update for the last step
                current_step = pd.DataFrame({
                    'date': [forecast_dates[i]],
                    'value': [prediction]
                })
                current_history = pd.concat([current_history, current_step])
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals (using prediction_stds from XGBoost)
        # This is a simplified approach as XGBoost doesn't provide direct confidence intervals
        lower_bound = []
        upper_bound = []
        
        # Calculate RMSE on training data as a proxy for prediction error
        X_train = self._create_features(self.history)[self.feature_names]
        X_train_scaled = self.scaler.transform(X_train)
        y_train = self._create_features(self.history)['value']
        y_pred_train = self.model.predict(X_train_scaled)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        
        # Use RMSE to construct confidence intervals, increasing width over time
        for i, value in enumerate(forecast_values):
            margin = rmse * (1 + 0.1 * i)  # Increase uncertainty over time
            lower = value - 1.96 * margin  # ~95% confidence interval
            upper = value + 1.96 * margin
            
            if self.ensure_minimum:
                lower = max(lower, self.minimum_value)
                
            lower_bound.append(lower)
            upper_bound.append(upper)
        
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
        
        # Create future dataframe
        future_df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'value': np.zeros(len(dates))  # Dummy values, will be replaced
        })
        
        # Generate predictions one at a time
        predictions = []
        current_history = self.history.copy()
        
        for i in range(len(dates)):
            # Create a one-step future dataframe
            one_step = future_df.iloc[i:i+1].copy()
            
            # Check if this date is before the last date in history
            current_date = one_step['date'].iloc[0]
            last_history_date = current_history['date'].max()
            
            if current_date <= last_history_date:
                # For dates in the past, use actual values from history
                historical_value = current_history[current_history['date'] == current_date]['value']
                if len(historical_value) > 0:
                    predictions.append(historical_value.iloc[0])
                    continue
            
            # Create features for this step
            features_df = self._create_features(
                pd.concat([current_history, future_df.iloc[:i]]).sort_values('date') if i > 0 else current_history, 
                is_training=False
            ).tail(1)
            
            # Extract features
            X = features_df[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                prediction = max(prediction, self.minimum_value)
            
            # Store prediction
            predictions.append(prediction)
            
            # Update history for next step
            if i < len(dates) - 1:  # No need to update for the last date
                current_step = pd.DataFrame({
                    'date': [current_date],
                    'value': [prediction]
                })
                current_history = pd.concat([current_history, current_step])
                current_history = current_history.sort_values('date')
        
        return np.array(predictions)


class LSTMForecaster(BaseForecaster):
    """
    LSTM Neural Network Forecasting Model.
    
    LSTM (Long Short-Term Memory) networks are a type of recurrent neural network
    capable of learning long-term dependencies, making them suitable for time series
    forecasting. This implementation includes sequence-based input formatting and
    proper normalization.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize LSTM-specific parameters from configuration"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow package is required. Install with 'pip install tensorflow'")
        
        # LSTM parameters
        self.sequence_length = self.config.get('sequence_length', 5)
        self.units = self.config.get('units', [50, 25])
        self.dropout = self.config.get('dropout', 0.2)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.validation_split = self.config.get('validation_split', 0.2)
        self.patience = self.config.get('patience', 10)  # For early stopping
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.shuffle = self.config.get('shuffle', False)  # Typically False for time series
        self.random_state = self.config.get('random_state', 42)
        
        # Scaling method: 'minmax' or 'standard'
        self.scaling = self.config.get('scaling', 'minmax')
        
        # Whether to include date-based features
        self.include_date_features = self.config.get('include_date_features', True)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Internal storage
        self.model = None
        self.history = None
        self.scaler_x = None
        self.scaler_y = None
        self.features = None
        self.feature_columns = None  # Store feature column names
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence data for LSTM.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple with X sequences and y values
        """
        # Get value column and feature columns
        values = data['value'].values
        
        # Create date-based features if requested
        if self.include_date_features:
            # Convert date to datetime if needed
            if data['date'].dtype == 'object':
                data['date'] = pd.to_datetime(data['date'])
                
            # Create date features
            data['year'] = data['date'].dt.year
            data['month'] = data['date'].dt.month
            data['quarter'] = data['date'].dt.quarter
            
            # Create cyclical features for month and quarter
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['quarter_sin'] = np.sin(2 * np.pi * data['quarter'] / 4)
            data['quarter_cos'] = np.cos(2 * np.pi * data['quarter'] / 4)
            
            # Create trend feature
            min_date = data['date'].min()
            data['trend'] = [(d - min_date).days / 365.25 for d in data['date']]
            
            # Store feature columns
            self.feature_columns = [
                'value', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos', 'trend'
            ]
            
            # Extract features
            features = data[self.feature_columns].values
        else:
            # Only use the value column as a feature
            self.feature_columns = ['value']
            features = values.reshape(-1, 1)
        
        # Scale features
        if self.scaling == 'minmax':
            self.scaler_x = MinMaxScaler()
        else:
            self.scaler_x = StandardScaler()
            
        self.scaler_y = MinMaxScaler()
            
        # Fit scalers
        features_scaled = self.scaler_x.fit_transform(features)
        values_scaled = self.scaler_y.fit_transform(values.reshape(-1, 1))
        
        # Store scaled features for future use
        self.features = features_scaled
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(features_scaled) - self.sequence_length):
            # Create sequence of features
            sequence = features_scaled[i:i + self.sequence_length]
            X.append(sequence)
            
            # Target is the next value after the sequence
            target = values_scaled[i + self.sequence_length]
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple) -> tf.keras.Model:
        """
        Build the LSTM model.
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Compiled Keras model
        """
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        
        # Create sequential model
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.units):
            if i == 0:
                # First layer needs input shape
                model.add(KerasLSTM(
                    units=units,
                    return_sequences=(i < len(self.units) - 1),
                    input_shape=input_shape
                ))
            else:
                # Subsequent layers
                model.add(KerasLSTM(
                    units=units,
                    return_sequences=(i < len(self.units) - 1)
                ))
            
            # Add dropout if configured
            if self.dropout > 0:
                model.add(Dropout(self.dropout))
        
        # Output layer (single value prediction)
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def fit(self, data: pd.DataFrame) -> 'LSTMForecaster':
        """
        Fit the LSTM model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Make sure we have enough data
        if len(data) <= self.sequence_length:
            raise ValueError(f"Not enough data points. Need more than sequence_length ({self.sequence_length}).")
        
        # Store historical data
        self.history = data.copy()
        
        # Create sequences
        X, y = self._create_sequences(data)
        
        # Build model
        self.model = self._build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        # Fit model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[early_stopping],
                shuffle=self.shuffle,
                verbose=0
            )
        
        logger.info(f"Fitted LSTM model with {self.units} units and {self.sequence_length} sequence length")
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
        
        # Get last date from historical data
        last_date = self.history['date'].iloc[-1]
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        # Generate future dates
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
        
        # Generate forecast step by step
        forecast_values = []
        
        # Start with the last sequence from history
        current_features = self.features[-self.sequence_length:].copy()
        
        # Generate predictions one at a time
        for i in range(periods):
            # Take the most recent sequence
            sequence = current_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            # Make prediction
            prediction_scaled = self.model.predict(sequence, verbose=0)[0][0]
            
            # Inverse transform the prediction
            prediction = self.scaler_y.inverse_transform(
                np.array([[prediction_scaled]])
            )[0][0]
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                prediction = max(prediction, self.minimum_value)
            
            # Store prediction
            forecast_values.append(prediction)
            
            # Create features for the next step
            if i < periods - 1:  # No need to update for the last step
                # Create date for the current prediction
                current_date = forecast_dates[i]
                
                # Create new feature array
                if self.include_date_features:
                    # Create date features
                    month = current_date.month
                    quarter = (month - 1) // 3 + 1
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    quarter_sin = np.sin(2 * np.pi * quarter / 4)
                    quarter_cos = np.cos(2 * np.pi * quarter / 4)
                    
                    # Calculate trend
                    min_date = self.history['date'].min()
                    if isinstance(min_date, str):
                        min_date = pd.to_datetime(min_date)
                    trend = (current_date - min_date).days / 365.25
                    
                    # Create feature array
                    new_features = np.array([[
                        prediction_scaled,  # Value is the prediction we just made
                        month_sin, month_cos, quarter_sin, quarter_cos, trend
                    ]])
                else:
                    # Only use the value as a feature
                    new_features = np.array([[prediction_scaled]])
                
                # Scale the new features
                new_scaled = new_features
                
                # Update the current features with the new prediction
                current_features = np.vstack([current_features, new_scaled])
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals
        # For LSTM, we use a simple approach based on the validation error
        # A more sophisticated approach would involve bootstrapping or MC Dropout
        
        # Calculate MSE on a validation set
        X, y = self._create_sequences(self.history)
        split_idx = int(len(X) * (1 - self.validation_split))
        X_val, y_val = X[split_idx:], y[split_idx:]
        
        # Get predictions
        y_pred_scaled = self.model.predict(X_val, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_true = self.scaler_y.inverse_transform(y_val)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Create confidence intervals using RMSE
        lower_bound = []
        upper_bound = []
        
        for i, value in enumerate(forecast_values):
            # Increase uncertainty over time
            margin = rmse * (1 + 0.2 * i)
            lower = value - 1.96 * margin  # ~95% confidence interval
            upper = value + 1.96 * margin
            
            if self.ensure_minimum:
                lower = max(lower, self.minimum_value)
                
            lower_bound.append(lower)
            upper_bound.append(upper)
        
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
        
        # Convert dates to datetime
        dates = pd.to_datetime(dates)
        
        # Check if any dates are in the past
        last_history_date = pd.to_datetime(self.history['date'].max())
        future_mask = dates > last_history_date
        
        # For dates in the past, get values from history
        predictions = np.zeros(len(dates))
        for i, date in enumerate(dates):
            if date <= last_history_date:
                # Try to find exact match in history
                historical_value = self.history[self.history['date'] == date]['value']
                if len(historical_value) > 0:
                    predictions[i] = historical_value.iloc[0]
                else:
                    # Find closest date in history
                    closest_date = self.history['date'].iloc[
                        np.argmin(np.abs((self.history['date'] - date).dt.total_seconds()))
                    ]
                    predictions[i] = self.history[self.history['date'] == closest_date]['value'].iloc[0]
        
        # Generate forecasts for future dates
        if any(future_mask):
            # Generate future dates sequence starting from the last history date
            future_dates = dates[future_mask]
            
            # Generate forecasts incrementally
            future_forecasts = []
            
            # Start with the last sequence from history
            current_features = self.features[-self.sequence_length:].copy()
            
            # Keep track of predicted dates
            predicted_dates = [last_history_date]
            predicted_values_scaled = []
            
            # Predict until we reach all required dates
            while any(d > predicted_dates[-1] for d in future_dates):
                # Determine the next date to predict
                if len(predicted_dates) == 1:
                    # First prediction after history
                    freq = pd.infer_freq(self.history['date'])
                    next_date = pd.date_range(
                        start=predicted_dates[-1],
                        periods=2,
                        freq=freq
                    )[1]
                else:
                    # Use the same frequency as between the last two predictions
                    freq = pd.infer_freq(pd.DatetimeIndex(predicted_dates[-2:]))
                    next_date = pd.date_range(
                        start=predicted_dates[-1],
                        periods=2,
                        freq=freq
                    )[1]
                
                # Make prediction for this date
                sequence = current_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
                prediction_scaled = self.model.predict(sequence, verbose=0)[0][0]
                predicted_values_scaled.append(prediction_scaled)
                
                # Add the predicted date
                predicted_dates.append(next_date)
                
                # Update features for next prediction
                if self.include_date_features:
                    # Create date features
                    month = next_date.month
                    quarter = (month - 1) // 3 + 1
                    month_sin = np.sin(2 * np.pi * month / 12)
                    month_cos = np.cos(2 * np.pi * month / 12)
                    quarter_sin = np.sin(2 * np.pi * quarter / 4)
                    quarter_cos = np.cos(2 * np.pi * quarter / 4)
                    
                    # Calculate trend
                    min_date = pd.to_datetime(self.history['date'].min())
                    trend = (next_date - min_date).days / 365.25
                    
                    # Create feature array
                    new_features = np.array([[
                        prediction_scaled,  # Value is the prediction we just made
                        month_sin, month_cos, quarter_sin, quarter_cos, trend
                    ]])
                else:
                    # Only use the value as a feature
                    new_features = np.array([[prediction_scaled]])
                
                # Update the current features with the new prediction
                current_features = np.vstack([current_features, new_features])
                
                # Check if we've generated enough predictions
                if all(d <= next_date for d in future_dates):
                    break
            
            # Convert predicted values to original scale
            predicted_values = self.scaler_y.inverse_transform(
                np.array(predicted_values_scaled).reshape(-1, 1)
            ).flatten()
            
            # Create DataFrame with predictions
            predictions_df = pd.DataFrame({
                'date': predicted_dates[1:],  # Skip the last history date
                'value': predicted_values
            })
            
            # For each future date, find the closest predicted date
            for i, date in enumerate(dates):
                if future_mask[i]:
                    closest_idx = np.argmin(np.abs(
                        (predictions_df['date'] - date).dt.total_seconds()
                    ))
                    predictions[i] = predictions_df['value'].iloc[closest_idx]
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                predictions = np.maximum(predictions, self.minimum_value)
        
        return predictions