"""
Advanced Ensemble Forecasting Module - Sophisticated ensemble techniques

This module provides implementations of more sophisticated ensemble forecasting methods
including stacking, boosting, and bagging approaches for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import logging
from datetime import datetime, timedelta
import warnings

from src.global_forecasting.base_forecaster import BaseForecaster
from src.global_forecasting.ensemble import WeightedEnsembleForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StackingEnsembleForecaster(BaseForecaster):
    """
    Stacking Ensemble Forecaster.
    
    Stacking is an ensemble technique that trains a meta-model to combine the predictions
    of multiple base models. This implementation trains base forecasters on the full dataset,
    then uses their predictions on a validation set to train a meta-model, which makes the
    final prediction by combining the outputs of the base models.
    
    This approach can capture complex relationships between different forecasting methods
    and is particularly effective when different models perform well under different conditions.
    """

    def _initialize_parameters(self) -> None:
        """Initialize stacking ensemble parameters from configuration"""
        # List of base forecasters to use in the ensemble
        self.base_forecasters = []
        
        # The meta-model forecaster
        self.meta_forecaster = None
        self.meta_forecaster_type = self.config.get('meta_forecaster_type', 'LinearRegression')
        self.meta_forecaster_params = self.config.get('meta_forecaster_params', {})
        
        # Validation strategy
        self.validation_strategy = self.config.get('validation_strategy', 'holdout')
        self.validation_size = self.config.get('validation_size', 0.2)
        self.cross_val_folds = self.config.get('cross_val_folds', 3)
        
        # Data preprocessing
        self.normalize_inputs = self.config.get('normalize_inputs', True)
        self.include_original_features = self.config.get('include_original_features', False)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store data
        self.history_df = None
        self.validation_df = None
        self.test_df = None
        
        # Store feature importance
        self.feature_importance = None
        
        # Store individual forecasts
        self.individual_forecasts = {}
        
        # Store scalers
        self.scaler_X = None
        self.scaler_y = None
        
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
                            self.add_base_forecaster(forecaster)
                            logger.info(f"Added {forecaster.name} as base forecaster from config")
                        except Exception as e:
                            logger.error(f"Error adding base forecaster from config: {str(e)}")
    
    def add_base_forecaster(self, forecaster: BaseForecaster) -> 'StackingEnsembleForecaster':
        """
        Add a base forecaster to the ensemble.
        
        Args:
            forecaster: A forecaster to add to the ensemble
            
        Returns:
            Self for method chaining
        """
        if not isinstance(forecaster, BaseForecaster):
            raise TypeError("forecaster must be an instance of BaseForecaster")
        
        # Add the forecaster to the list
        self.base_forecasters.append(forecaster)
        logger.info(f"Added {forecaster.name} as base forecaster")
        
        return self
    
    def clear_base_forecasters(self) -> None:
        """Remove all base forecasters from the ensemble"""
        self.base_forecasters = []
        self.fitted = False
        logger.info("Cleared all base forecasters")
    
    def _create_meta_features(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create meta-features from base forecasters' predictions.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
            is_training: Whether this is for training (will fit base forecasters if True)
            
        Returns:
            Tuple of (X, y) where X is meta-features and y is target values
        """
        # Extract dates and values
        dates = pd.to_datetime(data['date'])
        values = data['value'].values
        
        # Initialize meta-features array
        meta_features = np.zeros((len(dates), len(self.base_forecasters)))
        
        # For each base forecaster
        for i, forecaster in enumerate(self.base_forecasters):
            try:
                if is_training:
                    # Fit and predict in-sample
                    fit_data = data.copy()
                    forecaster.fit(fit_data)
                    predictions = forecaster._predict_for_dates(dates.values)
                else:
                    # Just predict using already fitted forecaster
                    if not forecaster.fitted:
                        logger.warning(f"{forecaster.name} not fitted. Skipping.")
                        continue
                    predictions = forecaster._predict_for_dates(dates.values)
                
                # Add predictions as a feature
                meta_features[:, i] = predictions
                
            except Exception as e:
                logger.error(f"Error creating meta-features for {forecaster.name}: {str(e)}")
                # Fill with mean of values
                meta_features[:, i] = np.mean(values)
        
        # Optionally add original features (time-based features)
        if self.include_original_features:
            # Extract year and convert to continuous numeric value
            years = dates.dt.year.values + dates.dt.month.values / 12
            years = years.reshape(-1, 1)
            
            # Concatenate with meta-features
            meta_features = np.hstack([years, meta_features])
        
        # Normalize meta-features if requested
        if self.normalize_inputs and is_training:
            from sklearn.preprocessing import StandardScaler
            self.scaler_X = StandardScaler()
            meta_features = self.scaler_X.fit_transform(meta_features)
            
            # Also normalize target values
            self.scaler_y = StandardScaler()
            values_norm = self.scaler_y.fit_transform(values.reshape(-1, 1)).flatten()
            
            return meta_features, values_norm
        elif self.normalize_inputs and not is_training:
            if self.scaler_X is not None and self.scaler_y is not None:
                meta_features = self.scaler_X.transform(meta_features)
                return meta_features, values
            else:
                logger.warning("Scalers not available. Using unnormalized features.")
                return meta_features, values
        else:
            return meta_features, values
    
    def _create_meta_forecaster(self):
        """
        Create the meta-forecaster based on configuration.
        
        Returns:
            Initialized meta-forecaster
        """
        if self.meta_forecaster_type == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**self.meta_forecaster_params)
        
        elif self.meta_forecaster_type == 'Ridge':
            from sklearn.linear_model import Ridge
            return Ridge(**self.meta_forecaster_params)
        
        elif self.meta_forecaster_type == 'Lasso':
            from sklearn.linear_model import Lasso
            return Lasso(**self.meta_forecaster_params)
        
        elif self.meta_forecaster_type == 'ElasticNet':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(**self.meta_forecaster_params)
        
        elif self.meta_forecaster_type == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**self.meta_forecaster_params)
        
        elif self.meta_forecaster_type == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**self.meta_forecaster_params)
        
        elif self.meta_forecaster_type == 'SVR':
            from sklearn.svm import SVR
            return SVR(**self.meta_forecaster_params)
        
        else:
            logger.warning(f"Unknown meta-forecaster type: {self.meta_forecaster_type}. Using LinearRegression.")
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
    
    def _get_feature_importance(self):
        """
        Get feature importance from meta-forecaster if available.
        
        Returns:
            Dictionary mapping forecaster names to importance scores
        """
        feature_importance = {}
        
        # Check if meta-forecaster supports feature importance
        if hasattr(self.meta_forecaster, 'coef_'):
            # Linear models have coef_ attribute
            coefficients = self.meta_forecaster.coef_
            
            # If include_original_features is True, skip the first feature
            start_idx = 1 if self.include_original_features else 0
            
            for i, forecaster in enumerate(self.base_forecasters):
                if i + start_idx < len(coefficients):
                    feature_importance[forecaster.name] = abs(coefficients[i + start_idx])
        
        elif hasattr(self.meta_forecaster, 'feature_importances_'):
            # Tree-based models have feature_importances_ attribute
            importance = self.meta_forecaster.feature_importances_
            
            # If include_original_features is True, skip the first feature
            start_idx = 1 if self.include_original_features else 0
            
            for i, forecaster in enumerate(self.base_forecasters):
                if i + start_idx < len(importance):
                    feature_importance[forecaster.name] = importance[i + start_idx]
        
        return feature_importance

    def fit(self, data: pd.DataFrame) -> 'StackingEnsembleForecaster':
        """
        Fit the stacking ensemble using the provided data.
        
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
        self.history_df = data.copy()
        
        # Check if we have any base forecasters
        if not self.base_forecasters:
            logger.warning("No base forecasters in the ensemble. Add base forecasters before fitting.")
            self.fitted = False
            return self
        
        # Split data into training and validation sets
        if self.validation_strategy == 'holdout':
            # Use a holdout set of most recent data
            n_samples = len(data)
            n_val = int(n_samples * self.validation_size)
            
            if n_val < 1:
                logger.warning("Validation size too small. Using 1 sample.")
                n_val = 1
            
            # Split data
            train_df = data.iloc[:-n_val].copy()
            val_df = data.iloc[-n_val:].copy()
            
            logger.info(f"Split data: {len(train_df)} training samples, {len(val_df)} validation samples")
            
            # Fit base forecasters on training data
            for forecaster in self.base_forecasters:
                try:
                    forecaster.fit(train_df)
                except Exception as e:
                    logger.error(f"Error fitting {forecaster.name}: {str(e)}")
            
            # Create meta-features for validation set
            X_val, y_val = self._create_meta_features(val_df, is_training=False)
            
            # Create and fit meta-forecaster
            self.meta_forecaster = self._create_meta_forecaster()
            self.meta_forecaster.fit(X_val, y_val)
            
            # Store validation data
            self.validation_df = val_df
            
        elif self.validation_strategy == 'cross-val':
            # Use cross-validation
            # Divide data into k chunks
            n_samples = len(data)
            fold_size = n_samples // self.cross_val_folds
            
            # Collect meta-features for all chunks
            all_X = []
            all_y = []
            
            for fold in range(self.cross_val_folds):
                # Define validation fold
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size if fold < self.cross_val_folds - 1 else n_samples
                
                # Split data
                train_indices = list(range(0, val_start)) + list(range(val_end, n_samples))
                val_indices = list(range(val_start, val_end))
                
                train_df = data.iloc[train_indices].copy()
                val_df = data.iloc[val_indices].copy()
                
                # Fit base forecasters on training data
                for forecaster in self.base_forecasters:
                    try:
                        forecaster.fit(train_df)
                    except Exception as e:
                        logger.error(f"Error fitting {forecaster.name} on fold {fold}: {str(e)}")
                
                # Create meta-features for validation fold
                X_val, y_val = self._create_meta_features(val_df, is_training=False)
                
                # Store meta-features
                all_X.append(X_val)
                all_y.append(y_val)
            
            # Combine all meta-features
            X_combined = np.vstack(all_X)
            y_combined = np.concatenate(all_y)
            
            # Create and fit meta-forecaster on combined meta-features
            self.meta_forecaster = self._create_meta_forecaster()
            self.meta_forecaster.fit(X_combined, y_combined)
            
            # Refit base forecasters on full data
            for forecaster in self.base_forecasters:
                try:
                    forecaster.fit(data)
                except Exception as e:
                    logger.error(f"Error refitting {forecaster.name} on full data: {str(e)}")
        
        else:
            logger.warning(f"Unknown validation strategy: {self.validation_strategy}. Using holdout.")
            # Default to simple full data fitting
            
            # Fit base forecasters on all data
            for forecaster in self.base_forecasters:
                try:
                    forecaster.fit(data)
                except Exception as e:
                    logger.error(f"Error fitting {forecaster.name}: {str(e)}")
            
            # Create meta-features for all data
            X_train, y_train = self._create_meta_features(data, is_training=True)
            
            # Create and fit meta-forecaster
            self.meta_forecaster = self._create_meta_forecaster()
            self.meta_forecaster.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = self._get_feature_importance()
        
        # Log feature importance
        if self.feature_importance:
            logger.info("Meta-model feature importance:")
            for name, importance in self.feature_importance.items():
                logger.info(f"  {name}: {importance:.4f}")
        
        self.fitted = True
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.fitted:
            raise ValueError("Stacking ensemble must be fitted before forecasting")
        
        if not self.base_forecasters:
            raise ValueError("No base forecasters in the ensemble")
        
        if self.meta_forecaster is None:
            raise ValueError("Meta-forecaster not available")
        
        # Generate future dates
        last_date = self.history_df['date'].iloc[-1]
        forecast_dates = []
        
        if frequency == 'Y':
            # Yearly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(years=i))
        elif frequency == 'Q':
            # Quarterly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(months=3*i))
        elif frequency == 'M':
            # Monthly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(months=i))
        else:
            # Default to yearly
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(years=i))
        
        # Generate forecasts from all base models
        forecasts = {}
        meta_features = np.zeros((periods, len(self.base_forecasters)))
        
        # Clear previous individual forecasts
        self.individual_forecasts = {}
        
        for i, forecaster in enumerate(self.base_forecasters):
            try:
                # Generate forecast
                forecast_df = forecaster.forecast(periods, frequency)
                
                # Store forecast
                self.individual_forecasts[forecaster.name] = forecast_df.copy()
                
                # Extract values and store in meta-features array
                meta_features[:, i] = forecast_df['value'].values
                
                # Store in forecasts dictionary
                forecasts[forecaster.name] = forecast_df['value'].values
                
            except Exception as e:
                logger.error(f"Error forecasting with {forecaster.name}: {str(e)}")
                # Fill with last value from history
                last_value = self.history_df['value'].iloc[-1]
                meta_features[:, i] = last_value
                forecasts[forecaster.name] = np.array([last_value] * periods)
        
        # Add original features if configured
        if self.include_original_features:
            # Extract year and convert to continuous numeric value
            years = np.array([d.year + d.month / 12 for d in forecast_dates])
            years = years.reshape(-1, 1)
            
            # Concatenate with meta-features
            meta_features = np.hstack([years, meta_features])
        
        # Normalize meta-features if needed
        if self.normalize_inputs and self.scaler_X is not None:
            meta_features = self.scaler_X.transform(meta_features)
        
        # Generate ensemble forecast using meta-forecaster
        ensemble_values = self.meta_forecaster.predict(meta_features)
        
        # Denormalize if needed
        if self.normalize_inputs and self.scaler_y is not None:
            ensemble_values = self.scaler_y.inverse_transform(ensemble_values.reshape(-1, 1)).flatten()
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            ensemble_values = np.maximum(ensemble_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': ensemble_values
        })
        
        # Create confidence intervals
        # Use the variance of base model predictions as a proxy for uncertainty
        forecasts_array = np.array([values for values in forecasts.values()])
        std_devs = np.std(forecasts_array, axis=0)
        
        # Fix: Use 1.96 standard deviations for proper 95% CI
        lower_bounds = ensemble_values - 1.96 * std_devs
        upper_bounds = ensemble_values + 1.96 * std_devs
        
        # Ensure lower bounds are non-negative if minimum value is 0
        if self.ensure_minimum and self.minimum_value >= 0:
            lower_bounds = np.maximum(lower_bounds, self.minimum_value)
        
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
        historical_mask = dates_sorted.isin(self.history_df['date'])
        historical_indices = historical_mask.nonzero()[0]
        
        # Check which dates are in the future
        future_mask = ~historical_mask
        future_indices = future_mask.nonzero()[0]
        
        # Initialize results array
        results = np.zeros(len(dates_sorted))
        
        # For historical dates, use the actual historical values
        for i in historical_indices:
            date = dates_sorted.iloc[i]
            idx = self.history_df[self.history_df['date'] == date].index
            if len(idx) > 0:
                results[i] = self.history_df.loc[idx[0], 'value']
            else:
                # This shouldn't happen given the mask check
                results[i] = np.nan
        
        # For future dates, generate predictions
        if len(future_indices) > 0:
            # Create temporary DataFrame for future dates
            future_dates = dates_sorted.iloc[future_indices]
            temp_df = pd.DataFrame({'date': future_dates})
            
            # Generate meta-features from base forecasters
            meta_features = np.zeros((len(future_dates), len(self.base_forecasters)))
            
            for i, forecaster in enumerate(self.base_forecasters):
                try:
                    if forecaster.fitted:
                        # Generate predictions for future dates
                        predictions = forecaster._predict_for_dates(future_dates.values)
                        meta_features[:, i] = predictions
                    else:
                        logger.warning(f"{forecaster.name} not fitted. Using fallback.")
                        # Fill with last value from history
                        last_value = self.history_df['value'].iloc[-1]
                        meta_features[:, i] = last_value
                except Exception as e:
                    logger.error(f"Error predicting with {forecaster.name}: {str(e)}")
                    # Fill with last value from history
                    last_value = self.history_df['value'].iloc[-1]
                    meta_features[:, i] = last_value
            
            # Add original features if configured
            if self.include_original_features:
                # Extract year and convert to continuous numeric value
                years = np.array([d.year + d.month / 12 for d in future_dates])
                years = years.reshape(-1, 1)
                
                # Concatenate with meta-features
                meta_features = np.hstack([years, meta_features])
            
            # Normalize meta-features if needed
            if self.normalize_inputs and self.scaler_X is not None:
                meta_features = self.scaler_X.transform(meta_features)
            
            # Generate ensemble predictions using meta-forecaster
            predictions = self.meta_forecaster.predict(meta_features)
            
            # Denormalize if needed
            if self.normalize_inputs and self.scaler_y is not None:
                predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
            
            # Apply minimum value if configured
            if self.ensure_minimum:
                predictions = np.maximum(predictions, self.minimum_value)
            
            # Assign to results
            results[future_indices] = predictions
        
        # Reorder results to match original dates order
        reorder_idx = np.argsort(np.argsort(dates_pd))
        return results[reorder_idx]
    
    def plot_ensemble(self, figsize: Tuple[int, int] = (10, 6),
                     include_individual: bool = True,
                     include_history: bool = True,
                     include_importance: bool = True) -> Any:
        """
        Plot the ensemble forecast with individual forecasts.
        
        Args:
            figsize: Figure size tuple (width, height)
            include_individual: Whether to include individual forecasts
            include_history: Whether to include historical data
            include_importance: Whether to show feature importance in legend
            
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
            if include_history and self.history_df is not None:
                ax.plot(self.history_df['date'], self.history_df['value'], 'k-', linewidth=2, label='Historical')
            
            # Plot individual forecasts if requested
            if include_individual and self.individual_forecasts:
                for name, forecast_df in self.individual_forecasts.items():
                    # Get importance if available
                    importance = self.feature_importance.get(name, None) if self.feature_importance else None
                    
                    if include_importance and importance is not None:
                        label = f"{name} (imp={importance:.3f})"
                    else:
                        label = name
                        
                    ax.plot(forecast_df['date'], forecast_df['value'], '--', alpha=0.5, label=label)
            
            # Plot ensemble forecast (always)
            ax.plot(self.forecast_result['date'], self.forecast_result['value'], 
                   'b-', linewidth=3, label='Stacking Ensemble')
            
            # Plot confidence intervals if available
            if self.confidence_intervals is not None:
                ci = self.confidence_intervals
                ax.fill_between(ci['date'], ci['lower'], ci['upper'], 
                               color='b', alpha=0.1, label='Confidence Interval')
            
            ax.set_title("Stacking Ensemble Forecast")
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None


class BoostingEnsembleForecaster(BaseForecaster):
    """
    Boosting Ensemble Forecaster.
    
    Boosting is an ensemble technique that trains models sequentially, with each model
    focusing on the errors of the previous ones. This implementation adapts gradient
    boosting for time series forecasting by training a sequence of forecasters, where
    each one tries to correct the residuals of the combined previous forecasters.
    
    This approach can reduce bias and is particularly effective when base models have
    high bias (underfitting) but low variance.
    """

    def _initialize_parameters(self) -> None:
        """Initialize boosting ensemble parameters from configuration"""
        # List of base forecaster types to use in the ensemble
        self.base_forecaster_type = self.config.get('base_forecaster_type', 'MovingAverage')
        self.base_forecaster_params = self.config.get('base_forecaster_params', {})
        
        # Number of boosting iterations (number of base models)
        self.n_estimators = self.config.get('n_estimators', 5)
        
        # Learning rate (shrinkage)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        
        # Whether to use a fixed base forecaster type or a list of different types
        self.use_diverse_forecasters = self.config.get('use_diverse_forecasters', False)
        self.diverse_forecaster_types = self.config.get('diverse_forecaster_types', [])
        
        # Loss function to use
        self.loss = self.config.get('loss', 'squared_error')  # 'squared_error', 'absolute_error'
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Base models - will be populated during fitting
        self.models = []
        
        # Store data
        self.history_df = None
        
        # Store initial predictions and residuals history
        self.initial_prediction = None
        self.residuals_history = []
        
        # Store individual forecasts
        self.individual_forecasts = {}

    def _create_base_forecaster(self, iteration: int) -> BaseForecaster:
        """
        Create a new base forecaster for a given iteration.
        
        Args:
            iteration: The current iteration (0-based)
            
        Returns:
            A new base forecaster instance
        """
        from src.global_forecasting import create_forecaster
        
        if self.use_diverse_forecasters and self.diverse_forecaster_types:
            # Use a different forecaster type for each iteration (cycling if needed)
            index = iteration % len(self.diverse_forecaster_types)
            forecaster_type = self.diverse_forecaster_types[index]
            
            # Create a copy of params in case we need to modify them
            params = self.base_forecaster_params.copy()
            
            # If params has forecaster-specific settings, merge them
            forecaster_specific_key = f"{forecaster_type}_params"
            if forecaster_specific_key in self.base_forecaster_params:
                params.update(self.base_forecaster_params[forecaster_specific_key])
            
            # Create forecaster
            try:
                return create_forecaster(forecaster_type, params)
            except Exception as e:
                logger.error(f"Error creating {forecaster_type} forecaster: {str(e)}")
                # Fall back to default
                return create_forecaster(self.base_forecaster_type, self.base_forecaster_params)
        else:
            # Use the same type for all forecasters
            return create_forecaster(self.base_forecaster_type, self.base_forecaster_params)

    def fit(self, data: pd.DataFrame) -> 'BoostingEnsembleForecaster':
        """
        Fit the boosting ensemble using the provided data.
        
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
        self.history_df = data.copy()
        
        # Reset models list
        self.models = []
        self.residuals_history = []
        
        # Step 1: Initial prediction (mean of the target)
        y = data['value'].values
        self.initial_prediction = np.mean(y)
        logger.info(f"Initial prediction (mean): {self.initial_prediction:.4f}")
        
        # Current prediction starts with the initial prediction
        current_prediction = np.full_like(y, self.initial_prediction)
        
        # Step 2: Iteratively fit residual models
        for i in range(self.n_estimators):
            # Calculate residuals
            if self.loss == 'squared_error':
                residuals = y - current_prediction
            elif self.loss == 'absolute_error':
                # Fix: Use proper absolute error (with magnitude information)
                residuals = y - current_prediction  # Keep magnitude for proper boosting
            else:
                # Default to squared error
                residuals = y - current_prediction
            
            # Store residuals for later analysis
            self.residuals_history.append(residuals.copy())
            
            # Create residual dataset
            residual_data = data.copy()
            residual_data['value'] = residuals
            
            # Create and fit a new base forecaster
            model = self._create_base_forecaster(i)
            model.fit(residual_data)
            
            # Add to models list
            self.models.append(model)
            logger.info(f"Fitted model {i+1}/{self.n_estimators}: {model.name}")
            
            # Update current prediction with scaled contribution from new model
            model_contrib = model._predict_for_dates(data['date'].values)
            current_prediction += self.learning_rate * model_contrib
        
        self.fitted = True
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.fitted:
            raise ValueError("Boosting ensemble must be fitted before forecasting")
        
        if not self.models:
            raise ValueError("No models available in the ensemble")
        
        # Generate future dates
        last_date = self.history_df['date'].iloc[-1]
        forecast_dates = []
        
        if frequency == 'Y':
            # Yearly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(years=i))
        elif frequency == 'Q':
            # Quarterly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(months=3*i))
        elif frequency == 'M':
            # Monthly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(months=i))
        else:
            # Default to yearly
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(years=i))
        
        # Start with initial prediction
        ensemble_values = np.full(periods, self.initial_prediction)
        
        # Clear previous individual forecasts
        self.individual_forecasts = {}
        
        # Add contribution from each model
        for i, model in enumerate(self.models):
            try:
                # Generate forecast
                forecast_df = model.forecast(periods, frequency)
                
                # Store forecast for this component
                self.individual_forecasts[f"Model_{i+1}_{model.name}"] = forecast_df.copy()
                
                # Add scaled contribution to ensemble forecast
                ensemble_values += self.learning_rate * forecast_df['value'].values
                
            except Exception as e:
                logger.error(f"Error forecasting with model {i+1}: {str(e)}")
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            ensemble_values = np.maximum(ensemble_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': ensemble_values
        })
        
        # Generate confidence intervals
        # Use the residuals history to estimate prediction error
        if self.residuals_history:
            # Calculate standard deviation of residuals
            all_residuals = np.concatenate(self.residuals_history)
            residual_std = np.std(all_residuals)
            
            # Create confidence intervals (95% CI assuming normal distribution)
            lower_bounds = ensemble_values - 1.96 * residual_std
            upper_bounds = ensemble_values + 1.96 * residual_std
            
            # Ensure lower bounds are non-negative if minimum value is 0
            if self.ensure_minimum and self.minimum_value >= 0:
                lower_bounds = np.maximum(lower_bounds, self.minimum_value)
        else:
            # Fallback to simple bounds
            lower_bounds = ensemble_values * 0.9
            upper_bounds = ensemble_values * 1.1
        
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
        
        # Start with initial prediction
        predictions = np.full(len(dates_pd), self.initial_prediction)
        
        # Add contribution from each model
        for model in self.models:
            try:
                if model.fitted:
                    # Generate predictions
                    model_contrib = model._predict_for_dates(dates_pd.values)
                    predictions += self.learning_rate * model_contrib
                else:
                    logger.warning(f"{model.name} not fitted. Skipping.")
            except Exception as e:
                logger.error(f"Error predicting with {model.name}: {str(e)}")
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            predictions = np.maximum(predictions, self.minimum_value)
        
        return predictions
    
    def plot_ensemble(self, figsize: Tuple[int, int] = (10, 6),
                     include_individual: bool = True,
                     include_history: bool = True,
                     include_residuals: bool = False) -> Any:
        """
        Plot the ensemble forecast.
        
        Args:
            figsize: Figure size tuple (width, height)
            include_individual: Whether to include individual model contributions
            include_history: Whether to include historical data
            include_residuals: Whether to include a residuals plot
            
        Returns:
            Matplotlib figure object or None if plotting is not possible
        """
        if not self.fitted or self.forecast_result is None:
            logger.warning("Cannot plot: ensemble not fitted or no forecast available")
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            if include_residuals and self.residuals_history:
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.5), 
                                             gridspec_kw={'height_ratios': [3, 1]})
            else:
                # Create figure with just forecast plot
                fig, ax1 = plt.subplots(figsize=figsize)
            
            # Plot historical data if available and requested
            if include_history and self.history_df is not None:
                ax1.plot(self.history_df['date'], self.history_df['value'], 'k-', linewidth=2, label='Historical')
            
            # Plot individual model contributions if requested
            if include_individual and self.individual_forecasts and len(self.models) <= 5:
                # Only show individual models if there aren't too many
                for name, forecast_df in self.individual_forecasts.items():
                    ax1.plot(forecast_df['date'], forecast_df['value'], '--', alpha=0.3, label=name)
            
            # Plot ensemble forecast (always)
            ax1.plot(self.forecast_result['date'], self.forecast_result['value'], 
                   'b-', linewidth=3, label='Boosting Ensemble')
            
            # Plot confidence intervals if available
            if self.confidence_intervals is not None:
                ci = self.confidence_intervals
                ax1.fill_between(ci['date'], ci['lower'], ci['upper'], 
                               color='b', alpha=0.1, label='Confidence Interval')
            
            ax1.set_title("Boosting Ensemble Forecast")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot residuals if requested
            if include_residuals and self.residuals_history:
                # Combine all residuals
                all_residuals = np.concatenate(self.residuals_history)
                
                # Create bins for histogram
                bins = np.linspace(all_residuals.min(), all_residuals.max(), 30)
                
                # Plot histogram of residuals
                ax2.hist(all_residuals, bins=bins, alpha=0.7)
                ax2.set_title("Residuals Distribution")
                ax2.set_xlabel("Residual Value")
                ax2.set_ylabel("Frequency")
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None


class BaggingEnsembleForecaster(BaseForecaster):
    """
    Bagging Ensemble Forecaster.
    
    Bagging (Bootstrap Aggregating) is an ensemble technique that trains multiple models
    on random subsamples of the training data, then combines their predictions. This
    implementation adapts bagging for time series forecasting by creating bootstrapped
    samples of the original time series, fitting a forecaster to each sample, and then
    averaging the predictions.
    
    This approach can reduce variance and is particularly effective when base models have
    high variance (overfitting) but low bias.
    """

    def _initialize_parameters(self) -> None:
        """Initialize bagging ensemble parameters from configuration"""
        # List of base forecaster types to use in the ensemble
        self.base_forecaster_type = self.config.get('base_forecaster_type', 'ARIMA')
        self.base_forecaster_params = self.config.get('base_forecaster_params', {})
        
        # Number of bootstrap samples (number of base models)
        self.n_estimators = self.config.get('n_estimators', 10)
        
        # Bootstrap parameters
        self.subsample_size = self.config.get('subsample_size', 0.8)  # Ratio of data to use in each subsample
        self.bootstrap = self.config.get('bootstrap', True)  # Whether to sample with replacement
        
        # Combination method
        self.combination_method = self.config.get('combination_method', 'mean')  # 'mean', 'median', 'weighted'
        
        # Weights for weighted combination
        self.weights = self.config.get('weights', None)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Models - will be populated during fitting
        self.models = []
        
        # Store data
        self.history_df = None
        self.subsamples = []
        
        # Store individual forecasts
        self.individual_forecasts = {}
        
        # Store model errors
        self.model_errors = {}

    def _create_bootstrap_sample(self, data: pd.DataFrame, sample_idx: int) -> pd.DataFrame:
        """
        Create a bootstrap sample from the data.
        
        Args:
            data: Original DataFrame with 'date' and 'value' columns
            sample_idx: Index of the sample (for reproducibility)
            
        Returns:
            DataFrame with bootstrapped sample
        """
        n_samples = len(data)
        subsample_size = int(n_samples * self.subsample_size)
        
        # Set random seed for reproducibility
        np.random.seed(42 + sample_idx)
        
        if self.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n_samples, size=subsample_size, replace=True)
        else:
            # Sample without replacement
            indices = np.random.choice(n_samples, size=subsample_size, replace=False)
        
        # Sort indices to preserve temporal order
        indices = np.sort(indices)
        
        # Create bootstrapped sample
        bootstrap_sample = data.iloc[indices].copy()
        
        return bootstrap_sample

    def fit(self, data: pd.DataFrame) -> 'BaggingEnsembleForecaster':
        """
        Fit the bagging ensemble using the provided data.
        
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
        self.history_df = data.copy()
        
        # Reset models and subsamples
        self.models = []
        self.subsamples = []
        
        # Create bootstrap samples and fit models
        for i in range(self.n_estimators):
            try:
                # Create bootstrap sample
                bootstrap_sample = self._create_bootstrap_sample(data, i)
                self.subsamples.append(bootstrap_sample)
                
                # Create forecaster
                from src.global_forecasting import create_forecaster
                forecaster = create_forecaster(self.base_forecaster_type, self.base_forecaster_params)
                
                # Fit forecaster on bootstrap sample
                forecaster.fit(bootstrap_sample)
                
                # Add to models list
                self.models.append(forecaster)
                logger.info(f"Fitted model {i+1}/{self.n_estimators}: {forecaster.name}")
                
                # Calculate in-sample error
                predictions = forecaster._predict_for_dates(data['date'].values)
                errors = data['value'].values - predictions
                mse = np.mean(errors ** 2)
                self.model_errors[i] = mse
                
            except Exception as e:
                logger.error(f"Error fitting model {i+1}: {str(e)}")
        
        # If no models were successfully fitted, raise an error
        if not self.models:
            raise ValueError("No models could be fitted successfully")
        
        # Calculate weights for weighted combination
        if self.combination_method == 'weighted':
            if self.weights is None:
                # Calculate weights based on model errors
                errors = np.array([self.model_errors.get(i, float('inf')) for i in range(len(self.models))])
                
                # Convert errors to weights (inverse error)
                non_zero_errors = np.maximum(errors, 1e-10)  # Avoid division by zero
                inverse_errors = 1.0 / non_zero_errors
                
                # Normalize weights
                self.weights = inverse_errors / np.sum(inverse_errors)
                
                logger.info("Calculated weights based on model errors")
            elif len(self.weights) != len(self.models):
                # If weights are provided but don't match number of models
                logger.warning("Provided weights do not match number of models. Using equal weights.")
                self.weights = np.ones(len(self.models)) / len(self.models)
        
        self.fitted = True
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.fitted:
            raise ValueError("Bagging ensemble must be fitted before forecasting")
        
        if not self.models:
            raise ValueError("No models available in the ensemble")
        
        # Generate future dates
        last_date = self.history_df['date'].iloc[-1]
        forecast_dates = []
        
        if frequency == 'Y':
            # Yearly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(years=i))
        elif frequency == 'Q':
            # Quarterly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(months=3*i))
        elif frequency == 'M':
            # Monthly forecasts
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(months=i))
        else:
            # Default to yearly
            for i in range(1, periods + 1):
                forecast_dates.append(last_date + pd.DateOffset(years=i))
        
        # Generate forecasts from all models
        all_forecasts = []
        
        # Clear previous individual forecasts
        self.individual_forecasts = {}
        
        for i, model in enumerate(self.models):
            try:
                # Generate forecast
                forecast_df = model.forecast(periods, frequency)
                
                # Store forecast
                self.individual_forecasts[f"Model_{i+1}_{model.name}"] = forecast_df.copy()
                
                # Extract values
                all_forecasts.append(forecast_df['value'].values)
                
            except Exception as e:
                logger.error(f"Error forecasting with model {i+1}: {str(e)}")
        
        # If no forecasts could be generated, raise an error
        if not all_forecasts:
            raise ValueError("No forecasts could be generated successfully")
        
        # Convert to numpy array
        all_forecasts = np.array(all_forecasts)
        
        # Combine forecasts based on combination method
        if self.combination_method == 'mean':
            ensemble_values = np.mean(all_forecasts, axis=0)
        elif self.combination_method == 'median':
            ensemble_values = np.median(all_forecasts, axis=0)
        elif self.combination_method == 'weighted':
            # Using pre-calculated weights
            ensemble_values = np.zeros(periods)
            
            for i in range(len(self.models)):
                if i < len(all_forecasts) and i < len(self.weights):
                    ensemble_values += self.weights[i] * all_forecasts[i]
        else:
            # Default to mean
            logger.warning(f"Unknown combination method: {self.combination_method}. Using mean.")
            ensemble_values = np.mean(all_forecasts, axis=0)
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            ensemble_values = np.maximum(ensemble_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': ensemble_values
        })
        
        # Generate confidence intervals based on model diversity
        # Calculate standard deviation across model forecasts
        forecast_std = np.std(all_forecasts, axis=0)
        
        # Create 95% confidence intervals
        lower_bounds = ensemble_values - 1.96 * forecast_std
        upper_bounds = ensemble_values + 1.96 * forecast_std
        
        # Ensure lower bounds are non-negative if minimum value is 0
        if self.ensure_minimum and self.minimum_value >= 0:
            lower_bounds = np.maximum(lower_bounds, self.minimum_value)
        
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
        
        # Generate predictions from all models
        all_predictions = []
        
        for model in self.models:
            try:
                if model.fitted:
                    # Generate predictions
                    predictions = model._predict_for_dates(dates_pd.values)
                    all_predictions.append(predictions)
                else:
                    logger.warning(f"{model.name} not fitted. Skipping.")
            except Exception as e:
                logger.error(f"Error predicting with {model.name}: {str(e)}")
        
        # If no predictions could be generated, return fallback values
        if not all_predictions:
            # Use last value from history as fallback
            if self.history_df is not None and len(self.history_df) > 0:
                last_value = self.history_df['value'].iloc[-1]
            else:
                last_value = 0
            return np.full(len(dates_pd), last_value)
        
        # Convert to numpy array
        all_predictions = np.array(all_predictions)
        
        # Combine predictions based on combination method
        if self.combination_method == 'mean':
            ensemble_values = np.mean(all_predictions, axis=0)
        elif self.combination_method == 'median':
            ensemble_values = np.median(all_predictions, axis=0)
        elif self.combination_method == 'weighted':
            # Using pre-calculated weights
            ensemble_values = np.zeros(len(dates_pd))
            
            for i in range(len(self.models)):
                if i < len(all_predictions) and i < len(self.weights):
                    ensemble_values += self.weights[i] * all_predictions[i]
        else:
            # Default to mean
            ensemble_values = np.mean(all_predictions, axis=0)
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            ensemble_values = np.maximum(ensemble_values, self.minimum_value)
        
        return ensemble_values
    
    def plot_ensemble(self, figsize: Tuple[int, int] = (10, 6),
                     include_individual: bool = True,
                     include_history: bool = True,
                     include_samples: bool = False) -> Any:
        """
        Plot the ensemble forecast.
        
        Args:
            figsize: Figure size tuple (width, height)
            include_individual: Whether to include individual model forecasts
            include_history: Whether to include historical data
            include_samples: Whether to include bootstrap samples
            
        Returns:
            Matplotlib figure object or None if plotting is not possible
        """
        if not self.fitted or self.forecast_result is None:
            logger.warning("Cannot plot: ensemble not fitted or no forecast available")
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            if include_samples and self.subsamples and len(self.subsamples) <= 5:
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.5), 
                                             gridspec_kw={'height_ratios': [3, 2]})
            else:
                # Create figure with just forecast plot
                fig, ax1 = plt.subplots(figsize=figsize)
            
            # Plot historical data if available and requested
            if include_history and self.history_df is not None:
                ax1.plot(self.history_df['date'], self.history_df['value'], 'k-', linewidth=2, label='Historical')
            
            # Plot individual model forecasts if requested
            if include_individual and self.individual_forecasts and len(self.models) <= 10:
                # Only show individual models if there aren't too many
                for name, forecast_df in self.individual_forecasts.items():
                    ax1.plot(forecast_df['date'], forecast_df['value'], '--', alpha=0.2, label=name if len(self.models) <= 5 else None)
            
            # Plot ensemble forecast (always)
            ax1.plot(self.forecast_result['date'], self.forecast_result['value'], 
                   'b-', linewidth=3, label='Bagging Ensemble')
            
            # Plot confidence intervals if available
            if self.confidence_intervals is not None:
                ci = self.confidence_intervals
                ax1.fill_between(ci['date'], ci['lower'], ci['upper'], 
                               color='b', alpha=0.1, label='Confidence Interval')
            
            ax1.set_title("Bagging Ensemble Forecast")
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot bootstrap samples if requested
            if include_samples and self.subsamples and len(self.subsamples) <= 5:
                # Plot a few sample subsamples
                for i, sample in enumerate(self.subsamples[:5]):
                    ax2.plot(sample['date'], sample['value'], 'o-', markersize=3, 
                            alpha=0.7, label=f'Sample {i+1}')
                
                # Also plot original data
                if self.history_df is not None:
                    ax2.plot(self.history_df['date'], self.history_df['value'], 'k-', 
                            linewidth=2, label='Original Data')
                
                ax2.set_title("Bootstrap Samples")
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None