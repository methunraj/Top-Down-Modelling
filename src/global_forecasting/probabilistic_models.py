"""
Probabilistic Forecasting Models

This module implements probabilistic time series forecasting methods that provide
full predictive distributions rather than just point forecasts.

Models implemented:
- BayesianStructuralTimeSeriesForecaster: Flexible state space models with priors
- GaussianProcessForecaster: Non-parametric kernel-based probabilistic forecasts
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


class BayesianStructuralTimeSeriesForecaster(BaseForecaster):
    """
    Bayesian Structural Time Series Forecaster.
    
    This model implements Bayesian structural time series forecasting, which combines
    state space models for time series with Bayesian methods. It allows for flexible
    incorporation of trend, seasonality, and regression components, along with full
    predictive distributions.
    
    BSTS models are particularly well-suited for:
    - Causal impact analysis
    - Forecasting with uncertainty quantification
    - Time series with structural changes
    - Incorporating prior knowledge
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize BSTS model parameters from configuration."""
        # Model components
        self.use_trend = self.config.get('use_trend', True)
        self.trend_type = self.config.get('trend_type', 'local_linear')  # 'local_level' or 'local_linear'
        
        # Seasonality
        self.use_seasonal = self.config.get('use_seasonal', True)
        self.seasonal_periods = self.config.get('seasonal_periods', None)
        
        # Regression components
        self.use_regression = self.config.get('use_regression', False)
        self.static_regressors = self.config.get('static_regressors', None)
        self.dynamic_regressors = self.config.get('dynamic_regressors', None)
        
        # MCMC parameters
        self.niter = self.config.get('niter', 1000)
        self.nburn = self.config.get('nburn', 200)
        self.prior_level_sd = self.config.get('prior_level_sd', 0.01)
        self.prior_slope_sd = self.config.get('prior_slope_sd', 0.01)
        
        # Check for TensorFlow Probability
        try:
            import tensorflow_probability as tfp
            self.tfp_available = True
        except ImportError:
            self.tfp_available = False
            logger.warning("TensorFlow Probability not available. Using CmdStan as fallback if available.")
        
        # Check for CmdStanPy
        try:
            import cmdstanpy
            self.cmdstanpy_available = True
        except ImportError:
            self.cmdstanpy_available = False
        
        # Check for PyStan
        try:
            import pystan
            self.pystan_available = True
        except ImportError:
            self.pystan_available = False
        
        if not self.tfp_available and not self.cmdstanpy_available and not self.pystan_available:
            logger.warning("Neither TensorFlow Probability nor Stan is available. Some functionality may be limited.")
        
        # Initialize model storage
        self.model = None
        self.fitted = False
        self.posterior_samples = None

    def _build_tfp_model(self, data):
        """
        Build the model using TensorFlow Probability.
        
        Args:
            data: Time series data for model building
            
        Returns:
            TensorFlow Probability model
        """
        try:
            import tensorflow as tf
            import tensorflow_probability as tfp
            from tensorflow_probability import sts
            
            # Convert data to TensorFlow tensor
            observed_time_series = tf.convert_to_tensor(data, dtype=tf.float32)
            
            # Initialize model components list
            model_components = []
            
            # Add trend component
            if self.use_trend:
                if self.trend_type == 'local_level':
                    # Local level (random walk) trend
                    trend = sts.LocalLevel(
                        observed_time_series=observed_time_series,
                        level_scale_prior=tfp.distributions.LogNormal(
                            loc=tf.math.log(self.prior_level_sd), scale=1.0)
                    )
                else:
                    # Local linear trend (includes slope)
                    trend = sts.LocalLinearTrend(
                        observed_time_series=observed_time_series,
                        level_scale_prior=tfp.distributions.LogNormal(
                            loc=tf.math.log(self.prior_level_sd), scale=1.0),
                        slope_scale_prior=tfp.distributions.LogNormal(
                            loc=tf.math.log(self.prior_slope_sd), scale=1.0)
                    )
                model_components.append(trend)
            
            # Add seasonal component(s)
            if self.use_seasonal and self.seasonal_periods:
                for period in self.seasonal_periods:
                    seasonal = sts.Seasonal(
                        num_seasons=period,
                        observed_time_series=observed_time_series,
                        allow_drift=False
                    )
                    model_components.append(seasonal)
            
            # Add regression component if needed
            if self.use_regression and (self.static_regressors is not None or self.dynamic_regressors is not None):
                # Process regressors
                regressors = []
                regressor_names = []
                
                if self.static_regressors is not None:
                    for name, values in self.static_regressors.items():
                        regressors.append(tf.convert_to_tensor(values, dtype=tf.float32))
                        regressor_names.append(name)
                
                if self.dynamic_regressors is not None:
                    for name, values in self.dynamic_regressors.items():
                        # For dynamic regressors, we need a 2D array where each column is a regressor
                        dynamic_vals = np.array(values)
                        for i in range(dynamic_vals.shape[1]):
                            regressors.append(tf.convert_to_tensor(dynamic_vals[:, i], dtype=tf.float32))
                            regressor_names.append(f"{name}_{i+1}")
                
                # Create regressor component if we have any regressors
                if regressors:
                    design_matrix = tf.stack(regressors, axis=-1)
                    regression = sts.LinearRegression(
                        design_matrix=design_matrix,
                        name='regression'
                    )
                    model_components.append(regression)
            
            # Combine components into a single model
            model = sts.Sum(model_components, observed_time_series=observed_time_series)
            
            return model
            
        except Exception as e:
            logger.error(f"Error building TensorFlow Probability model: {str(e)}")
            raise

    def _build_stan_model(self):
        """
        Build a Stan model string for Bayesian structural time series.
        
        Returns:
            Stan model code as string
        """
        # This is a simplified BSTS model for Stan
        model_code = """
        data {
            int<lower=1> T;                   // Number of time points
            vector[T] y;                      // Observed time series
            int<lower=0, upper=1> use_trend;  // Whether to include trend
            int<lower=0, upper=1> trend_type; // 0: local level, 1: local linear
            int<lower=0, upper=1> use_seasonal; // Whether to include seasonal component
            int<lower=0> num_seasons;         // Number of seasons (0 if not using seasonal)
        }
        
        parameters {
            real<lower=0> obs_sigma;          // Observation noise std
            real<lower=0> level_sigma;        // Level noise std
            real<lower=0> slope_sigma;        // Slope noise std (used if trend_type = 1)
            real<lower=0> seasonal_sigma;     // Seasonal noise std
            
            vector[T] level;                  // State for level
            vector[T] slope;                  // State for slope (used if trend_type = 1)
            vector[num_seasons] seasonal_init; // Initial seasonal states
        }
        
        transformed parameters {
            vector[T] mu;                     // Mean of the observed process
            vector[T] season;                 // Seasonal component
            
            // Initialize seasonal component
            if (use_seasonal && num_seasons > 0) {
                season = rep_vector(0, T);
                for (i in 1:min(T, num_seasons)) {
                    season[i] = seasonal_init[i];
                }
                // Fill the rest of the seasonal component
                for (i in (num_seasons+1):T) {
                    season[i] = season[i-num_seasons];
                }
            } else {
                season = rep_vector(0, T);
            }
            
            // Compute the mean
            if (use_trend) {
                if (trend_type == 0) {
                    // Local level model
                    mu = level + season;
                } else {
                    // Local linear trend model
                    mu = level + season;
                }
            } else {
                mu = season;
            }
        }
        
        model {
            // Priors
            obs_sigma ~ cauchy(0, 5);
            level_sigma ~ cauchy(0, 5);
            slope_sigma ~ cauchy(0, 5);
            seasonal_sigma ~ cauchy(0, 5);
            
            // Initial states
            level[1] ~ normal(0, 10);
            if (trend_type == 1) {
                slope[1] ~ normal(0, 1);
            }
            
            if (use_seasonal && num_seasons > 0) {
                seasonal_init ~ normal(0, 1);
            }
            
            // State transitions
            if (use_trend) {
                if (trend_type == 0) {
                    // Local level model
                    for (t in 2:T) {
                        level[t] ~ normal(level[t-1], level_sigma);
                    }
                } else {
                    // Local linear trend model
                    for (t in 2:T) {
                        level[t] ~ normal(level[t-1] + slope[t-1], level_sigma);
                        slope[t] ~ normal(slope[t-1], slope_sigma);
                    }
                }
            }
            
            // Observations
            y ~ normal(mu, obs_sigma);
        }
        
        generated quantities {
            vector[T] y_rep;     // Replicated data for posterior predictive checks
            vector[T] residuals; // Residuals
            
            for (t in 1:T) {
                y_rep[t] = normal_rng(mu[t], obs_sigma);
                residuals[t] = y[t] - mu[t];
            }
        }
        """
        return model_code

    def fit(self, data: pd.DataFrame) -> 'BayesianStructuralTimeSeriesForecaster':
        """
        Fit the Bayesian Structural Time Series model to historical data.
        
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
        if self.use_seasonal and self.seasonal_periods is None:
            # Try to infer seasonality from data frequency
            try:
                freq = pd.infer_freq(self.dates)
                
                if freq is None:
                    # Calculate average gap between observations
                    date_diffs = np.diff(self.dates.astype(np.int64) // 10**9 // 86400)
                    avg_days_diff = np.mean(date_diffs)
                    
                    if 28 <= avg_days_diff <= 31:
                        # Monthly data
                        self.seasonal_periods = [12]
                    elif 90 <= avg_days_diff <= 92:
                        # Quarterly data
                        self.seasonal_periods = [4]
                    elif 7 <= avg_days_diff <= 7.5:
                        # Weekly data
                        self.seasonal_periods = [52]
                    else:
                        # Default or no clear seasonality
                        self.seasonal_periods = []
                else:
                    # Infer from pandas frequency
                    if freq.startswith('M'):
                        # Monthly data
                        self.seasonal_periods = [12]
                    elif freq.startswith('Q'):
                        # Quarterly data
                        self.seasonal_periods = [4]
                    elif freq.startswith('W'):
                        # Weekly data
                        self.seasonal_periods = [52]
                    elif freq.startswith('D'):
                        # Daily data
                        self.seasonal_periods = [7]  # Weekly seasonality
                    else:
                        # Default or no clear seasonality
                        self.seasonal_periods = []
            except Exception as e:
                logger.warning(f"Could not automatically detect seasonality: {str(e)}")
                self.seasonal_periods = []
        
        # Use either TensorFlow Probability or Stan based on availability
        if self.tfp_available:
            try:
                import tensorflow as tf
                import tensorflow_probability as tfp
                
                # Build the model
                logger.info("Building TensorFlow Probability model...")
                model = self._build_tfp_model(self.values)
                
                # Create the variational surrogate posteriors
                logger.info("Setting up variational inference...")
                surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(
                    model=model
                )
                
                # Set up variational inference
                logger.info("Running variational inference...")
                elbo_loss_curve = tfp.vi.fit_surrogate_posterior(
                    target_log_prob_fn=model.joint_log_prob(observed_time_series=self.values),
                    surrogate_posterior=surrogate_posterior,
                    optimizer=tf.optimizers.Adam(learning_rate=0.1),
                    num_steps=self.niter
                )
                
                # Sample from posterior
                logger.info("Sampling from posterior...")
                samples = surrogate_posterior.sample(self.niter)
                
                # Store the model and samples
                self.model = model
                self.posterior_samples = samples
                self.surrogate_posterior = surrogate_posterior
                self.elbo_loss_curve = elbo_loss_curve
                
                # Create posterior predictive samples
                # This computes 1-step-ahead predictions for the training data
                logger.info("Computing posterior predictive samples...")
                one_step_ahead_mean, one_step_ahead_scale = tfp.sts.one_step_predictive(
                    model,
                    observed_time_series=self.values,
                    parameter_samples=samples
                )
                
                # Store the one-step-ahead predictions
                self.one_step_ahead_mean = one_step_ahead_mean.numpy()
                self.one_step_ahead_scale = one_step_ahead_scale.numpy()
                
                # Calculate fitted values and residuals
                self.fitted_values = self.one_step_ahead_mean
                self.residuals = self.values - self.fitted_values
                
                self.fitted = True
                logger.info("TensorFlow Probability BSTS model fitted successfully")
                
            except Exception as e:
                logger.error(f"Error during TensorFlow Probability model fitting: {str(e)}")
                self.fitted = False
                # Don't raise, try next implementation
                
        if not self.fitted and self.cmdstanpy_available:
            try:
                import cmdstanpy
                
                # Build the Stan model
                model_code = self._build_stan_model()
                
                # Create a temporary model file
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.stan', delete=False) as f:
                    f.write(model_code.encode('utf-8'))
                    stan_file = f.name
                
                try:
                    # Compile the model
                    logger.info("Compiling Stan model...")
                    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
                    
                    # Prepare data for Stan
                    stan_data = {
                        'T': len(self.values),
                        'y': self.values,
                        'use_trend': 1 if self.use_trend else 0,
                        'trend_type': 1 if self.trend_type == 'local_linear' else 0,
                        'use_seasonal': 1 if self.use_seasonal and self.seasonal_periods else 0,
                        'num_seasons': self.seasonal_periods[0] if self.use_seasonal and self.seasonal_periods else 0
                    }
                    
                    # Fit the model
                    logger.info("Fitting Stan model...")
                    fit = model.sample(
                        data=stan_data,
                        iter_sampling=self.niter,
                        iter_warmup=self.nburn,
                        chains=4
                    )
                    
                    # Store results
                    self.model = model
                    self.stan_fit = fit
                    
                    # Extract posterior samples
                    samples = fit.stan_variables()
                    self.posterior_samples = samples
                    
                    # Extract fitted values and residuals
                    self.fitted_values = samples['mu'].mean(axis=0)
                    self.residuals = self.values - self.fitted_values
                    
                    self.fitted = True
                    logger.info("Stan BSTS model fitted successfully")
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(stan_file)
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"Error during CmdStanPy model fitting: {str(e)}")
                self.fitted = False
                # Don't raise, try next implementation
                
        if not self.fitted and self.pystan_available:
            try:
                import pystan
                
                # Build the Stan model
                model_code = self._build_stan_model()
                
                # Compile the model
                logger.info("Compiling Stan model...")
                model = pystan.StanModel(model_code=model_code)
                
                # Prepare data for Stan
                stan_data = {
                    'T': len(self.values),
                    'y': self.values,
                    'use_trend': 1 if self.use_trend else 0,
                    'trend_type': 1 if self.trend_type == 'local_linear' else 0,
                    'use_seasonal': 1 if self.use_seasonal and self.seasonal_periods else 0,
                    'num_seasons': self.seasonal_periods[0] if self.use_seasonal and self.seasonal_periods else 0
                }
                
                # Fit the model
                logger.info("Fitting Stan model...")
                fit = model.sampling(
                    data=stan_data,
                    iter=self.niter + self.nburn,
                    warmup=self.nburn,
                    chains=4
                )
                
                # Store results
                self.model = model
                self.stan_fit = fit
                
                # Extract posterior samples
                samples = fit.extract()
                self.posterior_samples = samples
                
                # Extract fitted values and residuals
                self.fitted_values = samples['mu'].mean(axis=0)
                self.residuals = self.values - self.fitted_values
                
                self.fitted = True
                logger.info("Stan BSTS model fitted successfully")
                
            except Exception as e:
                logger.error(f"Error during PyStan model fitting: {str(e)}")
                self.fitted = False
                # Don't raise, try next implementation
                
        if not self.fitted:
            # Fallback to a simple implementation if neither TF Probability nor Stan is available
            logger.warning("Neither TensorFlow Probability nor Stan is available. Using simplified implementation.")
            try:
                from statsmodels.tsa.statespace.structural import UnobservedComponents
                
                # Determine trend type
                # UnobservedComponents expects: 'level', 'trend', 'llevel', 'lltrend', 'rw', 'rwdrift' or None
                if self.use_trend:
                    if self.trend_type == 'local_level':
                        trend_spec = 'llevel'  # Local level (random walk)
                    else:
                        trend_spec = 'lltrend'  # Local linear trend
                else:
                    trend_spec = None
                
                # Determine seasonal type
                if self.use_seasonal and self.seasonal_periods and len(self.seasonal_periods) > 0:
                    seasonal_spec = self.seasonal_periods[0]
                else:
                    seasonal_spec = None
                
                # Create and fit the model
                logger.info("Fitting UnobservedComponents model...")
                model = UnobservedComponents(
                    self.values,
                    level='llevel' if trend_spec == 'llevel' else False,
                    trend=True if trend_spec == 'lltrend' else False,
                    seasonal=seasonal_spec,
                    irregular=True,
                    stochastic_level=True if trend_spec == 'llevel' else False,
                    stochastic_trend=True if trend_spec == 'lltrend' else False,
                    stochastic_seasonal=True if seasonal_spec else False
                )
                
                fit = model.fit()
                
                # Store results
                self.model = fit
                self.fitted_values = fit.fittedvalues
                self.residuals = self.values - self.fitted_values
                
                self.fitted = True
                logger.info("Simplified BSTS model fitted successfully")
                
            except Exception as e:
                logger.error(f"Error during simplified model fitting: {str(e)}")
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
        
        # Generate forecasts based on the modeling approach used
        if self.tfp_available and hasattr(self, 'surrogate_posterior'):
            try:
                import tensorflow as tf
                import tensorflow_probability as tfp
                
                # Get posterior samples
                samples = self.posterior_samples
                
                # Generate forecast
                logger.info("Generating TensorFlow Probability forecast...")
                forecast_dist = tfp.sts.forecast(
                    model=self.model,
                    observed_time_series=self.values,
                    parameter_samples=samples,
                    num_steps_forecast=periods
                )
                
                # Extract mean and confidence intervals
                forecast_mean = forecast_dist.mean().numpy()
                forecast_stddev = forecast_dist.stddev().numpy()
                
                # Calculate confidence intervals
                lower = forecast_mean - 1.96 * forecast_stddev
                upper = forecast_mean + 1.96 * forecast_stddev
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'value': forecast_mean
                })
                
                # Store forecast result and confidence intervals
                self.forecast_result = forecast_df
                self.forecast_dates = future_dates
                
                self.confidence_intervals = pd.DataFrame({
                    'date': future_dates,
                    'value': forecast_mean,
                    'lower': lower,
                    'upper': upper
                })
                
                return forecast_df
                
            except Exception as e:
                logger.error(f"Error during TensorFlow Probability forecasting: {str(e)}")
                raise
                
        elif hasattr(self, 'stan_fit'):
            # Forecasting with Stan model results
            try:
                # For both cmdstanpy and pystan, we need to do a bit of manual forecasting
                # Extract mean estimates for components
                samples = self.posterior_samples
                
                # Start with the last value from the training data
                last_level = samples['level'][:, -1].mean()
                
                # For local linear trend, also get the last slope
                if self.use_trend and self.trend_type == 'local_linear':
                    last_slope = samples['slope'][:, -1].mean()
                else:
                    last_slope = 0.0
                
                # Get observation and level noise scales
                obs_sigma = samples['obs_sigma'].mean()
                level_sigma = samples['level_sigma'].mean()
                
                # Initialize arrays for forecasts
                forecast_mean = np.zeros(periods)
                forecast_lower = np.zeros(periods)
                forecast_upper = np.zeros(periods)
                
                # Simple forecasting for each period
                # This is a simplified version - in a real implementation we would
                # propagate full uncertainty from the posterior
                for i in range(periods):
                    # Forecast level
                    if i == 0:
                        level = last_level
                        if self.use_trend and self.trend_type == 'local_linear':
                            level += last_slope
                    else:
                        level = forecast_mean[i-1]
                        if self.use_trend and self.trend_type == 'local_linear':
                            level += last_slope
                    
                    # Store mean forecast
                    forecast_mean[i] = level
                    
                    # Calculate cumulative uncertainty (grows with forecast horizon)
                    # This is a simplified version - real forecast uncertainty would
                    # account for parameter uncertainty from the posterior
                    forecast_stddev = np.sqrt(obs_sigma**2 + (i+1) * level_sigma**2)
                    forecast_lower[i] = level - 1.96 * forecast_stddev
                    forecast_upper[i] = level + 1.96 * forecast_stddev
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'value': forecast_mean
                })
                
                # Store forecast result and confidence intervals
                self.forecast_result = forecast_df
                self.forecast_dates = future_dates
                
                self.confidence_intervals = pd.DataFrame({
                    'date': future_dates,
                    'value': forecast_mean,
                    'lower': forecast_lower,
                    'upper': forecast_upper
                })
                
                return forecast_df
                
            except Exception as e:
                logger.error(f"Error during Stan model forecasting: {str(e)}")
                raise
                
        else:
            # Fallback to statsmodels forecasting
            try:
                # Get the fitted statsmodels model
                fit = self.model
                
                # Generate forecast
                forecast_result = fit.get_forecast(steps=periods)
                
                # Extract mean and confidence intervals
                forecast_mean = forecast_result.predicted_mean
                confidence_int = forecast_result.conf_int(alpha=0.05)
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'value': forecast_mean
                })
                
                # Store forecast result and confidence intervals
                self.forecast_result = forecast_df
                self.forecast_dates = future_dates
                
                self.confidence_intervals = pd.DataFrame({
                    'date': future_dates,
                    'value': forecast_mean,
                    'lower': confidence_int[:, 0] if isinstance(confidence_int, np.ndarray) else confidence_int.iloc[:, 0],
                    'upper': confidence_int[:, 1] if isinstance(confidence_int, np.ndarray) else confidence_int.iloc[:, 1]
                })
                
                return forecast_df
                
            except Exception as e:
                logger.error(f"Error during statsmodels forecasting: {str(e)}")
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
        
        # For historical dates, use the fitted values
        for i in historical_indices:
            date = dates_sorted.iloc[i]
            idx = np.where(self.dates == date)[0]
            if len(idx) > 0:
                results[i] = self.fitted_values[idx[0]]
            else:
                # This shouldn't happen given the mask check
                results[i] = np.nan
        
        # For future dates, we need to forecast
        if len(future_indices) > 0:
            # Find the maximum number of periods to forecast
            last_date = self.dates.iloc[-1]
            
            # Calculate forecast periods needed for each future date
            # This is a rough calculation based on years difference
            periods = []
            for date in dates_sorted.iloc[future_indices]:
                if pd.notna(date):
                    # Calculate years difference
                    years_diff = (date.year - last_date.year) + (date.month - last_date.month) / 12
                    # Add a buffer to ensure we forecast far enough
                    periods.append(int(np.ceil(max(1, years_diff * 12))))
                else:
                    periods.append(1)  # Default if date is NaT
            
            # Maximum number of periods to forecast
            max_periods = max(periods)
            
            # Generate forecast
            forecast_df = self.forecast(periods=max_periods, frequency='M')
            forecast_dates = forecast_df['date']
            forecast_values = forecast_df['value'].values
            
            # For each future date, find the closest forecasted date
            for i, idx in enumerate(future_indices):
                date = dates_sorted.iloc[idx]
                if pd.notna(date):
                    # Find closest forecasted date
                    date_diff = np.abs((forecast_dates - date).days)
                    closest_idx = np.argmin(date_diff)
                    results[idx] = forecast_values[closest_idx]
                else:
                    # Handle NaT by using last observed value
                    results[idx] = self.values[-1]
        
        # Reorder results to match original dates order
        reorder_idx = np.argsort(np.argsort(dates_pd))
        return results[reorder_idx]
        
    def get_components(self) -> Dict[str, np.ndarray]:
        """
        Get decomposed time series components (trend, seasonality, etc.)
        
        Returns:
            Dictionary of component arrays
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting components")
            
        components = {}
        
        # Extract components based on modeling approach
        if self.tfp_available and hasattr(self, 'surrogate_posterior'):
            try:
                import tensorflow as tf
                import tensorflow_probability as tfp
                
                # Get posterior samples
                samples = self.posterior_samples
                
                # Decompose the time series
                decomp = tfp.sts.decompose_by_component(
                    self.model,
                    observed_time_series=self.values,
                    parameter_samples=samples
                )
                
                # Convert to numpy arrays and store
                for component_name, component_dist in decomp.items():
                    components[component_name] = component_dist.mean().numpy()
                
            except Exception as e:
                logger.error(f"Error extracting components from TensorFlow Probability model: {str(e)}")
                
        elif hasattr(self, 'stan_fit'):
            # Extract components from Stan fit
            try:
                samples = self.posterior_samples
                
                if self.use_trend:
                    components['trend'] = samples['level'].mean(axis=0)
                    
                    if self.trend_type == 'local_linear' and 'slope' in samples:
                        components['slope'] = samples['slope'].mean(axis=0)
                
                if self.use_seasonal and 'season' in samples:
                    components['seasonal'] = samples['season'].mean(axis=0)
                
            except Exception as e:
                logger.error(f"Error extracting components from Stan model: {str(e)}")
                
        else:
            # Extract components from statsmodels model
            try:
                # Get statsmodels results object
                results = self.model
                
                # Statsmodels stores components in the results
                if hasattr(results, 'states'):
                    states = results.states
                    
                    if hasattr(states, 'smoothed_state'):
                        state_names = results.model.state_names
                        smoothed_states = states.smoothed_state
                        
                        for i, name in enumerate(state_names):
                            if name in ['level', 'trend', 'seasonal']:
                                components[name] = smoothed_states[i]
                
            except Exception as e:
                logger.error(f"Error extracting components from statsmodels model: {str(e)}")
        
        return components


class GaussianProcessForecaster(BaseForecaster):
    """
    Gaussian Process Forecaster.
    
    Gaussian Processes are a flexible, non-parametric Bayesian approach for time series
    forecasting. They model the time series as a collection of random variables with a 
    multivariate Gaussian distribution, defined by mean and covariance (kernel) functions.
    
    GPs naturally provide uncertainty quantification for forecasts, can handle irregular
    time series, and can incorporate various patterns through kernel design.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Gaussian Process model parameters from configuration."""
        # Kernel configuration
        self.kernel_type = self.config.get('kernel_type', 'RBF')
        
        # RBF (Radial Basis Function) kernel parameters
        self.rbf_length_scale = self.config.get('rbf_length_scale', 1.0)
        self.rbf_length_scale_bounds = self.config.get('rbf_length_scale_bounds', (0.1, 10.0))
        
        # Periodic kernel parameters
        self.periodic_length_scale = self.config.get('periodic_length_scale', 1.0)
        self.periodic_periodicity = self.config.get('periodic_periodicity', 1.0)
        
        # Linear kernel parameters
        self.linear_variance = self.config.get('linear_variance', 1.0)
        
        # Noise level
        self.alpha = self.config.get('alpha', 0.1)
        
        # Normalization
        self.normalize_y = self.config.get('normalize_y', True)
        
        # Optimizer settings
        self.n_restarts_optimizer = self.config.get('n_restarts_optimizer', 5)
        
        # Initialize model
        self.gp_model = None
        self.scaler = None
        self.fitted = False
        
        # Try to import scikit-learn
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, ExpSineSquared, RationalQuadratic
            self.has_sklearn = True
        except ImportError:
            logger.warning("scikit-learn not available. Will try to use GPyTorch if available.")
            self.has_sklearn = False
            
            # Try to import GPyTorch
            try:
                import gpytorch
                self.has_gpytorch = True
            except ImportError:
                logger.warning("GPyTorch not available. Falling back to minimal implementation.")
                self.has_gpytorch = False

    def _build_sklearn_kernel(self):
        """
        Build the appropriate kernel for scikit-learn GP model.
        
        Returns:
            sklearn.gaussian_process.kernels object
        """
        from sklearn.gaussian_process.kernels import (
            RBF, ConstantKernel, WhiteKernel, Matern, 
            ExpSineSquared, RationalQuadratic, DotProduct
        )
        
        # Base kernel based on user selection
        if self.kernel_type == 'RBF':
            # Radial Basis Function kernel (smooth functions)
            base_kernel = RBF(
                length_scale=self.rbf_length_scale,
                length_scale_bounds=self.rbf_length_scale_bounds
            )
            
        elif self.kernel_type == 'Matern':
            # Matern kernel (less smooth than RBF)
            base_kernel = Matern(
                length_scale=self.rbf_length_scale,
                length_scale_bounds=self.rbf_length_scale_bounds,
                nu=1.5  # Controls smoothness (0.5, 1.5, 2.5 are common values)
            )
            
        elif self.kernel_type == 'RationalQuadratic':
            # Rational Quadratic kernel (mixture of RBF kernels)
            base_kernel = RationalQuadratic(
                length_scale=self.rbf_length_scale,
                alpha=1.0  # Scale mixture parameter
            )
            
        elif self.kernel_type == 'Periodic':
            # Periodic kernel (for seasonal/cyclic patterns)
            base_kernel = ExpSineSquared(
                length_scale=self.periodic_length_scale,
                periodicity=self.periodic_periodicity,
                periodicity_bounds=(0.1, 100.0)
            )
            
        elif self.kernel_type == 'Linear':
            # Linear kernel (for linear trends)
            base_kernel = DotProduct(sigma_0=self.linear_variance)
            
        elif self.kernel_type == 'Composite':
            # Composite kernel combining several patterns
            # For example: trend + seasonality + noise
            trend_kernel = RBF(
                length_scale=self.rbf_length_scale * 10,  # Longer scale for trend
                length_scale_bounds=(1.0, 100.0)
            )
            
            seasonal_kernel = ExpSineSquared(
                length_scale=1.0,
                periodicity=self.periodic_periodicity,
                periodicity_bounds=(0.1, 100.0)
            )
            
            noise_kernel = WhiteKernel(
                noise_level=0.1,
                noise_level_bounds=(1e-5, 1.0)
            )
            
            base_kernel = trend_kernel + seasonal_kernel + noise_kernel
            
        else:
            # Default to RBF if unknown kernel type
            logger.warning(f"Unknown kernel type: {self.kernel_type}. Using RBF kernel.")
            base_kernel = RBF(
                length_scale=self.rbf_length_scale,
                length_scale_bounds=self.rbf_length_scale_bounds
            )
        
        # Scale the kernel by a constant
        kernel = ConstantKernel(1.0, constant_value_bounds=(0.1, 10.0)) * base_kernel
        
        return kernel

    def _build_gpytorch_model(self, train_x, train_y):
        """
        Build a GPyTorch model for time series forecasting.
        
        Args:
            train_x: Training inputs (time points)
            train_y: Training targets (observations)
            
        Returns:
            GPyTorch model and likelihood
        """
        import torch
        import gpytorch
        
        # Define GPyTorch model
        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, kernel_type):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                
                # Select kernel based on user configuration
                if kernel_type == 'RBF':
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel()
                    )
                elif kernel_type == 'Matern':
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.MaternKernel(nu=1.5)
                    )
                elif kernel_type == 'Periodic':
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.PeriodicKernel()
                    )
                elif kernel_type == 'Linear':
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.LinearKernel()
                    )
                elif kernel_type == 'Composite':
                    # Composite kernel: RBF (trend) + Periodic (seasonality)
                    rbf_kernel = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel()
                    )
                    periodic_kernel = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.PeriodicKernel()
                    )
                    self.covar_module = rbf_kernel + periodic_kernel
                else:
                    # Default to RBF
                    self.covar_module = gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.RBFKernel()
                    )
                
            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)
        
        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(train_x, train_y, likelihood, self.kernel_type)
        
        return model, likelihood

    def fit(self, data: pd.DataFrame) -> 'GaussianProcessForecaster':
        """
        Fit the Gaussian Process model to historical data.
        
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
        
        # Convert dates to relative time values for the GP
        # First date is 0.0, subsequent dates are days since first date / 365.25
        first_date = self.dates.min()
        self.date_values = (self.dates - first_date).dt.total_seconds() / (86400 * 365.25)  # Convert to years
        
        # Try to fit the model using scikit-learn, GPyTorch, or minimal implementation
        if self.has_sklearn:
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.preprocessing import StandardScaler
                
                # Build the kernel
                kernel = self._build_sklearn_kernel()
                
                # Normalize the target values if requested
                if self.normalize_y:
                    self.scaler = StandardScaler()
                    y_train = self.scaler.fit_transform(self.values.reshape(-1, 1)).flatten()
                else:
                    y_train = self.values
                
                # Create and fit the GP model
                logger.info(f"Fitting Gaussian Process with {self.kernel_type} kernel...")
                self.gp_model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=self.alpha,  # Noise parameter
                    normalize_y=False,  # We handle normalization separately
                    n_restarts_optimizer=self.n_restarts_optimizer,
                    random_state=42
                )
                
                self.gp_model.fit(self.date_values.values.reshape(-1, 1), y_train)
                
                # Get fitted values
                self.fitted_values, self.fitted_std = self.gp_model.predict(
                    self.date_values.values.reshape(-1, 1), return_std=True
                )
                
                # Denormalize if necessary
                if self.normalize_y:
                    self.fitted_values = self.scaler.inverse_transform(
                        self.fitted_values.reshape(-1, 1)
                    ).flatten()
                    self.fitted_std = self.fitted_std * self.scaler.scale_[0]
                
                # Calculate residuals
                self.residuals = self.values - self.fitted_values
                
                self.fitted = True
                logger.info("Gaussian Process model fitted successfully")
                
            except Exception as e:
                logger.error(f"Error during scikit-learn Gaussian Process fitting: {str(e)}")
                self.fitted = False
                
        elif self.has_gpytorch:
            try:
                import torch
                import gpytorch
                from torch.optim import Adam
                from sklearn.preprocessing import StandardScaler
                
                # Convert data to PyTorch tensors
                X_train = torch.tensor(self.date_values.values, dtype=torch.float32).reshape(-1, 1)
                
                # Normalize the target values if requested
                if self.normalize_y:
                    self.scaler = StandardScaler()
                    y_train_np = self.scaler.fit_transform(self.values.reshape(-1, 1)).flatten()
                    y_train = torch.tensor(y_train_np, dtype=torch.float32)
                else:
                    y_train = torch.tensor(self.values, dtype=torch.float32)
                
                # Initialize model and likelihood
                self.gpytorch_likelihood, self.gpytorch_model = self._build_gpytorch_model(X_train, y_train)
                
                # Find optimal model hyperparameters
                logger.info("Optimizing GPyTorch model hyperparameters...")
                self.gpytorch_model.train()
                self.gpytorch_likelihood.train()
                
                # Use the Adam optimizer
                optimizer = Adam(self.gpytorch_model.parameters(), lr=0.1)
                
                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gpytorch_likelihood, self.gpytorch_model)
                
                # Training loop
                training_iterations = 100
                for i in range(training_iterations):
                    optimizer.zero_grad()
                    output = self.gpytorch_model(X_train)
                    loss = -mll(output, y_train)
                    loss.backward()
                    optimizer.step()
                    
                    if (i+1) % 20 == 0:
                        logger.info(f"Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}")
                
                # Set model to evaluation mode
                self.gpytorch_model.eval()
                self.gpytorch_likelihood.eval()
                
                # Get fitted values
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = self.gpytorch_model(X_train)
                    self.fitted_values = predictions.mean.numpy()
                    self.fitted_std = predictions.stddev.numpy()
                
                # Denormalize if necessary
                if self.normalize_y:
                    self.fitted_values = self.scaler.inverse_transform(
                        self.fitted_values.reshape(-1, 1)
                    ).flatten()
                    self.fitted_std = self.fitted_std * self.scaler.scale_[0]
                
                # Calculate residuals
                self.residuals = self.values - self.fitted_values
                
                self.fitted = True
                logger.info("GPyTorch model fitted successfully")
                
            except Exception as e:
                logger.error(f"Error during GPyTorch model fitting: {str(e)}")
                self.fitted = False
                
        else:
            # Minimal fallback implementation
            logger.warning("Using minimal GP implementation. Consider installing scikit-learn for full functionality.")
            try:
                from scipy.optimize import minimize
                import numpy as np
                
                # Simplified RBF kernel function
                def kernel(X1, X2, params):
                    """RBF kernel with noise"""
                    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
                    length_scale = params[0]
                    signal_variance = params[1]
                    noise_variance = params[2]
                    K = signal_variance * np.exp(-0.5 * sqdist / length_scale ** 2)
                    if X1 is X2:
                        K = K + noise_variance * np.eye(len(X1))
                    return K
                
                # Negative log likelihood function to minimize
                def nll(params):
                    # Unpack parameters
                    length_scale, signal_variance, noise_variance = params
                    if length_scale <= 0 or signal_variance <= 0 or noise_variance <= 0:
                        return 1e9  # Penalty for invalid parameters
                    
                    # Compute kernel matrix
                    K = kernel(X, X, params)
                    
                    # Compute log likelihood (with stability safeguards)
                    try:
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                        nll = 0.5 * np.dot(y, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * len(X) * np.log(2 * np.pi)
                        return nll
                    except np.linalg.LinAlgError:
                        return 1e9  # Penalty for numerical issues
                
                # Prepare training data
                X = self.date_values.values.reshape(-1, 1)
                
                # Normalize the target values
                if self.normalize_y:
                    # Simple normalization
                    y_mean = np.mean(self.values)
                    y_std = np.std(self.values)
                    if y_std == 0:
                        y_std = 1.0  # Prevent division by zero
                    y = (self.values - y_mean) / y_std
                    self.y_mean = y_mean
                    self.y_std = y_std
                else:
                    y = self.values
                    self.y_mean = 0.0
                    self.y_std = 1.0
                
                # Initial parameter values: [length_scale, signal_variance, noise_variance]
                initial_params = [self.rbf_length_scale, 1.0, self.alpha]
                
                # Optimize parameters
                logger.info("Optimizing GP hyperparameters...")
                bounds = [(0.01, 10.0), (0.01, 10.0), (0.001, 1.0)]
                result = minimize(nll, initial_params, bounds=bounds, method='L-BFGS-B')
                
                # Store optimized parameters
                self.opt_params = result.x
                logger.info(f"Optimized parameters: length_scale={self.opt_params[0]:.3f}, "
                           f"signal_variance={self.opt_params[1]:.3f}, "
                           f"noise_variance={self.opt_params[2]:.3f}")
                
                # Compute kernel matrix with optimized parameters
                K = kernel(X, X, self.opt_params)
                
                # Store data and parameters for prediction
                self.X_train = X
                self.y_train = y
                self.K_train = K
                
                # Compute fitted values
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    self.alpha_train = alpha
                    
                    # Fitted values are just the training predictions
                    self.fitted_values = np.dot(K, alpha)
                    
                    # Compute predicted variance for each training point
                    v = np.linalg.solve(L, np.eye(len(X)))
                    self.fitted_var = self.opt_params[1] - np.sum(v ** 2, axis=0)
                    self.fitted_std = np.sqrt(np.maximum(0, self.fitted_var))
                    
                    # Denormalize
                    self.fitted_values = self.fitted_values * self.y_std + self.y_mean
                    self.fitted_std = self.fitted_std * self.y_std
                    
                    # Calculate residuals
                    self.residuals = self.values - self.fitted_values
                    
                    self.fitted = True
                    logger.info("Minimal GP implementation fitted successfully")
                    
                except np.linalg.LinAlgError as e:
                    logger.error(f"Linear algebra error during fitting: {str(e)}")
                    self.fitted = False
                    
            except Exception as e:
                logger.error(f"Error during minimal GP implementation: {str(e)}")
                self.fitted = False
        
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
        
        # Convert future dates to model input format
        first_date = self.dates.min()
        future_date_values = (future_dates - first_date).total_seconds() / (86400 * 365.25)  # Convert to years
        
        # Generate forecasts based on the modeling approach used
        if self.has_sklearn and hasattr(self, 'gp_model'):
            try:
                # Use scikit-learn GP model to predict
                logger.info("Generating scikit-learn GP forecast...")
                pred_mean, pred_std = self.gp_model.predict(
                    future_date_values.values.reshape(-1, 1), return_std=True
                )
                
                # Denormalize if needed
                if self.normalize_y:
                    pred_mean = self.scaler.inverse_transform(pred_mean.reshape(-1, 1)).flatten()
                    pred_std = pred_std * self.scaler.scale_[0]
                
                # Calculate confidence intervals
                lower = pred_mean - 1.96 * pred_std
                upper = pred_mean + 1.96 * pred_std
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'value': pred_mean
                })
                
                # Store forecast result and confidence intervals
                self.forecast_result = forecast_df
                self.forecast_dates = future_dates
                
                self.confidence_intervals = pd.DataFrame({
                    'date': future_dates,
                    'value': pred_mean,
                    'lower': lower,
                    'upper': upper
                })
                
                return forecast_df
                
            except Exception as e:
                logger.error(f"Error during scikit-learn GP forecasting: {str(e)}")
                raise
                
        elif self.has_gpytorch and hasattr(self, 'gpytorch_model'):
            try:
                import torch
                import gpytorch
                
                # Convert to PyTorch tensor
                X_test = torch.tensor(future_date_values.values, dtype=torch.float32).reshape(-1, 1)
                
                # Generate forecast
                logger.info("Generating GPyTorch forecast...")
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = self.gpytorch_model(X_test)
                    pred_mean = predictions.mean.numpy()
                    pred_std = predictions.stddev.numpy()
                
                # Denormalize if needed
                if self.normalize_y:
                    pred_mean = self.scaler.inverse_transform(pred_mean.reshape(-1, 1)).flatten()
                    pred_std = pred_std * self.scaler.scale_[0]
                
                # Calculate confidence intervals
                lower = pred_mean - 1.96 * pred_std
                upper = pred_mean + 1.96 * pred_std
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'value': pred_mean
                })
                
                # Store forecast result and confidence intervals
                self.forecast_result = forecast_df
                self.forecast_dates = future_dates
                
                self.confidence_intervals = pd.DataFrame({
                    'date': future_dates,
                    'value': pred_mean,
                    'lower': lower,
                    'upper': upper
                })
                
                return forecast_df
                
            except Exception as e:
                logger.error(f"Error during GPyTorch forecasting: {str(e)}")
                raise
                
        else:
            # Minimal fallback implementation
            try:
                # Use stored parameters and data for prediction
                X_train = self.X_train
                alpha = self.alpha_train
                
                # Prepare test points
                X_test = future_date_values.values.reshape(-1, 1)
                
                # Define kernel function (same as in fit)
                def kernel(X1, X2, params):
                    """RBF kernel"""
                    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
                    length_scale = params[0]
                    signal_variance = params[1]
                    K = signal_variance * np.exp(-0.5 * sqdist / length_scale ** 2)
                    return K
                
                # Compute cross-covariance matrix
                K_s = kernel(X_train, X_test, self.opt_params)
                
                # Compute predictive mean
                pred_mean = np.dot(K_s.T, alpha)
                
                # Compute predictive variance
                K_ss = kernel(X_test, X_test, self.opt_params)
                L = np.linalg.cholesky(self.K_train)
                v = np.linalg.solve(L, K_s)
                pred_var = K_ss.diagonal() - np.sum(v ** 2, axis=0)
                pred_std = np.sqrt(np.maximum(0, pred_var))
                
                # Denormalize
                pred_mean = pred_mean * self.y_std + self.y_mean
                pred_std = pred_std * self.y_std
                
                # Calculate confidence intervals
                lower = pred_mean - 1.96 * pred_std
                upper = pred_mean + 1.96 * pred_std
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'value': pred_mean
                })
                
                # Store forecast result and confidence intervals
                self.forecast_result = forecast_df
                self.forecast_dates = future_dates
                
                self.confidence_intervals = pd.DataFrame({
                    'date': future_dates,
                    'value': pred_mean,
                    'lower': lower,
                    'upper': upper
                })
                
                return forecast_df
                
            except Exception as e:
                logger.error(f"Error during minimal GP forecasting: {str(e)}")
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
        
        # For historical dates, use the fitted values
        for i in historical_indices:
            date = dates_sorted.iloc[i]
            idx = np.where(self.dates == date)[0]
            if len(idx) > 0:
                results[i] = self.fitted_values[idx[0]]
            else:
                # This shouldn't happen given the mask check
                results[i] = np.nan
        
        # For future dates, generate predictions
        if len(future_indices) > 0:
            # Convert future dates to model input format
            first_date = self.dates.min()
            future_dates = dates_sorted.iloc[future_indices]
            future_date_values = (future_dates - first_date).dt.total_seconds() / (86400 * 365.25)
            
            # Generate predictions based on the modeling approach used
            if self.has_sklearn and hasattr(self, 'gp_model'):
                # Use scikit-learn GP model
                pred_mean, _ = self.gp_model.predict(
                    future_date_values.values.reshape(-1, 1), return_std=True
                )
                
                # Denormalize if needed
                if self.normalize_y:
                    pred_mean = self.scaler.inverse_transform(pred_mean.reshape(-1, 1)).flatten()
                
                # Assign to results
                results[future_indices] = pred_mean
                
            elif self.has_gpytorch and hasattr(self, 'gpytorch_model'):
                import torch
                import gpytorch
                
                # Convert to PyTorch tensor
                X_test = torch.tensor(future_date_values.values, dtype=torch.float32).reshape(-1, 1)
                
                # Generate predictions
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions = self.gpytorch_model(X_test)
                    pred_mean = predictions.mean.numpy()
                
                # Denormalize if needed
                if self.normalize_y:
                    pred_mean = self.scaler.inverse_transform(pred_mean.reshape(-1, 1)).flatten()
                
                # Assign to results
                results[future_indices] = pred_mean
                
            else:
                # Minimal fallback implementation
                # Use stored parameters and data for prediction
                X_train = self.X_train
                alpha = self.alpha_train
                
                # Prepare test points
                X_test = future_date_values.values.reshape(-1, 1)
                
                # Define kernel function (same as in fit)
                def kernel(X1, X2, params):
                    """RBF kernel"""
                    sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
                    length_scale = params[0]
                    signal_variance = params[1]
                    K = signal_variance * np.exp(-0.5 * sqdist / length_scale ** 2)
                    return K
                
                # Compute cross-covariance matrix
                K_s = kernel(X_train, X_test, self.opt_params)
                
                # Compute predictive mean
                pred_mean = np.dot(K_s.T, alpha)
                
                # Denormalize
                pred_mean = pred_mean * self.y_std + self.y_mean
                
                # Assign to results
                results[future_indices] = pred_mean
        
        # Reorder results to match original dates order
        reorder_idx = np.argsort(np.argsort(dates_pd))
        return results[reorder_idx]