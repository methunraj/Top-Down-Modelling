"""
Market-Specific Forecasting Models

This module implements specialized forecasting models designed specifically for 
technology market forecasting and diffusion of innovations.

Models implemented:
- Fisher-Pry: Classic substitution model for technology adoption
- Harvey Logistic: Modified logistic curve used for tech market forecasting
- Norton-Bass: Successive technology generations model
- Lotka-Volterra: Competition model useful for competing technologies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from scipy import optimize
from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FisherPryForecaster(BaseForecaster):
    """
    Fisher-Pry Technology Substitution Model.
    
    This model is used to forecast technology substitution processes. It is particularly
    useful for forecasting the replacement of an old technology by a new one, using an
    S-curve based on a hyperbolic tangent function.
    
    The Fisher-Pry model is based on the assumption that the fractional rate of 
    substitution of a new technology for an old one is proportional to the 
    remaining amount of the old technology still in use.
    
    Formula: F(t) = 1 / (1 + exp(-2α(t - t₀)))
    where:
    - F(t) is the market share of the new technology at time t
    - α is the rate of technology substitution
    - t₀ is the time when the new technology reaches 50% market share
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Fisher-Pry model parameters from configuration."""
        # Rate of technology substitution (default: 0.3)
        self.alpha = self.config.get('alpha', 0.3)
        
        # Initial estimate for t0 (time to 50% substitution) - will be fitted if None
        self.t0 = self.config.get('t0', None)
        
        # Maximum market saturation level (default: 1.0 = 100%)
        self.saturation = self.config.get('saturation', 1.0)
        
        # Whether to fit the saturation level (default: False)
        self.fit_saturation = self.config.get('fit_saturation', False)
        
        # Base year for time calculation (default: use min year in data)
        self.base_year = self.config.get('base_year', None)
        
        # Store historical data
        self.history_df = None

    def _fisher_pry_function(self, t, alpha, t0, saturation=1.0):
        """
        The Fisher-Pry substitution function.
        
        Args:
            t: Time points
            alpha: Rate of substitution
            t0: Time to 50% substitution
            saturation: Maximum market saturation level
            
        Returns:
            Market share values
        """
        return saturation / (1 + np.exp(-2 * alpha * (t - t0)))
    
    def _error_function(self, params, t, y):
        """
        Error function for optimization.
        
        Args:
            params: Model parameters [alpha, t0, saturation] or [alpha, t0]
            t: Time points
            y: Actual values
            
        Returns:
            Sum of squared errors
        """
        if self.fit_saturation:
            alpha, t0, saturation = params
        else:
            alpha, t0 = params
            saturation = self.saturation
            
        predictions = self._fisher_pry_function(t, alpha, t0, saturation)
        return np.sum((y - predictions) ** 2)

    def fit(self, data: pd.DataFrame) -> 'FisherPryForecaster':
        """
        Fit the Fisher-Pry model to historical data.
        
        Args:
            data: DataFrame containing historical data with 'date' and 'value' columns
                
        Returns:
            Self for method chaining
        """
        # Comprehensive data validation
        validation_errors = self._validate_input_data(data, ['date', 'value'])
        if validation_errors:
            raise ValueError(f"Data validation failed: {'; '.join(validation_errors)}")
        
        # Clean data if needed
        data = self._handle_data_quality_issues(data)
        
        # Fisher-Pry specific validation
        if len(data) < 3:
            raise ValueError("Fisher-Pry model requires at least 3 data points")
        
        # Check for monotonic data (Fisher-Pry expects growth patterns)
        if not (data['value'].diff().dropna() >= 0).all():
            logger.warning("Fisher-Pry model expects monotonic (non-decreasing) data for best results")
        
        # Store historical data
        self.history_df = data.copy()
        
        # Extract year from dates
        data['year'] = pd.DatetimeIndex(data['date']).year
        
        # Determine base year if not provided
        if self.base_year is None:
            self.base_year = data['year'].min()
        
        # Convert to relative time (years since base_year)
        t = data['year'].values - self.base_year
        y = data['value'].values
        
        # Normalize values if needed
        if np.max(y) > 1.0 and not self.fit_saturation:
            y = y / np.max(y)
            logger.info("Normalized values to range [0,1] for fitting")
        
        # Initial parameter estimates
        if self.t0 is None:
            # Estimate t0 as the time when the curve crosses 0.5
            if np.any(y >= 0.5 * self.saturation):
                idx = np.argmin(np.abs(y - 0.5 * self.saturation))
                initial_t0 = t[idx]
            else:
                # If no values cross 0.5, estimate based on data trend
                initial_t0 = np.mean(t)
        else:
            initial_t0 = self.t0 - self.base_year  # Convert to relative time
        
        initial_alpha = 0.3  # Initial guess for alpha
        
        # Perform optimization
        try:
            if self.fit_saturation:
                initial_saturation = np.max(y) * 1.2  # Initial guess for saturation
                initial_params = [initial_alpha, initial_t0, initial_saturation]
                bounds = [(0.01, 2.0), (min(t) - 10, max(t) + 10), (max(y), max(y) * 5)]
            else:
                initial_params = [initial_alpha, initial_t0]
                bounds = [(0.01, 2.0), (min(t) - 10, max(t) + 10)]
                
            result = optimize.minimize(
                self._error_function, 
                initial_params, 
                args=(t, y),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                if self.fit_saturation:
                    self.alpha, self.t0_rel, self.saturation = result.x
                else:
                    self.alpha, self.t0_rel = result.x
                    
                # Convert t0 back to calendar year
                self.t0 = self.t0_rel + self.base_year
                
                logger.info(f"Fitted Fisher-Pry model: alpha={self.alpha:.4f}, "
                           f"t0={self.t0:.2f}, saturation={self.saturation:.4f}")
                
                # Create model object for reference
                self.model = {
                    'alpha': self.alpha,
                    't0': self.t0,
                    'saturation': self.saturation,
                    'base_year': self.base_year
                }
                
                self.fitted = True
                
            else:
                logger.warning(f"Optimization failed: {result.message}")
                self.fitted = False
                
        except Exception as e:
            logger.error(f"Error during Fisher-Pry model fitting: {str(e)}")
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
            
        # Get the last date from historical data
        last_date = self.history_df['date'].max()
        
        # Generate future dates based on frequency
        if frequency == 'Y':
            # Yearly forecasting
            last_year = pd.DatetimeIndex([last_date])[0].year
            future_years = range(last_year + 1, last_year + periods + 1)
            future_dates = pd.to_datetime([f"{year}-01-01" for year in future_years])
            
        elif frequency == 'Q':
            # Quarterly forecasting
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=3),
                periods=periods,
                freq='Q'
            )
            
        elif frequency == 'M':
            # Monthly forecasting
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=periods,
                freq='M'
            )
            
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Extract years for the forecast
        forecast_years = pd.DatetimeIndex(future_dates).year
        
        # Calculate relative time values
        t_forecast = forecast_years - self.base_year
        
        # Generate predictions using the fitted model
        predictions = self._fisher_pry_function(
            t_forecast, self.alpha, self.t0 - self.base_year, self.saturation)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'value': predictions
        })
        
        # Store forecast result
        self.forecast_result = forecast_df
        self.forecast_dates = future_dates
        
        # Generate confidence intervals (simple approach for Fisher-Pry)
        lower_bound = predictions * 0.9  # 10% below forecast
        upper_bound = predictions * 1.1  # 10% above forecast
        
        self.confidence_intervals = pd.DataFrame({
            'date': future_dates,
            'value': predictions,
            'lower': lower_bound,
            'upper': upper_bound
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
            
        # Convert dates to years
        years = pd.DatetimeIndex(dates).year
        
        # Calculate relative time values
        t_predict = years - self.base_year
        
        # Generate predictions
        predictions = self._fisher_pry_function(
            t_predict, self.alpha, self.t0 - self.base_year, self.saturation)
            
        return predictions


class HarveyLogisticForecaster(BaseForecaster):
    """
    Harvey Logistic Model for Technology Forecasting.
    
    The Harvey Logistic model is a modified logistic curve used for technology forecasting.
    It is particularly useful for forecasting the growth and market saturation of technologies
    where both the saturation level and growth rates need to be estimated from historical data.
    
    The model uses a three-parameter logistic function and is typically 
    estimated in its log-transformed form.
    
    Formula: ln(y) = ln(α) - ln(1 + βe^(-γt))
    where:
    - y is the technology penetration level
    - α is the saturation level
    - β is related to the initial penetration level
    - γ is the growth rate parameter
    - t is time
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Harvey Logistic model parameters from configuration."""
        # Saturation level (maximum market size) - will be fitted if None
        self.saturation = self.config.get('saturation', None)
        
        # Initial value of beta parameter - will be fitted
        self.beta = self.config.get('beta', None)
        
        # Growth rate parameter - will be fitted
        self.gamma = self.config.get('gamma', None)
        
        # Base year for time calculation (default: use min year in data)
        self.base_year = self.config.get('base_year', None)
        
        # Store historical data
        self.history_df = None

    def _harvey_logistic_function(self, t, saturation, beta, gamma):
        """
        The Harvey Logistic function.
        
        Args:
            t: Time points
            saturation: Saturation level
            beta: Initial level parameter
            gamma: Growth rate parameter
            
        Returns:
            Market penetration values
        """
        return saturation / (1 + beta * np.exp(-gamma * t))
    
    def _log_harvey_logistic(self, t, log_saturation, log_beta, gamma):
        """
        Log-transformed Harvey Logistic function for fitting.
        
        Args:
            t: Time points
            log_saturation: Natural log of saturation level
            log_beta: Natural log of beta
            gamma: Growth rate parameter
            
        Returns:
            Log of market penetration values
        """
        return log_saturation - np.log(1 + np.exp(log_beta - gamma * t))
    
    def _error_function(self, params, t, log_y):
        """
        Error function for optimization (log-transformed).
        
        Args:
            params: Model parameters [log_saturation, log_beta, gamma]
            t: Time points
            log_y: Log of actual values
            
        Returns:
            Sum of squared errors in log space
        """
        log_saturation, log_beta, gamma = params
        predictions = self._log_harvey_logistic(t, log_saturation, log_beta, gamma)
        return np.sum((log_y - predictions) ** 2)

    def fit(self, data: pd.DataFrame) -> 'HarveyLogisticForecaster':
        """
        Fit the Harvey Logistic model to historical data.
        
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
        
        # Extract year from dates
        data['year'] = pd.DatetimeIndex(data['date']).year
        
        # Determine base year if not provided
        if self.base_year is None:
            self.base_year = data['year'].min()
        
        # Convert to relative time (years since base_year)
        t = data['year'].values - self.base_year
        y = data['value'].values
        
        # Ensure all values are positive for log transformation
        if np.any(y <= 0):
            raise ValueError("Harvey Logistic model requires all values to be positive for log transformation")
        
        # Log transform the data
        log_y = np.log(y)
        
        # Initial parameter estimates
        if self.saturation is None:
            # Estimate saturation as 2x the maximum observed value
            initial_saturation = np.max(y) * 2
        else:
            initial_saturation = self.saturation
        
        initial_log_saturation = np.log(initial_saturation)
        
        # Beta relates to initial level
        if self.beta is None:
            # Estimate beta based on first observation
            y0 = y[0]
            initial_beta = (initial_saturation / y0) - 1 if y0 > 0 else 10
        else:
            initial_beta = self.beta
            
        initial_log_beta = np.log(max(initial_beta, 1e-6))  # Ensure positive
        
        # Gamma is the growth rate
        if self.gamma is None:
            # Start with a moderate growth rate
            initial_gamma = 0.2
        else:
            initial_gamma = self.gamma
        
        # Perform optimization
        try:
            initial_params = [initial_log_saturation, initial_log_beta, initial_gamma]
            
            # Bounds for parameters (in log space for saturation and beta)
            # Saturation must be greater than max observed
            min_log_saturation = np.log(np.max(y))
            
            bounds = [
                (min_log_saturation, min_log_saturation + np.log(10)),  # log_saturation
                (-5.0, 10.0),  # log_beta (very wide range)
                (0.01, 2.0)    # gamma (growth rate)
            ]
            
            result = optimize.minimize(
                self._error_function, 
                initial_params, 
                args=(t, log_y),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                log_saturation, log_beta, self.gamma = result.x
                
                # Transform back from log space
                self.saturation = np.exp(log_saturation)
                self.beta = np.exp(log_beta)
                
                logger.info(f"Fitted Harvey Logistic model: saturation={self.saturation:.4f}, "
                           f"beta={self.beta:.4f}, gamma={self.gamma:.4f}")
                
                # Create model object for reference
                self.model = {
                    'saturation': self.saturation,
                    'beta': self.beta,
                    'gamma': self.gamma,
                    'base_year': self.base_year
                }
                
                self.fitted = True
                
            else:
                logger.warning(f"Optimization failed: {result.message}")
                self.fitted = False
                
        except Exception as e:
            logger.error(f"Error during Harvey Logistic model fitting: {str(e)}")
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
            
        # Get the last date from historical data
        last_date = self.history_df['date'].max()
        
        # Generate future dates based on frequency
        if frequency == 'Y':
            # Yearly forecasting
            last_year = pd.DatetimeIndex([last_date])[0].year
            future_years = range(last_year + 1, last_year + periods + 1)
            future_dates = pd.to_datetime([f"{year}-01-01" for year in future_years])
            
        elif frequency == 'Q':
            # Quarterly forecasting
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=3),
                periods=periods,
                freq='Q'
            )
            
        elif frequency == 'M':
            # Monthly forecasting
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=periods,
                freq='M'
            )
            
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Extract years for the forecast
        forecast_years = pd.DatetimeIndex(future_dates).year
        
        # Calculate relative time values
        t_forecast = forecast_years - self.base_year
        
        # Generate predictions using the fitted model
        predictions = self._harvey_logistic_function(
            t_forecast, self.saturation, self.beta, self.gamma)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'value': predictions
        })
        
        # Store forecast result
        self.forecast_result = forecast_df
        self.forecast_dates = future_dates
        
        # Generate confidence intervals
        # For Harvey model, uncertainty increases with time
        lower_bound = np.maximum(0, predictions * (1 - 0.05 * np.sqrt(np.arange(1, len(predictions) + 1) / 10)))
        upper_bound = predictions * (1 + 0.05 * np.sqrt(np.arange(1, len(predictions) + 1) / 10))
        
        self.confidence_intervals = pd.DataFrame({
            'date': future_dates,
            'value': predictions,
            'lower': lower_bound,
            'upper': upper_bound
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
            
        # Convert dates to years
        years = pd.DatetimeIndex(dates).year
        
        # Calculate relative time values
        t_predict = years - self.base_year
        
        # Generate predictions
        predictions = self._harvey_logistic_function(
            t_predict, self.saturation, self.beta, self.gamma)
            
        return predictions


class NortonBassForecaster(BaseForecaster):
    """
    Norton-Bass Model for Successive Technology Generations.
    
    The Norton-Bass model extends the Bass diffusion model to handle multiple generations 
    of a technology. It's particularly useful for forecasting markets where new product 
    generations replace older ones (like smartphones, software versions, etc.)
    
    For each generation, the model estimates:
    - m: market potential parameter
    - p: coefficient of innovation (external influence)
    - q: coefficient of imitation (internal influence)
    
    The model accounts for how later generations cannibalize sales from earlier ones.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Norton-Bass model parameters from configuration."""
        # Number of generations to model (default: 1)
        self.num_generations = self.config.get('num_generations', 1)
        
        # Initial estimates for market potential for each generation
        # Will be fitted if not provided
        self.m_values = self.config.get('m_values', [None] * self.num_generations)
        if len(self.m_values) != self.num_generations:
            logger.warning("Length of m_values does not match num_generations. Using defaults.")
            self.m_values = [None] * self.num_generations
            
        # Initial estimates for innovation coefficient (p) for each generation
        # Will be fitted if not provided
        self.p_values = self.config.get('p_values', [0.01] * self.num_generations)
        if len(self.p_values) != self.num_generations:
            self.p_values = [0.01] * self.num_generations
            
        # Initial estimates for imitation coefficient (q) for each generation
        # Will be fitted if not provided
        self.q_values = self.config.get('q_values', [0.1] * self.num_generations)
        if len(self.q_values) != self.num_generations:
            self.q_values = [0.1] * self.num_generations
            
        # Introduction times for each generation (years since start)
        # Must be provided in ascending order (gen1=0, gen2=t2, gen3=t3, etc.)
        self.intro_times = self.config.get('intro_times', [0] * self.num_generations)
        if len(self.intro_times) != self.num_generations:
            logger.warning("Length of intro_times does not match num_generations.")
            if self.num_generations > 1:
                # Make a guess at reasonable introduction times
                self.intro_times = [0] + [i*5 for i in range(1, self.num_generations)]
            else:
                self.intro_times = [0]
                
        # Base year (start of first generation)
        self.base_year = self.config.get('base_year', None)
        
        # Store historical data
        self.history_df = None
        
    def _bass_f(self, t, p, q):
        """
        Bass cumulative adoption function F(t).
        
        Args:
            t: Time since introduction
            p: Coefficient of innovation
            q: Coefficient of imitation
            
        Returns:
            Cumulative adoption fraction
        """
        # Handle negative times (before introduction)
        if np.isscalar(t):
            if t <= 0:
                return 0
        else:
            t = np.maximum(t, 0)
            
        return (1 - np.exp(-(p + q) * t)) / (1 + (q/p) * np.exp(-(p + q) * t))

    def _norton_bass_model(self, t, params):
        """
        Norton-Bass model for multiple technology generations.
        
        Args:
            t: Array of time points
            params: List of parameters [m1, p1, q1, m2, p2, q2, ...]
            
        Returns:
            Tuple of arrays (total adoption, adoption by generation)
        """
        # Unpack parameters
        params_per_gen = 3  # m, p, q for each generation
        m_values = params[0::params_per_gen]
        p_values = params[1::params_per_gen]
        q_values = params[2::params_per_gen]
        
        # Ensure t is an array
        t = np.atleast_1d(t)
        
        # Initialize arrays for results
        n_periods = len(t)
        adoptions_by_gen = np.zeros((self.num_generations, n_periods))
        
        # Calculate F(t) for each generation
        F_values = np.zeros((self.num_generations, n_periods))
        for i in range(self.num_generations):
            # Time since introduction of this generation
            t_since_intro = t - self.intro_times[i]
            
            # Calculate F(t) for this generation
            for j in range(n_periods):
                F_values[i, j] = self._bass_f(t_since_intro[j], p_values[i], q_values[i])
        
        # Calculate adoptions based on Norton-Bass model
        for i in range(self.num_generations):
            # Base adoption for this generation
            adoptions_by_gen[i] = m_values[i] * F_values[i]
            
            # Adjust for cannibalization by later generations
            for j in range(i+1, self.num_generations):
                adoptions_by_gen[i] = adoptions_by_gen[i] * (1 - F_values[j])
                
        # Total adoption across all generations
        total_adoption = np.sum(adoptions_by_gen, axis=0)
        
        return total_adoption, adoptions_by_gen

    def _error_function(self, params, t, y):
        """
        Error function for optimization.
        
        Args:
            params: Model parameters [m1, p1, q1, m2, p2, q2, ...]
            t: Time points
            y: Actual values
            
        Returns:
            Sum of squared errors
        """
        predicted, _ = self._norton_bass_model(t, params)
        return np.sum((y - predicted) ** 2)

    def fit(self, data: pd.DataFrame) -> 'NortonBassForecaster':
        """
        Fit the Norton-Bass model to historical data.
        
        Args:
            data: DataFrame containing historical data with 'date', 'value', and optionally
                 'generation' columns. If 'generation' is not provided, all data is assumed
                 to be total market values.
                
        Returns:
            Self for method chaining
        """
        # Validate input data
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Store historical data
        self.history_df = data.copy()
        
        # Extract year from dates
        data['year'] = pd.DatetimeIndex(data['date']).year
        
        # Determine base year if not provided
        if self.base_year is None:
            self.base_year = data['year'].min()
        
        # Convert to relative time (years since base_year)
        t = data['year'].values - self.base_year
        
        # Check if generation-specific data is provided
        has_generation_data = 'generation' in data.columns
        
        if has_generation_data:
            # Handle data with generation information
            # Pivot to get values by generation
            pivot_df = data.pivot_table(
                index='year', 
                columns='generation', 
                values='value', 
                aggfunc='sum'
            ).reset_index()
            
            # Ensure we have data for all generations
            for i in range(1, self.num_generations + 1):
                if i not in pivot_df.columns:
                    logger.warning(f"No data found for generation {i}, using zeros")
                    pivot_df[i] = 0
                    
            # Sort by year
            pivot_df = pivot_df.sort_values('year')
            
            # Extract time and values
            t = pivot_df['year'].values - self.base_year
            y_by_gen = np.array([pivot_df[i].values for i in range(1, self.num_generations + 1)])
            
            # Total market is sum across generations
            y = np.sum(y_by_gen, axis=0)
            
        else:
            # Use total market data only
            y = data['value'].values
            
        # Initialize parameters for optimization
        params_for_opt = []
        bounds = []
        
        # For each generation, we need m, p, and q
        for i in range(self.num_generations):
            # Market potential (m)
            if self.m_values[i] is None:
                # If market potential not specified, estimate from data
                if i == 0:
                    # For first generation, use 2x max observed value
                    initial_m = np.max(y) * 2
                else:
                    # For later generations, use last generation's estimate
                    initial_m = params_for_opt[0] * 0.8  # 80% of previous gen
            else:
                initial_m = self.m_values[i]
                
            params_for_opt.append(initial_m)
            bounds.append((np.max(y) * 0.5, np.max(y) * 10))  # m bounds
            
            # Innovation coefficient (p)
            initial_p = self.p_values[i]
            params_for_opt.append(initial_p)
            bounds.append((0.001, 0.05))  # p bounds - typically small
            
            # Imitation coefficient (q)
            initial_q = self.q_values[i]
            params_for_opt.append(initial_q)
            bounds.append((0.01, 0.5))  # q bounds - typically larger than p
        
        # Perform optimization
        try:
            result = optimize.minimize(
                self._error_function, 
                params_for_opt, 
                args=(t, y),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                # Extract optimized parameters
                optimized_params = result.x
                
                # Unpack parameters
                self.m_values = optimized_params[0::3]
                self.p_values = optimized_params[1::3]
                self.q_values = optimized_params[2::3]
                
                # Create model object for reference
                self.model = {
                    'm_values': self.m_values,
                    'p_values': self.p_values,
                    'q_values': self.q_values,
                    'intro_times': self.intro_times,
                    'base_year': self.base_year
                }
                
                # Log the fitted parameters
                for i in range(self.num_generations):
                    intro_year = self.base_year + self.intro_times[i]
                    logger.info(f"Generation {i+1} (introduced in {intro_year}): "
                               f"m={self.m_values[i]:.2f}, p={self.p_values[i]:.4f}, "
                               f"q={self.q_values[i]:.4f}")
                
                self.fitted = True
                
                # Store the generation-specific adoptions
                _, self.gen_adoptions = self._norton_bass_model(t, optimized_params)
                
            else:
                logger.warning(f"Optimization failed: {result.message}")
                self.fitted = False
                
        except Exception as e:
            logger.error(f"Error during Norton-Bass model fitting: {str(e)}")
            self.fitted = False
            
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate a forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
                
        Returns:
            DataFrame with forecasted values for total market
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
            
        # Get the last date from historical data
        last_date = self.history_df['date'].max()
        
        # Generate future dates based on frequency
        if frequency == 'Y':
            # Yearly forecasting
            last_year = pd.DatetimeIndex([last_date])[0].year
            future_years = range(last_year + 1, last_year + periods + 1)
            future_dates = pd.to_datetime([f"{year}-01-01" for year in future_years])
            
        elif frequency == 'Q':
            # For simplicity, we'll convert quarterly to fractional years
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=3),
                periods=periods,
                freq='Q'
            )
            
        elif frequency == 'M':
            # For simplicity, we'll convert monthly to fractional years
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=periods,
                freq='M'
            )
            
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Extract years for the forecast
        forecast_years = pd.DatetimeIndex(future_dates).year
        
        # For quarterly and monthly frequencies, add the fractional component
        if frequency == 'Q':
            forecast_years = forecast_years + (pd.DatetimeIndex(future_dates).month - 1) / 12
        elif frequency == 'M':
            forecast_years = forecast_years + (pd.DatetimeIndex(future_dates).month - 1) / 12
        
        # Calculate relative time values
        t_forecast = forecast_years - self.base_year
        
        # Create parameter list for model
        params = []
        for i in range(self.num_generations):
            params.extend([self.m_values[i], self.p_values[i], self.q_values[i]])
        
        # Generate predictions using the fitted model
        total_predictions, gen_predictions = self._norton_bass_model(t_forecast, params)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'value': total_predictions
        })
        
        # Store forecast result
        self.forecast_result = forecast_df
        self.forecast_dates = future_dates
        
        # Create a more detailed forecast that includes generation-specific values
        detailed_forecast = forecast_df.copy()
        for i in range(self.num_generations):
            detailed_forecast[f'gen_{i+1}'] = gen_predictions[i]
        
        self.detailed_forecast = detailed_forecast
        
        # Generate confidence intervals (simple approach for now)
        # Uncertainty increases with forecast horizon
        horizon_factor = np.sqrt(np.arange(1, len(total_predictions) + 1) / 10)
        lower_bound = total_predictions * (1 - 0.05 * horizon_factor)
        upper_bound = total_predictions * (1 + 0.05 * horizon_factor)
        
        self.confidence_intervals = pd.DataFrame({
            'date': future_dates,
            'value': total_predictions,
            'lower': lower_bound,
            'upper': upper_bound
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
            
        # Convert dates to years
        years = pd.DatetimeIndex(dates).year
        
        # Calculate relative time values
        t_predict = years - self.base_year
        
        # Create parameter list for model
        params = []
        for i in range(self.num_generations):
            params.extend([self.m_values[i], self.p_values[i], self.q_values[i]])
        
        # Generate predictions
        total_predictions, _ = self._norton_bass_model(t_predict, params)
            
        return total_predictions
        
    def get_generation_forecast(self) -> pd.DataFrame:
        """
        Get generation-specific forecasts.
        
        Returns:
            DataFrame with forecasted values for each generation
        """
        if self.forecast_result is None:
            raise ValueError("Must run forecast() before getting generation-specific forecasts")
            
        return self.detailed_forecast


class LotkaVolterraForecaster(BaseForecaster):
    """
    Lotka-Volterra Competition Model for Technology Forecasting.
    
    This model adapts the ecological competition model to analyze and forecast 
    competing technologies in a market. It's particularly useful for analyzing 
    market share dynamics between competing technologies or products.
    
    The model uses coupled differential equations to represent how technologies
    compete for limited resources (market share).
    
    For two competing technologies, the model is:
    dx/dt = r_x * x * (K_x - x - α_xy * y) / K_x
    dy/dt = r_y * y * (K_y - y - α_yx * x) / K_y
    
    where:
    - x, y are the market shares or adoption levels of technologies X and Y
    - r_x, r_y are the intrinsic growth rates
    - K_x, K_y are the carrying capacities (maximum market potential)
    - α_xy is the effect of Y on X
    - α_yx is the effect of X on Y
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Lotka-Volterra model parameters from configuration."""
        # Number of competing technologies (default: 2)
        self.num_techs = self.config.get('num_techs', 2)
        
        # Names of technologies (for readability)
        self.tech_names = self.config.get('tech_names', [f"Tech{i+1}" for i in range(self.num_techs)])
        
        # Intrinsic growth rates for each technology
        self.growth_rates = self.config.get('growth_rates', [0.5] * self.num_techs)
        
        # Carrying capacities for each technology
        self.capacities = self.config.get('capacities', [1.0] * self.num_techs)
        
        # Competition matrix (alpha values)
        # alpha[i,j] represents effect of tech j on tech i
        if 'competition_matrix' in self.config:
            self.competition_matrix = np.array(self.config['competition_matrix'])
        else:
            # Default: moderate competition between all technologies
            self.competition_matrix = np.ones((self.num_techs, self.num_techs)) * 0.5
            # No self-competition (diagonal is 0)
            np.fill_diagonal(self.competition_matrix, 0)
            
        # Time step for numerical integration
        self.dt = self.config.get('dt', 0.1)
        
        # Base year for time calculation
        self.base_year = self.config.get('base_year', None)
        
        # Store historical data
        self.history_df = None

    def _lotka_volterra_system(self, state, t, growth_rates, capacities, competition):
        """
        Lotka-Volterra competition differential equations.
        
        Args:
            state: Current state of all technologies
            t: Time (unused, but required by ODE solvers)
            growth_rates: Growth rates for each technology
            capacities: Carrying capacities for each technology
            competition: Competition matrix
            
        Returns:
            State derivatives (growth rates for all technologies)
        """
        derivatives = np.zeros_like(state)
        
        for i in range(len(state)):
            # Calculate total competition effect
            competition_effect = 0
            for j in range(len(state)):
                if i != j:  # Don't include self-competition
                    competition_effect += competition[i, j] * state[j]
            
            # Lotka-Volterra equation
            derivatives[i] = growth_rates[i] * state[i] * (
                (capacities[i] - state[i] - competition_effect) / capacities[i])
        
        return derivatives
    
    def _fit_parameters(self, t, data):
        """
        Fit the model parameters using optimization.
        
        Args:
            t: Time points
            data: Observed values for each technology
            
        Returns:
            Fitted parameters dictionary
        """
        from scipy.integrate import odeint
        
        # Get initial adoption values
        initial_state = data[:, 0]
        
        # Define error function for optimization
        def error_function(params):
            # Unpack parameters
            n = self.num_techs
            growth_rates = params[:n]
            capacities = params[n:2*n]
            
            # For competition matrix, use only upper triangle to reduce parameters
            comp_flat = params[2*n:]
            competition = np.zeros((n, n))
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    competition[i, j] = comp_flat[idx]
                    competition[j, i] = comp_flat[idx]
                    idx += 1
            
            # Simulate the system
            pred = odeint(
                self._lotka_volterra_system, 
                initial_state, 
                t, 
                args=(growth_rates, capacities, competition)
            )
            
            # Calculate error
            return np.sum((pred - data) ** 2)
        
        # Initial parameter estimates
        init_params = []
        bounds = []
        
        # Growth rates
        init_params.extend(self.growth_rates)
        bounds.extend([(0.01, 2.0) for _ in range(self.num_techs)])
        
        # Carrying capacities
        init_params.extend(self.capacities)
        
        # Estimate reasonable bounds for capacities
        max_values = np.max(data, axis=1)
        bounds.extend([(max_val, max_val * 5) for max_val in max_values])
        
        # Competition coefficients (only upper triangle)
        for i in range(self.num_techs):
            for j in range(i+1, self.num_techs):
                init_params.append(self.competition_matrix[i, j])
                bounds.append((0.01, 2.0))
        
        # Perform optimization
        try:
            result = optimize.minimize(
                error_function, 
                init_params, 
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                # Unpack optimized parameters
                params = result.x
                n = self.num_techs
                growth_rates = params[:n]
                capacities = params[n:2*n]
                
                # Reconstruct competition matrix
                competition = np.zeros((n, n))
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        competition[i, j] = params[2*n + idx]
                        competition[j, i] = params[2*n + idx]
                        idx += 1
                
                return {
                    'growth_rates': growth_rates,
                    'capacities': capacities,
                    'competition_matrix': competition
                }
                
            else:
                logger.warning(f"Parameter optimization failed: {result.message}")
                return None
                
        except Exception as e:
            logger.error(f"Error during parameter optimization: {str(e)}")
            return None

    def fit(self, data: pd.DataFrame) -> 'LotkaVolterraForecaster':
        """
        Fit the Lotka-Volterra competition model to historical data.
        
        Args:
            data: DataFrame with columns 'date', 'technology', and 'value'.
                 Where 'technology' identifies the specific technology.
                
        Returns:
            Self for method chaining
        """
        from scipy.integrate import odeint
        
        # Validate input data
        if not all(col in data.columns for col in ['date', 'technology', 'value']):
            raise ValueError("Data must contain 'date', 'technology', and 'value' columns")
        
        # Store historical data
        self.history_df = data.copy()
        
        # Extract year from dates
        data['year'] = pd.DatetimeIndex(data['date']).year
        
        # Determine base year if not provided
        if self.base_year is None:
            self.base_year = data['year'].min()
        
        # Validate technology names and map to indices
        techs_in_data = data['technology'].unique()
        tech_to_idx = {}
        
        if len(techs_in_data) != self.num_techs:
            logger.warning(f"Number of technologies in data ({len(techs_in_data)}) "
                          f"doesn't match configuration ({self.num_techs}). "
                          f"Using technologies from data.")
            self.num_techs = len(techs_in_data)
            
        for i, tech in enumerate(techs_in_data):
            tech_to_idx[tech] = i
            
        # Update tech names if necessary
        if len(self.tech_names) != self.num_techs:
            self.tech_names = list(techs_in_data)
        
        # Pivot the data to get time series for each technology
        pivot_df = data.pivot_table(
            index='year', 
            columns='technology', 
            values='value', 
            aggfunc='sum'
        ).fillna(0).reset_index()
        
        # Sort by year
        pivot_df = pivot_df.sort_values('year')
        
        # Extract time and values
        years = pivot_df['year'].values
        t = years - self.base_year
        tech_values = np.array([pivot_df[tech].values for tech in self.tech_names])
        
        # Update model parameters if specified
        if self.growth_rates is None or len(self.growth_rates) != self.num_techs:
            self.growth_rates = [0.5] * self.num_techs
            
        if self.capacities is None or len(self.capacities) != self.num_techs:
            # Estimate initial capacities as 2x max observed value
            self.capacities = [np.max(tech_values[i]) * 2 for i in range(self.num_techs)]
            
        if self.competition_matrix.shape != (self.num_techs, self.num_techs):
            # Reset competition matrix with moderate competition
            self.competition_matrix = np.ones((self.num_techs, self.num_techs)) * 0.5
            np.fill_diagonal(self.competition_matrix, 0)  # No self-competition
        
        # Fit model parameters
        fitted_params = self._fit_parameters(t, tech_values)
        
        if fitted_params:
            # Update with fitted parameters
            self.growth_rates = fitted_params['growth_rates']
            self.capacities = fitted_params['capacities']
            self.competition_matrix = fitted_params['competition_matrix']
            
            # Simulate with fitted parameters
            initial_state = tech_values[:, 0]
            fitted_values = odeint(
                self._lotka_volterra_system, 
                initial_state, 
                t, 
                args=(self.growth_rates, self.capacities, self.competition_matrix)
            ).T
            
            # Create model object
            self.model = {
                'growth_rates': self.growth_rates,
                'capacities': self.capacities,
                'competition_matrix': self.competition_matrix,
                'tech_names': self.tech_names,
                'base_year': self.base_year,
                'fitted_values': fitted_values
            }
            
            self.fitted = True
            
            # Log fitted parameters
            logger.info("Fitted Lotka-Volterra model parameters:")
            for i, tech in enumerate(self.tech_names):
                logger.info(f"{tech}: growth_rate={self.growth_rates[i]:.4f}, "
                           f"capacity={self.capacities[i]:.2f}")
                
            logger.info("Competition matrix:")
            for i, tech_i in enumerate(self.tech_names):
                for j, tech_j in enumerate(self.tech_names):
                    if i != j:
                        logger.info(f"Effect of {tech_j} on {tech_i}: "
                                   f"{self.competition_matrix[i, j]:.4f}")
            
        else:
            logger.warning("Failed to fit Lotka-Volterra model parameters")
            self.fitted = False
            
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate a forecast for the specified number of periods.
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast (Y=yearly, Q=quarterly, M=monthly)
                
        Returns:
            DataFrame with forecasted values for all technologies
        """
        from scipy.integrate import odeint
        
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
            
        # Get the last date from historical data
        last_date = self.history_df['date'].max()
        
        # Generate future dates based on frequency
        if frequency == 'Y':
            # Yearly forecasting
            last_year = pd.DatetimeIndex([last_date])[0].year
            future_years = range(last_year + 1, last_year + periods + 1)
            future_dates = pd.to_datetime([f"{year}-01-01" for year in future_years])
            dt_years = 1.0
            
        elif frequency == 'Q':
            # Quarterly forecasting
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=3),
                periods=periods,
                freq='Q'
            )
            dt_years = 0.25
            
        elif frequency == 'M':
            # Monthly forecasting
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=periods,
                freq='M'
            )
            dt_years = 1/12
            
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Create detailed time points for numerical integration
        # (more points than just the forecast dates for better accuracy)
        steps_per_period = max(1, int(dt_years / self.dt))
        total_steps = periods * steps_per_period
        
        # Last time point from fitted data
        last_fitted_year = pd.DatetimeIndex([last_date])[0].year
        last_t = last_fitted_year - self.base_year
        
        # Time points for forecast
        t_detailed = np.linspace(last_t, last_t + periods * dt_years, total_steps + 1)
        
        # Get last state from fitted model
        last_state = np.array([self.model['fitted_values'][i, -1] for i in range(self.num_techs)])
        
        # Forecast using numerical integration
        forecast_detailed = odeint(
            self._lotka_volterra_system, 
            last_state, 
            t_detailed, 
            args=(self.growth_rates, self.capacities, self.competition_matrix)
        )
        
        # Extract values at forecast dates
        forecast_indices = np.arange(steps_per_period, total_steps + 1, steps_per_period)
        forecast_values = forecast_detailed[forecast_indices]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({'date': future_dates})
        
        # Add forecasted values for each technology
        for i, tech in enumerate(self.tech_names):
            forecast_df[tech] = forecast_values[:, i]
        
        # Calculate total market
        forecast_df['value'] = forecast_df[self.tech_names].sum(axis=1)
        
        # Store forecast result
        self.forecast_result = forecast_df
        self.forecast_dates = future_dates
        
        # Generate confidence intervals for total market
        # Simple approach: increasing uncertainty with forecast horizon
        total_market = forecast_df['value'].values
        horizon_factor = np.sqrt(np.arange(1, len(total_market) + 1) / 10)
        lower_bound = total_market * (1 - 0.05 * horizon_factor)
        upper_bound = total_market * (1 + 0.05 * horizon_factor)
        
        self.confidence_intervals = pd.DataFrame({
            'date': future_dates,
            'value': total_market,
            'lower': lower_bound,
            'upper': upper_bound
        })
        
        return forecast_df
    
    def _predict_for_dates(self, dates: np.ndarray) -> np.ndarray:
        """
        Generate predictions for specific dates.
        
        Args:
            dates: Array of dates to predict for
                
        Returns:
            Array of predictions for total market
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
            
        # This is a simplified implementation that only returns total market
        # For a more detailed implementation, would need to do numerical integration
        # For arbitrary dates which is complex for this model
        
        # Convert dates to years
        years = pd.DatetimeIndex(dates).year
        
        # For each year, we'll find the closest match in our forecast
        # If we don't have a forecast yet, do a quick one
        if self.forecast_result is None:
            # Do a 30-year forecast to have data to interpolate from
            self.forecast(30)
            
        # Get the forecast years
        forecast_years = pd.DatetimeIndex(self.forecast_result['date']).year
        
        # Initialize results array
        results = np.zeros(len(years))
        
        # For each requested year, find closest match in forecast
        for i, year in enumerate(years):
            if year <= self.base_year:
                # Before our data, return 0
                results[i] = 0
            elif year in forecast_years:
                # Exact match
                idx = np.where(forecast_years == year)[0][0]
                results[i] = self.forecast_result['value'].iloc[idx]
            else:
                # Need to interpolate
                # Find closest years before and after
                before = forecast_years[forecast_years < year]
                after = forecast_years[forecast_years > year]
                
                if len(before) == 0:
                    # Before our forecast range, use first forecast
                    results[i] = self.forecast_result['value'].iloc[0]
                elif len(after) == 0:
                    # After our forecast range, use last forecast
                    results[i] = self.forecast_result['value'].iloc[-1]
                else:
                    # Interpolate
                    year_before = before.max()
                    year_after = after.min()
                    idx_before = np.where(forecast_years == year_before)[0][0]
                    idx_after = np.where(forecast_years == year_after)[0][0]
                    
                    val_before = self.forecast_result['value'].iloc[idx_before]
                    val_after = self.forecast_result['value'].iloc[idx_after]
                    
                    # Linear interpolation
                    weight = (year - year_before) / (year_after - year_before)
                    results[i] = val_before + weight * (val_after - val_before)
                    
        return results
    
    def get_technology_forecast(self) -> pd.DataFrame:
        """
        Get technology-specific forecasts.
        
        Returns:
            DataFrame with forecasted values for each technology
        """
        if self.forecast_result is None:
            raise ValueError("Must run forecast() before getting technology-specific forecasts")
            
        return self.forecast_result[['date'] + self.tech_names]