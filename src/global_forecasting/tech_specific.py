"""
Technology-Specific Forecasting Module - Methods optimized for technology markets

This module provides implementations of forecasting methods that are particularly
well-suited for technology markets, including Bass Diffusion, Gompertz Curve,
and Technology S-Curve models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit

from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BassDiffusionForecaster(BaseForecaster):
    """
    Bass Diffusion Model for technology adoption forecasting.
    
    The Bass diffusion model is commonly used to forecast the adoption of new products
    and technologies. It models the entire lifecycle from introduction to saturation,
    based on innovation and imitation factors.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Bass Diffusion model parameters from configuration"""
        # Innovation coefficient (p)
        self.innovation = self.config.get('innovation', 0.03)
        
        # Imitation coefficient (q)
        self.imitation = self.config.get('imitation', 0.38)
        
        # Market potential (m)
        self.market_potential = self.config.get('market_potential', None)
        
        # Whether to fix parameters during fitting
        self.fix_innovation = self.config.get('fix_innovation', False)
        self.fix_imitation = self.config.get('fix_imitation', False)
        self.fix_potential = self.config.get('fix_potential', False)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store parameters and fitted data
        self.params = None
        self.history = None
        self.history_cumulative = None
        self.time_periods = None
    
    def _bass_model(self, t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
        """
        Bass diffusion model function.
        
        Args:
            t: Time periods
            p: Innovation coefficient
            q: Imitation coefficient
            m: Market potential
            
        Returns:
            Predicted cumulative adoptions
        """
        return m * (1 - np.exp(-(p + q) * t)) / (1 + (q/p) * np.exp(-(p + q) * t))
    
    def _bass_model_noncumulative(self, t: np.ndarray, p: float, q: float, m: float) -> np.ndarray:
        """
        Non-cumulative Bass diffusion model (adoption rate).
        
        Args:
            t: Time periods
            p: Innovation coefficient
            q: Imitation coefficient
            m: Market potential
            
        Returns:
            Predicted adoption rate (non-cumulative)
        """
        return m * (p + q)**2 * np.exp(-(p + q) * t) / (p * (1 + (q/p) * np.exp(-(p + q) * t))**2)
    
    def fit(self, data: pd.DataFrame) -> 'BassDiffusionForecaster':
        """
        Fit the Bass Diffusion model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
                'value' should be non-cumulative adoptions (not cumulative)
            
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
        
        # Calculate time periods (assuming equal time intervals)
        start_date = data['date'].min()
        self.time_periods = [(date - start_date).days / 365.25 for date in data['date']]  # Convert to years
        
        # Calculate cumulative adoptions
        cumulative = data['value'].cumsum()
        self.history_cumulative = cumulative.values
        
        # Initial guesses for parameters
        p0 = [self.innovation, self.imitation, self.market_potential or cumulative.iloc[-1] * 2]
        
        # Set bounds for parameters
        lower_bounds = [0.001, 0.01, cumulative.iloc[-1]]
        upper_bounds = [0.5, 0.99, cumulative.iloc[-1] * 10]
        
        # Fix parameters if requested
        if self.fix_innovation or self.fix_imitation or self.fix_potential:
            param_mask = [not self.fix_innovation, not self.fix_imitation, not self.fix_potential]
            
            if all(not mask for mask in param_mask):
                logger.warning("All parameters are fixed. Skipping fitting.")
                self.params = [self.innovation, self.imitation, self.market_potential or cumulative.iloc[-1] * 2]
                self.fitted = True
                return self
            
            # Define the wrapper function that only varies non-fixed parameters
            def wrapper_func(t, *args):
                # Build full parameter list, using fixed values for fixed parameters
                full_params = []
                arg_idx = 0
                
                for idx, fixed in enumerate([self.fix_innovation, self.fix_imitation, self.fix_potential]):
                    if fixed:
                        # Use the fixed value
                        if idx == 0:
                            full_params.append(self.innovation)
                        elif idx == 1:
                            full_params.append(self.imitation)
                        else:
                            full_params.append(self.market_potential)
                    else:
                        # Use the fitted value
                        full_params.append(args[arg_idx])
                        arg_idx += 1
                
                return self._bass_model(t, *full_params)
            
            # Filter bounds for non-fixed parameters
            filtered_lower = [lower_bounds[i] for i in range(len(param_mask)) if param_mask[i]]
            filtered_upper = [upper_bounds[i] for i in range(len(param_mask)) if param_mask[i]]
            filtered_p0 = [p0[i] for i in range(len(param_mask)) if param_mask[i]]
            
            try:
                # Fit the model
                popt, _ = curve_fit(
                    wrapper_func, 
                    self.time_periods, 
                    self.history_cumulative,
                    p0=filtered_p0,
                    bounds=(filtered_lower, filtered_upper),
                    maxfev=5000
                )
                
                # Reconstruct full parameter list
                full_params = []
                popt_idx = 0
                
                for idx, fixed in enumerate([self.fix_innovation, self.fix_imitation, self.fix_potential]):
                    if fixed:
                        # Use the fixed value
                        if idx == 0:
                            full_params.append(self.innovation)
                        elif idx == 1:
                            full_params.append(self.imitation)
                        else:
                            full_params.append(self.market_potential)
                    else:
                        # Use the fitted value
                        full_params.append(popt[popt_idx])
                        popt_idx += 1
                
                self.params = full_params
                
            except Exception as e:
                logger.error(f"Curve fitting failed: {str(e)}")
                logger.warning("Using initial parameters")
                self.params = p0
        else:
            # Fit all parameters
            try:
                popt, _ = curve_fit(
                    self._bass_model, 
                    self.time_periods, 
                    self.history_cumulative,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=5000
                )
                self.params = popt
                
            except Exception as e:
                logger.error(f"Curve fitting failed: {str(e)}")
                logger.warning("Using initial parameters")
                self.params = p0
        
        # Update instance variables with fitted parameters
        self.innovation, self.imitation, self.market_potential = self.params
        
        logger.info(f"Fitted Bass model: p={self.innovation:.4f}, q={self.imitation:.4f}, m={self.market_potential:.2f}")
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
        
        # Calculate time periods for forecast
        start_date = self.history['date'].min()
        last_period = self.time_periods[-1]
        
        if frequency == 'Y':
            forecast_periods = [last_period + i for i in range(1, periods + 1)]
        elif frequency == 'Q':
            forecast_periods = [last_period + i/4 for i in range(1, periods + 1)]
        elif frequency == 'M':
            forecast_periods = [last_period + i/12 for i in range(1, periods + 1)]
        
        # Calculate cumulative adoptions for forecast periods
        forecast_cumulative = self._bass_model(
            np.array(forecast_periods), 
            self.innovation, 
            self.imitation, 
            self.market_potential
        )
        
        # Calculate non-cumulative adoptions (difference from previous period)
        last_cumulative = self.history_cumulative[-1]
        
        if frequency == 'Y':
            # For yearly, just take the difference between consecutive periods
            forecast_values = np.zeros(periods)
            forecast_values[0] = forecast_cumulative[0] - last_cumulative
            for i in range(1, periods):
                forecast_values[i] = forecast_cumulative[i] - forecast_cumulative[i-1]
        else:
            # For quarterly or monthly, use the non-cumulative formula directly
            forecast_values = self._bass_model_noncumulative(
                np.array(forecast_periods),
                self.innovation,
                self.imitation,
                self.market_potential
            )
            
            # Scale to match the frequency
            if frequency == 'Q':
                forecast_values = forecast_values / 4  # Quarterly values
            elif frequency == 'M':
                forecast_values = forecast_values / 12  # Monthly values
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals (simple implementation)
        history_values = self.history['value'].values
        residuals = np.abs(
            self.history_cumulative - 
            self._bass_model(np.array(self.time_periods), self.innovation, self.imitation, self.market_potential)
        )
        avg_residual = np.mean(residuals)
        
        lower_bound = []
        upper_bound = []
        
        for i, value in enumerate(forecast_values):
            # Increase uncertainty with time
            ci_width = avg_residual * (1 + i * 0.2)
            lower = max(value - ci_width, 0)
            upper = value + ci_width
            
            lower_bound.append(lower)
            upper_bound.append(upper)
        
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


class GompertzCurveForecaster(BaseForecaster):
    """
    Gompertz Curve Model for technology adoption forecasting.
    
    The Gompertz curve is an asymmetric S-shaped curve that is often used to model
    biological growth, market saturation, and technology adoption. It's especially
    useful for modeling growth processes that start slowly, accelerate, and then
    approach an asymptote.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Gompertz Curve model parameters from configuration"""
        # Asymptote parameter (a) - upper limit of growth
        self.asymptote = self.config.get('asymptote', None)
        
        # Displacement parameter (b) - horizontal shift
        self.displacement = self.config.get('displacement', 1.0)
        
        # Growth rate parameter (c) - controls growth rate
        self.growth_rate = self.config.get('growth_rate', 0.1)
        
        # Whether to fix parameters during fitting
        self.fix_asymptote = self.config.get('fix_asymptote', False)
        self.fix_displacement = self.config.get('fix_displacement', False)
        self.fix_growth_rate = self.config.get('fix_growth_rate', False)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store parameters and fitted data
        self.params = None
        self.history = None
        self.history_cumulative = None
        self.time_periods = None
    
    def _gompertz_curve(self, t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Gompertz curve function.
        
        Args:
            t: Time periods
            a: Asymptote parameter
            b: Displacement parameter
            c: Growth rate parameter
            
        Returns:
            Predicted cumulative adoptions
        """
        return a * np.exp(-b * np.exp(-c * t))
    
    def _gompertz_derivative(self, t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Derivative of the Gompertz curve (non-cumulative).
        
        Args:
            t: Time periods
            a: Asymptote parameter
            b: Displacement parameter
            c: Growth rate parameter
            
        Returns:
            Predicted adoption rate (non-cumulative)
        """
        return a * b * c * np.exp(-b * np.exp(-c * t) - c * t)
    
    def fit(self, data: pd.DataFrame) -> 'GompertzCurveForecaster':
        """
        Fit the Gompertz Curve model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
                'value' should be non-cumulative adoptions (not cumulative)
            
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
        
        # Calculate time periods (assuming equal time intervals)
        start_date = data['date'].min()
        self.time_periods = [(date - start_date).days / 365.25 for date in data['date']]  # Convert to years
        
        # Calculate cumulative adoptions
        cumulative = data['value'].cumsum()
        self.history_cumulative = cumulative.values
        
        # Initial guesses for parameters
        max_observed = cumulative.iloc[-1]
        estimated_asymptote = self.asymptote or max_observed * 2
        p0 = [estimated_asymptote, self.displacement, self.growth_rate]
        
        # Set bounds for parameters
        lower_bounds = [max_observed, 0.01, 0.01]
        upper_bounds = [max_observed * 10, 10.0, 1.0]
        
        # Fix parameters if requested
        if self.fix_asymptote or self.fix_displacement or self.fix_growth_rate:
            param_mask = [not self.fix_asymptote, not self.fix_displacement, not self.fix_growth_rate]
            
            if all(not mask for mask in param_mask):
                logger.warning("All parameters are fixed. Skipping fitting.")
                self.params = [self.asymptote or estimated_asymptote, self.displacement, self.growth_rate]
                self.fitted = True
                return self
            
            # Define the wrapper function that only varies non-fixed parameters
            def wrapper_func(t, *args):
                # Build full parameter list, using fixed values for fixed parameters
                full_params = []
                arg_idx = 0
                
                for idx, fixed in enumerate([self.fix_asymptote, self.fix_displacement, self.fix_growth_rate]):
                    if fixed:
                        # Use the fixed value
                        if idx == 0:
                            full_params.append(self.asymptote or estimated_asymptote)
                        elif idx == 1:
                            full_params.append(self.displacement)
                        else:
                            full_params.append(self.growth_rate)
                    else:
                        # Use the fitted value
                        full_params.append(args[arg_idx])
                        arg_idx += 1
                
                return self._gompertz_curve(t, *full_params)
            
            # Filter bounds for non-fixed parameters
            filtered_lower = [lower_bounds[i] for i in range(len(param_mask)) if param_mask[i]]
            filtered_upper = [upper_bounds[i] for i in range(len(param_mask)) if param_mask[i]]
            filtered_p0 = [p0[i] for i in range(len(param_mask)) if param_mask[i]]
            
            try:
                # Fit the model
                popt, _ = curve_fit(
                    wrapper_func, 
                    self.time_periods, 
                    self.history_cumulative,
                    p0=filtered_p0,
                    bounds=(filtered_lower, filtered_upper),
                    maxfev=5000
                )
                
                # Reconstruct full parameter list
                full_params = []
                popt_idx = 0
                
                for idx, fixed in enumerate([self.fix_asymptote, self.fix_displacement, self.fix_growth_rate]):
                    if fixed:
                        # Use the fixed value
                        if idx == 0:
                            full_params.append(self.asymptote or estimated_asymptote)
                        elif idx == 1:
                            full_params.append(self.displacement)
                        else:
                            full_params.append(self.growth_rate)
                    else:
                        # Use the fitted value
                        full_params.append(popt[popt_idx])
                        popt_idx += 1
                
                self.params = full_params
                
            except Exception as e:
                logger.error(f"Curve fitting failed: {str(e)}")
                logger.warning("Using initial parameters")
                self.params = p0
        else:
            # Fit all parameters
            try:
                popt, _ = curve_fit(
                    self._gompertz_curve, 
                    self.time_periods, 
                    self.history_cumulative,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=5000
                )
                self.params = popt
                
            except Exception as e:
                logger.error(f"Curve fitting failed: {str(e)}")
                logger.warning("Using initial parameters")
                self.params = p0
        
        # Update instance variables with fitted parameters
        self.asymptote, self.displacement, self.growth_rate = self.params
        
        logger.info(f"Fitted Gompertz model: a={self.asymptote:.2f}, b={self.displacement:.4f}, c={self.growth_rate:.4f}")
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
        
        # Calculate time periods for forecast
        start_date = self.history['date'].min()
        last_period = self.time_periods[-1]
        
        if frequency == 'Y':
            forecast_periods = [last_period + i for i in range(1, periods + 1)]
        elif frequency == 'Q':
            forecast_periods = [last_period + i/4 for i in range(1, periods + 1)]
        elif frequency == 'M':
            forecast_periods = [last_period + i/12 for i in range(1, periods + 1)]
        
        # Calculate cumulative adoptions for forecast periods
        forecast_cumulative = self._gompertz_curve(
            np.array(forecast_periods), 
            self.asymptote, 
            self.displacement, 
            self.growth_rate
        )
        
        # Calculate non-cumulative adoptions (difference from previous period)
        last_cumulative = self.history_cumulative[-1]
        
        if frequency == 'Y':
            # For yearly, just take the difference between consecutive periods
            forecast_values = np.zeros(periods)
            forecast_values[0] = forecast_cumulative[0] - last_cumulative
            for i in range(1, periods):
                forecast_values[i] = forecast_cumulative[i] - forecast_cumulative[i-1]
        else:
            # For quarterly or monthly, use the derivative formula directly
            forecast_values = self._gompertz_derivative(
                np.array(forecast_periods),
                self.asymptote,
                self.displacement,
                self.growth_rate
            )
            
            # Scale to match the frequency
            if frequency == 'Q':
                forecast_values = forecast_values / 4  # Quarterly values
            elif frequency == 'M':
                forecast_values = forecast_values / 12  # Monthly values
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals (simple implementation)
        history_values = self.history['value'].values
        residuals = np.abs(
            self.history_cumulative - 
            self._gompertz_curve(np.array(self.time_periods), self.asymptote, self.displacement, self.growth_rate)
        )
        avg_residual = np.mean(residuals)
        
        lower_bound = []
        upper_bound = []
        
        for i, value in enumerate(forecast_values):
            # Increase uncertainty with time
            ci_width = avg_residual * (1 + i * 0.2)
            lower = max(value - ci_width, 0)
            upper = value + ci_width
            
            lower_bound.append(lower)
            upper_bound.append(upper)
        
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


class TechnologySCurveForecaster(BaseForecaster):
    """
    Technology S-Curve Model for multi-stage technology adoption forecasting.
    
    This model extends the standard S-curve to handle multiple phases of
    technology adoption, which is common in technology markets where
    different segments adopt at different rates.
    """
    
    def _initialize_parameters(self) -> None:
        """Initialize Technology S-Curve model parameters from configuration"""
        # Number of adoption phases
        self.n_phases = self.config.get('n_phases', 2)
        
        # Phase parameters
        self.phase_params = self.config.get('phase_params', None)
        
        # Whether to estimate phase parameters automatically
        self.auto_params = self.config.get('auto_params', True)
        
        # Market potential for each phase (if not auto-estimated)
        # If not provided, will be estimated during fitting
        self.market_potentials = self.config.get('market_potentials', None)
        
        # Whether to ensure a minimum value in forecasts
        self.ensure_minimum = self.config.get('ensure_minimum', True)
        
        # Minimum value allowed in forecast
        self.minimum_value = self.config.get('minimum_value', 0)
        
        # Store parameters and fitted data
        self.params = None
        self.history = None
        self.history_cumulative = None
        self.time_periods = None
    
    def _multi_phase_s_curve(self, t: np.ndarray, *params) -> np.ndarray:
        """
        Multi-phase S-curve function.
        
        This function combines multiple logistic curves to model different
        adoption phases in technology markets.
        
        Args:
            t: Time periods
            *params: Parameters for each phase (3 params per phase)
                For each phase i:
                    params[3*i]: Market potential for phase i
                    params[3*i+1]: Midpoint for phase i
                    params[3*i+2]: Steepness for phase i
            
        Returns:
            Predicted cumulative adoptions
        """
        result = np.zeros_like(t, dtype=float)
        
        # Each phase has 3 parameters: L (capacity), k (midpoint), and r (steepness)
        n_phases = len(params) // 3
        
        for i in range(n_phases):
            L = params[3*i]      # Market potential for this phase
            k = params[3*i+1]    # Midpoint (time when growth rate is maximum)
            r = params[3*i+2]    # Steepness
            
            # Logistic function
            phase_curve = L / (1 + np.exp(-r * (t - k)))
            result += phase_curve
        
        return result
    
    def _multi_phase_derivative(self, t: np.ndarray, *params) -> np.ndarray:
        """
        Derivative of the multi-phase S-curve (non-cumulative).
        
        Args:
            t: Time periods
            *params: Parameters for each phase
            
        Returns:
            Predicted adoption rate (non-cumulative)
        """
        result = np.zeros_like(t, dtype=float)
        
        # Each phase has 3 parameters: L (capacity), k (midpoint), and r (steepness)
        n_phases = len(params) // 3
        
        for i in range(n_phases):
            L = params[3*i]      # Market potential for this phase
            k = params[3*i+1]    # Midpoint
            r = params[3*i+2]    # Steepness
            
            # Derivative of logistic function
            exp_term = np.exp(-r * (t - k))
            phase_derivative = (L * r * exp_term) / ((1 + exp_term) ** 2)
            result += phase_derivative
        
        return result
    
    def fit(self, data: pd.DataFrame) -> 'TechnologySCurveForecaster':
        """
        Fit the Technology S-Curve model to historical data.
        
        Args:
            data: DataFrame with 'date' and 'value' columns
                'value' should be non-cumulative adoptions (not cumulative)
            
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
        
        # Calculate time periods (assuming equal time intervals)
        start_date = data['date'].min()
        self.time_periods = [(date - start_date).days / 365.25 for date in data['date']]  # Convert to years
        
        # Calculate cumulative adoptions
        cumulative = data['value'].cumsum()
        self.history_cumulative = cumulative.values
        
        # Use provided phase parameters if available
        if not self.auto_params and self.phase_params is not None and len(self.phase_params) == self.n_phases * 3:
            self.params = self.phase_params
            logger.info("Using provided phase parameters")
            self.fitted = True
            return self
        
        # Generate initial guesses for parameters
        # The challenge here is to automatically determine good starting points for multiple phases
        
        # For a single phase, it's straightforward
        if self.n_phases == 1:
            # Total market potential
            if self.market_potentials and len(self.market_potentials) >= 1:
                L = self.market_potentials[0]
            else:
                L = cumulative.iloc[-1] * 2  # Estimate total potential as 2x current adoption
            
            # Midpoint (estimate as halfway through time range)
            k = np.mean(self.time_periods)
            
            # Steepness (start with moderate value)
            r = 0.5
            
            initial_params = [L, k, r]
            
        else:
            # For multiple phases, divide the total market into phases
            initial_params = []
            
            # Analyze trend change points to identify potential phase boundaries
            try:
                from scipy.signal import find_peaks
                
                # Smooth the data to reduce noise
                window_size = max(2, len(self.time_periods) // 5)
                smoothed = pd.Series(self.history['value'].values).rolling(window=window_size, min_periods=1).mean()
                
                # Find peaks in the adoption rate (potential inflection points)
                peaks, _ = find_peaks(smoothed, distance=max(2, len(self.time_periods) // 10))
                
                # If we found peaks, use them to estimate phase boundaries
                if len(peaks) > 0 and len(peaks) <= self.n_phases:
                    # Use actual peaks found
                    peak_times = [self.time_periods[p] for p in peaks]
                else:
                    # Not enough peaks found, distribute phases evenly
                    peak_times = np.linspace(min(self.time_periods), max(self.time_periods), self.n_phases)
                
                # Adjust number of phases if needed
                self.n_phases = len(peak_times)
                
                # Estimate potential for each phase (distribute total potential)
                if self.market_potentials and len(self.market_potentials) >= self.n_phases:
                    phase_potentials = self.market_potentials[:self.n_phases]
                else:
                    # Distribute potential across phases
                    total_potential = cumulative.iloc[-1] * 2
                    phase_potentials = [total_potential / self.n_phases] * self.n_phases
                
                # Add parameters for each phase
                for i in range(self.n_phases):
                    L = phase_potentials[i]     # Potential for this phase
                    k = peak_times[i]           # Midpoint at the peak
                    r = 0.5                     # Initial steepness
                    
                    initial_params.extend([L, k, r])
                
            except Exception as e:
                logger.error(f"Error estimating phase parameters: {str(e)}")
                
                # Fall back to evenly distributed phases
                total_potential = cumulative.iloc[-1] * 2
                phase_potential = total_potential / self.n_phases

                # Distribute phases evenly across the time range
                t_range = max(self.time_periods) - min(self.time_periods)
                t_start = min(self.time_periods)

                for i in range(self.n_phases):
                    L = phase_potential
                    k = t_start + t_range * (i + 0.5) / self.n_phases
                    r = 0.5

                    initial_params.extend([L, k, r])
        
        # Set bounds for parameters
        # Each phase has three parameters: L (potential), k (midpoint), r (steepness)
        lower_bounds = []
        upper_bounds = []

        # Calculate time range for bounds
        t_range = max(self.time_periods) - min(self.time_periods)

        for i in range(self.n_phases):
            # L: Market potential (must be positive)
            lower_bounds.append(cumulative.iloc[-1] / (2 * self.n_phases))
            upper_bounds.append(cumulative.iloc[-1] * 5)

            # k: Midpoint (can be outside observed range)
            lower_bounds.append(min(self.time_periods) - t_range * 0.5)
            upper_bounds.append(max(self.time_periods) + t_range * 2)

            # r: Steepness (must be positive)
            lower_bounds.append(0.01)
            upper_bounds.append(5.0)
        
        # Fit the model
        try:
            popt, _ = curve_fit(
                self._multi_phase_s_curve, 
                self.time_periods, 
                self.history_cumulative,
                p0=initial_params,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000
            )
            self.params = popt
            
        except Exception as e:
            logger.error(f"Curve fitting failed: {str(e)}")
            logger.warning("Using initial parameters")
            self.params = initial_params
        
        # Log the fitted parameters
        logger.info(f"Fitted {self.n_phases}-phase S-curve model")
        for i in range(self.n_phases):
            L = self.params[3*i]
            k = self.params[3*i+1]
            r = self.params[3*i+2]
            logger.info(f"Phase {i+1}: Potential={L:.2f}, Midpoint={k:.2f}, Steepness={r:.4f}")
        
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
        
        # Calculate time periods for forecast
        start_date = self.history['date'].min()
        last_period = self.time_periods[-1]
        
        if frequency == 'Y':
            forecast_periods = [last_period + i for i in range(1, periods + 1)]
        elif frequency == 'Q':
            forecast_periods = [last_period + i/4 for i in range(1, periods + 1)]
        elif frequency == 'M':
            forecast_periods = [last_period + i/12 for i in range(1, periods + 1)]
        
        # Calculate cumulative adoptions for forecast periods
        forecast_cumulative = self._multi_phase_s_curve(
            np.array(forecast_periods), 
            *self.params
        )
        
        # Calculate non-cumulative adoptions (difference from previous period)
        last_cumulative = self.history_cumulative[-1]
        
        if frequency == 'Y':
            # For yearly, just take the difference between consecutive periods
            forecast_values = np.zeros(periods)
            forecast_values[0] = forecast_cumulative[0] - last_cumulative
            for i in range(1, periods):
                forecast_values[i] = forecast_cumulative[i] - forecast_cumulative[i-1]
        else:
            # For quarterly or monthly, use the derivative formula directly
            forecast_values = self._multi_phase_derivative(
                np.array(forecast_periods),
                *self.params
            )
            
            # Scale to match the frequency
            if frequency == 'Q':
                forecast_values = forecast_values / 4  # Quarterly values
            elif frequency == 'M':
                forecast_values = forecast_values / 12  # Monthly values
        
        # Apply minimum value if configured
        if self.ensure_minimum:
            forecast_values = np.maximum(forecast_values, self.minimum_value)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'value': forecast_values
        })
        
        # Create confidence intervals (simple implementation)
        history_values = self.history['value'].values
        residuals = np.abs(
            self.history_cumulative - 
            self._multi_phase_s_curve(np.array(self.time_periods), *self.params)
        )
        avg_residual = np.mean(residuals)
        
        lower_bound = []
        upper_bound = []
        
        for i, value in enumerate(forecast_values):
            # Increase uncertainty with time
            ci_width = avg_residual * (1 + i * 0.2)
            lower = max(value - ci_width, 0)
            upper = value + ci_width
            
            lower_bound.append(lower)
            upper_bound.append(upper)
        
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