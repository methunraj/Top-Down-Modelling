"""
Gradient Harmonization Algorithm Module - Advanced market trajectory smoothing

This module provides sophisticated algorithms for harmonizing growth trajectories
across countries and regions, ensuring realistic, consistent market transitions while
preserving important inflection points and market dynamics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GradientHarmonizer:
    """
    Advanced trajectory smoothing and harmonization for market forecasts
    
    This class provides sophisticated algorithms for smoothing and harmonizing
    market growth patterns while preserving key dynamics and ensuring consistency
    across countries, regions, and the global total.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the GradientHarmonizer
        
        Args:
            config_manager: Configuration manager instance for accessing settings
        """
        self.config_manager = config_manager
        
        # Get harmonization settings from configuration
        self.harmonization_settings = self.config_manager.get_value(
            'market_distribution.harmonization', {}
        )
        
        # Default settings if not specified in config
        self.default_settings = {
            'method': 'adaptive',  # Options: 'adaptive', 'monotonic', 'gaussian', 'polynomial'
            'boundary_enforcement': 'strict',  # Options: 'strict', 'relaxed', 'none'
            'smoothing_strength': 0.5,  # 0.0 to 1.0 (higher = more smoothing)
            'preserve_inflection': True,  # Whether to preserve inflection points
            'transition_zone': 2,  # Years of transition between historical and forecast
            'global_consistency': True,  # Enforce global total consistency
            'regional_consistency': True,  # Enforce regional consistency
            'tier_specific_settings': {
                'tier1': {'smoothing_strength': 0.4},
                'tier2': {'smoothing_strength': 0.5},
                'tier3': {'smoothing_strength': 0.6}
            },
            'target_growth_rates': {
                'default': 15.0,
                'tier1': 12.0,
                'tier2': 18.0,
                'tier3': 25.0
            },
            'endpoint_behavior': 'natural',  # Options: 'natural', 'zero_slope', 'match_trend'
            'inflection_detection': {
                'enabled': True,
                'sensitivity': 0.6,  # 0.0 to 1.0
                'min_prominence': 0.2
            }
        }
        
        # Merge config settings with defaults
        self._initialize_settings()
        
        # Track countries processed
        self.processed_countries = set()
        
        # Cache for harmonized trajectories
        self.trajectory_cache = {}
    
    def _initialize_settings(self) -> None:
        """
        Initialize settings by merging configuration with defaults
        """
        self.settings = self.default_settings.copy()
        
        # Update with values from configuration
        for key, value in self.harmonization_settings.items():
            if key in self.settings:
                # Handle nested dictionaries
                if isinstance(value, dict) and isinstance(self.settings[key], dict):
                    self.settings[key].update(value)
                else:
                    self.settings[key] = value
        
        # Log settings
        logger.info(f"Gradient Harmonization initialized with method: {self.settings['method']}")
        
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update harmonization settings with new values
        
        Args:
            settings: Dictionary containing updated settings
        """
        logger.info("Updating gradient harmonization settings")
        
        if not settings:
            return
            
        # Update settings dictionary
        for key, value in settings.items():
            if key in self.settings:
                # Handle nested dictionaries
                if isinstance(value, dict) and isinstance(self.settings[key], dict):
                    self.settings[key].update(value)
                else:
                    self.settings[key] = value
        
        # Reset caches
        self.trajectory_cache = {}
        self.processed_countries = set()
        
        logger.info(f"Gradient Harmonization settings updated to method: {self.settings['method']}")
        if 'smoothing_strength' in settings:
            logger.info(f"Smoothing strength updated to: {self.settings['smoothing_strength']}")
        if 'tier_specific_settings' in settings:
            logger.info(f"Updated tier-specific settings")
        if 'target_growth_rates' in settings:
            logger.info(f"Updated target growth rates")
        logger.info(f"Smoothing strength: {self.settings['smoothing_strength']}")
    
    def harmonize_market_trajectories(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply gradient harmonization to market data trajectories
        
        Args:
            market_data: DataFrame with market forecast data
            
        Returns:
            DataFrame with harmonized trajectories
        """
        logger.info("Applying gradient harmonization to market trajectories")
        
        # Create a copy to avoid modifying the original
        harmonized_df = market_data.copy()
        
        # Get column mappings
        id_col = 'idGeo'  # Default
        
        # Reset processed countries
        self.processed_countries = set()
        self.trajectory_cache = {}
        
        # Check if we have tier information
        use_tiers = 'tier' in harmonized_df.columns
        
        # Process countries only first (not regions)
        is_country = ~harmonized_df['is_region'] if 'is_region' in harmonized_df.columns else pd.Series(True, index=harmonized_df.index)
        country_ids = harmonized_df[is_country][id_col].unique()
        
        # Get all years in chronological order
        all_years = sorted(harmonized_df['Year'].unique())
        
        # Identify historical vs forecast years
        is_forecast = harmonized_df['is_forecast'] if 'is_forecast' in harmonized_df.columns else None
        
        if is_forecast is None:
            # Try to infer historical/forecast boundary from data
            # Typically, there's a change in growth pattern at this boundary
            growth_rates = self._calculate_growth_rates(harmonized_df, id_col)
            if not growth_rates.empty:
                historical_end_year = self._detect_forecast_start(growth_rates, all_years)
                historical_years = [y for y in all_years if y <= historical_end_year]
                forecast_years = [y for y in all_years if y > historical_end_year]
            else:
                # Can't infer, assume first half is historical and second half is forecast
                mid_point = len(all_years) // 2
                historical_years = all_years[:mid_point]
                forecast_years = all_years[mid_point:]
        else:
            # Use the is_forecast column to separate
            forecast_mask = harmonized_df['is_forecast'] == True
            historical_mask = ~forecast_mask
            
            historical_years = sorted(harmonized_df[historical_mask]['Year'].unique())
            forecast_years = sorted(harmonized_df[forecast_mask]['Year'].unique())
        
        # Determine transition years (where special handling is needed)
        transition_zone = self.settings['transition_zone']
        if historical_years and forecast_years:
            transition_years = []
            for year in all_years:
                if (year >= max(historical_years) - transition_zone + 1 and 
                    year <= min(forecast_years) + transition_zone - 1):
                    transition_years.append(year)
        else:
            transition_years = []
        
        logger.info(f"Historical years: {historical_years}")
        logger.info(f"Forecast years: {forecast_years}")
        logger.info(f"Transition years: {transition_years}")
        
        # Process each country
        for country_id in country_ids:
            country_mask = harmonized_df[id_col] == country_id
            country_data = harmonized_df.loc[country_mask].sort_values(by='Year')
            
            if len(country_data) < 2:
                continue
            
            # Get country tier if available
            tier = None
            if use_tiers:
                tier_values = country_data['tier'].unique()
                if len(tier_values) > 0 and not pd.isna(tier_values[0]):
                    tier = int(tier_values[0])
            
            # Select harmonization method based on settings
            smoothed_values, smoothed_growth = self._harmonize_country_trajectory(
                country_data, 
                tier,
                historical_years,
                forecast_years,
                transition_years,
                all_years
            )
            
            # Update values in the main DataFrame
            for year, value in zip(all_years, smoothed_values):
                year_mask = (harmonized_df[id_col] == country_id) & (harmonized_df['Year'] == year)
                harmonized_df.loc[year_mask, 'Value'] = value
            
            # Store growth rates if needed
            if 'Growth_Rate' in harmonized_df.columns:
                for year, growth in zip(all_years[1:], smoothed_growth):
                    year_mask = (harmonized_df[id_col] == country_id) & (harmonized_df['Year'] == year)
                    harmonized_df.loc[year_mask, 'Growth_Rate'] = growth
            
            # Mark this country as processed
            self.processed_countries.add(country_id)
        
        # Process regions if present and regional consistency is enabled
        if 'is_region' in harmonized_df.columns and self.settings['regional_consistency']:
            harmonized_df = self._process_regions(harmonized_df, id_col, all_years)
        
        # Enforce global consistency if required
        if self.settings['global_consistency']:
            harmonized_df = self._enforce_global_consistency(harmonized_df, all_years)
        
        return harmonized_df
    
    def _calculate_growth_rates(self, data: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """
        Calculate year-over-year growth rates for each country
        
        Args:
            data: DataFrame with market data
            id_col: ID column name
            
        Returns:
            DataFrame with growth rates
        """
        growth_rates = []
        
        for country_id in data[id_col].unique():
            country_mask = data[id_col] == country_id
            country_data = data.loc[country_mask].sort_values(by='Year')
            
            if len(country_data) < 2:
                continue
            
            # Calculate growth rates
            country_data = country_data.copy()
            country_data['Growth_Rate'] = country_data['Value'].pct_change() * 100
            
            # Add to results (excluding first year with NaN growth rate)
            growth_rates.append(country_data.iloc[1:])
        
        if not growth_rates:
            return pd.DataFrame()
            
        # Combine all data
        return pd.concat(growth_rates, ignore_index=True)
    
    def _detect_forecast_start(self, growth_rates: pd.DataFrame, all_years: List[int]) -> int:
        """
        Detect the likely boundary between historical and forecast data
        
        Args:
            growth_rates: DataFrame with growth rates
            all_years: List of all years in chronological order
            
        Returns:
            Year that marks the end of historical data
        """
        # Group by year and calculate median absolute deviation of growth rates
        year_stats = growth_rates.groupby('Year')['Growth_Rate'].agg(['median', 'std']).reset_index()
        
        if len(year_stats) < 3:
            # Not enough years to detect a pattern
            return all_years[len(all_years) // 2]
        
        # Look for a significant change in median or standard deviation
        changes = []
        for i in range(1, len(year_stats)):
            median_change = abs(year_stats.iloc[i]['median'] - year_stats.iloc[i-1]['median'])
            std_change = abs(year_stats.iloc[i]['std'] - year_stats.iloc[i-1]['std'])
            
            # Normalize changes
            if year_stats.iloc[i-1]['median'] != 0:
                median_change /= abs(year_stats.iloc[i-1]['median'])
            if year_stats.iloc[i-1]['std'] != 0:
                std_change /= year_stats.iloc[i-1]['std']
            
            changes.append({
                'year': year_stats.iloc[i]['Year'],
                'change_metric': median_change + std_change
            })
        
        # Find the year with the maximum change
        max_change = max(changes, key=lambda x: x['change_metric'])
        forecast_start_year = max_change['year']
        
        # The year before forecast_start_year is the last historical year
        historical_end_year = forecast_start_year - 1
        
        # Ensure it's a year in our dataset
        if historical_end_year in all_years:
            return historical_end_year
        else:
            # Find the closest year in all_years
            return min(all_years, key=lambda y: abs(y - historical_end_year))
    
    def _harmonize_country_trajectory(self, country_data: pd.DataFrame, tier: Optional[int], 
                                     historical_years: List[int], forecast_years: List[int],
                                     transition_years: List[int], all_years: List[int]) -> Tuple[List[float], List[float]]:
        """
        Harmonize trajectory for a single country
        
        Args:
            country_data: DataFrame with data for a single country
            tier: Country tier (optional)
            historical_years: List of historical years
            forecast_years: List of forecast years
            transition_years: List of transition years
            all_years: List of all years
            
        Returns:
            Tuple of (harmonized values, harmonized growth rates)
        """
        # Check if we've already processed this trajectory
        country_id = country_data['idGeo'].iloc[0]
        if country_id in self.trajectory_cache:
            return self.trajectory_cache[country_id]
        
        # Get the original values
        years_map = {year: i for i, year in enumerate(all_years)}
        values = np.zeros(len(all_years))
        
        for _, row in country_data.iterrows():
            year = row['Year']
            if year in years_map:
                values[years_map[year]] = row['Value']
        
        # Get tier-specific settings
        tier_key = f"tier{tier}" if tier is not None else None
        
        smoothing_strength = self.settings['smoothing_strength']
        if tier_key and tier_key in self.settings['tier_specific_settings']:
            tier_settings = self.settings['tier_specific_settings'][tier_key]
            if 'smoothing_strength' in tier_settings:
                smoothing_strength = tier_settings['smoothing_strength']
        
        # Get target growth rate based on tier
        if tier_key and tier_key in self.settings['target_growth_rates']:
            target_growth = self.settings['target_growth_rates'][tier_key]
        else:
            target_growth = self.settings['target_growth_rates']['default']
        
        # Select method based on settings
        method = self.settings['method']
        
        if method == 'adaptive':
            smoothed_values = self._adaptive_harmonization(values, all_years, historical_years, 
                                                          forecast_years, transition_years, 
                                                          smoothing_strength, target_growth)
        elif method == 'monotonic':
            smoothed_values = self._monotonic_harmonization(values, all_years, historical_years, 
                                                           forecast_years, transition_years,
                                                           smoothing_strength, target_growth)
        elif method == 'gaussian':
            smoothed_values = self._gaussian_harmonization(values, all_years, historical_years, 
                                                          forecast_years, transition_years,
                                                          smoothing_strength)
        elif method == 'polynomial':
            smoothed_values = self._polynomial_harmonization(values, all_years, historical_years, 
                                                            forecast_years, transition_years,
                                                            smoothing_strength)
        else:
            logger.warning(f"Unknown harmonization method: {method}, using adaptive")
            smoothed_values = self._adaptive_harmonization(values, all_years, historical_years, 
                                                          forecast_years, transition_years, 
                                                          smoothing_strength, target_growth)
        
        # Calculate growth rates from smoothed values
        smoothed_growth = []
        for i in range(1, len(smoothed_values)):
            if smoothed_values[i-1] > 0:
                growth = (smoothed_values[i] / smoothed_values[i-1] - 1) * 100
            else:
                growth = 0.0
            smoothed_growth.append(growth)
        
        # Store in cache
        self.trajectory_cache[country_id] = (smoothed_values, smoothed_growth)
        
        return smoothed_values, smoothed_growth
    
    def _adaptive_harmonization(self, values: np.ndarray, all_years: List[int], 
                               historical_years: List[int], forecast_years: List[int],
                               transition_years: List[int], smoothing_strength: float,
                               target_growth: float) -> np.ndarray:
        """
        Apply adaptive harmonization, automatically selecting the best technique
        based on the data characteristics
        
        Args:
            values: Array of market values
            all_years: List of all years
            historical_years: List of historical years
            forecast_years: List of forecast years
            transition_years: List of transition years
            smoothing_strength: Strength of smoothing to apply
            target_growth: Target long-term growth rate
            
        Returns:
            Array of harmonized values
        """
        # Detect characteristics of the time series
        # 1. Calculate growth rates
        growth_rates = np.zeros(len(values) - 1)
        for i in range(1, len(values)):
            if values[i-1] > 0:
                growth_rates[i-1] = (values[i] / values[i-1] - 1) * 100
        
        # 2. Detect volatility
        volatility = np.std(growth_rates)
        
        # 3. Check for monotonicity (consistently increasing or decreasing)
        increases = np.sum(growth_rates > 0)
        decreases = np.sum(growth_rates < 0)
        monotonic_ratio = max(increases, decreases) / len(growth_rates) if len(growth_rates) > 0 else 0
        
        # 4. Detect inflection points if enabled
        inflection_points = []
        if self.settings['inflection_detection']['enabled']:
            inflection_points = self._detect_inflection_points(values, all_years)
        
        # Select technique based on characteristics
        if monotonic_ratio > 0.8:
            # Mostly monotonic, use monotonic interpolation
            harmonized = self._monotonic_harmonization(values, all_years, historical_years, 
                                                     forecast_years, transition_years,
                                                     smoothing_strength, target_growth)
        elif volatility > 15.0:
            # High volatility, use more aggressive gaussian smoothing
            harmonized = self._gaussian_harmonization(values, all_years, historical_years, 
                                                    forecast_years, transition_years,
                                                    smoothing_strength * 1.5)
        elif len(inflection_points) > 0:
            # Significant inflection points, use polynomial to preserve them
            harmonized = self._polynomial_harmonization(values, all_years, historical_years, 
                                                      forecast_years, transition_years,
                                                      smoothing_strength)
        else:
            # Default to a balanced approach
            # Start with monotonic harmonization
            harmonized1 = self._monotonic_harmonization(values, all_years, historical_years, 
                                                      forecast_years, transition_years,
                                                      smoothing_strength, target_growth)
            
            # Also do gaussian harmonization
            harmonized2 = self._gaussian_harmonization(values, all_years, historical_years, 
                                                     forecast_years, transition_years,
                                                     smoothing_strength)
            
            # Blend the two methods
            weights = np.linspace(0.7, 0.3, len(values))  # Favor monotonic early, gaussian later
            harmonized = weights * harmonized1 + (1 - weights) * harmonized2
        
        # Apply special handling for transition years and enforce boundaries
        harmonized = self._apply_transition_and_boundaries(
            harmonized, values, all_years, historical_years, forecast_years, transition_years
        )
        
        # Apply convergence to target growth rate for later forecast years
        if forecast_years:
            harmonized = self._apply_growth_convergence(
                harmonized, all_years, forecast_years, target_growth
            )
        
        return harmonized
    
    def _monotonic_harmonization(self, values: np.ndarray, all_years: List[int], 
                                historical_years: List[int], forecast_years: List[int],
                                transition_years: List[int], smoothing_strength: float,
                                target_growth: float) -> np.ndarray:
        """
        Apply monotonic harmonization using PCHIP interpolation
        
        Args:
            values: Array of market values
            all_years: List of all years
            historical_years: List of historical years
            forecast_years: List of forecast years
            transition_years: List of transition years
            smoothing_strength: Strength of smoothing to apply
            target_growth: Target long-term growth rate
            
        Returns:
            Array of harmonized values
        """
        # Create points for interpolation
        x_points = np.array(all_years, dtype=float)
        y_points = np.array(values, dtype=float)
        
        # Apply pre-smoothing to reduce noise
        if smoothing_strength > 0:
            # Calculate window size based on smoothing strength
            window_size = max(3, int(len(y_points) * smoothing_strength * 0.2) * 2 + 1)
            window_size = min(window_size, len(y_points) - 1)  # Ensure window isn't too large
            
            # Apply smoothing only to historical data for the first pass
            historical_indices = [i for i, year in enumerate(all_years) if year in historical_years]
            
            if historical_indices:
                # Use pandas rolling window for flexibility
                series = pd.Series(y_points)
                smoothed = series.copy()
                
                # Smooth only historical indices
                for i in historical_indices:
                    if i < window_size // 2 or i >= len(y_points) - window_size // 2:
                        continue  # Skip edges without enough data
                    
                    window_values = series.iloc[i - window_size // 2:i + window_size // 2 + 1]
                    smoothed.iloc[i] = window_values.mean()
                
                y_points = smoothed.values
        
        # Use monotonic interpolation with PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
        try:
            # Create the interpolator with original points
            interpolator = PchipInterpolator(x_points, y_points)
            
            # Generate smoother curve by evaluating at intermediate points
            x_dense = np.linspace(min(x_points), max(x_points), len(x_points) * 5)
            y_dense = interpolator(x_dense)
            
            # Apply additional smoothing to the dense curve
            if smoothing_strength > 0:
                sigma = max(1.0, smoothing_strength * 5.0)
                y_dense = gaussian_filter1d(y_dense, sigma=sigma)
            
            # Re-sample at the original years
            interpolator_smooth = PchipInterpolator(x_dense, y_dense)
            harmonized = interpolator_smooth(x_points)
            
            # Ensure all values are non-negative
            harmonized = np.maximum(harmonized, 0)
            
        except Exception as e:
            logger.warning(f"Error in monotonic interpolation: {str(e)}")
            # Fallback to original values
            harmonized = y_points
        
        # Apply special handling for transition years and enforce boundaries
        harmonized = self._apply_transition_and_boundaries(
            harmonized, values, all_years, historical_years, forecast_years, transition_years
        )
        
        # Apply convergence to target growth rate for later forecast years
        if forecast_years:
            harmonized = self._apply_growth_convergence(
                harmonized, all_years, forecast_years, target_growth
            )
        
        return harmonized
    
    def _gaussian_harmonization(self, values: np.ndarray, all_years: List[int], 
                               historical_years: List[int], forecast_years: List[int],
                               transition_years: List[int], smoothing_strength: float) -> np.ndarray:
        """
        Apply gaussian smoothing to the time series
        
        Args:
            values: Array of market values
            all_years: List of all years
            historical_years: List of historical years
            forecast_years: List of forecast years
            transition_years: List of transition years
            smoothing_strength: Strength of smoothing to apply
            
        Returns:
            Array of harmonized values
        """
        # Calculate sigma based on smoothing strength and number of years
        sigma = max(0.5, smoothing_strength * 3.0)
        
        # Get indices for different time periods
        historical_indices = np.array([i for i, year in enumerate(all_years) if year in historical_years])
        forecast_indices = np.array([i for i, year in enumerate(all_years) if year in forecast_years])
        
        # Apply different smoothing to historical and forecast values
        harmonized = values.copy()
        
        # Apply stronger smoothing to forecast values
        if len(forecast_indices) > 0:
            if self.settings['boundary_enforcement'] != 'strict':
                # Apply Gaussian filter to all values
                harmonized = gaussian_filter1d(values, sigma=sigma)
            else:
                # Apply Gaussian filter separately to historical and forecast
                if len(historical_indices) > 0:
                    # Smoother historical data with lighter smoothing
                    hist_sigma = sigma * 0.7
                    if len(historical_indices) > 3:  # Need enough points for effective smoothing
                        hist_values = values[historical_indices]
                        hist_smoothed = gaussian_filter1d(hist_values, sigma=hist_sigma)
                        harmonized[historical_indices] = hist_smoothed
                
                if len(forecast_indices) > 0:
                    # Stronger smoothing for forecast period
                    forecast_sigma = sigma * 1.3
                    if len(forecast_indices) > 3:  # Need enough points for effective smoothing
                        forecast_values = values[forecast_indices]
                        forecast_smoothed = gaussian_filter1d(forecast_values, sigma=forecast_sigma)
                        harmonized[forecast_indices] = forecast_smoothed
        else:
            # Only historical data
            harmonized = gaussian_filter1d(values, sigma=sigma)
        
        # Ensure non-negative values
        harmonized = np.maximum(harmonized, 0)
        
        # Apply special handling for transition years and enforce boundaries
        harmonized = self._apply_transition_and_boundaries(
            harmonized, values, all_years, historical_years, forecast_years, transition_years
        )
        
        return harmonized
    
    def _polynomial_harmonization(self, values: np.ndarray, all_years: List[int], 
                                 historical_years: List[int], forecast_years: List[int],
                                 transition_years: List[int], smoothing_strength: float) -> np.ndarray:
        """
        Apply polynomial harmonization using Akima spline
        
        Args:
            values: Array of market values
            all_years: List of all years
            historical_years: List of historical years
            forecast_years: List of forecast years
            transition_years: List of transition years
            smoothing_strength: Strength of smoothing to apply
            
        Returns:
            Array of harmonized values
        """
        # Create points for interpolation
        x_points = np.array(all_years, dtype=float)
        y_points = np.array(values, dtype=float)
        
        # Apply pre-smoothing to reduce noise
        if smoothing_strength > 0:
            # Calculate window size based on smoothing strength
            window_size = max(3, int(len(y_points) * smoothing_strength * 0.3) * 2 + 1)
            window_size = min(window_size, len(y_points) - 1)  # Ensure window isn't too large
            
            # Apply moving average smoothing
            y_smoothed = pd.Series(y_points).rolling(
                window=window_size, center=True, min_periods=1
            ).mean().values
            
            # Control how much of the smoothing to apply based on smoothing strength
            alpha = smoothing_strength
            y_points = (1 - alpha) * y_points + alpha * y_smoothed
        
        # Use Akima spline for smoother interpolation that preserves important inflection points
        try:
            # Detect and preserve key points if inflection detection is enabled
            if self.settings['inflection_detection']['enabled']:
                inflection_points = self._detect_inflection_points(values, all_years)
                
                # If we found inflection points, use them for improved interpolation
                if inflection_points:
                    # Add more weight to inflection points by duplicating them in the input
                    x_with_inflections = x_points.copy()
                    y_with_inflections = y_points.copy()
                    
                    for year, idx in inflection_points:
                        # Add points slightly before and after to influence the curve
                        idx_in_x = list(all_years).index(year)
                        x_with_inflections = np.append(x_with_inflections, [year - 0.1, year + 0.1])
                        y_with_inflections = np.append(y_with_inflections, [y_points[idx_in_x], y_points[idx_in_x]])
                    
                    # Sort by x for interpolation
                    sort_idx = np.argsort(x_with_inflections)
                    x_with_inflections = x_with_inflections[sort_idx]
                    y_with_inflections = y_with_inflections[sort_idx]
                    
                    # Create the interpolator with the enriched points
                    interpolator = Akima1DInterpolator(x_with_inflections, y_with_inflections)
                else:
                    # No inflection points, use regular interpolation
                    interpolator = Akima1DInterpolator(x_points, y_points)
            else:
                # Inflection detection disabled, use regular interpolation
                interpolator = Akima1DInterpolator(x_points, y_points)
            
            # Generate the harmonized values
            harmonized = interpolator(x_points)
            
            # Ensure all values are non-negative
            harmonized = np.maximum(harmonized, 0)
            
        except Exception as e:
            logger.warning(f"Error in polynomial interpolation: {str(e)}")
            # Fallback to original values
            harmonized = y_points
        
        # Apply special handling for transition years and enforce boundaries
        harmonized = self._apply_transition_and_boundaries(
            harmonized, values, all_years, historical_years, forecast_years, transition_years
        )
        
        return harmonized
    
    def _detect_inflection_points(self, values: np.ndarray, all_years: List[int]) -> List[Tuple[int, int]]:
        """
        Detect inflection points in the time series
        
        Args:
            values: Array of market values
            all_years: List of all years
            
        Returns:
            List of (year, index) tuples representing inflection points
        """
        if len(values) < 5:
            return []  # Not enough points to detect inflection
        
        # Calculate first and second derivatives
        first_derivative = np.diff(values)
        second_derivative = np.diff(first_derivative)
        
        # Detect sign changes in the second derivative
        sign_changes = np.zeros_like(second_derivative, dtype=bool)
        for i in range(1, len(second_derivative)):
            if second_derivative[i-1] * second_derivative[i] < 0:
                sign_changes[i] = True
        
        # Extract indices of sign changes
        change_indices = np.where(sign_changes)[0]
        
        # Filter for significant inflection points
        inflection_points = []
        sensitivity = self.settings['inflection_detection']['sensitivity']
        min_prominence = self.settings['inflection_detection']['min_prominence']
        
        for idx in change_indices:
            # Need +2 because second_derivative is 2 elements shorter than values
            value_idx = idx + 2
            if value_idx >= len(values):
                continue
                
            # Calculate a measure of "prominence" - how significant this inflection is
            # Use average of adjacent second derivatives as a measure
            if idx > 0 and idx < len(second_derivative) - 1:
                prominence = abs(second_derivative[idx] - second_derivative[idx-1])
                prominence += abs(second_derivative[idx+1] - second_derivative[idx])
                prominence /= 2
                
                # Scale by the value to make it relative
                if values[value_idx] > 0:
                    prominence /= values[value_idx]
                
                # Only keep significant inflections based on sensitivity and min_prominence
                if prominence > min_prominence * (2 - sensitivity):
                    inflection_points.append((all_years[value_idx], value_idx))
        
        return inflection_points
    
    def _apply_transition_and_boundaries(self, harmonized: np.ndarray, original: np.ndarray,
                                       all_years: List[int], historical_years: List[int], 
                                       forecast_years: List[int], transition_years: List[int]) -> np.ndarray:
        """
        Apply special handling for transition years and enforce boundaries
        
        Args:
            harmonized: Array of harmonized values
            original: Array of original values
            all_years: List of all years
            historical_years: List of historical years
            forecast_years: List of forecast years
            transition_years: List of transition years
            
        Returns:
            Array of harmonized values with transition handling and boundaries enforced
        """
        result = harmonized.copy()
        
        # Get indices for different time periods
        historical_indices = np.array([i for i, year in enumerate(all_years) if year in historical_years])
        forecast_indices = np.array([i for i, year in enumerate(all_years) if year in forecast_years])
        transition_indices = np.array([i for i, year in enumerate(all_years) if year in transition_years])
        
        # Apply boundary enforcement
        boundary_method = self.settings['boundary_enforcement']
        
        if boundary_method == 'strict':
            # Keep historical data exactly as original
            if len(historical_indices) > 0:
                result[historical_indices] = original[historical_indices]
                
        elif boundary_method == 'relaxed':
            # Keep the first and last historical points fixed, allow smoothing in between
            if len(historical_indices) > 0:
                # Fix first historical point
                result[historical_indices[0]] = original[historical_indices[0]]
                
                # Fix last historical point if it's not part of the transition zone
                last_historical = historical_indices[-1]
                if all_years[last_historical] not in transition_years:
                    result[last_historical] = original[last_historical]
        
        # Apply special handling for transition years
        if len(transition_indices) > 0:
            # Use weighted average between original and harmonized for transition years
            for idx in transition_indices:
                # Calculate position in transition zone
                year = all_years[idx]
                if year in historical_years:
                    # For historical transitions, favor original more
                    position = np.array(transition_years).tolist().index(year) / len(transition_years)
                    weight = 1.0 - position
                else:
                    # For forecast transitions, gradually shift from original to harmonized
                    position = np.array(transition_years).tolist().index(year) / len(transition_years)
                    weight = 1.0 - position
                
                # Apply weighted average
                result[idx] = weight * original[idx] + (1.0 - weight) * harmonized[idx]
        
        return result
    
    def _apply_growth_convergence(self, harmonized: np.ndarray, all_years: List[int], 
                                forecast_years: List[int], target_growth: float) -> np.ndarray:
        """
        Apply convergence to target growth rate for later forecast years
        
        Args:
            harmonized: Array of harmonized values
            all_years: List of all years
            forecast_years: List of forecast years
            target_growth: Target annual growth rate percentage
            
        Returns:
            Array with convergence to target growth applied
        """
        result = harmonized.copy()
        
        # Only apply if we have enough forecast years
        if len(forecast_years) < 3:
            return result
        
        # Get indices for forecast years
        forecast_indices = np.array([i for i, year in enumerate(all_years) if year in forecast_years])
        
        # Calculate a gradual convergence factor for each year
        target_factor = 1.0 + target_growth / 100.0  # Convert percentage to factor
        
        # Skip the first forecast year to maintain continuity
        for i in range(1, len(forecast_indices)):
            idx = forecast_indices[i]
            prev_idx = idx - 1
            
            # Calculate position in forecast period (0 to 1)
            position = i / (len(forecast_indices) - 1) if len(forecast_indices) > 1 else 1.0
            
            # Calculate weights for blending
            # Initial growth is maintained, gradually shifting to target growth
            original_weight = max(0.0, 1.0 - position * 2)
            target_weight = 1.0 - original_weight
            
            # Calculate growth factor
            if result[prev_idx] > 0:
                current_factor = result[idx] / result[prev_idx]
                
                # Blend current factor with target factor
                blended_factor = original_weight * current_factor + target_weight * target_factor
                
                # Apply the blended factor
                result[idx] = result[prev_idx] * blended_factor
        
        return result
    
    def _process_regions(self, market_data: pd.DataFrame, id_col: str, all_years: List[int]) -> pd.DataFrame:
        """
        Process regions ensuring consistency with constituent countries
        
        Args:
            market_data: DataFrame with market data
            id_col: ID column name
            all_years: List of all years
            
        Returns:
            DataFrame with processed regional data
        """
        # Create a copy to avoid modifying the original
        result_df = market_data.copy()
        
        # Find region definitions
        region_definitions = {}
        
        # Try to get regions from configuration
        region_hierarchy = self.config_manager.get_region_definitions()
        
        if region_hierarchy:
            # Config-based region definitions
            # First, build a mapping from country to regions
            country_to_regions = {}
            
            # Map regions to their constituent entities
            for region, constituents in region_hierarchy.items():
                for constituent in constituents:
                    if constituent not in country_to_regions:
                        country_to_regions[constituent] = []
                    country_to_regions[constituent].append(region)
            
            # Now build the reverse mapping from region to countries/subregions
            for region, constituents in region_hierarchy.items():
                region_definitions[region] = constituents
        else:
            # No region definitions in config, try to infer from market_data
            is_region = result_df['is_region'] if 'is_region' in result_df.columns else None
            is_subregion = result_df['is_subregion'] if 'is_subregion' in result_df.columns else None
            
            if is_region is not None:
                # Look for parent_region fields
                if 'parent_region' in result_df.columns:
                    # Build region definitions from parent_region field
                    region_rows = result_df[is_region]
                    
                    for _, region_row in region_rows.iterrows():
                        region_id = region_row[id_col]
                        region_name = region_row['Country']
                        
                        # Find all entities that have this region as parent
                        children_mask = result_df['parent_region'] == region_id
                        children = result_df.loc[children_mask, id_col].unique()
                        
                        region_definitions[region_name] = list(children)
        
        # Process each region
        for region_name, constituents in region_definitions.items():
            # Find region ID
            region_rows = result_df[result_df['Country'] == region_name]
            
            if region_rows.empty:
                continue
            
            region_id = region_rows[id_col].iloc[0]
            
            # Skip if we don't have the region in our data
            if region_id not in result_df[id_col].values:
                continue
            
            # Find all constituent country/region IDs
            constituent_ids = []
            for constituent in constituents:
                constituent_rows = result_df[result_df['Country'] == constituent]
                if not constituent_rows.empty:
                    constituent_ids.append(constituent_rows[id_col].iloc[0])
            
            # Skip if no constituent countries/regions found
            if not constituent_ids:
                continue
            
            # Recompute region values for each year based on constituents
            for year in all_years:
                year_mask = result_df['Year'] == year
                
                # Get values for constituents in this year
                constituent_values = result_df[year_mask & result_df[id_col].isin(constituent_ids)]['Value']
                
                # Compute total
                region_total = constituent_values.sum()
                
                # Update region value
                region_mask = (result_df[id_col] == region_id) & (result_df['Year'] == year)
                result_df.loc[region_mask, 'Value'] = region_total
        
        return result_df
    
    def _enforce_global_consistency(self, market_data: pd.DataFrame, all_years: List[int]) -> pd.DataFrame:
        """
        Enforce global consistency by adjusting country values to match global total
        
        Args:
            market_data: DataFrame with market data
            all_years: List of all years
            
        Returns:
            DataFrame with global consistency enforced
        """
        # Create a copy to avoid modifying the original
        result_df = market_data.copy()
        
        # Check if we have global totals
        has_global = False
        global_id = None
        
        # Try to find the global total row
        global_terms = ['world', 'global', 'total', 'worldwide']
        for term in global_terms:
            global_rows = result_df[result_df['Country'].str.lower().str.contains(term, na=False)]
            if not global_rows.empty:
                has_global = True
                global_id = global_rows['idGeo'].iloc[0]
                break
        
        if not has_global:
            return result_df  # No global total, nothing to enforce
        
        # Get the global values for each year
        global_values = {}
        for year in all_years:
            global_mask = (result_df['idGeo'] == global_id) & (result_df['Year'] == year)
            if global_mask.any():
                global_values[year] = result_df.loc[global_mask, 'Value'].iloc[0]
        
        # Adjust country values to match global total
        for year in all_years:
            if year not in global_values:
                continue
                
            global_total = global_values[year]
            
            # Get all rows for this year (excluding global and regions)
            is_country = ~result_df['is_region'] if 'is_region' in result_df.columns else pd.Series(True, index=result_df.index)
            year_mask = (result_df['Year'] == year) & is_country & (result_df['idGeo'] != global_id)
            
            country_rows = result_df.loc[year_mask]
            country_total = country_rows['Value'].sum()
            
            if country_total > 0 and abs(country_total - global_total) / global_total > 0.001:
                # Need to adjust country values
                scaling_factor = global_total / country_total
                
                # Apply scaling factor
                result_df.loc[year_mask, 'Value'] = result_df.loc[year_mask, 'Value'] * scaling_factor
        
        return result_df