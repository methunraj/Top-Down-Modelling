"""
Market Dynamics Engine - Advanced market phase and growth modeling

This module provides a sophisticated engine for modeling country-specific market growth
patterns, automatically detecting market phases, implementing appropriate growth constraints,
and handling market disruptions and inflection points.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import logging
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketDynamicsEngine:
    """
    Advanced market dynamics modeling engine
    
    This class provides functionality to model country-specific growth patterns,
    automatically detect market phases, implement appropriate growth constraints,
    and handle market disruptions and step changes.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the MarketDynamicsEngine
        
        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager
        
        # Market phase definitions
        self.market_phases = {
            'introduction': {
                'description': 'Introduction Phase',
                'growth_characteristics': {
                    'min_growth_rate': -20.0,
                    'max_growth_rate': 100.0,
                    'volatility_tolerance': 0.5,
                    'typical_duration': (1, 3)  # years
                }
            },
            'growth': {
                'description': 'Growth Phase',
                'growth_characteristics': {
                    'min_growth_rate': 15.0,
                    'max_growth_rate': 60.0,
                    'volatility_tolerance': 0.3,
                    'typical_duration': (3, 7)  # years
                }
            },
            'maturity': {
                'description': 'Maturity Phase',
                'growth_characteristics': {
                    'min_growth_rate': -5.0,
                    'max_growth_rate': 20.0,
                    'volatility_tolerance': 0.2,
                    'typical_duration': (5, 15)  # years
                }
            },
            'decline': {
                'description': 'Decline Phase',
                'growth_characteristics': {
                    'min_growth_rate': -50.0,
                    'max_growth_rate': 5.0,
                    'volatility_tolerance': 0.4,
                    'typical_duration': (3, 10)  # years
                }
            },
            'disruption': {
                'description': 'Market Disruption',
                'growth_characteristics': {
                    'min_growth_rate': -70.0,
                    'max_growth_rate': 150.0,
                    'volatility_tolerance': 0.8,
                    'typical_duration': (1, 3)  # years
                }
            }
        }
        
        # Growth model templates
        self.growth_models = {
            'bass': self._bass_diffusion_model,
            'gompertz': self._gompertz_model,
            'logistic': self._logistic_model,
            'exponential': self._exponential_model,
            'linear': self._linear_model
        }
        
        # Growth model parameters by country and region
        self.country_models = {}
        self.region_models = {}
        
        # Market phase assignments
        self.market_phases_by_country = {}
        self.market_phases_by_region = {}
        
        # Inflection points
        self.inflection_points = {}
        
        # Load settings from config manager if provided
        if config_manager:
            self._load_settings()
    
    def _load_settings(self):
        """Load dynamics settings from configuration"""
        if not self.config_manager:
            return
            
        # Get market dynamics settings
        distribution_settings = self.config_manager.get_market_distribution_settings()
        dynamics_settings = distribution_settings.get('market_dynamics', {})
        
        # Update market phase definitions if provided
        custom_phases = dynamics_settings.get('market_phases', {})
        for phase, settings in custom_phases.items():
            if phase in self.market_phases:
                # Update existing phase
                for key, value in settings.items():
                    self.market_phases[phase][key] = value
            else:
                # Add new phase
                self.market_phases[phase] = settings
        
        # Update growth models if provided
        custom_models = dynamics_settings.get('growth_models', {})
        for model, enabled in custom_models.items():
            if not enabled and model in self.growth_models:
                # Remove disabled model
                del self.growth_models[model]
    
    def analyze_market_dynamics(self, market_data: pd.DataFrame, 
                              id_col: str = 'idGeo', 
                              country_col: str = 'Country',
                              year_col: str = 'Year',
                              value_col: str = 'Value') -> pd.DataFrame:
        """
        Analyze market dynamics and determine phases for each country/region
        
        Args:
            market_data: DataFrame with market data
            id_col: Name of ID column
            country_col: Name of country name column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            DataFrame with added market phase and growth model information
        """
        # Make a copy to avoid modifying the original
        result_df = market_data.copy()
        
        # Ensure market data has required columns
        required_cols = [id_col, country_col, year_col, value_col]
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Required column '{col}' not found in market data")
        
        # Calculate growth rates if not present
        if 'growth_rate' not in result_df.columns:
            result_df = self._calculate_growth_rates(result_df, country_col, year_col, value_col)
        
        # Add volatility metric
        result_df = self._calculate_volatility(result_df, country_col, year_col, 'growth_rate')
        
        # Detect market phases for each country and year
        result_df = self._detect_market_phases(result_df, country_col, year_col, value_col, 'growth_rate', 'volatility')
        
        # Detect inflection points
        inflection_points = self._detect_inflection_points(result_df, country_col, year_col, value_col)
        self.inflection_points = inflection_points
        
        # Fit growth models for each country
        country_models = self._fit_growth_models(result_df, country_col, year_col, value_col)
        self.country_models = country_models
        
        # Add model projections to the data
        result_df = self._add_model_projections(result_df, country_col, year_col, value_col)
        
        # Handle regions if present
        if 'region_type' in result_df.columns:
            region_data = result_df[result_df['region_type'] == 'region']
            
            if not region_data.empty:
                # Analyze region dynamics
                region_phases = self._detect_market_phases(region_data, country_col, year_col, value_col, 'growth_rate', 'volatility')
                self.market_phases_by_region = {region: phases for region, phases in region_phases.items()}
                
                # Fit growth models for regions
                region_models = self._fit_growth_models(region_data, country_col, year_col, value_col)
                self.region_models = region_models
        
        return result_df
    
    def apply_dynamic_constraints(self, market_data: pd.DataFrame,
                                id_col: str = 'idGeo', 
                                country_col: str = 'Country',
                                year_col: str = 'Year',
                                value_col: str = 'Value') -> pd.DataFrame:
        """
        Apply growth constraints based on market dynamics phases
        
        Args:
            market_data: DataFrame with market data
            id_col: Name of ID column
            country_col: Name of country name column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            DataFrame with constrained growth rates
        """
        # Make a copy to avoid modifying the original
        result_df = market_data.copy()
        
        # Ensure we have market phase data
        if 'market_phase' not in result_df.columns:
            # Analyze dynamics first
            result_df = self.analyze_market_dynamics(result_df, id_col, country_col, year_col, value_col)
        
        # Get years
        years = sorted(result_df[year_col].unique())
        
        # Separate historical and forecast years (assume last year is dividing point)
        historical_years = years[:-1]
        forecast_years = years[-1:]
        
        # Check if we need to calculate growth rates
        if 'growth_rate' not in result_df.columns:
            result_df = self._calculate_growth_rates(result_df, country_col, year_col, value_col)
        
        # Process each country
        for country in result_df[country_col].unique():
            country_data = result_df[result_df[country_col] == country].copy()
            
            # Skip if not enough data
            if len(country_data) < 2:
                continue
            
            # Check market phase for the latest year
            latest_year = max(country_data[year_col])
            latest_phase_data = country_data[country_data[year_col] == latest_year]['market_phase']
            if len(latest_phase_data) == 0:
                continue
            latest_phase = latest_phase_data.iloc[0]
            
            # Get growth constraints for this phase
            if latest_phase in self.market_phases:
                phase_constraints = self.market_phases[latest_phase]['growth_characteristics']
                min_growth = phase_constraints['min_growth_rate']
                max_growth = phase_constraints['max_growth_rate']
                
                # Apply constraints to forecast years
                for year in forecast_years:
                    mask = (result_df[country_col] == country) & (result_df[year_col] == year)
                    
                    # Skip if no data for this year
                    if not any(mask):
                        continue
                    
                    current_growth = result_df.loc[mask, 'growth_rate'].iloc[0]
                    
                    # Apply constraints
                    if current_growth < min_growth:
                        result_df.loc[mask, 'growth_rate'] = min_growth
                        logger.debug(f"Constrained growth for {country} in {year} to minimum: {min_growth}%")
                    elif current_growth > max_growth:
                        result_df.loc[mask, 'growth_rate'] = max_growth
                        logger.debug(f"Constrained growth for {country} in {year} to maximum: {max_growth}%")
                    
                    # Recalculate value based on constrained growth
                    if year > historical_years[-1]:
                        # Find previous year value
                        prev_year = max([y for y in years if y < year])
                        prev_mask = (result_df[country_col] == country) & (result_df[year_col] == prev_year)
                        
                        if any(prev_mask):
                            prev_value = result_df.loc[prev_mask, value_col].iloc[0]
                            constrained_growth = result_df.loc[mask, 'growth_rate'].iloc[0] / 100.0
                            new_value = prev_value * (1 + constrained_growth)
                            
                            # Update value
                            result_df.loc[mask, value_col] = new_value
        
        return result_df
    
    def apply_market_specific_models(self, market_data: pd.DataFrame,
                                   forecast_years: List[int],
                                   id_col: str = 'idGeo', 
                                   country_col: str = 'Country',
                                   year_col: str = 'Year',
                                   value_col: str = 'Value') -> pd.DataFrame:
        """
        Apply best-fit market-specific growth models to forecast future values
        
        Args:
            market_data: DataFrame with historical market data
            forecast_years: List of years to forecast
            id_col: Name of ID column
            country_col: Name of country name column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            DataFrame with model-based forecasts
        """
        # Make a copy to avoid modifying the original
        historical_df = market_data.copy()
        
        # Ensure we have fitted models
        if not self.country_models:
            # Fit models first
            self._fit_growth_models(historical_df, country_col, year_col, value_col)
        
        # Get historical years
        historical_years = sorted(historical_df[year_col].unique())
        
        # Create forecast DataFrame
        forecast_rows = []
        
        # Process each country
        for country in historical_df[country_col].unique():
            country_data = historical_df[historical_df[country_col] == country]
            
            # Skip if not enough data
            if len(country_data) < 2:
                continue
            
            # Get country ID
            country_id = country_data[id_col].iloc[0]
            
            # Get country's growth model
            country_model = self.country_models.get(country, None)
            
            if country_model:
                model_name = country_model['model_name']
                params = country_model['params']
                
                # Generate forecasts for each year
                for year in forecast_years:
                    # Use model to predict value
                    model_func = self.growth_models[model_name]
                    year_index = year - historical_years[0]
                    
                    if model_name == 'bass':
                        predicted_value = model_func(year_index, *params)
                    else:
                        predicted_value = model_func(year_index, *params)
                    
                    # Calculate growth rate
                    if year > historical_years[-1]:
                        # Find closest previous year
                        prev_year = max([y for y in forecast_years + historical_years if y < year])
                        
                        # Find previous value
                        prev_rows = [row for row in forecast_rows if row[country_col] == country and row[year_col] == prev_year]
                        
                        if prev_rows:
                            prev_value = prev_rows[0][value_col]
                        else:
                            # Use last historical value
                            prev_value = country_data[country_data[year_col] == max(country_data[year_col])][value_col].iloc[0]
                        
                        # Calculate growth rate
                        growth_rate = (predicted_value / prev_value - 1) * 100
                    else:
                        growth_rate = 0.0
                    
                    # Add to forecast rows
                    forecast_rows.append({
                        id_col: country_id,
                        country_col: country,
                        year_col: year,
                        value_col: predicted_value,
                        'growth_rate': growth_rate,
                        'model_name': model_name,
                        'market_phase': self._predict_market_phase(country, year, growth_rate)
                    })
            else:
                # No model available, use simple trend projection
                latest_data = country_data.sort_values(year_col).tail(3)
                
                if len(latest_data) >= 2:
                    # Calculate average growth rate
                    avg_growth = latest_data['growth_rate'].mean() if 'growth_rate' in latest_data.columns else 5.0
                    latest_value = latest_data[value_col].iloc[-1]
                    
                    # Generate forecasts for each year
                    for year in forecast_years:
                        years_ahead = year - historical_years[-1]
                        predicted_value = latest_value * (1 + avg_growth/100) ** years_ahead
                        
                        # Add to forecast rows
                        forecast_rows.append({
                            id_col: country_id,
                            country_col: country,
                            year_col: year,
                            value_col: predicted_value,
                            'growth_rate': avg_growth,
                            'model_name': 'trend',
                            'market_phase': self._predict_market_phase(country, year, avg_growth)
                        })
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecast_rows)
        
        # Combine historical and forecast data
        result_df = pd.concat([historical_df, forecast_df], ignore_index=True)
        
        return result_df
    
    def _calculate_growth_rates(self, data: pd.DataFrame, 
                              country_col: str, 
                              year_col: str,
                              value_col: str) -> pd.DataFrame:
        """
        Calculate year-over-year growth rates for each country/region
        
        Args:
            data: DataFrame with market data
            country_col: Name of country column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            DataFrame with added growth_rate column
        """
        result = data.copy()
        result['growth_rate'] = 0.0
        
        # Process each country/region
        for entity in result[country_col].unique():
            entity_data = result[result[country_col] == entity].sort_values(year_col)
            
            if len(entity_data) > 1:
                # Calculate year-over-year growth rates
                entity_data['growth_rate'] = entity_data[value_col].pct_change() * 100
                
                # Update the main DataFrame
                for idx, row in entity_data.iterrows():
                    result.loc[idx, 'growth_rate'] = row['growth_rate']
        
        return result
    
    def _calculate_volatility(self, data: pd.DataFrame, 
                           country_col: str, 
                           year_col: str,
                           rate_col: str) -> pd.DataFrame:
        """
        Calculate growth rate volatility for each country/region
        
        Args:
            data: DataFrame with market data
            country_col: Name of country column
            year_col: Name of year column
            rate_col: Name of growth rate column
            
        Returns:
            DataFrame with added volatility column
        """
        result = data.copy()
        result['volatility'] = 0.0
        
        # Process each country/region
        for entity in result[country_col].unique():
            entity_data = result[result[country_col] == entity].sort_values(year_col)
            
            if len(entity_data) > 2:
                # Calculate rolling standard deviation of growth rates
                # with a window of 3 years (or less if fewer years available)
                window_size = min(3, len(entity_data))
                
                # Use rolling standard deviation as volatility measure
                rolling_std = entity_data[rate_col].rolling(window=window_size, min_periods=2).std()
                
                # Fill NaN values with 0
                rolling_std = rolling_std.fillna(0)
                
                # Update the main DataFrame
                for idx, std_val in zip(entity_data.index, rolling_std):
                    result.loc[idx, 'volatility'] = std_val
        
        return result
    
    def _detect_market_phases(self, data: pd.DataFrame, 
                           country_col: str, 
                           year_col: str,
                           value_col: str,
                           rate_col: str,
                           volatility_col: str) -> pd.DataFrame:
        """
        Detect market phases for each country/region and year
        
        Args:
            data: DataFrame with market data
            country_col: Name of country column
            year_col: Name of year column
            value_col: Name of value column
            rate_col: Name of growth rate column
            volatility_col: Name of volatility column
            
        Returns:
            DataFrame with added market_phase column
        """
        result = data.copy()
        result['market_phase'] = 'unknown'
        
        # Store phase assignments for each country
        country_phases = {}
        
        # Process each country/region
        for entity in result[country_col].unique():
            entity_data = result[result[country_col] == entity].sort_values(year_col)
            
            if len(entity_data) < 3:
                # Not enough data for phase detection
                for idx in entity_data.index:
                    result.loc[idx, 'market_phase'] = 'insufficient_data'
                continue
            
            # Calculate market share if not present
            if 'market_share' not in entity_data.columns:
                total_by_year = data.groupby(year_col)[value_col].sum()
                
                for idx, row in entity_data.iterrows():
                    year = row[year_col]
                    value = row[value_col]
                    total_value = total_by_year[year]
                    share = value / total_value * 100 if total_value > 0 else 0
                    result.loc[idx, 'market_share'] = share
            
            # Extract features for phase detection
            features = []
            years = []
            
            for idx, row in entity_data.iterrows():
                growth_rate = row[rate_col] if not pd.isna(row[rate_col]) else 0
                volatility = row[volatility_col] if not pd.isna(row[volatility_col]) else 0
                market_share = row['market_share'] if 'market_share' in row and not pd.isna(row['market_share']) else 0
                
                feature_vector = [growth_rate, volatility, market_share]
                features.append(feature_vector)
                years.append(row[year_col])
            
            # Convert to numpy array
            features = np.array(features)
            
            # Handle missing values
            features = np.nan_to_num(features)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Detect phases using GMM clustering
            phase_labels = []
            
            # Try to find optimal number of components (2-5)
            best_bic = np.inf
            best_gmm = None
            
            for n_components in range(2, min(6, len(scaled_features) + 1)):
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(scaled_features)
                bic = gmm.bic(scaled_features)
                
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            
            if best_gmm:
                # Assign phases based on GMM
                cluster_labels = best_gmm.predict(scaled_features)
                
                # Map clusters to market phases
                cluster_phases = self._map_clusters_to_phases(cluster_labels, features)
                
                # Assign phases in time order
                phase_sequence = []
                
                for i, year in enumerate(years):
                    cluster = cluster_labels[i]
                    phase = cluster_phases[cluster]
                    phase_sequence.append(phase)
                    
                    # Update result DataFrame
                    year_mask = (result[country_col] == entity) & (result[year_col] == year)
                    result.loc[year_mask, 'market_phase'] = phase
                
                # Post-process to ensure logical phase transitions
                smoothed_phases = self._smooth_phase_sequence(phase_sequence)
                
                for i, year in enumerate(years):
                    year_mask = (result[country_col] == entity) & (result[year_col] == year)
                    result.loc[year_mask, 'market_phase'] = smoothed_phases[i]
                
                # Store phases for this country
                country_phases[entity] = dict(zip(years, smoothed_phases))
        
        # Store market phases by country
        self.market_phases_by_country = country_phases
        
        return result
    
    def _map_clusters_to_phases(self, cluster_labels: np.ndarray, features: np.ndarray) -> Dict[int, str]:
        """
        Map cluster labels to market phases based on feature characteristics
        
        Args:
            cluster_labels: Array of cluster labels
            features: Array of feature values [growth_rate, volatility, market_share]
            
        Returns:
            Dictionary mapping cluster indices to phase names
        """
        # Calculate average growth rate and volatility for each cluster
        cluster_stats = {}
        
        for cluster in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster
            cluster_features = features[cluster_mask]
            
            avg_growth = np.mean(cluster_features[:, 0])
            avg_volatility = np.mean(cluster_features[:, 1])
            avg_share = np.mean(cluster_features[:, 2])
            
            cluster_stats[cluster] = {
                'avg_growth': avg_growth,
                'avg_volatility': avg_volatility,
                'avg_share': avg_share
            }
        
        # Map clusters to phases
        cluster_phases = {}
        
        for cluster, stats in cluster_stats.items():
            # Determine phase based on growth rate and volatility
            if stats['avg_growth'] > 30:
                if stats['avg_share'] < 5:
                    phase = 'introduction'
                else:
                    phase = 'growth'
            elif stats['avg_growth'] > 15:
                phase = 'growth'
            elif stats['avg_growth'] > -5:
                if stats['avg_volatility'] > 15:
                    phase = 'disruption'
                else:
                    phase = 'maturity'
            else:
                if stats['avg_volatility'] > 20:
                    phase = 'disruption'
                else:
                    phase = 'decline'
            
            cluster_phases[cluster] = phase
        
        return cluster_phases
    
    def _smooth_phase_sequence(self, phase_sequence: List[str]) -> List[str]:
        """
        Smooth the phase sequence to ensure logical transitions
        
        Args:
            phase_sequence: List of detected phases in chronological order
            
        Returns:
            Smoothed sequence of phases
        """
        # Define valid transitions
        valid_transitions = {
            'introduction': ['introduction', 'growth', 'decline'],
            'growth': ['growth', 'maturity', 'disruption'],
            'maturity': ['maturity', 'decline', 'disruption', 'growth'],
            'decline': ['decline', 'disruption', 'introduction'],
            'disruption': ['disruption', 'introduction', 'growth', 'maturity', 'decline']
        }
        
        # Handle special cases
        valid_transitions['insufficient_data'] = list(valid_transitions.keys())
        valid_transitions['unknown'] = list(valid_transitions.keys())
        
        # Make a copy of the sequence
        smoothed = phase_sequence.copy()
        
        # Apply smoothing using a sliding window
        window_size = 3
        half_window = window_size // 2
        
        for i in range(len(smoothed)):
            # Get window around current position
            start = max(0, i - half_window)
            end = min(len(smoothed), i + half_window + 1)
            window = smoothed[start:end]
            
            # Count phases in window
            phase_counts = {}
            for phase in window:
                if phase not in phase_counts:
                    phase_counts[phase] = 0
                phase_counts[phase] += 1
            
            # Find most common phase
            most_common = max(phase_counts.items(), key=lambda x: x[1])[0]
            
            # Check if current phase is different and needs to be changed
            current_phase = smoothed[i]
            if current_phase != most_common:
                # Check if transition is valid
                if most_common in valid_transitions.get(current_phase, []):
                    smoothed[i] = most_common
        
        return smoothed
    
    def _detect_inflection_points(self, data: pd.DataFrame,
                                country_col: str,
                                year_col: str,
                                value_col: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect inflection points in market growth curves
        
        Args:
            data: DataFrame with market data
            country_col: Name of country column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            Dictionary mapping country names to lists of inflection points
        """
        inflection_points = {}
        
        # Process each country/region
        for entity in data[country_col].unique():
            entity_data = data[data[country_col] == entity].sort_values(year_col)
            
            if len(entity_data) < 5:
                # Not enough data for inflection point detection
                continue
            
            # Get years and values
            years = entity_data[year_col].values
            values = entity_data[value_col].values
            
            # Calculate first and second derivatives
            first_deriv = np.gradient(values)
            second_deriv = np.gradient(first_deriv)
            
            # Detect sign changes in second derivative
            sign_changes = np.where(np.diff(np.signbit(second_deriv)))[0]
            
            # Create inflection point records
            entity_inflections = []
            
            for idx in sign_changes:
                year = years[idx]
                value = values[idx]
                
                # Determine if acceleration or deceleration
                if second_deriv[idx] < 0 and second_deriv[idx+1] > 0:
                    inflection_type = 'deceleration'
                else:
                    inflection_type = 'acceleration'
                
                # Add phase transition if applicable
                phase_transition = None
                if 'market_phase' in entity_data.columns:
                    # Check if phase changed at this point
                    if idx < len(years) - 1:
                        phase_before = entity_data[entity_data[year_col] == years[idx]]['market_phase'].iloc[0]
                        phase_after = entity_data[entity_data[year_col] == years[idx+1]]['market_phase'].iloc[0]
                        
                        if phase_before != phase_after:
                            phase_transition = f"{phase_before} → {phase_after}"
                
                entity_inflections.append({
                    'year': year,
                    'value': value,
                    'type': inflection_type,
                    'phase_transition': phase_transition
                })
            
            inflection_points[entity] = entity_inflections
        
        return inflection_points
    
    def _fit_growth_models(self, data: pd.DataFrame,
                         country_col: str,
                         year_col: str,
                         value_col: str) -> Dict[str, Dict[str, Any]]:
        """
        Fit growth models to historical data for each country
        
        Args:
            data: DataFrame with market data
            country_col: Name of country column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            Dictionary mapping country names to fitted model parameters
        """
        country_models = {}
        
        # Process each country/region
        for entity in data[country_col].unique():
            entity_data = data[data[country_col] == entity].sort_values(year_col)
            
            if len(entity_data) < 5:
                # Not enough data for model fitting
                continue
            
            # Get years and values
            years = entity_data[year_col].values
            values = entity_data[value_col].values
            
            # Convert years to sequential indices
            year_indices = np.arange(len(years))
            
            # Try fitting different models
            models_results = {}
            
            for model_name, model_func in self.growth_models.items():
                try:
                    if model_name == 'bass':
                        # Bass diffusion needs initial guesses
                        p0 = [max(values), 0.03, 0.38]
                        params, _ = curve_fit(model_func, year_indices, values, p0=p0, maxfev=10000)
                    else:
                        # Estimate initial parameters
                        if model_name == 'gompertz':
                            p0 = [max(values) * 1.5, 0.5, 0.1]
                        elif model_name == 'logistic':
                            p0 = [max(values) * 1.5, 1.0, 0.5]
                        elif model_name == 'exponential':
                            p0 = [values[0], 0.1]
                        else:  # linear
                            p0 = [values[0], (values[-1] - values[0]) / len(values)]
                        
                        params, _ = curve_fit(model_func, year_indices, values, p0=p0, maxfev=10000)
                    
                    # Calculate predicted values
                    predicted = np.array([model_func(x, *params) for x in year_indices])
                    
                    # Calculate metrics
                    mse = np.mean((values - predicted) ** 2)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((values - predicted) / values)) * 100
                    
                    # Store results
                    models_results[model_name] = {
                        'params': params,
                        'rmse': rmse,
                        'mape': mape
                    }
                except Exception as e:
                    logger.debug(f"Error fitting {model_name} model for {entity}: {str(e)}")
            
            if models_results:
                # Select best model based on RMSE
                best_model = min(models_results.items(), key=lambda x: x[1]['rmse'])
                model_name = best_model[0]
                metrics = best_model[1]
                
                # Store model information
                country_models[entity] = {
                    'model_name': model_name,
                    'params': metrics['params'],
                    'rmse': metrics['rmse'],
                    'mape': metrics['mape']
                }
                
                logger.debug(f"Selected {model_name} model for {entity} with RMSE: {metrics['rmse']:.2f}")
        
        return country_models
    
    def _add_model_projections(self, data: pd.DataFrame,
                             country_col: str,
                             year_col: str,
                             value_col: str) -> pd.DataFrame:
        """
        Add model projections to the data
        
        Args:
            data: DataFrame with market data
            country_col: Name of country column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            DataFrame with added model_value column
        """
        result = data.copy()
        result['model_value'] = np.nan
        result['model_name'] = ''
        
        # Process each country
        for entity, model_info in self.country_models.items():
            entity_data = result[result[country_col] == entity]
            
            if entity_data.empty:
                continue
            
            model_name = model_info['model_name']
            params = model_info['params']
            model_func = self.growth_models[model_name]
            
            # Get minimum year as reference
            min_year = entity_data[year_col].min()
            
            # Calculate model values
            for idx, row in entity_data.iterrows():
                year = row[year_col]
                year_index = year - min_year
                
                try:
                    model_value = model_func(year_index, *params)
                    result.loc[idx, 'model_value'] = model_value
                    result.loc[idx, 'model_name'] = model_name
                except Exception as e:
                    logger.debug(f"Error calculating model value for {entity} in {year}: {str(e)}")
        
        return result
    
    def _predict_market_phase(self, country: str, year: int, growth_rate: float) -> str:
        """
        Predict market phase for a future year
        
        Args:
            country: Country name
            year: Future year
            growth_rate: Projected growth rate
            
        Returns:
            Predicted market phase
        """
        # Check if country has phase data
        if country not in self.market_phases_by_country:
            # Determine based on growth rate only
            if growth_rate > 30:
                return 'growth'
            elif growth_rate > 10:
                return 'maturity'
            elif growth_rate > -5:
                return 'maturity'
            else:
                return 'decline'
        
        # Get current phase
        country_phases = self.market_phases_by_country[country]
        current_years = sorted(country_phases.keys())
        
        if not current_years:
            return 'unknown'
        
        last_year = current_years[-1]
        current_phase = country_phases[last_year]
        
        # Calculate years in current phase
        phase_years = [y for y, p in country_phases.items() if p == current_phase]
        years_in_phase = len(phase_years)
        
        # Get typical duration for this phase
        phase_info = self.market_phases.get(current_phase, {})
        typical_duration = phase_info.get('growth_characteristics', {}).get('typical_duration', (0, 999))
        
        # Check if we should transition to next phase
        if years_in_phase >= typical_duration[1]:
            # Phase duration exceeded, likely to transition
            if current_phase == 'introduction':
                return 'growth'
            elif current_phase == 'growth':
                return 'maturity'
            elif current_phase == 'maturity':
                return 'decline'
            elif current_phase == 'decline':
                return 'decline'  # Stay in decline
            else:
                return current_phase  # Stay in current phase
        elif years_in_phase >= typical_duration[0]:
            # Within typical duration range, could transition based on growth rate
            if current_phase == 'introduction' and growth_rate > 30:
                return 'growth'
            elif current_phase == 'growth' and growth_rate < 15:
                return 'maturity'
            elif current_phase == 'maturity' and growth_rate < 0:
                return 'decline'
            else:
                return current_phase  # Stay in current phase
        else:
            # Still early in phase, likely to stay
            return current_phase
    
    def _bass_diffusion_model(self, t, m, p, q):
        """
        Bass diffusion model for technology adoption
        
        Args:
            t: Time index
            m: Market potential (saturation level)
            p: Coefficient of innovation
            q: Coefficient of imitation
            
        Returns:
            Predicted value
        """
        return m * (1 - np.exp(-(p + q) * t)) / (1 + (q/p) * np.exp(-(p + q) * t))
    
    def _gompertz_model(self, t, a, b, c):
        """
        Gompertz growth model
        
        Args:
            t: Time index
            a: Asymptote (saturation level)
            b: Displacement along x-axis
            c: Growth rate
            
        Returns:
            Predicted value
        """
        return a * np.exp(-b * np.exp(-c * t))
    
    def _logistic_model(self, t, a, b, c):
        """
        Logistic growth model (S-curve)
        
        Args:
            t: Time index
            a: Asymptote (saturation level)
            b: Midpoint location
            c: Growth rate
            
        Returns:
            Predicted value
        """
        return a / (1 + np.exp(-c * (t - b)))
    
    def _exponential_model(self, t, a, b):
        """
        Exponential growth model
        
        Args:
            t: Time index
            a: Initial value
            b: Growth rate
            
        Returns:
            Predicted value
        """
        return a * np.exp(b * t)
    
    def _linear_model(self, t, a, b):
        """
        Linear growth model
        
        Args:
            t: Time index
            a: Intercept
            b: Slope
            
        Returns:
            Predicted value
        """
        return a + b * t
    
    def visualize_country_dynamics(self, market_data: pd.DataFrame, 
                                 country: str,
                                 country_col: str = 'Country',
                                 year_col: str = 'Year',
                                 value_col: str = 'Value') -> plt.Figure:
        """
        Create visualization of market dynamics for a specific country
        
        Args:
            market_data: DataFrame with market data
            country: Country to visualize
            country_col: Name of country column
            year_col: Name of year column
            value_col: Name of value column
            
        Returns:
            Matplotlib figure with visualization
        """
        # Filter data for the specified country
        country_data = market_data[market_data[country_col] == country].sort_values(year_col)
        
        if country_data.empty:
            raise ValueError(f"No data found for country: {country}")
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Market Dynamics Analysis: {country}", fontsize=16)
        
        # Plot 1: Market value and growth model
        ax1 = axes[0, 0]
        
        # Plot actual values
        years = country_data[year_col].values
        values = country_data[value_col].values
        ax1.plot(years, values, 'o-', label='Actual Value')
        
        # Plot model values if available
        if 'model_value' in country_data.columns and not country_data['model_value'].isna().all():
            model_values = country_data['model_value'].values
            model_name = country_data['model_name'].iloc[0] if not country_data['model_name'].isna().all() else 'Unknown'
            ax1.plot(years, model_values, 'r--', label=f'Model: {model_name}')
        
        # Mark inflection points if available
        if country in self.inflection_points:
            for point in self.inflection_points[country]:
                point_year = point['year']
                point_value = point['value']
                point_type = point['type']
                
                if point_type == 'acceleration':
                    marker = '^'
                    color = 'g'
                else:  # deceleration
                    marker = 'v'
                    color = 'r'
                
                ax1.plot(point_year, point_value, marker=marker, markersize=10, 
                        color=color, label=f"{point_type.capitalize()} ({point_year})")
        
        ax1.set_title("Market Value and Growth Model")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Market Value")
        ax1.grid(True, alpha=0.3)
        
        # Remove duplicate labels
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
        # Plot 2: Growth rates and market phases
        ax2 = axes[0, 1]
        
        # Plot growth rates
        if 'growth_rate' in country_data.columns:
            growth_rates = country_data['growth_rate'].values
            ax2.plot(years, growth_rates, 'o-', label='Growth Rate (%)')
            
            # Color background by market phase if available
            if 'market_phase' in country_data.columns:
                phases = country_data['market_phase'].unique()
                
                # Define colors for each phase
                phase_colors = {
                    'introduction': 'lightgreen',
                    'growth': 'lightblue',
                    'maturity': 'lightyellow',
                    'decline': 'mistyrose',
                    'disruption': 'lavender'
                }
                
                # Plot colored background for each phase region
                for phase in phases:
                    phase_data = country_data[country_data['market_phase'] == phase]
                    if not phase_data.empty:
                        phase_years = phase_data[year_col].values
                        min_year = min(phase_years)
                        max_year = max(phase_years)
                        
                        ax2.axvspan(min_year, max_year, 
                                   alpha=0.3, 
                                   color=phase_colors.get(phase, 'lightgray'),
                                   label=f"Phase: {phase.capitalize()}")
        
        ax2.set_title("Growth Rate and Market Phases")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Growth Rate (%)")
        ax2.grid(True, alpha=0.3)
        
        # Remove duplicate labels
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())
        
        # Plot 3: Market share evolution
        ax3 = axes[1, 0]
        
        if 'market_share' in country_data.columns:
            market_shares = country_data['market_share'].values
            ax3.plot(years, market_shares, 'o-', label='Market Share (%)')
            
            # Add trend line
            try:
                slope, intercept, _, _, _ = linregress(range(len(years)), market_shares)
                trend_line = intercept + slope * np.arange(len(years))
                ax3.plot(years, trend_line, 'r--', label='Trend')
            except Exception:
                pass
        
        ax3.set_title("Market Share Evolution")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Market Share (%)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Growth volatility
        ax4 = axes[1, 1]
        
        if 'volatility' in country_data.columns:
            volatility = country_data['volatility'].values
            ax4.plot(years, volatility, 'o-', label='Growth Volatility')
            
            # Add threshold lines
            ax4.axhline(y=10, color='g', linestyle='--', alpha=0.5, label='Low Volatility')
            ax4.axhline(y=20, color='y', linestyle='--', alpha=0.5, label='Medium Volatility')
            ax4.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='High Volatility')
        
        ax4.set_title("Growth Volatility")
        ax4.set_xlabel("Year")
        ax4.set_ylabel("Volatility")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig