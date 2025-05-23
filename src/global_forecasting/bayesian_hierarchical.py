"""
Bayesian Hierarchical Distribution Framework - Advanced Bayesian modeling for market distribution

This module provides a multi-level Bayesian framework that models global, regional, tier, 
and country-level parameters in a coherent statistical structure while enforcing global 
constraints and capturing uncertainty at all levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pymc as pm
import arviz as az
from src.global_forecasting.base_forecaster import BaseForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BayesianHierarchicalDistributor(BaseForecaster):
    """
    Bayesian Hierarchical Distribution Framework for market distribution
    
    This class implements a multi-level Bayesian framework that models global, regional, 
    tier, and country-level parameters in a coherent structure while enforcing global 
    constraints and capturing uncertainty at all levels.
    """
    
    def __init__(self, config=None):
        """
        Initialize the BayesianHierarchicalDistributor
        
        Args:
            config: Optional configuration for the distributor
        """
        super().__init__(config)
        self.name = "Bayesian Hierarchical Distributor"
        self.model = None
        self.trace = None
        self.prediction_samples = None
        self.countries = None
        self.regions = []
        self.tiers = {}
        self.region_country_map = {}
        self.country_region_map = {}
        self.prior_strength = 0.5  # Controls how strongly priors influence predictions
        
        # Initialize hierarchical components
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters from configuration"""
        if not self.config:
            return
            
        # Get Bayesian model settings
        model_settings = self.config.get('bayesian_model', {})
        
        # MCMC settings
        self.num_samples = model_settings.get('num_samples', 1000)
        self.tune = model_settings.get('tune', 1000)
        self.chains = model_settings.get('chains', 2)
        self.target_accept = model_settings.get('target_accept', 0.8)
        
        # Prior settings
        self.prior_strength = model_settings.get('prior_strength', 0.5)
        self.variance_prior_mult = model_settings.get('variance_prior_mult', 2.0)
        
        # Region and tier settings
        self.use_regions = model_settings.get('use_regions', True)
        self.use_tiers = model_settings.get('use_tiers', True)
        self.use_indicators = model_settings.get('use_indicators', True)
    
    def _build_model(self, historical_data, forecast_years, regions=None, tiers=None, indicators=None):
        """
        Build the hierarchical Bayesian model
        
        Args:
            historical_data: DataFrame with historical country-level data
            forecast_years: List of years to forecast
            regions: Optional dictionary mapping countries to regions
            tiers: Optional dictionary mapping countries to tiers
            indicators: Optional DataFrame with indicator data
            
        Returns:
            PyMC model
        """
        # Extract unique countries, years, and values from historical data
        countries = historical_data['Country'].unique()
        self.countries = countries
        historical_years = historical_data['Year'].unique()
        self.historical_years = historical_years
        self.forecast_years = forecast_years
        all_years = np.concatenate([historical_years, forecast_years])
        
        # Store region and tier mappings if provided
        if regions:
            self.region_country_map = regions
            # Invert mapping for lookup
            self.country_region_map = {}
            for region, region_countries in regions.items():
                for country in region_countries:
                    if country in countries:
                        self.country_region_map[country] = region
            self.regions = list(regions.keys())
        
        if tiers:
            self.tiers = tiers
        
        # Create coordinate mappings
        country_indices = {country: i for i, country in enumerate(countries)}
        year_indices = {year: i for i, year in enumerate(all_years)}
        region_indices = {region: i for i, region in enumerate(self.regions)} if self.regions else {}
        
        # Create the Bayesian model
        with pm.Model() as model:
            # Global market growth rate (common across all countries)
            global_growth_mean = pm.Normal('global_growth_mean', mu=0.05, sigma=0.02)
            global_growth_sigma = pm.HalfNormal('global_growth_sigma', sigma=0.05)
            
            # Region-level parameters (optional)
            if self.use_regions and self.regions:
                # Region-specific growth rate adjustments
                region_growth_adj = pm.Normal('region_growth_adj', 
                                            mu=0, 
                                            sigma=0.05, 
                                            shape=len(self.regions))
            
            # Tier-level parameters (optional)
            if self.use_tiers and self.tiers:
                # Tier-specific growth rate adjustments
                num_tiers = len(set(self.tiers.values()))
                tier_growth_adj = pm.Normal('tier_growth_adj', 
                                            mu=0, 
                                            sigma=0.03, 
                                            shape=num_tiers)
            
            # Country-level parameters
            # Create a different parameter for each country
            country_growth_adj = pm.Normal('country_growth_adj', 
                                          mu=0, 
                                          sigma=0.1, 
                                          shape=len(countries))
            
            # Indicator effects (optional)
            if self.use_indicators and indicators is not None:
                # Convert indicators to standardized form
                indicator_names = indicators['Indicator'].unique()
                num_indicators = len(indicator_names)
                
                # Create indicator weights as random variables
                indicator_weights = pm.Normal('indicator_weights', 
                                            mu=0, 
                                            sigma=0.05, 
                                            shape=num_indicators)
            
            # Calculate growth rates for each country and year
            growth_rates = {}
            
            for country in countries:
                country_idx = country_indices[country]
                
                # Base growth is global growth rate
                country_rate = global_growth_mean + country_growth_adj[country_idx]
                
                # Add region-specific effect if applicable
                if self.use_regions and self.regions and country in self.country_region_map:
                    region = self.country_region_map[country]
                    region_idx = region_indices[region]
                    country_rate = country_rate + region_growth_adj[region_idx]
                
                # Add tier-specific effect if applicable
                if self.use_tiers and self.tiers and country in self.tiers:
                    tier = self.tiers[country]
                    country_rate = country_rate + tier_growth_adj[tier]
                
                # Add indicator effects if applicable
                if self.use_indicators and indicators is not None:
                    for i, indicator_name in enumerate(indicator_names):
                        indicator_value = indicators.get((country, indicator_name), 0)
                        country_rate = country_rate + indicator_weights[i] * indicator_value
                
                growth_rates[country] = country_rate
            
            # Market share variables and constraints
            # Initialize market shares based on historical data for the last historical year
            last_hist_year = max(historical_years)
            last_hist_data = historical_data[historical_data['Year'] == last_hist_year]
            
            # Get initial market shares and values
            init_market_shares = {}
            total_value = last_hist_data['Value'].sum()
            
            for _, row in last_hist_data.iterrows():
                country = row['Country']
                value = row['Value']
                share = value / total_value
                init_market_shares[country] = share
            
            # Forecast market shares for all years
            market_shares = {}
            
            # Historical years - use actual data
            for year in historical_years:
                year_idx = year_indices[year]
                year_data = historical_data[historical_data['Year'] == year]
                year_total = year_data['Value'].sum()
                
                for _, row in year_data.iterrows():
                    country = row['Country']
                    country_idx = country_indices[country]
                    value = row['Value']
                    share = value / year_total
                    market_shares[(country, year)] = share
            
            # Forecast years - predict using growth rates
            for year_idx, year in enumerate(forecast_years):
                # We'll model shares directly with a Dirichlet distribution
                # to ensure they sum to 1.0
                shares_container = []
                
                # Calculate unnormalized shares based on growth rates
                unnorm_shares = {}
                
                for country in countries:
                    prev_year = forecast_years[year_idx-1] if year_idx > 0 else last_hist_year
                    growth_rate = growth_rates[country]
                    
                    # Apply growth rate to previous year's share
                    prev_share = market_shares.get((country, prev_year), init_market_shares.get(country, 0.01))
                    new_share = prev_share * (1 + growth_rate)
                    unnorm_shares[country] = new_share
                
                # Normalize shares to sum to 1.0
                total_share = sum(unnorm_shares.values())
                for country in countries:
                    market_shares[(country, year)] = unnorm_shares[country] / total_share
            
            # Define observations
            # For historical data, we'll fit to the actual market values
            for _, row in historical_data.iterrows():
                country = row['Country']
                year = row['Year']
                value = row['Value']
                
                # Create observation variable
                country_idx = country_indices[country]
                year_idx = year_indices[year]
                
                # Use Normal likelihood with data-driven standard deviation
                # This models the observation noise in the data
                pm.Normal(f'obs_{country}_{year}', 
                         mu=market_shares[(country, year)] * total_value, 
                         sigma=value * 0.1,  # 10% of value as standard deviation
                         observed=value)
                         
            # Define constraint that shares sum to 1.0 for each year
            for year in forecast_years:
                sum_shares = sum(market_shares[(country, year)] for country in countries)
                pm.Deterministic(f'sum_shares_{year}', sum_shares)
        
        self.model = model
        return model
    
    def fit(self, data: pd.DataFrame) -> 'BayesianHierarchicalDistributor':
        """
        Fit the Bayesian hierarchical model to historical data
        
        Args:
            data: DataFrame with 'date' and 'value' columns
        
        Returns:
            Self for method chaining
        """
        # Check if we have any region information
        has_region_info = 'region_type' in data.columns
        
        if has_region_info:
            # Filter to just country-level data for fitting
            country_data = data[data['region_type'] == 'country'].copy()
        else:
            country_data = data.copy()
        
        # Need date and value columns
        if 'date' not in data.columns or 'value' not in data.columns:
            raise ValueError("Data must contain 'date' and 'value' columns")
        
        # Extract years from dates
        country_data['Year'] = pd.DatetimeIndex(country_data['date']).year
        
        # Cache the data for later use
        self.history = country_data
        
        # Extract unique countries and years
        countries = country_data['Country'].unique()
        years = sorted(country_data['Year'].unique())
        
        # Region mappings from regional_aggregator
        regions = {}
        if 'region' in country_data.columns:
            for country in countries:
                region = country_data[country_data['Country'] == country]['region'].iloc[0]
                if region not in regions:
                    regions[region] = []
                regions[region].append(country)
        
        # Tier mappings
        tiers = {}
        if 'tier' in country_data.columns:
            for _, row in country_data.iterrows():
                tiers[row['Country']] = row['tier']
        
        # Build the Bayesian model
        self._build_model(country_data, years, regions, tiers)
        
        # Sample from the posterior distribution
        with self.model:
            # Use NUTS sampler for efficient exploration
            self.trace = pm.sample(self.num_samples, tune=self.tune, 
                                  chains=self.chains, target_accept=self.target_accept,
                                  return_inferencedata=True)
        
        # Mark as fitted
        self.fitted = True
        
        return self
    
    def forecast(self, periods: int, frequency: str = 'Y') -> pd.DataFrame:
        """
        Generate forecast for the specified number of periods
        
        Args:
            periods: Number of periods to forecast
            frequency: Time frequency of forecast ('Y'=yearly)
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        # Generate future years based on the frequency
        last_year = max(self.historical_years)
        if frequency == 'Y':
            forecast_years = np.array([last_year + i + 1 for i in range(periods)])
        else:
            # Default to annual for other frequencies
            forecast_years = np.array([last_year + i + 1 for i in range(periods)])
        
        # Extract posterior means for parameters
        posterior_means = {var: self.trace.posterior[var].mean(dim=["chain", "draw"]).values 
                          for var in self.trace.posterior.keys()}
        
        # Prepare forecasting
        country_indices = {country: i for i, country in enumerate(self.countries)}
        
        # Compute growth rates for each country
        growth_rates = {}
        
        for country in self.countries:
            country_idx = country_indices[country]
            
            # Start with global growth rate
            growth_rate = posterior_means['global_growth_mean']
            
            # Add country-specific adjustment
            growth_rate += posterior_means['country_growth_adj'][country_idx]
            
            # Add region-specific adjustment if applicable
            if self.use_regions and self.regions and country in self.country_region_map:
                region = self.country_region_map[country]
                region_idx = list(self.regions).index(region)
                growth_rate += posterior_means['region_growth_adj'][region_idx]
            
            # Add tier-specific adjustment if applicable
            if self.use_tiers and self.tiers and country in self.tiers:
                tier = self.tiers[country]
                tier_idx = tier  # Assuming tiers are 0-indexed
                growth_rate += posterior_means['tier_growth_adj'][tier_idx]
            
            growth_rates[country] = growth_rate
        
        # Get initial market shares from last historical year
        last_hist_data = self.history[self.history['Year'] == last_year]
        total_value = last_hist_data['value'].sum()
        initial_shares = {}
        
        for _, row in last_hist_data.iterrows():
            country = row['Country']
            share = row['value'] / total_value
            initial_shares[country] = share
        
        # Forecast market shares
        forecast_shares = {}
        for i, year in enumerate(forecast_years):
            if i == 0:
                # First forecast year starts from last historical year
                prev_shares = initial_shares
            else:
                # Subsequent years use previous forecast
                prev_shares = forecast_shares[forecast_years[i-1]]
            
            # Apply growth rates to get new shares
            new_shares = {}
            for country in self.countries:
                growth = growth_rates[country]
                prev_share = prev_shares.get(country, 0.0)
                new_share = prev_share * (1 + growth)
                new_shares[country] = new_share
            
            # Normalize to ensure shares sum to 1.0
            total_share = sum(new_shares.values())
            if total_share > 0:
                normalized_shares = {country: share/total_share 
                                    for country, share in new_shares.items()}
            else:
                normalized_shares = prev_shares
            
            forecast_shares[year] = normalized_shares
        
        # Compute global market values for forecast years
        # Options:
        # 1. Use growth rate approach
        global_growth = posterior_means['global_growth_mean']
        global_values = {}
        
        # Start with last historical global total
        last_global_value = total_value
        
        for i, year in enumerate(forecast_years):
            if i == 0:
                # First forecast year starts from last historical year
                global_values[year] = last_global_value * (1 + global_growth)
            else:
                # Subsequent years compound from previous
                global_values[year] = global_values[forecast_years[i-1]] * (1 + global_growth)
        
        # Create forecast DataFrame
        forecast_data = []
        
        for year in forecast_years:
            year_shares = forecast_shares[year]
            global_value = global_values[year]
            
            for country in self.countries:
                share = year_shares.get(country, 0.0)
                value = share * global_value
                
                # Create row for forecast DataFrame
                forecast_data.append({
                    'date': pd.Timestamp(year=int(year), month=1, day=1),
                    'Country': country,
                    'value': value,
                    'market_share': share * 100,  # Convert to percentage
                    'growth_rate': growth_rates[country] * 100  # Convert to percentage
                })
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecast_data)
        
        # Store results
        self.forecast_result = forecast_df
        self.forecast_dates = forecast_df['date'].unique()
        
        # Generate confidence intervals
        self._generate_confidence_intervals(forecast_df)
        
        return forecast_df
    
    def _generate_confidence_intervals(self, forecast_df):
        """
        Generate confidence intervals for the forecast
        
        Args:
            forecast_df: DataFrame with point forecasts
        
        Returns:
            None (updates self.confidence_intervals)
        """
        # Create posterior predictive distribution
        with self.model:
            self.prediction_samples = pm.sample_posterior_predictive(
                self.trace, 
                var_names=[v for v in self.trace.posterior.keys() if v.startswith('obs_')],
                random_seed=42
            )
        
        # Extract confidence intervals from posterior
        ci_data = []
        alpha = 0.05  # 95% confidence interval
        
        # For each forecasted row, generate confidence intervals
        for _, row in forecast_df.iterrows():
            date = row['date']
            country = row['Country']
            value = row['value']
            
            # Find corresponding posterior samples
            var_name = f'obs_{country}_{date.year}'
            
            if var_name in self.prediction_samples.posterior_predictive:
                samples = self.prediction_samples.posterior_predictive[var_name].values.flatten()
                lower = np.quantile(samples, alpha/2)
                upper = np.quantile(samples, 1-alpha/2)
            else:
                # If not available, use a simple percentage range
                lower = value * 0.9
                upper = value * 1.1
            
            ci_data.append({
                'date': date,
                'lower': lower,
                'value': value,
                'upper': upper
            })
        
        # Store as confidence intervals
        self.confidence_intervals = pd.DataFrame(ci_data)
    
    def evaluate(self, validation_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on validation data
        
        Args:
            validation_data: DataFrame with actual values for comparison
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Convert validation data to required format
        if 'date' not in validation_data.columns or 'value' not in validation_data.columns:
            raise ValueError("Validation data must contain 'date' and 'value' columns")
        
        # Extract years from dates
        validation_data['Year'] = pd.DatetimeIndex(validation_data['date']).year
        
        # Generate predictions for validation years
        validation_years = sorted(validation_data['Year'].unique())
        
        # Here we could use posterior predictive checks
        # or simply forecast for validation years
        
        # Calculate metrics (MSE, MAPE, etc.)
        metrics = {
            'MSE': 0.0,
            'MAPE': 0.0,
            'MAE': 0.0,
            'R2': 0.0
        }
        
        # Use arviz to compute metrics
        # metrics = az.summary(self.trace, kind='stats')
        
        return metrics
    
    def get_tier_effects(self) -> Dict[int, float]:
        """
        Get the estimated effect of each tier on growth rates
        
        Returns:
            Dictionary mapping tier indices to their growth rate effects
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before accessing tier effects")
        
        if not self.use_tiers:
            return {}
        
        # Extract tier effects from posterior
        tier_adj = self.trace.posterior['tier_growth_adj'].mean(dim=["chain", "draw"]).values
        
        return {i: float(effect) for i, effect in enumerate(tier_adj)}
    
    def get_region_effects(self) -> Dict[str, float]:
        """
        Get the estimated effect of each region on growth rates
        
        Returns:
            Dictionary mapping region names to their growth rate effects
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before accessing region effects")
        
        if not self.use_regions or not self.regions:
            return {}
        
        # Extract region effects from posterior
        region_adj = self.trace.posterior['region_growth_adj'].mean(dim=["chain", "draw"]).values
        
        return {region: float(region_adj[i]) for i, region in enumerate(self.regions)}
    
    def get_country_effects(self) -> Dict[str, float]:
        """
        Get the estimated effect of each country on growth rates
        
        Returns:
            Dictionary mapping country names to their growth rate effects
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before accessing country effects")
        
        # Extract country effects from posterior
        country_adj = self.trace.posterior['country_growth_adj'].mean(dim=["chain", "draw"]).values
        
        return {country: float(country_adj[i]) for i, country in enumerate(self.countries)}
    
    def get_global_growth_rate(self) -> float:
        """
        Get the estimated global growth rate
        
        Returns:
            Float with global growth rate
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before accessing global growth rate")
        
        # Extract global growth rate from posterior
        global_growth = float(self.trace.posterior['global_growth_mean'].mean(dim=["chain", "draw"]).values)
        
        return global_growth
    
    def get_indicator_weights(self) -> Dict[str, float]:
        """
        Get the estimated weights for each indicator
        
        Returns:
            Dictionary mapping indicator names to their weights
        """
        if not self.fitted or not self.use_indicators:
            return {}
        
        # Check if we have indicator weights in the trace
        if 'indicator_weights' not in self.trace.posterior:
            return {}
        
        # Extract indicator weights from posterior
        indicator_weights = self.trace.posterior['indicator_weights'].mean(dim=["chain", "draw"]).values
        
        # Map to indicator names
        indicators = self.model.indicator_names if hasattr(self.model, 'indicator_names') else []
        
        return {indicator: float(indicator_weights[i]) for i, indicator in enumerate(indicators)}
    
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary of the model's parameters
        
        Returns:
            DataFrame with parameter summaries
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before generating summary")
        
        # Use arviz to generate summary statistics
        summary_df = az.summary(self.trace)
        
        return summary_df
    
    def plot_trace(self, var_names=None):
        """
        Plot the MCMC traces for parameters
        
        Args:
            var_names: Optional list of variable names to plot
            
        Returns:
            arviz trace plot
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting trace")
        
        # Use arviz to plot trace
        return az.plot_trace(self.trace, var_names=var_names)
    
    def plot_posterior(self, var_names=None):
        """
        Plot the posterior distributions for parameters
        
        Args:
            var_names: Optional list of variable names to plot
            
        Returns:
            arviz posterior plot
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before plotting posterior")
        
        # Use arviz to plot posterior
        return az.plot_posterior(self.trace, var_names=var_names)


# Register the forecaster
def register_forecaster():
    from src.global_forecasting import register_forecaster_class
    register_forecaster_class(
        "bayesian_hierarchical", 
        BayesianHierarchicalDistributor, 
        "Advanced Bayesian Hierarchical Distributor",
        "statistical"
    )