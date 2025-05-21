"""
Market Distributor Module - Universal market share distribution logic

This module provides a market-agnostic approach to distributing global market values
across countries using dynamic, data-driven share allocation algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.cluster import KMeans

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketDistributor:
    """
    Universal market distributor for any market type
    
    This class provides functionality to distribute global market values across
    countries using data-driven algorithms that adapt to any market structure.
    """
    
    def __init__(self, config_manager, data_loader, indicator_analyzer):
        """
        Initialize the MarketDistributor
        
        Args:
            config_manager: Configuration manager instance for accessing settings
            data_loader: Data loader instance for accessing market data
            indicator_analyzer: Indicator analyzer instance for applying adjustments
        """
        self.config_manager = config_manager
        self.data_loader = data_loader
        self.indicator_analyzer = indicator_analyzer
        
        # Get market distribution settings
        self.distribution_settings = self.config_manager.get_market_distribution_settings()
        
        # Initialize dynamic tier classification
        self.tiers = None
        self.tier_thresholds = None
    
    def distribute_market(self) -> pd.DataFrame:
        """
        Distribute the global market forecast across countries
        
        Returns:
            DataFrame containing distributed market values for all countries and years
        """
        # Load required data
        global_forecast = self.data_loader.load_global_forecast()
        country_historical = self.data_loader.load_country_historical()
        
        # Get column names from mapping
        global_mapping = self.config_manager.get_column_mapping('global_forecast')
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        
        global_year_col = global_mapping.get('year_column', 'Year')
        global_value_col = global_mapping.get('value_column', 'Value')
        global_type_col = global_mapping.get('type_column', 'Type')
        
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
        
        # Get redistribution start year from configuration (if specified)
        redistribution_start_year = self.distribution_settings.get('redistribution_start_year', None)
        
        # CRITICAL CHANGE: If redistribution_start_year is set, we'll process data in two completely separate paths
        if redistribution_start_year is not None:
            logger.info(f"Using redistribution_start_year={redistribution_start_year}. Processing data in two separate paths.")
            
            # Path 1: Preserve historical data exactly as is (before redistribution_start_year)
            preserved_historical = country_historical[country_historical['Year'] < redistribution_start_year].copy()
            preserved_years = sorted(preserved_historical['Year'].unique())
            
            if not preserved_historical.empty:
                logger.info(f"Preserving historical data for years: {preserved_years}")
            
            # Path 2: Process data from redistribution_start_year onward
            # First, filter data to only include years from redistribution_start_year onward
            filtered_country_historical = country_historical[country_historical['Year'] >= redistribution_start_year].copy()
            filtered_years = sorted(filtered_country_historical['Year'].unique())
            
            if filtered_country_historical.empty:
                logger.warning("No historical data found for years >= redistribution_start_year.")
                # If no historical data available for processing, use the most recent year before redistribution_start_year
                most_recent_year = max(preserved_historical['Year']) if not preserved_historical.empty else None
                if most_recent_year:
                    logger.info(f"Using data from year {most_recent_year} as reference for forecasting.")
                    filtered_country_historical = country_historical[country_historical['Year'] == most_recent_year].copy()
                    filtered_country_historical['Year'] = redistribution_start_year  # Adjust year to redistribution_start_year
                    filtered_years = [redistribution_start_year]
            
            logger.info(f"Processing data for years: {filtered_years}")
            
            # Now process only the filtered data through the normal pipeline
            historical_shares = self._calculate_historical_shares(
                filtered_country_historical, filtered_years, id_col, country_col)
            
            # Get the latest year for tier determination
            latest_historical_year = max(filtered_years) if filtered_years else redistribution_start_year
            
            # Detect market tiers
            if self.distribution_settings.get('tier_determination', 'auto') == 'auto':
                self._determine_tiers(historical_shares, latest_historical_year)
            else:
                self._load_manual_tiers()
            
            # Get forecast years from global forecast (only years after redistribution_start_year)
            forecast_mask = (global_forecast[global_type_col] == 'Forecast') & (global_forecast[global_year_col] >= redistribution_start_year)
            forecast_years = sorted(global_forecast.loc[forecast_mask, global_year_col].unique())
            
            # Project market shares for forecast years
            projected_shares = self._project_market_shares(historical_shares, forecast_years, id_col, country_col)
            
            # Apply indicator adjustments
            if self.indicator_analyzer:
                projected_shares = self.indicator_analyzer.apply_indicator_adjustments(projected_shares)
            
            # Apply growth constraints
            growth_constrained_shares = self._apply_growth_constraints(projected_shares, historical_shares, id_col)
            
            # Combine historical (filtered) and projected shares
            combined_shares = pd.concat([historical_shares, growth_constrained_shares], ignore_index=True)
            
            # Calculate absolute market values using global forecast
            distributed_market = self._calculate_distributed_values(combined_shares, global_forecast, global_year_col, global_value_col)
            
            # Apply smoothing
            smoothed_market = self._apply_smoothing(distributed_market)
            
            # Now merge the preserved historical data with the processed data
            # Ensure the preserved data has all necessary columns
            if not preserved_historical.empty:
                final_result = pd.concat([preserved_historical, smoothed_market], ignore_index=True)
                final_result = final_result.sort_values(by=[id_col, 'Year'])
            else:
                final_result = smoothed_market
            
            return final_result
            
        else:
            # Original path - process all years
            logger.info("No redistribution_start_year specified. Processing all years.")
            
            # Get years for history and forecast
            historical_years = sorted(country_historical['Year'].unique())
            latest_historical_year = max(historical_years)
            
            # Get all years from global forecast
            all_years = sorted(global_forecast[global_year_col].unique())
            
            # Separate historical and forecast years
            forecast_mask = global_forecast[global_type_col] == 'Forecast'
            forecast_years = sorted(global_forecast.loc[forecast_mask, global_year_col].unique())
            
            # Calculate historical market shares
            historical_shares = self._calculate_historical_shares(
                country_historical, historical_years, id_col, country_col)
            
            # Detect market tiers automatically if configured to do so
            if self.distribution_settings.get('tier_determination', 'auto') == 'auto':
                self._determine_tiers(historical_shares, latest_historical_year)
            else:
                # Use manual tier settings from configuration
                self._load_manual_tiers()
            
            # Project market shares for forecast years
            projected_shares = self._project_market_shares(historical_shares, forecast_years, id_col, country_col)
            
            # Apply indicator adjustments to projected shares if indicators are available
            if self.indicator_analyzer:
                projected_shares = self.indicator_analyzer.apply_indicator_adjustments(projected_shares)
            
            # Apply growth constraints based on market dynamics
            growth_constrained_shares = self._apply_growth_constraints(projected_shares, historical_shares, id_col)
            
            # Combine historical and projected shares
            combined_shares = pd.concat([historical_shares, growth_constrained_shares], ignore_index=True)
            
            # Calculate absolute market values using global forecast
            distributed_market = self._calculate_distributed_values(combined_shares, global_forecast, global_year_col, global_value_col)
            
            # Apply smoothing to ensure realistic growth patterns
            smoothed_market = self._apply_smoothing(distributed_market)
            
            return smoothed_market
    
    def _calculate_historical_shares(self, country_data: pd.DataFrame, years: List[int], 
                                    id_col: str, country_col: str) -> pd.DataFrame:
        """
        Calculate historical market shares for all countries and years
        
        Args:
            country_data: DataFrame with country historical data
            years: List of historical years
            id_col: Name of country ID column
            country_col: Name of country name column
            
        Returns:
            DataFrame with historical market shares
        """
        # Create a copy to avoid modifying the original
        df = country_data.copy()
        
        # Initialize result dataframe for shares
        shares_list = []
        
        # Process each year
        for year in years:
            year_data = df[df['Year'] == year].copy()
            
            # Calculate total market size for the year
            total_market = year_data['Value'].sum()
            
            if total_market <= 0:
                logger.warning(f"Total market size for year {year} is zero or negative")
                continue
            
            # Calculate market share for each country
            year_data['market_share'] = year_data['Value'] / total_market * 100
            
            # Keep relevant columns
            shares_list.append(year_data[[id_col, country_col, 'Year', 'Value', 'market_share']])
        
        # Combine all years
        historical_shares = pd.concat(shares_list, ignore_index=True)
        
        return historical_shares
    
    def _determine_tiers(self, historical_shares: pd.DataFrame, latest_year: int) -> None:
        """
        Automatically determine market tiers based on share distribution
        
        Args:
            historical_shares: DataFrame with historical market shares
            latest_year: Most recent historical year
        """
        # Get the latest year data
        latest_data = historical_shares[historical_shares['Year'] == latest_year].copy()
        
        # Sort by market share descending
        latest_data = latest_data.sort_values(by='market_share', ascending=False)
        
        # Get shares as a numpy array for clustering
        shares = latest_data['market_share'].values.reshape(-1, 1)
        
        # Determine optimal number of clusters using silhouette score
        from sklearn.metrics import silhouette_score
        
        max_clusters = min(8, len(shares) - 1)  # At most 8 tiers, and must be less than number of countries
        max_clusters = max(3, max_clusters)  # At least 3 tiers
        
        best_score = -1
        best_n_clusters = 3  # Default to 3 clusters
        
        # Try different numbers of clusters
        for n_clusters in range(3, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(shares)
                silhouette_avg = silhouette_score(shares, cluster_labels)
                
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_n_clusters = n_clusters
            except Exception as e:
                logger.warning(f"Error in cluster analysis with {n_clusters} clusters: {str(e)}")
        
        # Apply the best clustering
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        latest_data['tier'] = kmeans.fit_predict(shares)
        
        # Remap cluster numbers to ensure tier 0 has highest shares
        tier_map = {}
        for tier in range(best_n_clusters):
            tier_mean = latest_data[latest_data['tier'] == tier]['market_share'].mean()
            tier_map[tier] = tier_mean
        
        # Sort tiers by mean share descending
        sorted_tiers = sorted(tier_map.items(), key=lambda x: x[1], reverse=True)
        tier_remap = {old_tier: new_tier for new_tier, (old_tier, _) in enumerate(sorted_tiers)}
        
        # Apply remapping
        latest_data['tier'] = latest_data['tier'].map(tier_remap)
        
        # Calculate tier thresholds
        self.tier_thresholds = []
        
        for tier in range(best_n_clusters):
            tier_data = latest_data[latest_data['tier'] == tier]
            
            # Find minimum share in this tier
            min_share = tier_data['market_share'].min()
            
            # For highest tier (0), use the minimum share as threshold
            # For other tiers, use the average of this tier's min and next tier's max
            if tier < best_n_clusters - 1:
                next_tier_data = latest_data[latest_data['tier'] == tier + 1]
                next_tier_max = next_tier_data['market_share'].max()
                threshold = (min_share + next_tier_max) / 2
            else:
                threshold = min_share / 2  # For lowest tier, use half of its minimum
            
            self.tier_thresholds.append(threshold)
        
        # Store tiers for later use
        self.tiers = best_n_clusters
        
        logger.info(f"Automatically determined {best_n_clusters} market tiers with thresholds: {self.tier_thresholds}")
    
    def _load_manual_tiers(self) -> None:
        """
        Load manually configured tier settings from configuration
        """
        manual_tiers = self.distribution_settings.get('manual_tiers', {})
        
        if not manual_tiers:
            logger.warning("No manual tier settings found in configuration, using defaults")
            # Default to 3 tiers with reasonable thresholds
            self.tiers = 3
            self.tier_thresholds = [5.0, 1.0, 0.1]
            return
        
        # Extract tier thresholds
        thresholds = []
        for tier_key in sorted(manual_tiers.keys()):
            tier = manual_tiers[tier_key]
            threshold = tier.get('share_threshold', 0)
            thresholds.append(threshold)
        
        # Ensure thresholds are in descending order
        thresholds = sorted(thresholds, reverse=True)
        
        self.tiers = len(thresholds)
        self.tier_thresholds = thresholds
        
        logger.info(f"Loaded {self.tiers} manual market tiers with thresholds: {self.tier_thresholds}")
    
    def _assign_country_tier(self, share: float) -> int:
        """
        Assign a tier to a country based on its market share
        
        Args:
            share: Market share percentage
            
        Returns:
            Tier index (0 for highest tier)
        """
        if self.tier_thresholds is None:
            # Default tier assignment if not determined yet
            if share >= 5.0:
                return 0
            elif share >= 1.0:
                return 1
            else:
                return 2
        
        # Assign based on calculated thresholds
        for tier, threshold in enumerate(self.tier_thresholds):
            if share >= threshold:
                return tier
        
        # If below all thresholds, assign to lowest tier
        return len(self.tier_thresholds)
    
    def _project_market_shares(self, historical_shares: pd.DataFrame, forecast_years: List[int],
                              id_col: str, country_col: str) -> pd.DataFrame:
        """
        Project market shares for forecast years
        
        Args:
            historical_shares: DataFrame with historical market shares
            forecast_years: List of forecast years
            id_col: Name of country ID column
            country_col: Name of country name column
            
        Returns:
            DataFrame with projected market shares for forecast years
        """
        # Get the latest historical year
        latest_year = historical_shares['Year'].max()
        latest_shares = historical_shares[historical_shares['Year'] == latest_year].copy()
        
        # Calculate the trend for each country from historical data
        trend_df = self._calculate_share_trends(historical_shares, id_col)
        
        # Merge trends with latest shares
        projection_base = pd.merge(
            latest_shares,
            trend_df,
            on=id_col,
            how='left'
        )
        
        # Fill missing trends with neutral value (1.0 = no change)
        projection_base['trend_factor'] = projection_base['trend_factor'].fillna(1.0)
        
        # Assign tiers to countries
        projection_base['tier'] = projection_base['market_share'].apply(self._assign_country_tier)
        
        # Project shares for forecast years
        projected_shares_list = []
        
        for year in forecast_years:
            # Calculate years since the latest historical year
            years_ahead = year - latest_year
            
            # Make a copy for this forecast year
            year_projection = projection_base.copy()
            year_projection['Year'] = year
            
            # Apply trends with diminishing effect for later years
            # Use a sigmoid-like diminishing factor
            diminishing_factor = 2.0 / (1.0 + np.exp(0.3 * years_ahead)) - 0.5
            
            # Apply trends with tier-specific constraints
            for tier in range(self.tiers if self.tiers else 3):
                tier_mask = year_projection['tier'] == tier
                
                # Get tier-specific constraints
                max_change = self._get_tier_max_share_change(tier, years_ahead)
                
                # Apply trend with tier-specific constraints
                tier_df = year_projection.loc[tier_mask].copy()
                
                # Calculate raw trend projection
                raw_trend = tier_df['trend_factor'] ** (diminishing_factor * years_ahead)
                
                # Constrain the change
                constrained_trend = np.clip(raw_trend, 1.0 - max_change, 1.0 + max_change)
                
                # Apply constrained trend
                year_projection.loc[tier_mask, 'market_share'] = year_projection.loc[tier_mask, 'market_share'] * constrained_trend
            
            # Normalize shares to sum to 100%
            total_share = year_projection['market_share'].sum()
            if total_share > 0:
                year_projection['market_share'] = year_projection['market_share'] / total_share * 100
            
            # Add to results
            projected_shares_list.append(year_projection)
        
        # Combine all forecast years
        projected_shares = pd.concat(projected_shares_list, ignore_index=True)
        
        # Keep relevant columns
        relevant_cols = [id_col, country_col, 'Year', 'market_share', 'tier']
        projected_shares = projected_shares[relevant_cols]
        
        return projected_shares
    
    def _calculate_share_trends(self, historical_shares: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """
        Calculate market share trends for each country from historical data
        
        Args:
            historical_shares: DataFrame with historical market shares
            id_col: Name of country ID column
            
        Returns:
            DataFrame with country IDs and trend factors
        """
        # Get all years sorted
        years = sorted(historical_shares['Year'].unique())
        
        # Need at least 2 years to calculate trend
        if len(years) < 2:
            logger.warning("Not enough historical years to calculate trends")
            return pd.DataFrame({id_col: [], 'trend_factor': []})
        
        # Calculate moving average share change for each country
        trend_factors = []
        countries = historical_shares[id_col].unique()
        
        for country_id in countries:
            country_data = historical_shares[historical_shares[id_col] == country_id].copy()
            
            # Sort by year
            country_data = country_data.sort_values(by='Year')
            
            if len(country_data) < 2:
                # Not enough data for this country
                trend_factors.append({
                    id_col: country_id,
                    'trend_factor': 1.0  # Neutral trend
                })
                continue
            
            # Calculate year-over-year share changes
            country_data['prev_share'] = country_data['market_share'].shift(1)
            country_data['share_change'] = country_data['market_share'] / country_data['prev_share']
            
            # Remove first row (no previous share)
            country_data = country_data.dropna(subset=['share_change'])
            
            if country_data.empty:
                trend_factors.append({
                    id_col: country_id,
                    'trend_factor': 1.0  # Neutral trend
                })
                continue
            
            # Calculate weighted average of share changes (more recent years have higher weight)
            weights = np.linspace(0.5, 1.0, len(country_data))
            weighted_avg_change = np.average(country_data['share_change'], weights=weights)
            
            # Apply dampening to extreme trends
            if weighted_avg_change > 1.0:
                dampened_change = 1.0 + (weighted_avg_change - 1.0) * 0.7
            elif weighted_avg_change < 1.0:
                dampened_change = 1.0 - (1.0 - weighted_avg_change) * 0.7
            else:
                dampened_change = 1.0
            
            trend_factors.append({
                id_col: country_id,
                'trend_factor': dampened_change
            })
        
        return pd.DataFrame(trend_factors)
    
    def _get_tier_max_share_change(self, tier: int, years_ahead: int) -> float:
        """
        Get the maximum allowed share change for a tier
        
        Args:
            tier: Tier index (0 for highest tier)
            years_ahead: Number of years ahead from latest historical year
            
        Returns:
            Maximum allowed proportional change in market share
        """
        # Get tier-specific settings
        tier_settings = None
        
        if self.distribution_settings.get('tier_determination', 'auto') != 'auto':
            # Get from manual tier settings
            manual_tiers = self.distribution_settings.get('manual_tiers', {})
            tier_key = f"tier{tier+1}"
            
            if tier_key in manual_tiers:
                tier_settings = manual_tiers[tier_key]
        
        # Default settings based on tier
        if tier == 0:  # Top tier
            base_max_change = 0.2  # 20% max change for tier 1
        elif tier == 1:  # Middle tier
            base_max_change = 0.3  # 30% max change for tier 2
        else:  # Lower tiers
            base_max_change = 0.5  # 50% max change for lower tiers
        
        # Override with manual settings if available
        if tier_settings and tier_settings.get('max_share_change', 'auto') != 'auto':
            base_max_change = tier_settings.get('max_share_change') / 100.0
        
        # Scale max change based on years ahead (allows more change for later years)
        scaled_max_change = base_max_change * (1.0 + 0.2 * (years_ahead - 1))
        
        # Cap at reasonable values
        return min(scaled_max_change, 1.0)
    
    def _apply_growth_constraints(self, projected_shares: pd.DataFrame, 
                                 historical_shares: pd.DataFrame,
                                 id_col: str) -> pd.DataFrame:
        """
        Apply growth constraints to projected market shares
        
        Args:
            projected_shares: DataFrame with projected market shares
            historical_shares: DataFrame with historical market shares
            id_col: Name of country ID column
            
        Returns:
            DataFrame with growth-constrained market shares
        """
        # Get growth constraint settings
        growth_settings = self.distribution_settings.get('growth_constraints', {})
        determination_method = growth_settings.get('determination_method', 'auto')
        
        # Get redistribution start year from configuration (if specified)
        redistribution_start_year = self.distribution_settings.get('redistribution_start_year', None)
        
        # Determine constraints
        if determination_method == 'auto':
            # Calculate reasonable constraints from historical data
            constraints = self._calculate_dynamic_growth_constraints(historical_shares)
        else:
            # Use manual constraints
            manual_constraints = growth_settings.get('manual_constraints', {})
            constraints = {
                'max_growth_rate': manual_constraints.get('max_growth_rate', 60),
                'min_growth_rate': manual_constraints.get('min_growth_rate', -30),
                'apply_scaling': manual_constraints.get('apply_scaling_by_market_size', True)
            }
        
        # Get the latest historical year or redistribution start year, whichever is later
        if redistribution_start_year is not None:
            reference_year = max(historical_shares['Year'].max(), redistribution_start_year)
            latest_shares = historical_shares[historical_shares['Year'] == reference_year].copy()
        else:
            latest_year = historical_shares['Year'].max()
            latest_shares = historical_shares[historical_shares['Year'] == latest_year].copy()
        
        # Prepare for distributing global market
        constrained_shares_list = []
        
        # Get forecast years
        forecast_years = sorted(projected_shares['Year'].unique())
        
        # Process each year
        prev_year_data = latest_shares
        
        for year in forecast_years:
            # Get projected shares for this year
            year_projection = projected_shares[projected_shares['Year'] == year].copy()
            
            # Merge with previous year data to calculate growth
            merged = pd.merge(
                year_projection,
                prev_year_data[[id_col, 'market_share']],
                on=id_col,
                how='left',
                suffixes=('', '_prev')
            )
            
            # Fill missing previous shares with small values
            merged['market_share_prev'] = merged['market_share_prev'].fillna(0.01)
            
            # Calculate market share growth rate
            merged['growth_rate'] = (merged['market_share'] / merged['market_share_prev'] - 1) * 100
            
            # Apply constraints based on market size
            if constraints.get('apply_scaling', True):
                # Scale constraints based on market size
                for i, row in merged.iterrows():
                    prev_share = row['market_share_prev']
                    
                    # Smaller markets can grow/shrink faster
                    size_factor = 1.0
                    if prev_share < 1.0:
                        size_factor = 2.0
                    elif prev_share < 5.0:
                        size_factor = 1.5
                    
                    max_growth = constraints['max_growth_rate'] * size_factor
                    min_growth = constraints['min_growth_rate'] * size_factor
                    
                    # Apply constraints
                    current_growth = row['growth_rate']
                    
                    if current_growth > max_growth:
                        # Constrain growth rate
                        new_share = row['market_share_prev'] * (1 + max_growth / 100)
                        merged.loc[i, 'market_share'] = new_share
                    elif current_growth < min_growth:
                        # Constrain decline rate
                        new_share = row['market_share_prev'] * (1 + min_growth / 100)
                        merged.loc[i, 'market_share'] = new_share
            else:
                # Apply uniform constraints to all countries
                max_growth = constraints['max_growth_rate']
                min_growth = constraints['min_growth_rate']
                
                # Constrain growth rates
                merged.loc[merged['growth_rate'] > max_growth, 'market_share'] = (
                    merged.loc[merged['growth_rate'] > max_growth, 'market_share_prev'] * (1 + max_growth / 100)
                )
                
                merged.loc[merged['growth_rate'] < min_growth, 'market_share'] = (
                    merged.loc[merged['growth_rate'] < min_growth, 'market_share_prev'] * (1 + min_growth / 100)
                )
            
            # Normalize shares to sum to 100%
            total_share = merged['market_share'].sum()
            if total_share > 0:
                merged['market_share'] = merged['market_share'] / total_share * 100
            
            # Keep relevant columns
            relevant_cols = [id_col, 'Country', 'Year', 'market_share', 'tier']
            year_result = merged[relevant_cols]
            
            # Add to results
            constrained_shares_list.append(year_result)
            
            # Use this year as the previous year for the next iteration
            prev_year_data = year_result.copy()
        
        # Combine all forecast years
        constrained_shares = pd.concat(constrained_shares_list, ignore_index=True)
        
        return constrained_shares
    
    def _calculate_dynamic_growth_constraints(self, historical_shares: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate dynamic growth constraints based on historical data
        
        Args:
            historical_shares: DataFrame with historical market shares
            
        Returns:
            Dictionary with growth constraint parameters
        """
        # Need at least 2 years to calculate historical growth rates
        years = sorted(historical_shares['Year'].unique())
        if len(years) < 2:
            logger.warning("Not enough historical years to calculate dynamic growth constraints")
            return {
                'max_growth_rate': 60,  # Default values
                'min_growth_rate': -30,
                'apply_scaling': True
            }
        
        # Calculate year-over-year growth rates for all countries
        growth_rates = []
        
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            prev_data = historical_shares[historical_shares['Year'] == prev_year]
            curr_data = historical_shares[historical_shares['Year'] == curr_year]
            
            # Merge data to calculate growth
            merged = pd.merge(
                curr_data[['idGeo', 'market_share']],
                prev_data[['idGeo', 'market_share']],
                on='idGeo',
                how='inner',
                suffixes=('_curr', '_prev')
            )
            
            # Calculate growth rates
            merged['growth_rate'] = (merged['market_share_curr'] / merged['market_share_prev'] - 1) * 100
            
            growth_rates.extend(merged['growth_rate'].tolist())
        
        if not growth_rates:
            logger.warning("No growth rates could be calculated from historical data")
            return {
                'max_growth_rate': 60,  # Default values
                'min_growth_rate': -30,
                'apply_scaling': True
            }
        
        # Calculate statistics for growth rates
        growth_rates = np.array(growth_rates)
        q90 = np.percentile(growth_rates, 90)  # 90th percentile for max growth
        q10 = np.percentile(growth_rates, 10)  # 10th percentile for min growth
        
        # Apply reasonable limits
        max_growth = min(max(q90 * 1.5, 40), 100)  # Between 40% and 100%
        min_growth = max(min(q10 * 1.5, -20), -50)  # Between -20% and -50%
        
        logger.info(f"Calculated dynamic growth constraints: max={max_growth:.1f}%, min={min_growth:.1f}%")
        
        return {
            'max_growth_rate': max_growth,
            'min_growth_rate': min_growth,
            'apply_scaling': True
        }
    
    def _calculate_distributed_values(self, combined_shares: pd.DataFrame,
                                     global_forecast: pd.DataFrame,
                                     global_year_col: str,
                                     global_value_col: str) -> pd.DataFrame:
        """
        Calculate absolute market values for each country using global forecast
        
        Args:
            combined_shares: DataFrame with market shares for all years
            global_forecast: DataFrame with global market forecast
            global_year_col: Name of year column in global forecast
            global_value_col: Name of value column in global forecast
            
        Returns:
            DataFrame with distributed market values
        """
        # Create a copy to avoid modifying the original
        result_df = combined_shares.copy()
        
        # Get redistribution start year from configuration (if specified)
        redistribution_start_year = self.distribution_settings.get('redistribution_start_year', None)
        
        # Create a mapping from year to global market value
        global_values = {}
        for _, row in global_forecast.iterrows():
            year = row[global_year_col]
            value = row[global_value_col]
            global_values[year] = value
        
        # Calculate distributed values for each year
        for year in result_df['Year'].unique():
            # Skip years before redistribution_start_year if specified
            # Note: This check is now managed in the distribute_market method by separating preserved data
            # We don't need to skip anything here as the preserved data is added back later
            if year not in global_values:
                logger.warning(f"Year {year} not found in global forecast, skipping")
                continue
            
            global_value = global_values[year]
            year_mask = result_df['Year'] == year
            
            # Calculate absolute value based on market share
            result_df.loc[year_mask, 'Value'] = result_df.loc[year_mask, 'market_share'] * global_value / 100
        
        return result_df
    
    def _apply_smoothing(self, distributed_market: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smoothing to ensure realistic growth patterns
        
        Args:
            distributed_market: DataFrame with distributed market values
            
        Returns:
            DataFrame with smoothed market values
        """
        # Create a copy to avoid modifying the original
        smoothed_df = distributed_market.copy()
        
        # Get redistribution start year from configuration (if specified)
        # Note: We no longer need to check this here since data separation is now handled
        # in the distribute_market method, but keeping it for robustness
        redistribution_start_year = self.distribution_settings.get('redistribution_start_year', None)
        
        # Calculate growth rates
        years = sorted(smoothed_df['Year'].unique())
        if len(years) < 2:
            return smoothed_df
        
        # Add a column for growth rates
        smoothed_df['Growth_Rate'] = np.nan
        
        # Calculate growth rates for each country over time
        countries = smoothed_df['idGeo'].unique()
        country_tier_map = {}
        
        # Determine tier for each country if available
        if 'tier' in smoothed_df.columns:
            for country_id in countries:
                country_data = smoothed_df[smoothed_df['idGeo'] == country_id]
                if 'tier' in country_data.columns and not country_data['tier'].isna().all():
                    country_tier_map[country_id] = country_data['tier'].iloc[0]
        
        # Enhanced smoothing parameters based on market dynamics
        # For markets showing high volatility, apply stronger smoothing
        # These parameters control the amount of smoothing based on market tier
        tier_smoothing_params = {
            1: {'window': 3, 'min_periods': 1, 'center': True, 'max_growth': 35, 'min_growth': -15},  # Tier 1 (leaders)
            2: {'window': 3, 'min_periods': 1, 'center': True, 'max_growth': 40, 'min_growth': -20},  # Tier 2 (established)
            3: {'window': 5, 'min_periods': 1, 'center': True, 'max_growth': 45, 'min_growth': -25},  # Tier 3 (emerging)
            None: {'window': 4, 'min_periods': 1, 'center': True, 'max_growth': 40, 'min_growth': -20}  # Default
        }
        
        # Apply smoothing for each country
        for country_id in countries:
            country_mask = smoothed_df['idGeo'] == country_id
            country_data = smoothed_df.loc[country_mask].sort_values(by='Year')
            
            if len(country_data) < 2:
                continue
            
            # Get tier for smoothing parameters
            tier = country_tier_map.get(country_id, None)
            smoothing_params = tier_smoothing_params.get(tier, tier_smoothing_params[None])
            
            # Calculate year-over-year growth rates for all years
            country_data['Growth_Rate'] = country_data['Value'].pct_change() * 100
            
            # Handle extreme values before smoothing
            # Cap extreme growth rates to make smoothing more effective
            extreme_cap = 80.0  # Cap extreme growth rates
            extreme_floor = -40.0  # Floor for extreme negative growth
            country_data['Growth_Rate'] = country_data['Growth_Rate'].clip(lower=extreme_floor, upper=extreme_cap)
            
            # Apply standard smoothing to all years in this dataset
            # (since data separation is handled in distribute_market)
            if len(country_data) >= 3:
                # First pass: rolling average to handle outliers
                country_data['Smoothed_Growth'] = country_data['Growth_Rate'].rolling(
                    window=smoothing_params['window'], 
                    min_periods=smoothing_params['min_periods'],
                    center=smoothing_params['center']
                ).mean()
                
                # Fill NaN values with nearest non-NaN value for growth rates
                country_data['Smoothed_Growth'] = country_data['Smoothed_Growth'].bfill().ffill()
                
                # Second pass: Apply exponential weighted average for trend smoothing
                country_data['Smoothed_Growth'] = country_data['Smoothed_Growth'].ewm(span=3, adjust=False).mean()
                
                # Apply growth constraints based on tier
                country_data['Smoothed_Growth'] = country_data['Smoothed_Growth'].clip(
                    lower=smoothing_params['min_growth'],
                    upper=smoothing_params['max_growth']
                )
                
                # Apply progressive smoothing for future years to ensure convergence to stable growth
                if len(years) > 3:  # Only apply if we have enough years
                    # Find first year
                    first_year = country_data['Year'].min()
                    
                    # Calculate target long-term growth rate (can be industry-specific)
                    # Adjust target growth rates to be more realistic for each tier
                    target_growth = 15.0 if tier == 1 else 20.0 if tier == 2 else 25.0
                    convergence_rate = 0.25  # How quickly to converge to target growth
                    
                    for i in range(1, len(country_data)):
                        year = country_data.iloc[i]['Year']
                        # Progress towards stable growth for future years
                        years_into_future = max(0, year - first_year)
                        weight = min(0.9, convergence_rate * years_into_future)
                        
                        # Blend current smoothed growth with target long-term growth
                        current_growth = country_data.iloc[i]['Smoothed_Growth']
                        country_data.iloc[i, country_data.columns.get_loc('Smoothed_Growth')] = (
                            (1 - weight) * current_growth + weight * target_growth
                        )
                
                    # Apply smoothed growth rates to calculate values
                    base_value = country_data.iloc[0]['Value']
                    smoothed_values = [base_value]
                    
                    for i in range(1, len(country_data)):
                        smoothed_growth = country_data.iloc[i]['Smoothed_Growth']
                        if not np.isnan(smoothed_growth):
                            new_value = smoothed_values[-1] * (1 + smoothed_growth / 100)
                            smoothed_values.append(new_value)
                            country_data.iloc[i, country_data.columns.get_loc('Value')] = new_value
            
            # Update the main DataFrame
            for idx, row in country_data.iterrows():
                smoothed_df.loc[idx, 'Growth_Rate'] = row['Growth_Rate']
                smoothed_df.loc[idx, 'Value'] = row['Value']
        
        # Ensure total global value is preserved for each year
        for year in years:
            year_mask = smoothed_df['Year'] == year
            original_total = distributed_market.loc[year_mask, 'Value'].sum()
            smoothed_total = smoothed_df.loc[year_mask, 'Value'].sum()
            
            if smoothed_total > 0:
                # Scale values to preserve total
                scaling_factor = original_total / smoothed_total
                smoothed_df.loc[year_mask, 'Value'] = smoothed_df.loc[year_mask, 'Value'] * scaling_factor
        
        # Recalculate market share based on new values
        if 'market_share' in smoothed_df.columns:
            for year in years:
                year_mask = smoothed_df['Year'] == year
                year_total = smoothed_df.loc[year_mask, 'Value'].sum()
                if year_total > 0:
                    smoothed_df.loc[year_mask, 'market_share'] = (
                        smoothed_df.loc[year_mask, 'Value'] / year_total * 100
                    )
        
        # Recalculate growth rates with final values
        for country_id in countries:
            country_mask = smoothed_df['idGeo'] == country_id
            country_data = smoothed_df.loc[country_mask].sort_values(by='Year')
            
            if len(country_data) >= 2:
                country_growth_rates = country_data['Value'].pct_change() * 100
                for idx, growth_rate in zip(country_data.index[1:], country_growth_rates[1:]):
                    smoothed_df.loc[idx, 'Growth_Rate'] = growth_rate
                    
        return smoothed_df 