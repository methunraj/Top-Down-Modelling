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
from src.distribution.regional_aggregator import RegionalAggregator
from src.distribution.gradient_harmonization import GradientHarmonizer
from src.utils.math_utils import (
    safe_divide, normalize_to_sum, calculate_growth_rate,
    apply_growth_bounds, validate_data_consistency
)

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
        self.causal_integration = None
        
        # Get market distribution settings
        self.distribution_settings = self.config_manager.get_market_distribution_settings()
        
        # Initialize dynamic tier classification
        self.tiers = None
        self.tier_thresholds = None
        
        # Initialize regional aggregator
        self.regional_aggregator = RegionalAggregator(config_manager)
        
        # Initialize gradient harmonizer
        self.gradient_harmonizer = GradientHarmonizer(config_manager)
        
        # Load region definitions from provided hierarchy definition if available
        region_definitions = self.config_manager.get_region_definitions()
        region_metadata = self.config_manager.get_region_metadata()
        
        if region_definitions:
            logger.info(f"Loaded {len(region_definitions)} region definitions from configuration")
            self.regional_aggregator.set_hierarchy_definition(region_definitions, region_metadata)
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update distribution settings with new values
        
        Args:
            settings: Dictionary containing updated settings
        """
        logger.info("Updating market distribution settings")
        
        # Update distribution settings
        if settings:
            self.distribution_settings.update(settings)
            
            # If settings include tier-related changes, reset tiers
            if any(key in settings for key in ['tier_determination', 'manual_tiers', 'kmeans_params']):
                logger.info("Resetting tier classification due to settings update")
                self.tiers = None
                self.tier_thresholds = None
            
            # Update gradient harmonization settings if needed
            if 'smoothing' in settings and hasattr(self, 'gradient_harmonizer'):
                self.gradient_harmonizer.update_settings(settings.get('smoothing', {}))
            
            logger.info("Market distribution settings updated successfully")
    
    def safe_market_share_calculation(self, value: float, total: float, min_share: Optional[float] = None) -> float:
        """
        Safely calculate market share with configurable minimum
        
        Args:
            value: The country's market value
            total: The total market value
            min_share: Minimum allowed share (optional)
            
        Returns:
            Calculated market share as a percentage
            
        Raises:
            ValueError: If total market size is not positive
        """
        if total <= 0:
            raise ValueError(f"Total market size must be positive, got {total}")
        
        share = (value / total) * 100.0
        
        if min_share is not None and share < min_share:
            logger.warning(f"Market share {share:.4f}% below minimum {min_share}%, using minimum")
            return min_share
            
        return share
    
    def handle_zero_market(self, year_data: pd.DataFrame, year: int) -> Optional[pd.DataFrame]:
        """
        Handle zero or negative market size consistently
        
        Args:
            year_data: DataFrame with country data for the year
            year: The year being processed
            
        Returns:
            Processed DataFrame or None if year should be skipped
        """
        handling_method = self.distribution_settings.get('zero_market_handling', 'skip')
        
        if handling_method == 'skip':
            logger.warning(f"Skipping year {year} due to zero market size")
            return None
            
        elif handling_method == 'interpolate':
            logger.info(f"Will interpolate values for year {year} in post-processing")
            # Return a marker DataFrame that will be interpolated later
            year_data['market_share'] = np.nan
            year_data['Value'] = np.nan
            year_data['_needs_interpolation'] = True
            return year_data
            
        elif handling_method == 'equal_distribution':
            # Fix: Set both shares and values to zero for mathematical consistency
            logger.info(f"Applying equal distribution for year {year} with zero market")
            n_countries = len(year_data)
            if n_countries > 0:
                # For zero market, both shares and values should be zero
                year_data['market_share'] = 0.0
                year_data['Value'] = 0.0
                year_data['_zero_market'] = True
                logger.warning(f"Zero market for year {year}: setting all shares and values to 0")
                return year_data
            else:
                logger.error(f"No countries available for year {year}")
                return None
                
        else:
            raise ValueError(f"Invalid zero_market_handling method: {handling_method}")
    
    def handle_single_country(self, country_id: str, country_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Properly handle single country markets
        
        Args:
            country_id: The ID of the single country
            country_data: DataFrame with the country's data
            
        Returns:
            Dictionary with tier configuration for single country
        """
        # Get growth constraints from settings or use defaults
        single_country_constraints = self.distribution_settings.get(
            'single_country_constraints',
            {'min_growth_rate': -10.0, 'max_growth_rate': 20.0}
        )
        
        return {
            'tiers': {1: [country_id]},  # Use tier 1 for consistency
            'tier_thresholds': [100.0],   # Single country has 100% share
            'growth_constraints': single_country_constraints,
            'tier_count': 1
        }
    
    def get_valid_growth_bounds(self, market_phase: Optional[str] = None, tier: Optional[int] = None) -> Dict[str, float]:
        """
        Get dynamic growth bounds based on market context
        
        Args:
            market_phase: Market phase (emerging, growth, mature, declining)
            tier: Market tier number
            
        Returns:
            Dictionary with min and max valid growth multipliers
        """
        # Default bounds from settings or fallback values
        default_bounds = self.distribution_settings.get('growth_bounds', {
            'min': 0.1,  # 90% decline max
            'max': 10.0  # 900% growth max
        })
        
        # Market phase specific bounds
        phase_bounds = {
            'emerging': {'min': 0.01, 'max': 50.0},    # Very high growth possible
            'growth': {'min': 0.05, 'max': 20.0},      # High growth phase
            'mature': {'min': 0.5, 'max': 5.0},        # Stable phase
            'declining': {'min': 0.1, 'max': 2.0}      # Limited growth
        }
        
        # Tier specific adjustments (if provided)
        tier_adjustments = {
            1: {'min_mult': 1.5, 'max_mult': 0.7},  # Top tier: more stable
            2: {'min_mult': 1.2, 'max_mult': 0.9},
            3: {'min_mult': 1.0, 'max_mult': 1.0},
            4: {'min_mult': 0.8, 'max_mult': 1.2}   # Lower tier: more volatile
        }
        
        # Start with defaults
        bounds = default_bounds.copy()
        
        # Apply market phase bounds if specified
        if market_phase and market_phase in phase_bounds:
            bounds = phase_bounds[market_phase].copy()
        
        # Apply tier adjustments if specified
        if tier and tier in tier_adjustments:
            adj = tier_adjustments[tier]
            bounds['min'] *= adj['min_mult']
            bounds['max'] *= adj['max_mult']
        
        return bounds
    
    def _detect_market_phase(self, country_data: pd.DataFrame) -> str:
        """
        Detect market phase based on historical growth patterns
        
        Args:
            country_data: Historical data for a country
            
        Returns:
            Market phase: 'emerging', 'growth', 'mature', or 'declining'
        """
        if len(country_data) < 2:
            return 'growth'  # Default assumption
        
        # Calculate average growth rate
        growth_rates = country_data['share_change'].values
        avg_growth = np.mean(growth_rates[np.isfinite(growth_rates)])
        
        # Calculate volatility
        volatility = np.std(growth_rates[np.isfinite(growth_rates)])
        
        # Classify based on growth and volatility patterns
        if avg_growth > 2.0 and volatility > 0.5:
            return 'emerging'
        elif avg_growth > 1.2:
            return 'growth'
        elif avg_growth > 0.8 and volatility < 0.3:
            return 'mature'
        else:
            return 'declining'
    
    def distribute_market(self) -> pd.DataFrame:
        """
        Distribute the global market forecast across countries
        
        Returns:
            DataFrame containing distributed market values for all countries and years
        """
        # Load required data
        global_forecast = self.data_loader.load_global_forecast()
        country_historical = self.data_loader.load_country_historical()
        
        # CRITICAL DEBUG: Log years in source data
        logger.info("=== YEAR TRACKING DEBUG START ===")
        if 'Year' in global_forecast.columns:
            global_years = sorted(global_forecast['Year'].unique())
            logger.info(f"Global forecast years: {global_years}")
            # Check for gaps in global forecast years
            if global_years:
                expected_years = set(range(min(global_years), max(global_years) + 1))
                missing_global_years = expected_years - set(global_years)
                if missing_global_years:
                    logger.warning(f"GAPS in global forecast years: {sorted(missing_global_years)}")
        else:
            logger.warning("No 'Year' column found in global forecast!")
            
        if 'Year' in country_historical.columns:
            country_years = sorted(country_historical['Year'].unique())
            logger.info(f"Country historical years: {country_years}")
        else:
            logger.warning("No 'Year' column found in country historical!")
        
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
        logger.info(f"Redistribution start year: {redistribution_start_year}")
        
        # Perform pre-flight validation of years
        self._validate_year_continuity(global_forecast, country_historical, global_year_col, redistribution_start_year)
        
        # Get indicators data if available
        indicators = None
        try:
            indicators = self.config_manager.get_indicators()
        except Exception as e:
            logger.warning(f"Could not retrieve indicators from config: {str(e)}")
            indicators = None
        
        # CRITICAL CHANGE: If redistribution_start_year is set, we'll process data in two completely separate paths
        if redistribution_start_year is not None:
            logger.info(f"Using redistribution_start_year={redistribution_start_year}. Processing data in two separate paths. "
                       f"Global forecast: {global_forecast.shape[0]} rows, "
                       f"Country historical: {country_historical.shape[0]} rows, "
                       f"Indicators: {len(indicators) if indicators else 0}")
            
            # Path 1: Preserve historical data exactly as is (before redistribution_start_year)
            preserved_historical = country_historical[country_historical['Year'] < redistribution_start_year].copy()
            preserved_years = sorted(preserved_historical['Year'].unique())
            
            if not preserved_historical.empty:
                logger.info(f"Preserving historical data for years: {preserved_years}")
                # CRITICAL: Ensure preserved data has all necessary columns
                # Add market_share column if missing
                if 'market_share' not in preserved_historical.columns:
                    # Calculate market shares for preserved data
                    for year in preserved_years:
                        year_data = preserved_historical[preserved_historical['Year'] == year]
                        total_value = year_data['Value'].sum()
                        if total_value > 0:
                            preserved_historical.loc[preserved_historical['Year'] == year, 'market_share'] = \
                                (preserved_historical.loc[preserved_historical['Year'] == year, 'Value'] / total_value) * 100
            
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
            
            # CRITICAL FIX: Get ALL years from global forecast >= redistribution_start_year
            # Don't filter by Type here - we want both Historical and Forecast years that are >= redistribution_start_year
            all_years_to_process = sorted(global_forecast[global_forecast[global_year_col] >= redistribution_start_year][global_year_col].unique())
            logger.info(f"All years >= {redistribution_start_year} in global forecast: {all_years_to_process}")
            
            # ENHANCEMENT: Also check for any missing years in the global forecast and warn
            if all_years_to_process:
                min_year = min(all_years_to_process)
                max_year = max(all_years_to_process)
                expected_years = set(range(min_year, max_year + 1))
                actual_years = set(all_years_to_process)
                missing_years = expected_years - actual_years
                if missing_years:
                    logger.warning(f"CRITICAL: Missing years in global forecast: {sorted(missing_years)}")
                    logger.info("These years will be interpolated during value calculation")
            
            # Validate year continuity for processing
            if all_years_to_process:
                year_gaps = self._find_year_gaps(all_years_to_process)
                if year_gaps:
                    logger.warning(f"Found gaps in years to process: {year_gaps}")
                    # Note: We don't automatically fill gaps here, but log them for awareness
            
            # Separate into historical and forecast years for logging
            historical_years_to_process = sorted(global_forecast[
                (global_forecast[global_type_col] == 'Historical') & 
                (global_forecast[global_year_col] >= redistribution_start_year)
            ][global_year_col].unique())
            
            forecast_years_to_process = sorted(global_forecast[
                (global_forecast[global_type_col] == 'Forecast') & 
                (global_forecast[global_year_col] >= redistribution_start_year)
            ][global_year_col].unique())
            
            logger.info(f"Historical years >= {redistribution_start_year}: {historical_years_to_process}")
            logger.info(f"Forecast years >= {redistribution_start_year}: {forecast_years_to_process}")
            
            # CRITICAL FIX: Fill any gaps in years to ensure continuity
            if all_years_to_process:
                min_year = min(all_years_to_process)
                max_year = max(all_years_to_process)
                # Create a complete range of years including any gaps
                forecast_years = list(range(min_year, max_year + 1))
                logger.info(f"Filled year gaps - processing complete range: {forecast_years}")
            else:
                forecast_years = all_years_to_process
            
            # Project market shares for all years >= redistribution_start_year (including historical years in that range)
            logger.info(f"Projecting shares for {len(forecast_years)} years: {forecast_years}")
            projected_shares = self._project_market_shares(historical_shares, forecast_years, id_col, country_col)
            
            # Apply indicator adjustments
            if self.indicator_analyzer:
                projected_shares = self.indicator_analyzer.apply_indicator_adjustments(projected_shares)
                
            # Apply causal indicator adjustments if enabled
            if self.causal_integration:
                logger.info("Applying causal indicator adjustments to projected shares")
                projected_shares = self.causal_integration.apply_causal_adjustments(projected_shares)
            
            # Apply growth constraints
            growth_constrained_shares = self._apply_growth_constraints(projected_shares, historical_shares, id_col)
            
            # Combine historical (filtered) and projected shares
            logger.info(f"Historical shares years: {sorted(historical_shares['Year'].unique())}")
            logger.info(f"Growth constrained shares years: {sorted(growth_constrained_shares['Year'].unique())}")
            
            # CRITICAL FIX: Ensure no overlap between historical and projected years when combining
            # Remove any years from historical_shares that are already in growth_constrained_shares
            projected_years = set(growth_constrained_shares['Year'].unique())
            historical_shares_filtered = historical_shares[~historical_shares['Year'].isin(projected_years)]
            
            combined_shares = pd.concat([historical_shares_filtered, growth_constrained_shares], ignore_index=True)
            combined_shares = combined_shares.sort_values(by=['Year', id_col])
            logger.info(f"Combined shares include years: {sorted(combined_shares['Year'].unique())}")
            
            # Calculate absolute market values using global forecast
            distributed_market = self._calculate_distributed_values(combined_shares, global_forecast, global_year_col, global_value_col, redistribution_start_year)
            
            # Apply smoothing
            smoothed_market = self._apply_smoothing(distributed_market)
            
            # Now merge the preserved historical data with the processed data
            # Ensure the preserved data has all necessary columns
            if not preserved_historical.empty:
                logger.info(f"Merging preserved years {preserved_years} with processed years {sorted(smoothed_market['Year'].unique())}")
                
                # Ensure both dataframes have the same columns for proper concatenation
                all_columns = set(preserved_historical.columns) | set(smoothed_market.columns)
                for col in all_columns:
                    if col not in preserved_historical.columns:
                        preserved_historical[col] = None
                    if col not in smoothed_market.columns:
                        smoothed_market[col] = None
                
                # Ensure column order is consistent
                common_columns = sorted(list(all_columns))
                preserved_historical = preserved_historical[common_columns]
                smoothed_market = smoothed_market[common_columns]
                
                final_result = pd.concat([preserved_historical, smoothed_market], ignore_index=True)
                final_result = final_result.sort_values(by=[id_col, 'Year'])
                
                # Validate no duplicate years for any country
                duplicates = final_result.groupby([id_col, 'Year']).size()
                duplicates = duplicates[duplicates > 1]
                if not duplicates.empty:
                    logger.warning(f"Found duplicate entries after merge: {duplicates}")
                    # Remove duplicates, keeping the processed data over preserved when there's overlap
                    final_result = final_result.drop_duplicates(subset=[id_col, 'Year'], keep='last')
                
                logger.info(f"Final result includes years: {sorted(final_result['Year'].unique())}")
            else:
                final_result = smoothed_market
            
            # Apply regional aggregation if enabled
            if self.distribution_settings.get('enable_regional_aggregation', True):
                # Perform hierarchical regional aggregation
                final_result = self.regional_aggregator.aggregate_hierarchical(
                    final_result, id_col=id_col, country_col=country_col, value_col='Value'
                )
                
                # Enforce regional consistency
                final_result = self.regional_aggregator.enforce_regional_consistency(
                    final_result, id_col=id_col, country_col=country_col, value_col='Value', 
                    method=self.distribution_settings.get('regional_consistency_method', 'hybrid')
                )
            
            # CRITICAL: Final validation for redistribution path
            logger.info("Performing final validation of distributed totals (redistribution path)...")
            
            # Recreate global values mapping for validation
            global_values_final = {}
            for _, row in global_forecast.iterrows():
                year = row[global_year_col]
                value = row[global_value_col]
                global_values_final[year] = value
            
            for year in sorted(final_result['Year'].unique()):
                year_mask = final_result['Year'] == year
                distributed_total = final_result.loc[year_mask, 'Value'].sum()
                
                # Handle interpolated years
                if year not in global_values_final:
                    interpolated_value = self._interpolate_missing_year(year, global_values_final, redistribution_start_year)
                    if interpolated_value is not None:
                        logger.info(f"Using interpolated global value for year {year}: {interpolated_value:.2f}")
                        global_values_final[year] = interpolated_value
                    else:
                        logger.warning(f"Cannot find or interpolate global value for year {year}, skipping validation")
                        continue
                
                if year in global_values_final:
                    global_total = global_values_final[year]
                    deviation = abs(distributed_total - global_total)
                    deviation_pct = (deviation / global_total * 100) if global_total > 0 else 0
                    
                    if deviation > 0.001:  # More than $0.001 deviation
                        logger.warning(f"FINAL VALIDATION: Year {year} - "
                                     f"Distributed: {distributed_total:,.2f}, Global: {global_total:,.2f}, "
                                     f"Deviation: {deviation:,.2f} ({deviation_pct:.4f}%)")
                        
                        # Force correction
                        if distributed_total > 0:
                            correction_factor = global_total / distributed_total
                            final_result.loc[year_mask, 'Value'] = final_result.loc[year_mask, 'Value'] * correction_factor
                            logger.info(f"Applied correction factor {correction_factor:.8f} for year {year}")
            
            return final_result
            
        else:
            # Original path - process all years
            logger.info("No redistribution_start_year specified. Processing all years.")
            
            # Get years for history and forecast
            country_historical_years = sorted(country_historical['Year'].unique())
            latest_country_historical_year = max(country_historical_years)
            
            # Get all years from global forecast
            all_years = sorted(global_forecast[global_year_col].unique())
            
            # Debug: Check Type column values
            if global_type_col in global_forecast.columns:
                type_values = global_forecast[global_type_col].unique()
                logger.info(f"Type column values in global forecast: {type_values}")
            else:
                logger.warning(f"Type column '{global_type_col}' not found in global forecast!")
            
            # Get historical years FROM GLOBAL FORECAST (not just country data)
            historical_mask = global_forecast[global_type_col] == 'Historical'
            global_historical_years = sorted(global_forecast.loc[historical_mask, global_year_col].unique())
            
            # Separate forecast years
            forecast_mask = global_forecast[global_type_col] == 'Forecast'
            forecast_years = sorted(global_forecast.loc[forecast_mask, global_year_col].unique())
            
            # CRITICAL FIX: If no forecast years found due to missing Type column, infer from country data
            if len(forecast_years) == 0:
                logger.warning("No forecast years found using Type column. Inferring from year ranges...")
                # All years in global forecast that are after the latest country historical year
                forecast_years = sorted([y for y in all_years if y > latest_country_historical_year])
                logger.info(f"Inferred forecast years: {forecast_years}")
                
                # Also infer historical years if needed
                if len(global_historical_years) == 0:
                    global_historical_years = sorted([y for y in all_years if y <= latest_country_historical_year])
                    logger.info(f"Inferred global historical years: {global_historical_years}")
            
            logger.info(f"Country historical years: {country_historical_years}")
            logger.info(f"Global historical years: {global_historical_years}")
            logger.info(f"Forecast years: {forecast_years}")
            logger.info(f"All years in global forecast: {all_years}")
            
            # Calculate historical market shares for years that have country data
            historical_shares_from_country = self._calculate_historical_shares(
                country_historical, country_historical_years, id_col, country_col)
            
            # For global historical years missing in country data, 
            # use the latest available country year's shares
            missing_historical_years = set(global_historical_years) - set(country_historical_years)
            if missing_historical_years:
                logger.info(f"Global forecast has historical data for years {sorted(missing_historical_years)} "
                           f"not in country data. Using shares from {latest_country_historical_year}")
                
                # Get shares from the latest country historical year
                latest_shares = historical_shares_from_country[
                    historical_shares_from_country['Year'] == latest_country_historical_year
                ].copy()
                
                # Create entries for missing years using latest shares
                missing_years_shares = []
                for year in sorted(missing_historical_years):
                    year_shares = latest_shares.copy()
                    year_shares['Year'] = year
                    missing_years_shares.append(year_shares)
                
                # Combine all historical shares
                historical_shares = pd.concat(
                    [historical_shares_from_country] + missing_years_shares, 
                    ignore_index=True
                )
            else:
                historical_shares = historical_shares_from_country
            
            # Detect market tiers automatically if configured to do so
            if self.distribution_settings.get('tier_determination', 'auto') == 'auto':
                # Use the latest year that has actual country data for tier determination
                self._determine_tiers(historical_shares, latest_country_historical_year)
            else:
                # Use manual tier settings from configuration
                self._load_manual_tiers()
            
            # CRITICAL FIX: Include any missing years in the forecast
            # Get all years that should be in the result
            all_result_years = sorted(set(list(historical_shares['Year'].unique()) + forecast_years))
            
            # Check for gaps and fill them
            if all_result_years:
                min_year = min(all_result_years)
                max_year = max(all_result_years)
                complete_years = list(range(min_year, max_year + 1))
                missing_in_result = set(complete_years) - set(all_result_years)
                if missing_in_result:
                    logger.warning(f"Missing years detected in result: {sorted(missing_in_result)}")
                    # Add missing years to forecast_years if they're not in historical
                    hist_years_set = set(historical_shares['Year'].unique())
                    for year in sorted(missing_in_result):
                        if year not in hist_years_set:
                            forecast_years.append(year)
                    forecast_years = sorted(forecast_years)
                    logger.info(f"Updated forecast years to include missing: {forecast_years}")
            
            # Project market shares for forecast years
            projected_shares = self._project_market_shares(historical_shares, forecast_years, id_col, country_col)
            
            # Apply indicator adjustments to projected shares if indicators are available
            if self.indicator_analyzer:
                projected_shares = self.indicator_analyzer.apply_indicator_adjustments(projected_shares)
                
            # Apply causal indicator adjustments if enabled
            if self.causal_integration:
                logger.info("Applying causal indicator adjustments to projected shares")
                projected_shares = self.causal_integration.apply_causal_adjustments(projected_shares)
            
            # Apply growth constraints based on market dynamics
            growth_constrained_shares = self._apply_growth_constraints(projected_shares, historical_shares, id_col)
            
            # Combine historical and projected shares
            combined_shares = pd.concat([historical_shares, growth_constrained_shares], ignore_index=True)
            logger.info(f"Combined shares include years: {sorted(combined_shares['Year'].unique())}")
            
            # Calculate absolute market values using global forecast
            distributed_market = self._calculate_distributed_values(combined_shares, global_forecast, global_year_col, global_value_col, redistribution_start_year)
            
            # Apply smoothing to ensure realistic growth patterns
            smoothed_market = self._apply_smoothing(distributed_market)
            
            # Log totals before regional aggregation
            logger.info("Checking totals before regional aggregation...")
            for year in sorted(smoothed_market['Year'].unique()):
                year_total = smoothed_market[smoothed_market['Year'] == year]['Value'].sum()
                logger.info(f"Year {year}: Total before regional aggregation = {year_total:,.2f}")
            
            # Apply regional aggregation if enabled
            if self.distribution_settings.get('enable_regional_aggregation', True):
                # Perform hierarchical regional aggregation
                smoothed_market = self.regional_aggregator.aggregate_hierarchical(
                    smoothed_market, id_col=id_col, country_col=country_col, value_col='Value'
                )
                
                # Enforce regional consistency
                smoothed_market = self.regional_aggregator.enforce_regional_consistency(
                    smoothed_market, id_col=id_col, country_col=country_col, value_col='Value',
                    method=self.distribution_settings.get('regional_consistency_method', 'hybrid')
                )
                
                # Log totals after regional aggregation
                logger.info("Checking totals after regional aggregation...")
                for year in sorted(smoothed_market['Year'].unique()):
                    year_total = smoothed_market[smoothed_market['Year'] == year]['Value'].sum()
                    logger.info(f"Year {year}: Total after regional aggregation = {year_total:,.2f}")
            
            # FINAL VALIDATION: Ensure distributed totals match global forecast exactly
            logger.info("Performing final validation of distributed totals...")
            
            # Recreate global values mapping for validation
            global_values_final = {}
            for _, row in global_forecast.iterrows():
                year = row[global_year_col]
                value = row[global_value_col]
                global_values_final[year] = value
            
            for year in sorted(smoothed_market['Year'].unique()):
                year_mask = smoothed_market['Year'] == year
                distributed_total = smoothed_market.loc[year_mask, 'Value'].sum()
                
                # CRITICAL FIX: Handle interpolated years properly
                if year not in global_values_final:
                    # Try to interpolate the global value for this year
                    interpolated_value = self._interpolate_missing_year(year, global_values_final, redistribution_start_year)
                    if interpolated_value is not None:
                        logger.info(f"Using interpolated global value for year {year}: {interpolated_value:.2f}")
                        global_values_final[year] = interpolated_value
                    else:
                        logger.warning(f"Cannot find or interpolate global value for year {year}, skipping validation")
                        continue
                
                if year in global_values_final:
                    global_total = global_values_final[year]
                    deviation = abs(distributed_total - global_total)
                    deviation_pct = (deviation / global_total * 100) if global_total > 0 else 0
                    
                    if deviation > 0.001:  # More than $0.001 deviation (very strict)
                        logger.warning(f"FINAL VALIDATION: Year {year} - "
                                     f"Distributed: {distributed_total:,.2f}, Global: {global_total:,.2f}, "
                                     f"Deviation: {deviation:,.2f} ({deviation_pct:.4f}%)")
                        
                        # ALWAYS force correction to ensure exact match
                        if distributed_total > 0:
                            correction_factor = global_total / distributed_total
                            smoothed_market.loc[year_mask, 'Value'] = smoothed_market.loc[year_mask, 'Value'] * correction_factor
                            
                            # Verify correction worked
                            new_total = smoothed_market.loc[year_mask, 'Value'].sum()
                            final_deviation = abs(new_total - global_total)
                            if final_deviation > 0.001:
                                logger.error(f"Correction failed! New total: {new_total:,.2f}, still deviates by {final_deviation:,.2f}")
                            else:
                                logger.info(f"Applied correction factor {correction_factor:.8f} for year {year} - now matches exactly")
                    else:
                        logger.info(f"Year {year}: Already matches exactly (deviation: ${deviation:.6f})")
            
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
                logger.warning(f"Total market size for year {year} is zero or negative ({total_market})")
                # Use the dedicated handler for zero market situations
                processed_data = self.handle_zero_market(year_data, year)
                if processed_data is not None:
                    # Keep relevant columns including any markers
                    cols_to_keep = [id_col, country_col, 'Year', 'Value', 'market_share']
                    # Add marker columns if they exist
                    for marker_col in ['_needs_interpolation', '_zero_market']:
                        if marker_col in processed_data.columns:
                            cols_to_keep.append(marker_col)
                    shares_list.append(processed_data[cols_to_keep])
                continue
            
            # Calculate market share for each country using safe method
            year_data['market_share'] = year_data['Value'].apply(
                lambda v: self.safe_market_share_calculation(v, total_market)
            )
            
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
        # Get column mappings
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        
        # Get the latest year data
        latest_data = historical_shares[historical_shares['Year'] == latest_year].copy()
        
        # Sort by market share descending
        latest_data = latest_data.sort_values(by='market_share', ascending=False)
        
        # Get shares as a numpy array for clustering
        shares = latest_data['market_share'].values.reshape(-1, 1)
        
        # Determine optimal number of clusters using silhouette score
        from sklearn.metrics import silhouette_score
        
        max_clusters = min(8, len(shares))  # At most 8 tiers, and can't exceed number of countries
        min_clusters = min(3, len(shares))  # At least 3 tiers, but not more than available countries
        max_clusters = max(min_clusters, max_clusters)  # Ensure max >= min
        
        best_score = -1
        best_n_clusters = min_clusters  # Default to minimum viable clusters
        
        # Handle edge case: only one country
        if len(shares) == 1:
            # Handle single country case properly
            country_id = latest_data.iloc[0][id_col]
            single_config = self.handle_single_country(country_id, latest_data)
            
            latest_data['tier'] = 1  # Use tier 1 for consistency
            self.tiers = single_config['tiers']
            self.tier_thresholds = single_config['tier_thresholds']
            
            logger.info(f"Single country case: {country_id} assigned to tier 1 with 100% market share")
            return
        
        # Try different numbers of clusters
        for n_clusters in range(min_clusters, max_clusters + 1):
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
        
        # Calculate tier thresholds with improved logic
        self.tier_thresholds = []
        
        for tier in range(best_n_clusters):
            tier_data = latest_data[latest_data['tier'] == tier]
            
            if tier_data.empty:
                # Handle empty tier case
                if tier == 0:
                    threshold = latest_data['market_share'].max() * 0.8
                else:
                    threshold = self.tier_thresholds[-1] * 0.5 if self.tier_thresholds else 0.1
                self.tier_thresholds.append(threshold)
                continue
            
            # Find minimum share in this tier
            min_share = tier_data['market_share'].min()
            max_share = tier_data['market_share'].max()
            
            # Improved threshold calculation
            if tier == 0:
                # For highest tier, use minimum share in tier
                threshold = min_share
            elif tier < best_n_clusters - 1:
                # For middle tiers, use geometric mean between this tier's min and next tier's max
                next_tier_data = latest_data[latest_data['tier'] == tier + 1]
                if not next_tier_data.empty:
                    next_tier_max = next_tier_data['market_share'].max()
                    # Use geometric mean to avoid unrealistic thresholds
                    threshold = np.sqrt(min_share * next_tier_max)
                else:
                    threshold = min_share * 0.5
            else:
                # For lowest tier, ensure reasonable minimum threshold
                threshold = max(min_share * 0.3, 0.01)  # At least 0.01% market share
            
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
            
            # Validate market shares before normalization
            total_share = year_projection['market_share'].sum()
            
            # Check for mathematical consistency before normalization
            share_deviation = abs(total_share - 100.0)
            if share_deviation > 10.0:  # More than 10% deviation warrants investigation
                logger.warning(f"Year {year}: Market shares sum to {total_share:.2f}% (deviation: {share_deviation:.2f}%)")
            
            # Check for invalid individual shares
            invalid_shares = year_projection['market_share'].isna() | (year_projection['market_share'] < 0)
            if invalid_shares.any():
                invalid_countries = year_projection[invalid_shares]['Country'].tolist()
                logger.error(f"Year {year}: Invalid market shares for countries: {invalid_countries}")
                # Set invalid shares to 0 before normalization
                year_projection.loc[invalid_shares, 'market_share'] = 0.0
                total_share = year_projection['market_share'].sum()
            
            # Normalize shares to sum to 100% with enhanced safety checks
            if total_share > 0 and not np.isnan(total_share) and not np.isinf(total_share):
                # Store original shares for validation
                original_shares = year_projection['market_share'].copy()
                year_projection['market_share'] = year_projection['market_share'] / total_share * 100
                
                # Validate normalization didn't introduce errors
                new_total = year_projection['market_share'].sum()
                if abs(new_total - 100.0) > 0.01:  # 0.01% tolerance for floating point
                    logger.error(f"Year {year}: Normalization failed, total after normalization: {new_total:.6f}%")
            else:
                # Enhanced fallback with equal distribution
                logger.warning(f"Year {year}: Invalid total share {total_share}, applying equal distribution")
                n_countries = len(year_projection)
                if n_countries > 0:
                    year_projection['market_share'] = 100.0 / n_countries
                    logger.info(f"Year {year}: Applied equal distribution of {100.0/n_countries:.4f}% to {n_countries} countries")
                else:
                    logger.error(f"Year {year}: No countries found for distribution - cannot apply equal distribution")
                    # Create a placeholder entry to maintain data integrity
                    continue
            
            # Add to results
            projected_shares_list.append(year_projection)
        
        # Combine all forecast years
        projected_shares = pd.concat(projected_shares_list, ignore_index=True)
        
        # Critical: Validate mathematical consistency after projection
        self._validate_market_share_consistency(projected_shares, "after market share projection")
        
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
            # Safe division for share change calculation with minimum threshold
            MIN_SHARE_THRESHOLD = 1e-6
            country_data['share_change'] = np.where(
                (country_data['prev_share'] > MIN_SHARE_THRESHOLD) & (~pd.isna(country_data['prev_share'])),
                country_data['market_share'] / country_data['prev_share'],
                1.0  # No change if previous share was zero or near-zero
            )
            
            # Remove first row (no previous share)
            country_data = country_data.dropna(subset=['share_change'])
            
            if country_data.empty:
                trend_factors.append({
                    id_col: country_id,
                    'trend_factor': 1.0  # Neutral trend
                })
                continue
            
            # Calculate weighted average of share changes with improved logic
            weights = np.linspace(0.5, 1.0, len(country_data))
            
            # Remove outliers and invalid values before calculating average
            valid_changes = country_data['share_change']
            valid_changes = valid_changes[np.isfinite(valid_changes)]  # Remove NaN and inf
            
            # Use dynamic bounds based on market context
            # Try to detect market phase from historical data
            market_phase = self._detect_market_phase(country_data)
            bounds = self.get_valid_growth_bounds(market_phase=market_phase)
            
            valid_changes = valid_changes[(valid_changes > bounds['min']) & (valid_changes < bounds['max'])]
            
            if len(valid_changes) == 0:
                # No valid changes, use neutral trend
                weighted_avg_change = 1.0
            else:
                # Recalculate weights for valid data only
                weights = weights[-len(valid_changes):]
                weighted_avg_change = np.average(valid_changes, weights=weights)
            
            # Apply more conservative dampening to extreme trends
            if weighted_avg_change > 1.0:
                # Limit growth trends more aggressively
                dampened_change = 1.0 + (weighted_avg_change - 1.0) * 0.5
                dampened_change = min(dampened_change, 1.3)  # Max 30% annual growth
            elif weighted_avg_change < 1.0:
                # Limit decline trends more conservatively
                dampened_change = 1.0 - (1.0 - weighted_avg_change) * 0.5
                dampened_change = max(dampened_change, 0.8)  # Max 20% annual decline
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
        
        # Get the appropriate reference year
        if redistribution_start_year is not None:
            # When redistribution year is set, we need the year just before redistribution
            # or the redistribution year itself if it's in historical data
            hist_years = sorted(historical_shares['Year'].unique())
            
            if redistribution_start_year in hist_years:
                reference_year = redistribution_start_year
            else:
                # Find the latest year before redistribution_start_year
                years_before = [y for y in hist_years if y < redistribution_start_year]
                if years_before:
                    reference_year = max(years_before)
                else:
                    # If no years before, use the earliest available
                    reference_year = min(hist_years)
            
            logger.info(f"Using reference year {reference_year} for growth constraints (redistribution year: {redistribution_start_year})")
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
            
            # Calculate growth rates safely
            merged['growth_rate'] = merged.apply(
                lambda row: calculate_growth_rate(
                    row['market_share'], 
                    row['market_share_prev'],
                    max_rate=200.0,  # Allow higher rates for market share changes
                    min_rate=-90.0   # Market share can drop significantly
                ), axis=1
            )
            
            # Apply constraints based on market size
            if constraints.get('apply_scaling', True):
                # First, handle invalid market_share_prev values
                invalid_prev = (merged['market_share_prev'].isna()) | (merged['market_share_prev'] <= 0)
                if invalid_prev.any():
                    min_share = self.distribution_settings.get('min_market_share', 0.01)
                    merged.loc[invalid_prev, 'market_share_prev'] = min_share
                    invalid_countries = merged.loc[invalid_prev, id_col].tolist()
                    logger.warning(f"Set minimum share for countries with invalid prev_share: {invalid_countries}")
                
                # Vectorized size factor calculation
                size_factors = np.where(merged['market_share_prev'] < 1.0, 2.0,
                                      np.where(merged['market_share_prev'] < 5.0, 1.5, 1.0))
                
                # Vectorized constraint calculation
                max_growth_vec = constraints['max_growth_rate'] * size_factors
                min_growth_vec = constraints['min_growth_rate'] * size_factors
                
                # Apply constraints vectorized
                over_max = merged['growth_rate'] > max_growth_vec
                under_min = merged['growth_rate'] < min_growth_vec
                
                # Constrain growth rates
                merged.loc[over_max, 'market_share'] = (
                    merged.loc[over_max, 'market_share_prev'] * (1 + max_growth_vec[over_max] / 100)
                )
                merged.loc[under_min, 'market_share'] = (
                    merged.loc[under_min, 'market_share_prev'] * (1 + min_growth_vec[under_min] / 100)
                )
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
            
            # Normalize shares to sum to 100% using safe method
            merged['market_share'] = normalize_to_sum(
                merged['market_share'],
                target_sum=100.0,
                min_value=self.distribution_settings.get('min_market_share', 0.001)
            )
            
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
    
    def _validate_market_share_consistency(self, data: pd.DataFrame, context: str = "") -> None:
        """
        Validate mathematical consistency of market shares.
        
        Args:
            data: DataFrame with market share data
            context: Description of when this validation is being performed
        """
        if data.empty:
            logger.warning(f"Market share validation ({context}): No data to validate")
            return
        
        # Check each year separately
        years = data['Year'].unique()
        for year in years:
            year_data = data[data['Year'] == year]
            total_share = year_data['market_share'].sum()
            
            # Check sum consistency (should be 100%)
            deviation = abs(total_share - 100.0)
            if deviation > 0.01:  # 0.01% tolerance
                if deviation > 1.0:  # More than 1% is critical
                    logger.error(f"CRITICAL: Market shares for year {year} sum to {total_share:.4f}% "
                               f"(deviation: {deviation:.4f}%) - {context}")
                else:
                    logger.warning(f"Market shares for year {year} sum to {total_share:.4f}% "
                                 f"(deviation: {deviation:.4f}%) - {context}")
            
            # Check for invalid individual shares
            invalid_shares = (year_data['market_share'] < 0) | year_data['market_share'].isna()
            if invalid_shares.any():
                invalid_count = invalid_shares.sum()
                logger.error(f"Year {year}: {invalid_count} countries have invalid market shares - {context}")
            
            # Check for unrealistic shares (>50% for any single country might be suspicious)
            high_shares = year_data['market_share'] > 50.0
            if high_shares.any():
                high_countries = year_data[high_shares]['Country'].tolist()
                logger.info(f"Year {year}: Countries with >50% market share: {high_countries} - {context}")
    
    def _validate_global_consistency(self, country_data: pd.DataFrame, global_forecast: pd.DataFrame, 
                                   year: int, context: str = "") -> None:
        """
        Validate that country-level values sum to global forecast values.
        
        Args:
            country_data: DataFrame with country-level data
            global_forecast: DataFrame with global forecast data
            year: Year to validate
            context: Description of validation context
        """
        if country_data.empty or global_forecast.empty:
            logger.warning(f"Global consistency validation ({context}): Missing data for year {year}")
            return
        
        # Get global value for the year
        global_year_data = global_forecast[global_forecast['Year'] == year]
        if global_year_data.empty:
            logger.warning(f"No global forecast data for year {year} - {context}")
            return
        
        global_value = global_year_data['Value'].iloc[0]
        
        # Sum country values
        country_total = country_data['Value'].sum()
        
        # Check consistency
        deviation_abs = abs(country_total - global_value)
        deviation_pct = (deviation_abs / global_value * 100) if global_value > 0 else float('inf')
        
        if deviation_pct > 1.0:  # More than 1% deviation
            logger.error(f"CRITICAL: Year {year} - Country values sum to {country_total:,.0f}, "
                        f"global forecast is {global_value:,.0f} "
                        f"(deviation: {deviation_pct:.2f}%) - {context}")
        elif deviation_pct > 0.1:  # More than 0.1% deviation
            logger.warning(f"Year {year} - Country values sum to {country_total:,.0f}, "
                         f"global forecast is {global_value:,.0f} "
                         f"(deviation: {deviation_pct:.2f}%) - {context}")
    
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
    
    def set_causal_integration(self, causal_integration) -> None:
        """
        Set the causal integration component for enhanced indicator analysis
        
        Args:
            causal_integration: CausalIndicatorIntegration instance
        """
        self.causal_integration = causal_integration
        logger.info("Causal indicator integration enabled for market distribution")
    
    def _calculate_distributed_values(self, combined_shares: pd.DataFrame,
                                     global_forecast: pd.DataFrame,
                                     global_year_col: str,
                                     global_value_col: str,
                                     redistribution_start_year: Optional[int] = None) -> pd.DataFrame:
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
        logger.info(f"Years to calculate values for: {sorted(result_df['Year'].unique())}")
        logger.info(f"Years available in global forecast: {sorted(global_values.keys())}")
        
        for year in sorted(result_df['Year'].unique()):
            # Skip years before redistribution_start_year if specified
            # Note: This check is now managed in the distribute_market method by separating preserved data
            # We don't need to skip anything here as the preserved data is added back later
            if year not in global_values:
                logger.warning(f"Year {year} not found in global forecast - will interpolate")
                # Enhanced interpolation logic with validation
                interpolated_value = self._interpolate_missing_year(year, global_values, redistribution_start_year)
                if interpolated_value is None:
                    logger.error(f"Cannot interpolate value for year {year}. Trying alternative approaches...")
                    # CRITICAL FIX: Try to use average growth rate as fallback
                    if len(global_values) >= 2:
                        sorted_years = sorted(global_values.keys())
                        # Calculate average growth rate
                        growth_rates = []
                        for i in range(1, len(sorted_years)):
                            prev_val = global_values[sorted_years[i-1]]
                            curr_val = global_values[sorted_years[i]]
                            if prev_val > 0:
                                growth_rate = (curr_val / prev_val) - 1
                                growth_rates.append(growth_rate)
                        
                        if growth_rates:
                            avg_growth = sum(growth_rates) / len(growth_rates)
                            # Find closest year with data
                            closest_year = min(global_values.keys(), key=lambda y: abs(y - year))
                            base_value = global_values[closest_year]
                            years_diff = year - closest_year
                            interpolated_value = base_value * ((1 + avg_growth) ** years_diff)
                            logger.info(f"Used average growth rate {avg_growth:.2%} to interpolate year {year}: {interpolated_value:.2f}")
                        else:
                            logger.error(f"Cannot calculate growth rate for interpolation. Skipping year {year}")
                            continue
                    else:
                        logger.error(f"Not enough data points for interpolation. Skipping year {year}")
                        continue
                global_value = interpolated_value
            else:
                global_value = global_values[year]
            year_mask = result_df['Year'] == year
            year_data = result_df[year_mask]
            
            # CRITICAL: Validate market shares before calculating values
            total_share = year_data['market_share'].sum()
            if abs(total_share - 100.0) > 0.01:  # More than 0.01% deviation
                logger.error(f"CRITICAL: Year {year} market shares sum to {total_share:.4f}% before value calculation")
                # Normalize to ensure mathematical consistency
                if total_share > 0:
                    result_df.loc[year_mask, 'market_share'] = result_df.loc[year_mask, 'market_share'] / total_share * 100
                    logger.info(f"Year {year}: Normalized market shares to sum to 100%")
            
            # Calculate absolute value based on market share
            result_df.loc[year_mask, 'Value'] = result_df.loc[year_mask, 'market_share'] * global_value / 100
            
            # CRITICAL: Validate that values sum to global forecast
            calculated_total = result_df.loc[year_mask, 'Value'].sum()
            deviation_pct = abs(calculated_total - global_value) / global_value * 100
            if deviation_pct > 0.01:  # More than 0.01% deviation
                logger.error(f"CRITICAL: Year {year} calculated values sum to {calculated_total:,.0f}, "
                           f"should be {global_value:,.0f} (deviation: {deviation_pct:.4f}%)")
                # Force exact alignment
                if calculated_total > 0:
                    scaling_factor = global_value / calculated_total
                    result_df.loc[year_mask, 'Value'] = result_df.loc[year_mask, 'Value'] * scaling_factor
                    logger.info(f"Year {year}: Applied scaling factor {scaling_factor:.6f} to ensure global consistency")
        
        return result_df
    
    def _apply_smoothing(self, distributed_market: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smoothing to ensure realistic growth patterns
        
        Args:
            distributed_market: DataFrame with distributed market values
            
        Returns:
            DataFrame with smoothed market values
        """
        # Check if gradient harmonization is enabled
        use_gradient_harmonization = self.distribution_settings.get('use_gradient_harmonization', True)
        
        if use_gradient_harmonization:
            # Use the advanced Gradient Harmonization Algorithm
            logger.info("Applying Gradient Harmonization Algorithm for trajectory smoothing")
            
            # Mark forecast vs historical data if not already marked
            if 'is_forecast' not in distributed_market.columns:
                # Attempt to infer from type column if available
                if 'Type' in distributed_market.columns:
                    distributed_market['is_forecast'] = distributed_market['Type'] == 'Forecast'
                else:
                    # Can't determine, algorithm will infer from data patterns
                    pass
            
            # Apply the gradient harmonization
            return self.gradient_harmonizer.harmonize_market_trajectories(distributed_market)
        
        # Fall back to the original smoothing method if gradient harmonization is disabled
        logger.info("Using legacy smoothing method (gradient harmonization disabled)")
        
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
            
            if smoothed_total > 0 and original_total > 0:
                # Scale values to preserve total
                scaling_factor = original_total / smoothed_total
                smoothed_df.loc[year_mask, 'Value'] = smoothed_df.loc[year_mask, 'Value'] * scaling_factor
            elif original_total == 0:
                # Handle zero market case - set all values to zero
                smoothed_df.loc[year_mask, 'Value'] = 0.0
        
        # Recalculate market share based on new values
        if 'market_share' in smoothed_df.columns:
            for year in years:
                year_mask = smoothed_df['Year'] == year
                year_total = smoothed_df.loc[year_mask, 'Value'].sum()
                if year_total > 0:
                    smoothed_df.loc[year_mask, 'market_share'] = (
                        smoothed_df.loc[year_mask, 'Value'] / year_total * 100
                    )
        
        # CRITICAL: Final validation of mathematical consistency
        self._validate_market_share_consistency(smoothed_df, "after gradient harmonization and smoothing")
        
        # Recalculate growth rates with final values
        for country_id in countries:
            country_mask = smoothed_df['idGeo'] == country_id
            country_data = smoothed_df.loc[country_mask].sort_values(by='Year')
            
            if len(country_data) >= 2:
                country_growth_rates = country_data['Value'].pct_change() * 100
                for idx, growth_rate in zip(country_data.index[1:], country_growth_rates[1:]):
                    smoothed_df.loc[idx, 'Growth_Rate'] = growth_rate
                    
        return smoothed_df
    
    def _validate_year_continuity(self, global_forecast: pd.DataFrame, country_historical: pd.DataFrame, 
                                  global_year_col: str, redistribution_start_year: Optional[int] = None) -> None:
        """
        Validate year continuity in data and log any issues
        
        Args:
            global_forecast: Global forecast data
            country_historical: Country historical data
            global_year_col: Column name for years in global forecast
            redistribution_start_year: Optional redistribution start year
        """
        # Get all years from both datasets
        global_years = sorted(global_forecast[global_year_col].unique())
        country_years = sorted(country_historical['Year'].unique())
        
        logger.info(f"Year validation - Global forecast years: {global_years}")
        logger.info(f"Year validation - Country historical years: {country_years}")
        
        # Check for gaps in global forecast
        if global_years:
            expected_years = list(range(min(global_years), max(global_years) + 1))
            missing_global_years = set(expected_years) - set(global_years)
            if missing_global_years:
                logger.warning(f"GAPS found in global forecast years: {sorted(missing_global_years)}")
        
        # If redistribution year is set, validate continuity around it
        if redistribution_start_year:
            # Check if redistribution year exists in global forecast
            if redistribution_start_year not in global_years:
                logger.warning(f"Redistribution start year {redistribution_start_year} not found in global forecast!")
                
            # Check years around redistribution year
            years_around = [y for y in global_years if abs(y - redistribution_start_year) <= 2]
            logger.info(f"Years around redistribution year {redistribution_start_year}: {years_around}")
            
            # Ensure we have at least one year before and after if possible
            years_before = [y for y in global_years if y < redistribution_start_year]
            years_after = [y for y in global_years if y > redistribution_start_year]
            
            if not years_before and redistribution_start_year > min(global_years):
                logger.warning(f"No years found before redistribution year {redistribution_start_year}")
            if not years_after and redistribution_start_year < max(global_years):
                logger.warning(f"No years found after redistribution year {redistribution_start_year}")
    
    def _find_year_gaps(self, years: List[int]) -> List[int]:
        """
        Find gaps in a list of years
        
        Args:
            years: List of years
            
        Returns:
            List of missing years
        """
        if not years:
            return []
        
        min_year = min(years)
        max_year = max(years)
        expected_years = set(range(min_year, max_year + 1))
        actual_years = set(years)
        
        return sorted(list(expected_years - actual_years))
    
    def _fill_year_gaps(self, years: List[int]) -> List[int]:
        """
        Fill gaps in years list by adding missing years
        
        Args:
            years: List of years with potential gaps
            
        Returns:
            Complete list of years without gaps
        """
        if not years:
            return years
            
        min_year = min(years)
        max_year = max(years)
        return list(range(min_year, max_year + 1))
    
    def _interpolate_missing_year(self, year: int, global_values: Dict[int, float], 
                                  redistribution_start_year: Optional[int] = None) -> Optional[float]:
        """
        Interpolate value for a missing year with enhanced logic
        
        Args:
            year: The missing year
            global_values: Dictionary of year to value mappings
            redistribution_start_year: Optional redistribution start year for context
            
        Returns:
            Interpolated value or None if cannot interpolate
        """
        if not global_values:
            return None
            
        # ENHANCEMENT: Increase flexibility for finding adjacent years
        max_gap = 5  # Allow up to 5 years gap for interpolation
        all_years = sorted(global_values.keys())
        
        # Find the closest years before and after
        years_before = [y for y in all_years if y < year]
        years_after = [y for y in all_years if y > year]
        
        # If we can't find any years within max_gap, use all available years
        adjacent_years = sorted([y for y in global_values.keys() if abs(y - year) <= max_gap])
        
        if not adjacent_years and (years_before or years_after):
            logger.warning(f"No years within {max_gap} years of {year}, using all available years")
            adjacent_years = all_years
        
        # Get years before and after
        before_years = [y for y in adjacent_years if y < year]
        after_years = [y for y in adjacent_years if y > year]
        
        # If redistribution year is set and this is near it, be more careful
        if redistribution_start_year and abs(year - redistribution_start_year) <= 1:
            logger.info(f"Year {year} is near redistribution year {redistribution_start_year}, using careful interpolation")
            
        # Try linear interpolation if we have both before and after
        if before_years and after_years:
            before_year = max(before_years)
            after_year = min(after_years)
            before_value = global_values[before_year]
            after_value = global_values[after_year]
            
            # Linear interpolation
            weight = (year - before_year) / (after_year - before_year)
            interpolated_value = before_value + weight * (after_value - before_value)
            
            logger.info(f"Interpolated value for year {year}: {interpolated_value:.2f} "
                       f"(linear between {before_year}={before_value:.2f} and {after_year}={after_value:.2f})")
            return interpolated_value
            
        # If only before years, use trend extrapolation
        elif len(before_years) >= 2:
            # Get last two years for trend
            year1 = before_years[-2]
            year2 = before_years[-1]
            value1 = global_values[year1]
            value2 = global_values[year2]
            
            # Calculate trend
            trend = (value2 - value1) / (year2 - year1)
            extrapolated_value = value2 + trend * (year - year2)
            
            logger.info(f"Extrapolated value for year {year}: {extrapolated_value:.2f} "
                       f"(trend from {year1}-{year2})")
            return extrapolated_value
            
        # If only after years, use reverse trend
        elif len(after_years) >= 2:
            # Get first two years for trend
            year1 = after_years[0]
            year2 = after_years[1]
            value1 = global_values[year1]
            value2 = global_values[year2]
            
            # Calculate trend
            trend = (value2 - value1) / (year2 - year1)
            extrapolated_value = value1 - trend * (year1 - year)
            
            logger.info(f"Reverse extrapolated value for year {year}: {extrapolated_value:.2f} "
                       f"(trend from {year1}-{year2})")
            return extrapolated_value
            
        # Last resort: use nearest year
        elif adjacent_years:
            nearest_year = min(adjacent_years, key=lambda y: abs(y - year))
            value = global_values[nearest_year]
            logger.warning(f"Using nearest year {nearest_year} value for year {year}: {value:.2f}")
            return value
            
        return None 