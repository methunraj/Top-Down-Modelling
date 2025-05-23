"""
Streamlit Distribution Interface Module

This module provides components and utilities for configuring and executing
the market distribution process through the Streamlit interface.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.distribution.market_distributor import MarketDistributor
from src.config.config_manager import ConfigurationManager
from src.global_forecasting.base_forecaster import BaseForecaster
from src.global_forecasting import create_forecaster, get_available_forecasters

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def render_tier_configuration(market_distributor: MarketDistributor, config_manager: ConfigurationManager) -> Dict[str, Any]:
    """
    Render tier configuration interface.
    
    Args:
        market_distributor: MarketDistributor instance
        config_manager: ConfigurationManager instance
        
    Returns:
        Dictionary with updated tier configuration
    """
    st.subheader("Market Tier Configuration")
    st.markdown("""
    Market tiers group countries based on their market share.
    This affects how growth rates are calculated and constrained.
    """)
    
    # Get tier configuration from market_distributor
    distribution_settings = market_distributor.distribution_settings
    
    # Tier determination method
    tier_method = st.radio(
        "Tier Determination Method",
        options=["Automatic", "Manual"],
        index=0 if distribution_settings.get('tier_determination', 'auto') == 'auto' else 1,
        horizontal=True,
        help="How tiers are determined: 'Auto' uses clustering, 'Manual' uses thresholds"
    )
    
    # Convert to expected value
    tier_determination = 'auto' if tier_method == "Automatic" else 'manual'
    
    # Additional settings based on method
    if tier_determination == 'auto':
        # K-means parameters
        st.subheader("K-means Clustering Parameters")
        kmeans_params = distribution_settings.get('kmeans_params', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_clusters = st.slider(
                "Minimum Clusters",
                min_value=2,
                max_value=5,
                value=kmeans_params.get('min_clusters', 3),
                help="Minimum number of clusters to consider"
            )
        
        with col2:
            max_clusters = st.slider(
                "Maximum Clusters",
                min_value=3,
                max_value=10,
                value=kmeans_params.get('max_clusters', 8),
                help="Maximum number of clusters to consider"
            )
        
        # Update settings
        kmeans_params = {
            'min_clusters': min_clusters,
            'max_clusters': max_clusters,
            'random_state': 42,
            'n_init': 10
        }
        
        # Show explanation
        st.info("""
        The system will use silhouette scores to determine the optimal number of clusters within this range.
        Clusters will be sorted by market share to determine tiers, with Tier 1 having the highest share.
        """)
        
        # Return updated settings
        return {
            'tier_determination': tier_determination,
            'kmeans_params': kmeans_params
        }
    else:
        # Manual tier settings
        st.subheader("Manual Tier Thresholds")
        st.markdown("Define market share thresholds for each tier.")
        
        # Get manual tier settings
        manual_tiers = distribution_settings.get('manual_tiers', {})
        
        # Tier 1 (Market Leaders)
        tier1_threshold = st.slider(
            "Tier 1 Threshold (%)",
            min_value=1.0,
            max_value=30.0,
            value=float(manual_tiers.get('tier_1', {}).get('share_threshold', 10.0)),
            step=0.1,
            help="Countries with market share >= this percentage are in Tier 1"
        )
        
        # Tier 2 (Established Markets)
        tier2_threshold = st.slider(
            "Tier 2 Threshold (%)",
            min_value=0.1,
            max_value=10.0,
            value=float(manual_tiers.get('tier_2', {}).get('share_threshold', 1.0)),
            step=0.1,
            help="Countries with market share >= this percentage but < Tier 1 threshold are in Tier 2"
        )
        
        # Tier 3 (Emerging Markets)
        tier3_threshold = st.slider(
            "Tier 3 Threshold (%)",
            min_value=0.01,
            max_value=1.0,
            value=float(manual_tiers.get('tier_3', {}).get('share_threshold', 0.1)),
            step=0.01,
            help="Countries with market share >= this percentage but < Tier 2 threshold are in Tier 3"
        )
        
        # Create manual tiers dictionary
        manual_tiers = {
            'tier_1': {
                'share_threshold': tier1_threshold,
                'description': "Market Leaders"
            },
            'tier_2': {
                'share_threshold': tier2_threshold,
                'description': "Established Markets"
            },
            'tier_3': {
                'share_threshold': tier3_threshold,
                'description': "Emerging Markets"
            }
        }
        
        # Manual tier assignment
        if 'country_historical' in st.session_state and st.session_state.country_historical is not None:
            st.subheader("Manual Country Assignment")
            st.markdown("Optionally, override tier assignment for specific countries.")
            
            # Get country mapping
            try:
                column_mapping = config_manager.get_column_mapping('country_historical')
                id_col = column_mapping.get('id_column', 'idGeo')
                country_col = column_mapping.get('country_column', 'Country')
            except Exception:
                id_col = 'idGeo'
                country_col = 'Country'
            
            # Get countries from data
            df = st.session_state.country_historical
            countries = df[country_col].unique()
            
            # Let user assign countries to specific tiers
            country_tiers = {}
            
            with st.expander("Override Country Tiers"):
                # Group countries into sets of 5 for easier selection
                for i in range(0, len(countries), 5):
                    country_group = countries[i:i+5]
                    cols = st.columns(len(country_group))
                    
                    for idx, country in enumerate(country_group):
                        with cols[idx]:
                            tier = st.selectbox(
                                f"{country}",
                                options=["Auto", "Tier 1", "Tier 2", "Tier 3"],
                                index=0,
                                key=f"tier_{country}"
                            )
                            
                            # Only store non-auto assignments
                            if tier != "Auto":
                                # Convert to numeric tier (0-indexed)
                                tier_num = int(tier.split()[1]) - 1
                                country_tiers[country] = tier_num
            
            # Return updated settings
            return {
                'tier_determination': tier_determination,
                'manual_tiers': manual_tiers,
                'country_tiers': country_tiers
            }
        
        # Return updated settings without country assignments
        return {
            'tier_determination': tier_determination,
            'manual_tiers': manual_tiers
        }


def render_growth_constraints(market_distributor: MarketDistributor) -> Dict[str, Any]:
    """
    Render growth constraints interface.
    
    Args:
        market_distributor: MarketDistributor instance
        
    Returns:
        Dictionary with updated growth constraint configuration
    """
    st.subheader("Growth Constraints")
    st.markdown("""
    Growth constraints ensure realistic growth patterns in the forecast.
    Different market tiers can have different constraints.
    """)
    
    # Get growth constraint configuration
    distribution_settings = market_distributor.distribution_settings
    growth_constraints = distribution_settings.get('growth_constraints', {})
    
    # Determination method
    constraint_method = st.radio(
        "Constraint Determination",
        options=["Automatic", "Manual"],
        index=0 if growth_constraints.get('determination_method', 'auto') == 'auto' else 1,
        horizontal=True,
        help="How constraints are determined: 'Auto' calculates from historical data, 'Manual' uses defined values"
    )
    
    # Convert to expected value
    determination_method = 'auto' if constraint_method == "Automatic" else 'manual'
    
    # Additional settings based on method
    if determination_method == 'auto':
        # Auto parameters
        st.info("""
        Growth constraints will be calculated automatically from historical data.
        
        The system analyzes historical growth patterns and sets appropriate constraints based on:
        - Historical growth volatility
        - Market tier characteristics
        - Data quality and coverage
        """)
        
        # Constraint factor
        constraint_factor = st.slider(
            "Constraint Factor",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
            help="Adjust constraint strictness: values above 1.0 allow wider growth ranges, values below 1.0 enforce tighter constraints"
        )
        
        # Return updated settings
        return {
            'determination_method': determination_method,
            'constraint_factor': constraint_factor
        }
    else:
        # Manual constraints
        st.subheader("Manual Growth Constraints")
        
        # Get manual constraints
        manual_constraints = growth_constraints.get('manual_constraints', {})
        
        # Tier 1 (Market Leaders)
        st.markdown("#### Tier 1 (Market Leaders)")
        tier1_max = st.slider(
            "Maximum Growth Rate (%)",
            min_value=10.0,
            max_value=80.0,
            value=float(manual_constraints.get('tier1_max_growth', 35.0)),
            step=1.0,
            key="tier1_max",
            help="Maximum allowed year-over-year growth for Tier 1 countries"
        )
        
        tier1_min = st.slider(
            "Minimum Growth Rate (%)",
            min_value=-40.0,
            max_value=0.0,
            value=float(manual_constraints.get('tier1_min_growth', -15.0)),
            step=1.0,
            key="tier1_min",
            help="Minimum allowed year-over-year growth for Tier 1 countries"
        )
        
        # Tier 2 (Established Markets)
        st.markdown("#### Tier 2 (Established Markets)")
        tier2_max = st.slider(
            "Maximum Growth Rate (%)",
            min_value=15.0,
            max_value=100.0,
            value=float(manual_constraints.get('tier2_max_growth', 40.0)),
            step=1.0,
            key="tier2_max",
            help="Maximum allowed year-over-year growth for Tier 2 countries"
        )
        
        tier2_min = st.slider(
            "Minimum Growth Rate (%)",
            min_value=-50.0,
            max_value=0.0,
            value=float(manual_constraints.get('tier2_min_growth', -20.0)),
            step=1.0,
            key="tier2_min",
            help="Minimum allowed year-over-year growth for Tier 2 countries"
        )
        
        # Tier 3 (Emerging Markets)
        st.markdown("#### Tier 3 (Emerging Markets)")
        tier3_max = st.slider(
            "Maximum Growth Rate (%)",
            min_value=20.0,
            max_value=120.0,
            value=float(manual_constraints.get('tier3_max_growth', 45.0)),
            step=1.0,
            key="tier3_max",
            help="Maximum allowed year-over-year growth for Tier 3 countries"
        )
        
        tier3_min = st.slider(
            "Minimum Growth Rate (%)",
            min_value=-60.0,
            max_value=0.0,
            value=float(manual_constraints.get('tier3_min_growth', -25.0)),
            step=1.0,
            key="tier3_min",
            help="Minimum allowed year-over-year growth for Tier 3 countries"
        )
        
        # Apply scaling by market size
        apply_scaling = st.checkbox(
            "Apply Scaling by Market Size",
            value=manual_constraints.get('apply_scaling_by_market_size', True),
            help="Allow smaller markets to grow faster and decline slower than larger markets"
        )
        
        # Create manual constraints dictionary
        manual_constraints = {
            'tier1_max_growth': tier1_max,
            'tier1_min_growth': tier1_min,
            'tier2_max_growth': tier2_max,
            'tier2_min_growth': tier2_min,
            'tier3_max_growth': tier3_max,
            'tier3_min_growth': tier3_min,
            'apply_scaling_by_market_size': apply_scaling
        }
        
        # Return updated settings
        return {
            'determination_method': determination_method,
            'manual_constraints': manual_constraints
        }


def render_indicator_configuration(market_distributor: MarketDistributor) -> Dict[str, Any]:
    """
    Render indicator configuration interface.
    
    Args:
        market_distributor: MarketDistributor instance
        
    Returns:
        Dictionary with updated indicator configuration
    """
    st.subheader("Indicator Configuration")
    
    # Check if indicators are available
    if not hasattr(st.session_state, 'indicators') or not st.session_state.indicators:
        st.warning("No indicators added. Indicators can enhance distribution accuracy.")
        
        if st.button("Add Indicators"):
            st.session_state.active_page = "Data Input"
            st.rerun()
        
        return {}
    
    # Get configuration settings
    distribution_settings = market_distributor.distribution_settings
    
    # Weight transformation parameters
    st.markdown("#### Weight Transformation")
    transformation = st.selectbox(
        "Transformation Method",
        options=["log", "squared", "sigmoid", "linear"],
        index=0,
        help="Method for transforming indicator values"
    )
    
    # Significance method
    significance_method = st.selectbox(
        "Significance Method",
        options=["continuous", "stepped"],
        index=0,
        help="Method for applying significance: 'continuous' uses smooth scaling, 'stepped' uses discrete thresholds"
    )
    
    # Individual indicator weights
    st.markdown("#### Indicator Weights")
    indicator_weights = {}
    
    # Safely iterate over indicators if they exist
    if hasattr(st.session_state, 'indicators') and st.session_state.indicators:
        for name, details in st.session_state.indicators.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{name}** ({details['meta']['type']})")
            
            with col2:
                if details['meta']['weight'] == "auto":
                    st.text("Auto")
                    indicator_weights[name] = "auto"
                else:
                    new_weight = st.number_input(
                        f"Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(details['meta']['weight']),
                        step=0.01,
                        key=f"weight_{name}"
                    )
                    indicator_weights[name] = new_weight
    else:
        st.info("No indicators configured. Add indicators in the Data Input section for enhanced distribution accuracy.")
    
    # Return updated settings
    return {
        'transformation': transformation,
        'significance_method': significance_method,
        'indicator_weights': indicator_weights
    }


def render_smoothing_configuration(market_distributor: MarketDistributor) -> Dict[str, Any]:
    """
    Render smoothing configuration interface.
    
    Args:
        market_distributor: MarketDistributor instance
        
    Returns:
        Dictionary with updated smoothing configuration
    """
    st.subheader("Smoothing Parameters")
    st.markdown("""
    Smoothing ensures realistic growth patterns by reducing volatility in the forecast.
    Different market tiers can have different smoothing parameters.
    """)
    
    # Get smoothing configuration
    distribution_settings = market_distributor.distribution_settings
    smoothing = distribution_settings.get('smoothing', {})
    
    # Enable smoothing
    enable_smoothing = st.checkbox(
        "Enable Smoothing",
        value=smoothing.get('enabled', True),
        help="Apply smoothing to ensure realistic growth patterns"
    )
    
    if not enable_smoothing:
        return {'enabled': False}
    
    # Tier-specific smoothing parameters
    st.markdown("#### Tier-Specific Smoothing Parameters")
    
    # Get tier smoothing settings
    tier_smoothing = smoothing.get('tier_smoothing', {})
    
    # Tier 1 (Market Leaders)
    st.markdown("##### Tier 1 (Market Leaders)")
    
    tier1_smoothing = tier_smoothing.get('tier_1', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tier1_window = st.slider(
            "Window Size",
            min_value=1,
            max_value=7,
            value=tier1_smoothing.get('window', 3),
            key="smooth_t1_window",
            help="Size of the rolling window for initial smoothing (larger = more smoothing)"
        )
    
    with col2:
        tier1_max = st.slider(
            "Maximum Growth (%)",
            min_value=10.0,
            max_value=50.0,
            value=float(tier1_smoothing.get('max_growth', 35.0)),
            step=1.0,
            key="smooth_t1_max",
            help="Maximum growth rate after smoothing"
        )
    
    with col3:
        tier1_min = st.slider(
            "Minimum Growth (%)",
            min_value=-30.0,
            max_value=0.0,
            value=float(tier1_smoothing.get('min_growth', -15.0)),
            step=1.0,
            key="smooth_t1_min",
            help="Minimum growth rate after smoothing"
        )
    
    tier1_target = st.slider(
        "Target Growth Rate (%)",
        min_value=5.0,
        max_value=25.0,
        value=float(tier1_smoothing.get('target_growth', 15.0)),
        step=0.5,
        key="smooth_t1_target",
        help="Long-term target growth rate for convergence"
    )
    
    # Tier 2 (Established Markets)
    st.markdown("##### Tier 2 (Established Markets)")
    
    tier2_smoothing = tier_smoothing.get('tier_2', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tier2_window = st.slider(
            "Window Size",
            min_value=1,
            max_value=7,
            value=tier2_smoothing.get('window', 3),
            key="smooth_t2_window",
            help="Size of the rolling window for initial smoothing (larger = more smoothing)"
        )
    
    with col2:
        tier2_max = st.slider(
            "Maximum Growth (%)",
            min_value=10.0,
            max_value=60.0,
            value=float(tier2_smoothing.get('max_growth', 40.0)),
            step=1.0,
            key="smooth_t2_max",
            help="Maximum growth rate after smoothing"
        )
    
    with col3:
        tier2_min = st.slider(
            "Minimum Growth (%)",
            min_value=-35.0,
            max_value=0.0,
            value=float(tier2_smoothing.get('min_growth', -20.0)),
            step=1.0,
            key="smooth_t2_min",
            help="Minimum growth rate after smoothing"
        )
    
    tier2_target = st.slider(
        "Target Growth Rate (%)",
        min_value=10.0,
        max_value=30.0,
        value=float(tier2_smoothing.get('target_growth', 20.0)),
        step=0.5,
        key="smooth_t2_target",
        help="Long-term target growth rate for convergence"
    )
    
    # Tier 3 (Emerging Markets)
    st.markdown("##### Tier 3 (Emerging Markets)")
    
    tier3_smoothing = tier_smoothing.get('tier_3', {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tier3_window = st.slider(
            "Window Size",
            min_value=1,
            max_value=9,
            value=tier3_smoothing.get('window', 5),
            key="smooth_t3_window",
            help="Size of the rolling window for initial smoothing (larger = more smoothing)"
        )
    
    with col2:
        tier3_max = st.slider(
            "Maximum Growth (%)",
            min_value=15.0,
            max_value=70.0,
            value=float(tier3_smoothing.get('max_growth', 45.0)),
            step=1.0,
            key="smooth_t3_max",
            help="Maximum growth rate after smoothing"
        )
    
    with col3:
        tier3_min = st.slider(
            "Minimum Growth (%)",
            min_value=-40.0,
            max_value=0.0,
            value=float(tier3_smoothing.get('min_growth', -25.0)),
            step=1.0,
            key="smooth_t3_min",
            help="Minimum growth rate after smoothing"
        )
    
    tier3_target = st.slider(
        "Target Growth Rate (%)",
        min_value=15.0,
        max_value=35.0,
        value=float(tier3_smoothing.get('target_growth', 25.0)),
        step=0.5,
        key="smooth_t3_target",
        help="Long-term target growth rate for convergence"
    )
    
    # Convergence rate
    convergence_rate = st.slider(
        "Convergence Rate",
        min_value=0.1,
        max_value=0.5,
        value=float(smoothing.get('convergence_rate', 0.25)),
        step=0.01,
        help="How quickly growth rates converge to target values (higher = faster convergence)"
    )
    
    # Create updated smoothing configuration
    updated_smoothing = {
        'enabled': True,
        'convergence_rate': convergence_rate,
        'tier_smoothing': {
            'tier_1': {
                'window': tier1_window,
                'min_periods': 1,
                'max_growth': tier1_max,
                'min_growth': tier1_min,
                'target_growth': tier1_target
            },
            'tier_2': {
                'window': tier2_window,
                'min_periods': 1,
                'max_growth': tier2_max,
                'min_growth': tier2_min,
                'target_growth': tier2_target
            },
            'tier_3': {
                'window': tier3_window,
                'min_periods': 1,
                'max_growth': tier3_max,
                'min_growth': tier3_min,
                'target_growth': tier3_target
            }
        }
    }
    
    return updated_smoothing


def render_redistribution_settings(market_distributor: MarketDistributor) -> Dict[str, Any]:
    """
    Render redistribution settings interface.
    
    Args:
        market_distributor: MarketDistributor instance
        
    Returns:
        Dictionary with updated redistribution settings
    """
    st.subheader("Redistribution Settings")
    
    # Get redistribution settings
    distribution_settings = market_distributor.distribution_settings
    redistribution_start_year = distribution_settings.get('redistribution_start_year', None)
    
    # Enable redistribution from specific year
    enable_redistribution = st.checkbox(
        "Enable Redistribution from Specific Year",
        value=redistribution_start_year is not None,
        help="Preserve historical data exactly as-is before a specific year"
    )
    
    if enable_redistribution:
        # Get available years from data
        available_years = []
        
        if 'country_historical' in st.session_state and st.session_state.country_historical is not None:
            df = st.session_state.country_historical
            if 'Year' in df.columns:
                available_years = sorted(df['Year'].unique())
        
        if available_years:
            # Choose redistribution start year from available years
            redistribution_year = st.selectbox(
                "Redistribution Start Year",
                options=available_years,
                index=len(available_years) // 2 if redistribution_start_year is None else available_years.index(redistribution_start_year) if redistribution_start_year in available_years else len(available_years) // 2,
                help="Historical data before this year will be preserved exactly as-is"
            )
        else:
            # Manual input if years not available
            redistribution_year = st.number_input(
                "Redistribution Start Year",
                min_value=2010,
                max_value=2025,
                value=int(redistribution_start_year) if redistribution_start_year is not None else 2020,
                help="Historical data before this year will be preserved exactly as-is"
            )
        
        st.info(f"Historical data before {redistribution_year} will be preserved exactly as-is.")
        
        # Return updated settings
        return {'redistribution_start_year': redistribution_year}
    else:
        # Return empty dict to indicate no redistribution
        return {}


def render_global_forecast_interface() -> Dict[str, Any]:
    """
    Render global forecast interface.
    
    Returns:
        Dictionary with global forecast configuration
    """
    st.header("Global Market Forecasting")
    
    # Check if we have historical global data
    if 'global_forecast' not in st.session_state or st.session_state.global_forecast is None:
        st.warning("No global market data available. Please upload data first.")
        
        if st.button("Go to Data Input"):
            st.session_state.active_page = "Data Input"
            st.rerun()
        
        return {}
    
    # Get the data
    global_df = st.session_state.global_forecast
    
    # Identify columns
    # In a real implementation, get these from config
    year_col = 'Year'
    value_col = 'Value'
    
    # Define type_col for compatibility with downstream code
    type_col = '_forecast_type'  # This is our internal marker column
    
    # Detect if there's existing forecast data
    has_existing_forecast = False
    forecast_data = pd.DataFrame()
    
    # Sort the data by year to identify last historical year
    sorted_df = global_df.sort_values(by=year_col)
    years = sorted_df[year_col].unique()
    
    # Determine the forecast horizon year if specified by the user
    forecast_horizon_year = None
    if 'forecast_horizon_year' in st.session_state:
        forecast_horizon_year = st.session_state.forecast_horizon_year
    
    # If a forecast horizon is set, split the data
    if forecast_horizon_year is not None:
        historical_data = sorted_df[sorted_df[year_col] < forecast_horizon_year].copy()
        forecast_data = sorted_df[sorted_df[year_col] >= forecast_horizon_year].copy()
        has_existing_forecast = not forecast_data.empty
        
        # Check if we have any historical data
        if historical_data.empty:
            st.error(f"No historical data found before year {forecast_horizon_year}. Please adjust the forecast horizon year to include some historical data.")
            st.info(f"Available years in data: {sorted(sorted_df[year_col].unique())}")
    else:
        # Treat all data as historical if no horizon is set
        historical_data = sorted_df.copy()
    
    # Allow user to set the forecast horizon
    if forecast_horizon_year is None:
        # Get min and max year
        min_year = int(years.min()) if years.size > 0 else 2000
        max_year = int(years.max()) if years.size > 0 else 2030
        default_horizon = min(max_year + 1, 2030)  # Default to next year after last data point
        
        st.markdown("#### Set Historical Data Range")
        forecast_horizon_year = st.slider(
            "Select the first year of forecast period:",
            min_value=min_year + 1,
            max_value=max_year + 20,
            value=default_horizon,
            help="Data before this year will be treated as historical, data from this year onward will be treated as forecast"
        )
        st.session_state.forecast_horizon_year = forecast_horizon_year
        
        # Re-split the data based on the selected horizon
        historical_data = sorted_df[sorted_df[year_col] < forecast_horizon_year].copy()
        forecast_data = sorted_df[sorted_df[year_col] >= forecast_horizon_year].copy()
        has_existing_forecast = not forecast_data.empty
        
        # Check if we have any historical data after re-split
        if historical_data.empty:
            st.error(f"No historical data found before year {forecast_horizon_year}. Please adjust the forecast horizon year to include some historical data.")
            st.info(f"Available years in data: {sorted(sorted_df[year_col].unique())}")
    
    # Display information about the data split
    st.info(f"Data up to {forecast_horizon_year-1} will be treated as historical. Forecasting will start from {forecast_horizon_year}.")
    
    # If we have existing forecast data after the split
    if has_existing_forecast:
        st.info(f"Existing forecast data found for {len(forecast_data)} years from {forecast_horizon_year}.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_existing = st.radio(
                "Use existing forecast?",
                options=["Yes", "No"],
                index=0,
                    horizontal=True
                )
            
            if use_existing == "Yes":
                # Display existing forecast
                st.subheader("Existing Global Forecast")
                
                # Plot the data
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=historical_data[year_col],
                    y=historical_data[value_col],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8, color='blue')
                ))
                
                # Add forecast data
                fig.add_trace(go.Scatter(
                    x=forecast_data[year_col],
                    y=forecast_data[value_col],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=8, color='red')
                ))
                
                # Update layout
                fig.update_layout(
                    title='Global Market Forecast',
                    xaxis_title='Year',
                    yaxis_title='Market Value',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Return original data
                return {'use_existing': True}
    # This section has been replaced by the more intuitive approach above
    
    # If we don't have forecast data or user chose not to use existing
    st.subheader("Generate New Global Forecast")
    
    # Get available forecasting methods
    forecaster_categories = get_available_forecasters()
    
    # Method selection
    st.markdown("#### Select Forecasting Method")
    
    # Select a category
    category = st.selectbox(
        "Method Category",
        options=list(forecaster_categories.keys()),
        index=0
    )
    
    # Select a method from the category
    method = st.selectbox(
        "Forecasting Method",
        options=forecaster_categories[category],
        index=0
    )
    
    # Forecast horizon
    st.markdown("#### Forecast Horizon")
    
    # Get the latest historical year
    if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
        # Always use historical_data, which by this point contains the right subset of data 
        # (either filtered by Type or all data)
        latest_year = historical_data[year_col].max()
    else:
        latest_year = 2023  # Default
    
    # Calculate a reasonable forecast horizon
    suggested_horizon = 5
    
    # Forecast horizon selection
    horizon = st.slider(
        "Years to Forecast",
        min_value=1,
        max_value=20,
        value=suggested_horizon,
        help="Number of years to forecast into the future"
    )
    
    # Method-specific parameters
    st.markdown("#### Method Parameters")
    
    # Add different parameters based on the selected method
    method_params = {}
    
    if method == "CAGR":
        # CAGR-specific parameters
        cagr_value = st.slider(
            "CAGR Value (%)",
            min_value=-20.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="Compound Annual Growth Rate to apply"
        )
        
        method_params = {
            'fixed_cagr': cagr_value
        }
    
    elif method == "Moving Average":
        # Moving Average parameters
        window_size = st.slider(
            "Window Size",
            min_value=2,
            max_value=10,
            value=3,
            help="Number of periods to include in the moving average"
        )
        
        weighted = st.checkbox(
            "Use Weighted Moving Average",
            value=False,
            help="Give more weight to recent periods"
        )
        
        method_params = {
            'window_size': window_size,
            'weighted': weighted
        }
    
    elif method == "Exponential Smoothing":
        # Exponential Smoothing parameters
        smoothing_type = st.radio(
            "Smoothing Type",
            options=["simple", "double", "triple"],
            index=1,
            horizontal=True,
            help="Simple: level only, Double: level and trend, Triple: level, trend, and seasonality"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha = st.slider(
                "Alpha (Level)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help="Smoothing factor for level (0=no learning, 1=no smoothing)"
            )
        
        with col2:
            beta = st.slider(
                "Beta (Trend)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Smoothing factor for trend"
            ) if smoothing_type in ["double", "triple"] else None
        
        with col3:
            gamma = st.slider(
                "Gamma (Seasonal)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Smoothing factor for seasonality"
            ) if smoothing_type == "triple" else None
        
        method_params = {
            'smoothing_type': smoothing_type,
            'alpha': alpha
        }
        
        if beta is not None:
            method_params['beta'] = beta
        
        if gamma is not None:
            method_params['gamma'] = gamma
            
            # For triple exponential smoothing, need seasonal periods
            method_params['seasonal_periods'] = st.slider(
                "Seasonal Periods",
                min_value=2,
                max_value=12,
                value=4,
                help="Number of periods in a seasonal cycle"
            )
    
    elif method == "ARIMA":
        # ARIMA parameters
        st.markdown("#### ARIMA Parameters (p, d, q)")
        st.markdown("p = AR order, d = differencing, q = MA order")
        
        auto_order = st.checkbox(
            "Auto-determine optimal parameters",
            value=True,
            help="Automatically find the best ARIMA order using statistical methods"
        )
        
        if auto_order:
            # Information criterion
            criterion = st.radio(
                "Information Criterion",
                options=["aic", "bic"],
                index=0,
                horizontal=True,
                help="AIC tends to select more complex models, BIC tends to select simpler models"
            )
            
            # Maximum values to consider
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_p = st.slider(
                    "Max p",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Maximum AR order to consider"
                )
            
            with col2:
                max_d = st.slider(
                    "Max d",
                    min_value=0,
                    max_value=2,
                    value=2,
                    help="Maximum differencing order to consider"
                )
            
            with col3:
                max_q = st.slider(
                    "Max q",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help="Maximum MA order to consider"
                )
            
            method_params = {
                'auto_order': True,
                'criterion': criterion,
                'max_p': max_p,
                'max_d': max_d,
                'max_q': max_q
            }
        else:
            # Manual order selection
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p = st.slider(
                    "p (AR order)",
                    min_value=0,
                    max_value=10,
                    value=1,
                    help="Number of lag observations"
                )
            
            with col2:
                d = st.slider(
                    "d (Differencing)",
                    min_value=0,
                    max_value=2,
                    value=1,
                    help="Number of times to difference data"
                )
            
            with col3:
                q = st.slider(
                    "q (MA order)",
                    min_value=0,
                    max_value=10,
                    value=1,
                    help="Size of moving average window"
                )
            
            method_params = {
                'auto_order': False,
                'p': p,
                'd': d,
                'q': q
            }
    
    elif method == "SARIMA":
        # SARIMA parameters
        st.markdown("#### SARIMA Parameters (p, d, q)(P, D, Q, s)")
        st.markdown("p, d, q = non-seasonal, P, D, Q = seasonal, s = periods per season")
        
        auto_order = st.checkbox(
            "Auto-determine optimal parameters",
            value=True,
            help="Automatically find the best SARIMA order using statistical methods"
        )
        
        # Seasonal period
        seasonal_period = st.selectbox(
            "Seasonal Period",
            options=[1, 4, 12],
            index=1,
            help="Periods per season (1=none, 4=quarterly, 12=monthly)"
        )
        
        if auto_order:
            # Information criterion
            criterion = st.radio(
                "Information Criterion",
                options=["aic", "bic"],
                index=0,
                horizontal=True,
                help="AIC tends to select more complex models, BIC tends to select simpler models"
            )
            
            method_params = {
                'auto_order': True,
                'criterion': criterion,
                's': seasonal_period
            }
        else:
            # Manual order selection
            st.markdown("##### Non-seasonal parameters (p, d, q)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p = st.slider(
                    "p (AR order)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Non-seasonal AR order"
                )
            
            with col2:
                d = st.slider(
                    "d (Differencing)",
                    min_value=0,
                    max_value=2,
                    value=1,
                    help="Non-seasonal differencing"
                )
            
            with col3:
                q = st.slider(
                    "q (MA order)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Non-seasonal MA order"
                )
            
            st.markdown("##### Seasonal parameters (P, D, Q)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                P = st.slider(
                    "P (Seasonal AR)",
                    min_value=0,
                    max_value=3,
                    value=1,
                    help="Seasonal AR order"
                )
            
            with col2:
                D = st.slider(
                    "D (Seasonal diff)",
                    min_value=0,
                    max_value=1,
                    value=0,
                    help="Seasonal differencing"
                )
            
            with col3:
                Q = st.slider(
                    "Q (Seasonal MA)",
                    min_value=0,
                    max_value=3,
                    value=1,
                    help="Seasonal MA order"
                )
            
            method_params = {
                'auto_order': False,
                'p': p, 'd': d, 'q': q,
                'P': P, 'D': D, 'Q': Q, 's': seasonal_period
            }
    
    elif method == "Regression":
        # Regression parameters
        model_type = st.selectbox(
            "Regression Model Type",
            options=["linear", "polynomial", "ridge", "lasso", "elasticnet", "randomforest"],
            index=0,
            help="Type of regression model to use for forecasting"
        )
        
        method_params = {
            'model_type': model_type,
            'include_trend': True,
            'include_seasonal': True
        }
        
        # Additional parameters based on model type
        if model_type == "polynomial":
            method_params['poly_degree'] = st.slider(
                "Polynomial Degree",
                min_value=2,
                max_value=5,
                value=2,
                help="Degree of polynomial features (higher = more flexible but can overfit)"
            )
        
        elif model_type in ["ridge", "lasso", "elasticnet"]:
            method_params['alpha'] = st.slider(
                "Alpha (Regularization Strength)",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                help="Regularization strength (higher = more regularization)"
            )
            
            if model_type == "elasticnet":
                method_params['l1_ratio'] = st.slider(
                    "L1 Ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Mix between L1 and L2 (0=Ridge, 1=Lasso)"
                )
        
        elif model_type == "randomforest":
            col1, col2 = st.columns(2)
            
            with col1:
                method_params['n_estimators'] = st.slider(
                    "Number of Trees",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Number of trees in the forest"
                )
            
            with col2:
                max_depth = st.slider(
                    "Max Tree Depth",
                    min_value=3,
                    max_value=20,
                    value=10,
                    help="Maximum depth of trees (None=unlimited)"
                )
                if max_depth < 20:
                    method_params['max_depth'] = max_depth
        
        # Feature engineering options
        st.markdown("#### Feature Engineering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_lag = st.checkbox(
                "Include Lag Features",
                value=False,
                help="Use previous values as features (autoregressive)"
            )
        
        if include_lag:
            method_params['include_lag'] = True
            method_params['lag_periods'] = [1, 2, 3]  # Default lags
    
    elif method == "Prophet":
        # Prophet parameters
        st.markdown("#### Prophet Parameters")
        
        # Growth type
        growth = st.radio(
            "Growth Type",
            options=["linear", "logistic"],
            index=0,
            horizontal=True,
            help="Linear growth allows unbounded growth, logistic growth has a capacity"
        )
        
        method_params = {
            'growth': growth
        }
        
        # Capacity for logistic growth
        if growth == "logistic":
            # Estimate max capacity as 2-3x the maximum historical value
            max_value = 0
            if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                if 'Value' in st.session_state.global_forecast.columns:
                    max_value = st.session_state.global_forecast['Value'].max() * 3
            
            cap = st.number_input(
                "Capacity (Upper Limit)",
                min_value=0,
                value=int(max_value) if max_value > 0 else 10000,
                help="Maximum possible value for forecast (required for logistic growth)"
            )
            method_params['cap'] = cap
        
        # Seasonality mode
        seasonality_mode = st.radio(
            "Seasonality Mode",
            options=["additive", "multiplicative"],
            index=0,
            horizontal=True,
            help="Additive seasonality is more appropriate when seasonal fluctuations are roughly constant; multiplicative when they increase with the trend"
        )
        method_params['seasonality_mode'] = seasonality_mode
        
        # Advanced parameters in expander
        with st.expander("Advanced Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                method_params['yearly_seasonality'] = st.selectbox(
                    "Yearly Seasonality",
                    options=["auto", True, False],
                    index=0,
                    help="Whether to include yearly seasonality"
                )
            
            with col2:
                method_params['changepoint_prior_scale'] = st.slider(
                    "Changepoint Prior Scale",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.05,
                    step=0.001,
                    format="%.3f",
                    help="Controls flexibility of the trend, larger values allow more flexibility"
                )
        
    elif method == "XGBoost":
        # XGBoost parameters
        st.markdown("#### XGBoost Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider(
                "Number of Trees",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of trees in the ensemble"
            )
        
        with col2:
            max_depth = st.slider(
                "Max Tree Depth",
                min_value=1,
                max_value=15,
                value=3,
                help="Maximum depth of trees"
            )
        
        method_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }
        
        # Feature engineering options
        st.markdown("#### Feature Engineering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_date_features = st.checkbox(
                "Include Date Features",
                value=True,
                help="Create features from date (year, month, quarter, etc.)"
            )
            method_params['include_date_features'] = include_date_features
        
        with col2:
            lag_size = st.slider(
                "Lag Features",
                min_value=1,
                max_value=12,
                value=3,
                help="Number of previous values to use as features"
            )
            method_params['lag_size'] = lag_size
        
        # Advanced parameters in expander
        with st.expander("Advanced Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                    help="Step size shrinkage to prevent overfitting"
                )
                method_params['learning_rate'] = learning_rate
            
            with col2:
                subsample = st.slider(
                    "Subsample Ratio",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    help="Fraction of samples used for training trees"
                )
                method_params['subsample'] = subsample
    
    elif method == "LSTM":
        # LSTM parameters
        st.markdown("#### LSTM Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_length = st.slider(
                "Sequence Length",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of time steps to use as input sequence"
            )
        
        with col2:
            units_str = st.text_input(
                "LSTM Units (comma-separated)",
                value="50,25",
                help="Number of units in each LSTM layer, comma-separated"
            )
            units = [int(u.strip()) for u in units_str.split(",")]
        
        method_params = {
            'sequence_length': sequence_length,
            'units': units
        }
        
        # Training parameters
        st.markdown("#### Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider(
                "Max Epochs",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Maximum number of training epochs"
            )
            method_params['epochs'] = epochs
        
        with col2:
            patience = st.slider(
                "Early Stopping Patience",
                min_value=3,
                max_value=50,
                value=10,
                help="Number of epochs without improvement before stopping"
            )
            method_params['patience'] = patience
        
        # Feature engineering
        include_date_features = st.checkbox(
            "Include Date Features",
            value=True,
            help="Create features from date components"
        )
        method_params['include_date_features'] = include_date_features
    
    elif method == "Bass Diffusion":
        # Bass Diffusion parameters
        col1, col2 = st.columns(2)
        
        with col1:
            innovation = st.slider(
                "Innovation Coefficient (p)",
                min_value=0.001,
                max_value=0.1,
                value=0.03,
                step=0.001,
                help="Coefficient of innovation (external influence)"
            )
        
        with col2:
            imitation = st.slider(
                "Imitation Coefficient (q)",
                min_value=0.1,
                max_value=0.9,
                value=0.38,
                step=0.01,
                help="Coefficient of imitation (internal influence)"
            )
        
        # Market potential
        market_estimate = 0
        if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
            if not historical_data.empty:
                market_estimate = historical_data[value_col].max() * 3
            else:
                market_estimate = global_df[value_col].max() * 3
        
        market_potential = st.number_input(
            "Market Potential",
            min_value=0,
            value=int(market_estimate) if market_estimate > 0 else 10000,
            help="Maximum market size that will eventually be reached"
        )
        
        method_params = {
            'innovation': innovation,
            'imitation': imitation,
            'market_potential': market_potential
        }
    
    elif method == "Gompertz Curve":
        # Gompertz Curve parameters
        col1, col2 = st.columns(2)
        
        with col1:
            displacement = st.slider(
                "Displacement (b)",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Displacement parameter (higher = more initial lag)"
            )
        
        with col2:
            growth_rate = st.slider(
                "Growth Rate (c)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="Growth rate parameter (higher = faster growth)"
            )
        
        # Asymptote (market potential)
        market_estimate = 0
        if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
            if not historical_data.empty:
                market_estimate = historical_data[value_col].max() * 3
            else:
                market_estimate = global_df[value_col].max() * 3
        
        asymptote = st.number_input(
            "Asymptote (Market Potential)",
            min_value=0,
            value=int(market_estimate) if market_estimate > 0 else 10000,
            help="Maximum market size that will eventually be reached"
        )
        
        method_params = {
            'displacement': displacement,
            'growth_rate': growth_rate,
            'asymptote': asymptote
        }
    
    elif method == "Technology S-Curve":
        # Technology S-Curve parameters
        n_phases = st.slider(
            "Number of Adoption Phases",
            min_value=1,
            max_value=3,
            value=2,
            help="Number of distinct adoption phases in the technology lifecycle"
        )
        
        auto_params = st.checkbox(
            "Auto-estimate phase parameters",
            value=True,
            help="Automatically estimate parameters for each phase"
        )
        
        # If manual parameter estimation
        if not auto_params:
            st.warning("Manual parameter estimation is complex. Simplified controls provided.")
            
            # Market potentials for each phase
            phase_potentials = {}
            
            # Get estimated total market size
            market_estimate = 0
            if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                # Check if we have historical data:
                if not historical_data.empty:
                    market_estimate = historical_data[value_col].max() * 3
                else:
                    market_estimate = global_df[value_col].max() * 3
            
            # Total market potential
            total_potential = st.number_input(
                "Total Market Potential",
                min_value=0,
                value=int(market_estimate) if market_estimate > 0 else 10000,
                help="Maximum total market size across all phases"
            )
            
            # For each phase, specify the percentage
            for i in range(n_phases):
                phase_pct = st.slider(
                    f"Phase {i+1} Percentage",
                    min_value=1,
                    max_value=100,
                    value=int(100 / n_phases),
                    help=f"Percentage of total market for phase {i+1}"
                )
                
                phase_potentials[i] = total_potential * phase_pct / 100
        
        method_params = {
            'n_phases': n_phases,
            'auto_params': auto_params
        }
        
        if not auto_params:
            method_params['market_potentials'] = list(phase_potentials.values())
            
    elif method == "Fisher-Pry":
        # Fisher-Pry parameters
        st.markdown("#### Fisher-Pry Technology Substitution Model Parameters")
        st.markdown("This model forecasts technology substitution using an S-curve. It is particularly useful for forecasting replacement of an old technology by a new one.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alpha = st.slider(
                "Substitution Rate (alpha)",
                min_value=0.05,
                max_value=1.0,
                value=0.3,
                step=0.01,
                help="Rate of technology substitution (higher = faster adoption)"
            )
        
        with col2:
            auto_t0 = st.checkbox(
                "Auto-estimate t0 (time to 50% adoption)",
                value=True,
                help="Automatically estimate when technology reaches 50% adoption"
            )
            
            if not auto_t0:
                # Get the latest historical year
                if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                    latest_year = historical_data[year_col].max()
                    t0 = st.slider(
                        "Time to 50% Adoption (t0)",
                        min_value=int(latest_year),
                        max_value=int(latest_year) + 20,
                        value=int(latest_year) + 5,
                        help="Year when new technology reaches 50% market share"
                    )
                else:
                    t0 = st.number_input(
                        "Time to 50% Adoption (t0)",
                        min_value=2020,
                        max_value=2050,
                        value=2030,
                        help="Year when new technology reaches 50% market share"
                    )
        
        # Market saturation level
        fit_saturation = st.checkbox(
            "Fit Saturation Level",
            value=False,
            help="Automatically determine maximum market potential"
        )
        
        if not fit_saturation:
            # Estimate market saturation as 2x the maximum historical value
            market_estimate = 0
            if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                # Check if we have historical data:
                if not historical_data.empty:
                    market_estimate = historical_data[value_col].max() * 2
                else:
                    market_estimate = global_df[value_col].max() * 2
            
            saturation = st.number_input(
                "Market Saturation Level",
                min_value=0,
                value=int(market_estimate) if market_estimate > 0 else 10000,
                help="Maximum market size that will eventually be reached"
            )
        
        # Create parameters dictionary
        method_params = {
            'alpha': alpha,
            'fit_saturation': fit_saturation
        }
        
        if not auto_t0:
            method_params['t0'] = t0
            
        if not fit_saturation:
            method_params['saturation'] = saturation
            
    elif method == "Harvey Logistic":
        # Harvey Logistic parameters
        st.markdown("#### Harvey Logistic Model Parameters")
        st.markdown("The Harvey Logistic model is a modified logistic curve used for technology forecasting with variable market saturation.")
        
        # Fit saturation automatically or specify manually
        fit_saturation = st.checkbox(
            "Fit Saturation Level",
            value=True,
            help="Automatically determine maximum market potential"
        )
        
        if not fit_saturation:
            # Estimate saturation as 3x the maximum historical value
            market_estimate = 0
            if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                # Check if we have historical data:
                if not historical_data.empty:
                    market_estimate = historical_data[value_col].max() * 3
                else:
                    market_estimate = global_df[value_col].max() * 3
            
            saturation = st.number_input(
                "Market Saturation Level",
                min_value=0,
                value=int(market_estimate) if market_estimate > 0 else 10000,
                help="Maximum market size that will eventually be reached"
            )
        
        # Growth parameters
        col1, col2 = st.columns(2)
        
        with col1:
            beta_auto = st.checkbox(
                "Auto-estimate Initial Level (beta)",
                value=True,
                help="Automatically estimate beta parameter relating to initial market level"
            )
            
            if not beta_auto:
                beta = st.slider(
                    "Initial Level Parameter (beta)",
                    min_value=0.1,
                    max_value=100.0,
                    value=10.0,
                    help="Parameter related to initial adoption level (higher = lower initial adoption)"
                )
        
        with col2:
            gamma_auto = st.checkbox(
                "Auto-estimate Growth Rate (gamma)",
                value=True,
                help="Automatically estimate gamma parameter (growth rate)"
            )
            
            if not gamma_auto:
                gamma = st.slider(
                    "Growth Rate Parameter (gamma)",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.2,
                    step=0.01,
                    help="Growth rate parameter (higher = faster growth)"
                )
        
        # Create parameters dictionary
        method_params = {}
        
        if not fit_saturation:
            method_params['saturation'] = saturation
            
        if not beta_auto:
            method_params['beta'] = beta
            
        if not gamma_auto:
            method_params['gamma'] = gamma
            
    elif method == "Norton-Bass":
        # Norton-Bass parameters
        st.markdown("#### Norton-Bass Model for Successive Technology Generations")
        st.markdown("This model extends the Bass diffusion model to handle multiple generations of a technology.")
        
        # Number of generations
        num_generations = st.slider(
            "Number of Technology Generations",
            min_value=1,
            max_value=4,
            value=2,
            help="Number of successive technology generations to model"
        )
        
        # Introduction years for each generation
        st.markdown("##### Introduction Years")
        intro_times = [0]  # First generation always starts at relative time 0
        
        # Get the base year for reference
        base_year = None
        if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
            if not historical_data.empty:
                base_year = historical_data[year_col].min()
                
        if base_year is None:
            base_year = 2010  # Default if no data available
            
        st.text(f"Generation 1 introduced in {base_year} (base year)")
        
        # For each subsequent generation, specify introduction year
        for i in range(1, num_generations):
            intro_year = st.number_input(
                f"Generation {i+1} Introduction Year",
                min_value=int(base_year) + 1,
                max_value=int(base_year) + 30,
                value=int(base_year) + i*5,  # Default to every 5 years
                key=f"intro_year_{i}"
            )
            # Convert to relative time
            intro_times.append(intro_year - base_year)
        
        # Parameters for each generation
        st.markdown("##### Generation Parameters")
        st.info("You can either auto-estimate all parameters or specify them for each generation.")
        
        auto_params = st.checkbox(
            "Auto-estimate All Parameters",
            value=True,
            help="Automatically estimate all parameters for all generations"
        )
        
        if not auto_params:
            m_values = []
            p_values = []
            q_values = []
            
            for i in range(num_generations):
                st.markdown(f"**Generation {i+1}**")
                
                # Market potential for this generation
                market_estimate = 0
                if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                    if not historical_data.empty:
                        # For subsequent generations, estimate smaller potential
                        factor = 1.0 if i == 0 else 0.8 ** i
                        market_estimate = historical_data[value_col].max() * 2 * factor
                
                m = st.number_input(
                    f"Market Potential (m{i+1})",
                    min_value=0,
                    value=int(market_estimate) if market_estimate > 0 else 10000 // (i+1),
                    key=f"m_{i}",
                    help=f"Maximum market potential for generation {i+1}"
                )
                m_values.append(m)
                
                # Innovation coefficient
                col1, col2 = st.columns(2)
                
                with col1:
                    p = st.slider(
                        f"Innovation Coefficient (p{i+1})",
                        min_value=0.001,
                        max_value=0.1,
                        value=0.03,
                        step=0.001,
                        key=f"p_{i}",
                        help=f"Coefficient of innovation for generation {i+1}"
                    )
                    p_values.append(p)
                
                with col2:
                    q = st.slider(
                        f"Imitation Coefficient (q{i+1})",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.38,
                        step=0.01,
                        key=f"q_{i}",
                        help=f"Coefficient of imitation for generation {i+1}"
                    )
                    q_values.append(q)
        
        # Create parameters dictionary
        method_params = {
            'num_generations': num_generations,
            'intro_times': intro_times,
            'base_year': base_year
        }
        
        if not auto_params:
            method_params['m_values'] = m_values
            method_params['p_values'] = p_values
            method_params['q_values'] = q_values
            
    elif method == "Lotka-Volterra":
        # Lotka-Volterra parameters
        st.markdown("#### Lotka-Volterra Competition Model Parameters")
        st.markdown("This model adapts ecological competition equations to analyze competition between technologies.")
        
        # Number of competing technologies
        num_techs = st.slider(
            "Number of Competing Technologies",
            min_value=2,
            max_value=5,
            value=2,
            help="Number of competing technologies in the market"
        )
        
        # Names for each technology
        st.markdown("##### Technology Names")
        tech_names = []
        for i in range(num_techs):
            name = st.text_input(
                f"Name for Technology {i+1}",
                value=f"Tech{i+1}",
                key=f"tech_name_{i}"
            )
            tech_names.append(name)
        
        # Auto or manual parameters
        auto_params = st.checkbox(
            "Auto-estimate All Parameters",
            value=True,
            help="Automatically estimate growth rates, capacities, and competition coefficients"
        )
        
        if not auto_params:
            # Growth rates for each technology
            st.markdown("##### Growth Rates")
            growth_rates = []
            
            for i in range(num_techs):
                growth = st.slider(
                    f"Growth Rate for {tech_names[i]}",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key=f"growth_{i}",
                    help=f"Intrinsic growth rate for {tech_names[i]}"
                )
                growth_rates.append(growth)
            
            # Carrying capacities for each technology
            st.markdown("##### Market Capacities")
            capacities = []
            
            # Estimate total market size
            market_estimate = 0
            if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                if not historical_data.empty:
                    market_estimate = historical_data[value_col].max() * 3
            
            for i in range(num_techs):
                capacity = st.number_input(
                    f"Market Capacity for {tech_names[i]}",
                    min_value=0,
                    value=int(market_estimate / num_techs) if market_estimate > 0 else 10000 // num_techs,
                    key=f"capacity_{i}",
                    help=f"Maximum potential market size for {tech_names[i]}"
                )
                capacities.append(capacity)
            
            # Competition matrix
            st.markdown("##### Competition Coefficients")
            st.markdown("These values determine how much each technology affects others. Higher values indicate stronger competition.")
            
            competition_matrix = np.zeros((num_techs, num_techs))
            
            # Only show upper triangle to reduce number of parameters
            for i in range(num_techs):
                for j in range(i+1, num_techs):
                    competition = st.slider(
                        f"Effect of {tech_names[j]} on {tech_names[i]}",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.5,
                        step=0.1,
                        key=f"comp_{i}_{j}",
                        help=f"How strongly {tech_names[j]} competes with {tech_names[i]}"
                    )
                    competition_matrix[i, j] = competition
                    competition_matrix[j, i] = competition  # Assume symmetric competition for simplicity
        
        # Time step for numerical integration
        dt = st.slider(
            "Time Step for Integration",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Time step for numerical solution (smaller = more accurate but slower)"
        )
        
        # Create parameters dictionary
        method_params = {
            'num_techs': num_techs,
            'tech_names': tech_names,
            'dt': dt
        }
        
        if not auto_params:
            method_params['growth_rates'] = growth_rates
            method_params['capacities'] = capacities
            method_params['competition_matrix'] = competition_matrix.tolist()  # Convert to list for JSON serialization
            
    elif method == "Bayesian Structural Time Series":
        # BSTS parameters
        st.markdown("#### Bayesian Structural Time Series Parameters")
        st.markdown("This model combines state space models with Bayesian methods for flexible, interpretable forecasting with uncertainty quantification.")
        
        # Model components
        st.markdown("##### Model Components")
        col1, col2 = st.columns(2)
        
        with col1:
            use_trend = st.checkbox(
                "Include Trend Component",
                value=True,
                help="Include trend component in the model"
            )
            
            if use_trend:
                trend_type = st.radio(
                    "Trend Type",
                    options=["Local Level", "Local Linear"],
                    index=1,
                    horizontal=True,
                    help="Local level: random walk, Local linear: includes slope"
                )
                # Convert to internal format
                trend_type_internal = "local_level" if trend_type == "Local Level" else "local_linear"
                
        with col2:
            use_seasonal = st.checkbox(
                "Include Seasonal Component",
                value=True,
                help="Include seasonal component in the model"
            )
            
            if use_seasonal:
                # Get data frequency to suggest seasonal periods
                suggested_periods = []
                if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                    dates = pd.to_datetime(historical_data['Year'], format='%Y')
                    freq = pd.infer_freq(dates)
                    
                    if freq is None or freq.startswith('A'):  # Annual data or unknown
                        suggested_periods = []  # No obvious seasonality
                    elif freq.startswith('Q'):  # Quarterly data
                        suggested_periods = [4]  # Quarterly seasonality
                    elif freq.startswith('M'):  # Monthly data
                        suggested_periods = [12]  # Monthly seasonality
                
                # Allow user to specify seasonal periods
                seasonal_periods_str = st.text_input(
                    "Seasonal Periods (comma-separated)",
                    value=",".join(map(str, suggested_periods)) if suggested_periods else "",
                    help="Periods for seasonal components (e.g., 12 for monthly data with yearly seasonality)"
                )
                
                if seasonal_periods_str.strip():
                    try:
                        seasonal_periods = [int(p.strip()) for p in seasonal_periods_str.split(',') if p.strip()]
                    except ValueError:
                        st.warning("Invalid seasonal periods. Using auto-detection.")
                        seasonal_periods = None
                else:
                    seasonal_periods = None
        
        # Regression components
        use_regression = st.checkbox(
            "Include Regression Component",
            value=False,
            help="Include regression component with external predictors"
        )
        
        # MCMC parameters
        with st.expander("Advanced Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                niter = st.slider(
                    "MCMC Iterations",
                    min_value=100,
                    max_value=2000,
                    value=1000,
                    step=100,
                    help="Number of MCMC iterations"
                )
                
                nburn = st.slider(
                    "Burn-in Period",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Number of burn-in iterations"
                )
            
            with col2:
                prior_level_sd = st.slider(
                    "Prior Level StdDev",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    format="%.3f",
                    help="Prior standard deviation for level component"
                )
                
                prior_slope_sd = st.slider(
                    "Prior Slope StdDev",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    format="%.3f",
                    help="Prior standard deviation for slope component"
                )
        
        # Create parameters dictionary
        method_params = {
            'use_trend': use_trend,
            'trend_type': trend_type_internal if use_trend else None,
            'use_seasonal': use_seasonal,
            'seasonal_periods': seasonal_periods,
            'use_regression': use_regression,
            'niter': niter,
            'nburn': nburn,
            'prior_level_sd': prior_level_sd,
            'prior_slope_sd': prior_slope_sd
        }
        
    elif method == "Gaussian Process":
        # Gaussian Process parameters
        st.markdown("#### Gaussian Process Parameters")
        st.markdown("Gaussian Processes provide flexible, non-parametric forecasting with uncertainty quantification.")
        
        # Kernel selection
        kernel_type = st.selectbox(
            "Kernel Type",
            options=["RBF", "Matern", "Periodic", "Linear", "RationalQuadratic", "Composite"],
            index=0,
            help="Kernel function determines the covariance structure and patterns the model can capture"
        )
        
        # Kernel-specific parameters
        if kernel_type in ["RBF", "Matern", "RationalQuadratic"]:
            rbf_length_scale = st.slider(
                "Length Scale",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Length scale parameter (higher = smoother functions)"
            )
            
            # Range for length scale bounds
            col1, col2 = st.columns(2)
            with col1:
                lb_scale = st.number_input(
                    "Length Scale Lower Bound",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    format="%.2f",
                    help="Lower bound for length scale optimization"
                )
            with col2:
                ub_scale = st.number_input(
                    "Length Scale Upper Bound",
                    min_value=1.0,
                    max_value=20.0,
                    value=10.0,
                    format="%.2f",
                    help="Upper bound for length scale optimization"
                )
            rbf_length_scale_bounds = (lb_scale, ub_scale)
            
        elif kernel_type == "Periodic":
            # Periodic kernel parameters
            col1, col2 = st.columns(2)
            
            with col1:
                periodic_length_scale = st.slider(
                    "Length Scale",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Length scale parameter"
                )
            
            with col2:
                # Try to guess a periodicity based on data
                suggested_period = 1.0
                if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                    dates = pd.to_datetime(historical_data['Year'], format='%Y')
                    freq = pd.infer_freq(dates)
                    
                    if freq is None or freq.startswith('A'):  # Annual data
                        suggested_period = 1.0
                    elif freq.startswith('Q'):  # Quarterly data
                        suggested_period = 0.25
                    elif freq.startswith('M'):  # Monthly data
                        suggested_period = 1.0  # Annual period for monthly data
                
                periodic_periodicity = st.number_input(
                    "Periodicity",
                    min_value=0.1,
                    max_value=10.0,
                    value=suggested_period,
                    step=0.1,
                    help="Period length (in same units as x-axis)"
                )
                
        elif kernel_type == "Linear":
            # Linear kernel parameter
            linear_variance = st.slider(
                "Variance",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Variance parameter for linear kernel"
            )
        
        elif kernel_type == "Composite":
            # Composite kernel (RBF + Periodic) parameters
            st.info("The composite kernel combines RBF (for trend) and Periodic (for seasonality) components.")
            
            # RBF component
            st.markdown("##### RBF Component (Trend)")
            rbf_length_scale = st.slider(
                "RBF Length Scale",
                min_value=1.0,
                max_value=20.0,
                value=10.0,
                step=0.5,
                help="Length scale for RBF component (higher = smoother trend)"
            )
            
            # Periodic component
            st.markdown("##### Periodic Component (Seasonality)")
            periodic_periodicity = st.number_input(
                "Periodicity",
                min_value=0.1,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Period length (in same units as x-axis)"
            )
        
        # Noise level
        alpha = st.slider(
            "Noise Level (alpha)",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Noise level for observations (higher = more weight to regularization)"
        )
        
        # Other general settings
        col1, col2 = st.columns(2)
        
        with col1:
            normalize_y = st.checkbox(
                "Normalize Target Values",
                value=True,
                help="Normalize target values before fitting"
            )
        
        with col2:
            n_restarts_optimizer = st.slider(
                "Optimizer Restarts",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of restarts for optimizer to avoid local minima"
            )
        
        # Create parameters dictionary
        method_params = {
            'kernel_type': kernel_type,
            'alpha': alpha,
            'normalize_y': normalize_y,
            'n_restarts_optimizer': n_restarts_optimizer
        }
        
        # Add kernel-specific parameters
        if kernel_type in ["RBF", "Matern", "RationalQuadratic"]:
            method_params.update({
                'rbf_length_scale': rbf_length_scale,
                'rbf_length_scale_bounds': rbf_length_scale_bounds
            })
        elif kernel_type == "Periodic":
            method_params.update({
                'periodic_length_scale': periodic_length_scale,
                'periodic_periodicity': periodic_periodicity
            })
        elif kernel_type == "Linear":
            method_params.update({
                'linear_variance': linear_variance
            })
        elif kernel_type == "Composite":
            method_params.update({
                'rbf_length_scale': rbf_length_scale,
                'periodic_periodicity': periodic_periodicity
            })
            
    elif method == "TBATS":
        # TBATS parameters
        st.markdown("#### TBATS Model Parameters")
        st.markdown("TBATS (Trigonometric, Box-Cox transform, ARMA errors, Trend, and Seasonal components) is ideal for time series with complex seasonality.")
        
        # Box-Cox transformation
        use_box_cox = st.selectbox(
            "Box-Cox Transformation",
            options=["Auto", "Yes", "No"],
            index=0,
            help="Box-Cox transformation can help stabilize variance in the time series"
        )
        
        if use_box_cox == "Yes":
            box_cox_lambda = st.slider(
                "Box-Cox Lambda",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Lambda parameter for Box-Cox transformation (0=log, 1=no transform)"
            )
        else:
            box_cox_lambda = None
        
        # Trend and seasonal components
        col1, col2 = st.columns(2)
        
        with col1:
            use_trend = st.checkbox(
                "Include Trend",
                value=True,
                help="Include trend component in the model"
            )
        
        with col2:
            use_damped_trend = st.checkbox(
                "Use Damped Trend",
                value=False,
                help="Damped trend can prevent unrealistic long-term forecasts"
            ) if use_trend else False
        
        # Seasonal periods
        use_seasonal = st.checkbox(
            "Include Seasonality",
            value=True,
            help="Include seasonal components in the model"
        )
        
        if use_seasonal:
            # Get data frequency to suggest seasonal periods
            suggested_periods = []
            if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                dates = pd.to_datetime(historical_data['Year'], format='%Y')
                freq = pd.infer_freq(dates)
                
                if freq is None or freq.startswith('A'):  # Annual data or unknown
                    suggested_periods = []  # No obvious seasonality
                elif freq.startswith('Q'):  # Quarterly data
                    suggested_periods = [4]  # Quarterly seasonality
                elif freq.startswith('M'):  # Monthly data
                    suggested_periods = [12]  # Monthly seasonality
            
            # Allow user to specify seasonal periods
            seasonal_periods_str = st.text_input(
                "Seasonal Periods (comma-separated)",
                value=",".join(map(str, suggested_periods)) if suggested_periods else "",
                help="Periods for seasonal components (e.g., 12 for monthly data with yearly seasonality)"
            )
            
            if seasonal_periods_str.strip():
                try:
                    seasonal_periods = [int(p.strip()) for p in seasonal_periods_str.split(',') if p.strip()]
                except ValueError:
                    st.warning("Invalid seasonal periods. Using auto-detection.")
                    seasonal_periods = None
            else:
                seasonal_periods = None
        else:
            seasonal_periods = None
        
        # ARMA errors
        use_arma_errors = st.checkbox(
            "Use ARMA Errors",
            value=True,
            help="Include ARMA errors to capture autocorrelation in residuals"
        )
        
        if use_arma_errors:
            # ARMA order
            col1, col2 = st.columns(2)
            
            with col1:
                arma_p = st.slider(
                    "ARMA p (AR order)",
                    min_value=0,
                    max_value=5,
                    value=2,
                    help="Autoregressive order"
                )
            
            with col2:
                arma_q = st.slider(
                    "ARMA q (MA order)",
                    min_value=0,
                    max_value=5,
                    value=2,
                    help="Moving average order"
                )
            
            arma_order = (arma_p, arma_q)
        else:
            arma_order = None
        
        # Performance settings
        with st.expander("Advanced Settings"):
            show_warnings = st.checkbox(
                "Show TBATS Warnings",
                value=False,
                help="Show warnings during model fitting"
            )
            
            n_jobs = st.slider(
                "Number of Parallel Jobs",
                min_value=1,
                max_value=8,
                value=1,
                help="Number of parallel jobs for model selection"
            )
        
        # Create parameters dictionary
        method_params = {
            'use_box_cox': True if use_box_cox == "Yes" else (None if use_box_cox == "Auto" else False),
            'box_cox_lambda': box_cox_lambda if use_box_cox == "Yes" else None,
            'use_trend': use_trend,
            'use_damped_trend': use_damped_trend,
            'seasonal_periods': seasonal_periods,
            'use_arma_errors': use_arma_errors,
            'arma_order': arma_order,
            'show_warnings': show_warnings,
            'n_jobs': n_jobs
        }
        
    elif method == "NBEATS":
        # NBEATS parameters
        st.markdown("#### NBEATS Model Parameters")
        st.markdown("Neural Basis Expansion Analysis for Time Series (NBEATS) is a deep learning model for time series forecasting without feature engineering.")
        
        # Input sequence and forecast parameters
        col1, col2 = st.columns(2)
        
        with col1:
            lookback = st.slider(
                "Lookback Window Size",
                min_value=4,
                max_value=20,
                value=10,
                help="Number of past time steps to use as input"
            )
        
        with col2:
            forecast_horizon = st.slider(
                "Forecast Horizon",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of time steps predicted by the model at once"
            )
        
        # Model architecture options
        st.markdown("#### Model Architecture")
        
        # Stack types
        stack_types = st.multiselect(
            "Stack Types",
            options=["Trend Block", "Seasonality Block", "Generic Block"],
            default=["Trend Block", "Seasonality Block"],
            help="Types of building blocks to use in the NBEATS model"
        )
        
        # Convert user-friendly names to actual stack types
        stack_type_mapping = {
            "Trend Block": "trend_block",
            "Seasonality Block": "seasonality_block",
            "Generic Block": "generic_block"
        }
        stack_types_internal = [stack_type_mapping[s] for s in stack_types]
        
        col1, col2 = st.columns(2)
        
        with col1:
            nb_blocks_per_stack = st.slider(
                "Blocks per Stack",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of blocks in each stack"
            )
        
        with col2:
            hidden_layer_units = st.slider(
                "Hidden Layer Units",
                min_value=32,
                max_value=512,
                value=256,
                step=32,
                help="Number of units in hidden layers"
            )
        
        # Training parameters in expander
        with st.expander("Training Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.slider(
                    "Batch Size",
                    min_value=16,
                    max_value=256,
                    value=128,
                    step=16,
                    help="Batch size for training"
                )
                
                max_epochs = st.slider(
                    "Maximum Epochs",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Maximum number of training epochs"
                )
            
            with col2:
                patience = st.slider(
                    "Early Stopping Patience",
                    min_value=3,
                    max_value=20,
                    value=10,
                    help="Number of epochs without improvement before stopping"
                )
                
                learning_rate = st.number_input(
                    "Learning Rate",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.001,
                    format="%.4f",
                    help="Learning rate for optimization"
                )
            
            normalize = st.checkbox(
                "Normalize Data",
                value=True,
                help="Normalize input data before training"
            )
        
        # Create parameters dictionary
        method_params = {
            'lookback': lookback,
            'forecast_horizon': forecast_horizon,
            'stack_types': stack_types_internal,
            'nb_blocks_per_stack': nb_blocks_per_stack,
            'hidden_layer_units': hidden_layer_units,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'patience': patience,
            'learning_rate': learning_rate,
            'normalize': normalize
        }
        
    elif method == "Hybrid ETS-ARIMA":
        # Hybrid ETS-ARIMA parameters
        st.markdown("#### Hybrid ETS-ARIMA Model Parameters")
        st.markdown("This model combines Exponential Smoothing (ETS) and ARIMA to leverage the strengths of both approaches.")
        
        # Auto parameter selection options
        col1, col2 = st.columns(2)
        
        with col1:
            auto_ets = st.checkbox(
                "Auto-select ETS Parameters",
                value=True,
                help="Automatically select the best ETS model parameters"
            )
        
        with col2:
            auto_arima = st.checkbox(
                "Auto-select ARIMA Parameters",
                value=True,
                help="Automatically select the best ARIMA model parameters"
            )
        
        # ETS parameters
        st.markdown("#### ETS Parameters")
        
        if not auto_ets:
            # Manual ETS parameters
            col1, col2 = st.columns(2)
            
            with col1:
                ets_trend = st.selectbox(
                    "Trend Type",
                    options=["None", "Additive", "Multiplicative"],
                    index=1,
                    help="Type of trend component in ETS model"
                )
                
                # Convert to internal format
                ets_trend_internal = None if ets_trend == "None" else ("add" if ets_trend == "Additive" else "mul")
                
                ets_damped_trend = st.checkbox(
                    "Damped Trend",
                    value=False,
                    help="Whether to damp the trend component"
                ) if ets_trend != "None" else False
            
            with col2:
                ets_seasonal = st.selectbox(
                    "Seasonal Type",
                    options=["None", "Additive", "Multiplicative"],
                    index=1,
                    help="Type of seasonal component in ETS model"
                )
                
                # Convert to internal format
                ets_seasonal_internal = None if ets_seasonal == "None" else ("add" if ets_seasonal == "Additive" else "mul")
                
                # Get data frequency to suggest seasonal periods
                suggested_period = 0
                if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                    dates = pd.to_datetime(historical_data['Year'], format='%Y')
                    freq = pd.infer_freq(dates)
                    
                    if freq is None or freq.startswith('A'):  # Annual data or unknown
                        suggested_period = 0  # No obvious seasonality
                    elif freq.startswith('Q'):  # Quarterly data
                        suggested_period = 4  # Quarterly seasonality
                    elif freq.startswith('M'):  # Monthly data
                        suggested_period = 12  # Monthly seasonality
                
                if ets_seasonal != "None":
                    ets_seasonal_periods = st.number_input(
                        "Seasonal Periods",
                        min_value=0,
                        max_value=365,
                        value=suggested_period,
                        help="Number of periods in a seasonal cycle (0=None, 4=quarterly, 12=monthly)"
                    )
                    
                    # Convert 0 to None
                    ets_seasonal_periods = None if ets_seasonal_periods == 0 else ets_seasonal_periods
                else:
                    ets_seasonal_periods = None
        
        # ARIMA parameters
        st.markdown("#### ARIMA Parameters")
        
        if not auto_arima:
            # Manual ARIMA parameters
            st.markdown("##### ARIMA Order (p, d, q)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                arima_p = st.slider(
                    "p (AR Order)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Autoregressive order"
                )
            
            with col2:
                arima_d = st.slider(
                    "d (Differencing)",
                    min_value=0,
                    max_value=2,
                    value=1,
                    help="Differencing order"
                )
            
            with col3:
                arima_q = st.slider(
                    "q (MA Order)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Moving average order"
                )
            
            st.markdown("##### Seasonal ARIMA Order (P, D, Q, s)")
            include_seasonal = st.checkbox(
                "Include Seasonal Component",
                value=True,
                help="Include seasonal component in ARIMA model"
            )
            
            if include_seasonal:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    arima_P = st.slider(
                        "P (Seasonal AR)",
                        min_value=0,
                        max_value=2,
                        value=1,
                        help="Seasonal autoregressive order"
                    )
                
                with col2:
                    arima_D = st.slider(
                        "D (Seasonal Diff)",
                        min_value=0,
                        max_value=1,
                        value=0,
                        help="Seasonal differencing order"
                    )
                
                with col3:
                    arima_Q = st.slider(
                        "Q (Seasonal MA)",
                        min_value=0,
                        max_value=2,
                        value=1,
                        help="Seasonal moving average order"
                    )
                
                with col4:
                    arima_s = st.number_input(
                        "s (Seasonal Periods)",
                        min_value=0,
                        max_value=365,
                        value=suggested_period,
                        help="Number of periods in seasonal cycle"
                    )
                    
                # Prepare seasonal order
                arima_seasonal_order = (arima_P, arima_D, arima_Q, arima_s) if arima_s > 0 else None
            else:
                arima_seasonal_order = None
        
        # Component weights
        st.markdown("#### Component Weights")
        ets_weight = st.slider(
            "ETS Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Weight for ETS component (ARIMA weight = 1 - ETS weight)"
        )
        arima_weight = 1.0 - ets_weight
        st.text(f"ARIMA Weight: {arima_weight:.2f}")
        
        # Create parameters dictionary
        method_params = {
            'auto_ets': auto_ets,
            'auto_arima': auto_arima,
            'ets_weight': ets_weight,
            'arima_weight': arima_weight
        }
        
        if not auto_ets:
            method_params.update({
                'ets_trend': ets_trend_internal,
                'ets_damped_trend': ets_damped_trend,
                'ets_seasonal': ets_seasonal_internal,
                'ets_seasonal_periods': ets_seasonal_periods
            })
        
        if not auto_arima:
            method_params.update({
                'arima_order': (arima_p, arima_d, arima_q),
                'arima_seasonal_order': arima_seasonal_order
            })

    
    # Generate forecast button
    if st.button("Generate Forecast"):
        try:
            # Create a progress indicator
            progress_msg = st.info("Initializing forecasting process...")
            
            # Handle ensemble methods differently
            if method in ["Simple Average Ensemble", "Weighted Ensemble"]:
                # Get selected individual forecasters for the ensemble
                st.info("For ensemble methods, please select the underlying forecasters to include.")

                # Define a list of individual forecasters to choose from
                individual_forecasters = []
                for category in forecaster_categories:
                    if category != "Ensemble Methods":
                        individual_forecasters.extend(forecaster_categories[category])

                selected_forecasters = st.multiselect(
                    "Select forecasters to include in the ensemble",
                    options=individual_forecasters,
                    default=individual_forecasters[:2] if len(individual_forecasters) >= 2 else individual_forecasters
                )

                if not selected_forecasters:
                    st.error("Please select at least one forecaster for the ensemble")
                    return {}

                # Get the data for fitting - by this point historical_data is properly set
                fit_data = historical_data.copy()
                
                # Check if we have data
                if fit_data.empty:
                    st.error("No historical data available for forecasting. Please ensure you have data before the forecast horizon year.")
                    return {'result': None}

                # Convert to the format expected by the forecaster
                fit_df = pd.DataFrame({
                    'date': pd.to_datetime(fit_data[year_col], format='%Y'),
                    'value': fit_data[value_col]
                })

                # Set up base forecasters in the config
                base_forecasters = []
                for selected in selected_forecasters:
                    # Create simple parameters for each forecaster
                    # In a real implementation, we would get proper parameters
                    simple_params = {}
                    base_forecasters.append({
                        'name': selected,
                        'params': simple_params
                    })

                # Add base forecasters to the ensemble config
                method_params['base_forecasters'] = base_forecasters

                # Create the ensemble forecaster
                progress_msg.info("Creating ensemble forecaster...")
                try:
                    forecaster = create_forecaster(method, method_params)
                    if forecaster is None:
                        st.error(f"Failed to create forecaster '{method}'. Forecaster is None.")
                        return {'result': None}
                except Exception as e:
                    st.error(f"Error creating ensemble forecaster: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    return {'result': None}

                # For each selected forecaster, create, fit, and add to the ensemble
                for selected in selected_forecasters:
                    try:
                        # Create the individual forecaster
                        progress_msg.info(f"Creating and fitting {selected} forecaster...")
                        individual = create_forecaster(selected, {})
                        if individual is None:
                            st.warning(f"Failed to create forecaster '{selected}'. Skipping.")
                            continue

                        # Fit the individual forecaster
                        individual.fit(fit_df)

                        # Add to the ensemble
                        forecaster.add_forecaster(individual)
                    except Exception as e:
                        st.warning(f"Error adding {selected} to ensemble: {str(e)}")

                # Fit the ensemble with the data
                progress_msg.info("Fitting ensemble forecaster...")
                try:
                    forecaster.fit(fit_df)
                except Exception as e:
                    st.error(f"Error fitting ensemble forecaster: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    return {'result': None}
            else:
                # Regular non-ensemble forecaster
                # Create a forecaster instance
                progress_msg.info(f"Creating {method} forecaster...")
                try:
                    forecaster = create_forecaster(method, method_params)
                    if forecaster is None:
                        st.error(f"Failed to create forecaster '{method}'. Forecaster is None.")
                        return {'result': None}
                    
                except Exception as e:
                    st.error(f"Error creating forecaster: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    return {'result': None}

                # Get the data for fitting - by this point historical_data is properly set
                fit_data = historical_data.copy()
                
                # Check if we have data
                if fit_data.empty:
                    st.error("No historical data available for forecasting. Please ensure you have data before the forecast horizon year.")
                    return {'result': None}

                # Convert to the format expected by the forecaster
                fit_df = pd.DataFrame({
                    'date': pd.to_datetime(fit_data[year_col], format='%Y'),
                    'value': fit_data[value_col]
                })

                # Fit the forecaster
                progress_msg.info(f"Fitting {method} forecaster to historical data...")
                try:
                    forecaster.fit(fit_df)
                except Exception as e:
                    st.error(f"Error fitting forecaster: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    return {'result': None}

            # Generate forecast
            progress_msg.info(f"Generating forecast for {horizon} periods...")
            try:
                forecast_df = forecaster.forecast(horizon, frequency='Y')
                if forecast_df is None:
                    st.error("Forecast method returned None. Check implementation of forecast method.")
                    return {'result': None}
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                return {'result': None}
            
            # Get confidence intervals
            progress_msg.info("Calculating confidence intervals...")
            try:
                ci_df = forecaster.get_confidence_intervals()
            except Exception as e:
                logger.error(f"Error getting confidence intervals: {str(e)}")
                ci_df = None
            
            # Convert back to the original format for consistency
            forecast_columns = {
                year_col: [int(date.year) for date in forecast_df['date']],
                value_col: forecast_df['value']
            }
            
            # Add forecast type marker
            forecast_columns[type_col] = 'Forecast'
                
            forecast_result = pd.DataFrame(forecast_columns)
            
            # Create a combined DataFrame with both historical and forecasted data
            # Create a copy of historical data with forecast marker
            historical_copy = historical_data[[year_col, value_col]].copy()
            historical_copy[type_col] = 'Historical'
            
            combined_data = pd.concat([
                historical_copy,
                forecast_result
            ]).sort_values(by=year_col)
            
            # Display the forecast
            progress_msg.success("Forecast generated successfully!")
            st.subheader("Forecast Result")
            
            # Plot the data
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical_data[year_col],
                y=historical_data[value_col],
                mode='lines+markers',
                name='Historical',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='blue')
            ))
            
            # Add forecast data
            fig.add_trace(go.Scatter(
                x=forecast_result[year_col],
                y=forecast_result[value_col],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8, color='red')
            ))
            
            # Add confidence intervals if available
            if ci_df is not None:
                # Lower bound
                fig.add_trace(go.Scatter(
                    x=[int(date.year) for date in ci_df['date']],
                    y=ci_df['lower'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
                    showlegend=False
                ))
                
                # Upper bound
                fig.add_trace(go.Scatter(
                    x=[int(date.year) for date in ci_df['date']],
                    y=ci_df['upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title=f'Global Market Forecast ({method})',
                xaxis_title='Year',
                yaxis_title='Market Value',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to save the forecast
            if st.button("Save Forecast"):
                # Store in session state
                st.session_state.global_forecast = combined_data
                
                # Also update global_forecast in the data_loader if present
                if hasattr(st.session_state, 'data_loader'):
                    st.session_state.data_loader.global_forecast = combined_data
                
                st.success("Forecast saved successfully!")
            
            # Return forecast configuration
            return {
                'method': method,
                'params': method_params,
                'horizon': horizon,
                'result': combined_data
            }
        
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return {'result': None}

    # Default return - ensure we always return something
    return {'result': None}


def render_distribution_interface(market_distributor=None, config_manager=None) -> dict:
    """
    Render the market distribution interface.
    
    Args:
        market_distributor: MarketDistributor instance
        config_manager: ConfigurationManager instance
        
    Returns:
        Dictionary with distribution settings
    """
    st.header("Market Distribution Configuration")
    
    # Initialize session state indicators if not present
    if not hasattr(st.session_state, 'indicators'):
        st.session_state.indicators = {}
    
    # Check if we have necessary components
    if config_manager is None:
        st.warning("Configuration manager not initialized.")
        return {}
        
    # Try to create a market distributor if one wasn't provided
    if market_distributor is None:
        try:
            from src.data_processing.data_loader import DataLoader
            from src.indicators.indicator_analyzer import IndicatorAnalyzer
            
            # Initialize data loader
            data_loader = DataLoader(config_manager)
            
            # Initialize indicator analyzer
            indicator_analyzer = IndicatorAnalyzer(config_manager, data_loader)
            
            # Create market distributor
            market_distributor = MarketDistributor(config_manager, data_loader, indicator_analyzer)
            st.success("Created a new market distributor.")
        except Exception as e:
            st.error(f"Error creating market distributor: {str(e)}")
            return {}
    
    # Initialize settings dictionary
    distribution_settings = {}
    
    # Create tabs for different configuration groups
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tier Configuration", 
        "Growth Constraints", 
        "Indicators", 
        "Smoothing", 
        "Redistribution"
    ])
    
    # Tier Configuration
    with tab1:
        tier_settings = render_tier_configuration(market_distributor, config_manager)
        distribution_settings.update(tier_settings)
    
    # Growth Constraints
    with tab2:
        growth_settings = render_growth_constraints(market_distributor)
        distribution_settings['growth_constraints'] = growth_settings
    
    # Indicator Configuration
    with tab3:
        indicator_settings = render_indicator_configuration(market_distributor)
        distribution_settings['indicators'] = indicator_settings
    
    # Smoothing Configuration
    with tab4:
        smoothing_settings = render_smoothing_configuration(market_distributor)
        distribution_settings['smoothing'] = smoothing_settings
    
    # Redistribution Settings
    with tab5:
        redistribution_settings = render_redistribution_settings(market_distributor)
        distribution_settings.update(redistribution_settings)
    
    # Apply button
    if st.button("Apply Distribution Settings"):
        # Update market distributor settings
        try:
            # First try to update settings using the update_settings method
            if hasattr(market_distributor, 'update_settings'):
                # Call the update_settings method
                market_distributor.update_settings(distribution_settings)
                
                # Success message
                st.success("Distribution settings updated successfully!")
            # Fallback to direct attribute updates if the method doesn't exist
            elif hasattr(market_distributor, 'distribution_settings'):
                # Log a warning about using fallback method
                logger.warning("Using fallback method to update settings - update_settings method not found")
                
                # Update settings
                market_distributor.distribution_settings.update(distribution_settings)
                
                # Reset tiers if tier settings changed
                if any(key in distribution_settings for key in ['tier_determination', 'manual_tiers', 'kmeans_params']):
                    if hasattr(market_distributor, 'tiers'):
                        market_distributor.tiers = None
                    if hasattr(market_distributor, 'tier_thresholds'):
                        market_distributor.tier_thresholds = None
                
                # Update gradient harmonizer settings if available
                if 'smoothing' in distribution_settings and hasattr(market_distributor, 'gradient_harmonizer'):
                    harmonizer = market_distributor.gradient_harmonizer
                    if hasattr(harmonizer, 'update_settings'):
                        harmonizer.update_settings(distribution_settings.get('smoothing', {}))
                    elif hasattr(harmonizer, 'settings'):
                        # Update settings for gradient harmonizer
                        for key, value in distribution_settings.get('smoothing', {}).items():
                            harmonizer.settings[key] = value
                            
                # Success message
                st.success("Distribution settings updated successfully!")
            else:
                st.warning("Market distributor does not have a distribution_settings attribute")
            
            # Update the session state
            if 'market_distributor' in st.session_state:
                st.session_state.market_distributor = market_distributor
                
            # Save to configuration if desired
            if st.checkbox("Save to configuration file", value=False):
                # Update configuration directly
                if hasattr(config_manager, 'config'):
                    # Update market_distribution settings
                    if 'market_distribution' not in config_manager.config:
                        config_manager.config['market_distribution'] = {}
                    
                    # Update each setting section
                    for key, value in distribution_settings.items():
                        config_manager.config['market_distribution'][key] = value
                    
                    # Save the file if path exists
                    if hasattr(config_manager, 'config_path') and config_manager.config_path:
                        config_manager.save_config(config_manager.config_path)
                        st.success(f"Settings saved to {config_manager.config_path}")
                else:
                    st.error("Configuration manager is not properly initialized")
        except Exception as e:
            st.error(f"Error updating settings: {str(e)}")
    
    # Run Market Distribution button
    st.header("Run Market Distribution")
    st.markdown("""
    Click the button below to run the market distribution process using the current settings.
    This will distribute the global market forecast across countries and generate market projections.
    """)
    
    # Check if required data is available
    data_ready = True
    error_msg = ""
    
    if 'global_forecast' not in st.session_state or st.session_state.global_forecast is None:
        data_ready = False
        error_msg += "Global forecast data is missing. "
        
    if 'country_historical' not in st.session_state or st.session_state.country_historical is None:
        data_ready = False
        error_msg += "Country historical data is missing. "
    
    if not data_ready:
        st.warning(f"Cannot run distribution: {error_msg}")
        if st.button("Go to Data Input"):
            st.session_state.active_page = "Data Input"
            st.rerun()
    else:
        # Run distribution button
        if st.button("Run Market Distribution", key="run_distribution_btn"):
            try:
                with st.spinner("Running market distribution..."):
                    # Call the distribute_market method
                    if market_distributor is not None:
                        # Run the distribution
                        distributed_market = market_distributor.distribute_market()
                        
                        # Store result in session state
                        st.session_state.distributed_market = distributed_market
                        
                        # Display summary
                        st.success("Market distribution completed successfully!")
                        
                        # Show distribution summary
                        if distributed_market is not None:
                            total_countries = distributed_market['Country'].nunique() if 'Country' in distributed_market.columns else 0
                            years = sorted(distributed_market['Year'].unique()) if 'Year' in distributed_market.columns else []
                            first_year = min(years) if years else "N/A"
                            last_year = max(years) if years else "N/A"
                            
                            # Display summary
                            st.subheader("Distribution Summary")
                            st.markdown(f"""
                            - **Total Countries/Regions**: {total_countries}
                            - **Forecast Period**: {first_year} - {last_year}
                            - **Total Years**: {len(years)}
                            """)
                            
                            # Option to view visualization
                            if st.button("View Visualization"):
                                st.session_state.active_page = "Visualization"
                                st.rerun()
                    else:
                        st.error("Market distributor is not initialized. Cannot run distribution.")
            except Exception as e:
                st.error(f"Error running market distribution: {str(e)}")
                logger.exception("Error in market distribution")
    
    return distribution_settings