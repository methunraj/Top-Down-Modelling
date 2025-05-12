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
    type_col = 'Type'
    
    if type_col in global_df.columns:
        # Separate historical and forecast data
        historical_data = global_df[global_df[type_col] == 'Historical']
        forecast_data = global_df[global_df[type_col] == 'Forecast']
        
        # Check if we have both historical and forecast data
        if historical_data.empty:
            st.warning("No historical data found.")
            return {}
        
        if not forecast_data.empty:
            st.info("Forecast data is already available in the uploaded data.")
            
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
        if type_col in global_df.columns:
            latest_year = historical_data[year_col].max()
        else:
            latest_year = global_df[year_col].max()
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
            if type_col in global_df.columns:
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
            if type_col in global_df.columns:
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
                if type_col in global_df.columns:
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
    
    # Generate forecast button
    if st.button("Generate Forecast"):
        try:
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

                # Get the data for fitting
                if type_col in global_df.columns:
                    fit_data = historical_data.copy()
                else:
                    fit_data = global_df.copy()

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
                forecaster = create_forecaster(method, method_params)

                # For each selected forecaster, create, fit, and add to the ensemble
                for selected in selected_forecasters:
                    try:
                        # Create the individual forecaster
                        individual = create_forecaster(selected, {})

                        # Fit the individual forecaster
                        individual.fit(fit_df)

                        # Add to the ensemble
                        forecaster.add_forecaster(individual)
                    except Exception as e:
                        st.warning(f"Error adding {selected} to ensemble: {str(e)}")

                # Fit the ensemble with the data
                forecaster.fit(fit_df)
            else:
                # Regular non-ensemble forecaster
                # Create a forecaster instance
                forecaster = create_forecaster(method, method_params)

                # Get the data for fitting
                if type_col in global_df.columns:
                    fit_data = historical_data.copy()
                else:
                    fit_data = global_df.copy()

                # Convert to the format expected by the forecaster
                fit_df = pd.DataFrame({
                    'date': pd.to_datetime(fit_data[year_col], format='%Y'),
                    'value': fit_data[value_col]
                })

                # Fit the forecaster
                forecaster.fit(fit_df)

            # Generate forecast
            forecast_df = forecaster.forecast(horizon, frequency='Y')
            
            # Get confidence intervals
            try:
                ci_df = forecaster.get_confidence_intervals()
            except Exception as e:
                logger.error(f"Error getting confidence intervals: {str(e)}")
                ci_df = None
            
            # Convert back to the original format for consistency
            forecast_result = pd.DataFrame({
                year_col: [int(date.year) for date in forecast_df['date']],
                value_col: forecast_df['value'],
                type_col: 'Forecast'
            })
            
            # Create a combined DataFrame with both historical and forecasted data
            combined_data = pd.concat([
                historical_data[[year_col, value_col, type_col]],
                forecast_result
            ]).sort_values(by=year_col)
            
            # Display the forecast
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
            return {}
    
    # Return empty dict if forecast not generated
    return {}


def render_distribution_interface(config_manager: ConfigurationManager) -> Dict[str, Any]:
    """
    Render market distribution interface.
    
    Args:
        config_manager: ConfigurationManager instance
        
    Returns:
        Dictionary with distribution results and configuration
    """
    st.title("Market Distribution")
    
    # Check if we have the required data
    if 'global_forecast' not in st.session_state or st.session_state.global_forecast is None:
        st.warning("Please upload global forecast data first!")
        if st.button("Go to Data Input"):
            st.session_state.active_page = "Data Input"
            st.rerun()
        return {}
    
    if 'country_historical' not in st.session_state or st.session_state.country_historical is None:
        st.warning("Please upload country historical data first!")
        if st.button("Go to Data Input"):
            st.session_state.active_page = "Data Input"
            st.rerun()
        return {}
    
    # Create tabs for different aspects of distribution
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Tier Configuration", "Growth Constraints", "Indicators", "Smoothing", "Redistribution"
    ])
    
    # Create MarketDistributor instance
    market_distributor = None
    
    try:
        # Ensure the data directory exists
        import os
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save the global forecast to a file if it's in session state but not on disk
        global_forecast_path = "data/global_forecast.xlsx"
        if 'global_forecast' in st.session_state and not os.path.exists(global_forecast_path):
            st.info("Saving global forecast data to file...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(global_forecast_path), exist_ok=True)
                # Save to Excel
                st.session_state.global_forecast.to_excel(global_forecast_path, index=False)
                st.success(f"Global forecast data saved to {global_forecast_path}")
            except Exception as e:
                st.warning(f"Error saving global forecast data: {str(e)}")

        # Save the country historical data to a file if it's in session state but not on disk
        country_data_path = "data/country_data.xlsx"
        if 'country_historical' in st.session_state and not os.path.exists(country_data_path):
            st.info("Saving country historical data to file...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(country_data_path), exist_ok=True)
                # Save to Excel
                st.session_state.country_historical.to_excel(country_data_path, index=False)
                st.success(f"Country historical data saved to {country_data_path}")
            except Exception as e:
                st.warning(f"Error saving country historical data: {str(e)}")

        # Create indicators directory if it doesn't exist
        indicators_dir = "data/indicators"
        if not os.path.exists(indicators_dir):
            os.makedirs(indicators_dir, exist_ok=True)

        # Save indicator data if available in session state
        if hasattr(st.session_state, 'indicators') and st.session_state.indicators:
            for name, details in st.session_state.indicators.items():
                indicator_path = f"data/indicators/{name.lower().replace(' ', '_')}.xlsx"
                if not os.path.exists(indicator_path):
                    try:
                        details['data'].to_excel(indicator_path, index=False)
                        st.success(f"Saved indicator data to {indicator_path}")
                    except Exception as e:
                        st.warning(f"Error saving indicator data {name}: {str(e)}")

        # Get data_loader from session or create new one
        if 'data_loader' not in st.session_state:
            from src.data_processing.data_loader import DataLoader
            data_loader = DataLoader(config_manager)

            # Override data with session data
            data_loader.global_forecast = st.session_state.global_forecast
            data_loader.country_historical = st.session_state.country_historical

            st.session_state.data_loader = data_loader
        else:
            data_loader = st.session_state.data_loader
        
        # Get indicator_analyzer from session or create new one
        if 'indicator_analyzer' not in st.session_state:
            from src.indicators.indicator_analyzer import IndicatorAnalyzer
            indicator_analyzer = IndicatorAnalyzer(config_manager, data_loader)
            
            # Add indicators from session if available
            if hasattr(st.session_state, 'indicators') and st.session_state.indicators:
                for name, details in st.session_state.indicators.items():
                    indicator_analyzer.add_indicator(
                        indicator_name=details['meta']['name'],
                        indicator_data=details['data'],
                        indicator_type=details['meta']['type'],
                        weight=details['meta']['weight']
                    )
            
            st.session_state.indicator_analyzer = indicator_analyzer
        else:
            indicator_analyzer = st.session_state.indicator_analyzer
        
        # Create MarketDistributor
        market_distributor = MarketDistributor(config_manager, data_loader, indicator_analyzer)
        
        # Store configuration updates
        updated_config = {}
        
        # Tab 1: Tier Configuration
        with tab1:
            tier_config = render_tier_configuration(market_distributor, config_manager)
            updated_config['tier_config'] = tier_config
        
        # Tab 2: Growth Constraints
        with tab2:
            growth_config = render_growth_constraints(market_distributor)
            updated_config['growth_config'] = growth_config
        
        # Tab 3: Indicators
        with tab3:
            indicator_config = render_indicator_configuration(market_distributor)
            updated_config['indicator_config'] = indicator_config
        
        # Tab 4: Smoothing
        with tab4:
            smoothing_config = render_smoothing_configuration(market_distributor)
            updated_config['smoothing_config'] = smoothing_config
        
        # Tab 5: Redistribution
        with tab5:
            redistribution_config = render_redistribution_settings(market_distributor)
            updated_config['redistribution_config'] = redistribution_config
    
    except Exception as e:
        st.error(f"Error initializing market distributor: {str(e)}")
        return {}
    
    # Run distribution
    st.header("Run Distribution")
    
    # Update market distributor with new configuration if needed
    if updated_config:
        try:
            # Update tier configuration
            if 'tier_config' in updated_config:
                tier_config = updated_config['tier_config']
                
                if 'tier_determination' in tier_config:
                    market_distributor.distribution_settings['tier_determination'] = tier_config['tier_determination']
                
                if 'kmeans_params' in tier_config:
                    market_distributor.distribution_settings['kmeans_params'] = tier_config['kmeans_params']
                
                if 'manual_tiers' in tier_config:
                    market_distributor.distribution_settings['manual_tiers'] = tier_config['manual_tiers']
                
                if 'country_tiers' in tier_config:
                    market_distributor.distribution_settings['country_tiers'] = tier_config['country_tiers']
            
            # Update growth constraints
            if 'growth_config' in updated_config:
                growth_config = updated_config['growth_config']
                
                if 'growth_constraints' not in market_distributor.distribution_settings:
                    market_distributor.distribution_settings['growth_constraints'] = {}
                
                if 'determination_method' in growth_config:
                    market_distributor.distribution_settings['growth_constraints']['determination_method'] = growth_config['determination_method']
                
                if 'constraint_factor' in growth_config:
                    market_distributor.distribution_settings['growth_constraints']['constraint_factor'] = growth_config['constraint_factor']
                
                if 'manual_constraints' in growth_config:
                    market_distributor.distribution_settings['growth_constraints']['manual_constraints'] = growth_config['manual_constraints']
            
            # Update indicator configuration
            if 'indicator_config' in updated_config and indicator_analyzer:
                indicator_config = updated_config['indicator_config']
                
                # Update indicator weights
                if 'indicator_weights' in indicator_config:
                    for name, weight in indicator_config['indicator_weights'].items():
                        indicator_analyzer.set_indicator_weight(name, weight)
                
                # Update transformation method
                if 'transformation' in indicator_config:
                    if 'weight_params' not in config_manager.config:
                        config_manager.config['weight_params'] = {}
                    
                    config_manager.config['weight_params']['transformation'] = indicator_config['transformation']
                
                # Update significance method
                if 'significance_method' in indicator_config:
                    if 'weight_params' not in config_manager.config:
                        config_manager.config['weight_params'] = {}
                    
                    config_manager.config['weight_params']['significance_method'] = indicator_config['significance_method']
            
            # Update smoothing configuration
            if 'smoothing_config' in updated_config:
                smoothing_config = updated_config['smoothing_config']
                market_distributor.distribution_settings['smoothing'] = smoothing_config
            
            # Update redistribution settings
            if 'redistribution_config' in updated_config:
                redistribution_config = updated_config['redistribution_config']
                
                if 'redistribution_start_year' in redistribution_config:
                    market_distributor.distribution_settings['redistribution_start_year'] = redistribution_config['redistribution_start_year']
                elif 'redistribution_start_year' in market_distributor.distribution_settings:
                    # Remove redistribution_start_year if it was disabled
                    del market_distributor.distribution_settings['redistribution_start_year']
        
        except Exception as e:
            st.error(f"Error updating market distributor configuration: {str(e)}")
    
    # Configuration summary
    with st.expander("Configuration Summary"):
        st.json(market_distributor.distribution_settings)
    
    # Run distribution button
    if st.button("Run Market Distribution"):
        try:
            # Show progress message
            progress_msg = st.info("Running market distribution...")
            
            # Run the distribution
            distributed_market = market_distributor.distribute_market()
            
            # Update progress
            progress_msg.info("Distribution completed! Preparing results...")
            
            # Store result in session state
            st.session_state.distributed_market = distributed_market
            
            # Show success message
            st.success("Market distribution completed successfully!")
            
            # Extract some summary statistics for display
            total_countries = distributed_market['Country'].nunique()
            total_years = distributed_market['Year'].nunique()
            years = sorted(distributed_market['Year'].unique())
            first_year = min(years)
            last_year = max(years)
            
            # Display summary statistics
            st.subheader("Distribution Result Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Countries", total_countries)
            
            with col2:
                st.metric("Total Years", total_years)
            
            with col3:
                st.metric("Year Range", f"{first_year} - {last_year}")
            
            # Show a preview of the results
            st.subheader("Results Preview")
            
            # Group by year to show total market size
            yearly_totals = distributed_market.groupby('Year')['Value'].sum().reset_index()
            yearly_totals = yearly_totals.sort_values(by='Year')
            
            # Create a chart of total market by year
            fig = px.line(
                yearly_totals,
                x='Year',
                y='Value',
                title='Total Market Size by Year',
                labels={'Value': 'Market Size'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top countries for the latest year
            latest_year = max(years)
            latest_data = distributed_market[distributed_market['Year'] == latest_year]
            latest_data = latest_data.sort_values(by='Value', ascending=False)
            
            st.subheader(f"Top 10 Countries ({latest_year})")
            
            # Create a bar chart of top 10 countries
            top10 = latest_data.head(10)
            fig = px.bar(
                top10,
                x='Country',
                y='Value',
                title=f'Top 10 Countries by Market Size ({latest_year})',
                labels={'Value': 'Market Size'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to view detailed results
            if st.button("View Detailed Results"):
                st.session_state.active_page = "Visualization"
                st.rerun()
            
            # Return distribution results
            return {
                'distributed_market': distributed_market,
                'config': market_distributor.distribution_settings
            }
        
        except Exception as e:
            st.error(f"Error running market distribution: {str(e)}")
            return {}
    
    # Return empty dict if distribution not run
    return {}