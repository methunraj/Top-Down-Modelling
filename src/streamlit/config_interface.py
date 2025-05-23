"""
Streamlit Configuration Interface Module

This module provides components and utilities for interacting with the
configuration system through the Streamlit interface, allowing users
to view, edit, and save configuration files.
"""

import os
import yaml
import json
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

from src.config.config_manager import ConfigurationManager

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> ConfigurationManager:
    """
    Load a configuration file and return a configuration manager instance.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigurationManager instance
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        config_manager = ConfigurationManager(config_path)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config_manager
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise ValueError(f"Invalid configuration file: {str(e)}")


def save_config_file(config_data: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration data to a file.
    
    Args:
        config_data: Configuration data dictionary
        config_path: Path to save the configuration file
        
    Raises:
        ValueError: If the configuration data is invalid or cannot be saved
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save based on file extension
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
        elif config_path.endswith('.json'):
            with open(config_path, 'w') as file:
                json.dump(config_data, file, indent=2)
        else:
            # Default to YAML
            with open(config_path, 'w') as file:
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Successfully saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        raise ValueError(f"Failed to save configuration: {str(e)}")


def render_config_selector() -> Optional[str]:
    """
    Render a configuration file selector.
    
    Returns:
        Path to the selected configuration file or None if no file selected
    """
    st.subheader("Configuration File")
    
    # Option to use existing config or create new
    config_option = st.radio(
        "Configuration Option",
        options=["Load Existing Configuration", "Create New Configuration"],
        index=0,
        horizontal=True
    )
    
    if config_option == "Load Existing Configuration":
        # Find available config files
        config_dir = "config"
        if not os.path.exists(config_dir):
            st.warning(f"Configuration directory not found: {config_dir}")
            return None
        
        config_files = [f for f in os.listdir(config_dir) 
                       if f.endswith(('.yaml', '.yml', '.json')) and 
                       os.path.isfile(os.path.join(config_dir, f))]
        
        if not config_files:
            st.warning("No configuration files found in the config directory")
            return None
        
        # Select a config file
        selected_config = st.selectbox(
            "Select Configuration File",
            options=config_files,
            index=0
        )
        
        config_path = os.path.join(config_dir, selected_config)
        
        # Option to preview the selected config
        if st.checkbox("Preview Configuration"):
            try:
                with open(config_path, 'r') as file:
                    content = file.read()
                st.code(content, language='yaml')
            except Exception as e:
                st.error(f"Error reading configuration file: {str(e)}")
        
        return config_path
    else:
        # Create new configuration
        new_config_name = st.text_input(
            "New Configuration Name",
            value="market_config.yaml"
        )
        
        # Ensure the name has a valid extension
        if not new_config_name.endswith(('.yaml', '.yml', '.json')):
            new_config_name += '.yaml'
        
        config_path = os.path.join("config", new_config_name)
        
        # Check if the file already exists
        if os.path.exists(config_path):
            st.warning(f"Configuration file already exists: {config_path}")
            if st.checkbox("Overwrite existing file"):
                return config_path
            return None
        
        return config_path


def render_project_settings(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render project settings section of the configuration interface.
    
    Args:
        config_data: Current configuration data
        
    Returns:
        Updated configuration data
    """
    st.header("Project Settings")
    
    # Get current project settings
    project_settings = config_data.get('project', {})
    
    # Project information
    project_name = st.text_input(
        "Project Name",
        value=project_settings.get('name', "Universal Market Forecast")
    )
    
    market_type = st.text_input(
        "Market Type",
        value=project_settings.get('market_type', "Technology Market")
    )
    
    version = st.text_input(
        "Version",
        value=project_settings.get('version', "1.0")
    )
    
    description = st.text_area(
        "Description",
        value=project_settings.get('description', "Universal market forecasting project")
    )
    
    # Update config data
    config_data['project'] = {
        'name': project_name,
        'version': version,
        'market_type': market_type,
        'description': description
    }
    
    return config_data


def render_data_sources(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render data sources section of the configuration interface.

    Args:
        config_data: Current configuration data

    Returns:
        Updated configuration data
    """
    st.header("Data Sources")

    # Initialize data_sources if it doesn't exist
    if 'data_sources' not in config_data:
        config_data['data_sources'] = {}

    # Get current data sources
    data_sources = config_data.get('data_sources', {})

    # Create tabs for different data sources
    tab_global, tab_country, tab_indicators = st.tabs([
        "Global Forecast", "Country Historical", "Indicators"
    ])

    # Tab 1: Global Forecast
    with tab_global:
        st.subheader("Global Market Forecast Data")

        global_forecast = data_sources.get('global_forecast', {})

        global_path = st.text_input(
            "Data File Path",
            value=global_forecast.get('path', "data/global_forecast.xlsx")
        )

        global_sheet = st.text_input(
            "Sheet Name",
            value=global_forecast.get('sheet_name', "Sheet1")
        )

        # Column identifiers
        global_identifier = global_forecast.get('identifier', {})

        st.markdown("**Column Identifiers**")
        col1, col2, col3 = st.columns(3)

        with col1:
            value_column = st.text_input(
                "Value Column",
                value=global_identifier.get('value_column', "Value")
            )

        with col2:
            year_column = st.text_input(
                "Year Column",
                value=global_identifier.get('year_column', "Year")
            )

        with col3:
            type_column = st.text_input(
                "Type Column",
                value=global_identifier.get('type_column', "Type")
            )

        # Update config data
        config_data['data_sources']['global_forecast'] = {
            'path': global_path,
            'sheet_name': global_sheet,
            'identifier': {
                'value_column': value_column,
                'year_column': year_column,
                'type_column': type_column
            }
        }
    
    # Tab 2: Country Historical
    with tab_country:
        st.subheader("Country Historical Data")
        
        country_historical = data_sources.get('country_historical', {})
        
        country_path = st.text_input(
            "Data File Path",
            value=country_historical.get('path', "data/country_data.xlsx")
        )
        
        country_sheet = st.text_input(
            "Sheet Name",
            value=country_historical.get('sheet_name', "Sheet1"),
            key="country_sheet"
        )
        
        # Data format
        data_format = st.radio(
            "Data Format",
            options=["wide", "long"],
            index=0 if country_historical.get('format', "wide") == "wide" else 1,
            horizontal=True
        )
        
        # Column identifiers
        country_identifier = country_historical.get('identifier', {})
        
        st.markdown("**Column Identifiers**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            id_column = st.text_input(
                "ID Column",
                value=country_identifier.get('id_column', "idGeo")
            )
        
        with col2:
            name_column = st.text_input(
                "Name Column",
                value=country_identifier.get('name_column', "Country")
            )
        
        with col3:
            market_column = st.text_input(
                "Market Column",
                value=country_identifier.get('market_column', "nameVertical")
            )
        
        # Update config data
        config_data['data_sources']['country_historical'] = {
            'path': country_path,
            'sheet_name': country_sheet,
            'format': data_format,
            'identifier': {
                'id_column': id_column,
                'name_column': name_column,
                'market_column': market_column
            }
        }
    
    # Tab 3: Indicators
    with tab_indicators:
        st.subheader("Market Indicators")
        
        # Get existing indicators
        indicators = data_sources.get('indicators', [])
        
        # Display existing indicators
        if indicators:
            st.markdown("### Existing Indicators")
            
            for i, indicator in enumerate(indicators):
                with st.expander(f"{indicator.get('name', f'Indicator {i+1}')}"):
                    # Show indicator details
                    st.text(f"Path: {indicator.get('path', 'N/A')}")
                    st.text(f"Type: {indicator.get('type', 'value')}")
                    st.text(f"Weight: {indicator.get('weight', 'auto')}")
                    
                    # Option to remove
                    if st.button(f"Remove Indicator {i+1}"):
                        indicators.pop(i)
                        st.success(f"Removed indicator {i+1}")
                        st.rerun()
        
        # Add new indicator
        st.markdown("### Add New Indicator")
        with st.form("add_indicator"):
            new_name = st.text_input("Indicator Name")
            new_path = st.text_input("Data File Path")
            new_sheet = st.text_input("Sheet Name", value="Sheet1")
            new_id_column = st.text_input("ID Column", value="idGeo")
            new_country_column = st.text_input("Country Column", value="Country")
            new_format = st.radio(
                "Data Format",
                options=["wide", "long"],
                index=0,
                horizontal=True
            )
            new_type = st.radio(
                "Indicator Type",
                options=["value", "rank"],
                index=0,
                horizontal=True
            )
            new_weight = st.radio(
                "Weight Calculation",
                options=["auto", "manual"],
                index=0,
                horizontal=True
            )
            
            if new_weight == "manual":
                new_weight_value = st.slider(
                    "Weight Value",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01
                )
            
            submit = st.form_submit_button("Add Indicator")
            
            if submit and new_name and new_path:
                # Create new indicator
                new_indicator = {
                    'name': new_name,
                    'path': new_path,
                    'sheet_name': new_sheet,
                    'id_column': new_id_column,
                    'country_column': new_country_column,
                    'format': new_format,
                    'type': new_type,
                    'weight': new_weight_value if new_weight == "manual" else "auto"
                }
                
                # Add to indicators list
                indicators.append(new_indicator)
                
                st.success(f"Added indicator: {new_name}")
        
        # Update config data
        config_data['data_sources']['indicators'] = indicators
    
    return config_data


def render_column_mapping(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render column mapping section of the configuration interface.

    Args:
        config_data: Current configuration data

    Returns:
        Updated configuration data
    """
    st.header("Column Mapping")

    # Initialize column_mapping if it doesn't exist
    if 'column_mapping' not in config_data:
        config_data['column_mapping'] = {}

    # Get current column mapping
    column_mapping = config_data.get('column_mapping', {})

    # Create tabs for different mappings
    tab_global, tab_country, tab_indicators = st.tabs([
        "Global Forecast", "Country Historical", "Indicators"
    ])

    # Tab 1: Global Forecast Mapping
    with tab_global:
        st.subheader("Global Forecast Column Mapping")

        global_mapping = column_mapping.get('global_forecast', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            year_column = st.text_input(
                "Year Column",
                value=global_mapping.get('year_column', "Year"),
                key="global_year_col"
            )

        with col2:
            value_column = st.text_input(
                "Value Column",
                value=global_mapping.get('value_column', "Value"),
                key="global_value_col"
            )

        with col3:
            type_column = st.text_input(
                "Type Column",
                value=global_mapping.get('type_column', "Type"),
                key="global_type_col"
            )

        # Update config data
        config_data['column_mapping']['global_forecast'] = {
            'year_column': year_column,
            'value_column': value_column,
            'type_column': type_column
        }
    
    # Tab 2: Country Historical Mapping
    with tab_country:
        st.subheader("Country Historical Column Mapping")
        
        country_mapping = column_mapping.get('country_historical', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            id_column = st.text_input(
                "ID Column",
                value=country_mapping.get('id_column', "idGeo"),
                key="country_id_col"
            )
        
        with col2:
            country_column = st.text_input(
                "Country Column",
                value=country_mapping.get('country_column', "Country"),
                key="country_name_col"
            )
        
        with col3:
            vertical_column = st.text_input(
                "Vertical Column",
                value=country_mapping.get('vertical_column', "nameVertical"),
                key="country_vertical_col"
            )
        
        # Update config data
        config_data['column_mapping']['country_historical'] = {
            'id_column': id_column,
            'country_column': country_column,
            'vertical_column': vertical_column
        }
    
    # Tab 3: Indicators Mapping
    with tab_indicators:
        st.subheader("Indicators Column Mapping")
        
        indicators_mapping = column_mapping.get('indicators', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            id_column = st.text_input(
                "ID Column",
                value=indicators_mapping.get('id_column', "idGeo"),
                key="indicator_id_col"
            )
        
        with col2:
            country_column = st.text_input(
                "Country Column",
                value=indicators_mapping.get('country_column', "Country"),
                key="indicator_country_col"
            )
        
        # Update config data
        config_data['column_mapping']['indicators'] = {
            'id_column': id_column,
            'country_column': country_column
        }
    
    return config_data


def render_market_distribution(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render market distribution section of the configuration interface.

    Args:
        config_data: Current configuration data

    Returns:
        Updated configuration data
    """
    st.header("Market Distribution Settings")

    # Initialize market_distribution if it doesn't exist
    if 'market_distribution' not in config_data:
        config_data['market_distribution'] = {}

    # Get current market distribution settings
    market_distribution = config_data.get('market_distribution', {})

    # Create tabs for different aspects of distribution
    tab_tiers, tab_growth, tab_smoothing, tab_redistribution = st.tabs([
        "Tier Configuration", "Growth Constraints", "Smoothing", "Redistribution"
    ])
    
    # Tab 1: Tier Configuration
    with tab_tiers:
        st.subheader("Market Tier Configuration")
        
        # Tier determination method
        tier_determination = st.radio(
            "Tier Determination Method",
            options=["auto", "manual"],
            index=0 if market_distribution.get('tier_determination', "auto") == "auto" else 1,
            horizontal=True,
            help="Method for determining market tiers: 'auto' uses K-means clustering, 'manual' uses defined thresholds"
        )
        
        # For manual tier determination
        if tier_determination == "manual":
            st.markdown("#### Manual Tier Settings")
            
            # Get manual tiers
            manual_tiers = market_distribution.get('manual_tiers', {})
            
            # Tier 1 settings
            st.markdown("##### Tier 1 (Market Leaders)")
            tier1_threshold = st.slider(
                "Share Threshold (%)",
                min_value=0.1,
                max_value=50.0,
                value=float(manual_tiers.get('tier_1', {}).get('share_threshold', 5.0)),
                step=0.1,
                help="Countries with market share >= this threshold are classified as Tier 1"
            )
            tier1_description = st.text_input(
                "Description",
                value=manual_tiers.get('tier_1', {}).get('description', "Market Leaders"),
                key="tier1_description"
            )
            
            # Tier 2 settings
            st.markdown("##### Tier 2 (Established Markets)")
            tier2_threshold = st.slider(
                "Share Threshold (%)",
                min_value=0.01,
                max_value=20.0,
                value=float(manual_tiers.get('tier_2', {}).get('share_threshold', 1.0)),
                step=0.1,
                help="Countries with market share >= this threshold but < Tier 1 threshold are classified as Tier 2"
            )
            tier2_description = st.text_input(
                "Description",
                value=manual_tiers.get('tier_2', {}).get('description', "Established Markets"),
                key="tier2_description"
            )
            
            # Tier 3 settings
            st.markdown("##### Tier 3 (Emerging Markets)")
            tier3_threshold = st.slider(
                "Share Threshold (%)",
                min_value=0.001,
                max_value=5.0,
                value=float(manual_tiers.get('tier_3', {}).get('share_threshold', 0.1)),
                step=0.01,
                help="Countries with market share >= this threshold but < Tier 2 threshold are classified as Tier 3"
            )
            tier3_description = st.text_input(
                "Description",
                value=manual_tiers.get('tier_3', {}).get('description', "Emerging Markets"),
                key="tier3_description"
            )
            
            # Update manual tiers
            market_distribution['manual_tiers'] = {
                'tier_1': {
                    'share_threshold': tier1_threshold,
                    'description': tier1_description
                },
                'tier_2': {
                    'share_threshold': tier2_threshold,
                    'description': tier2_description
                },
                'tier_3': {
                    'share_threshold': tier3_threshold,
                    'description': tier3_description
                }
            }
        else:
            # Auto tier determination
            st.markdown("#### K-means Clustering Parameters")
            
            # Get kmeans params
            kmeans_params = market_distribution.get('kmeans_params', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_clusters = st.slider(
                    "Minimum Clusters",
                    min_value=2,
                    max_value=6,
                    value=kmeans_params.get('min_clusters', 3),
                    help="Minimum number of clusters (tiers) to consider"
                )
            
            with col2:
                max_clusters = st.slider(
                    "Maximum Clusters",
                    min_value=3,
                    max_value=10,
                    value=kmeans_params.get('max_clusters', 8),
                    help="Maximum number of clusters (tiers) to consider"
                )
            
            # Update kmeans params
            market_distribution['kmeans_params'] = {
                'min_clusters': min_clusters,
                'max_clusters': max_clusters,
                'random_state': 42,
                'n_init': 10
            }
        
        # Update tier determination method
        market_distribution['tier_determination'] = tier_determination
    
    # Tab 2: Growth Constraints
    with tab_growth:
        st.subheader("Growth Constraints")
        
        # Get growth constraints
        growth_constraints = market_distribution.get('growth_constraints', {})
        
        # Determination method
        determination_method = st.radio(
            "Constraint Determination Method",
            options=["auto", "manual"],
            index=0 if growth_constraints.get('determination_method', "auto") == "auto" else 1,
            horizontal=True,
            help="Method for determining growth constraints: 'auto' calculates from historical data, 'manual' uses defined values"
        )
        
        # For manual constraint determination
        if determination_method == "manual":
            st.markdown("#### Manual Growth Constraints")
            
            # Get manual constraints
            manual_constraints = growth_constraints.get('manual_constraints', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_growth_rate = st.slider(
                    "Maximum Growth Rate (%)",
                    min_value=10.0,
                    max_value=100.0,
                    value=float(manual_constraints.get('max_growth_rate', 60)),
                    step=1.0,
                    help="Maximum allowed year-over-year growth rate"
                )
            
            with col2:
                min_growth_rate = st.slider(
                    "Minimum Growth Rate (%)",
                    min_value=-50.0,
                    max_value=0.0,
                    value=float(manual_constraints.get('min_growth_rate', -30)),
                    step=1.0,
                    help="Minimum allowed year-over-year growth rate (negative = decline)"
                )
            
            apply_scaling = st.checkbox(
                "Apply Scaling by Market Size",
                value=manual_constraints.get('apply_scaling_by_market_size', True),
                help="If enabled, smaller markets are allowed to grow faster than larger markets"
            )
            
            # Update manual constraints
            growth_constraints['manual_constraints'] = {
                'max_growth_rate': max_growth_rate,
                'min_growth_rate': min_growth_rate,
                'apply_scaling_by_market_size': apply_scaling
            }
        else:
            # Auto constraint determination
            st.info(
                "Growth constraints will be calculated automatically from historical data. "
                "This typically results in max growth of 40-80% and min growth of -20 to -40%, "
                "depending on historical volatility."
            )
        
        # Update determination method
        growth_constraints['determination_method'] = determination_method
        
        # Update growth constraints
        market_distribution['growth_constraints'] = growth_constraints
    
    # Tab 3: Smoothing
    with tab_smoothing:
        st.subheader("Smoothing Settings")
        
        # Get smoothing settings
        smoothing = market_distribution.get('smoothing', {})
        
        # Enable smoothing
        smoothing_enabled = st.checkbox(
            "Enable Smoothing",
            value=smoothing.get('enabled', True),
            help="Apply smoothing to ensure realistic growth patterns"
        )
        
        if smoothing_enabled:
            # Tier-specific smoothing
            st.markdown("#### Tier-Specific Smoothing Parameters")
            
            # Get tier smoothing
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
                tier1_max_growth = st.slider(
                    "Maximum Growth (%)",
                    min_value=10.0,
                    max_value=50.0,
                    value=float(tier1_smoothing.get('max_growth', 35)),
                    step=1.0,
                    key="smooth_t1_max",
                    help="Maximum growth rate after smoothing"
                )
            
            with col3:
                tier1_min_growth = st.slider(
                    "Minimum Growth (%)",
                    min_value=-30.0,
                    max_value=0.0,
                    value=float(tier1_smoothing.get('min_growth', -15)),
                    step=1.0,
                    key="smooth_t1_min",
                    help="Minimum growth rate after smoothing"
                )
            
            tier1_target = st.slider(
                "Target Growth Rate (%)",
                min_value=5.0,
                max_value=25.0,
                value=float(tier1_smoothing.get('target_growth', 15)),
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
                tier2_max_growth = st.slider(
                    "Maximum Growth (%)",
                    min_value=10.0,
                    max_value=60.0,
                    value=float(tier2_smoothing.get('max_growth', 40)),
                    step=1.0,
                    key="smooth_t2_max",
                    help="Maximum growth rate after smoothing"
                )
            
            with col3:
                tier2_min_growth = st.slider(
                    "Minimum Growth (%)",
                    min_value=-35.0,
                    max_value=0.0,
                    value=float(tier2_smoothing.get('min_growth', -20)),
                    step=1.0,
                    key="smooth_t2_min",
                    help="Minimum growth rate after smoothing"
                )
            
            tier2_target = st.slider(
                "Target Growth Rate (%)",
                min_value=10.0,
                max_value=30.0,
                value=float(tier2_smoothing.get('target_growth', 20)),
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
                tier3_max_growth = st.slider(
                    "Maximum Growth (%)",
                    min_value=15.0,
                    max_value=70.0,
                    value=float(tier3_smoothing.get('max_growth', 45)),
                    step=1.0,
                    key="smooth_t3_max",
                    help="Maximum growth rate after smoothing"
                )
            
            with col3:
                tier3_min_growth = st.slider(
                    "Minimum Growth (%)",
                    min_value=-40.0,
                    max_value=0.0,
                    value=float(tier3_smoothing.get('min_growth', -25)),
                    step=1.0,
                    key="smooth_t3_min",
                    help="Minimum growth rate after smoothing"
                )
            
            tier3_target = st.slider(
                "Target Growth Rate (%)",
                min_value=15.0,
                max_value=35.0,
                value=float(tier3_smoothing.get('target_growth', 25)),
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
            
            # Update tier-specific smoothing
            tier_smoothing = {
                'tier_1': {
                    'window': tier1_window,
                    'min_periods': 1,
                    'max_growth': tier1_max_growth,
                    'min_growth': tier1_min_growth,
                    'target_growth': tier1_target
                },
                'tier_2': {
                    'window': tier2_window,
                    'min_periods': 1,
                    'max_growth': tier2_max_growth,
                    'min_growth': tier2_min_growth,
                    'target_growth': tier2_target
                },
                'tier_3': {
                    'window': tier3_window,
                    'min_periods': 1,
                    'max_growth': tier3_max_growth,
                    'min_growth': tier3_min_growth,
                    'target_growth': tier3_target
                }
            }
            
            # Update smoothing settings
            smoothing['tier_smoothing'] = tier_smoothing
            smoothing['convergence_rate'] = convergence_rate
        
        # Update enabled status
        smoothing['enabled'] = smoothing_enabled
        
        # Update smoothing settings
        market_distribution['smoothing'] = smoothing
    
    # Tab 4: Redistribution
    with tab_redistribution:
        st.subheader("Redistribution Settings")
        
        # Enable redistribution
        enable_redistribution = st.checkbox(
            "Enable Redistribution from Specific Year",
            value='redistribution_start_year' in market_distribution,
            help="Preserve historical data exactly as-is before a specific year"
        )
        
        if enable_redistribution:
            redistribution_year = st.number_input(
                "Redistribution Start Year",
                min_value=2010,
                max_value=2025,
                value=int(market_distribution.get('redistribution_start_year', 2020)),
                help="Historical data before this year will be preserved exactly as-is"
            )
            
            # Add redistribution year to settings
            market_distribution['redistribution_start_year'] = redistribution_year
        elif 'redistribution_start_year' in market_distribution:
            # Remove redistribution year if disabled
            del market_distribution['redistribution_start_year']
    
    # Update market distribution settings
    config_data['market_distribution'] = market_distribution
    
    return config_data


def render_output_settings(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render output settings section of the configuration interface.

    Args:
        config_data: Current configuration data

    Returns:
        Updated configuration data
    """
    st.header("Output Settings")

    # Initialize output if it doesn't exist
    if 'output' not in config_data:
        config_data['output'] = {}

    # Get current output settings
    output_settings = config_data.get('output', {})
    
    # Save path
    save_path = st.text_input(
        "Output Directory",
        value=output_settings.get('save_path', "data/output/"),
        help="Directory where output files will be saved"
    )
    
    # Output formats
    available_formats = ["xlsx", "csv", "json"]
    selected_formats = st.multiselect(
        "Output Formats",
        options=available_formats,
        default=output_settings.get('formats', ["xlsx"]),
        help="File formats to generate for output data"
    )
    
    # Visualization settings
    st.subheader("Visualization Settings")
    
    # Get visualization settings
    visualizations = output_settings.get('visualizations', {})
    
    # Basic settings
    col1, col2 = st.columns(2)
    
    with col1:
        enable_visualizations = st.checkbox(
            "Enable Visualizations",
            value=visualizations.get('enabled', True),
            help="Generate visualization charts and figures"
        )
    
    with col2:
        save_format = st.selectbox(
            "Image Format",
            options=["png", "jpg", "svg", "pdf"],
            index=0 if visualizations.get('save_format', "png") == "png" else 
                  1 if visualizations.get('save_format', "png") == "jpg" else
                  2 if visualizations.get('save_format', "png") == "svg" else 3,
            help="Format for saving visualization images"
        )
    
    # DPI for raster formats
    if save_format in ["png", "jpg"]:
        dpi = st.slider(
            "Resolution (DPI)",
            min_value=72,
            max_value=600,
            value=visualizations.get('dpi', 300),
            step=72,
            help="Resolution for raster image formats (PNG, JPG)"
        )
    else:
        dpi = 300  # Default for vector formats
    
    # Visualization types
    st.subheader("Visualization Types")
    
    # Get visualization types
    vis_types = visualizations.get('types', [])
    
    # Market Size Visualization
    st.markdown("#### Market Size Visualization")
    include_market_size = st.checkbox(
        "Include Market Size Visualization",
        value=any(v.get('name') == 'market_size' for v in vis_types),
        key="include_market_size"
    )
    
    if include_market_size:
        market_size_settings = next((v for v in vis_types if v.get('name') == 'market_size'), {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_size_title = st.text_input(
                "Title Template",
                value=market_size_settings.get('title', "${market_type} Market Size by Country"),
                key="market_size_title"
            )
        
        with col2:
            top_n_countries = st.number_input(
                "Top N Countries",
                min_value=5,
                max_value=20,
                value=market_size_settings.get('top_n_countries', 10),
                key="market_size_top_n"
            )
        
        # Update or add market size visualization
        if any(v.get('name') == 'market_size' for v in vis_types):
            for i, v in enumerate(vis_types):
                if v.get('name') == 'market_size':
                    vis_types[i] = {
                        'name': 'market_size',
                        'title': market_size_title,
                        'top_n_countries': top_n_countries
                    }
                    break
        else:
            vis_types.append({
                'name': 'market_size',
                'title': market_size_title,
                'top_n_countries': top_n_countries
            })
    else:
        # Remove market size visualization if not included
        vis_types = [v for v in vis_types if v.get('name') != 'market_size']
    
    # Growth Rate Visualization
    st.markdown("#### Growth Rate Visualization")
    include_growth_rates = st.checkbox(
        "Include Growth Rate Visualization",
        value=any(v.get('name') == 'growth_rates' for v in vis_types),
        key="include_growth_rates"
    )
    
    if include_growth_rates:
        growth_rate_settings = next((v for v in vis_types if v.get('name') == 'growth_rates'), {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            growth_rate_title = st.text_input(
                "Title Template",
                value=growth_rate_settings.get('title', "${market_type} Growth Rate Analysis"),
                key="growth_rate_title"
            )
        
        with col2:
            growth_top_n = st.number_input(
                "Top N Countries",
                min_value=5,
                max_value=20,
                value=growth_rate_settings.get('top_n_countries', 10),
                key="growth_rate_top_n"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_yoy = st.checkbox(
                "Show Year-over-Year Growth",
                value=growth_rate_settings.get('show_yoy', True),
                key="show_yoy"
            )
        
        with col2:
            show_cagr = st.checkbox(
                "Show CAGR",
                value=growth_rate_settings.get('show_cagr', True),
                key="show_cagr"
            )
        
        # Update or add growth rate visualization
        if any(v.get('name') == 'growth_rates' for v in vis_types):
            for i, v in enumerate(vis_types):
                if v.get('name') == 'growth_rates':
                    vis_types[i] = {
                        'name': 'growth_rates',
                        'title': growth_rate_title,
                        'top_n_countries': growth_top_n,
                        'show_yoy': show_yoy,
                        'show_cagr': show_cagr
                    }
                    break
        else:
            vis_types.append({
                'name': 'growth_rates',
                'title': growth_rate_title,
                'top_n_countries': growth_top_n,
                'show_yoy': show_yoy,
                'show_cagr': show_cagr
            })
    else:
        # Remove growth rate visualization if not included
        vis_types = [v for v in vis_types if v.get('name') != 'growth_rates']
    
    # Market Share Visualization
    st.markdown("#### Market Share Visualization")
    include_market_share = st.checkbox(
        "Include Market Share Visualization",
        value=any(v.get('name') == 'market_share' for v in vis_types),
        key="include_market_share"
    )
    
    if include_market_share:
        market_share_settings = next((v for v in vis_types if v.get('name') == 'market_share'), {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_share_title = st.text_input(
                "Title Template",
                value=market_share_settings.get('title', "${market_type} Market Share Analysis"),
                key="market_share_title"
            )
        
        with col2:
            share_top_n = st.number_input(
                "Top N Countries",
                min_value=5,
                max_value=20,
                value=market_share_settings.get('top_n_countries', 10),
                key="market_share_top_n"
            )
        
        show_regional = st.checkbox(
            "Show Regional Breakdown",
            value=market_share_settings.get('show_regional', True),
            key="show_regional"
        )
        
        # Update or add market share visualization
        if any(v.get('name') == 'market_share' for v in vis_types):
            for i, v in enumerate(vis_types):
                if v.get('name') == 'market_share':
                    vis_types[i] = {
                        'name': 'market_share',
                        'title': market_share_title,
                        'top_n_countries': share_top_n,
                        'show_regional': show_regional
                    }
                    break
        else:
            vis_types.append({
                'name': 'market_share',
                'title': market_share_title,
                'top_n_countries': share_top_n,
                'show_regional': show_regional
            })
    else:
        # Remove market share visualization if not included
        vis_types = [v for v in vis_types if v.get('name') != 'market_share']
    
    # Update visualization settings
    visualizations['enabled'] = enable_visualizations
    visualizations['save_format'] = save_format
    visualizations['dpi'] = dpi
    visualizations['types'] = vis_types
    
    # Update output settings
    output_settings['save_path'] = save_path
    output_settings['formats'] = selected_formats
    output_settings['visualizations'] = visualizations
    
    # Update config data
    config_data['output'] = output_settings
    
    return config_data


def render_advanced_settings(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Render advanced settings section of the configuration interface.

    Args:
        config_data: Current configuration data

    Returns:
        Updated configuration data
    """
    st.header("Advanced Settings")

    # Weight calculation parameters
    st.subheader("Weight Calculation Parameters")

    # Initialize weight_params if it doesn't exist
    if 'weight_params' not in config_data:
        config_data['weight_params'] = {}

    # Get weight parameters
    weight_params = config_data.get('weight_params', {})
    
    # Transformation method
    transformation = st.selectbox(
        "Transformation Method",
        options=["log", "squared", "sigmoid", "linear"],
        index=0 if weight_params.get('transformation', "log") == "log" else
              1 if weight_params.get('transformation', "log") == "squared" else
              2 if weight_params.get('transformation', "log") == "sigmoid" else 3,
        help="Method for transforming correlation values"
    )
    
    # Parameters specific to each transformation
    if transformation == "log":
        col1, col2 = st.columns(2)
        
        with col1:
            log_base = st.number_input(
                "Log Base",
                min_value=1.1,
                max_value=10.0,
                value=float(weight_params.get('log_base', 2.0)),
                step=0.1,
                help="Base for logarithmic transformation"
            )
        
        with col2:
            log_scale = st.number_input(
                "Log Scale",
                min_value=0.1,
                max_value=10.0,
                value=float(weight_params.get('log_scale', 1.0)),
                step=0.1,
                help="Scaling factor for logarithmic transformation"
            )
    elif transformation == "sigmoid":
        sigmoid_steepness = st.slider(
            "Sigmoid Steepness",
            min_value=1.0,
            max_value=10.0,
            value=float(weight_params.get('sigmoid_steepness', 5.0)),
            step=0.1,
            help="Steepness parameter for sigmoid transformation"
        )
    
    # Significance method
    significance_method = st.radio(
        "Significance Method",
        options=["continuous", "stepped"],
        index=0 if weight_params.get('significance_method', "continuous") == "continuous" else 1,
        horizontal=True,
        help="Method for adjusting significance: 'continuous' uses smooth scaling, 'stepped' uses discrete thresholds"
    )
    
    # Filtering options
    st.markdown("#### Filtering Settings")
    
    # Get filtering options
    filtering = weight_params.get('filtering', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_correlation = st.slider(
            "Minimum Correlation",
            min_value=0.0,
            max_value=0.5,
            value=float(filtering.get('min_correlation', 0.1)),
            step=0.01,
            help="Minimum correlation value to consider (absolute value)"
        )
    
    with col2:
        max_p_value = st.slider(
            "Maximum p-value",
            min_value=0.01,
            max_value=0.2,
            value=float(filtering.get('max_p_value', 0.1)),
            step=0.01,
            help="Maximum p-value to consider significant"
        )
    
    with col3:
        min_coverage = st.slider(
            "Minimum Coverage",
            min_value=0.1,
            max_value=0.9,
            value=float(filtering.get('min_coverage', 0.3)),
            step=0.1,
            help="Minimum data coverage ratio required"
        )
    
    # Update filtering settings
    weight_params['filtering'] = {
        'min_correlation': min_correlation,
        'max_p_value': max_p_value,
        'min_coverage': min_coverage
    }
    
    # Update transformation-specific parameters
    weight_params['transformation'] = transformation
    
    if transformation == "log":
        weight_params['log_base'] = log_base
        weight_params['log_scale'] = log_scale
    elif transformation == "sigmoid":
        weight_params['sigmoid_steepness'] = sigmoid_steepness
    
    # Update significance method
    weight_params['significance_method'] = significance_method
    
    # Add visualization option
    weight_params['visualize_weights'] = st.checkbox(
        "Visualize Weights",
        value=weight_params.get('visualize_weights', True),
        help="Generate weight comparison charts"
    )
    
    # Update weight parameters
    config_data['weight_params'] = weight_params
    
    return config_data


def render_config_interface(config_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Render the complete configuration interface.
    
    Args:
        config_data: Current configuration data (optional)
        
    Returns:
        Updated configuration data
    """
    st.title("Configuration Settings")
    
    # Initialize configuration if not provided
    if config_data is None:
        config_data = {
            'project': {},
            'data_sources': {
                'global_forecast': {},
                'country_historical': {},
                'indicators': []
            },
            'column_mapping': {
                'global_forecast': {},
                'country_historical': {},
                'indicators': {}
            },
            'market_distribution': {},
            'output': {},
            'weight_params': {}
        }
    
    # Create tabs for different configuration sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Project", "Data Sources", "Column Mapping", 
        "Market Distribution", "Output", "Advanced"
    ])
    
    # Tab 1: Project Settings
    with tab1:
        config_data = render_project_settings(config_data)
    
    # Tab 2: Data Sources
    with tab2:
        config_data = render_data_sources(config_data)
    
    # Tab 3: Column Mapping
    with tab3:
        config_data = render_column_mapping(config_data)
    
    # Tab 4: Market Distribution
    with tab4:
        config_data = render_market_distribution(config_data)
    
    # Tab 5: Output Settings
    with tab5:
        config_data = render_output_settings(config_data)
    
    # Tab 6: Advanced Settings
    with tab6:
        config_data = render_advanced_settings(config_data)
    
    return config_data