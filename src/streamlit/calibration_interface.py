"""
Auto-Calibration Interface Module

This module provides components and utilities for configuring and executing
the auto-calibration system through the Streamlit interface.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime

from src.market_analysis.market_analyzer import MarketAnalyzer
from src.global_forecasting.auto_calibration import AutoCalibrator
from src.config.config_manager import ConfigurationManager

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_mock_accuracy_metrics() -> Dict[str, Any]:
    """Create mock accuracy metrics for demonstration purposes"""
    return {
        'overall': {
            'mape': 15.3,
            'rmse': 12.7,
            'r2': 0.847,
            'bias': -2.1
        },
        '1_year_back': {
            'aggregate': {
                'mape': 12.8,
                'rmse': 10.2,
                'r2': 0.891,
                'bias': -1.5
            },
            'countries': {
                'USA': {
                    'name': 'United States',
                    'mape': 8.4,
                    'rmse': 7.2,
                    'r2': 0.923,
                    'bias': -0.8,
                    'sample_size': 12
                },
                'CHN': {
                    'name': 'China',
                    'mape': 14.2,
                    'rmse': 11.8,
                    'r2': 0.876,
                    'bias': -2.1,
                    'sample_size': 12
                },
                'DEU': {
                    'name': 'Germany',
                    'mape': 11.7,
                    'rmse': 9.3,
                    'r2': 0.901,
                    'bias': -1.2,
                    'sample_size': 12
                },
                'JPN': {
                    'name': 'Japan',
                    'mape': 16.5,
                    'rmse': 13.4,
                    'r2': 0.834,
                    'bias': -2.8,
                    'sample_size': 12
                },
                'GBR': {
                    'name': 'United Kingdom',
                    'mape': 13.9,
                    'rmse': 11.1,
                    'r2': 0.867,
                    'bias': -1.9,
                    'sample_size': 12
                }
            }
        },
        '2_year_back': {
            'aggregate': {
                'mape': 18.6,
                'rmse': 15.4,
                'r2': 0.798,
                'bias': -2.9
            },
            'countries': {
                'USA': {
                    'name': 'United States',
                    'mape': 13.2,
                    'rmse': 10.8,
                    'r2': 0.887,
                    'bias': -1.4,
                    'sample_size': 24
                },
                'CHN': {
                    'name': 'China',
                    'mape': 21.7,
                    'rmse': 18.9,
                    'r2': 0.741,
                    'bias': -4.2,
                    'sample_size': 24
                },
                'DEU': {
                    'name': 'Germany',
                    'mape': 16.8,
                    'rmse': 13.7,
                    'r2': 0.823,
                    'bias': -2.3,
                    'sample_size': 24
                },
                'JPN': {
                    'name': 'Japan',
                    'mape': 22.1,
                    'rmse': 19.2,
                    'r2': 0.756,
                    'bias': -3.8,
                    'sample_size': 24
                },
                'GBR': {
                    'name': 'United Kingdom',
                    'mape': 19.4,
                    'rmse': 16.1,
                    'r2': 0.789,
                    'bias': -3.1,
                    'sample_size': 24
                }
            }
        }
    }


def render_calibration_interface(config_manager: ConfigurationManager, market_analyzer: Optional[MarketAnalyzer] = None) -> Dict[str, Any]:
    """
    Render auto-calibration configuration and evaluation interface.
    
    Args:
        config_manager: ConfigurationManager instance
        market_analyzer: Optional MarketAnalyzer instance
        
    Returns:
        Dictionary with updated calibration configuration
    """
    st.title("Auto-Calibrating Learning System")
    
    # Create market analyzer if not provided
    if market_analyzer is None and config_manager is not None:
        try:
            market_analyzer = MarketAnalyzer(config_manager)
        except Exception as e:
            st.error(f"Error creating market analyzer: {str(e)}")
            st.error("Please configure your project first.")
            return {}
    
    # Check if auto_calibrator is available
    if market_analyzer is None or not hasattr(market_analyzer, 'auto_calibrator'):
        st.error("Auto-calibrator component not available.")
        return {}
    
    # Get calibration configuration
    calibration_settings = market_analyzer.config_manager.get_value('market_distribution.calibration', {})
    
    # Create tabs for different aspects of calibration
    tab1, tab2, tab3, tab4 = st.tabs([
        "Calibration Settings", 
        "Evaluation Metrics", 
        "Component Performance", 
        "Country Adjustments"
    ])
    
    # Tab 1: Calibration Settings
    with tab1:
        st.header("Calibration Configuration")
        st.markdown("""
        The Auto-Calibrating Learning System continuously evaluates and automatically recalibrates 
        forecasting models based on historical accuracy, ensuring that the system learns from its 
        past performance and adapts to changing market conditions.
        """)
        
        # Enable auto-calibration
        enable_calibration = st.checkbox(
            "Enable Auto-Calibration",
            value=calibration_settings.get('enabled', True),
            help="Enable the Auto-Calibrating Learning System"
        )
        
        if not enable_calibration:
            return {'enabled': False}
        
        # Recalibration strategy
        st.subheader("Calibration Strategy")
        
        recalibration_strategy = st.radio(
            "Recalibration Strategy",
            options=["adaptive", "gradual", "aggressive"],
            index=0 if calibration_settings.get('recalibration_strategy', 'adaptive') == 'adaptive' else 
                 (1 if calibration_settings.get('recalibration_strategy', 'adaptive') == 'gradual' else 2),
            horizontal=True,
            help="How aggressively to adjust parameters based on performance"
        )
        
        # Strategy description
        if recalibration_strategy == "adaptive":
            st.info("""
            **Adaptive Strategy**: Adjusts parameters based on confidence level. Makes larger adjustments 
            when confidence is low and smaller adjustments when confidence is high.
            """)
        elif recalibration_strategy == "gradual":
            st.info("""
            **Gradual Strategy**: Makes smaller, consistent adjustments over time regardless of 
            confidence level. Best for stable forecasting environments.
            """)
        else:  # aggressive
            st.info("""
            **Aggressive Strategy**: Makes larger adjustments to quickly adapt to new patterns or 
            correct significant errors. Best when accuracy needs rapid improvement.
            """)
        
        # Learning rate
        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.5,
            value=float(calibration_settings.get('learning_rate', 0.15)),
            step=0.01,
            help="How quickly to adapt weights (higher = faster adaptation but potentially less stable)"
        )
        
        # Memory length
        memory_length = st.slider(
            "Memory Length",
            min_value=1,
            max_value=10,
            value=int(calibration_settings.get('memory_length', 5)),
            step=1,
            help="Number of past calibrations to consider (higher = more stable but slower to adapt)"
        )
        
        # Component weights
        st.subheader("Component Weights")
        st.markdown("Set the relative importance of each component for calibration:")
        
        # Get current component weights
        component_weights = calibration_settings.get('component_weights', {
            'tier_classification': 0.2,
            'causal_integration': 0.2,
            'gradient_harmonization': 0.15,
            'distribution_method': 0.3,
            'regional_aggregation': 0.15
        })
        
        # Allow adjusting component weights
        col1, col2 = st.columns(2)
        
        with col1:
            tier_weight = st.slider(
                "Tier Classification Weight",
                min_value=0.05,
                max_value=0.5,
                value=float(component_weights.get('tier_classification', 0.2)),
                step=0.05,
                help="Weight for tier classification component"
            )
            
            causal_weight = st.slider(
                "Causal Integration Weight",
                min_value=0.05,
                max_value=0.5,
                value=float(component_weights.get('causal_integration', 0.2)),
                step=0.05,
                help="Weight for causal indicator integration component"
            )
            
            gradient_weight = st.slider(
                "Gradient Harmonization Weight",
                min_value=0.05,
                max_value=0.5,
                value=float(component_weights.get('gradient_harmonization', 0.15)),
                step=0.05,
                help="Weight for gradient harmonization component"
            )
        
        with col2:
            distribution_weight = st.slider(
                "Distribution Method Weight",
                min_value=0.05,
                max_value=0.5,
                value=float(component_weights.get('distribution_method', 0.3)),
                step=0.05,
                help="Weight for distribution method component"
            )
            
            regional_weight = st.slider(
                "Regional Aggregation Weight",
                min_value=0.05,
                max_value=0.5,
                value=float(component_weights.get('regional_aggregation', 0.15)),
                step=0.05,
                help="Weight for regional aggregation component"
            )
        
        # Auto-adjust weights
        auto_adjust_weights = st.checkbox(
            "Auto-Adjust Component Weights",
            value=calibration_settings.get('auto_adjust_weights', True),
            help="Automatically adjust component weights based on performance"
        )
        
        # Apply country-specific adjustments
        apply_country_adjustments = st.checkbox(
            "Apply Country-Specific Adjustments",
            value=calibration_settings.get('apply_country_specific_adjustments', True),
            help="Apply targeted adjustments to countries with systematic forecast bias"
        )
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            # Confidence thresholds
            st.subheader("Confidence Thresholds")
            
            # Get current confidence thresholds
            confidence_thresholds = calibration_settings.get('confidence_thresholds', {
                'high': 0.85,
                'medium': 0.7,
                'low': 0.5
            })
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_confidence = st.slider(
                    "High Confidence Threshold",
                    min_value=0.7,
                    max_value=0.95,
                    value=float(confidence_thresholds.get('high', 0.85)),
                    step=0.05,
                    help="Threshold for high confidence (conservative adjustments)"
                )
            
            with col2:
                medium_confidence = st.slider(
                    "Medium Confidence Threshold",
                    min_value=0.5,
                    max_value=0.8,
                    value=float(confidence_thresholds.get('medium', 0.7)),
                    step=0.05,
                    help="Threshold for medium confidence (moderate adjustments)"
                )
            
            with col3:
                low_confidence = st.slider(
                    "Low Confidence Threshold",
                    min_value=0.3,
                    max_value=0.6,
                    value=float(confidence_thresholds.get('low', 0.5)),
                    step=0.05,
                    help="Threshold for low confidence (aggressive adjustments)"
                )
            
            # Evaluation periods
            st.subheader("Evaluation Periods")
            
            # Get current evaluation periods
            evaluation_periods = calibration_settings.get('evaluation_periods', [1, 3, 5])
            
            periods = []
            if st.checkbox("1-Year Evaluation", value=1 in evaluation_periods):
                periods.append(1)
            if st.checkbox("3-Year Evaluation", value=3 in evaluation_periods):
                periods.append(3)
            if st.checkbox("5-Year Evaluation", value=5 in evaluation_periods):
                periods.append(5)
            
            # Add custom period
            add_custom = st.checkbox("Add Custom Evaluation Period", value=False)
            if add_custom:
                custom_period = st.number_input(
                    "Custom Period (Years)",
                    min_value=2,
                    max_value=10,
                    value=7 if 7 in evaluation_periods else 2
                )
                if custom_period not in periods:
                    periods.append(custom_period)
            
            # Save calibration history
            save_history = st.checkbox(
                "Save Calibration History",
                value=calibration_settings.get('save_calibration_history', True),
                help="Save calibration history to disk"
            )
            
            # Backup frequency
            if save_history:
                backup_frequency = st.number_input(
                    "Backup Frequency",
                    min_value=1,
                    max_value=20,
                    value=int(calibration_settings.get('backup_frequency', 5)),
                    help="Create backup of calibration history every N calibrations"
                )
            else:
                backup_frequency = 5
        
        # Update component weights dictionary
        component_weights = {
            'tier_classification': tier_weight,
            'causal_integration': causal_weight,
            'gradient_harmonization': gradient_weight,
            'distribution_method': distribution_weight,
            'regional_aggregation': regional_weight
        }
        
        # Update confidence thresholds dictionary
        confidence_thresholds = {
            'high': high_confidence,
            'medium': medium_confidence,
            'low': low_confidence
        }
        
        # Create updated calibration settings
        updated_settings = {
            'enabled': enable_calibration,
            'recalibration_strategy': recalibration_strategy,
            'learning_rate': learning_rate,
            'memory_length': memory_length,
            'component_weights': component_weights,
            'auto_adjust_weights': auto_adjust_weights,
            'apply_country_specific_adjustments': apply_country_adjustments,
            'confidence_thresholds': confidence_thresholds,
            'evaluation_periods': periods,
            'save_calibration_history': save_history,
            'backup_frequency': backup_frequency
        }
    
    # Tab 2: Evaluation Metrics
    with tab2:
        st.header("Forecast Accuracy Evaluation")
        
        # Check if we have historical data for comparison
        if 'country_historical' not in st.session_state or st.session_state.country_historical is None:
            st.warning("No historical country data available for evaluation.")
            st.info("Please upload country historical data to enable forecast evaluation.")
            return updated_settings
        
        # Check if we have distributed market data
        if 'distributed_market' not in st.session_state or st.session_state.distributed_market is None:
            st.info("No market forecast available. Please generate a forecast first.")
            return updated_settings
        
        # Evaluation controls
        st.subheader("Evaluation Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics to evaluate
            st.markdown("**Metrics to Evaluate**")
            
            include_mape = st.checkbox("Mean Absolute Percentage Error (MAPE)", value=True)
            include_rmse = st.checkbox("Root Mean Squared Error (RMSE)", value=True)
            include_r2 = st.checkbox("R-squared (R²)", value=True)
            include_bias = st.checkbox("Systematic Bias", value=True)
        
        with col2:
            # Evaluation scope
            st.markdown("**Evaluation Scope**")
            
            eval_scope = st.radio(
                "Scope",
                options=["All Countries", "Top Countries", "Custom Countries"],
                index=0
            )
            
            if eval_scope == "Top Countries":
                top_n = st.number_input("Number of Top Countries", min_value=5, max_value=50, value=20)
            elif eval_scope == "Custom Countries":
                # Get countries from data
                country_col = 'Country'  # Default
                try:
                    column_mapping = market_analyzer.config_manager.get_column_mapping('country_historical')
                    country_col = column_mapping.get('country_column', 'Country')
                except Exception:
                    pass
                
                df = st.session_state.country_historical
                countries = sorted(df[country_col].unique())
                
                selected_countries = st.multiselect(
                    "Select Countries",
                    options=countries,
                    default=countries[:10] if len(countries) > 10 else countries
                )
        
        # Run evaluation
        if st.button("Evaluate Forecast Accuracy"):
            with st.spinner("Evaluating forecast accuracy..."):
                try:
                    # Get metric selection
                    metrics = []
                    if include_mape:
                        metrics.append('mape')
                    if include_rmse:
                        metrics.append('rmse')
                    if include_r2:
                        metrics.append('r2')
                    if include_bias:
                        metrics.append('bias')
                    
                    # Set up evaluation filters based on scope
                    eval_filters = {}
                    if eval_scope == "Top Countries":
                        eval_filters['top_n'] = top_n
                    elif eval_scope == "Custom Countries":
                        eval_filters['countries'] = selected_countries
                    
                    # Check if forecast data is available
                    forecast_available = False
                    debug_info = []
                    
                    # Check multiple possible locations for forecast data
                    if hasattr(market_analyzer, 'distributed_market') and market_analyzer.distributed_market is not None:
                        forecast_available = True
                        debug_info.append("✅ distributed_market found")
                    else:
                        debug_info.append("❌ distributed_market not found")
                    
                    if hasattr(market_analyzer, 'global_forecast') and market_analyzer.global_forecast is not None:
                        forecast_available = True
                        debug_info.append("✅ global_forecast found")
                    else:
                        debug_info.append("❌ global_forecast not found")
                    
                    if hasattr(market_analyzer, 'forecast_data') and market_analyzer.forecast_data is not None:
                        forecast_available = True
                        debug_info.append("✅ forecast_data found")
                    else:
                        debug_info.append("❌ forecast_data not found")
                    
                    # Check session state for forecast data
                    if 'forecast_results' in st.session_state and st.session_state.forecast_results is not None:
                        forecast_available = True
                        debug_info.append("✅ forecast_results in session_state found")
                    else:
                        debug_info.append("❌ forecast_results in session_state not found")
                    
                    if 'distributed_market' in st.session_state and st.session_state.distributed_market is not None:
                        forecast_available = True
                        debug_info.append("✅ distributed_market in session_state found")
                        # Try to set it on the market_analyzer if not already set
                        if not hasattr(market_analyzer, 'distributed_market') or market_analyzer.distributed_market is None:
                            market_analyzer.distributed_market = st.session_state.distributed_market
                            debug_info.append("✅ Set distributed_market on market_analyzer from session_state")
                    else:
                        debug_info.append("❌ distributed_market in session_state not found")
                    
                    if 'global_forecast' in st.session_state and st.session_state.global_forecast is not None:
                        forecast_available = True
                        debug_info.append("✅ global_forecast in session_state found")
                        # Try to set it on the market_analyzer if not already set
                        if not hasattr(market_analyzer, 'global_forecast') or market_analyzer.global_forecast is None:
                            market_analyzer.global_forecast = st.session_state.global_forecast
                            debug_info.append("✅ Set global_forecast on market_analyzer from session_state")
                    else:
                        debug_info.append("❌ global_forecast in session_state not found")
                    
                    if not forecast_available:
                        st.error("❌ No forecast data available for evaluation!")
                        
                        # Show debug information
                        with st.expander("🔍 Debug Information"):
                            st.write("Checked the following locations for forecast data:")
                            for info in debug_info:
                                st.write(f"- {info}")
                            
                            st.write("\n**Available market_analyzer attributes:**")
                            if hasattr(market_analyzer, '__dict__'):
                                for attr in sorted(dir(market_analyzer)):
                                    if not attr.startswith('_'):
                                        try:
                                            value = getattr(market_analyzer, attr)
                                            if value is not None:
                                                st.write(f"- {attr}: {type(value).__name__}")
                                        except:
                                            pass
                            
                            st.write("\n**Session state keys:**")
                            session_keys = [key for key in st.session_state.keys() if 'forecast' in key.lower() or 'market' in key.lower()]
                            if session_keys:
                                for key in sorted(session_keys):
                                    try:
                                        value = st.session_state[key]
                                        if value is not None:
                                            if hasattr(value, 'shape'):
                                                st.write(f"- {key}: {type(value).__name__} with shape {value.shape}")
                                            else:
                                                st.write(f"- {key}: {type(value).__name__}")
                                    except:
                                        st.write(f"- {key}: (error accessing)")
                            else:
                                st.write("No forecast or market related keys found in session state")
                        
                        st.markdown("""
                        **To evaluate forecast accuracy, you need to:**
                        1. Go to the **Global Forecasting** page
                        2. Generate a market forecast first
                        3. Return to this page to evaluate accuracy
                        """)
                        if st.button("🔮 Go to Global Forecasting"):
                            st.session_state.active_page = "Global Forecasting"
                            st.rerun()
                        return {}
                    
                    # Check if historical data is available
                    if not hasattr(market_analyzer, 'data_loader') or market_analyzer.data_loader is None:
                        st.error("❌ No data loader available!")
                        st.markdown("Please configure your project and load data first.")
                        return {}
                    
                    st.info("📊 Found forecast data. Evaluating accuracy...")
                    
                    # Check if auto-calibration is enabled
                    calibration_enabled = market_analyzer.config_manager.get_value('market_distribution.calibration.enabled', False)
                    if not calibration_enabled:
                        st.warning("⚠️ Auto-calibration is currently disabled in your configuration.")
                        if st.button("🔧 Enable Auto-Calibration"):
                            # Enable auto-calibration in the config
                            market_analyzer.config_manager.set_value('market_distribution.calibration.enabled', True)
                            market_analyzer.config_manager.set_value('market_distribution.calibration.evaluation_periods', [1, 2, 3])
                            market_analyzer.config_manager.set_value('market_distribution.calibration.metrics', ['mape', 'rmse', 'r2', 'bias'])
                            st.success("✅ Auto-calibration enabled! Click 'Evaluate Forecast Accuracy' again.")
                            st.rerun()
                    
                    # Debug information
                    with st.expander("🔍 Debug Information"):
                        forecast_shape = market_analyzer.distributed_market.shape if market_analyzer.distributed_market is not None else "None"
                        st.write(f"Forecast data shape: {forecast_shape}")
                        
                        try:
                            historical_data = market_analyzer.data_loader.load_country_historical()
                            st.write(f"Historical data shape: {historical_data.shape}")
                            st.write(f"Historical data columns: {list(historical_data.columns)}")
                            if len(historical_data) > 0:
                                # Check for date column (could be 'date', 'Year', or other)
                                date_col = None
                                for col in ['date', 'Date', 'year', 'Year', 'time', 'Time']:
                                    if col in historical_data.columns:
                                        date_col = col
                                        break
                                
                                if date_col:
                                    st.write(f"Historical data {date_col} range: {historical_data[date_col].min()} to {historical_data[date_col].max()}")
                                    
                                    # Show sample countries
                                    if 'Country' in historical_data.columns:
                                        sample_countries = historical_data['Country'].unique()[:5]
                                        st.write(f"Sample countries in historical data: {list(sample_countries)}")
                                    elif 'idGeo' in historical_data.columns:
                                        sample_countries = historical_data['idGeo'].unique()[:5]
                                        st.write(f"Sample country IDs in historical data: {list(sample_countries)}")
                                else:
                                    st.write("No date/year column found in historical data")
                            
                            # Check forecast data as well
                            if hasattr(market_analyzer, 'distributed_market') and market_analyzer.distributed_market is not None:
                                forecast_data = market_analyzer.distributed_market
                                st.write(f"Forecast data shape: {forecast_data.shape}")
                                st.write(f"Forecast data columns: {list(forecast_data.columns)}")
                                
                                if 'Year' in forecast_data.columns:
                                    forecast_years = sorted(forecast_data['Year'].unique())
                                    st.write(f"Forecast years: {forecast_years[:5]}...{forecast_years[-5:] if len(forecast_years) > 5 else ''}")
                                
                                if 'Country' in forecast_data.columns:
                                    sample_forecast_countries = forecast_data['Country'].unique()[:5]
                                    st.write(f"Sample countries in forecast data: {list(sample_forecast_countries)}")
                                elif 'idGeo' in forecast_data.columns:
                                    sample_forecast_countries = forecast_data['idGeo'].unique()[:5]
                                    st.write(f"Sample country IDs in forecast data: {list(sample_forecast_countries)}")
                        except Exception as e:
                            st.write(f"Historical data error: {str(e)}")
                    
                    # Run evaluation with error handling
                    try:
                        accuracy_metrics = market_analyzer.evaluate_forecast_accuracy()
                        
                        if not accuracy_metrics:
                            st.warning("⚠️ No accuracy metrics returned. This could be due to:")
                            st.markdown("""
                            - **Insufficient historical data**: Need overlapping years between forecast and historical data
                            - **Column name mismatch**: Historical data needs columns: Country, Year, Value (or idGeo, Year, Value)
                            - **Data format issues**: Check that data types and formats match expected structure
                            - **No common time periods**: Forecast and historical data must have overlapping years
                            """)
                            
                            # Offer to show mock results for demonstration
                            if st.button("📊 Show Mock Results for Testing"):
                                st.session_state.accuracy_metrics = create_mock_accuracy_metrics()
                                st.success("✅ Mock accuracy metrics loaded for demonstration!")
                                st.rerun()
                            
                            return {}
                        
                        # Store metrics in session state
                        st.session_state.accuracy_metrics = accuracy_metrics
                        
                        # Display success message
                        st.success("✅ Forecast accuracy evaluation completed!")
                        
                        # Display overall metrics
                        st.subheader("Overall Accuracy Metrics")
                        
                        if 'overall' in accuracy_metrics:
                            overall = accuracy_metrics['overall']
                            
                            metric_cols = st.columns(len(overall))
                            
                            for i, (metric, value) in enumerate(overall.items()):
                                with metric_cols[i]:
                                    if value is None or value == 'N/A':
                                        st.metric(metric.upper(), "N/A")
                                    elif metric == 'mape':
                                        st.metric("MAPE", f"{value:.2f}%")
                                    elif metric == 'rmse':
                                        st.metric("RMSE", f"{value:.2f}")
                                    elif metric == 'r2':
                                        st.metric("R²", f"{value:.4f}")
                                    elif metric == 'bias':
                                        st.metric("Bias", f"{value:.2f}%")
                                    else:
                                        st.metric(metric.upper(), f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                        
                        # Display period-specific metrics
                        st.subheader("Metrics by Evaluation Period")
                        
                        period_tabs = []
                        for period in accuracy_metrics:
                            if period != 'overall':
                                period_tabs.append(period)
                        
                        if period_tabs:
                            period_tab = st.radio("Evaluation Period", options=period_tabs, horizontal=True)
                            
                            if period_tab in accuracy_metrics:
                                period_data = accuracy_metrics[period_tab]
                                
                                # Display aggregate metrics for this period
                                st.markdown(f"**Aggregate Metrics for {period_tab}**")
                                
                                if 'aggregate' in period_data:
                                    agg_metrics = period_data['aggregate']
                                    
                                    agg_cols = st.columns(len(agg_metrics))
                                    
                                    for i, (metric, value) in enumerate(agg_metrics.items()):
                                        with agg_cols[i]:
                                            if value is None or value == 'N/A':
                                                st.metric(metric.upper(), "N/A")
                                            elif metric == 'mape':
                                                st.metric("MAPE", f"{value:.2f}%")
                                            elif metric == 'rmse':
                                                st.metric("RMSE", f"{value:.2f}")
                                            elif metric == 'r2':
                                                st.metric("R²", f"{value:.4f}")
                                            elif metric == 'bias':
                                                st.metric("Bias", f"{value:.2f}%")
                                            else:
                                                st.metric(metric.upper(), f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                                
                                # Display country-specific metrics
                                if 'countries' in period_data:
                                    st.markdown(f"**Country-Specific Metrics for {period_tab}**")
                                    
                                    # Convert to DataFrame for easier display
                                    countries_data = []
                                    for country_id, metrics in period_data['countries'].items():
                                        country_row = {
                                            'Country ID': country_id,
                                            'Country': metrics.get('name', 'Unknown')
                                        }
                                        
                                        # Add metrics
                                        for metric, value in metrics.items():
                                            if metric not in ['name', 'sample_size']:
                                                # Handle None values
                                                display_value = value if value is not None else 'N/A'
                                                if metric == 'mape':
                                                    country_row['MAPE (%)'] = display_value
                                                elif metric == 'rmse':
                                                    country_row['RMSE'] = display_value
                                                elif metric == 'r2':
                                                    country_row['R²'] = display_value
                                                elif metric == 'bias':
                                                    country_row['Bias (%)'] = display_value
                                        
                                        countries_data.append(country_row)
                                    
                                    # Create DataFrame
                                    if countries_data:
                                        countries_df = pd.DataFrame(countries_data)
                                        
                                        # Sort by MAPE (higher first)
                                        if 'MAPE (%)' in countries_df.columns:
                                            # Handle N/A values in sorting by converting them to a large number
                                            def sort_key(x):
                                                if x == 'N/A' or x is None:
                                                    return float('inf')
                                                try:
                                                    return float(x)
                                                except:
                                                    return float('inf')
                                            
                                            countries_df['_sort_key'] = countries_df['MAPE (%)'].apply(sort_key)
                                            countries_df = countries_df.sort_values(by='_sort_key', ascending=False)
                                            countries_df = countries_df.drop(columns=['_sort_key'])
                                        
                                        # Display as table
                                        st.dataframe(countries_df)
                        
                        # Plot country performance
                        st.subheader("Country Performance Visualization")
                        
                        # Find period with most country data
                        best_period = None
                        max_countries = 0
                        
                        for period, period_data in accuracy_metrics.items():
                            if period != 'overall' and 'countries' in period_data:
                                num_countries = len(period_data['countries'])
                                if num_countries > max_countries:
                                    max_countries = num_countries
                                    best_period = period
                        
                        if best_period:
                            period_data = accuracy_metrics[best_period]
                            
                            # Create plot data
                            plot_data = []
                            for country_id, metrics in period_data['countries'].items():
                                if 'name' in metrics and 'mape' in metrics and metrics['mape'] is not None:
                                    # Only include countries with valid MAPE values
                                    try:
                                        mape_value = float(metrics['mape'])
                                        plot_data.append({
                                            'Country': metrics['name'],
                                            'MAPE (%)': mape_value
                                        })
                                    except (ValueError, TypeError):
                                        # Skip countries with invalid MAPE values
                                        continue
                            
                            if plot_data:
                                plot_df = pd.DataFrame(plot_data)
                                
                                # Sort by MAPE
                                plot_df = plot_df.sort_values(by='MAPE (%)', ascending=True)
                                
                                # Limit to top 20 for better visualization
                                if len(plot_df) > 20:
                                    plot_df = plot_df.head(20)
                                
                                # Plot using Plotly
                                fig = px.bar(
                                    plot_df,
                                    y='Country',
                                    x='MAPE (%)',
                                    orientation='h',
                                    title=f"Forecast Error by Country ({best_period})",
                                    labels={'MAPE (%)': 'Mean Absolute Percentage Error (%)'},
                                    color='MAPE (%)',
                                    color_continuous_scale='Viridis'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error evaluating forecast accuracy: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                except Exception as e:
                    st.error(f"Error in forecast evaluation process: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Tab 3: Component Performance
    with tab3:
        st.header("Component Performance Analysis")
        
        # Check if we have calibration history
        if not hasattr(market_analyzer.auto_calibrator, 'calibration_history') or not market_analyzer.auto_calibrator.calibration_history:
            st.info("No calibration history available. Perform evaluation first.")
            return updated_settings
        
        # Display component performance if available
        if hasattr(market_analyzer.auto_calibrator, 'component_performance') and market_analyzer.auto_calibrator.component_performance:
            st.subheader("Component Weights")
            
            # Extract component weights
            component_weights_data = []
            for component, data in market_analyzer.auto_calibrator.component_performance.items():
                component_weights_data.append({
                    'Component': component.replace('_', ' ').title(),
                    'Weight': data['weight']
                })
            
            # Create DataFrame and sort by weight
            weights_df = pd.DataFrame(component_weights_data)
            weights_df = weights_df.sort_values(by='Weight', ascending=False)
            
            # Plot using Plotly
            fig = px.bar(
                weights_df,
                y='Component',
                x='Weight',
                orientation='h',
                title="Component Weights",
                labels={'Weight': 'Relative Weight'},
                color='Weight',
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Component impact analysis
            st.subheader("Component Impact Analysis")
            st.info("""
            Impact scores indicate each component's contribution to forecast error. Higher scores
            suggest the component may need adjustment to improve forecast accuracy.
            """)
            
            # Extract component impacts if available
            if hasattr(market_analyzer.auto_calibrator, 'component_impacts'):
                impact_data = []
                for component, impact in market_analyzer.auto_calibrator.component_impacts.items():
                    impact_data.append({
                        'Component': component.replace('_', ' ').title(),
                        'Impact Score': impact
                    })
                
                # Create DataFrame and sort by impact
                impact_df = pd.DataFrame(impact_data)
                impact_df = impact_df.sort_values(by='Impact Score', ascending=False)
                
                # Plot using Plotly
                fig = px.bar(
                    impact_df,
                    y='Component',
                    x='Impact Score',
                    orientation='h',
                    title="Component Impact on Forecast Error",
                    labels={'Impact Score': 'Error Contribution Score'},
                    color='Impact Score',
                    color_continuous_scale='Reds'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Calibration history
        st.subheader("Calibration History")
        
        # Extract metrics from calibration history
        calibration_history = market_analyzer.auto_calibrator.calibration_history
        
        if calibration_history:
            # Extract metrics over time
            history_data = []
            for calibration in calibration_history:
                if 'metrics' in calibration and 'overall' in calibration['metrics']:
                    history_row = {
                        'Calibration ID': calibration.get('calibration_id', 0),
                        'Date': calibration.get('date', '')
                    }
                    
                    # Add metrics
                    for metric, value in calibration['metrics']['overall'].items():
                        if metric == 'mape':
                            history_row['MAPE (%)'] = value
                        elif metric == 'rmse':
                            history_row['RMSE'] = value
                        elif metric == 'r2':
                            history_row['R²'] = value
                        elif metric == 'bias':
                            history_row['Bias (%)'] = value
                    
                    history_data.append(history_row)
            
            if history_data:
                # Create DataFrame
                history_df = pd.DataFrame(history_data)
                
                # Plot metrics over time
                if len(history_df) > 1:
                    st.markdown("**Forecast Accuracy Trend**")
                    
                    # Plot MAPE trend
                    if 'MAPE (%)' in history_df.columns:
                        fig = px.line(
                            history_df,
                            x='Calibration ID',
                            y='MAPE (%)',
                            title="MAPE Trend Over Calibrations",
                            markers=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot R² trend
                    if 'R²' in history_df.columns:
                        fig = px.line(
                            history_df,
                            x='Calibration ID',
                            y='R²',
                            title="R² Trend Over Calibrations",
                            markers=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Display calibration history table
                st.markdown("**Calibration History Details**")
                st.dataframe(history_df)
        
        # Perform new calibration
        st.subheader("Perform Model Calibration")
        st.markdown("""
        Calibrate forecasting models based on evaluation results. This will adjust component
        parameters and weights to improve future forecast accuracy.
        """)
        
        if st.button("Calibrate Models"):
            with st.spinner("Performing model calibration..."):
                try:
                    # Run calibration
                    calibration_report = market_analyzer.calibrate_models()
                    
                    if calibration_report:
                        # Store in session state
                        st.session_state.calibration_report = calibration_report
                        
                        # Display calibration results
                        st.success("Model calibration completed successfully!")
                        
                        # Calibration details
                        st.subheader("Calibration Details")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Calibration ID", calibration_report.get('calibration_id', 'N/A'))
                        
                        with col2:
                            confidence = calibration_report.get('confidence_score', 0)
                            st.metric("Confidence Score", f"{confidence:.2f}")
                        
                        with col3:
                            approach = calibration_report.get('approach', 'unknown')
                            st.metric("Calibration Approach", approach.title())
                        
                        # Parameter changes
                        if 'parameter_changes' in calibration_report and calibration_report['parameter_changes']:
                            st.subheader("Parameter Changes")
                            
                            param_changes = calibration_report['parameter_changes']
                            
                            for component, changes in param_changes.items():
                                with st.expander(f"{component.replace('_', ' ').title()} Changes"):
                                    # Iterate through parameters
                                    for param, values in changes.items():
                                        if isinstance(values, dict) and 'old' in values and 'new' in values:
                                            st.markdown(f"**{param}**")
                                            
                                            # Format values nicely
                                            if isinstance(values['old'], (list, tuple)) and isinstance(values['new'], (list, tuple)):
                                                # Show list values
                                                st.code(f"Old: {values['old']}\nNew: {values['new']}")
                                            elif isinstance(values['old'], dict) and isinstance(values['new'], dict):
                                                # Show dict values
                                                st.code(f"Old: {values['old']}\nNew: {values['new']}")
                                            else:
                                                # Show scalar values
                                                st.code(f"Old: {values['old']}\nNew: {values['new']}")
                        
                        # Weight changes
                        if 'weight_changes' in calibration_report and calibration_report['weight_changes']:
                            st.subheader("Weight Adjustments")
                            
                            weight_changes = calibration_report['weight_changes']
                            
                            # Create DataFrame for weight changes
                            weight_data = []
                            for component, changes in weight_changes.items():
                                weight_data.append({
                                    'Component': component.replace('_', ' ').title(),
                                    'Old Weight': changes.get('old', 0),
                                    'New Weight': changes.get('new', 0),
                                    'Change (%)': changes.get('change', 0) * 100
                                })
                            
                            # Create DataFrame
                            if weight_data:
                                weight_df = pd.DataFrame(weight_data)
                                
                                # Display as table
                                st.dataframe(weight_df)
                        
                        # Country adjustments
                        if ('country_adjustments' in calibration_report and 
                            calibration_report['country_adjustments']):
                            st.subheader("Country-Specific Adjustments")
                            
                            country_adjustments = calibration_report['country_adjustments']
                            
                            # Create DataFrame for country adjustments
                            country_data = []
                            for country_id, adjustment in country_adjustments.items():
                                country_data.append({
                                    'Country': adjustment.get('name', 'Unknown'),
                                    'Error (MAPE)': adjustment.get('error', 0),
                                    'Adjustment Strength': adjustment.get('adjustment_strength', 0)
                                })
                            
                            # Create DataFrame
                            if country_data:
                                country_df = pd.DataFrame(country_data)
                                
                                # Sort by error
                                country_df = country_df.sort_values(by='Error (MAPE)', ascending=False)
                                
                                # Display as table
                                st.dataframe(country_df)
                        
                        # Save calibration model
                        st.subheader("Save Calibration Model")
                        
                        if st.button("Save Calibration Model"):
                            with st.spinner("Saving calibration model..."):
                                try:
                                    model_path = market_analyzer.save_calibration_model()
                                    
                                    if model_path:
                                        st.success(f"Calibration model saved to: {model_path}")
                                except Exception as e:
                                    st.error(f"Error saving calibration model: {str(e)}")
                    else:
                        st.warning("No calibration performed. Make sure auto-calibration is enabled and evaluation has been run.")
                except Exception as e:
                    st.error(f"Error during model calibration: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    # Tab 4: Country Adjustments
    with tab4:
        st.header("Country-Specific Calibration")
        
        # Check if we have country performance data
        if (not hasattr(market_analyzer.auto_calibrator, 'country_performance') or 
            not market_analyzer.auto_calibrator.country_performance):
            st.info("No country performance data available. Perform evaluation first.")
            return updated_settings
        
        # Display country performance
        country_performance = market_analyzer.auto_calibrator.country_performance
        
        if country_performance:
            st.subheader("Country Performance Analysis")
            
            # Create DataFrame for country performance
            country_data = []
            for country_id, data in country_performance.items():
                if 'history' in data and data['history']:
                    # Calculate average metrics across history
                    mapes = [h.get('mape', 0) for h in data['history'] if 'mape' in h]
                    rmses = [h.get('rmse', 0) for h in data['history'] if 'rmse' in h]
                    r2s = [h.get('r2', 0) for h in data['history'] if 'r2' in h]
                    biases = [h.get('bias', 0) for h in data['history'] if 'bias' in h]
                    
                    avg_mape = np.mean(mapes) if mapes else 0
                    avg_rmse = np.mean(rmses) if rmses else 0
                    avg_r2 = np.mean(r2s) if r2s else 0
                    avg_bias = np.mean(biases) if biases else 0
                    
                    country_data.append({
                        'Country': data.get('name', 'Unknown'),
                        'Avg MAPE (%)': avg_mape,
                        'Avg RMSE': avg_rmse,
                        'Avg R²': avg_r2,
                        'Avg Bias (%)': avg_bias
                    })
            
            # Create DataFrame
            if country_data:
                country_df = pd.DataFrame(country_data)
                
                # Sort by MAPE
                country_df = country_df.sort_values(by='Avg MAPE (%)', ascending=False)
                
                # Display top countries with highest error
                st.markdown("**Top Countries with Highest Forecast Error**")
                st.dataframe(country_df.head(20))
                
                # Plot top 10 countries with highest error
                top_countries = country_df.head(10)
                
                fig = px.bar(
                    top_countries,
                    y='Country',
                    x='Avg MAPE (%)',
                    orientation='h',
                    title="Top 10 Countries with Highest Forecast Error",
                    labels={'Avg MAPE (%)': 'Average MAPE (%)'},
                    color='Avg MAPE (%)',
                    color_continuous_scale='Reds'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot bias distribution
                st.subheader("Forecast Bias Analysis")
                st.markdown("""
                Bias indicates systematic over-forecasting (positive values) or under-forecasting 
                (negative values). Countries with significant bias will receive specific adjustments.
                """)
                
                fig = px.scatter(
                    country_df,
                    x='Avg MAPE (%)',
                    y='Avg Bias (%)',
                    color='Avg MAPE (%)',
                    hover_name='Country',
                    title="Forecast Error vs. Bias",
                    labels={
                        'Avg MAPE (%)': 'Average MAPE (%)',
                        'Avg Bias (%)': 'Average Bias (%)'
                    },
                    color_continuous_scale='Viridis'
                )
                
                # Add zero line for bias
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Country-specific calibration controls
                st.subheader("Manual Country Adjustments")
                st.markdown("""
                Override automatic country-specific adjustments by setting manual adjustment factors.
                Positive values reduce forecasts, negative values increase forecasts.
                """)
                
                # Get current manual adjustments
                manual_adjustments = updated_settings.get('manual_country_adjustments', {})
                
                # Let user select countries to adjust
                with st.expander("Configure Manual Adjustments"):
                    # Select country
                    country_options = list(country_df['Country'])
                    selected_country = st.selectbox("Select Country", options=country_options)
                    
                    # Get country data
                    country_row = country_df[country_df['Country'] == selected_country].iloc[0]
                    
                    # Display country metrics
                    st.markdown(f"**Metrics for {selected_country}**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("MAPE", f"{country_row['Avg MAPE (%)']:.2f}%")
                    
                    with col2:
                        st.metric("RMSE", f"{country_row['Avg RMSE']:.2f}")
                    
                    with col3:
                        st.metric("Bias", f"{country_row['Avg Bias (%)']:.2f}%")
                    
                    # Adjustment controls
                    st.markdown("**Adjustment Settings**")
                    
                    # Get current adjustment for this country
                    current_adjustment = manual_adjustments.get(selected_country, 0)
                    
                    # Adjustment slider
                    adjustment_factor = st.slider(
                        "Adjustment Factor (%)",
                        min_value=-30.0,
                        max_value=30.0,
                        value=float(current_adjustment),
                        step=1.0,
                        help="Positive values reduce forecasts, negative values increase forecasts"
                    )
                    
                    # Apply adjustment button
                    if st.button("Apply Adjustment"):
                        # Update manual adjustments
                        manual_adjustments[selected_country] = adjustment_factor
                        
                        st.success(f"Manual adjustment of {adjustment_factor}% set for {selected_country}")
                
                # Display current manual adjustments
                if manual_adjustments:
                    st.markdown("**Current Manual Adjustments**")
                    
                    adjustment_data = []
                    for country, factor in manual_adjustments.items():
                        adjustment_data.append({
                            'Country': country,
                            'Adjustment Factor (%)': factor
                        })
                    
                    # Create DataFrame
                    adjustment_df = pd.DataFrame(adjustment_data)
                    
                    # Display as table
                    st.dataframe(adjustment_df)
                
                # Update settings with manual adjustments
                updated_settings['manual_country_adjustments'] = manual_adjustments
    
    # Return updated settings
    return updated_settings