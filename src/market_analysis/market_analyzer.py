"""
Market Analyzer Module - Universal Market Forecasting Framework

This module provides the main coordination functionality for the Universal
Market Forecasting Framework, integrating the various components to generate
market forecasts in a market-agnostic way.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

from src.config.config_manager import ConfigurationManager, create_default_config
from src.data_processing.data_loader import DataLoader
from src.indicators.indicator_analyzer import IndicatorAnalyzer
from src.indicators.causal_indicator_integration import CausalIndicatorIntegration
from src.distribution.market_distributor import MarketDistributor
from src.global_forecasting.auto_calibration import AutoCalibrator
from src.visualization.market_visualizer import MarketVisualizer
from src.utils.math_utils import format_market_value

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """
    Universal market analyzer for any market type
    
    This class coordinates the various components of the Universal Market
    Forecasting Framework to generate market forecasts for any market type.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the MarketAnalyzer
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        # Initialize configuration manager
        self.config_manager = ConfigurationManager(config_path)
        
        # Get project info
        self.project_info = self.config_manager.get_project_info()
        self.market_type = self.project_info.get('market_type', 'Market')
        
        logger.info(f"Initializing Market Analyzer for {self.market_type} market")
        
        # Initialize components
        self.data_loader = DataLoader(self.config_manager)
        
        # Add session state indicators to config if they exist (for Streamlit)
        self._add_session_indicators_to_config()
        
        self.indicator_analyzer = IndicatorAnalyzer(self.config_manager, self.data_loader)
        self.causal_integration = CausalIndicatorIntegration(self.config_manager, self.data_loader, self.indicator_analyzer)
        self.market_distributor = MarketDistributor(self.config_manager, self.data_loader, self.indicator_analyzer)
        self.market_visualizer = MarketVisualizer(self.config_manager, self.data_loader)
        self.auto_calibrator = AutoCalibrator(self.config_manager, self.data_loader)
        
        # Initialize results storage
        self.distributed_market = None
        
        # Store selected forecasting model
        self.forecasting_model = None
    
    def set_forecasting_model(self, model_name: str) -> None:
        """
        Set the forecasting model to use for market analysis
        
        Args:
            model_name: Name of the forecasting model to use
        """
        # Validate model name
        from src.global_forecasting import FORECASTER_NAMES
        
        if model_name in FORECASTER_NAMES:
            self.forecasting_model = model_name
            logger.info(f"Set forecasting model to: {model_name}")
            
            # Update market distributor configuration
            if hasattr(self.market_distributor, 'distribution_settings'):
                if 'forecasting' not in self.market_distributor.distribution_settings:
                    self.market_distributor.distribution_settings['forecasting'] = {}
                
                self.market_distributor.distribution_settings['forecasting']['method'] = model_name
        else:
            valid_models = list(FORECASTER_NAMES.keys())
            logger.warning(f"Unknown forecasting model: {model_name}")
            logger.warning(f"Valid models: {', '.join(valid_models)}")
    
    def calculate_cagr(self, start_value: float, end_value: float, years: int) -> Optional[float]:
        """
        Calculate CAGR with proper error handling
        
        Args:
            start_value: Initial value
            end_value: Final value
            years: Number of years
            
        Returns:
            CAGR as a decimal (e.g., 0.05 for 5%) or None if calculation fails
        """
        # Validate inputs
        if years <= 0:
            logger.error(f"Invalid period for CAGR calculation: {years} years")
            return None
        
        if start_value <= 0:
            logger.warning(f"Cannot calculate CAGR with non-positive start value: {start_value}")
            return None
            
        if end_value < 0:
            logger.warning(f"Cannot calculate CAGR with negative end value: {end_value}")
            return None
        
        # Handle edge case where end value is zero
        if end_value == 0:
            logger.warning("End value is zero, returning -100% CAGR")
            return -1.0
        
        try:
            # Calculate CAGR
            cagr = (end_value / start_value) ** (1 / years) - 1
            
            # Sanity check the result
            if abs(cagr) > 10:  # More than 1000% growth/decline per year is suspicious
                logger.warning(f"Extreme CAGR calculated: {cagr:.2%}")
            
            return cagr
            
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            logger.error(f"CAGR calculation failed: {e}")
            return None
    
    def analyze_market(self) -> pd.DataFrame:
        """
        Analyze the market and generate forecast
        
        Returns:
            DataFrame with distributed market values
        """
        try:
            logger.info("Starting market analysis")
            
            # Load market data
            logger.info("Loading market data")
            global_forecast = self.data_loader.load_global_forecast()
            country_historical = self.data_loader.load_country_historical()
            
            # Apply forecasting model if specified
            if self.forecasting_model:
                logger.info(f"Using forecasting model: {self.forecasting_model}")
                # Update market distributor configuration
                if hasattr(self.market_distributor, 'distribution_settings'):
                    if 'forecasting' not in self.market_distributor.distribution_settings:
                        self.market_distributor.distribution_settings['forecasting'] = {}
                    
                    self.market_distributor.distribution_settings['forecasting']['method'] = self.forecasting_model
            
            # Analyze indicators
            logger.info("Analyzing market indicators")
            indicator_analysis = self.indicator_analyzer.analyze_indicators()
            
            # Perform causal analysis if enabled
            causal_enabled = self.config_manager.get_value('indicators.enable_causal_analysis', False)
            if causal_enabled:
                logger.info("Performing causal indicator analysis")
                causal_analysis = self.causal_integration.analyze_causal_relationships()
                
                # Apply causal adjustments to market distribution if configured
                apply_causal = self.config_manager.get_value('indicators.apply_causal_adjustments', False)
                if apply_causal:
                    self.market_distributor.set_causal_integration(self.causal_integration)
            
            # Distribute market values
            logger.info("Distributing market values")
            self.distributed_market = self.market_distributor.distribute_market()
            
            # Ensure consistent column naming
            if 'Value' in self.distributed_market.columns and 'market_value' not in self.distributed_market.columns:
                self.distributed_market = self.distributed_market.rename(columns={'Value': 'market_value'})
            
            # Ensure we have market_share column
            if 'market_share' not in self.distributed_market.columns and 'market_value' in self.distributed_market.columns:
                # Calculate market share by year
                self.distributed_market['market_share'] = 0.0
                for year in self.distributed_market['Year'].unique():
                    year_mask = self.distributed_market['Year'] == year
                    year_total = self.distributed_market.loc[year_mask, 'market_value'].sum()
                    if year_total > 0:
                        self.distributed_market.loc[year_mask, 'market_share'] = (
                            self.distributed_market.loc[year_mask, 'market_value'] / year_total * 100
                        )
                    else:
                        # Handle zero total by setting equal shares or zero shares
                        num_countries = year_mask.sum()
                        if num_countries > 0:
                            self.distributed_market.loc[year_mask, 'market_share'] = 0.0
            
            # Apply auto-calibration if enabled
            auto_calibration_enabled = self.config_manager.get_value('market_distribution.calibration.enabled', False)
            if auto_calibration_enabled:
                logger.info("Applying auto-calibration to market forecast")
                self.distributed_market = self.auto_calibrator.apply_auto_calibration(self.distributed_market)
            
            logger.info("Market analysis completed successfully")
            return self.distributed_market
            
        except Exception as e:
            logger.error(f"Error during market analysis: {str(e)}")
            raise
    
    def generate_visualizations(self) -> List[str]:
        """
        Generate visualizations for the distributed market
        
        Returns:
            List of paths to generated visualization files
        """
        if self.distributed_market is None:
            logger.warning("No market data available for visualization")
            return []
        
        logger.info("Generating market visualizations")
        
        # Generate standard visualizations
        visualization_files = self.market_visualizer.generate_all_visualizations(self.distributed_market)
        
        # Generate Excel report
        try:
            excel_file = self.market_visualizer.generate_excel_report(self.distributed_market)
            visualization_files.append(excel_file)
        except ImportError as e:
            logger.warning(f"Could not generate Excel report: {str(e)}")
            logger.warning("Install xlsxwriter or openpyxl package to enable Excel report generation")
        except Exception as e:
            logger.warning(f"Error generating Excel report: {str(e)}")
        
        logger.info(f"Generated {len(visualization_files)} visualization files")
        return visualization_files
    
    def save_results(self, output_formats: List[str] = None) -> Dict[str, str]:
        """
        Save market analysis results to files
        
        Args:
            output_formats: List of output formats (csv, excel, json)
            
        Returns:
            Dictionary with output file paths
        """
        if self.distributed_market is None:
            logger.warning("No market data available to save")
            return {}
        
        if output_formats is None:
            output_formats = ['csv', 'excel']
        
        output_files = {}
        
        # Get output directory from config
        output_dir = self.config_manager.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Save in each requested format
        for fmt in output_formats:
            if fmt.lower() == 'csv':
                # Save as CSV
                csv_path = os.path.join(output_dir, f'{self.market_type}_Market_Forecast.csv')
                self.distributed_market.to_csv(csv_path, index=False)
                output_files['csv'] = csv_path
                logger.info(f"Saved market forecast to CSV: {csv_path}")
                
            elif fmt.lower() == 'excel':
                # Save as Excel
                excel_path = os.path.join(output_dir, f'{self.market_type}_Market_Forecast.xlsx')
                
                # Create wide format version for Excel
                try:
                    # Get column names from mapping
                    country_mapping = self.config_manager.get_column_mapping('country_historical')
                    id_col = country_mapping.get('id_column', 'idGeo')
                    name_col = country_mapping.get('name_column', 'Country')
                    
                    # Determine value column
                    value_column = None
                    for col in ['market_value', 'Value', 'value']:
                        if col in self.distributed_market.columns:
                            value_column = col
                            break
                    
                    if value_column and 'Year' in self.distributed_market.columns:
                        # Create pivot table with years as columns
                        wide_data = self.distributed_market.pivot_table(
                            index=[id_col, name_col], 
                            columns='Year',
                            values=value_column,
                            aggfunc='sum'
                        )
                        
                        # Save to Excel directly
                        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                            wide_data.to_excel(writer, sheet_name='Market Forecast')
                            
                            # Also include original data
                            self.distributed_market.to_excel(writer, sheet_name='Raw Data', index=False)
                        
                        output_files['excel'] = excel_path
                        logger.info(f"Saved market forecast to Excel (wide format): {excel_path}")
                    else:
                        # Fallback to standard format
                        self.distributed_market.to_excel(excel_path, index=False)
                        output_files['excel'] = excel_path
                        logger.info(f"Saved market forecast to Excel: {excel_path}")
                except Exception as e:
                    logger.warning(f"Error creating wide format Excel: {str(e)}")
                    # Fallback to standard format
                    self.distributed_market.to_excel(excel_path, index=False)
                    output_files['excel'] = excel_path
                    logger.info(f"Saved market forecast to Excel: {excel_path}")
                
            elif fmt.lower() == 'json':
                # Save as JSON
                json_path = os.path.join(output_dir, f'{self.market_type}_Market_Forecast.json')
                self.distributed_market.to_json(json_path, orient='records', date_format='iso')
                output_files['json'] = json_path
                logger.info(f"Saved market forecast to JSON: {json_path}")
        
        return output_files
    
    def calculate_market_statistics(self) -> Dict[str, Any]:
        """
        Calculate key statistics about the market forecast
        
        Returns:
            Dictionary with market statistics
        """
        if self.distributed_market is None:
            logger.warning("No market data available for statistics")
            return {}
        
        try:
            df = self.distributed_market
            
            # Fixed: Get value column name from configuration
            value_column = self.config_manager.get_column_mapping('global_forecast').get('value_column', 'Value')
            
            # Validate that the value column exists
            if value_column not in df.columns:
                logger.warning(f"Value column '{value_column}' not found in data, using 'Value' as fallback")
                value_column = 'Value'
                if value_column not in df.columns:
                    logger.error("Neither configured value column nor 'Value' column found in data")
                    return {}
            
            # Get years
            years = sorted(df['Year'].unique())
            first_year = min(years)
            last_year = max(years)
            
            # Get global values for first and last year - Fixed to use dynamic column name
            first_year_total = df[df['Year'] == first_year][value_column].sum()
            last_year_total = df[df['Year'] == last_year][value_column].sum()
            
            # Calculate global CAGR using safe method
            years_diff = last_year - first_year
            global_cagr = self.calculate_cagr(first_year_total, last_year_total, years_diff)
            
            # If CAGR calculation failed, use a default
            if global_cagr is None:
                logger.warning("Global CAGR calculation failed, using 0.0 as default")
                global_cagr = 0.0
            
            # Get top 5 countries in the final year - Fixed to use dynamic column name
            last_year_data = df[df['Year'] == last_year].copy()
            top_countries = last_year_data.sort_values(by=value_column, ascending=False).head(5)
            
            # Create country statistics
            country_stats = []
            for _, row in top_countries.iterrows():
                country_id = row['idGeo']
                country_name = row['Country']
                
                # Get first year value for this country - Fixed to use dynamic column name
                first_value = df[(df['Year'] == first_year) & (df['idGeo'] == country_id)][value_column].values
                
                if len(first_value) > 0:
                    # Calculate country CAGR using safe method
                    country_cagr = self.calculate_cagr(first_value[0], row[value_column], years_diff)
                else:
                    logger.warning(f"No historical data for country {country_name} in year {first_year}")
                    country_cagr = None
                
                country_stats.append({
                    'id': country_id,
                    'name': country_name,
                    'final_value': row[value_column],
                    'final_share': row[value_column] / last_year_total * 100,
                    'cagr': country_cagr * 100 if country_cagr is not None else None
                })
            
            # Compile statistics
            statistics = {
                'market_type': self.market_type,
                'years': {
                    'first': first_year,
                    'last': last_year,
                    'period': years_diff
                },
                'global': {
                    'first_year_value': first_year_total,
                    'last_year_value': last_year_total,
                    'growth_multiplier': last_year_total / first_year_total if first_year_total != 0 else 0.0,
                    'cagr': global_cagr * 100
                },
                'top_countries': country_stats
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}
    
    def create_default_config_file(self, output_path: str) -> str:
        """
        Create a default configuration file
        
        Args:
            output_path: Path to save the configuration file
            
        Returns:
            Path to the created configuration file
        """
        try:
            # Generate default configuration
            default_config = create_default_config()
            
            # Determine file format from extension
            file_ext = os.path.splitext(output_path)[1].lower()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Save configuration file
            config_manager = ConfigurationManager()
            config_manager.config = default_config
            config_manager.save_config(output_path)
            
            logger.info(f"Created default configuration file: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating default configuration file: {str(e)}")
            raise
    
    def print_summary(self) -> None:
        """
        Print a summary of the market forecast
        """
        if self.distributed_market is None:
            logger.warning("No market forecast available for summary")
            print("No market forecast available. Run analyze_market() first.")
            return
        
        statistics = self.calculate_market_statistics()
        
        if not statistics:
            print("Error calculating market statistics")
            return
        
        print(f"\n{'='*80}")
        print(f"  {self.market_type} MARKET FORECAST SUMMARY")
        print(f"{'='*80}")
        
        # Market period
        years = statistics['years']
        print(f"\nForecast Period: {years['first']} to {years['last']} ({years['period']} years)")
        
        # Global values
        global_stats = statistics['global']
        print(f"\nGlobal Market Value:")
        print(f"  {years['first']}: {format_market_value(global_stats['first_year_value'])}")
        print(f"  {years['last']}: {format_market_value(global_stats['last_year_value'])}")
        print(f"  Growth Multiple: {global_stats['growth_multiplier']:.2f}x")
        print(f"  CAGR: {global_stats['cagr']:.2f}%")
        
        # Top countries
        print(f"\nTop 5 Countries in {years['last']}:")
        for i, country in enumerate(statistics['top_countries'], 1):
            print(f"  {i}. {country['name']}")
            print(f"     Value: {format_market_value(country['final_value'])}")
            print(f"     Share: {country['final_share']:.2f}%")
            if country['cagr'] is not None:
                print(f"     CAGR: {country['cagr']:.2f}%")
            else:
                print(f"     CAGR: N/A")
        
        print(f"\n{'='*80}")
    
    def save_wide_format(self, file_path: Optional[str] = None) -> str:
        """
        Save market data in wide format (years as columns)
        
        Args:
            file_path: Path to save the Excel file (optional)
            
        Returns:
            Path to the saved Excel file
        """
        if self.distributed_market is None:
            logger.warning("No market data available to save")
            return ""
        
        # Get output directory if file_path not provided
        if file_path is None:
            output_dir = self.config_manager.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f'{self.market_type}_Market_Forecast_Wide.xlsx')
        
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        name_col = country_mapping.get('name_column', 'Country')
        
        # Determine value column
        value_column = None
        for col in ['market_value', 'Value', 'value']:
            if col in self.distributed_market.columns:
                value_column = col
                break
        
        if not value_column:
            logger.warning("No value column found in market data")
            return ""
        
        try:
            # Create pivot table with years as columns
            wide_data = self.distributed_market.pivot_table(
                index=[id_col, name_col], 
                columns='Year',
                values=value_column,
                aggfunc='sum'
            )
            
            # Save to Excel
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                wide_data.to_excel(writer, sheet_name='Market Forecast')
                
                # Also save market share in wide format if available
                if 'market_share' in self.distributed_market.columns:
                    share_data = self.distributed_market.pivot_table(
                        index=[id_col, name_col], 
                        columns='Year',
                        values='market_share',
                        aggfunc='sum'
                    )
                    share_data.to_excel(writer, sheet_name='Market Share')
            
            logger.info(f"Saved market forecast in wide format: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving wide format: {str(e)}")
            return ""
            
    def evaluate_forecast_accuracy(self, historical_data: Optional[pd.DataFrame] = None, 
                               actual_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate the accuracy of the current forecast against actual data
        
        Args:
            historical_data: Optional historical market data used for forecasting
            actual_data: Optional actual market data to compare against forecast
            
        Returns:
            Dictionary with accuracy metrics
        """
        if self.distributed_market is None:
            logger.warning("No forecast available for evaluation")
            return {}
        
        logger.info("Evaluating forecast accuracy")
        
        # If historical data not provided, load from data_loader
        if historical_data is None:
            try:
                historical_data = self.data_loader.load_country_historical()
                logger.info(f"Loaded historical data with {len(historical_data)} records")
            except Exception as e:
                logger.warning(f"Could not load historical data: {str(e)}")
                historical_data = pd.DataFrame()
        
        # Use auto_calibrator to evaluate forecast accuracy
        metrics = self.auto_calibrator.evaluate_forecast_accuracy(
            historical_data=historical_data,
            forecast_data=self.distributed_market,
            actual_data=actual_data
        )
        
        return metrics
    
    def calibrate_models(self) -> Dict[str, Any]:
        """
        Perform auto-calibration of forecasting models based on accuracy evaluation
        
        Returns:
            Dictionary with calibration changes
        """
        # Check if auto-calibration is enabled
        calibration_enabled = self.config_manager.get_value('market_distribution.calibration.enabled', False)
        
        if not calibration_enabled:
            logger.info("Auto-calibration is disabled in configuration")
            return {}
        
        if not hasattr(self, 'auto_calibrator'):
            logger.warning("Auto-calibrator component not initialized")
            return {}
        
        logger.info("Performing model auto-calibration")
        
        # Use auto_calibrator to calibrate models
        calibration_report = self.auto_calibrator.calibrate_models(market_analyzer=self)
        
        # Log calibration results
        if calibration_report:
            confidence = calibration_report.get('confidence_score', 0)
            approach = calibration_report.get('approach', 'unknown')
            
            logger.info(f"Auto-calibration complete with confidence score: {confidence:.2f}")
            logger.info(f"Used {approach} calibration approach")
            
            # Log parameter changes
            param_changes = calibration_report.get('parameter_changes', {})
            if param_changes:
                for component, changes in param_changes.items():
                    logger.info(f"Applied calibration to {component} component")
        
        return calibration_report
    
    def save_calibration_model(self, file_path: Optional[str] = None) -> str:
        """
        Save the current calibration model to a file
        
        Args:
            file_path: Optional path to save the model file
            
        Returns:
            Path to the saved model file
        """
        if not hasattr(self, 'auto_calibrator'):
            logger.warning("Auto-calibrator component not initialized")
            return ""
        
        return self.auto_calibrator.save_calibration_model(file_path)
    
    def load_calibration_model(self, file_path: str) -> bool:
        """
        Load a calibration model from a file
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Boolean indicating success
        """
        # Fixed: Add validation for auto_calibrator existence and method availability
        if not hasattr(self, 'auto_calibrator') or self.auto_calibrator is None:
            logger.warning("Auto-calibrator component not initialized")
            return False
        
        if not hasattr(self.auto_calibrator, 'load_calibration_model'):
            logger.warning("Auto-calibrator does not support loading calibration models")
            return False
        
        try:
            return self.auto_calibrator.load_calibration_model(file_path)
        except Exception as e:
            logger.error(f"Error loading calibration model: {e}")
            return False

    def _add_session_indicators_to_config(self) -> None:
        """
        Add indicators from session state to the config manager (for Streamlit integration)
        """
        try:
            # Check if we're in a Streamlit environment and have session state indicators
            import streamlit as st
            
            logger.info("🔍 DEBUGGING: Checking for session state indicators...")
            logger.info(f"🔍 DEBUGGING: hasattr(st, 'session_state'): {hasattr(st, 'session_state')}")
            
            if hasattr(st, 'session_state'):
                logger.info(f"🔍 DEBUGGING: hasattr(st.session_state, 'indicators'): {hasattr(st.session_state, 'indicators')}")
                
                if hasattr(st.session_state, 'indicators'):
                    logger.info(f"🔍 DEBUGGING: st.session_state.indicators: {st.session_state.indicators}")
                    logger.info(f"🔍 DEBUGGING: st.session_state.indicators type: {type(st.session_state.indicators)}")
                    
                    if st.session_state.indicators:
                        logger.info(f"🔍 DEBUGGING: Found {len(st.session_state.indicators)} indicators in session state")
                        logger.info(f"🔍 DEBUGGING: Indicator names: {list(st.session_state.indicators.keys())}")
                        
                        # Get current config
                        current_config = self.config_manager.config
                        logger.info(f"🔍 DEBUGGING: Current config keys: {list(current_config.keys())}")
                        
                        # Ensure data_sources exists
                        if 'data_sources' not in current_config:
                            current_config['data_sources'] = {}
                            logger.info("🔍 DEBUGGING: Created data_sources in config")
                        
                        # Initialize indicators list if it doesn't exist
                        if 'indicators' not in current_config['data_sources']:
                            current_config['data_sources']['indicators'] = []
                            logger.info("🔍 DEBUGGING: Created indicators list in config")
                        
                        # Clear existing indicators to avoid duplicates
                        current_config['data_sources']['indicators'] = []
                        logger.info("🔍 DEBUGGING: Cleared existing indicators from config")
                        
                        # Add each session indicator to config
                        for indicator_name, indicator_data in st.session_state.indicators.items():
                            logger.info(f"🔍 DEBUGGING: Processing indicator: {indicator_name}")
                            logger.info(f"🔍 DEBUGGING: Indicator data: {indicator_data}")
                            
                            meta = indicator_data.get('meta', {})
                            logger.info(f"🔍 DEBUGGING: Indicator meta: {meta}")
                            
                            indicator_config = {
                                'name': indicator_name,
                                'type': meta.get('type', 'value'),
                                'weight': meta.get('weight', 1.0),
                                'direction': meta.get('direction', 'positive'),
                                'description': meta.get('description', f"Uploaded indicator: {indicator_name}")
                            }
                            
                            current_config['data_sources']['indicators'].append(indicator_config)
                            logger.info(f"🔍 DEBUGGING: Added indicator config: {indicator_config}")
                        
                        # Update the config manager's config
                        self.config_manager.config = current_config
                        
                        final_indicators = current_config['data_sources']['indicators']
                        logger.info(f"🔍 DEBUGGING: Final indicators in config: {final_indicators}")
                        logger.info(f"✅ Added {len(st.session_state.indicators)} indicators to config from session state")
                    else:
                        logger.info("🔍 DEBUGGING: st.session_state.indicators is empty or None")
                else:
                    logger.info("🔍 DEBUGGING: st.session_state.indicators does not exist")
            else:
                logger.info("🔍 DEBUGGING: st.session_state does not exist")
                
        except ImportError:
            # Not in Streamlit environment, skip
            logger.info("🔍 DEBUGGING: Not in Streamlit environment (ImportError)")
            pass
        except Exception as e:
            logger.error(f"🔍 DEBUGGING: Error in _add_session_indicators_to_config: {str(e)}")
            logger.exception("🔍 DEBUGGING: Full exception details:")


def create_market_analyzer(config_path: Optional[str] = None) -> MarketAnalyzer:
    """
    Create a MarketAnalyzer instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MarketAnalyzer instance
    """
    return MarketAnalyzer(config_path) 