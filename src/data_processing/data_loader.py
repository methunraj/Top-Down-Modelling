"""
Data Loader Module - Universal data loading functionality for market forecasting

This module provides functionality to load and preprocess market data from various 
file formats (Excel, CSV) in different structures, making it universally applicable
to any market type.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Universal data loader for market forecasting data
    
    This class provides functionality to load data from various file formats and structures,
    making it adaptable to any market type without hard-coded assumptions.
    """
    
    def __init__(self, config_manager):
        """
        Initialize the DataLoader
        
        Args:
            config_manager: Configuration manager instance for accessing settings
        """
        self.config_manager = config_manager
        self.data_cache = {}  # Cache for loaded data to avoid redundant loading
    
    def load_global_forecast(self) -> pd.DataFrame:
        """
        Load the global market forecast data
        
        Returns:
            DataFrame containing the global market forecast data
        """
        source_path = self.config_manager.get_data_source_path('global_forecast')
        column_mapping = self.config_manager.get_column_mapping('global_forecast')
        
        # Get column names from mapping
        year_col = column_mapping.get('year_column', 'Year')
        value_col = column_mapping.get('value_column', 'Value')
        type_col = column_mapping.get('type_column', 'Type')
        
        logger.info(f"Loading global forecast data from {source_path}")
        
        # Load data with dynamic column detection
        df = self._load_file(source_path)
        
        # Validate required columns
        required_cols = [year_col, value_col, type_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in global forecast data")
        
        # Ensure year is in the correct format
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        # Check for missing values in critical columns
        if df[year_col].isna().any() or df[value_col].isna().any():
            logger.warning("Missing values found in global forecast data critical columns")
        
        # Calculate growth rates if not present
        if 'Growth_Rate' not in df.columns:
            df = df.sort_values(by=year_col)
            df['Growth_Rate'] = df[value_col].pct_change() * 100
        
        # Cache the data
        self.data_cache['global_forecast'] = df
        
        return df
    
    def load_country_historical(self) -> pd.DataFrame:
        """
        Load the country historical market data
        
        Returns:
            DataFrame containing the country historical market data
        """
        source_path = self.config_manager.get_data_source_path('country_historical')
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        
        # Get column names from mapping
        id_col = column_mapping.get('id_column', 'idGeo')
        country_col = column_mapping.get('country_column', 'Country')
        vertical_col = column_mapping.get('vertical_column', 'nameVertical')
        
        logger.info(f"Loading country historical data from {source_path}")
        
        # Load data with dynamic column detection
        df = self._load_file(source_path)
        
        # Detect data structure (wide vs long format)
        is_wide_format = self._detect_wide_format(df, id_col, country_col)
        
        if is_wide_format:
            logger.info("Detected wide format for country historical data")
            # Transform to long format for consistent processing
            df = self._wide_to_long_format(df, id_col, country_col, vertical_col)
        else:
            logger.info("Detected long format for country historical data")
            # Validate required columns for long format
            required_cols = [id_col, country_col, 'Year', 'Value']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in country historical data")
        
        # Ensure numeric columns are properly formatted
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
        
        # Check for missing values in critical columns
        if df[id_col].isna().any() or df['Year'].isna().any() or df['Value'].isna().any():
            logger.warning("Missing values found in country historical data critical columns")
        
        # Cache the data
        self.data_cache['country_historical'] = df
        
        return df
    
    def load_indicator(self, indicator_name: str) -> pd.DataFrame:
        """
        Load an indicator dataset
        
        Args:
            indicator_name: Name of the indicator to load
            
        Returns:
            DataFrame containing the indicator data
        """
        # Check if already in cache
        cache_key = f"indicator_{indicator_name}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Find indicator config
        indicators = self.config_manager.get_indicators()
        indicator_config = None
        
        for indicator in indicators:
            if indicator.get('name') == indicator_name:
                indicator_config = indicator
                break
        
        if not indicator_config:
            raise ValueError(f"Indicator '{indicator_name}' not found in configuration")
        
        # Get file path
        source_path = indicator_config.get('path')
        if not source_path:
            raise ValueError(f"Path not specified for indicator '{indicator_name}'")
        
        logger.info(f"Loading indicator data for '{indicator_name}' from {source_path}")
        
        # Get column names from mapping
        column_mapping = self.config_manager.get_column_mapping('indicators')
        id_col = indicator_config.get('id_column') or column_mapping.get('id_column', 'idGeo')
        country_col = column_mapping.get('country_column', 'Country')
        
        # Load data with dynamic column detection
        df = self._load_file(source_path)
        
        # Detect data structure (wide vs long format)
        is_wide_format = self._detect_wide_format(df, id_col, country_col)
        
        if is_wide_format:
            logger.info(f"Detected wide format for indicator '{indicator_name}'")
            # Transform to long format for consistent processing
            df = self._wide_to_long_format(df, id_col, country_col)
        else:
            logger.info(f"Detected long format for indicator '{indicator_name}'")
            # Validate required columns for long format
            year_col = column_mapping.get('year_column', 'Year')
            value_col = column_mapping.get('value_column', 'Value')
            
            required_cols = [id_col, year_col, value_col]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in indicator '{indicator_name}'")
            
            # Standardize column names
            df = df.rename(columns={year_col: 'Year', value_col: 'Value'})
        
        # Ensure numeric columns are properly formatted
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
        
        # Add indicator name column for identification
        df['Indicator'] = indicator_name
        
        # Cache the data
        self.data_cache[cache_key] = df
        
        return df
    
    def load_all_indicators(self) -> pd.DataFrame:
        """
        Load all configured indicators and combine into a single DataFrame
        
        Returns:
            DataFrame containing all indicator data
        """
        indicators = self.config_manager.get_indicators()
        
        if not indicators:
            logger.warning("No indicators configured")
            return pd.DataFrame()
        
        all_indicator_dfs = []
        
        for indicator in indicators:
            indicator_name = indicator.get('name')
            try:
                df = self.load_indicator(indicator_name)
                all_indicator_dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading indicator '{indicator_name}': {str(e)}")
        
        if not all_indicator_dfs:
            logger.warning("No indicator data was successfully loaded")
            return pd.DataFrame()
        
        # Combine all indicator data
        combined_df = pd.concat(all_indicator_dfs, ignore_index=True)
        
        return combined_df
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a data file with format detection
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame containing the data
            
        Raises:
            ValueError: If the file format is not supported or file not found
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif file_ext == '.csv':
                return pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {str(e)}")
    
    def _detect_wide_format(self, df: pd.DataFrame, id_col: str, country_col: str) -> bool:
        """
        Detect if the data is in wide format (years as columns) or long format
        
        Args:
            df: DataFrame to check
            id_col: Name of the ID column
            country_col: Name of the country column
            
        Returns:
            True if wide format, False if long format
        """
        # Check if required columns exist
        if id_col not in df.columns or country_col not in df.columns:
            raise ValueError(f"Required columns {id_col} or {country_col} not found")
        
        # If 'Year' and 'Value' columns exist, it's likely long format
        if 'Year' in df.columns and 'Value' in df.columns:
            return False
        
        # Check for year columns (numeric column names)
        year_columns = []
        for col in df.columns:
            try:
                # Check if column name can be converted to an integer
                year = int(col)
                # Check if it looks like a year (typically 1900-2100)
                if 1900 <= year <= 2100:
                    year_columns.append(col)
            except (ValueError, TypeError):
                continue
        
        # If we have year columns, it's likely wide format
        return len(year_columns) > 0
    
    def _wide_to_long_format(self, df: pd.DataFrame, id_col: str, country_col: str, 
                             vertical_col: Optional[str] = None) -> pd.DataFrame:
        """
        Convert data from wide format (years as columns) to long format
        
        Args:
            df: DataFrame in wide format
            id_col: Name of the ID column
            country_col: Name of the country column
            vertical_col: Name of the vertical/market column (optional)
            
        Returns:
            DataFrame in long format with 'Year' and 'Value' columns
        """
        # Identify year columns
        year_columns = []
        for col in df.columns:
            try:
                year = int(col)
                if 1900 <= year <= 2100:
                    year_columns.append(col)
            except (ValueError, TypeError):
                continue
        
        if not year_columns:
            raise ValueError("No year columns found in data")
        
        # Determine non-year columns to keep
        id_vars = [id_col, country_col]
        if vertical_col and vertical_col in df.columns:
            id_vars.append(vertical_col)
        
        # Add any other non-year columns that might be important
        other_cols = [col for col in df.columns if col not in year_columns and col not in id_vars]
        id_vars.extend(other_cols)
        
        # Convert to long format
        long_df = pd.melt(
            df, 
            id_vars=id_vars, 
            value_vars=year_columns,
            var_name='Year',
            value_name='Value'
        )
        
        # Ensure Year is numeric
        long_df['Year'] = pd.to_numeric(long_df['Year'], errors='coerce')
        
        return long_df
    
    def get_historical_years(self) -> List[int]:
        """
        Get the list of historical years from the country historical data
        
        Returns:
            List of historical years in ascending order
        """
        if 'country_historical' not in self.data_cache:
            self.load_country_historical()
        
        df = self.data_cache['country_historical']
        years = sorted(df['Year'].unique())
        
        return years
    
    def get_forecast_years(self) -> List[int]:
        """
        Get the list of forecast years from the global forecast data
        
        Returns:
            List of forecast years in ascending order
        """
        if 'global_forecast' not in self.data_cache:
            self.load_global_forecast()
        
        df = self.data_cache['global_forecast']
        column_mapping = self.config_manager.get_column_mapping('global_forecast')
        
        year_col = column_mapping.get('year_column', 'Year')
        type_col = column_mapping.get('type_column', 'Type')
        
        # Get years where type is 'Forecast'
        forecast_df = df[df[type_col] == 'Forecast']
        years = sorted(forecast_df[year_col].unique())
        
        return years
    
    def get_countries(self) -> pd.DataFrame:
        """
        Get the list of countries with their IDs
        
        Returns:
            DataFrame with country IDs and names
        """
        if 'country_historical' not in self.data_cache:
            self.load_country_historical()
        
        df = self.data_cache['country_historical']
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        
        id_col = column_mapping.get('id_column', 'idGeo')
        country_col = column_mapping.get('country_column', 'Country')
        
        # Get unique country information
        countries_df = df[[id_col, country_col]].drop_duplicates().sort_values(by=country_col)
        
        return countries_df
    
    def get_latest_historical_data(self) -> pd.DataFrame:
        """
        Get the most recent historical data for all countries
        
        Returns:
            DataFrame with the latest historical data
        """
        if 'country_historical' not in self.data_cache:
            self.load_country_historical()
        
        df = self.data_cache['country_historical']
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        
        id_col = column_mapping.get('id_column', 'idGeo')
        
        # Get the latest year
        latest_year = df['Year'].max()
        
        # Filter for the latest year
        latest_df = df[df['Year'] == latest_year].copy()
        
        return latest_df


def create_sample_global_forecast() -> pd.DataFrame:
    """
    Create a sample global forecast dataset for testing
    
    Returns:
        Sample global forecast DataFrame
    """
    years = list(range(2020, 2032))
    values = [
        51963860168.127,
        126656404051.53,
        67189149798.755,
        53339629642.902,
        77160954396.945,
        103182643919.19,
        138064531811.77,
        184894553246.36,
        247583133177.67,
        331580309478.98,
        394166066694.94,
        446970682157.38
    ]
    
    types = ['Historical'] * 5 + ['Forecast'] * 7
    
    df = pd.DataFrame({
        'Year': years,
        'Value': values,
        'Type': types
    })
    
    # Calculate growth rates
    df['Growth_Rate'] = df['Value'].pct_change() * 100
    
    return df 