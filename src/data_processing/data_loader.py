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
from src.utils.math_utils import validate_data_consistency, handle_outliers

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
        self.file_timestamps = {}  # Track file modification times for cache invalidation
    
    def _is_cache_valid(self, cache_key: str, file_path: str) -> bool:
        """
        Check if cached data is still valid based on file modification time
        
        Args:
            cache_key: Key used in cache
            file_path: Path to the source file
            
        Returns:
            True if cache is valid, False if file has been modified
        """
        if cache_key not in self.data_cache:
            return False
        
        if not os.path.exists(file_path):
            return False
        
        current_mtime = os.path.getmtime(file_path)
        cached_mtime = self.file_timestamps.get(cache_key)
        
        return cached_mtime is not None and current_mtime <= cached_mtime
    
    def _update_cache(self, cache_key: str, file_path: str, data: pd.DataFrame) -> None:
        """
        Update cache with new data and file timestamp
        
        Args:
            cache_key: Key to use in cache
            file_path: Path to the source file
            data: Data to cache
        """
        self.data_cache[cache_key] = data
        if os.path.exists(file_path):
            self.file_timestamps[cache_key] = os.path.getmtime(file_path)
    
    def clear_cache(self) -> None:
        """
        Clear all cached data - useful when user uploads new files
        """
        self.data_cache.clear()
        self.file_timestamps.clear()
        logger.info("Data cache cleared")
    
    def _validate_data(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate data for consistency and quality issues.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            Tuple of (validated_dataframe, list_of_warnings)
        """
        warnings = []
        df_validated = df.copy()
        
        # Check for empty dataframe
        if df_validated.empty:
            warnings.append(f"{data_type}: DataFrame is empty")
            return df_validated, warnings
        
        # Check for duplicate rows
        duplicates = df_validated.duplicated().sum()
        if duplicates > 0:
            warnings.append(f"{data_type}: Found {duplicates} duplicate rows, removing")
            df_validated = df_validated.drop_duplicates()
        
        # Type-specific validation
        if data_type == 'global_forecast':
            # Validate year and value columns
            if 'Year' in df_validated.columns:
                # Check for missing years
                years = df_validated['Year'].dropna().unique()
                if len(years) > 1:
                    min_year, max_year = int(min(years)), int(max(years))
                    expected_years = set(range(min_year, max_year + 1))
                    missing_years = expected_years - set(years)
                    if missing_years:
                        warnings.append(f"Missing years in global forecast: {sorted(missing_years)}")
                
                # Check for duplicate years per type
                if 'Type' in df_validated.columns:
                    for type_val in df_validated['Type'].unique():
                        type_data = df_validated[df_validated['Type'] == type_val]
                        year_counts = type_data['Year'].value_counts()
                        duplicated_years = year_counts[year_counts > 1].index.tolist()
                        if duplicated_years:
                            warnings.append(f"Duplicate entries for years {duplicated_years} in {type_val} data")
            
            # Validate values
            if 'Value' in df_validated.columns:
                # Check for negative values (might be invalid for some markets)
                neg_values = (df_validated['Value'] < 0).sum()
                if neg_values > 0:
                    warnings.append(f"Found {neg_values} negative values in global forecast")
                
                # Handle outliers
                df_validated['Value'] = handle_outliers(
                    df_validated['Value'],
                    method='iqr',
                    threshold=3.0,
                    replace_with='clip'
                )
        
        elif data_type == 'country_historical':
            # Check for missing country identifiers
            if 'idGeo' in df_validated.columns:
                missing_ids = df_validated['idGeo'].isna().sum()
                if missing_ids > 0:
                    warnings.append(f"Found {missing_ids} rows with missing country IDs")
                    df_validated = df_validated.dropna(subset=['idGeo'])
            
            # Validate values per country/year combination
            if all(col in df_validated.columns for col in ['idGeo', 'Year', 'Value']):
                # Check for duplicate country/year combinations
                duplicates = df_validated.groupby(['idGeo', 'Year']).size()
                multi_entries = duplicates[duplicates > 1]
                if not multi_entries.empty:
                    warnings.append(f"Found {len(multi_entries)} duplicate country/year combinations")
                    # Keep only the first occurrence
                    df_validated = df_validated.drop_duplicates(subset=['idGeo', 'Year'], keep='first')
                
                # Check for countries with very few data points
                country_counts = df_validated.groupby('idGeo').size()
                sparse_countries = country_counts[country_counts < 3].index.tolist()
                if sparse_countries:
                    warnings.append(f"{len(sparse_countries)} countries have less than 3 data points")
        
        # General value validation
        value_columns = [col for col in df_validated.columns if 'value' in col.lower()]
        for col in value_columns:
            if col in df_validated.columns:
                # Check for infinite values
                inf_values = np.isinf(df_validated[col]).sum()
                if inf_values > 0:
                    warnings.append(f"Found {inf_values} infinite values in {col}, replacing with NaN")
                    df_validated[col] = df_validated[col].replace([np.inf, -np.inf], np.nan)
                
                # Check for very large values that might indicate data errors
                if df_validated[col].notna().any():
                    max_val = df_validated[col].max()
                    min_val = df_validated[col].min()
                    if max_val > 1e15:  # Trillion scale check
                        warnings.append(f"Very large values detected in {col}: max={max_val:,.0f}")
                    if min_val < -1e15:
                        warnings.append(f"Very large negative values in {col}: min={min_val:,.0f}")
        
        return df_validated, warnings
    
    def _standardize_column_names(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """
        Standardize column names to expected format
        
        Args:
            df: DataFrame to standardize
            data_type: Type of data ('global_forecast', 'country_historical', 'indicators')
            
        Returns:
            DataFrame with standardized column names
        """
        df_copy = df.copy()
        
        if data_type == 'global_forecast':
            # Map common year column names
            year_patterns = ['year', 'yr', 'date', 'time_period', 'period']
            for col in df_copy.columns:
                if col.lower() in year_patterns or 'year' in col.lower():
                    if col != 'Year':
                        df_copy = df_copy.rename(columns={col: 'Year'})
                        logger.info(f"Mapped column '{col}' to 'Year'")
                    break
            
            # Map common value column names
            value_patterns = ['value', 'amount', 'total', 'market_size', 'revenue', 'sales']
            for col in df_copy.columns:
                if col.lower() in value_patterns or 'value' in col.lower():
                    if col != 'Value':
                        df_copy = df_copy.rename(columns={col: 'Value'})
                        logger.info(f"Mapped column '{col}' to 'Value'")
                    break
            
            # Map common type column names
            type_patterns = ['type', 'category', 'forecast_type', 'data_type']
            for col in df_copy.columns:
                if col.lower() in type_patterns or 'type' in col.lower():
                    if col != 'Type':
                        df_copy = df_copy.rename(columns={col: 'Type'})
                        logger.info(f"Mapped column '{col}' to 'Type'")
                    break
                    
        elif data_type == 'country_historical':
            # Map year column
            year_patterns = ['year', 'yr', 'date', 'time_period', 'period']
            for col in df_copy.columns:
                if col.lower() in year_patterns or 'year' in col.lower():
                    if col != 'Year':
                        df_copy = df_copy.rename(columns={col: 'Year'})
                        logger.info(f"Mapped column '{col}' to 'Year'")
                    break
            
            # Map value column
            value_patterns = ['value', 'amount', 'total', 'market_size', 'revenue', 'sales']
            for col in df_copy.columns:
                if col.lower() in value_patterns or 'value' in col.lower():
                    if col != 'Value':
                        df_copy = df_copy.rename(columns={col: 'Value'})
                        logger.info(f"Mapped column '{col}' to 'Value'")
                    break
            
            # Map country column
            country_patterns = ['country', 'nation', 'country_name', 'region']
            for col in df_copy.columns:
                if col.lower() in country_patterns or 'country' in col.lower():
                    if col != 'Country':
                        df_copy = df_copy.rename(columns={col: 'Country'})
                        logger.info(f"Mapped column '{col}' to 'Country'")
                    break
            
            # Map ID column
            id_patterns = ['id', 'country_id', 'geo_id', 'code', 'country_code']
            for col in df_copy.columns:
                if col.lower() in id_patterns or 'id' in col.lower():
                    if col != 'idGeo':
                        df_copy = df_copy.rename(columns={col: 'idGeo'})
                        logger.info(f"Mapped column '{col}' to 'idGeo'")
                    break
                    
        elif data_type == 'indicators':
            # Similar to country_historical but may have different patterns
            # Map year column
            year_patterns = ['year', 'yr', 'date', 'time_period', 'period']
            for col in df_copy.columns:
                if col.lower() in year_patterns or 'year' in col.lower():
                    if col != 'Year':
                        df_copy = df_copy.rename(columns={col: 'Year'})
                        logger.info(f"Mapped column '{col}' to 'Year'")
                    break
            
            # Map value column
            value_patterns = ['value', 'amount', 'total', 'indicator_value', 'score']
            for col in df_copy.columns:
                if col.lower() in value_patterns or 'value' in col.lower():
                    if col != 'Value':
                        df_copy = df_copy.rename(columns={col: 'Value'})
                        logger.info(f"Mapped column '{col}' to 'Value'")
                    break
            
            # Map country column
            country_patterns = ['country', 'nation', 'country_name', 'region']
            for col in df_copy.columns:
                if col.lower() in country_patterns or 'country' in col.lower():
                    if col != 'Country':
                        df_copy = df_copy.rename(columns={col: 'Country'})
                        logger.info(f"Mapped column '{col}' to 'Country'")
                    break
            
            # Map ID column
            id_patterns = ['id', 'country_id', 'geo_id', 'code', 'country_code']
            for col in df_copy.columns:
                if col.lower() in id_patterns or 'id' in col.lower():
                    if col != 'idGeo':
                        df_copy = df_copy.rename(columns={col: 'idGeo'})
                        logger.info(f"Mapped column '{col}' to 'idGeo'")
                    break
        
        return df_copy
    
    def _ensure_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide format data to long format if needed
        
        Args:
            df: DataFrame that might be in wide format
            
        Returns:
            DataFrame in long format
        """
        # Detect if data is in wide format (years as columns)
        year_columns = []
        for col in df.columns:
            # Check if column name looks like a year
            if isinstance(col, (int, float)) or (isinstance(col, str) and col.isdigit()):
                year = int(col)
                if 1900 <= year <= 2100:
                    year_columns.append(col)
        
        # If we found year columns, it's wide format - convert to long
        if len(year_columns) >= 2:
            logger.info(f"Detected wide format data with year columns: {year_columns}")
            
            # Identify ID and metadata columns (non-year columns)
            id_columns = [col for col in df.columns if col not in year_columns]
            
            try:
                # Melt the DataFrame to convert from wide to long
                df_long = pd.melt(
                    df, 
                    id_vars=id_columns,
                    value_vars=year_columns,
                    var_name='Year',
                    value_name='Value'
                )
                
                # Convert Year column to numeric
                df_long['Year'] = pd.to_numeric(df_long['Year'])
                
                # Remove rows with NaN values
                df_long = df_long.dropna(subset=['Value'])
                
                logger.info(f"Converted wide format to long format: {df.shape} -> {df_long.shape}")
                return df_long
                
            except Exception as e:
                logger.error(f"Failed to convert wide format to long format: {e}")
                logger.warning("Returning original data")
                return df
        
        # Data is already in long format or not convertible
        return df
    
    def load_global_forecast(self) -> pd.DataFrame:
        """
        Load the global market forecast data
        
        Returns:
            DataFrame containing the global market forecast data
        """
        # First, check if there's uploaded data in session state (for Streamlit)
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'global_forecast') and st.session_state.global_forecast is not None:
                logger.info("Using uploaded global forecast data from session state")
                df = st.session_state.global_forecast.copy()
                
                # First, try to standardize column names
                df = self._standardize_column_names(df, 'global_forecast')
                
                # Check if data is in wide format and convert to long (for global data this is less common but possible)
                df = self._ensure_long_format(df)
                
                # Apply same validation and standardization as file loading
                column_mapping = self.config_manager.get_column_mapping('global_forecast')
                year_col = column_mapping.get('year_column', 'Year')
                value_col = column_mapping.get('value_column', 'Value')
                type_col = column_mapping.get('type_column', 'Type')
                
                # Apply comprehensive validation
                df, warnings = self._validate_data(df, 'global_forecast')
                for warning in warnings:
                    logger.warning(f"Data validation: {warning}")
                
                # Apply old validation if we have the required columns
                if year_col in df.columns and value_col in df.columns:
                    try:
                        self._validate_global_forecast_data(df, year_col, value_col, type_col)
                    except ValueError as e:
                        logger.warning(f"Validation warning for uploaded data: {e}")
                    
                    # Ensure year is in the correct format
                    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
                    
                    # CRITICAL FIX: Ensure Type column exists and is populated
                    if type_col not in df.columns or df[type_col].isna().all():
                        logger.warning(f"Type column '{type_col}' missing or empty. Auto-inferring from forecast horizon...")
                        # Try to get forecast horizon from session state
                        forecast_horizon_year = None
                        try:
                            if hasattr(st.session_state, 'forecast_horizon_year'):
                                forecast_horizon_year = st.session_state.forecast_horizon_year
                                logger.info(f"Using forecast horizon year from session state: {forecast_horizon_year}")
                        except:
                            pass
                        
                        if forecast_horizon_year is not None:
                            # Set Type based on forecast horizon
                            df[type_col] = df[year_col].apply(
                                lambda y: 'Historical' if y < forecast_horizon_year else 'Forecast'
                            )
                            logger.info(f"Set Type column based on forecast horizon year {forecast_horizon_year}")
                        else:
                            # Fallback: assume all data is forecast if no horizon specified
                            logger.warning("No forecast horizon found. Marking all data as 'Forecast'")
                            df[type_col] = 'Forecast'
                    
                    # Calculate growth rates if not present
                    if 'Growth_Rate' not in df.columns:
                        df = df.sort_values(by=year_col)
                        # Calculate percentage change with proper handling of zeros and NaNs
                        prev_values = df[value_col].shift(1)
                        df['Growth_Rate'] = np.where(
                            (prev_values != 0) & (~pd.isna(prev_values)) & (~pd.isna(df[value_col])),
                            ((df[value_col] - prev_values) / prev_values) * 100,
                            0.0
                        )
                else:
                    logger.warning(f"Missing required columns in uploaded global forecast data. Found columns: {list(df.columns)}")
                
                return df
        except (ImportError, AttributeError):
            # Not in Streamlit context, continue with normal file loading
            pass
        
        source_path = self.config_manager.get_data_source_path('global_forecast')
        column_mapping = self.config_manager.get_column_mapping('global_forecast')
        
        # Check if cached data is still valid
        cache_key = 'global_forecast'
        if self._is_cache_valid(cache_key, source_path):
            logger.info(f"Using cached global forecast data")
            return self.data_cache[cache_key]
        
        # Get column names from mapping
        year_col = column_mapping.get('year_column', 'Year')
        value_col = column_mapping.get('value_column', 'Value')
        type_col = column_mapping.get('type_column', 'Type')
        
        logger.info(f"Loading fresh global forecast data from {source_path}")
        
        # Load data with dynamic column detection
        df = self._load_file(source_path)
        
        # Validate required columns
        required_cols = [year_col, value_col, type_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in global forecast data")
        
        # Apply comprehensive validation
        df, warnings = self._validate_data(df, 'global_forecast')
        for warning in warnings:
            logger.warning(f"Data validation: {warning}")
        
        # Additional specific validation
        self._validate_global_forecast_data(df, year_col, value_col, type_col)
        
        # Ensure year is in the correct format
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        # Check for missing values in critical columns after conversion
        if df[year_col].isna().any() or df[value_col].isna().any():
            invalid_years = df[df[year_col].isna()].shape[0]
            invalid_values = df[df[value_col].isna()].shape[0]
            raise ValueError(f"Data validation failed: {invalid_years} invalid years, {invalid_values} invalid values")
        
        # Calculate growth rates if not present
        if 'Growth_Rate' not in df.columns:
            df = df.sort_values(by=year_col)
            # Calculate percentage change with proper handling of zeros and NaNs
            prev_values = df[value_col].shift(1)
            df['Growth_Rate'] = np.where(
                (prev_values != 0) & (~pd.isna(prev_values)) & (~pd.isna(df[value_col])),
                ((df[value_col] - prev_values) / prev_values) * 100,
                0.0
            )
        
        # Update cache with new data and timestamp
        self._update_cache(cache_key, source_path, df)
        
        return df
    
    def load_country_historical(self) -> pd.DataFrame:
        """
        Load the country historical market data
        
        Returns:
            DataFrame containing the country historical market data
        """
        # First, check if there's uploaded data in session state (for Streamlit)
        try:
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'country_historical') and st.session_state.country_historical is not None:
                logger.info("Using uploaded country historical data from session state")
                df = st.session_state.country_historical.copy()
                
                # First, try to standardize column names and detect format
                df = self._standardize_column_names(df, 'country_historical')
                
                # Check if data is in wide format and convert to long
                df = self._ensure_long_format(df)
                
                # Apply same validation and standardization as file loading
                column_mapping = self.config_manager.get_column_mapping('country_historical')
                id_col = column_mapping.get('id_column', 'idGeo')
                country_col = column_mapping.get('country_column', 'Country')
                
                # Ensure numeric columns are properly formatted with strict validation
                if 'Year' in df.columns and 'Value' in df.columns and id_col in df.columns:
                    original_len = len(df)
                    
                    # Convert to numeric and track conversion failures
                    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                    df['Value'] = pd.to_numeric(df['Value'], errors='coerce') 
                    df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
                    
                    # Check for conversion failures and data integrity
                    year_failures = df['Year'].isna().sum()
                    value_failures = df['Value'].isna().sum()
                    id_failures = df[id_col].isna().sum()
                    
                    if year_failures > 0 or value_failures > 0 or id_failures > 0:
                        logger.warning(f"Data conversion issues in uploaded country data: Year={year_failures}, Value={value_failures}, ID={id_failures} out of {original_len} records")
                        # Clean up the data by removing invalid rows
                        df = df.dropna(subset=['Year', 'Value', id_col])
                        logger.info(f"Cleaned data: {len(df)} valid records remain")
                    
                    # Validate data ranges
                    if 'Year' in df.columns and not df['Year'].isna().all():
                        if (df['Year'] < 1900).any() or (df['Year'] > 2100).any():
                            logger.warning("Some year values are outside normal range (1900-2100)")
                    
                    if 'Value' in df.columns and not df['Value'].isna().all():
                        if (df['Value'] < 0).any():
                            negative_count = (df['Value'] < 0).sum()
                            logger.warning(f"Found {negative_count} negative values in uploaded country data")
                else:
                    logger.warning(f"Missing required columns in uploaded country data. Found columns: {list(df.columns)}")
                
                return df
        except (ImportError, AttributeError):
            # Not in Streamlit context, continue with normal file loading
            pass
        
        source_path = self.config_manager.get_data_source_path('country_historical')
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        
        # Check if cached data is still valid
        cache_key = 'country_historical'
        if self._is_cache_valid(cache_key, source_path):
            logger.info(f"Using cached country historical data")
            return self.data_cache[cache_key]
        
        # Get column names from mapping
        id_col = column_mapping.get('id_column', 'idGeo')
        country_col = column_mapping.get('country_column', 'Country')
        vertical_col = column_mapping.get('vertical_column', 'nameVertical')
        
        logger.info(f"Loading fresh country historical data from {source_path}")
        
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
        
        # Ensure numeric columns are properly formatted with strict validation
        original_len = len(df)
        
        # Convert to numeric and track conversion failures
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce') 
        df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
        
        # Check for conversion failures and data integrity
        year_failures = df['Year'].isna().sum()
        value_failures = df['Value'].isna().sum()
        id_failures = df[id_col].isna().sum()
        
        if year_failures > 0 or value_failures > 0 or id_failures > 0:
            error_msg = f"Data conversion failures in country historical data: Year={year_failures}, Value={value_failures}, ID={id_failures} out of {original_len} records"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate data ranges
        if (df['Year'] < 1900).any() or (df['Year'] > 2100).any():
            raise ValueError("Year values must be between 1900 and 2100")
        
        if (df['Value'] < 0).any():
            negative_count = (df['Value'] < 0).sum()
            raise ValueError(f"Found {negative_count} negative values in country historical data")
        
        # Update cache with new data and timestamp
        self._update_cache(cache_key, source_path, df)
        
        return df
    
    def load_indicator(self, indicator_name: str) -> pd.DataFrame:
        """
        Load an indicator dataset
        
        Args:
            indicator_name: Name of the indicator to load
            
        Returns:
            DataFrame containing the indicator data
        """
        # First, check if there's uploaded data in session state (for Streamlit)
        try:
            import streamlit as st
            if (hasattr(st, 'session_state') and 
                hasattr(st.session_state, 'indicators') and 
                st.session_state.indicators and 
                indicator_name in st.session_state.indicators):
                logger.info(f"Using uploaded indicator '{indicator_name}' from session state")
                indicator_obj = st.session_state.indicators[indicator_name]
                
                # Extract the data DataFrame from the indicator object
                if isinstance(indicator_obj, dict) and 'data' in indicator_obj:
                    df = indicator_obj['data'].copy()
                else:
                    # Fallback: assume the object is the DataFrame itself
                    df = indicator_obj.copy()
                
                # Apply column standardization and format conversion for indicators
                df = self._standardize_column_names(df, 'indicators')
                df = self._ensure_long_format(df)
                
                # Add indicator name column for identification
                df['Indicator'] = indicator_name
                
                return df
        except (ImportError, AttributeError):
            # Not in Streamlit context, continue with normal file loading
            pass
        
        # Generate cache key
        cache_key = f"indicator_{indicator_name}"
        
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
        
        # Check if cached data is still valid
        if self._is_cache_valid(cache_key, source_path):
            logger.info(f"Using cached indicator data for '{indicator_name}'")
            return self.data_cache[cache_key]
        
        logger.info(f"Loading fresh indicator data for '{indicator_name}' from {source_path}")
        
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
        
        # Update cache with new data and timestamp
        self._update_cache(cache_key, source_path, df)
        
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
                # Try to read Excel file with error recovery
                try:
                    df = pd.read_excel(file_path)
                except Exception as excel_error:
                    logger.error(f"Failed to read Excel file {file_path}: {excel_error}")
                    # Try reading as CSV as fallback
                    try:
                        logger.info(f"Attempting to read {file_path} as CSV fallback")
                        df = pd.read_csv(file_path)
                    except Exception:
                        raise ValueError(f"Failed to read file as Excel or CSV: {excel_error}")
                return df
            elif file_ext == '.csv':
                # Try reading CSV with different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        return pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError(f"Could not read CSV file {file_path} with any supported encoding")
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except ValueError:
            raise  # Re-raise ValueError as-is
        except Exception as e:
            logger.error(f"Unexpected error loading file {file_path}: {str(e)}")
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
        # Check if required columns exist - Fixed error message logic
        missing_cols = []
        if id_col not in df.columns:
            missing_cols.append(id_col)
        if country_col not in df.columns:
            missing_cols.append(country_col)
        
        if missing_cols:
            raise ValueError(f"Required columns not found: {', '.join(missing_cols)}")
        
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
        
        Raises:
            ValueError: If DataFrame is empty or has no valid year columns
        """
        # Enhanced: Add comprehensive DataFrame validation with fallback
        if df.empty:
            logger.warning("Empty DataFrame provided for wide-to-long conversion, returning empty result")
            # Return empty DataFrame with expected structure including ID columns
            required_columns = ['Year', 'Value']
            if id_col:
                required_columns.append(id_col)
            if country_col and country_col != id_col:
                required_columns.append(country_col)
            if vertical_col and vertical_col not in [id_col, country_col]:
                required_columns.append(vertical_col)
            return pd.DataFrame(columns=required_columns)
        
        if len(df.columns) == 0:
            logger.warning("DataFrame has no columns to process, returning empty result")
            # Return empty DataFrame with expected structure including ID columns
            required_columns = ['Year', 'Value']
            if id_col:
                required_columns.append(id_col)
            if country_col and country_col != id_col:
                required_columns.append(country_col)
            if vertical_col and vertical_col not in [id_col, country_col]:
                required_columns.append(vertical_col)
            return pd.DataFrame(columns=required_columns)
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
            logger.warning("No year columns found in data, checking if data is already in long format")
            # Check if data might already be in long format
            if 'Year' in df.columns and 'Value' in df.columns:
                logger.info("Data appears to already be in long format, returning as-is")
                return df[['Year', 'Value'] + [col for col in [id_col, country_col, vertical_col] if col and col in df.columns]]
            else:
                raise ValueError("No year columns found in data and not in expected long format")
        
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

    def _validate_global_forecast_data(self, df: pd.DataFrame, year_col: str, value_col: str, type_col: str) -> None:
        """
        Comprehensive validation for global forecast data
        
        Args:
            df: DataFrame to validate
            year_col: Name of year column
            value_col: Name of value column
            type_col: Name of type column
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Global forecast data is empty")
        
        # Validate year column
        if not pd.api.types.is_numeric_dtype(df[year_col]) and not all(df[year_col].astype(str).str.isdigit()):
            raise ValueError(f"Year column '{year_col}' contains non-numeric values")
        
        # Validate value column
        try:
            pd.to_numeric(df[value_col], errors='raise')
        except (ValueError, TypeError):
            raise ValueError(f"Value column '{value_col}' contains non-numeric values")
        
        # Check for negative values
        if (pd.to_numeric(df[value_col], errors='coerce') < 0).any():
            raise ValueError(f"Value column '{value_col}' contains negative values")
        
        # Validate year range
        numeric_years = pd.to_numeric(df[year_col], errors='coerce')
        if (numeric_years < 1900).any() or (numeric_years > 2100).any():
            raise ValueError("Year values must be between 1900 and 2100")
        
        # Validate type column
        valid_types = ['Historical', 'Forecast', 'Projection']
        if not df[type_col].isin(valid_types).all():
            invalid_types = df[~df[type_col].isin(valid_types)][type_col].unique()
            raise ValueError(f"Invalid type values found: {list(invalid_types)}. Valid types: {valid_types}")
        
        logger.info(f"Global forecast data validation passed: {len(df)} records")


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