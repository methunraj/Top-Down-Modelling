"""
Configuration Manager - Handles loading and processing of configuration files

This module provides functionality to load, validate, and access configuration settings
for the Universal Market Forecasting Framework.
"""

import os
import yaml
import json
from typing import Dict, List, Any, Optional, Union


class ConfigurationManager:
    """
    Manages configuration settings for the Universal Market Forecasting Framework
    
    This class handles loading configuration from YAML or JSON files, validating the
    configuration structure, and providing access to configuration settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Configuration Manager
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        """
        self.config: Dict[str, Any] = {}
        self.config_path = config_path
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a file
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        
        Raises:
            ValueError: If the configuration file format is not supported
            FileNotFoundError: If the configuration file does not exist
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if file_ext in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            elif file_ext == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
            
            self.config_path = config_path
            self._validate_config()
            
        except Exception as e:
            raise ValueError(f"Error loading configuration: {str(e)}")
    
    def _validate_config(self) -> None:
        """
        Validate the configuration structure
        
        Raises:
            ValueError: If the configuration is invalid
        """
        # Check for required top-level sections
        required_sections = ['project', 'data_sources']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data_sources section
        data_sources = self.config.get('data_sources', {})
        if not isinstance(data_sources, dict):
            raise ValueError("data_sources must be a dictionary")
        
        # Check for required data sources
        required_sources = ['global_forecast', 'country_historical']
        for source in required_sources:
            if source not in data_sources:
                raise ValueError(f"Missing required data source: {source}")
        
        # Validate indicators if present
        indicators = data_sources.get('indicators', [])
        if indicators and not isinstance(indicators, list):
            raise ValueError("indicators must be a list")
    
    def get_project_info(self) -> Dict[str, Any]:
        """Get project information from configuration"""
        return self.config.get('project', {})
    
    def get_data_source_path(self, source_name: str) -> str:
        """
        Get the file path for a data source
        
        Args:
            source_name: Name of the data source
            
        Returns:
            Path to the data source file
            
        Raises:
            ValueError: If the data source is not found in configuration
        """
        data_sources = self.config.get('data_sources', {})
        source = data_sources.get(source_name)
        
        if not source:
            # Check if it's an indicator
            indicators = data_sources.get('indicators', [])
            for indicator in indicators:
                if indicator.get('name') == source_name:
                    return indicator.get('path', '')
            
            raise ValueError(f"Data source not found: {source_name}")
        
        return source.get('path', '')
    
    def get_indicators(self) -> List[Dict[str, Any]]:
        """Get all configured indicator information"""
        data_sources = self.config.get('data_sources', {})
        return data_sources.get('indicators', [])
    
    def get_column_mapping(self, source_type: str) -> Dict[str, str]:
        """
        Get column mapping for a specific source type
        
        Args:
            source_type: Type of source (global_forecast, country_historical, indicators)
            
        Returns:
            Dictionary mapping logical column names to actual column names in the data
        """
        column_mapping = self.config.get('column_mapping', {})
        return column_mapping.get(source_type, {})
    
    def get_market_distribution_settings(self) -> Dict[str, Any]:
        """Get market distribution settings from configuration"""
        return self.config.get('market_distribution', {})
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output settings from configuration"""
        return self.config.get('output', {})
    
    def get_visualization_config(self, viz_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific visualization type
        
        Args:
            viz_type: Type of visualization (e.g., 'regional_analysis')
            
        Returns:
            Dictionary containing visualization configuration
        """
        output_settings = self.get_output_settings()
        visualization_types = output_settings.get('visualizations', {}).get('types', [])
        
        for viz_config in visualization_types:
            if viz_config.get('name') == viz_type:
                return viz_config
        
        return {}
    
    def get_output_directory(self) -> str:
        """
        Get the output directory from configuration
        
        Returns:
            Path to output directory
        """
        output_settings = self.get_output_settings()
        output_dir = output_settings.get('save_path', 'data/output/')
        return output_dir
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a value from the configuration using dot notation
        
        Args:
            key_path: Path to the value using dot notation (e.g., 'output.save_path')
            default: Default value to return if the key is not found
            
        Returns:
            Value from configuration or default if not found
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update_config(self, key_path: str, value: Any) -> None:
        """
        Update a value in the configuration using dot notation
        
        Args:
            key_path: Path to the value using dot notation (e.g., 'output.save_path')
            value: New value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the right level
        for i, key in enumerate(keys[:-1]):
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file
        
        Args:
            path: Path to save the configuration file (defaults to the original path)
            
        Raises:
            ValueError: If no path is provided and no original path exists
        """
        save_path = path or self.config_path
        
        if not save_path:
            raise ValueError("No path provided and no original path exists")
        
        file_ext = os.path.splitext(save_path)[1].lower()
        
        try:
            if file_ext in ['.yaml', '.yml']:
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif file_ext == '.json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_ext}")
        except Exception as e:
            raise ValueError(f"Error saving configuration: {str(e)}")


def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration structure
    
    Returns:
        Default configuration dictionary
    """
    return {
        "project": {
            "name": "Universal Market Estimator",
            "version": "1.0",
            "market_type": "Machine Learning"
        },
        "data_sources": {
            "global_forecast": {
                "path": "data/input/Market_Forecast.xlsx",
                "identifier": {
                    "value_column": "Value",
                    "year_column": "Year",
                    "type_column": "Type"
                }
            },
            "country_historical": {
                "path": "data/input/Country_Historical_data.xlsx",
                "identifier": {
                    "id_column": "idGeo",
                    "name_column": "Country",
                    "market_column": "nameVertical"
                }
            },
            "indicators": [
                {
                    "name": "indicator1",
                    "path": "data/input/indicator1.xlsx",
                    "id_column": "idGeo",
                    "weight": "auto"
                },
                {
                    "name": "indicator2",
                    "path": "data/input/indicator2.xlsx",
                    "id_column": "idGeo",
                    "weight": "auto"
                }
            ]
        },
        "column_mapping": {
            "global_forecast": {
                "year_column": "Year",
                "value_column": "Value",
                "type_column": "Type"
            },
            "country_historical": {
                "id_column": "idGeo",
                "country_column": "Country",
                "vertical_column": "nameVertical"
            },
            "indicators": {
                "id_column": "idGeo",
                "country_column": "Country",
                "year_column": "Year",
                "value_column": "Value"
            }
        },
        "market_distribution": {
            "tier_determination": "auto",
            "redistribution_start_year": null,  # Set to a specific year (e.g., 2020) to only redistribute from that year forward
            "manual_tiers": {
                "tier1": {
                    "description": "Market Leaders",
                    "share_threshold": 5.0,
                    "max_share_change": "auto"
                },
                "tier2": {
                    "description": "Established Markets",
                    "share_threshold": 1.0,
                    "max_share_change": "auto"
                }
            },
            "growth_constraints": {
                "determination_method": "auto",
                "manual_constraints": {
                    "max_growth_rate": 60,
                    "min_growth_rate": -30,
                    "apply_scaling_by_market_size": True
                }
            }
        },
        "output": {
            "save_path": "data/output/",
            "formats": ["xlsx", "csv", "json"],
            "visualizations": {
                "types": [
                    {
                        "name": "market_size",
                        "title": "${market_type} Market Size by Country",
                        "top_n_countries": 10
                    },
                    {
                        "name": "growth_rates",
                        "title": "${market_type} Growth Rates",
                        "top_n_countries": 15
                    },
                    {
                        "name": "cagr_analysis",
                        "title": "${market_type} CAGR Analysis",
                        "periods": [
                            {
                                "name": "Short-term",
                                "years": 3
                            },
                            {
                                "name": "Mid-term",
                                "years": 5
                            },
                            {
                                "name": "Long-term",
                                "years": 7
                            }
                        ]
                    }
                ]
            }
        }
    } 