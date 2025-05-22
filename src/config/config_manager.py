"""
Configuration Manager - Handles loading and processing of configuration files

This module provides functionality to load, validate, and access configuration settings
for the Universal Market Forecasting Framework.
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class IncludeLoader(yaml.SafeLoader):
    """Custom YAML loader that supports including external files"""
    
    def __init__(self, stream):
        # Fixed: Add validation for stream.name attribute
        if hasattr(stream, 'name') and stream.name:
            self._root = os.path.dirname(stream.name)
        else:
            # Fallback to current directory if stream has no name
            self._root = os.getcwd()
        super().__init__(stream)


def include_yaml_file(loader, node):
    """Process the !include directive in YAML files"""
    # Get the path to the included file
    filename = os.path.join(loader._root, loader.construct_scalar(node))
    
    # Check if the file exists
    if not os.path.exists(filename):
        # Try with .yaml and .yml extensions
        for ext in ['.yaml', '.yml']:
            if os.path.exists(filename + ext):
                filename += ext
                break
        else:  # No matching file found
            raise FileNotFoundError(f"Included file not found: {filename}")
    
    # Load and return the file content
    with open(filename, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=IncludeLoader)


# Register the include constructor
IncludeLoader.add_constructor('!include', include_yaml_file)


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
                    self.config = yaml.load(f, Loader=IncludeLoader)
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
        
        # Enhanced indicator validation
        if indicators:
            for i, indicator in enumerate(indicators):
                if not isinstance(indicator, dict):
                    raise ValueError(f"Indicator {i} must be a dictionary")
                if 'name' not in indicator:
                    raise ValueError(f"Indicator {i} must have a 'name' field")
                if 'path' not in indicator:
                    raise ValueError(f"Indicator {i} must have a 'path' field")
                    
                # Validate weight if present
                weight = indicator.get('weight')
                if weight is not None and weight != 'auto':
                    try:
                        weight_val = float(weight)
                        if not (0.0 <= weight_val <= 1.0):
                            raise ValueError(f"Indicator {i} weight must be between 0.0 and 1.0 or 'auto'")
                    except (ValueError, TypeError):
                        raise ValueError(f"Indicator {i} weight must be numeric or 'auto'")
        
        # Validate market distribution settings if present
        market_dist = self.config.get('market_distribution', {})
        if market_dist:
            # Validate tier determination
            tier_det = market_dist.get('tier_determination', 'auto')
            if tier_det not in ['auto', 'manual']:
                raise ValueError("tier_determination must be 'auto' or 'manual'")
            
            # Validate redistribution year if present
            redist_year = market_dist.get('redistribution_start_year')
            if redist_year is not None:
                try:
                    year_val = int(redist_year)
                    if not (1800 <= year_val <= 2200):
                        raise ValueError("redistribution_start_year must be between 1800 and 2200")
                except (ValueError, TypeError):
                    raise ValueError("redistribution_start_year must be an integer year")
        
        # Validate project information
        project_info = self.config.get('project', {})
        if 'name' not in project_info:
            logger.warning("Project name not specified in configuration")
        if 'market_type' not in project_info:
            logger.warning("Market type not specified in configuration")
    
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
    
    def get_region_definitions(self) -> Dict[str, List[str]]:
        """
        Get region definitions from configuration
        
        Returns:
            Dictionary mapping region names to lists of constituent countries/regions
        """
        return self.config.get('regions', {}).get('hierarchy_definition', {})
    
    def get_region_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get region metadata from configuration
        
        Returns:
            Dictionary mapping region names to their metadata
        """
        return self.config.get('regions', {}).get('region_metadata', {})
    
    def set_region_definitions(self, hierarchy_definition: Dict[str, List[str]], 
                              region_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Set or update region definitions in configuration
        
        Args:
            hierarchy_definition: Dictionary mapping region names to lists of constituent countries/regions
            region_metadata: Optional dictionary with region metadata
        
        Raises:
            ValueError: If hierarchy_definition is not properly formatted
        """
        # Fixed: Add validation for hierarchy_definition format
        if not isinstance(hierarchy_definition, dict):
            raise ValueError("hierarchy_definition must be a dictionary")
        
        for region_name, countries in hierarchy_definition.items():
            if not isinstance(region_name, str):
                raise ValueError(f"Region name must be a string, got {type(region_name)}")
            if not isinstance(countries, list):
                raise ValueError(f"Countries for region '{region_name}' must be a list, got {type(countries)}")
            if not all(isinstance(country, str) for country in countries):
                raise ValueError(f"All countries in region '{region_name}' must be strings")
        
        # Fixed: Add validation for region_metadata format
        if region_metadata is not None:
            if not isinstance(region_metadata, dict):
                raise ValueError("region_metadata must be a dictionary")
            for region_name, metadata in region_metadata.items():
                if not isinstance(region_name, str):
                    raise ValueError(f"Region name in metadata must be a string, got {type(region_name)}")
                if not isinstance(metadata, dict):
                    raise ValueError(f"Metadata for region '{region_name}' must be a dictionary, got {type(metadata)}")
        
        if 'regions' not in self.config:
            self.config['regions'] = {}
            
        self.config['regions']['hierarchy_definition'] = hierarchy_definition
        
        if region_metadata:
            self.config['regions']['region_metadata'] = region_metadata
    
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
            ],
            "enable_causal_analysis": True,
            "apply_causal_adjustments": True,
            "causal_analysis": {
                "method": "ensemble",
                "lags": 1,
                "alpha": 0.05,
                "regularization": "elastic_net",
                "interaction_detection": True
            }
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
            "redistribution_start_year": None,  # Set to a specific year (e.g., 2020) to only redistribute from that year forward
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
            },
            "use_gradient_harmonization": True,
            "harmonization": {
                "method": "adaptive",
                "smoothing_strength": 0.5,
                "preserve_inflection": True,
                "transition_zone": 2,
                "global_consistency": True,
                "regional_consistency": True,
                "boundary_enforcement": "relaxed",
                "target_growth_rates": {
                    "default": 15.0,
                    "tier1": 12.0,
                    "tier2": 18.0,
                    "tier3": 25.0
                },
                "inflection_detection": {
                    "enabled": True,
                    "sensitivity": 0.6
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
                    },
                    {
                        "name": "regional_analysis",
                        "title": "${market_type} Regional Analysis",
                        "specific_year": None,
                        "analysis_years": None
                    }
                ]
            }
        },
        "regions": {
            "hierarchy_definition": {
                "APAC": ["Pakistan", "New Zealand", "Bangladesh", "Bhutan", "Brunei Darussalam", "Myanmar", "Cambodia", 
                       "Vietnam", "Sri Lanka", "Mainland China", "Laos", "Taiwan", "Mongolia", "Timor-Leste", 
                       "Papua New Guinea", "Fiji", "Thailand", "South Korea", "India", "Australia", "China", 
                       "Hong Kong", "Indonesia", "Nepal", "Malaysia", "Philippines", "Singapore", "Japan"],
                "Americas": ["Central America", "South America", "North America", "Caribbean"],
                "EMEA": ["Europe", "Middle East", "Africa"],
                "Worldwide": ["APAC", "Americas", "EMEA"]
            },
            "region_metadata": {
                "APAC": {"description": "Asia Pacific region"},
                "Americas": {"description": "North, Central, and South America"},
                "EMEA": {"description": "Europe, Middle East, and Africa"},
                "Worldwide": {"description": "Global total"}
            }
        }
    }