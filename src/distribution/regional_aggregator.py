"""
Regional Aggregator Module - Universal support for regional market aggregation

This module provides functionality to aggregate country-level market data into 
regions and handle complex hierarchical regional structures while maintaining 
consistency at all aggregation levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import logging
import networkx as nx
from collections import defaultdict

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegionalAggregator:
    """
    Universal regional aggregator for market distribution
    
    This class provides functionality to aggregate country-level market values into
    regional groupings while maintaining mathematical consistency across the hierarchy.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the RegionalAggregator
        
        Args:
            config_manager: Optional configuration manager instance for accessing settings
        """
        self.config_manager = config_manager
        
        # Region definitions from configuration
        self.hierarchy_definition = {}
        self.region_metadata = {}
        self.region_to_idgeo = {}
        self.idgeo_to_region = {}
        
        # Region hierarchy graph for dependency resolution
        self.region_graph = None
        
        # Load region definitions
        if config_manager:
            self._load_region_definitions()
    
    def _load_region_definitions(self):
        """
        Load region definitions from configuration
        """
        if not self.config_manager:
            logger.warning("No configuration manager provided, using default regions")
            return
        
        # Get region definitions from config
        # Check both old (regions) and new (regional_aggregation) config paths
        regions_config = self.config_manager.config.get('regions', {})
        regional_aggregation = self.config_manager.config.get('regional_aggregation', {})
        
        hierarchy_definition = {}
        region_metadata = {}
        
        # Try to load from old config format
        if regions_config:
            hierarchy_definition = regions_config.get('hierarchy_definition', {})
            region_metadata = regions_config.get('region_metadata', {})
            
        # Try to load from new config format
        if not hierarchy_definition and regional_aggregation:
            # Check if region_definitions exists
            region_definitions = regional_aggregation.get('region_definitions', [])
            hierarchical = regional_aggregation.get('hierarchical', {})
            
            if region_definitions:
                # Convert region_definitions list to hierarchy_definition dict
                logger.info(f"Converting {len(region_definitions)} region definitions to hierarchy format")
                for region in region_definitions:
                    region_name = region.get('name')
                    countries = region.get('countries', [])
                    if region_name and countries:
                        hierarchy_definition[region_name] = countries
        
        if hierarchy_definition:
            self.hierarchy_definition = hierarchy_definition
            logger.info(f"Loaded {len(hierarchy_definition)} region definitions from configuration")
        else:
            logger.warning("No region definitions found in configuration")
        
        if region_metadata:
            self.region_metadata = region_metadata
            logger.info(f"Loaded metadata for {len(region_metadata)} regions")
            
        # Generate ID mappings for regions
        start_id = 90000  # Using high ID values to avoid conflicts with country IDs
        for i, region_name in enumerate(self.hierarchy_definition.keys()):
            region_id = start_id + i
            self.region_to_idgeo[region_name] = region_id
            self.idgeo_to_region[region_id] = region_name
        
        # Build region hierarchy graph
        self._build_region_hierarchy_graph()
    
    def _build_region_hierarchy_graph(self):
        """
        Build a directed graph representing the region hierarchy
        
        This graph is used for dependency resolution during hierarchical aggregation
        """
        self.region_graph = nx.DiGraph()
        
        # Add all regions as nodes
        for region in self.hierarchy_definition.keys():
            self.region_graph.add_node(region)
        
        # Add "Worldwide" as the root node if it doesn't exist
        if "Worldwide" not in self.region_graph:
            self.region_graph.add_node("Worldwide")
        
        # Add edges from child to parent regions
        for region, constituents in self.hierarchy_definition.items():
            for constituent in constituents:
                # Check if the constituent is a region (not a country)
                if constituent in self.hierarchy_definition:
                    # Add edge from child to parent
                    self.region_graph.add_edge(constituent, region)
                    
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.region_graph):
            logger.error("Region hierarchy contains cycles, which will cause inconsistencies")
            # Find and log cycles for debugging
            cycles = list(nx.simple_cycles(self.region_graph))
            for cycle in cycles:
                logger.error(f"Cycle detected: {' -> '.join(cycle)}")
    
    def set_hierarchy_definition(self, hierarchy_definition: Dict[str, List[str]],
                                 region_metadata: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Set or update the region hierarchy definition
        
        Args:
            hierarchy_definition: Dictionary mapping regions to their constituent countries/regions
            region_metadata: Optional dictionary with region metadata
        """
        self.hierarchy_definition = hierarchy_definition
        
        if region_metadata:
            self.region_metadata = region_metadata
        
        # Regenerate ID mappings
        start_id = 90000  # Using high ID values to avoid conflicts with country IDs
        for i, region_name in enumerate(self.hierarchy_definition.keys()):
            region_id = start_id + i
            self.region_to_idgeo[region_name] = region_id
            self.idgeo_to_region[region_id] = region_name
        
        # Rebuild hierarchy graph
        self._build_region_hierarchy_graph()
        
        logger.info(f"Updated hierarchy definition with {len(hierarchy_definition)} regions")
    
    def aggregate_by_regions(self, market_data: pd.DataFrame, 
                            id_col: str = 'idGeo', 
                            country_col: str = 'Country', 
                            value_col: str = 'Value') -> pd.DataFrame:
        """
        Aggregate country-level data into regions based on defined mappings
        
        Args:
            market_data: DataFrame with country-level market data
            id_col: Name of ID column
            country_col: Name of country name column
            value_col: Name of value column
            
        Returns:
            DataFrame with regional data appended to country data
        """
        if not self.hierarchy_definition:
            logger.warning("No region definitions available for aggregation")
            return market_data
        
        # Make a copy to avoid modifying the original
        result_df = market_data.copy()
        
        # Create a mapping of countries to regions
        country_to_regions = defaultdict(list)
        for region, constituents in self.hierarchy_definition.items():
            for country in constituents:
                # Only add direct mappings for countries (not regions)
                if country not in self.hierarchy_definition:
                    country_to_regions[country].append(region)
        
        # Get list of columns that should be aggregated (numeric columns)
        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
        
        # Ensure the value column is in the list
        if value_col not in numeric_cols:
            numeric_cols.append(value_col)
        
        # Prepare to store regional data
        regional_data = []
        
        # Process each year separately if 'Year' column present
        year_groups = [None]
        if 'Year' in result_df.columns:
            year_groups = result_df['Year'].unique()
            
        for year in year_groups:
            # Filter by year if applicable
            year_filter = (result_df['Year'] == year) if year is not None else slice(None)
            year_data = result_df[year_filter]
            
            # Aggregate data for each region
            for region, constituents in self.hierarchy_definition.items():
                # Filter data for countries in this region
                region_filter = year_data[country_col].isin(constituents)
                region_countries = year_data[region_filter]
                
                if region_countries.empty:
                    logger.debug(f"No data found for region '{region}' in year {year}")
                    continue
                
                # Sum all numeric columns
                agg_data = region_countries[numeric_cols].sum()
                
                # Create a row for the region
                region_row = {
                    id_col: self.region_to_idgeo.get(region, -1),
                    country_col: region,
                    **{col: agg_data[col] for col in numeric_cols}
                }
                
                # Add 'Year' if present
                if year is not None:
                    region_row['Year'] = year
                
                # Add 'region_type' column to identify as region
                region_row['region_type'] = 'region'
                
                # Add any region metadata
                if region in self.region_metadata:
                    for key, value in self.region_metadata[region].items():
                        if key not in region_row:
                            region_row[key] = value
                
                regional_data.append(region_row)
        
        # Combine original data with regional data
        if regional_data:
            # Add region_type column to original data (as 'country')
            result_df['region_type'] = 'country'
            
            # Combine with regional data
            regional_df = pd.DataFrame(regional_data)
            result_df = pd.concat([result_df, regional_df], ignore_index=True)
        
        return result_df
    
    def aggregate_hierarchical(self, market_data: pd.DataFrame,
                              id_col: str = 'idGeo',
                              country_col: str = 'Country',
                              value_col: str = 'Value') -> pd.DataFrame:
        """
        Perform hierarchical aggregation respecting region dependencies
        
        Args:
            market_data: DataFrame with country-level market data
            id_col: Name of ID column
            country_col: Name of country name column
            value_col: Name of value column
            
        Returns:
            DataFrame with hierarchically aggregated regional data
        """
        if not self.hierarchy_definition or not self.region_graph:
            logger.warning("No region hierarchy available for aggregation")
            return market_data
        
        # Make a copy to avoid modifying the original
        result_df = market_data.copy()
        
        # Get list of columns that should be aggregated (numeric columns)
        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
        
        # Ensure the value column is in the list
        if value_col not in numeric_cols:
            numeric_cols.append(value_col)
        
        # Prepare to store regional data
        regional_data = []
        
        # Process each year separately if 'Year' column present
        year_groups = [None]
        if 'Year' in result_df.columns:
            year_groups = result_df['Year'].unique()
        
        for year in year_groups:
            # Filter by year if applicable
            year_filter = (result_df['Year'] == year) if year is not None else slice(None)
            year_data = result_df[year_filter]
            
            # Get a topological sort of the region hierarchy to ensure 
            # we process regions in dependency order (children before parents)
            try:
                region_order = list(reversed(list(nx.topological_sort(self.region_graph))))
            except nx.NetworkXUnfeasible:
                logger.error("Region hierarchy contains cycles and cannot be sorted")
                continue
            
            # Create a dictionary to store intermediate results
            intermediate_results = {}
            
            # Process regions in order
            for region in region_order:
                # Get direct constituents
                constituents = self.hierarchy_definition.get(region, [])
                if not constituents:
                    logger.debug(f"No constituents found for region '{region}'")
                    continue
                
                # Split constituents into countries and sub-regions
                countries = []
                sub_regions = []
                
                for constituent in constituents:
                    if constituent in self.hierarchy_definition:
                        sub_regions.append(constituent)
                    else:
                        countries.append(constituent)
                
                # Aggregate country data
                country_filter = year_data[country_col].isin(countries)
                country_data = year_data[country_filter]
                
                # Get sub-region data from intermediate results
                sub_region_data = []
                for sub_region in sub_regions:
                    if sub_region in intermediate_results:
                        sub_region_data.append(intermediate_results[sub_region])
                
                # Combine country and sub-region data
                combined_data = pd.concat([country_data] + sub_region_data) if sub_region_data else country_data
                
                if combined_data.empty:
                    logger.debug(f"No data found for region '{region}' in year {year}")
                    continue
                
                # Sum all numeric columns
                agg_data = combined_data[numeric_cols].sum()
                
                # Create a row for the region
                region_row = {
                    id_col: self.region_to_idgeo.get(region, -1),
                    country_col: region,
                    **{col: agg_data[col] for col in numeric_cols}
                }
                
                # Add 'Year' if present
                if year is not None:
                    region_row['Year'] = year
                
                # Add 'region_type' column to identify as region
                region_row['region_type'] = 'region'
                
                # Add hierarchical metadata
                region_row['region_level'] = self._get_region_level(region)
                region_row['parent_regions'] = ','.join(list(self.region_graph.successors(region)))
                region_row['child_regions'] = ','.join(sub_regions)
                region_row['child_countries'] = ','.join(countries)
                
                # Add any region metadata
                if region in self.region_metadata:
                    for key, value in self.region_metadata[region].items():
                        if key not in region_row:
                            region_row[key] = value
                
                # Store as intermediate result
                intermediate_results[region] = pd.DataFrame([region_row])
                
                # Add to final results
                regional_data.append(region_row)
        
        # Combine original data with regional data
        if regional_data:
            # Add region_type column to original data (as 'country')
            result_df['region_type'] = 'country'
            
            # Combine with regional data
            regional_df = pd.DataFrame(regional_data)
            result_df = pd.concat([result_df, regional_df], ignore_index=True)
        
        return result_df
    
    def _get_region_level(self, region: str) -> int:
        """
        Get the hierarchical level of a region (higher numbers = higher in hierarchy)
        
        Args:
            region: Name of the region
            
        Returns:
            Integer level (0 = country/leaf, 1 = first level region, etc.)
        """
        if not self.region_graph or region not in self.region_graph:
            return 0
        
        # Find longest path from any leaf to this region
        max_path_length = 0
        
        # Find leaf nodes (nodes with no predecessors)
        leaf_nodes = [node for node in self.region_graph.nodes 
                     if self.region_graph.in_degree(node) == 0]
        
        for leaf in leaf_nodes:
            try:
                path_length = len(nx.shortest_path(self.region_graph, leaf, region)) - 1
                max_path_length = max(max_path_length, path_length)
            except nx.NetworkXNoPath:
                # No path from this leaf to the target region
                continue
        
        return max_path_length
    
    def validate_regional_consistency(self, market_data: pd.DataFrame,
                                     id_col: str = 'idGeo',
                                     country_col: str = 'Country',
                                     value_col: str = 'Value') -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate that regional aggregations are mathematically consistent
        
        Args:
            market_data: DataFrame with country and region-level market data
            id_col: Name of ID column
            country_col: Name of country name column
            value_col: Name of value column
            
        Returns:
            Tuple with (is_consistent, list_of_inconsistencies)
        """
        # Check if we have region data
        if 'region_type' not in market_data.columns:
            return True, []
        
        # Get region data
        region_data = market_data[market_data['region_type'] == 'region']
        if region_data.empty:
            return True, []
        
        # Process each year separately if 'Year' column present
        year_groups = [None]
        if 'Year' in market_data.columns:
            year_groups = market_data['Year'].unique()
        
        inconsistencies = []
        is_consistent = True
        
        for year in year_groups:
            # Filter by year if applicable
            year_filter = (market_data['Year'] == year) if year is not None else slice(None)
            year_data = market_data[year_filter]
            year_regions = year_data[year_data['region_type'] == 'region']
            
            # Check each region
            for _, region_row in year_regions.iterrows():
                region = region_row[country_col]
                region_value = region_row[value_col]
                
                # Get constituents
                constituents = self.hierarchy_definition.get(region, [])
                if not constituents:
                    continue
                
                # Calculate expected value from constituents
                constituent_data = year_data[year_data[country_col].isin(constituents)]
                if constituent_data.empty:
                    # Can't validate - no data for constituents
                    continue
                
                expected_value = constituent_data[value_col].sum()
                
                # Check if values match within tolerance
                # Use relative tolerance for large values, absolute for small values
                if max(region_value, expected_value) > 1000:
                    # For large values, use relative tolerance (0.01% = 0.0001)
                    tolerance = max(region_value, expected_value) * 0.0001
                else:
                    # For small values, use absolute tolerance
                    tolerance = 0.01
                
                difference = abs(region_value - expected_value)
                
                if difference > tolerance:
                    # Found an inconsistency
                    is_consistent = False
                    
                    inconsistency = {
                        'region': region,
                        'year': year,
                        'region_value': region_value,
                        'expected_value': expected_value,
                        'difference': difference,
                        'constituents': constituent_data[country_col].tolist()
                    }
                    
                    inconsistencies.append(inconsistency)
                    
                    logger.warning(
                        f"Inconsistency detected for region '{region}' (year={year}): "
                        f"Region value = {region_value}, Expected from constituents = {expected_value}"
                    )
        
        return is_consistent, inconsistencies
    
    def enforce_regional_consistency(self, market_data: pd.DataFrame,
                                    id_col: str = 'idGeo',
                                    country_col: str = 'Country',
                                    value_col: str = 'Value',
                                    method: str = 'bottom_up') -> pd.DataFrame:
        """
        Enforce consistency in regional aggregations using specified method
        
        Args:
            market_data: DataFrame with country and region-level market data
            id_col: Name of ID column
            country_col: Name of country name column
            value_col: Name of value column
            method: Consistency enforcement method ('bottom_up', 'top_down', or 'hybrid')
            
        Returns:
            DataFrame with consistent regional aggregations
        """
        # Make a copy to avoid modifying the original
        result_df = market_data.copy()
        
        # Check if we have region data
        if 'region_type' not in result_df.columns:
            return result_df
        
        # Get region data
        region_data = result_df[result_df['region_type'] == 'region']
        if region_data.empty:
            return result_df
        
        # Process each year separately if 'Year' column present
        year_groups = [None]
        if 'Year' in result_df.columns:
            year_groups = result_df['Year'].unique()
        
        for year in year_groups:
            # Filter by year if applicable
            year_filter = (result_df['Year'] == year) if year is not None else slice(None)
            
            # Get a topological sort of the region hierarchy
            try:
                if method == 'bottom_up':
                    # Process from leaf nodes upward
                    region_order = list(nx.topological_sort(self.region_graph))
                elif method == 'top_down':
                    # Process from root nodes downward
                    region_order = list(reversed(list(nx.topological_sort(self.region_graph))))
                else:  # hybrid
                    # Process in topological order, but with some special handling
                    region_order = list(nx.topological_sort(self.region_graph))
            except nx.NetworkXUnfeasible:
                logger.error("Region hierarchy contains cycles and cannot be sorted")
                continue
            
            # Process regions in order
            for region in region_order:
                # Get region row
                region_rows = result_df.loc[year_filter & 
                                        (result_df[country_col] == region) & 
                                        (result_df['region_type'] == 'region')]
                
                if region_rows.empty:
                    continue
                
                region_row_idx = region_rows.index[0]
                region_value = result_df.loc[region_row_idx, value_col]
                
                # Get constituents
                constituents = self.hierarchy_definition.get(region, [])
                if not constituents:
                    continue
                
                # Get constituent data
                constituent_mask = year_filter & result_df[country_col].isin(constituents)
                constituent_data = result_df[constituent_mask]
                
                if constituent_data.empty:
                    continue
                
                # Get current total from constituents
                constituent_total = constituent_data[value_col].sum()
                
                # Check if adjustment is needed
                tolerance = 0.0001  # Allow small floating point differences
                if abs(region_value - constituent_total) <= tolerance:
                    continue  # Already consistent
                
                if method == 'bottom_up' or (method == 'hybrid' and region == 'Worldwide'):
                    # Adjust region value to match constituents
                    result_df.loc[region_row_idx, value_col] = constituent_total
                    logger.debug(f"Adjusted region '{region}' value from {region_value} to {constituent_total}")
                    
                elif method == 'top_down' or method == 'hybrid':
                    # Adjust constituent values to match region total
                    if constituent_total == 0:
                        # Distribute evenly
                        new_values = [region_value / len(constituent_data)] * len(constituent_data)
                    else:
                        # Adjust proportionally
                        adjustment_factor = region_value / constituent_total
                        new_values = constituent_data[value_col] * adjustment_factor
                    
                    # Update constituent values
                    result_df.loc[constituent_mask, value_col] = new_values.values
                    logger.debug(
                        f"Adjusted constituents of region '{region}' "
                        f"(total {constituent_total} -> {region_value})"
                    )
        
        return result_df
    
    def get_region_constituents(self, region: str, include_nested: bool = False) -> List[str]:
        """
        Get all constituent countries for a region
        
        Args:
            region: Name of the region
            include_nested: Whether to include constituents of sub-regions
            
        Returns:
            List of country names
        """
        if region not in self.hierarchy_definition:
            return []
        
        direct_constituents = self.hierarchy_definition[region]
        
        if not include_nested:
            return [c for c in direct_constituents if c not in self.hierarchy_definition]
        
        # Include nested constituents
        all_constituents = set()
        to_process = direct_constituents.copy()
        
        while to_process:
            current = to_process.pop(0)
            
            if current in self.hierarchy_definition:
                # This is a region, add its constituents to processing list
                sub_constituents = self.hierarchy_definition[current]
                to_process.extend([c for c in sub_constituents if c not in all_constituents])
            else:
                # This is a country, add to results
                all_constituents.add(current)
        
        return list(all_constituents)
    
    def get_country_regions(self, country: str) -> List[str]:
        """
        Get all regions that a country belongs to
        
        Args:
            country: Name of the country
            
        Returns:
            List of region names
        """
        regions = []
        
        for region, constituents in self.hierarchy_definition.items():
            if country in constituents:
                regions.append(region)
                
                # Find parent regions recursively
                parent_regions = []
                to_check = [region]
                
                while to_check:
                    current = to_check.pop(0)
                    
                    for parent, parent_constituents in self.hierarchy_definition.items():
                        if current in parent_constituents and parent not in regions and parent not in parent_regions:
                            parent_regions.append(parent)
                            to_check.append(parent)
                
                regions.extend(parent_regions)
        
        return regions
    
    def get_region_hierarchy_levels(self) -> Dict[str, int]:
        """
        Get the hierarchical level of each region
        
        Returns:
            Dictionary mapping region names to their hierarchical levels
        """
        levels = {}
        
        for region in self.hierarchy_definition.keys():
            levels[region] = self._get_region_level(region)
        
        return levels