"""
Market Visualizer Module - Flexible visualization for market forecasts

This module provides functionality to generate visualizations and reports for
market forecasts in a market-agnostic way, making it universally applicable
to any market type.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from matplotlib.ticker import FuncFormatter
import re
import networkx as nx

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketVisualizer:
    """
    Universal market visualizer for any market type
    
    This class provides functionality to generate visualizations and reports for
    market forecasts in a market-agnostic way.
    """
    
    def __init__(self, config_manager, data_loader):
        """
        Initialize the MarketVisualizer
        
        Args:
            config_manager: Configuration manager instance for accessing settings
            data_loader: Data loader instance for accessing market data
        """
        self.config_manager = config_manager
        self.data_loader = data_loader
        
        # Get project info
        self.project_info = self.config_manager.get_project_info()
        self.market_type = self.project_info.get('market_type', 'Market')
        
        # Set default style for visualizations
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        
        # Define a color palette
        self.color_palette = sns.color_palette("viridis", 12)
        
        # Set default output directory
        self.output_dir = self.config_manager.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track created visualization files
        self.visualization_files = []
    
    def generate_all_visualizations(self, market_data: pd.DataFrame) -> List[str]:
        """
        Generate all configured visualizations
        
        Args:
            market_data: DataFrame with market forecast data
            
        Returns:
            List of paths to generated visualization files
        """
        # Reset visualization files list
        self.visualization_files = []
        
        # Get visualization configurations
        viz_configs = self.config_manager.get_value(
            'output.visualizations.types', 
            []
        )
        
        if not viz_configs:
            logger.warning("No visualization configurations found")
            return []
        
        # Generate each configured visualization
        for viz_config in viz_configs:
            viz_type = viz_config.get('name', '')
            
            try:
                if viz_type == 'market_size':
                    self._create_market_size_visualization(market_data, viz_config)
                elif viz_type == 'growth_rates':
                    self._create_growth_rates_visualization(market_data, viz_config)
                elif viz_type == 'cagr_analysis':
                    self._create_cagr_analysis_visualization(market_data, viz_config)
                elif viz_type == 'regional_analysis':
                    self._create_regional_analysis_visualization(market_data, viz_config)
                elif viz_type == 'causal_influence':
                    self._create_causal_influence_visualization(market_data, viz_config)
                else:
                    logger.warning(f"Unknown visualization type: {viz_type}")
            except Exception as e:
                logger.error(f"Error generating {viz_type} visualization: {str(e)}")
        
        return self.visualization_files
    
    def _create_market_size_visualization(self, market_data: pd.DataFrame, 
                                         config: Dict[str, Any]) -> str:
        """
        Create market size visualization
        
        Args:
            market_data: DataFrame with market forecast data
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Generating market size visualization")
        
        # Get configuration
        title = config.get('title', f"{self.market_type} Market Size by Country")
        title = self._replace_variables(title)
        top_n = config.get('top_n_countries', 10)
        
        # Get the latest year in the data
        latest_year = market_data['Year'].max()
        latest_data = market_data[market_data['Year'] == latest_year].copy()
        
        # Fixed: Get value column name and validate existence
        value_column = self.config_manager.get_column_mapping('global_forecast').get('value_column', 'Value')
        
        # Validate that the value column exists
        if value_column not in latest_data.columns:
            logger.warning(f"Value column '{value_column}' not found in data, using 'Value' as fallback")
            value_column = 'Value'
            if value_column not in latest_data.columns:
                logger.error("Neither configured value column nor 'Value' column found in data")
                return None
        
        # Get top N countries by market value
        top_countries = latest_data.sort_values(by=value_column, ascending=False).head(top_n)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Bar chart of market size - Fixed to use dynamic column name
        ax = sns.barplot(
            x='Country',
            y=value_column,
            data=top_countries,
            palette=self.color_palette[:top_n]
        )
        
        # Format y-axis as millions/billions
        ax.yaxis.set_major_formatter(FuncFormatter(self._format_value_axis))
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height()/1e6:.1f}M" if p.get_height() < 1e9 else f"{p.get_height()/1e9:.1f}B",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom'
            )
        
        # Customize plot
        plt.title(f"{title} ({latest_year})", fontsize=16)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Market Value', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the visualization
        output_file = os.path.join(self.output_dir, f"{self.market_type}_Market_Size_{latest_year}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved market size visualization to {output_file}")
        self.visualization_files.append(output_file)
        return output_file
    
    def _create_growth_rates_visualization(self, market_data: pd.DataFrame, 
                                         config: Dict[str, Any]) -> str:
        """
        Create growth rates visualization
        
        Args:
            market_data: DataFrame with market forecast data
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Generating growth rates visualization")
        
        # Get configuration
        title = config.get('title', f"{self.market_type} Growth Rates")
        title = self._replace_variables(title)
        top_n = config.get('top_n_countries', 15)
        
        # Calculate year-over-year growth rates
        growth_data = []
        
        # Group by country
        for country, group in market_data.groupby(['idGeo', 'Country']):
            country_id, country_name = country
            
            # Sort by year
            group = group.sort_values(by='Year')
            
            # Calculate growth rate
            group['Growth_Rate'] = group['Value'].pct_change() * 100
            
            # Add to results (excluding first year which has no growth rate)
            growth_data.append(group.iloc[1:])
        
        # Combine all data
        growth_df = pd.concat(growth_data, ignore_index=True)
        
        # Get the latest year in the data
        latest_year = growth_df['Year'].max()
        latest_growth = growth_df[growth_df['Year'] == latest_year].copy()
        
        # Get top N countries by growth rate
        top_growth = latest_growth.sort_values(by='Growth_Rate', ascending=False).head(top_n)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Bar chart of growth rates
        ax = sns.barplot(
            x='Country',
            y='Growth_Rate',
            data=top_growth,
            palette=self.color_palette[:top_n]
        )
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom'
            )
        
        # Customize plot
        plt.title(f"{title} ({latest_year})", fontsize=16)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the visualization
        output_file = os.path.join(self.output_dir, f"{self.market_type}_Growth_Rates_{latest_year}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved growth rates visualization to {output_file}")
        self.visualization_files.append(output_file)
        
        # Also create a growth trends visualization
        self._create_growth_trends_visualization(growth_df, config)
        
        return output_file
    
    def _create_growth_trends_visualization(self, growth_df: pd.DataFrame,
                                          config: Dict[str, Any]) -> str:
        """
        Create growth trends visualization
        
        Args:
            growth_df: DataFrame with growth rate data
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        # Get top countries by market value in the latest year
        latest_year = growth_df['Year'].max()
        latest_data = growth_df[growth_df['Year'] == latest_year].copy()
        
        top_n = min(8, len(latest_data))  # Limit to 8 countries for readability
        top_countries = latest_data.sort_values(by='Value', ascending=False).head(top_n)
        top_country_ids = top_countries['idGeo'].tolist()
        
        # Filter growth data to include only top countries
        top_growth_data = growth_df[growth_df['idGeo'].isin(top_country_ids)]
        
        # Create the visualization
        plt.figure(figsize=(14, 8))
        
        # Line chart of growth rates over time
        sns.lineplot(
            data=top_growth_data,
            x='Year',
            y='Growth_Rate',
            hue='Country',
            style='Country',
            markers=True,
            dashes=False,
            palette=self.color_palette[:top_n]
        )
        
        # Add horizontal line at 0%
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        
        # Customize plot
        title = config.get('title', f"{self.market_type} Growth Trends")
        title = self._replace_variables(title)
        plt.title(f"{title} (Top {top_n} Countries)", fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12)
        plt.legend(title='Country', fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the visualization
        output_file = os.path.join(self.output_dir, f"{self.market_type}_Growth_Trends.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved growth trends visualization to {output_file}")
        self.visualization_files.append(output_file)
        return output_file
    
    def _create_cagr_analysis_visualization(self, market_data: pd.DataFrame,
                                          config: Dict[str, Any]) -> str:
        """
        Create CAGR analysis visualization
        
        Args:
            market_data: DataFrame with market forecast data
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Generating CAGR analysis visualization")
        
        # Get configuration
        title = config.get('title', f"{self.market_type} CAGR Analysis")
        title = self._replace_variables(title)
        periods = config.get('periods', [])
        
        if not periods:
            # Default periods
            periods = [
                {'name': 'Short-term', 'years': 3},
                {'name': 'Mid-term', 'years': 5},
                {'name': 'Long-term', 'years': 7}
            ]
        
        # Get years range in data
        years = sorted(market_data['Year'].unique())
        min_year = min(years)
        max_year = max(years)
        total_years = max_year - min_year
        
        # Adjust periods based on available data
        adjusted_periods = []
        for period in periods:
            period_years = period.get('years', 5)
            if period_years <= total_years:
                adjusted_periods.append(period)
            else:
                logger.warning(f"Period {period['name']} ({period_years} years) exceeds available data range")
        
        if not adjusted_periods:
            logger.warning("No valid periods for CAGR analysis")
            return ""
        
        # Calculate CAGR for each country and period
        cagr_data = []
        
        for country, group in market_data.groupby(['idGeo', 'Country']):
            country_id, country_name = country
            
            # Get values for start and end years
            for period in adjusted_periods:
                period_name = period['name']
                period_years = period['years']
                
                # Calculate period start and end years
                end_year = max_year
                start_year = end_year - period_years
                
                # Get values
                start_value = group[group['Year'] == start_year]['Value'].values
                end_value = group[group['Year'] == end_year]['Value'].values
                
                if len(start_value) > 0 and len(end_value) > 0 and start_value[0] > 0 and period_years > 0:
                    # Calculate CAGR with proper validation
                    cagr = (end_value[0] / start_value[0]) ** (1 / period_years) - 1
                    
                    cagr_data.append({
                        'idGeo': country_id,
                        'Country': country_name,
                        'Period': period_name,
                        'Start_Year': start_year,
                        'End_Year': end_year,
                        'CAGR': cagr * 100  # Convert to percentage
                    })
        
        # Convert to DataFrame
        cagr_df = pd.DataFrame(cagr_data)
        
        if cagr_df.empty:
            logger.warning("No CAGR data available for visualization")
            return ""
        
        # Create the visualization
        plt.figure(figsize=(14, 10))
        
        # Get top countries by end-year value
        end_year_data = market_data[market_data['Year'] == max_year]
        top_n = min(10, len(end_year_data))
        top_countries = end_year_data.sort_values(by='Value', ascending=False).head(top_n)['Country'].tolist()
        
        # Filter CAGR data to include only top countries
        top_cagr_data = cagr_df[cagr_df['Country'].isin(top_countries)]
        
        # Create grouped bar chart
        ax = sns.barplot(
            data=top_cagr_data,
            x='Country',
            y='CAGR',
            hue='Period',
            palette=self.color_palette[:len(adjusted_periods)]
        )
        
        # Add value labels
        for i, bars in enumerate(ax.containers):
            ax.bar_label(bars, fmt='%.1f%%', fontsize=9)
        
        # Customize plot
        plt.title(title, fontsize=16)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('CAGR (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Period', fontsize=10)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the visualization
        output_file = os.path.join(self.output_dir, f"{self.market_type}_CAGR_Analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved CAGR analysis visualization to {output_file}")
        self.visualization_files.append(output_file)
        return output_file
    
    def _create_regional_analysis_visualization(self, market_data: pd.DataFrame,
                                               config: Dict[str, Any]) -> str:
        """
        Create regional analysis visualization
        
        Args:
            market_data: DataFrame with market forecast data
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Generating regional analysis visualization")
        
        # Check if data includes regions
        if 'is_region' not in market_data.columns or market_data['is_region'].sum() == 0:
            logger.warning("No regional data found for visualization")
            return ""
        
        # Get configuration
        title = config.get('title', f"{self.market_type} Regional Analysis")
        title = self._replace_variables(title)
        
        # Get specific year from config, or use the latest year
        specific_year = config.get('specific_year')
        analysis_years = config.get('analysis_years')
        years = sorted(market_data['Year'].unique())
        
        if specific_year and specific_year in years:
            years_to_analyze = [specific_year]
        elif analysis_years and isinstance(analysis_years, list):
            years_to_analyze = [year for year in analysis_years if year in years]
        else:
            # Default to latest year
            years_to_analyze = [max(years)]
        
        if not years_to_analyze:
            logger.warning("No valid years for regional analysis")
            return ""
        
        # Filter for regions only (top-level regions)
        region_data = market_data[market_data['is_region'] == True].copy()
        
        # Get top-level regions (excluding "Worldwide" or similar global totals)
        exclude_terms = ['world', 'global', 'total', 'worldwide']
        
        def is_top_level_region(row):
            return (row['is_region'] and 
                   not any(term in row['Country'].lower() for term in exclude_terms))
        
        top_regions = region_data[region_data.apply(is_top_level_region, axis=1)]
        
        if top_regions.empty:
            logger.warning("No top-level regions found for visualization")
            return ""
        
        # Create visualizations for each analysis year
        output_files = []
        
        for year in years_to_analyze:
            # Filter for the specific year
            year_data = top_regions[top_regions['Year'] == year].copy()
            
            if year_data.empty:
                logger.warning(f"No regional data found for year {year}")
                continue
            
            # Sort by value
            year_data = year_data.sort_values(by='Value', ascending=False)
            
            # Create pie chart
            plt.figure(figsize=(10, 10))
            plt.pie(
                year_data['Value'],
                labels=year_data['Country'],
                autopct='%1.1f%%',
                startangle=90,
                colors=self.color_palette[:len(year_data)],
                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                textprops={'fontsize': 12}
            )
            
            plt.title(f"{title} ({year})", fontsize=16)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
            # Save the visualization
            output_file = os.path.join(self.output_dir, f"{self.market_type}_Regional_Analysis_{year}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved regional analysis visualization for {year} to {output_file}")
            self.visualization_files.append(output_file)
            output_files.append(output_file)
        
        # If we have multiple years, create a trend visualization
        if len(years_to_analyze) > 1:
            self._create_regional_trend_chart(top_regions, years_to_analyze, config)
        
        # Return the first output file as reference
        return output_files[0] if output_files else ""
    
    def _create_regional_trend_chart(self, region_data: pd.DataFrame,
                                    years: List[int],
                                    config: Dict[str, Any]) -> str:
        """
        Create regional trend chart
        
        Args:
            region_data: DataFrame with regional data
            years: List of years to include
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        # Filter for the specified years
        trend_data = region_data[region_data['Year'].isin(years)].copy()
        
        if trend_data.empty:
            return ""
        
        # Create the visualization
        plt.figure(figsize=(14, 8))
        
        # Line chart of regional values over time
        sns.lineplot(
            data=trend_data,
            x='Year',
            y='Value',
            hue='Country',
            style='Country',
            markers=True,
            dashes=False,
            palette=self.color_palette[:len(trend_data['Country'].unique())]
        )
        
        # Format y-axis as millions/billions
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self._format_value_axis))
        
        # Customize plot
        title = config.get('title', f"{self.market_type} Regional Trends")
        title = self._replace_variables(title)
        plt.title(title, fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Market Value', fontsize=12)
        plt.legend(title='Region', fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the visualization
        output_file = os.path.join(self.output_dir, f"{self.market_type}_Regional_Trends.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved regional trends visualization to {output_file}")
        self.visualization_files.append(output_file)
        return output_file
    
    def _create_causal_influence_visualization(self, market_data: pd.DataFrame,
                                              config: Dict[str, Any]) -> str:
        """
        Create causal influence visualization
        
        Args:
            market_data: DataFrame with market forecast data
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Generating causal influence visualization")
        
        # Try to get causal strengths from indicator analyzer
        try:
            # Attempt to access causal integration via market analyzer
            from src.market_analysis.market_analyzer import MarketAnalyzer
            
            # Try to get our parent MarketAnalyzer instance by looking at the caller chain
            import inspect
            frame = inspect.currentframe()
            frames = inspect.getouterframes(frame)
            
            market_analyzer = None
            for caller_frame in frames:
                if 'self' in caller_frame.frame.f_locals:
                    self_obj = caller_frame.frame.f_locals['self']
                    if isinstance(self_obj, MarketAnalyzer):
                        market_analyzer = self_obj
                        break
            
            if market_analyzer and hasattr(market_analyzer, 'causal_integration'):
                causal_strengths = market_analyzer.causal_integration.get_causal_strengths()
            else:
                # Fallback to trying to create our own instance
                from src.indicators.causal_indicator_integration import CausalIndicatorIntegration
                from src.indicators.indicator_analyzer import IndicatorAnalyzer
                
                indicator_analyzer = IndicatorAnalyzer(self.config_manager, self.data_loader)
                causal_integration = CausalIndicatorIntegration(
                    self.config_manager, self.data_loader, indicator_analyzer
                )
                
                # See if we have pre-computed causal strengths
                causal_strengths = causal_integration.get_causal_strengths()
                
                # If no pre-computed strengths, we can't calculate them here
                if not causal_strengths:
                    logger.warning("No causal strengths available for visualization")
                    return ""
                
        except Exception as e:
            logger.warning(f"Error accessing causal integration: {str(e)}")
            return ""
        
        if not causal_strengths:
            logger.warning("No causal strengths available for visualization")
            return ""
        
        # Create DataFrame for visualization
        data = []
        for ind_name, strength in causal_strengths.items():
            data.append({
                'Indicator': ind_name,
                'Causal Strength': strength
            })
        
        causal_df = pd.DataFrame(data)
        
        # Sort by causal strength
        causal_df = causal_df.sort_values('Causal Strength', ascending=False)
        
        # Get top N indicators
        top_n = config.get('top_n_indicators', 10)
        top_indicators = causal_df.head(top_n)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        
        # Horizontal bar chart for better readability with indicator names
        ax = sns.barplot(
            y='Indicator',
            x='Causal Strength',
            data=top_indicators,
            palette=sns.color_palette("YlOrRd", n_colors=len(top_indicators))
        )
        
        # Add value labels
        for i, v in enumerate(top_indicators['Causal Strength']):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        # Customize plot
        title = config.get('title', f"{self.market_type} Causal Indicator Influence")
        title = self._replace_variables(title)
        plt.title(title, fontsize=16)
        plt.xlabel('Causal Strength (0-1 scale)', fontsize=12)
        plt.ylabel('Indicator', fontsize=12)
        plt.xlim(0, 1.1 * top_indicators['Causal Strength'].max())
        plt.tight_layout()
        
        # Save the visualization
        output_file = os.path.join(self.output_dir, f"{self.market_type}_Causal_Influence.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved causal influence visualization to {output_file}")
        self.visualization_files.append(output_file)
        
        # Create causal network graph if available
        try:
            if market_analyzer and hasattr(market_analyzer, 'causal_integration') and \
               hasattr(market_analyzer.causal_integration, 'causal_graph'):
                causal_graph = market_analyzer.causal_integration.causal_graph
                if causal_graph and len(causal_graph.edges()) > 0:
                    self._create_causal_network_visualization(causal_graph, config)
        except Exception as e:
            logger.warning(f"Error creating causal network visualization: {str(e)}")
        
        return output_file
    
    def _create_causal_network_visualization(self, causal_graph: nx.DiGraph,
                                            config: Dict[str, Any]) -> str:
        """
        Create causal network visualization
        
        Args:
            causal_graph: NetworkX directed graph with causal relationships
            config: Visualization configuration
            
        Returns:
            Path to the generated visualization file
        """
        logger.info("Generating causal network visualization")
        
        plt.figure(figsize=(14, 10))
        
        # Position nodes using a hierarchical layout
        pos = nx.spring_layout(causal_graph, seed=42)
        
        # Define node colors by type
        node_colors = []
        for node in causal_graph.nodes():
            node_type = causal_graph.nodes[node].get('type', '')
            if node_type == 'target':
                node_colors.append('red')
            elif node_type == 'indicator':
                node_colors.append('skyblue')
            elif node_type == 'lag':
                node_colors.append('lightgreen')
            else:
                node_colors.append('gray')
        
        # Get edge weights for thickness
        edge_weights = [causal_graph[u][v].get('weight', 0.1) * 3 for u, v in causal_graph.edges()]
        
        # Draw the graph
        nx.draw_networkx(
            causal_graph,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            font_size=10,
            width=edge_weights,
            edge_color='gray',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Market Value'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='Indicator'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Lagged Variable')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        title = config.get('title', f"{self.market_type} Causal Network")
        title = self._replace_variables(title)
        plt.title(title, fontsize=16)
        
        # Remove axes
        plt.axis('off')
        
        # Save the visualization
        output_file = os.path.join(self.output_dir, f"{self.market_type}_Causal_Network.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved causal network visualization to {output_file}")
        self.visualization_files.append(output_file)
        return output_file
    
    def generate_excel_report(self, market_data: pd.DataFrame) -> str:
        """
        Generate a comprehensive Excel report
        
        Args:
            market_data: DataFrame with market forecast data
            
        Returns:
            Path to the generated Excel file
        """
        logger.info("Generating Excel report")
        
        # Create Excel writer
        excel_path = os.path.join(self.output_dir, f"{self.market_type}_Market_Forecast_Report.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Add report sheets
                self._add_summary_sheet(workbook, writer, market_data)
                self._add_market_size_sheet(workbook, writer, market_data)
                self._add_growth_rates_sheet(workbook, writer, market_data)
                self._add_region_sheet(workbook, writer, market_data)
                
                # Try to add causal analysis sheet if available
                try:
                    self._add_causal_analysis_sheet(workbook, writer, market_data)
                except Exception as e:
                    logger.warning(f"Error adding causal analysis sheet: {str(e)}")
            
            logger.info(f"Generated Excel report: {excel_path}")
            self.visualization_files.append(excel_path)
            return excel_path
            
        except Exception as e:
            logger.error(f"Error generating Excel report: {str(e)}")
            return ""
    
    def _add_summary_sheet(self, workbook, writer, market_data: pd.DataFrame) -> None:
        """
        Add market summary sheet to Excel report
        
        Args:
            workbook: Excel workbook
            writer: Excel writer
            market_data: DataFrame with market forecast data
        """
        # Create summary sheet
        summary_sheet = workbook.add_worksheet('Market Summary')
        
        # Add formats
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        value_format = workbook.add_format({
            'num_format': '$#,##0.00',
            'border': 1
        })
        
        percent_format = workbook.add_format({
            'num_format': '0.00%',
            'border': 1
        })
        
        text_format = workbook.add_format({
            'border': 1
        })
        
        # Add title
        summary_sheet.merge_range('A1:F1', f"{self.market_type} Market Forecast Summary", title_format)
        
        # Add summary statistics
        row = 2
        
        # Market period
        years = sorted(market_data['Year'].unique())
        first_year = min(years)
        last_year = max(years)
        
        summary_sheet.write(row, 0, 'Analysis Period', header_format)
        summary_sheet.merge_range(f'B{row+1}:F{row+1}', f"{first_year} - {last_year}", text_format)
        row += 2
        
        # Global values
        first_year_data = market_data[market_data['Year'] == first_year]
        last_year_data = market_data[market_data['Year'] == last_year]
        
        # Filter to exclude regions for base calculations
        if 'is_region' in market_data.columns:
            first_year_data = first_year_data[~first_year_data['is_region']]
            last_year_data = last_year_data[~last_year_data['is_region']]
        
        first_year_total = first_year_data['Value'].sum()
        last_year_total = last_year_data['Value'].sum()
        
        # Calculate global CAGR with proper validation
        years_diff = last_year - first_year
        if years_diff > 0 and first_year_total > 0:
            global_cagr = (last_year_total / first_year_total) ** (1 / years_diff) - 1
        else:
            logger.warning(f"Cannot calculate global CAGR: years_diff={years_diff}, first_year_total={first_year_total}")
            global_cagr = 0.0
        
        summary_sheet.write(row, 0, 'Market Size', header_format)
        summary_sheet.write(row, 1, f"First Year ({first_year})", header_format)
        summary_sheet.write(row, 2, f"Last Year ({last_year})", header_format)
        summary_sheet.write(row, 3, 'Growth Multiple', header_format)
        summary_sheet.write(row, 4, 'CAGR', header_format)
        
        row += 1
        summary_sheet.write(row, 0, 'Global Market', text_format)
        summary_sheet.write(row, 1, first_year_total, value_format)
        summary_sheet.write(row, 2, last_year_total, value_format)
        summary_sheet.write(row, 3, last_year_total / first_year_total, text_format)
        summary_sheet.write(row, 4, global_cagr, percent_format)
        
        row += 2
        
        # Top countries table
        summary_sheet.write(row, 0, 'Top Countries', header_format)
        summary_sheet.merge_range(f'A{row+1}:F{row+1}', f"Top 10 Countries by Market Size ({last_year})", header_format)
        
        row += 2
        # Table headers
        summary_sheet.write(row, 0, 'Country', header_format)
        summary_sheet.write(row, 1, f"Value ({last_year})", header_format)
        summary_sheet.write(row, 2, 'Market Share', header_format)
        summary_sheet.write(row, 3, f"Value ({first_year})", header_format)
        summary_sheet.write(row, 4, 'Growth Multiple', header_format)
        summary_sheet.write(row, 5, 'CAGR', header_format)
        
        row += 1
        
        # Get top 10 countries by last year value
        top_countries = last_year_data.sort_values(by='Value', ascending=False).head(10)
        
        for _, country_row in top_countries.iterrows():
            country_id = country_row['idGeo']
            country_name = country_row['Country']
            last_year_value = country_row['Value']
            
            # Get first year value for this country
            first_year_value = first_year_data[first_year_data['idGeo'] == country_id]['Value'].values
            
            if len(first_year_value) > 0:
                first_year_value = first_year_value[0]
                growth_multiple = last_year_value / first_year_value
                country_cagr = (growth_multiple) ** (1 / years_diff) - 1
            else:
                first_year_value = 0
                growth_multiple = 0
                country_cagr = 0
            
            # Calculate market share
            market_share = last_year_value / last_year_total
            
            # Add to table
            summary_sheet.write(row, 0, country_name, text_format)
            summary_sheet.write(row, 1, last_year_value, value_format)
            summary_sheet.write(row, 2, market_share, percent_format)
            summary_sheet.write(row, 3, first_year_value, value_format)
            summary_sheet.write(row, 4, growth_multiple, text_format)
            summary_sheet.write(row, 5, country_cagr, percent_format)
            
            row += 1
        
        # Set column widths
        summary_sheet.set_column('A:A', 20)
        summary_sheet.set_column('B:F', 15)
    
    def _add_market_size_sheet(self, workbook, writer, market_data: pd.DataFrame) -> None:
        """
        Add market size sheet to Excel report
        
        Args:
            workbook: Excel workbook
            writer: Excel writer
            market_data: DataFrame with market forecast data
        """
        # Create pivot table with years as columns
        pivot_data = market_data.pivot_table(
            index=['idGeo', 'Country'],
            columns='Year',
            values='Value',
            aggfunc='sum'
        )
        
        # Write to Excel
        pivot_data.to_excel(writer, sheet_name='Market Size')
        
        # Get the worksheet
        worksheet = writer.sheets['Market Size']
        
        # Add title
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        # Determine range based on number of years
        years = sorted(market_data['Year'].unique())
        end_col = chr(ord('A') + len(years) + 1)
        worksheet.merge_range(f'A1:{end_col}1', f"{self.market_type} Market Size by Country", title_format)
        
        # Format the sheet
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply formats
        worksheet.set_column('A:A', 20)  # ID column
        worksheet.set_column('B:B', 25)  # Country name
        worksheet.set_column(f'C:{end_col}', 15)  # Year columns
    
    def _add_growth_rates_sheet(self, workbook, writer, market_data: pd.DataFrame) -> None:
        """
        Add growth rates sheet to Excel report
        
        Args:
            workbook: Excel workbook
            writer: Excel writer
            market_data: DataFrame with market forecast data
        """
        # Calculate year-over-year growth rates
        growth_data = []
        
        # Group by country
        for country, group in market_data.groupby(['idGeo', 'Country']):
            country_id, country_name = country
            
            # Sort by year
            group = group.sort_values(by='Year')
            
            # Calculate growth rate
            group['Growth_Rate'] = group['Value'].pct_change() * 100
            
            # Add to results (excluding first year which has no growth rate)
            growth_data.append(group.iloc[1:])
        
        # Combine all data
        growth_df = pd.concat(growth_data, ignore_index=True)
        
        # Create pivot table with years as columns
        pivot_data = growth_df.pivot_table(
            index=['idGeo', 'Country'],
            columns='Year',
            values='Growth_Rate',
            aggfunc='mean'
        )
        
        # Write to Excel
        pivot_data.to_excel(writer, sheet_name='Growth Rates')
        
        # Get the worksheet
        worksheet = writer.sheets['Growth Rates']
        
        # Add title
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        # Determine range based on number of years
        years = sorted(growth_df['Year'].unique())
        end_col = chr(ord('A') + len(years) + 1)
        worksheet.merge_range(f'A1:{end_col}1', f"{self.market_type} Growth Rates by Country (%)", title_format)
        
        # Format the sheet
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply formats
        worksheet.set_column('A:A', 20)  # ID column
        worksheet.set_column('B:B', 25)  # Country name
        worksheet.set_column(f'C:{end_col}', 15)  # Year columns
    
    def _add_region_sheet(self, workbook, writer, market_data: pd.DataFrame) -> None:
        """
        Add regional analysis sheet to Excel report
        
        Args:
            workbook: Excel workbook
            writer: Excel writer
            market_data: DataFrame with market forecast data
        """
        # Check if regional data is available
        if 'is_region' not in market_data.columns or market_data['is_region'].sum() == 0:
            return
        
        # Filter for regions only
        region_data = market_data[market_data['is_region'] == True].copy()
        
        # Create pivot table with years as columns
        pivot_data = region_data.pivot_table(
            index=['idGeo', 'Country'],
            columns='Year',
            values='Value',
            aggfunc='sum'
        )
        
        # Write to Excel
        pivot_data.to_excel(writer, sheet_name='Regional Analysis')
        
        # Get the worksheet
        worksheet = writer.sheets['Regional Analysis']
        
        # Add title
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        # Determine range based on number of years
        years = sorted(region_data['Year'].unique())
        end_col = chr(ord('A') + len(years) + 1)
        worksheet.merge_range(f'A1:{end_col}1', f"{self.market_type} Regional Analysis", title_format)
        
        # Format the sheet
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply formats
        worksheet.set_column('A:A', 20)  # ID column
        worksheet.set_column('B:B', 25)  # Region name
        worksheet.set_column(f'C:{end_col}', 15)  # Year columns
    
    def _add_causal_analysis_sheet(self, workbook, writer, market_data: pd.DataFrame) -> None:
        """
        Add causal analysis sheet to Excel report
        
        Args:
            workbook: Excel workbook
            writer: Excel writer
            market_data: DataFrame with market forecast data
        """
        # Try to get causal strengths from indicator analyzer
        try:
            # Attempt to access causal integration via market analyzer
            from src.market_analysis.market_analyzer import MarketAnalyzer
            
            # Try to get our parent MarketAnalyzer instance by looking at the caller chain
            import inspect
            frame = inspect.currentframe()
            frames = inspect.getouterframes(frame)
            
            market_analyzer = None
            for caller_frame in frames:
                if 'self' in caller_frame.frame.f_locals:
                    self_obj = caller_frame.frame.f_locals['self']
                    if isinstance(self_obj, MarketAnalyzer):
                        market_analyzer = self_obj
                        break
            
            if market_analyzer and hasattr(market_analyzer, 'causal_integration'):
                causal_strengths = market_analyzer.causal_integration.get_causal_strengths()
            else:
                # Fallback to trying to create our own instance
                from src.indicators.causal_indicator_integration import CausalIndicatorIntegration
                from src.indicators.indicator_analyzer import IndicatorAnalyzer
                
                indicator_analyzer = IndicatorAnalyzer(self.config_manager, self.data_loader)
                causal_integration = CausalIndicatorIntegration(
                    self.config_manager, self.data_loader, indicator_analyzer
                )
                
                # See if we have pre-computed causal strengths
                causal_strengths = causal_integration.get_causal_strengths()
        except Exception as e:
            logger.warning(f"Error accessing causal integration: {str(e)}")
            return
        
        if not causal_strengths:
            return
        
        # Create DataFrame for analysis
        data = []
        for ind_name, strength in causal_strengths.items():
            data.append({
                'Indicator': ind_name,
                'Causal Strength': strength
            })
        
        causal_df = pd.DataFrame(data)
        
        # Sort by causal strength
        causal_df = causal_df.sort_values('Causal Strength', ascending=False)
        
        # Create the sheet
        causal_df.to_excel(writer, sheet_name='Causal Analysis', index=False)
        
        # Get the worksheet
        worksheet = writer.sheets['Causal Analysis']
        
        # Add title
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        worksheet.merge_range('A1:B1', f"{self.market_type} Causal Indicator Analysis", title_format)
        
        # Format the sheet
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply header format
        worksheet.write(1, 0, 'Indicator', header_format)
        worksheet.write(1, 1, 'Causal Strength', header_format)
        
        # Write the data
        for i, (_, row) in enumerate(causal_df.iterrows()):
            worksheet.write(i + 2, 0, row['Indicator'])
            worksheet.write(i + 2, 1, row['Causal Strength'])
        
        # Set column widths
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 15)
        
        # Add explanation of causal strength
        row = len(causal_df) + 4
        
        explanation_format = workbook.add_format({
            'text_wrap': True
        })
        
        worksheet.merge_range(f'A{row}:B{row}', 'About Causal Strength:', header_format)
        row += 1
        
        explanation = (
            "Causal Strength measures the estimated causal impact of an indicator on market outcomes. "
            "Unlike correlation, which only shows association, causal strength attempts to quantify "
            "the actual influence an indicator has on market values. Values range from 0 (no causal effect) "
            "to 1 (strong causal effect). This analysis combines multiple causal inference methods "
            "including Granger causality tests, conditional independence tests, and feature importance "
            "from machine learning models."
        )
        
        worksheet.merge_range(f'A{row}:B{row+3}', explanation, explanation_format)
    
    def _format_value_axis(self, x, pos):
        """Format value axis to show values in K, M, B"""
        if x >= 1e9:
            return f'{x/1e9:.1f}B'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.1f}'
    
    def _replace_variables(self, text: str) -> str:
        """
        Replace variables in text with their values
        
        Args:
            text: Text with variables to replace
            
        Returns:
            Text with variables replaced
        """
        # Replace ${market_type} with actual market type
        text = text.replace('${market_type}', self.market_type)
        
        return text