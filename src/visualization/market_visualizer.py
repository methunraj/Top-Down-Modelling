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
        
        # Get project and output settings
        self.project_info = self.config_manager.get_project_info()
        self.output_settings = self.config_manager.get_output_settings()
        
        # Set up formatting
        self.market_type = self.project_info.get('market_type', 'Market')
        self.save_path = self.output_settings.get('save_path', 'data/output/')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)
        
        # Set default style for plots
        plt.style.use('seaborn-v0_8-pastel')
        sns.set_color_codes("pastel")
        
        # Configure plot sizes and styles
        self.figure_size = (12, 8)
        self.dpi = 100
        self.title_fontsize = 16
        self.axis_fontsize = 12
        self.legend_fontsize = 10
    
    def generate_all_visualizations(self, market_data: pd.DataFrame) -> List[str]:
        """
        Generate all configured visualizations
        
        Args:
            market_data: DataFrame with distributed market values
            
        Returns:
            List of file paths to generated visualizations
        """
        # Get visualization types from configuration
        visualization_settings = self.output_settings.get('visualizations', {})
        visualization_types = visualization_settings.get('types', [])
        
        if not visualization_types:
            logger.warning("No visualization types configured")
            return []
        
        generated_files = []
        
        # Generate each visualization type
        for viz_config in visualization_types:
            viz_name = viz_config.get('name', '')
            
            if not viz_name:
                continue
            
            # Replace placeholders in title
            title = viz_config.get('title', f'{self.market_type} Analysis')
            title = self._replace_placeholders(title)
            
            try:
                if viz_name == 'market_size':
                    file_path = self.create_market_size_chart(
                        market_data, 
                        title=title,
                        top_n=viz_config.get('top_n_countries', 10)
                    )
                    generated_files.append(file_path)
                
                elif viz_name == 'growth_rates':
                    generated_files = self.create_growth_rate_chart(
                        market_data, 
                        title=title,
                        top_n=viz_config.get('top_n_countries', 15)
                    )
                    generated_files.extend(generated_files)
                
                elif viz_name == 'cagr_analysis':
                    periods = viz_config.get('periods', [])
                    file_path = self.create_cagr_analysis(
                        market_data, 
                        title=title,
                        periods=periods
                    )
                    generated_files.append(file_path)
                
                elif viz_name == 'market_share':
                    file_path = self.create_market_share_chart(
                        market_data, 
                        title=title,
                        top_n=viz_config.get('top_n_countries', 10)
                    )
                    generated_files.append(file_path)
                
                elif viz_name == 'forecast_comparison':
                    file_path = self.create_forecast_comparison_chart(
                        market_data, 
                        title=title,
                        years=viz_config.get('years', [])
                    )
                    generated_files.append(file_path)
                
                elif viz_name == 'regional_analysis':
                    file_path = self.create_regional_analysis(
                        market_data, 
                        title=title,
                        specific_year=viz_config.get('specific_year'),
                        analysis_years=viz_config.get('analysis_years')
                    )
                    generated_files.append(file_path)
                
                elif viz_name == 'top_countries':
                    file_path = self.create_top_countries_chart(
                        market_data,
                        title=title,
                        top_n=viz_config.get('top_n_countries', 10),
                        specific_countries=viz_config.get('specific_countries'),
                        year=viz_config.get('year')
                    )
                    generated_files.append(file_path)
                
                else:
                    logger.warning(f"Unknown visualization type: {viz_name}")
            
            except Exception as e:
                logger.error(f"Error creating visualization '{viz_name}': {str(e)}")
        
        return generated_files
    
    def create_market_size_chart(self, market_data: pd.DataFrame, title: str = None, 
                                top_n: int = 10) -> str:
        """
        Create a stacked area chart of market size by country
        
        Args:
            market_data: DataFrame with distributed market values
            title: Chart title
            top_n: Number of top countries to show separately
            
        Returns:
            Path to the saved visualization file
        """
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
        
        # Use the final year for top countries selection
        final_year = market_data['Year'].max()
        final_year_data = market_data[market_data['Year'] == final_year].copy()
        
        # Get top N countries by value in final year
        top_countries = final_year_data.sort_values(by='Value', ascending=False).head(top_n)[id_col].tolist()
        
        # Prepare data for visualization
        pivot_data = []
        
        for year in sorted(market_data['Year'].unique()):
            year_data = market_data[market_data['Year'] == year].copy()
            
            # Split into top countries and others
            top_data = year_data[year_data[id_col].isin(top_countries)]
            other_data = year_data[~year_data[id_col].isin(top_countries)]
            
            # Sum values for other countries
            other_value = other_data['Value'].sum()
            
            # Add top countries data
            for _, row in top_data.iterrows():
                pivot_data.append({
                    'Year': year,
                    'Country': row[country_col],
                    'Value': row['Value']
                })
            
            # Add other countries
            pivot_data.append({
                'Year': year,
                'Country': 'Other Countries',
                'Value': other_value
            })
        
        # Convert to DataFrame
        pivot_df = pd.DataFrame(pivot_data)
        
        # Pivot for stacked chart
        chart_data = pivot_df.pivot(index='Year', columns='Country', values='Value')
        
        # Create the plot
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        ax = chart_data.plot(kind='area', stacked=True, figsize=self.figure_size, alpha=0.7)
        
        # Format axes
        plt.title(title or f'{self.market_type} Market Size by Country', fontsize=self.title_fontsize)
        plt.xlabel('Year', fontsize=self.axis_fontsize)
        plt.ylabel('Market Size', fontsize=self.axis_fontsize)
        
        # Format y-axis to show values in billions
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x/1e9:.1f}B'))
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=self.legend_fontsize, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.save_path, f'{self.market_type}_Market_Size_by_Country.png')
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Created market size chart: {filename}")
        
        return filename
    
    def create_growth_rate_chart(self, market_data: pd.DataFrame, title: str = None,
                                top_n: int = 15) -> List[str]:
        """
        Create separate charts for YoY and CAGR growth rates by country
        
        Args:
            market_data: DataFrame with distributed market values
            title: Chart title
            top_n: Number of top countries to show
            
        Returns:
            List of paths to the saved visualization files [cagr_file, yoy_file]
        """
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
        
        # Calculate CAGR for the forecast period
        years = sorted(market_data['Year'].unique())
        forecast_start = min(years)  # First year (historical)
        forecast_end = max(years)    # Last year (forecast)
        
        # Filter data for start and end years
        start_data = market_data[market_data['Year'] == forecast_start].copy()
        end_data = market_data[market_data['Year'] == forecast_end].copy()
        
        # Merge start and end data
        merged = pd.merge(
            end_data[[id_col, country_col, 'Value']],
            start_data[[id_col, 'Value']],
            on=id_col,
            how='inner',
            suffixes=('_end', '_start')
        )
        
        # Calculate CAGR
        years_diff = forecast_end - forecast_start
        merged['CAGR'] = (merged['Value_end'] / merged['Value_start']) ** (1 / years_diff) - 1
        
        # Get top countries by end value
        top_countries = merged.sort_values(by='Value_end', ascending=False).head(top_n)
        
        generated_files = []
        
        # Create CAGR Chart
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Sort countries by CAGR for better visualization
        top_countries = top_countries.sort_values(by='CAGR', ascending=False)
        colors = ['#ff9999' if x > 0 else '#99ccff' for x in top_countries['CAGR']]
        
        plt.bar(top_countries[country_col], top_countries['CAGR'] * 100, color=colors)
        
        # Format CAGR plot
        plt.title(f'{self.market_type} CAGR Growth Rates ({forecast_start}-{forecast_end})', 
                 fontsize=self.title_fontsize)
        plt.xlabel('Country', fontsize=self.axis_fontsize)
        plt.ylabel('CAGR (%)', fontsize=self.axis_fontsize)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right', fontsize=self.axis_fontsize - 2)
        
        # Add value labels on CAGR bars
        for i, v in enumerate(top_countries['CAGR'] * 100):
            plt.text(i, v + (1 if v >= 0 else -3), f'{v:.1f}%', 
                     ha='center', fontsize=self.axis_fontsize - 4)
        
        plt.tight_layout()
        
        # Save CAGR chart
        cagr_filename = os.path.join(self.save_path, f'{self.market_type}_CAGR_Growth_Rates_{forecast_start}-{forecast_end}.png')
        plt.savefig(cagr_filename)
        plt.close()
        
        logger.info(f"Created CAGR growth rate chart: {cagr_filename}")
        generated_files.append(cagr_filename)
        
        # Calculate YoY growth rates for each country
        yoy_data = []
        for country_id in top_countries[id_col]:
            country_data = market_data[market_data[id_col] == country_id].sort_values('Year')
            if len(country_data) > 1:
                country_data['YoY_Growth'] = country_data['Value'].pct_change() * 100
                yoy_data.append(country_data)
        
        if yoy_data:
            yoy_df = pd.concat(yoy_data)
            
            # Create YoY Growth Chart
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot YoY growth rates for each country
            for country in top_countries[country_col]:
                country_data = yoy_df[yoy_df[country_col] == country]
                plt.plot(country_data['Year'], country_data['YoY_Growth'], 
                        marker='o', label=country, linewidth=2)
            
            # Format YoY plot
            plt.title(f'{self.market_type} Year-over-Year Growth Rates', fontsize=self.title_fontsize)
            plt.xlabel('Year', fontsize=self.axis_fontsize)
            plt.ylabel('YoY Growth Rate (%)', fontsize=self.axis_fontsize)
            plt.grid(True, alpha=0.3)
            
            # Add legend with better placement
            plt.legend(fontsize=self.legend_fontsize, bbox_to_anchor=(1.05, 1), 
                      loc='upper left', borderaxespad=0.)
            
            plt.tight_layout()
            
            # Save YoY chart
            yoy_filename = os.path.join(self.save_path, f'{self.market_type}_YoY_Growth_Rates_{forecast_start}-{forecast_end}.png')
            plt.savefig(yoy_filename, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created YoY growth rate chart: {yoy_filename}")
            generated_files.append(yoy_filename)
        
        return generated_files
    
    def create_cagr_analysis(self, market_data: pd.DataFrame, title: str = None,
                            periods: List[Dict[str, Any]] = None) -> str:
        """
        Create a CAGR analysis chart for different time periods
        
        Args:
            market_data: DataFrame with distributed market values
            title: Chart title
            periods: List of period configurations
            
        Returns:
            Path to the saved visualization file
        """
        # Default periods if not provided
        if not periods:
            periods = [
                {'name': 'Short-term', 'years': 3},
                {'name': 'Mid-term', 'years': 5},
                {'name': 'Long-term', 'years': 7}
            ]
        
        # Get years
        years = sorted(market_data['Year'].unique())
        last_year = max(years)
        
        # Prepare data for each period
        cagr_data = []
        
        for period in periods:
            period_years = period.get('years', 5)
            period_name = period.get('name', f'{period_years}-Year')
            
            # Find start year for the period
            if last_year - period_years in years:
                start_year = last_year - period_years
            else:
                # Find closest available year
                available_years = [y for y in years if y < last_year]
                if not available_years:
                    logger.warning(f"No suitable start year for period {period_name}")
                    continue
                start_year = max(available_years)
            
            # Calculate global CAGR for this period
            start_total = market_data[market_data['Year'] == start_year]['Value'].sum()
            end_total = market_data[market_data['Year'] == last_year]['Value'].sum()
            
            years_diff = last_year - start_year
            cagr = (end_total / start_total) ** (1 / years_diff) - 1
            
            cagr_data.append({
                'Period': period_name,
                'CAGR': cagr * 100,
                'Years': f'{start_year}-{last_year}'
            })
        
        # Convert to DataFrame
        cagr_df = pd.DataFrame(cagr_data)
        
        if cagr_df.empty:
            logger.warning("No CAGR data available for analysis")
            return ""
        
        # Sort by period years (if present in period name)
        def extract_years(period_name):
            match = re.search(r'(\d+)', period_name)
            return int(match.group(1)) if match else 0
        
        cagr_df['sort_key'] = cagr_df['Period'].apply(extract_years)
        cagr_df = cagr_df.sort_values(by='sort_key')
        
        # Create the plot
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Create bar chart
        ax = plt.bar(cagr_df['Period'], cagr_df['CAGR'], 
                    color=sns.color_palette("Blues_d", len(cagr_df)))
        
        # Format axes
        plt.title(title or f'{self.market_type} CAGR Analysis', fontsize=self.title_fontsize)
        plt.xlabel('Time Period', fontsize=self.axis_fontsize)
        plt.ylabel('CAGR (%)', fontsize=self.axis_fontsize)
        plt.grid(axis='y', alpha=0.3)
        
        # Add year range labels
        for i, (_, row) in enumerate(cagr_df.iterrows()):
            plt.text(i, row['CAGR'] + 0.5, row['Years'], 
                    ha='center', fontsize=self.axis_fontsize - 4)
        
        # Add value labels on bars
        for i, v in enumerate(cagr_df['CAGR']):
            plt.text(i, v / 2, f'{v:.1f}%', 
                    ha='center', fontsize=self.axis_fontsize, color='white')
        
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.save_path, f'{self.market_type}_CAGR_Analysis.png')
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Created CAGR analysis chart: {filename}")
        
        return filename
    
    def create_market_share_chart(self, market_data: pd.DataFrame, title: str = None,
                                 top_n: int = 10) -> str:
        """
        Create a pie chart of market shares for the final forecast year
        
        Args:
            market_data: DataFrame with distributed market values
            title: Chart title
            top_n: Number of top countries to show separately
            
        Returns:
            Path to the saved visualization file
        """
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
        
        # Get available years
        years = sorted(market_data['Year'].unique())
        start_year = min(years)
        end_year = max(years)
        
        # Use the final year data
        final_year_data = market_data[market_data['Year'] == end_year].copy()
        
        # Calculate market shares
        total_value = final_year_data['Value'].sum()
        final_year_data['Share'] = final_year_data['Value'] / total_value * 100
        
        # Get top N countries
        top_data = final_year_data.sort_values(by='Value', ascending=False).head(top_n)
        
        # Combine remaining countries as "Others"
        other_data = final_year_data.sort_values(by='Value', ascending=False)[top_n:]
        other_share = other_data['Share'].sum()
        
        # Prepare data for pie chart
        labels = top_data[country_col].tolist()
        sizes = top_data['Share'].tolist()
        
        if other_share > 0:
            labels.append('Other Countries')
            sizes.append(other_share)
        
        # Create the plot
        plt.figure(figsize=self.figure_size, dpi=self.dpi)
        
        # Use a custom colormap
        colors = plt.cm.Spectral(np.linspace(0, 1, len(labels)))
        
        # Create pie chart
        plt.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors, 
               wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True})
        
        # Add legend
        plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=self.legend_fontsize)
        
        # Format chart
        plt.title(title or f'{self.market_type} Market Share ({start_year}-{end_year})', fontsize=self.title_fontsize)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.tight_layout()
        
        # Save the figure with year range in filename
        filename = os.path.join(self.save_path, f'{self.market_type}_Market_Share_{start_year}-{end_year}.png')
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Created market share chart: {filename}")
        
        return filename
    
    def create_forecast_comparison_chart(self, market_data: pd.DataFrame, title: str = None,
                                        years: List[int] = None) -> str:
        """
        Create a bar chart comparing market sizes for specific years
        
        Args:
            market_data: DataFrame with distributed market values
            title: Chart title
            years: List of years to compare (defaults to first, middle, and last)
            
        Returns:
            Path to the saved visualization file
        """
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
        
        # Get all years
        all_years = sorted(market_data['Year'].unique())
        
        # Default to first, middle, and last year if not specified
        if not years:
            if len(all_years) < 3:
                years = all_years
            else:
                years = [all_years[0], all_years[len(all_years) // 2], all_years[-1]]
        
        # Filter only available years
        years = [y for y in years if y in all_years]
        
        if not years:
            logger.warning("No valid years for forecast comparison")
            return ""
        
        # Prepare data for comparison
        comparison_data = []
        
        # Get top 5 countries from the last year
        last_year = max(years)
        last_year_data = market_data[market_data['Year'] == last_year].copy()
        top_countries = last_year_data.sort_values(by='Value', ascending=False).head(5)[id_col].tolist()
        
        # Get data for selected years and top countries
        for year in years:
            year_data = market_data[market_data['Year'] == year].copy()
            
            # Get global total
            global_total = year_data['Value'].sum()
            comparison_data.append({
                'Year': str(year),
                'Country': 'Global Total',
                'Value': global_total
            })
            
            # Get data for top countries
            top_data = year_data[year_data[id_col].isin(top_countries)]
            
            for _, row in top_data.iterrows():
                comparison_data.append({
                    'Year': str(year),
                    'Country': row[country_col],
                    'Value': row['Value']
                })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Pivot for grouped bar chart
        chart_data = comparison_df.pivot(index='Country', columns='Year', values='Value')
        
        # Ensure Global Total is first
        if 'Global Total' in chart_data.index:
            chart_data = pd.concat([
                chart_data.loc[['Global Total']],
                chart_data.drop('Global Total')
            ])
        
        # Create the plot
        plt.figure(figsize=(14, 8), dpi=self.dpi)
        
        # Create grouped bar chart
        ax = chart_data.plot(kind='bar', figsize=(14, 8))
        
        # Format axes
        plt.title(title or f'{self.market_type} Market Forecast Comparison', fontsize=self.title_fontsize)
        plt.xlabel('', fontsize=self.axis_fontsize)
        plt.ylabel('Market Size', fontsize=self.axis_fontsize)
        
        # Format y-axis to show values in billions
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x/1e9:.1f}B'))
        
        plt.grid(axis='y', alpha=0.3)
        plt.legend(fontsize=self.legend_fontsize, title='Year')
        
        # Add value labels with proper formatting
        def format_value(x):
            if x >= 1e9:
                return f'${x/1e9:.1f}B'
            else:
                return f'${x/1e6:.1f}M'
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, 
                        labels=[format_value(v) for v in container.datavalues],
                        fontsize=self.axis_fontsize - 4,
                        label_type='edge',
                        padding=4)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.save_path, f'{self.market_type}_Forecast_Comparison.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created forecast comparison chart: {filename}")
        
        return filename
    
    def create_regional_analysis(self, market_data: pd.DataFrame, title: str = None, 
                               specific_year: int = None, analysis_years: list = None) -> str:
        """
        Create a regional analysis chart showing market shares by region
        
        Args:
            market_data: DataFrame with distributed market values
            title: Chart title
            specific_year: Optional specific year to show regional shares for
            analysis_years: Optional list of years to analyze
            
        Returns:
            Path to the saved visualization file or list of file paths
        """
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
        
        # Get regional configuration
        viz_config = self.config_manager.get_visualization_config('regional_analysis')
        regions = viz_config.get('regions', [])
        
        if not regions:
            logger.warning("No regional configuration found")
            return ""
        
        # Create a mapping of countries to regions
        country_to_region = {}
        for region in regions:
            for country in region.get('countries', []):
                country_to_region[country] = region['name']
        
        # Add region information to market data
        market_data['Region'] = market_data[country_col].map(country_to_region)
        
        # Get available years
        available_years = sorted(market_data['Year'].unique())
        
        # Handle analysis years if provided
        if analysis_years:
            years_to_analyze = [year for year in analysis_years if year in available_years]
            if not years_to_analyze:
                logger.warning(f"None of the specified analysis years {analysis_years} found in data")
                return ""
        elif specific_year:
            if specific_year not in available_years:
                logger.warning(f"Specified year {specific_year} not found in data")
                return ""
            years_to_analyze = [specific_year]
        else:
            years_to_analyze = [max(available_years)]  # Default to most recent year
        
        generated_files = []
        
        # Create horizontal bar charts for each year
        for year in years_to_analyze:
            year_data = market_data[market_data['Year'] == year].copy()
            total_value = year_data['Value'].sum()
            
            # Group by region and calculate shares and values
            region_data = year_data.groupby('Region').agg({
                'Value': 'sum'
            }).reset_index()
            
            region_data['Share'] = region_data['Value'] / total_value * 100
            region_data = region_data.sort_values('Share', ascending=True)  # For bottom-to-top display
            
            # Create the plot
            plt.figure(figsize=(12, 6), dpi=self.dpi)
            
            # Create horizontal bar chart
            bars = plt.barh(region_data['Region'], region_data['Share'], 
                          color=plt.cm.Spectral(np.linspace(0, 1, len(region_data))))
            
            # Format chart
            plt.title(f'{self.market_type} Regional Market Share Distribution - {year}', 
                     fontsize=self.title_fontsize, pad=20)
            plt.xlabel('Market Share (%)', fontsize=self.axis_fontsize)
            plt.ylabel('Region', fontsize=self.axis_fontsize)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                value = region_data.iloc[i]['Value']
                share = region_data.iloc[i]['Share']
                
                # Format value string
                if value >= 1e9:
                    value_str = f'${value/1e9:.1f}B'
                else:
                    value_str = f'${value/1e6:.1f}M'
                
                # Add percentage and value labels
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f' {share:.1f}% ({value_str})',
                        va='center', fontsize=self.axis_fontsize - 2)
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save the figure
            filename = os.path.join(self.save_path, 
                                  f'{self.market_type}_Regional_Analysis_{year}.png')
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created regional analysis chart for {year}: {filename}")
            generated_files.append(filename)
        
        # If analyzing multiple years, create a trend chart
        if len(years_to_analyze) > 1:
            self._create_regional_trend_chart(market_data, years_to_analyze, title)
        
        return generated_files[0] if len(generated_files) == 1 else generated_files
    
    def _create_regional_trend_chart(self, market_data: pd.DataFrame, years: list, title: str = None):
        """
        Create a trend chart showing regional shares over multiple years
        """
        trend_data = []
        for year in years:
            year_data = market_data[market_data['Year'] == year].copy()
            total_value = year_data['Value'].sum()
            region_shares = year_data.groupby('Region')['Value'].sum() / total_value * 100
            
            for region, share in region_shares.items():
                trend_data.append({
                    'Year': year,
                    'Region': region,
                    'Share': share,
                    'Value': year_data[year_data['Region'] == region]['Value'].sum()
                })
        
        trend_df = pd.DataFrame(trend_data)
        
        plt.figure(figsize=(12, 6), dpi=self.dpi)
        
        # Create line plot for each region
        for region in trend_df['Region'].unique():
            region_data = trend_df[trend_df['Region'] == region]
            plt.plot(region_data['Year'], region_data['Share'], 
                    marker='o', label=region, linewidth=2)
        
        plt.title(f'{self.market_type} Regional Share Evolution ({min(years)}-{max(years)})',
                 fontsize=self.title_fontsize)
        plt.xlabel('Year', fontsize=self.axis_fontsize)
        plt.ylabel('Market Share (%)', fontsize=self.axis_fontsize)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=self.legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Add value labels for the final year
        final_year = max(years)
        final_data = trend_df[trend_df['Year'] == final_year]
        
        for _, row in final_data.iterrows():
            value = row['Value']
            if value >= 1e9:
                value_str = f'${value/1e9:.1f}B'
            else:
                value_str = f'${value/1e6:.1f}M'
            
            plt.text(row['Year'], row['Share'], 
                    f' {row["Share"]:.1f}%\n ({value_str})',
                    ha='left', va='bottom', fontsize=self.axis_fontsize - 4)
        
        plt.tight_layout()
        
        # Save the trend chart
        filename = os.path.join(self.save_path, 
                              f'{self.market_type}_Regional_Share_Evolution_{min(years)}-{max(years)}.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created regional share evolution chart: {filename}")
    
    def create_top_countries_chart(self, market_data: pd.DataFrame, title: str = None,
                                   top_n: int = 10, specific_countries: List[str] = None,
                                   year: int = None) -> str:
        """
        Create a horizontal bar chart showing market values for top N countries or specific countries
        
        Args:
            market_data: DataFrame with distributed market values
            title: Chart title
            top_n: Number of top countries to show (if specific_countries not provided)
            specific_countries: Optional list of specific countries to include
            year: Optional specific year to analyze (defaults to latest year)
            
        Returns:
            Path to the saved visualization file
        """
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
        
        # Use the specified year or latest year
        available_years = sorted(market_data['Year'].unique())
        target_year = year if year in available_years else max(available_years)
        
        # Filter data for target year
        year_data = market_data[market_data['Year'] == target_year].copy()
        total_value = year_data['Value'].sum()
        
        # Ensure consistent country naming
        country_name_mapping = {
            'China': 'Mainland China',
            'People\'s Republic of China': 'Mainland China',
            'PRC': 'Mainland China'
        }
        year_data[country_col] = year_data[country_col].replace(country_name_mapping)
        
        # Filter countries based on input
        if specific_countries:
            # Replace any variations of China in specific_countries
            specific_countries = [country_name_mapping.get(c, c) for c in specific_countries]
            filtered_data = year_data[year_data[country_col].isin(specific_countries)]
            if filtered_data.empty:
                logger.warning("None of the specified countries found in data")
                return ""
        else:
            filtered_data = year_data.nlargest(top_n, 'Value')
        
        # Sort by value descending for better visualization
        filtered_data = filtered_data.sort_values('Value', ascending=True)
        filtered_data['Share'] = filtered_data['Value'] / total_value * 100
        
        # Create the plot with larger figure size for better readability
        plt.figure(figsize=(14, 8), dpi=self.dpi)
        
        # Create horizontal bar chart with custom colors
        colors = plt.cm.Spectral(np.linspace(0, 1, len(filtered_data)))
        bars = plt.barh(filtered_data[country_col], filtered_data['Value'] / 1e9, color=colors)
        
        # Format chart with year in main title
        chart_title = f'{self.market_type} Top Countries Analysis - {target_year}'
        plt.title(chart_title, fontsize=self.title_fontsize + 2, pad=20)
        
        plt.xlabel('Market Size (Billion USD)', fontsize=self.axis_fontsize)
        plt.ylabel('Country', fontsize=self.axis_fontsize)
        
        # Add value and share labels on bars with improved formatting
        for i, bar in enumerate(bars):
            value = filtered_data.iloc[i]['Value']
            share = filtered_data.iloc[i]['Share']
            
            # Format value string with billions and millions
            if value >= 1e9:
                value_str = f'${value/1e9:.2f}B'
            else:
                value_str = f'${value/1e6:.1f}M'
            
            # Position the text at the end of each bar
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f'  {value_str} ({share:.1f}%)',
                    va='center', fontsize=self.axis_fontsize,
                    fontweight='bold')
        
        # Enhance grid and layout
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add a light background color for better contrast
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        plt.gcf().patch.set_facecolor('white')
        
        # Format x-axis to show billions
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:.0f}B'))
        
        # Add total market size in the title
        total_str = f'${total_value/1e9:.2f}B'
        plt.suptitle(f'Total Market Size: {total_str}', y=0.95, fontsize=self.title_fontsize - 2)
        
        plt.tight_layout()
        
        # Save the figure with year in filename
        filename_prefix = 'Selected_Countries' if specific_countries else f'Top_{len(filtered_data)}_Countries'
        filename = os.path.join(self.save_path, 
                              f'{self.market_type}_{filename_prefix}_Market_Size_{target_year}.png')
        plt.savefig(filename, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Created top countries market size chart: {filename}")
        return filename
    
    def generate_excel_report(self, market_data: pd.DataFrame) -> str:
        """
        Generate an Excel report with market data in wide format (years as columns)
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Path to the generated Excel file
        """
        # Create a writer for Excel file
        filename = os.path.join(self.save_path, f'{self.market_type}_Market_Forecast.xlsx')
        writer = pd.ExcelWriter(filename, engine='openpyxl')
        
        # Get column names from mapping
        country_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        name_col = country_mapping.get('name_column', 'Country')
        
        # Create a summary sheet with global market data
        global_data = self._extract_global_data(market_data)
        global_data.to_excel(writer, sheet_name='Global Market', index=False)
        
        # Determine which value column to use
        value_column = None
        for col in ['market_value', 'Value', 'value']:
            if col in market_data.columns:
                value_column = col
                break
        
        if not value_column:
            logger.warning("No value column found in market data")
            writer.close()
            return filename
        
        # Create a wide format version of the data (years as columns)
        if 'Year' in market_data.columns:
            # Create pivot table with years as columns
            wide_data = market_data.pivot_table(
                index=[id_col, name_col], 
                columns='Year',
                values=value_column,
                aggfunc='sum'
            ).reset_index()
            
            # Save to Excel
            wide_data.to_excel(writer, sheet_name='Market Data (Wide)', index=False)
            
            # Also save market share in wide format if available
            if 'market_share' in market_data.columns:
                share_data = market_data.pivot_table(
                    index=[id_col, name_col], 
                    columns='Year',
                    values='market_share',
                    aggfunc='sum'
                ).reset_index()
                
                share_data.to_excel(writer, sheet_name='Market Share (Wide)', index=False)
        
        # Also include the original data in a separate sheet
        market_data.to_excel(writer, sheet_name='Market Data (Long)', index=False)
        
        # Create a sheet with top countries for the latest year
        latest_year = market_data['Year'].max() if 'Year' in market_data.columns else None
        if latest_year:
            latest_data = market_data[market_data['Year'] == latest_year].copy()
            
            # Determine which column to use for sorting
            value_column = None
            for col in ['market_value', 'Value', 'value']:
                if col in latest_data.columns:
                    value_column = col
                    break
            
            if value_column:
                latest_data = latest_data.sort_values(value_column, ascending=False)
                latest_data.to_excel(writer, sheet_name=f'Top Countries {latest_year}', index=False)
        
        # Save the Excel file
        writer.close()
        
        logger.info(f"Generated Excel report: {filename}")
        return filename
    
    def _extract_global_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract global market data from the distributed market data
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            DataFrame with global market data by year
        """
        # Create a copy to avoid modifying the original
        df = market_data.copy()
        
        # Check if we have a Year column
        if 'Year' not in df.columns:
            return pd.DataFrame()
        
        # Determine which value column to use
        value_column = None
        for col in ['market_value', 'Value', 'value']:
            if col in df.columns:
                value_column = col
                break
        
        if not value_column:
            return pd.DataFrame()
        
        # Determine which original value column to use
        original_column = None
        for col in ['original_value', 'original_market_value', 'original_share']:
            if col in df.columns:
                original_column = col
                break
        
        # Group by year and sum values
        agg_dict = {value_column: 'sum'}
        if original_column:
            agg_dict[original_column] = 'sum'
        
        global_data = df.groupby('Year').agg(agg_dict).reset_index()
        
        # Rename columns for clarity
        rename_dict = {value_column: 'Total Market Value'}
        if original_column:
            rename_dict[original_column] = 'Original Market Value'
        
        global_data = global_data.rename(columns=rename_dict)
        
        return global_data
    
    def _replace_placeholders(self, text: str) -> str:
        """
        Replace placeholders in text with actual values
        
        Args:
            text: Text with placeholders
            
        Returns:
            Text with placeholders replaced
        """
        if not text:
            return text
        
        # Replace market_type placeholder
        text = text.replace('${market_type}', self.market_type)
        
        # Replace other project info placeholders
        for key, value in self.project_info.items():
            text = text.replace(f'${{{key}}}', str(value))
        
        return text 