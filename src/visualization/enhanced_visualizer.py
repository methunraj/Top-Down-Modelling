"""
Enhanced Market Visualizer Module - Modern Interactive Visualizations

This module provides advanced, interactive visualizations for market forecasts
implementing the improvements outlined in the visualization enhancement plan.
"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import json

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedMarketVisualizer:
    """
    Enhanced market visualizer with modern interactive charts
    
    This class provides advanced visualization functionality including:
    - Animated bar chart races
    - Interactive world maps
    - Dynamic comparison tools
    - Professional styling themes
    """
    
    def __init__(self, config_manager, data_loader):
        """
        Initialize the Enhanced Market Visualizer
        
        Args:
            config_manager: Configuration manager instance
            data_loader: Data loader instance
        """
        self.config_manager = config_manager
        self.data_loader = data_loader
        
        # Get project info
        self.project_info = self.config_manager.get_project_info()
        self.market_type = self.project_info.get('market_type', 'Market')
        
        # Set default output directory
        self.output_dir = self.config_manager.get_output_directory()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Professional color schemes
        self.color_schemes = {
            'corporate': px.colors.qualitative.Plotly,
            'professional': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'viridis': px.colors.sequential.Viridis,
            'blues': px.colors.sequential.Blues,
            'diverging': px.colors.diverging.RdBu,
            'colorblind_safe': ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', 
                               '#e6ab02', '#a6761d', '#666666']
        }
        
        # Set default theme
        self.current_theme = 'professional'
        
        # Track created visualization files
        self.visualization_files = []
    
    def create_animated_bar_race(self, market_data: pd.DataFrame, 
                                top_n: int = 10, 
                                title: str = None,
                                animation_speed: int = 800) -> go.Figure:
        """
        Create an animated bar chart race showing country rankings over time
        
        Args:
            market_data: DataFrame with market forecast data
            top_n: Number of top countries to show
            title: Chart title
            animation_speed: Animation speed in milliseconds
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating animated bar chart race")
        
        if title is None:
            title = f"{self.market_type} Market Size - Country Rankings Over Time"
        
        # Prepare data for animation
        years = sorted(market_data['Year'].unique())
        
        # Get top countries based on latest year
        latest_year = max(years)
        latest_data = market_data[market_data['Year'] == latest_year]
        top_countries = latest_data.nlargest(top_n, 'Value')['Country'].tolist()
        
        # Filter data to include only top countries
        filtered_data = market_data[market_data['Country'].isin(top_countries)].copy()
        
        # Create frames for animation
        frames = []
        for year in years:
            year_data = filtered_data[filtered_data['Year'] == year].copy()
            year_data = year_data.sort_values('Value', ascending=True)
            
            # Ensure we have data for all top countries (fill with 0 if missing)
            missing_countries = set(top_countries) - set(year_data['Country'].tolist())
            for country in missing_countries:
                missing_row = pd.DataFrame([{
                    'Country': country,
                    'Value': 0,
                    'Year': year
                }])
                year_data = pd.concat([year_data, missing_row], ignore_index=True)
            
            year_data = year_data.sort_values('Value', ascending=True)
            
            frame = go.Frame(
                data=[go.Bar(
                    x=year_data['Value'],
                    y=year_data['Country'],
                    orientation='h',
                    marker=dict(
                        color=year_data['Value'],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f"{val/1e6:.1f}M" if val < 1e9 else f"{val/1e9:.1f}B" for val in year_data['Value']],
                    textposition='outside',
                    textfont=dict(size=12)
                )],
                name=str(year)
            )
            frames.append(frame)
        
        # Create initial figure with first year data
        initial_data = filtered_data[filtered_data['Year'] == years[0]].copy()
        initial_data = initial_data.sort_values('Value', ascending=True)
        
        fig = go.Figure(
            data=[go.Bar(
                x=initial_data['Value'],
                y=initial_data['Country'],
                orientation='h',
                marker=dict(
                    color=initial_data['Value'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"{val/1e6:.1f}M" if val < 1e9 else f"{val/1e9:.1f}B" for val in initial_data['Value']],
                textposition='outside',
                textfont=dict(size=12)
            )],
            frames=frames
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, family="Arial Black"),
                x=0.5
            ),
            xaxis=dict(
                title="Market Value",
                titlefont=dict(size=14),
                tickformat=".0s"
            ),
            yaxis=dict(
                title="Country",
                titlefont=dict(size=14)
            ),
            height=600,
            width=1000,
            template="plotly_white",
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0.1,
                "y": 1.15,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": animation_speed, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 200}
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Year: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 200},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [{
                    "args": [[str(year)], {
                        "frame": {"duration": animation_speed, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 200}
                    }],
                    "label": str(year),
                    "method": "animate"
                } for year in years]
            }]
        )
        
        return fig
    
    def create_market_share_evolution(self, market_data: pd.DataFrame,
                                    top_n: int = 10,
                                    title: str = None) -> go.Figure:
        """
        Create streamlined area chart showing market share evolution (NOT stacked)
        
        Args:
            market_data: DataFrame with market forecast data
            top_n: Number of top countries to show
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating market share evolution chart")
        
        if title is None:
            title = f"{self.market_type} Market Share Evolution"
        
        # Get top countries based on latest year
        latest_year = market_data['Year'].max()
        latest_data = market_data[market_data['Year'] == latest_year]
        top_countries = latest_data.nlargest(top_n, 'Value')['Country'].tolist()
        
        # Filter data and calculate market share
        years = sorted(market_data['Year'].unique())
        share_data = []
        
        for year in years:
            year_data = market_data[market_data['Year'] == year]
            total_market = year_data['Value'].sum()
            
            for country in top_countries:
                country_data = year_data[year_data['Country'] == country]
                if not country_data.empty:
                    value = country_data['Value'].iloc[0]
                    share = (value / total_market) * 100
                else:
                    share = 0
                
                share_data.append({
                    'Year': year,
                    'Country': country,
                    'Market_Share': share
                })
        
        share_df = pd.DataFrame(share_data)
        
        # Create figure with multiple line traces
        fig = go.Figure()
        
        colors = self.color_schemes[self.current_theme]
        
        for i, country in enumerate(top_countries):
            country_data = share_df[share_df['Country'] == country]
            
            # Add line trace
            fig.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['Market_Share'],
                mode='lines+markers',
                name=country,
                line=dict(
                    color=colors[i % len(colors)],
                    width=3
                ),
                marker=dict(size=6),
                fill='tonexty' if i > 0 else 'tozeroy',
                fillcolor=colors[i % len(colors)].replace('rgb', 'rgba').replace(')', ', 0.1)'),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Year: %{x}<br>' +
                             'Market Share: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, family="Arial"),
                x=0.5
            ),
            xaxis=dict(
                title="Year",
                titlefont=dict(size=14),
                tickmode='linear',
                dtick=1
            ),
            yaxis=dict(
                title="Market Share (%)",
                titlefont=dict(size=14),
                ticksuffix="%"
            ),
            height=500,
            width=900,
            template="plotly_white",
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig
    
    def create_interactive_world_map(self, market_data: pd.DataFrame,
                                   year: int = None,
                                   title: str = None) -> go.Figure:
        """
        Create interactive choropleth world map
        
        Args:
            market_data: DataFrame with market forecast data
            year: Year to display (defaults to latest)
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating interactive world map")
        
        if year is None:
            year = market_data['Year'].max()
        
        if title is None:
            title = f"{self.market_type} Market Size by Country ({year})"
        
        # Filter data for the specified year
        year_data = market_data[market_data['Year'] == year].copy()
        
        # Create choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=year_data['Country'],
            z=year_data['Value'],
            locationmode='country names',
            colorscale='Viridis',
            autocolorscale=False,
            text=year_data['Country'],
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title="Market Value",
            hovertemplate='<b>%{text}</b><br>' +
                         'Market Value: %{z:,.0f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, family="Arial"),
                x=0.5
            ),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=600,
            width=1000
        )
        
        return fig
    
    def create_bubble_evolution(self, market_data: pd.DataFrame,
                              top_n: int = 15,
                              title: str = None) -> go.Figure:
        """
        Create animated bubble chart showing market size vs growth rate
        
        Args:
            market_data: DataFrame with market forecast data
            top_n: Number of top countries to show
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating bubble evolution chart")
        
        if title is None:
            title = f"{self.market_type} Market Size vs Growth Rate Evolution"
        
        # Calculate growth rates
        growth_data = []
        countries = market_data['Country'].unique()
        years = sorted(market_data['Year'].unique())
        
        for country in countries:
            country_data = market_data[market_data['Country'] == country].sort_values('Year')
            
            for i in range(1, len(country_data)):
                current_year = country_data.iloc[i]
                prev_year = country_data.iloc[i-1]
                
                growth_rate = ((current_year['Value'] / prev_year['Value']) - 1) * 100
                
                growth_data.append({
                    'Country': country,
                    'Year': current_year['Year'],
                    'Market_Size': current_year['Value'],
                    'Growth_Rate': growth_rate
                })
        
        growth_df = pd.DataFrame(growth_data)
        
        # Get top countries by latest year market size
        latest_year = max(years)
        latest_data = market_data[market_data['Year'] == latest_year]
        top_countries = latest_data.nlargest(top_n, 'Value')['Country'].tolist()
        
        # Filter to top countries
        growth_df = growth_df[growth_df['Country'].isin(top_countries)]
        
        # Create animated scatter plot
        fig = px.scatter(
            growth_df,
            x='Market_Size',
            y='Growth_Rate',
            size='Market_Size',
            color='Country',
            hover_name='Country',
            animation_frame='Year',
            size_max=60,
            range_x=[0, growth_df['Market_Size'].max() * 1.1],
            range_y=[growth_df['Growth_Rate'].min() - 5, growth_df['Growth_Rate'].max() + 5],
            title=title,
            labels={
                'Market_Size': 'Market Size',
                'Growth_Rate': 'Growth Rate (%)'
            }
        )
        
        # Add quadrant lines
        max_size = growth_df['Market_Size'].max()
        median_size = growth_df['Market_Size'].median()
        
        # Vertical line at median market size
        fig.add_vline(x=median_size, line_dash="dash", line_color="gray", opacity=0.5)
        # Horizontal line at 0% growth
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(
            x=median_size * 1.5, y=growth_df['Growth_Rate'].max() * 0.8,
            text="Stars<br>(High Growth, Large Market)",
            showarrow=False, font=dict(color="green", size=10)
        )
        
        fig.add_annotation(
            x=median_size * 0.5, y=growth_df['Growth_Rate'].max() * 0.8,
            text="Question Marks<br>(High Growth, Small Market)",
            showarrow=False, font=dict(color="orange", size=10)
        )
        
        fig.add_annotation(
            x=median_size * 1.5, y=growth_df['Growth_Rate'].min() * 0.8,
            text="Cash Cows<br>(Low Growth, Large Market)",
            showarrow=False, font=dict(color="blue", size=10)
        )
        
        fig.add_annotation(
            x=median_size * 0.5, y=growth_df['Growth_Rate'].min() * 0.8,
            text="Dogs<br>(Low Growth, Small Market)",
            showarrow=False, font=dict(color="red", size=10)
        )
        
        fig.update_layout(
            height=600,
            width=1000,
            template="plotly_white"
        )
        
        return fig
    
    def create_waterfall_chart(self, market_data: pd.DataFrame,
                             year_start: int, year_end: int,
                             top_n: int = 10,
                             title: str = None) -> go.Figure:
        """
        Create waterfall chart showing market changes
        
        Args:
            market_data: DataFrame with market forecast data
            year_start: Starting year
            year_end: Ending year
            top_n: Number of top contributors to show
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating waterfall chart")
        
        if title is None:
            title = f"{self.market_type} Market Change ({year_start}-{year_end})"
        
        # Get data for start and end years
        start_data = market_data[market_data['Year'] == year_start]
        end_data = market_data[market_data['Year'] == year_end]
        
        # Calculate changes
        changes = []
        
        start_total = start_data['Value'].sum()
        end_total = end_data['Value'].sum()
        
        # Get country-wise changes
        for country in start_data['Country']:
            start_value = start_data[start_data['Country'] == country]['Value'].iloc[0]
            end_country_data = end_data[end_data['Country'] == country]
            
            if not end_country_data.empty:
                end_value = end_country_data['Value'].iloc[0]
                change = end_value - start_value
            else:
                change = -start_value  # Country dropped out
            
            changes.append({
                'Country': country,
                'Change': change,
                'Abs_Change': abs(change)
            })
        
        # Check for new countries in end year
        new_countries = set(end_data['Country']) - set(start_data['Country'])
        for country in new_countries:
            end_value = end_data[end_data['Country'] == country]['Value'].iloc[0]
            changes.append({
                'Country': country,
                'Change': end_value,
                'Abs_Change': end_value
            })
        
        changes_df = pd.DataFrame(changes)
        
        # Get top contributors by absolute change
        top_changes = changes_df.nlargest(top_n, 'Abs_Change')
        other_change = changes_df[~changes_df['Country'].isin(top_changes['Country'])]['Change'].sum()
        
        # Prepare waterfall data
        waterfall_data = [
            ('Starting Value', start_total, 'absolute'),
        ]
        
        # Add top country contributions
        for _, row in top_changes.iterrows():
            waterfall_data.append((row['Country'], row['Change'], 'relative'))
        
        # Add others if significant
        if abs(other_change) > start_total * 0.01:  # If > 1% of starting value
            waterfall_data.append(('Others', other_change, 'relative'))
        
        waterfall_data.append(('Ending Value', end_total, 'total'))
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Market Change",
            orientation="v",
            measure=[item[2] for item in waterfall_data],
            x=[item[0] for item in waterfall_data],
            y=[item[1] for item in waterfall_data],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, family="Arial"),
                x=0.5
            ),
            xaxis=dict(title="Contributors"),
            yaxis=dict(title="Market Value", tickformat=".0s"),
            height=600,
            width=1000,
            template="plotly_white"
        )
        
        return fig
    
    def create_ranking_table_with_sparklines(self, market_data: pd.DataFrame,
                                           reference_year: int = None) -> go.Figure:
        """
        Create interactive ranking table with mini sparkline charts
        
        Args:
            market_data: DataFrame with market forecast data
            reference_year: Reference year for ranking (defaults to latest)
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating ranking table with sparklines")
        
        if reference_year is None:
            reference_year = market_data['Year'].max()
        
        # Get ranking data for reference year
        ref_data = market_data[market_data['Year'] == reference_year].sort_values('Value', ascending=False)
        
        # Calculate sparkline data and trends
        table_data = []
        years = sorted(market_data['Year'].unique())
        
        for rank, (_, row) in enumerate(ref_data.iterrows(), 1):
            country = row['Country']
            current_value = row['Value']
            
            # Get historical data for this country
            country_data = market_data[market_data['Country'] == country].sort_values('Year')
            
            # Calculate trend
            if len(country_data) > 1:
                first_value = country_data.iloc[0]['Value']
                cagr = ((current_value / first_value) ** (1 / (len(years) - 1)) - 1) * 100
                
                # Create mini sparkline (as text representation)
                values = country_data['Value'].tolist()
                max_val = max(values)
                min_val = min(values)
                
                # Normalize values for sparkline representation
                if max_val > min_val:
                    normalized = [(v - min_val) / (max_val - min_val) for v in values]
                    sparkline = ''.join(['▁' if v < 0.2 else '▂' if v < 0.4 else '▄' if v < 0.6 else '▆' if v < 0.8 else '█' for v in normalized])
                else:
                    sparkline = '▄' * len(values)
            else:
                cagr = 0
                sparkline = '▄'
            
            # Calculate rank change (if we have previous year data)
            if reference_year > min(years):
                prev_year_data = market_data[market_data['Year'] == reference_year - 1]
                if not prev_year_data.empty:
                    prev_rank_data = prev_year_data.sort_values('Value', ascending=False)
                    prev_rank_data = prev_rank_data.reset_index(drop=True)
                    
                    if country in prev_rank_data['Country'].values:
                        prev_rank = prev_rank_data[prev_rank_data['Country'] == country].index[0] + 1
                        rank_change = prev_rank - rank
                        
                        if rank_change > 0:
                            rank_change_str = f"↑{rank_change}"
                            rank_change_color = "green"
                        elif rank_change < 0:
                            rank_change_str = f"↓{abs(rank_change)}"
                            rank_change_color = "red"
                        else:
                            rank_change_str = "→"
                            rank_change_color = "gray"
                    else:
                        rank_change_str = "NEW"
                        rank_change_color = "blue"
                else:
                    rank_change_str = "-"
                    rank_change_color = "gray"
            else:
                rank_change_str = "-"
                rank_change_color = "gray"
            
            table_data.append({
                'Rank': rank,
                'Country': country,
                'Value': f"{current_value/1e6:.1f}M" if current_value < 1e9 else f"{current_value/1e9:.1f}B",
                'CAGR': f"{cagr:.1f}%",
                'Trend': sparkline,
                'Change': rank_change_str
            })
        
        # Create table figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Rank', 'Country', f'Value ({reference_year})', 'CAGR', 'Trend', 'Rank Change'],
                fill_color='lightblue',
                align='center',
                font=dict(size=12, color='white'),
                height=40
            ),
            cells=dict(
                values=[
                    [d['Rank'] for d in table_data],
                    [d['Country'] for d in table_data],
                    [d['Value'] for d in table_data],
                    [d['CAGR'] for d in table_data],
                    [d['Trend'] for d in table_data],
                    [d['Change'] for d in table_data]
                ],
                fill_color='white',
                align='center',
                font=dict(size=11),
                height=30
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=f"{self.market_type} Country Rankings with Trends ({reference_year})",
                font=dict(size=16, family="Arial"),
                x=0.5
            ),
            height=600,
            width=1000
        )
        
        return fig
    
    def create_executive_dashboard(self, market_data: pd.DataFrame) -> go.Figure:
        """
        Create executive summary dashboard with key metrics
        
        Args:
            market_data: DataFrame with market forecast data
            
        Returns:
            Plotly figure object with subplots
        """
        logger.info("Creating executive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Market Size Trend', 'Top 5 Countries', 'Growth Rate Distribution',
                'Regional Breakdown', 'CAGR Analysis', 'Market Concentration'
            ],
            specs=[
                [{"secondary_y": False}, {"type": "bar"}, {"type": "histogram"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        years = sorted(market_data['Year'].unique())
        latest_year = max(years)
        
        # 1. Market Size Trend
        total_by_year = market_data.groupby('Year')['Value'].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=total_by_year['Year'],
                y=total_by_year['Value'],
                mode='lines+markers',
                name='Total Market',
                line=dict(color='blue', width=3)
            ),
            row=1, col=1
        )
        
        # 2. Top 5 Countries
        top_5 = market_data[market_data['Year'] == latest_year].nlargest(5, 'Value')
        fig.add_trace(
            go.Bar(
                x=top_5['Country'],
                y=top_5['Value'],
                name='Top 5 Countries',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # 3. Growth Rate Distribution
        growth_rates = []
        for country in market_data['Country'].unique():
            country_data = market_data[market_data['Country'] == country].sort_values('Year')
            if len(country_data) > 1:
                first_val = country_data.iloc[0]['Value']
                last_val = country_data.iloc[-1]['Value']
                years_diff = len(years) - 1
                if years_diff > 0 and first_val > 0:
                    cagr = ((last_val / first_val) ** (1 / years_diff) - 1) * 100
                    growth_rates.append(cagr)
        
        fig.add_trace(
            go.Histogram(
                x=growth_rates,
                nbinsx=20,
                name='CAGR Distribution',
                marker_color='lightgreen'
            ),
            row=1, col=3
        )
        
        # 4. Regional Breakdown (simplified)
        # Group countries into regions (basic example)
        regions = {
            'North America': ['United States', 'Canada', 'Mexico'],
            'Europe': ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain'],
            'Asia Pacific': ['China', 'Japan', 'India', 'South Korea', 'Australia'],
            'Others': []
        }
        
        # Assign countries to regions
        latest_data = market_data[market_data['Year'] == latest_year]
        regional_data = {}
        
        for region, countries in regions.items():
            if region == 'Others':
                continue
            region_value = latest_data[latest_data['Country'].isin(countries)]['Value'].sum()
            if region_value > 0:
                regional_data[region] = region_value
        
        # Add remaining countries to Others
        assigned_countries = [c for region_countries in regions.values() for c in region_countries]
        others_value = latest_data[~latest_data['Country'].isin(assigned_countries)]['Value'].sum()
        if others_value > 0:
            regional_data['Others'] = others_value
        
        if regional_data:
            fig.add_trace(
                go.Pie(
                    labels=list(regional_data.keys()),
                    values=list(regional_data.values()),
                    name='Regional Share'
                ),
                row=2, col=1
            )
        
        # 5. CAGR Analysis (Top 10 countries)
        cagr_data = []
        for country in market_data['Country'].unique():
            country_data = market_data[market_data['Country'] == country].sort_values('Year')
            if len(country_data) > 1:
                first_val = country_data.iloc[0]['Value']
                last_val = country_data.iloc[-1]['Value']
                years_diff = len(years) - 1
                if years_diff > 0 and first_val > 0:
                    cagr = ((last_val / first_val) ** (1 / years_diff) - 1) * 100
                    cagr_data.append({'Country': country, 'CAGR': cagr, 'Latest_Value': last_val})
        
        cagr_df = pd.DataFrame(cagr_data)
        if not cagr_df.empty:
            top_cagr = cagr_df.nlargest(10, 'Latest_Value').sort_values('CAGR', ascending=True)
            
            fig.add_trace(
                go.Bar(
                    x=top_cagr['CAGR'],
                    y=top_cagr['Country'],
                    orientation='h',
                    name='CAGR by Country',
                    marker_color='orange'
                ),
                row=2, col=2
            )
        
        # 6. Market Concentration (Herfindahl Index over time)
        concentration_data = []
        for year in years:
            year_data = market_data[market_data['Year'] == year]
            total_market = year_data['Value'].sum()
            if total_market > 0:
                market_shares = (year_data['Value'] / total_market) ** 2
                hhi = market_shares.sum() * 10000  # Multiply by 10000 for standard HHI
                concentration_data.append({'Year': year, 'HHI': hhi})
        
        if concentration_data:
            conc_df = pd.DataFrame(concentration_data)
            fig.add_trace(
                go.Scatter(
                    x=conc_df['Year'],
                    y=conc_df['HHI'],
                    mode='lines+markers',
                    name='Market Concentration',
                    line=dict(color='red', width=2)
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1400,
            title_text=f"{self.market_type} Executive Dashboard",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def apply_professional_theme(self, fig: go.Figure, theme: str = 'professional') -> go.Figure:
        """
        Apply professional styling theme to figure
        
        Args:
            fig: Plotly figure object
            theme: Theme name ('professional', 'corporate', 'dark', etc.)
            
        Returns:
            Styled figure object
        """
        if theme == 'professional':
            fig.update_layout(
                template="plotly_white",
                font=dict(family="Arial, sans-serif", size=12, color="#2E2E2E"),
                title_font=dict(size=18, color="#1F1F1F"),
                plot_bgcolor="white",
                paper_bgcolor="white"
            )
        elif theme == 'corporate':
            fig.update_layout(
                template="plotly_white",
                font=dict(family="Helvetica, sans-serif", size=11, color="#333333"),
                title_font=dict(size=16, color="#003366"),
                plot_bgcolor="#FAFAFA",
                paper_bgcolor="white"
            )
        elif theme == 'dark':
            fig.update_layout(
                template="plotly_dark",
                font=dict(family="Arial, sans-serif", size=12, color="#E1E1E1"),
                title_font=dict(size=18, color="#FFFFFF"),
                plot_bgcolor="#2F2F2F",
                paper_bgcolor="#1E1E1E"
            )
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str, format: str = 'html') -> str:
        """
        Save figure to file
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            format: Output format ('html', 'png', 'pdf', 'svg')
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, f"{filename}.{format}")
        
        if format == 'html':
            fig.write_html(output_path)
        elif format == 'png':
            fig.write_image(output_path, format='png', width=1200, height=800, scale=2)
        elif format == 'pdf':
            fig.write_image(output_path, format='pdf', width=1200, height=800)
        elif format == 'svg':
            fig.write_image(output_path, format='svg', width=1200, height=800)
        
        self.visualization_files.append(output_path)
        logger.info(f"Saved visualization to {output_path}")
        return output_path