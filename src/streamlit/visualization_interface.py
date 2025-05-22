"""
Streamlit Visualization Interface Module

This module provides components and utilities for visualizing market data
and forecasts through the Streamlit interface.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import base64
from datetime import datetime

from src.config.config_manager import ConfigurationManager
from src.visualization.market_visualizer import MarketVisualizer

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def render_market_size_visualization(distributed_market: pd.DataFrame, config_manager: ConfigurationManager) -> None:
    """
    Render market size visualization.
    
    Args:
        distributed_market: DataFrame with distributed market data
        config_manager: ConfigurationManager instance
    """
    st.header("Market Size Visualization")
    
    # Get visualization options
    st.subheader("Display Options")
    col1, col2 = st.columns(2)
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
    except Exception:
        id_col = 'idGeo'
        country_col = 'Country'
    
    with col1:
        # Countries to include
        top_n = st.slider("Top N Countries", min_value=5, max_value=20, value=10, key="market_size_top_n")
        view_type = st.radio("View Type", options=["Stacked", "Individual", "Total"], horizontal=True, key="market_size_view_type")
    
    with col2:
        # Years to include
        all_years = sorted(distributed_market['Year'].unique())
        selected_years = st.multiselect(
            "Select Years", 
            options=all_years, 
            default=[min(all_years), max(all_years)],
            key="market_size_years"
        )
        
        if not selected_years:
            selected_years = all_years
        
        # Scale type
        scale = st.radio("Scale", options=["Linear", "Logarithmic"], horizontal=True, key="market_size_scale")
    
    # Filter data
    filtered_data = distributed_market.copy()
    
    # Filter by selected years
    if selected_years and len(selected_years) > 0:
        filtered_data = filtered_data[filtered_data['Year'].isin(selected_years)]
    
    # For individual or stacked views, get top N countries
    if view_type in ["Stacked", "Individual"]:
        # Get the latest year
        latest_year = max(filtered_data['Year'])
        
        # Get the top N countries in the latest year
        top_countries_data = filtered_data[filtered_data['Year'] == latest_year]
        top_countries_data = top_countries_data.sort_values(by='Value', ascending=False)
        top_countries = top_countries_data.head(top_n)[country_col].tolist()
        
        # Filter to only include top countries
        filtered_data = filtered_data[filtered_data[country_col].isin(top_countries)]
    
    # Sort data
    filtered_data = filtered_data.sort_values(by=['Year', 'Value'], ascending=[True, False])
    
    # Create visualization based on view type
    if view_type == "Total":
        # Total market size by year
        total_by_year = filtered_data.groupby('Year')['Value'].sum().reset_index()
        
        # Create a bar chart
        fig = px.bar(
            total_by_year,
            x='Year',
            y='Value',
            title='Total Market Size by Year',
            labels={'Value': 'Market Size', 'Year': 'Year'}
        )
        
        # Update y-axis scale if logarithmic
        if scale == "Logarithmic":
            fig.update_yaxes(type='log')
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    elif view_type == "Stacked":
        # Pivot data to create stacked bar chart
        pivot_data = filtered_data.pivot_table(
            index='Year',
            columns=country_col,
            values='Value',
            aggfunc='sum'
        ).fillna(0)
        
        # Reset index for Plotly
        pivot_data = pivot_data.reset_index()
        
        # Create a stacked bar chart
        fig = px.bar(
            pivot_data,
            x='Year',
            y=pivot_data.columns[1:],  # All columns except 'Year'
            title='Market Size by Country (Stacked)',
            labels={'value': 'Market Size', 'Year': 'Year', 'variable': 'Country'}
        )
        
        # Update y-axis scale if logarithmic
        if scale == "Logarithmic":
            fig.update_yaxes(type='log')
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Individual
        # Group by country and year, sum values
        grouped_data = filtered_data.groupby([country_col, 'Year'])['Value'].sum().reset_index()
        
        # Create a line chart
        fig = px.line(
            grouped_data,
            x='Year',
            y='Value',
            color=country_col,
            title='Market Size by Country (Individual)',
            labels={'Value': 'Market Size', 'Year': 'Year', country_col: 'Country'}
        )
        
        # Update y-axis scale if logarithmic
        if scale == "Logarithmic":
            fig.update_yaxes(type='log')
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Option to download the chart
    if st.button("Download Chart", key="download_market_size_chart"):
        # Create a high-resolution image
        fig.update_layout(
            width=1200,
            height=800,
        )
        
        # Save to a bytes buffer
        img_bytes = fig.to_image(format="png", scale=2)
        
        # Create a download link
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="market_size_chart.png">Download Chart as PNG</a>'
        st.markdown(href, unsafe_allow_html=True)


def render_growth_analysis(distributed_market: pd.DataFrame, config_manager: ConfigurationManager) -> None:
    """
    Render growth rate analysis visualization.
    
    Args:
        distributed_market: DataFrame with distributed market data
        config_manager: ConfigurationManager instance
    """
    st.header("Growth Rate Analysis")
    
    # Get visualization options
    st.subheader("Analysis Options")
    col1, col2 = st.columns(2)
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
    except Exception:
        id_col = 'idGeo'
        country_col = 'Country'
    
    with col1:
        # Growth metric
        growth_metric = st.radio(
            "Growth Metric", 
            options=["Year-over-Year", "CAGR"], 
            horizontal=True,
            key="growth_metric"
        )
        
        # Countries to include
        top_n = st.slider(
            "Top N Countries", 
            min_value=5, 
            max_value=20, 
            value=10, 
            key="growth_top_n"
        )
    
    with col2:
        # Period selection
        all_years = sorted(distributed_market['Year'].unique())
        
        if growth_metric == "CAGR":
            # Two-point CAGR
            start_year = st.selectbox(
                "Start Year",
                options=all_years[:-1],  # Exclude last year
                index=0,
                key="cagr_start_year"
            )
            
            # Only show years after start_year
            valid_end_years = [y for y in all_years if y > start_year]
            
            end_year = st.selectbox(
                "End Year",
                options=valid_end_years,
                index=len(valid_end_years) - 1,  # Default to last year
                key="cagr_end_year"
            )
        else:
            # YoY growth - select multiple years
            selected_years = st.multiselect(
                "Select Years",
                options=all_years[1:],  # Exclude first year since we need previous year for YoY
                default=all_years[-3:] if len(all_years) >= 3 else all_years[-1:],
                key="yoy_years"
            )
    
    # Calculate growth rates
    growth_data = distributed_market.copy()
    
    if growth_metric == "CAGR":
        # Calculate CAGR between start_year and end_year
        start_data = growth_data[growth_data['Year'] == start_year]
        end_data = growth_data[growth_data['Year'] == end_year]
        
        # Merge start and end data
        cagr_data = pd.merge(
            start_data,
            end_data,
            on=[id_col, country_col],
            suffixes=('_start', '_end')
        )
        
        # Calculate CAGR
        years_diff = end_year - start_year
        cagr_data['CAGR'] = ((cagr_data['Value_end'] / cagr_data['Value_start']) ** (1 / years_diff) - 1) * 100
        
        # Get top N countries by end year value
        top_countries = end_data.sort_values(by='Value', ascending=False).head(top_n)[country_col].tolist()
        
        # Filter to only include top countries
        cagr_data = cagr_data[cagr_data[country_col].isin(top_countries)]
        
        # Sort by CAGR
        cagr_data = cagr_data.sort_values(by='CAGR', ascending=False)
        
        # Create a bar chart
        fig = px.bar(
            cagr_data,
            x=country_col,
            y='CAGR',
            title=f'CAGR by Country ({start_year}-{end_year})',
            labels={'CAGR': 'CAGR (%)', country_col: 'Country'},
            color='CAGR',
            color_continuous_scale=px.colors.diverging.RdBu_r
        )
        
        # Add a reference line at 0%
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(cagr_data) - 0.5,
            y0=0,
            y1=0,
            line=dict(color='gray', width=1, dash='dash')
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Year-over-Year
        # If no years selected, use the last 3 years
        if not selected_years:
            selected_years = all_years[-3:] if len(all_years) >= 3 else all_years[-1:]
        
        # Calculate YoY growth for selected years
        yoy_data_list = []
        
        for year in selected_years:
            # Get data for current and previous year
            current_year_data = growth_data[growth_data['Year'] == year]
            prev_year_data = growth_data[growth_data['Year'] == year - 1]
            
            # Only proceed if we have data for both years
            if not current_year_data.empty and not prev_year_data.empty:
                # Merge current and previous year data
                year_data = pd.merge(
                    current_year_data,
                    prev_year_data,
                    on=[id_col, country_col],
                    suffixes=('_current', '_prev')
                )
                
                # Calculate YoY growth
                year_data['YoY_Growth'] = ((year_data['Value_current'] / year_data['Value_prev']) - 1) * 100
                
                # Add year column
                year_data['Growth_Year'] = year
                
                # Add to list
                yoy_data_list.append(year_data)
        
        # Combine all YoY data
        if yoy_data_list:
            yoy_data = pd.concat(yoy_data_list, ignore_index=True)
            
            # Get the latest year for determining top countries
            latest_year = max(yoy_data['Growth_Year'])
            latest_year_data = yoy_data[yoy_data['Growth_Year'] == latest_year]
            
            # Get top N countries by value in the latest year
            top_countries = latest_year_data.sort_values(by='Value_current', ascending=False).head(top_n)[country_col].tolist()
            
            # Filter to only include top countries
            yoy_data = yoy_data[yoy_data[country_col].isin(top_countries)]
            
            # Create visualization based on number of selected years
            if len(selected_years) == 1:
                # Single year - bar chart
                year_data = yoy_data[yoy_data['Growth_Year'] == selected_years[0]]
                year_data = year_data.sort_values(by='YoY_Growth', ascending=False)
                
                fig = px.bar(
                    year_data,
                    x=country_col,
                    y='YoY_Growth',
                    title=f'Year-over-Year Growth ({selected_years[0]-1} to {selected_years[0]})',
                    labels={'YoY_Growth': 'YoY Growth (%)', country_col: 'Country'},
                    color='YoY_Growth',
                    color_continuous_scale=px.colors.diverging.RdBu_r
                )
                
                # Add a reference line at 0%
                fig.add_shape(
                    type='line',
                    x0=-0.5,
                    x1=len(year_data) - 0.5,
                    y0=0,
                    y1=0,
                    line=dict(color='gray', width=1, dash='dash')
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # Multiple years - grouped bar chart or heatmap
                vis_type = st.radio(
                    "Visualization Type",
                    options=["Grouped Bar Chart", "Heatmap"],
                    horizontal=True,
                    key="yoy_vis_type"
                )
                
                if vis_type == "Grouped Bar Chart":
                    # Create a grouped bar chart
                    fig = px.bar(
                        yoy_data,
                        x=country_col,
                        y='YoY_Growth',
                        color='Growth_Year',
                        barmode='group',
                        title='Year-over-Year Growth by Country',
                        labels={'YoY_Growth': 'YoY Growth (%)', country_col: 'Country', 'Growth_Year': 'Year'}
                    )
                    
                    # Add a reference line at 0%
                    fig.add_shape(
                        type='line',
                        x0=-0.5,
                        x1=len(top_countries) - 0.5,
                        y0=0,
                        y1=0,
                        line=dict(color='gray', width=1, dash='dash')
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:  # Heatmap
                    # Pivot the data for heatmap
                    heatmap_data = yoy_data.pivot_table(
                        index=country_col,
                        columns='Growth_Year',
                        values='YoY_Growth'
                    )
                    
                    # Sort countries by average growth
                    heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]
                    
                    # Create a heatmap
                    fig = px.imshow(
                        heatmap_data,
                        text_auto='.1f',
                        aspect="auto",
                        title='Year-over-Year Growth Heatmap',
                        labels={'x': 'Year', 'y': 'Country', 'color': 'YoY Growth (%)'},
                        color_continuous_scale=px.colors.diverging.RdBu_r,
                        color_continuous_midpoint=0,
                        zmin=-20,  # Lower bound for color scale
                        zmax=20    # Upper bound for color scale
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No growth rate data available for the selected years.")
    
    # Option to download the chart
    if st.button("Download Chart", key="download_growth_chart"):
        # Create a high-resolution image
        fig.update_layout(
            width=1200,
            height=800,
        )
        
        # Save to a bytes buffer
        img_bytes = fig.to_image(format="png", scale=2)
        
        # Create a download link
        b64 = base64.b64encode(img_bytes).decode()
        chart_type = "cagr" if growth_metric == "CAGR" else "yoy_growth"
        href = f'<a href="data:image/png;base64,{b64}" download="{chart_type}_chart.png">Download Chart as PNG</a>'
        st.markdown(href, unsafe_allow_html=True)


def render_market_share_analysis(distributed_market: pd.DataFrame, config_manager: ConfigurationManager) -> None:
    """
    Render market share analysis visualization.
    
    Args:
        distributed_market: DataFrame with distributed market data
        config_manager: ConfigurationManager instance
    """
    st.header("Market Share Analysis")
    
    # Get visualization options
    st.subheader("Analysis Options")
    col1, col2 = st.columns(2)
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
    except Exception:
        id_col = 'idGeo'
        country_col = 'Country'
    
    with col1:
        # Reference year
        all_years = sorted(distributed_market['Year'].unique())
        reference_year = st.selectbox(
            "Reference Year",
            options=all_years,
            index=len(all_years) - 1,  # Default to last year
            key="share_reference_year"
        )
        
        # Countries to include
        top_n = st.slider(
            "Top N Countries",
            min_value=5,
            max_value=20,
            value=10,
            key="share_top_n"
        )
    
    with col2:
        # Visualization type
        chart_type = st.radio(
            "Chart Type",
            options=["Pie Chart", "Treemap", "Bar Chart"],
            horizontal=True,
            key="share_chart_type"
        )
        
        # Whether to group smaller countries
        show_others = st.checkbox(
            "Group Smaller Countries as 'Others'",
            value=True,
            key="share_group_others"
        )
    
    # Filter data for the reference year
    year_data = distributed_market[distributed_market['Year'] == reference_year].copy()
    
    # Calculate market share
    total_market = year_data['Value'].sum()
    year_data['Market_Share'] = year_data['Value'] / total_market * 100
    
    # Sort by market share
    year_data = year_data.sort_values(by='Market_Share', ascending=False)
    
    # Handle grouping of smaller countries
    if show_others and len(year_data) > top_n:
        # Get top N countries
        top_data = year_data.head(top_n)
        
        # Group remaining countries as 'Others'
        others_data = year_data.iloc[top_n:]
        others_value = others_data['Value'].sum()
        others_share = others_data['Market_Share'].sum()
        
        # Create a row for 'Others'
        others_row = {
            id_col: -1,
            country_col: 'Others',
            'Year': reference_year,
            'Value': others_value,
            'Market_Share': others_share
        }
        
        # Add other columns if present
        for col in year_data.columns:
            if col not in others_row:
                others_row[col] = None
        
        # Append 'Others' row to top data
        top_data = pd.concat([top_data, pd.DataFrame([others_row])], ignore_index=True)
        
        # Use the top data for visualization
        viz_data = top_data
    else:
        # Only use top N countries without grouping
        viz_data = year_data.head(top_n)
    
    # Create visualization based on chart type
    if chart_type == "Pie Chart":
        # Create a pie chart
        fig = px.pie(
            viz_data,
            values='Market_Share',
            names=country_col,
            title=f'Market Share by Country ({reference_year})',
            hole=0.3,  # Make it a donut chart
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        # Update traces
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='%{label}<br>Market Share: %{value:.1f}%<br>Value: %{customdata[0]:,.0f}',
            customdata=viz_data[['Value']]
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Treemap":
        # Create a treemap
        fig = px.treemap(
            viz_data,
            path=[country_col],
            values='Market_Share',
            title=f'Market Share by Country ({reference_year})',
            color='Market_Share',
            color_continuous_scale=px.colors.sequential.Blues,
            hover_data=['Value']
        )
        
        # Update traces
        fig.update_traces(
            textinfo='label+percent',
            hovertemplate='%{label}<br>Market Share: %{value:.1f}%<br>Value: %{customdata[0]:,.0f}'
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Bar Chart
        # Create a bar chart
        fig = px.bar(
            viz_data,
            x=country_col,
            y='Market_Share',
            title=f'Market Share by Country ({reference_year})',
            labels={'Market_Share': 'Market Share (%)', country_col: 'Country'},
            color='Market_Share',
            color_continuous_scale=px.colors.sequential.Blues,
            text='Market_Share'
        )
        
        # Add value labels
        fig.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='outside'
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Show market share evolution over time
    st.subheader("Market Share Evolution")
    
    # Get countries for evolution analysis
    evolution_countries = viz_data[country_col].tolist()
    if 'Others' in evolution_countries:
        evolution_countries.remove('Others')  # Remove 'Others' for evolution analysis
    
    # Filter years for evolution
    evolution_year_options = all_years.copy()
    
    # If too many years, select a subset
    if len(evolution_year_options) > 10:
        # Get first year, last year, and some years in between
        first_year = evolution_year_options[0]
        last_year = evolution_year_options[-1]
        step = len(evolution_year_options) // 4
        selected_indices = [0] + [i for i in range(step, len(evolution_year_options) - 1, step)] + [len(evolution_year_options) - 1]
        evolution_year_options = [evolution_year_options[i] for i in selected_indices]
    
    # Let user select years for evolution chart
    evolution_years = st.multiselect(
        "Select Years for Evolution Chart",
        options=all_years,
        default=evolution_year_options,
        key="share_evolution_years"
    )
    
    if evolution_years:
        # Filter data for selected countries and years
        evolution_data = distributed_market[
            (distributed_market[country_col].isin(evolution_countries)) &
            (distributed_market['Year'].isin(evolution_years))
        ].copy()
        
        # Calculate market share by year
        evolution_shares = []
        
        for year in evolution_years:
            year_data = distributed_market[distributed_market['Year'] == year]
            year_total = year_data['Value'].sum()
            
            # Calculate share for each selected country
            for country in evolution_countries:
                country_data = year_data[year_data[country_col] == country]
                
                if not country_data.empty:
                    country_value = country_data['Value'].iloc[0]
                    share = country_value / year_total * 100
                else:
                    country_value = 0
                    share = 0
                
                evolution_shares.append({
                    country_col: country,
                    'Year': year,
                    'Value': country_value,
                    'Market_Share': share
                })
        
        # Convert to DataFrame
        evolution_df = pd.DataFrame(evolution_shares)
        
        # Create line chart
        fig = px.line(
            evolution_df,
            x='Year',
            y='Market_Share',
            color=country_col,
            title='Market Share Evolution Over Time',
            labels={'Market_Share': 'Market Share (%)', 'Year': 'Year', country_col: 'Country'},
            markers=True
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Option to download the charts
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Share Chart", key="download_share_chart"):
            # Create a high-resolution image
            fig.update_layout(
                width=1200,
                height=800,
            )
            
            # Save to a bytes buffer
            img_bytes = fig.to_image(format="png", scale=2)
            
            # Create a download link
            b64 = base64.b64encode(img_bytes).decode()
            href = f'<a href="data:image/png;base64,{b64}" download="market_share_chart.png">Download Chart as PNG</a>'
            st.markdown(href, unsafe_allow_html=True)


def render_regional_analysis(distributed_market: pd.DataFrame, config_manager: ConfigurationManager) -> None:
    """
    Render regional analysis visualization.
    
    Args:
        distributed_market: DataFrame with distributed market data
        config_manager: ConfigurationManager instance
    """
    st.header("Regional Analysis")
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
    except Exception:
        id_col = 'idGeo'
        country_col = 'Country'
    
    # Get all countries from data
    all_countries = sorted(distributed_market[country_col].unique())
    
    # Default regions
    default_regions = {
        "North America": ["United States", "Canada", "Mexico"],
        "Europe": ["Germany", "United Kingdom", "France", "Italy", "Spain", "Netherlands", "Switzerland", "Sweden", "Belgium", "Austria"],
        "Asia Pacific": ["China", "Japan", "India", "South Korea", "Australia", "Singapore", "Taiwan", "Indonesia", "Thailand", "Malaysia"],
        "Latin America": ["Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Peru"],
        "Middle East & Africa": ["South Africa", "United Arab Emirates", "Saudi Arabia", "Turkey", "Israel", "Egypt"]
    }
    
    # Filter default regions to only include countries in the data
    filtered_regions = {}
    for region, countries in default_regions.items():
        filtered_countries = [c for c in countries if c in all_countries]
        if filtered_countries:
            filtered_regions[region] = filtered_countries
    
    # Region configuration
    st.subheader("Region Configuration")
    
    # Option to use default regions or manual configuration
    region_config = st.radio(
        "Region Configuration",
        options=["Use Default Regions", "Manual Configuration"],
        horizontal=True,
        key="region_config"
    )
    
    if region_config == "Use Default Regions":
        # Show default regions with option to modify
        with st.expander("View Default Regions"):
            for region, countries in filtered_regions.items():
                st.markdown(f"**{region}**: {', '.join(countries)}")
        
        regions = filtered_regions
    else:
        # Manual region configuration
        regions = {}
        
        # Option to add a new region
        with st.form("add_region_form"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                region_name = st.text_input("Region Name", value="New Region")
                # Enhanced: Add input validation for region name
                if region_name and not region_name.replace(" ", "").replace("-", "").replace("_", "").isalnum():
                    st.warning("Region name should only contain letters, numbers, spaces, hyphens, and underscores")
                    region_name = ""
            
            with col2:
                region_countries = st.multiselect(
                    "Countries",
                    options=all_countries,
                    default=[]
                )
            
            if st.form_submit_button("Add Region"):
                if region_name and region_countries:
                    regions[region_name] = region_countries
                    st.success(f"Added region: {region_name}")
                else:
                    st.error("Please provide a region name and select at least one country")
        
        # Show existing regions (from session state)
        if 'regional_analysis_regions' in st.session_state:
            regions = st.session_state.regional_analysis_regions
        
        # Display existing regions with option to edit
        if regions:
            st.subheader("Existing Regions")
            
            for region, countries in regions.items():
                with st.expander(f"{region} ({len(countries)} countries)"):
                    # Show countries in this region
                    st.write(", ".join(countries))
                    
                    # Option to remove region
                    if st.button(f"Remove {region}", key=f"remove_{region}"):
                        del regions[region]
                        st.success(f"Removed region: {region}")
                        st.rerun()
        
        # Save regions to session state
        st.session_state.regional_analysis_regions = regions
    
    # Check if we have any regions
    if not regions:
        st.warning("No regions defined. Please add at least one region.")
        return
    
    # Visualization options
    st.subheader("Visualization Options")
    col1, col2 = st.columns(2)
    
    with col1:
        # Reference year
        all_years = sorted(distributed_market['Year'].unique())
        reference_year = st.selectbox(
            "Reference Year",
            options=all_years,
            index=len(all_years) - 1,  # Default to last year
            key="regional_reference_year"
        )
        
        # Visualization type
        vis_type = st.radio(
            "Visualization Type",
            options=["Pie Chart", "Bar Chart"],
            horizontal=True,
            key="regional_vis_type"
        )
    
    with col2:
        # Metric to display
        metric = st.radio(
            "Metric",
            options=["Market Size", "Market Share", "Growth Rate"],
            horizontal=True,
            key="regional_metric"
        )
        
        # For growth rate, need start year
        if metric == "Growth Rate":
            # Find years before reference year
            prev_years = [y for y in all_years if y < reference_year]
            
            if prev_years:
                start_year = st.selectbox(
                    "Start Year",
                    options=prev_years,
                    index=len(prev_years) - 1,  # Default to latest previous year
                    key="regional_start_year"
                )
            else:
                st.warning("No previous years available for growth rate calculation.")
                start_year = None
        else:
            start_year = None
    
    # Process data for regional analysis
    region_data = []
    
    # Calculate metrics for each region
    for region, countries in regions.items():
        # Filter data for this region
        region_countries = distributed_market[distributed_market[country_col].isin(countries)]
        
        if metric == "Market Size" or metric == "Market Share":
            # Reference year data
            ref_year_data = region_countries[region_countries['Year'] == reference_year]
            ref_year_value = ref_year_data['Value'].sum()
            
            # For market share, get global total
            if metric == "Market Share":
                global_data = distributed_market[distributed_market['Year'] == reference_year]
                global_value = global_data['Value'].sum()
                ref_year_share = ref_year_value / global_value * 100
                
                region_data.append({
                    'Region': region,
                    'Value': ref_year_value,
                    'Share': ref_year_share,
                    'CountryCount': len(countries),
                    'Countries': countries
                })
            else:
                region_data.append({
                    'Region': region,
                    'Value': ref_year_value,
                    'CountryCount': len(countries),
                    'Countries': countries
                })
        
        elif metric == "Growth Rate" and start_year is not None:
            # Calculate growth rate between start_year and reference_year
            start_year_data = region_countries[region_countries['Year'] == start_year]
            ref_year_data = region_countries[region_countries['Year'] == reference_year]
            
            start_year_value = start_year_data['Value'].sum()
            ref_year_value = ref_year_data['Value'].sum()
            
            if start_year_value > 0:
                # Calculate CAGR
                years_diff = reference_year - start_year
                cagr = ((ref_year_value / start_year_value) ** (1 / years_diff) - 1) * 100
                
                region_data.append({
                    'Region': region,
                    'StartValue': start_year_value,
                    'EndValue': ref_year_value,
                    'Growth': cagr,
                    'CountryCount': len(countries),
                    'Countries': countries
                })
    
    # Convert to DataFrame
    region_df = pd.DataFrame(region_data)
    
    # Create visualization based on metric and type
    if not region_df.empty:
        if metric == "Market Size":
            # Sort by value
            region_df = region_df.sort_values(by='Value', ascending=False)
            
            if vis_type == "Pie Chart":
                # Create a pie chart
                fig = px.pie(
                    region_df,
                    values='Value',
                    names='Region',
                    title=f'Market Size by Region ({reference_year})',
                    hole=0.3,  # Make it a donut chart
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                # Update traces
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='%{label}<br>Market Size: %{value:,.0f}<br>Countries: %{customdata[0]}',
                    customdata=region_df[['CountryCount']]
                )
            else:  # Bar Chart
                # Create a bar chart
                fig = px.bar(
                    region_df,
                    x='Region',
                    y='Value',
                    title=f'Market Size by Region ({reference_year})',
                    labels={'Value': 'Market Size', 'Region': 'Region'},
                    color='Value',
                    color_continuous_scale=px.colors.sequential.Blues,
                    text='Value'
                )
                
                # Update traces
                fig.update_traces(
                    texttemplate='%{y:,.0f}',
                    textposition='outside'
                )
        
        elif metric == "Market Share":
            # Sort by share
            region_df = region_df.sort_values(by='Share', ascending=False)
            
            if vis_type == "Pie Chart":
                # Create a pie chart
                fig = px.pie(
                    region_df,
                    values='Share',
                    names='Region',
                    title=f'Market Share by Region ({reference_year})',
                    hole=0.3,  # Make it a donut chart
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                # Update traces
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='%{label}<br>Market Share: %{value:.1f}%<br>Value: %{customdata[0]:,.0f}<br>Countries: %{customdata[1]}',
                    customdata=region_df[['Value', 'CountryCount']]
                )
            else:  # Bar Chart
                # Create a bar chart
                fig = px.bar(
                    region_df,
                    x='Region',
                    y='Share',
                    title=f'Market Share by Region ({reference_year})',
                    labels={'Share': 'Market Share (%)', 'Region': 'Region'},
                    color='Share',
                    color_continuous_scale=px.colors.sequential.Blues,
                    text='Share'
                )
                
                # Update traces
                fig.update_traces(
                    texttemplate='%{y:.1f}%',
                    textposition='outside'
                )
        
        elif metric == "Growth Rate" and start_year is not None:
            # Sort by growth
            region_df = region_df.sort_values(by='Growth', ascending=False)
            
            if vis_type == "Pie Chart":
                # Pie chart doesn't make sense for growth rates
                st.warning("Pie chart not suitable for growth rates. Showing bar chart instead.")
                vis_type = "Bar Chart"
            
            # Create a bar chart
            fig = px.bar(
                region_df,
                x='Region',
                y='Growth',
                title=f'CAGR by Region ({start_year}-{reference_year})',
                labels={'Growth': 'CAGR (%)', 'Region': 'Region'},
                color='Growth',
                color_continuous_scale=px.colors.diverging.RdBu_r,
                color_continuous_midpoint=0,
                text='Growth'
            )
            
            # Update traces
            fig.update_traces(
                texttemplate='%{y:.1f}%',
                textposition='outside'
            )
            
            # Add a reference line at 0%
            fig.add_shape(
                type='line',
                x0=-0.5,
                x1=len(region_df) - 0.5,
                y0=0,
                y1=0,
                line=dict(color='gray', width=1, dash='dash')
            )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional details
        st.subheader("Regional Details")
        
        # Display detailed information for each region
        for idx, row in region_df.iterrows():
            with st.expander(f"{row['Region']} ({len(row['Countries'])} countries)"):
                st.write(f"**Countries**: {', '.join(row['Countries'])}")
                
                if metric == "Market Size":
                    st.write(f"**Market Size**: {row['Value']:,.0f}")
                elif metric == "Market Share":
                    st.write(f"**Market Share**: {row['Share']:.1f}%")
                    st.write(f"**Market Size**: {row['Value']:,.0f}")
                elif metric == "Growth Rate" and start_year is not None:
                    st.write(f"**CAGR ({start_year}-{reference_year})**: {row['Growth']:.1f}%")
                    st.write(f"**Start Value ({start_year})**: {row['StartValue']:,.0f}")
                    st.write(f"**End Value ({reference_year})**: {row['EndValue']:,.0f}")
    else:
        st.warning("No regional data available for the selected parameters.")

def render_data_table(distributed_market: pd.DataFrame, config_manager: ConfigurationManager) -> None:
    """
    Render data table with filtering and download options.
    
    Args:
        distributed_market: DataFrame with distributed market data
        config_manager: ConfigurationManager instance
    """
    st.header("Data Table")
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
    except Exception:
        id_col = 'idGeo'
        country_col = 'Country'
    
    # Filter options
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Country filter
        all_countries = sorted(distributed_market[country_col].unique())
        selected_countries = st.multiselect(
            "Countries",
            options=all_countries,
            default=[],
            key="table_countries"
        )
    
    with col2:
        # Year filter
        all_years = sorted(distributed_market['Year'].unique())
        selected_years = st.multiselect(
            "Years",
            options=all_years,
            default=[],
            key="table_years"
        )
    
    with col3:
        # Sort options
        sort_by = st.selectbox(
            "Sort By",
            options=["Country", "Year", "Value"],
            index=1,
            key="table_sort"
        )
        
        sort_order = st.radio(
            "Sort Order",
            options=["Ascending", "Descending"],
            horizontal=True,
            index=0,
            key="table_sort_order"
        )
    
    # Filter data
    filtered_data = distributed_market.copy()
    
    if selected_countries:
        filtered_data = filtered_data[filtered_data[country_col].isin(selected_countries)]
    
    if selected_years:
        filtered_data = filtered_data[filtered_data['Year'].isin(selected_years)]
    
    # Sort data
    if sort_by == "Country":
        filtered_data = filtered_data.sort_values(
            by=[country_col, 'Year'],
            ascending=sort_order == "Ascending"
        )
    elif sort_by == "Year":
        filtered_data = filtered_data.sort_values(
            by=['Year', country_col],
            ascending=sort_order == "Ascending"
        )
    else:  # Value
        filtered_data = filtered_data.sort_values(
            by=['Value', 'Year'],
            ascending=sort_order == "Ascending"
        )
    
    # Display data
    st.dataframe(
        filtered_data,
        use_container_width=True,
        height=400
    )
    
    # Download options
    st.subheader("Download Options")
    col1, col2 = st.columns(2)
    
    with col1:
        file_format = st.radio(
            "File Format",
            options=["CSV", "Excel", "JSON"],
            horizontal=True,
            key="table_format"
        )
    
    with col2:
        file_name = st.text_input(
            "File Name",
            value="market_data_export",
            key="table_filename"
        )
        # Enhanced: Add input validation for file name
        if file_name and not file_name.replace("_", "").replace("-", "").isalnum():
            st.warning("File name should only contain letters, numbers, hyphens, and underscores")
            file_name = "market_data_export"
    
    # Download button
    if st.button("Download Data", key="download_data_button"):
        try:
            # Create appropriate file extension
            if file_format == "CSV":
                file_ext = "csv"
                mime_type = "text/csv"
                
                # Convert to CSV
                csv_data = filtered_data.to_csv(index=False)
                b64 = base64.b64encode(csv_data.encode()).decode()
                
            elif file_format == "Excel":
                file_ext = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
                # Convert to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_data.to_excel(writer, index=False, sheet_name='Market Data')
                
                output.seek(0)
                b64 = base64.b64encode(output.read()).decode()
                
            else:  # JSON
                file_ext = "json"
                mime_type = "application/json"
                
                # Convert to JSON
                json_data = filtered_data.to_json(orient='records', date_format='iso')
                b64 = base64.b64encode(json_data.encode()).decode()
            
            # Create download link
            href = f'<a href="data:{mime_type};base64,{b64}" download="{file_name}.{file_ext}">Download {file_format}</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")


def render_calibration_metrics(distributed_market: pd.DataFrame, config_manager: ConfigurationManager) -> None:
    """
    Render calibration metrics visualization.
    
    Args:
        distributed_market: DataFrame with distributed market data
        config_manager: ConfigurationManager instance
    """
    st.header("Forecast Calibration Metrics")
    
    # Check if we have a market analyzer available
    if 'market_analyzer' not in st.session_state or st.session_state.market_analyzer is None:
        st.info("No calibration metrics available. Go to the Auto-Calibration page to evaluate and calibrate your models.")
        
        if st.button("Go to Auto-Calibration"):
            st.session_state.active_page = "Auto-Calibration"
            st.rerun()
        
        return
    
    # Get market analyzer
    market_analyzer = st.session_state.market_analyzer
    
    # Check if we have calibration metrics
    if not hasattr(market_analyzer, 'auto_calibrator') or not hasattr(market_analyzer.auto_calibrator, 'calibration_history'):
        st.info("No calibration metrics available. Go to the Auto-Calibration page to evaluate and calibrate your models.")
        
        if st.button("Go to Auto-Calibration"):
            st.session_state.active_page = "Auto-Calibration"
            st.rerun()
        
        return
    
    # Get calibration history
    calibration_history = market_analyzer.auto_calibrator.calibration_history
    
    if not calibration_history:
        st.info("No calibration history available. Go to the Auto-Calibration page to evaluate and calibrate your models.")
        
        if st.button("Go to Auto-Calibration"):
            st.session_state.active_page = "Auto-Calibration"
            st.rerun()
        
        return
    
    # Display calibration metrics
    st.subheader("Forecast Accuracy Metrics")
    
    # Extract metrics from calibration history
    latest_calibration = calibration_history[-1]
    
    if 'metrics' in latest_calibration and 'overall' in latest_calibration['metrics']:
        overall_metrics = latest_calibration['metrics']['overall']
        
        # Display metrics as cards
        metric_cols = st.columns(len(overall_metrics))
        
        for i, (metric, value) in enumerate(overall_metrics.items()):
            with metric_cols[i]:
                if metric == 'mape':
                    st.metric("MAPE", f"{value:.2f}%")
                elif metric == 'rmse':
                    st.metric("RMSE", f"{value:.2f}")
                elif metric == 'r2':
                    st.metric("R²", f"{value:.4f}")
                elif metric == 'bias':
                    st.metric("Bias", f"{value:.2f}%")
    
    # Display accuracy trends
    if len(calibration_history) > 1:
        st.subheader("Accuracy Trends")
        
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
    
    # Display country performance
    st.subheader("Country Performance Analysis")
    
    # Check if we have country performance data
    if hasattr(market_analyzer.auto_calibrator, 'country_performance') and market_analyzer.auto_calibrator.country_performance:
        country_performance = market_analyzer.auto_calibrator.country_performance
        
        if country_performance:
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
        else:
            st.info("No country performance data available.")
    else:
        st.info("No country performance data available.")
    
    # Display component weights
    st.subheader("Component Performance")
    
    # Check if we have component performance data
    if hasattr(market_analyzer.auto_calibrator, 'component_performance') and market_analyzer.auto_calibrator.component_performance:
        component_performance = market_analyzer.auto_calibrator.component_performance
        
        if component_performance:
            # Extract component weights
            component_weights_data = []
            for component, data in component_performance.items():
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
        else:
            st.info("No component performance data available.")
    else:
        st.info("No component performance data available.")
    
    # Call to action
    st.markdown("---")
    st.markdown("For more detailed analysis and calibration options, go to the Auto-Calibration page.")
    
    if st.button("Go to Auto-Calibration", key="go_to_calibration"):
        st.session_state.active_page = "Auto-Calibration"
        st.rerun()


def render_visualization_interface(config_manager: ConfigurationManager) -> None:
    """
    Render market visualization interface.
    
    Args:
        config_manager: ConfigurationManager instance
    """
    st.title("Market Visualization")
    
    # Check if we have distributed market data
    if 'distributed_market' not in st.session_state or st.session_state.distributed_market is None:
        st.warning("No distributed market data available for visualization.")
        
        # Show options based on what data we have
        if st.button("Run Market Distribution", key="run_distribution_button"):
            st.session_state.active_page = "Market Distribution"
            st.rerun()
        
        return
    
    # Create tabs for different visualization types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Market Size", "Growth Analysis", "Market Share", "Regional Analysis", "Data Table", "Calibration Metrics"
    ])
    
    # Get distributed market data
    distributed_market = st.session_state.distributed_market
    
    # Tab 1: Market Size Visualization
    with tab1:
        render_market_size_visualization(distributed_market, config_manager)
    
    # Tab 2: Growth Analysis
    with tab2:
        render_growth_analysis(distributed_market, config_manager)
    
    # Tab 3: Market Share
    with tab3:
        render_market_share_analysis(distributed_market, config_manager)
    
    # Tab 4: Regional Analysis
    with tab4:
        render_regional_analysis(distributed_market, config_manager)
    
    # Tab 5: Data Table
    with tab5:
        render_data_table(distributed_market, config_manager)
        
    # Tab 6: Calibration Metrics
    with tab6:
        render_calibration_metrics(distributed_market, config_manager)