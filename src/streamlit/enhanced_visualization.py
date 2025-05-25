"""
Enhanced Visualization Interface - Next Generation Market Visualizations

This module provides state-of-the-art, interactive visualizations for market data
with professional design, advanced interactivity, and comprehensive insights.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import io
import base64
from src.streamlit.yoy_change_viz import YoYChangeVisualizer

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Professional color schemes
COLOR_SCHEMES = {
    'corporate': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
                  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'],
    'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', 
                '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA'],
    'professional': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941', '#4B5D67',
                     '#87BBA2', '#F2CC8F', '#E07A5F', '#81B29A', '#3D5A80', '#98C1D9'],
    'modern': ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#6A4C93',
               '#8338EC', '#3A86FF', '#06FFA5', '#FB5607', '#FFBE0B', '#8338EC']
}

def format_value(value: float) -> str:
    """
    Format numeric values dynamically based on their scale.
    
    Args:
        value: The numeric value to format
        
    Returns:
        Formatted string with appropriate suffix (T, B, M, K) or decimal places
    """
    if value >= 1e12:  # Trillions
        return f"${value/1e12:.1f}T"
    elif value >= 1e9:  # Billions
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:  # Millions
        return f"${value/1e6:.1f}M"
    elif value >= 1e3:  # Thousands
        return f"${value/1e3:.0f}K"
    elif value > 0:  # Small positive values
        if value < 1:
            return f"${value:.4f}"
        else:
            return f"${value:,.0f}"
    else:  # Zero or negative
        return "$0"

def set_page_style():
    """Set enhanced page styling"""
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background: #d1edff;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .trend-up {
        color: #28a745;
        font-weight: bold;
    }
    
    .trend-down {
        color: #dc3545;
        font-weight: bold;
    }
    
    .trend-stable {
        color: #6c757d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create an enhanced metric card"""
    delta_html = ""
    if delta:
        color_class = f"trend-{delta_color}" if delta_color in ["up", "down", "stable"] else ""
        delta_html = f'<div class="{color_class}" style="font-size: 0.8em; margin-top: 0.5rem;">{delta}</div>'
    
    st.markdown(f"""
    <div class="kpi-card">
        <div style="font-size: 0.9em; opacity: 0.9;">{title}</div>
        <div style="font-size: 2em; font-weight: bold; margin: 0.5rem 0;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_insight_box(content: str, type: str = "info"):
    """Create an insight box with different styling"""
    css_class = f"{type}-box"
    st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

def render_executive_dashboard(distributed_market: pd.DataFrame, config_manager) -> None:
    """
    Render an executive dashboard with key insights and KPIs
    """
    st.title("📊 Executive Dashboard")
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        country_col = country_mapping.get('country_column', 'Country')
    except:
        country_col = 'Country'
    
    # Calculate key metrics
    # Ensure Year column is integer type to avoid comparison issues
    distributed_market['Year'] = distributed_market['Year'].astype(int)
    
    latest_year = distributed_market['Year'].max()
    earliest_year = distributed_market['Year'].min()
    total_years = latest_year - earliest_year + 1
    
    # Market size metrics - filter to country data only if regions exist
    if 'region_type' in distributed_market.columns:
        country_data = distributed_market[distributed_market['region_type'] == 'country']
    else:
        country_data = distributed_market
    
    latest_data = country_data[country_data['Year'] == latest_year]
    
    # Add debug info for data checking
    if 'Value' not in country_data.columns:
        st.error("Value column not found in data DataFrame")
        st.write("Available columns:", country_data.columns.tolist())
        return
    
    # Check for data issues
    value_sum = country_data['Value'].sum()
    if value_sum == 0 or pd.isna(value_sum):
        st.warning("⚠️ Data Issue Detected: All values are zero or missing.")
        with st.expander("🔍 Debug Information"):
            st.write(f"- Total rows in distributed_market: {len(distributed_market)}")
            st.write(f"- Total rows in country_data: {len(country_data)}")
            st.write(f"- Unique years: {sorted(country_data['Year'].unique())}")
            st.write(f"- Value column statistics:")
            st.write(country_data['Value'].describe())
            st.write(f"- Sample of data (first 10 rows):")
            st.write(country_data.head(10))
            st.write(f"- Unique countries: {country_data[country_col].unique()}")
            
            # Check if values are in different units
            max_value = country_data['Value'].max()
            if max_value > 0 and max_value < 1:
                st.info("💡 Values appear to be in fractional units. They may need to be multiplied by a scaling factor (e.g., 1000 or 1,000,000).")
    
    # Get the most recent year with meaningful data - use country data only
    year_totals = country_data.groupby('Year')['Value'].sum()
    
    
    # Use dynamic threshold: 1% of maximum yearly value or minimum of 100
    max_yearly_total = year_totals.max()
    threshold = max(100, max_yearly_total * 0.01)
    
    # Find years with meaningful data (above threshold)
    meaningful_years = year_totals[year_totals > threshold].index
    
    if len(meaningful_years) > 0:
        # Use the most recent year with meaningful data
        display_year = max(meaningful_years)
        latest_data = country_data[country_data['Year'] == display_year]
        total_market_size = latest_data['Value'].sum()
    else:
        # Fallback to original calculation
        total_market_size = latest_data['Value'].sum()
        display_year = latest_year
    
    # Debug: Check if we have data
    if total_market_size == 0 or total_market_size < 1000:
        st.warning(f"⚠️ Low/zero total market size: {total_market_size:,.0f} for year {display_year}")
        
        with st.expander("🔍 Debugging Information"):
            st.write("**Latest data sample:**")
            st.dataframe(latest_data.head())
            st.write("**Value column stats:**")
            st.write(latest_data['Value'].describe())
            
            # Check all years to find where data exists
            year_totals_debug = distributed_market.groupby('Year')['Value'].sum().sort_index()
            st.write("**Market size by year:**")
            st.write(year_totals_debug)
            
            # Check if there are regional entries affecting totals
            if 'region_type' in distributed_market.columns:
                region_totals = distributed_market.groupby(['Year', 'region_type'])['Value'].sum().unstack(fill_value=0)
                st.write("**Totals by region type:**")
                st.write(region_totals)
        
        # Find the last year with non-zero data
        non_zero_years = year_totals[year_totals > 0]
        if not non_zero_years.empty:
            last_good_year = non_zero_years.index[-1]
            st.info(f"📅 Using last year with data: {last_good_year} (total: {year_totals[last_good_year]:,.0f})")
            # Use that year's data instead
            latest_data = country_data[country_data['Year'] == last_good_year]
            total_market_size = latest_data['Value'].sum()
            display_year = last_good_year
    
    num_countries = latest_data[country_col].nunique()
    
    # Growth metrics
    if total_years > 1:
        earliest_data = country_data[country_data['Year'] == earliest_year]
        earliest_total = earliest_data['Value'].sum()
        if earliest_total > 0:
            cagr = ((total_market_size / earliest_total) ** (1 / (total_years - 1)) - 1) * 100
        else:
            cagr = 0
    else:
        cagr = 0
    
    # Top countries
    top_countries = latest_data.nlargest(3, 'Value')[country_col].tolist()
    top_3_share = latest_data.nlargest(3, 'Value')['Value'].sum() / total_market_size * 100 if total_market_size > 0 else 0
    
    # Create KPI cards
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Show which year we're displaying if it's different from latest
        year_label = f" ({display_year})" if 'display_year' in locals() and display_year != latest_year else f" ({latest_year})"
        # Smart formatting for market size with better handling of small values
        size_display = format_value(total_market_size)
        
        create_metric_card(
            f"Total Market Size{year_label}", 
            size_display,
            f"CAGR: {cagr:+.1f}%" if cagr != 0 else "Snapshot",
            "up" if cagr > 0 else "down" if cagr < 0 else "stable"
        )
    
    with col2:
        create_metric_card(
            "Countries Analyzed", 
            str(num_countries),
            f"Across {total_years} years",
            "stable"
        )
    
    with col3:
        create_metric_card(
            "Market Leaders", 
            f"Top 3: {top_3_share:.1f}%",
            f"{', '.join(top_countries[:2])}...",
            "stable"
        )
    
    with col4:
        # Determine forecast years dynamically
        # Assume forecast years are those after the last year with complete country data
        years_with_data = distributed_market.groupby('Year').size()
        # Find the transition point (could be based on data completeness or other criteria)
        # For now, we'll count all years as the horizon
        total_years_span = len(distributed_market['Year'].unique())
        create_metric_card(
            "Data Horizon", 
            f"{total_years_span} years",
            f"Through {latest_year}",
            "stable"
        )
    
    # Market overview chart
    st.markdown("### 🌍 Global Market Overview")
    
    # Create market size evolution
    yearly_totals = distributed_market.groupby('Year')['Value'].sum().reset_index()
    
    # Debug: Check for missing years
    all_years = set(range(int(yearly_totals['Year'].min()), int(yearly_totals['Year'].max()) + 1))
    present_years = set(yearly_totals['Year'].astype(int))
    missing_years = all_years - present_years
    if missing_years:
        st.warning(f"Missing data for years: {sorted(missing_years)}")
        st.info(f"Years in distributed_market: {sorted(distributed_market['Year'].unique())}")
        st.info(f"Years in yearly_totals: {sorted(yearly_totals['Year'].unique())}")
        
        # Check if 2024 exists in the raw data
        if 2024 in distributed_market['Year'].values:
            st.error("2024 exists in distributed_market but not in yearly_totals!")
            year_2024_data = distributed_market[distributed_market['Year'] == 2024]
            st.write(f"2024 data shape: {year_2024_data.shape}")
            st.write(f"2024 total value: {year_2024_data['Value'].sum()}")
    
    fig = go.Figure()
    
    # Plot as a single continuous line
    fig.add_trace(go.Scatter(
        x=yearly_totals['Year'],
        y=yearly_totals['Value'],
        mode='lines+markers',
        name='Market Size',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Market Size: $%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'Global Market Evolution ({earliest_year}-{latest_year})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Year',
        yaxis_title='Market Size (USD)',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="executive_dashboard_global_market_overview")
    
    # Top countries breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top 10 Countries by Market Size")
        top_10 = latest_data.nlargest(10, 'Value')
        top_10['Market Share'] = top_10['Value'] / total_market_size * 100
        
        # Create horizontal bar chart
        fig = px.bar(
            top_10.iloc[::-1],  # Reverse for better display
            x='Market Share',
            y=country_col,
            orientation='h',
            color='Market Share',
            color_continuous_scale='Blues',
            title=f'Market Share Distribution ({latest_year})'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            coloraxis_showscale=False
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Share: %{x:.1f}%<br>Value: $%{customdata:,.0f}<extra></extra>',
            customdata=top_10.iloc[::-1]['Value']
        )
        
        st.plotly_chart(fig, use_container_width=True, key="executive_dashboard_top_countries_bar")
    
    with col2:
        st.markdown("### 📊 Market Concentration Analysis")
        
        # Calculate market concentration
        top_1_share = latest_data.nlargest(1, 'Value')['Value'].sum() / total_market_size * 100
        top_5_share = latest_data.nlargest(5, 'Value')['Value'].sum() / total_market_size * 100
        top_10_share = latest_data.nlargest(10, 'Value')['Value'].sum() / total_market_size * 100
        
        concentration_data = pd.DataFrame({
            'Segment': ['Top 1', 'Top 2-5', 'Top 6-10', 'Others'],
            'Share': [
                top_1_share,
                top_5_share - top_1_share,
                top_10_share - top_5_share,
                100 - top_10_share
            ]
        })
        
        fig = px.pie(
            concentration_data,
            values='Share',
            names='Segment',
            color_discrete_sequence=COLOR_SCHEMES['professional'],
            title='Market Concentration Analysis'
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Share: %{value:.1f}%<extra></extra>'
        )
        
        fig.update_layout(height=400, template='plotly_white')
        
        st.plotly_chart(fig, use_container_width=True, key="executive_dashboard_concentration_pie")
    
    # Insights section
    st.markdown("### 💡 Key Insights")
    
    # Market concentration insight
    if top_3_share > 70:
        create_insight_box(
            f"🔴 <b>Highly Concentrated Market:</b> Top 3 countries control {top_3_share:.1f}% of the market, "
            f"indicating a concentrated market structure with {', '.join(top_countries)} as dominant players.",
            "warning"
        )
    elif top_3_share > 50:
        create_insight_box(
            f"🟡 <b>Moderately Concentrated:</b> Top 3 countries hold {top_3_share:.1f}% market share, "
            f"showing moderate concentration with {top_countries[0]} leading the market.",
            "info"
        )
    else:
        create_insight_box(
            f"🟢 <b>Fragmented Market:</b> Top 3 countries control only {top_3_share:.1f}% of the market, "
            f"indicating a fragmented market structure with opportunities for growth.",
            "success"
        )
    
    # Growth insight
    if cagr > 15:
        create_insight_box(
            f"🚀 <b>High Growth Market:</b> The market is experiencing exceptional growth at {cagr:.1f}% CAGR, "
            f"indicating strong expansion and investment opportunities.",
            "success"
        )
    elif cagr > 5:
        create_insight_box(
            f"📈 <b>Steady Growth:</b> The market shows healthy growth at {cagr:.1f}% CAGR, "
            f"suggesting stable expansion and sustainable development.",
            "info"
        )
    elif cagr > 0:
        create_insight_box(
            f"🐌 <b>Slow Growth:</b> The market is growing slowly at {cagr:.1f}% CAGR, "
            f"which may indicate market maturity or challenging conditions.",
            "warning"
        )
    else:
        create_insight_box(
            f"📉 <b>Declining Market:</b> The market is contracting at {cagr:.1f}% CAGR, "
            f"requiring strategic analysis and potential intervention.",
            "warning"
        )

def render_enhanced_market_size_visualization(distributed_market: pd.DataFrame, config_manager) -> None:
    """
    Enhanced market size visualization with multiple view options and interactivity
    """
    st.title("📈 Enhanced Market Size Analysis")
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        country_col = country_mapping.get('country_column', 'Country')
    except:
        country_col = 'Country'
    
    # Advanced controls
    st.markdown("### 🎛️ Visualization Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Area Chart", "Line Chart", "Bar Chart", "Animated Bubble", "Waterfall"],
            help="Select the type of visualization",
            key="enhanced_viz_chart_type"
        )
    
    with col2:
        time_scope = st.selectbox(
            "Time Scope",
            ["All Years", "Historical Only", "Forecast Only", "Custom Range"],
            help="Choose which years to display",
            key="enhanced_viz_time_scope"
        )
    
    with col3:
        countries_to_show = st.selectbox(
            "Countries",
            ["Top 10", "Top 15", "Top 20", "All Countries", "Custom Selection"],
            help="Select which countries to include",
            key="enhanced_viz_countries"
        )
    
    with col4:
        color_theme = st.selectbox(
            "Color Theme",
            ["Corporate", "Vibrant", "Professional", "Modern"],
            help="Choose color scheme",
            key="enhanced_viz_color_theme"
        )
    
    # Filter data based on selections
    filtered_data = distributed_market.copy()
    
    # Time filtering
    if time_scope == "Historical Only":
        filtered_data = filtered_data[filtered_data['Year'] <= 2023]
    elif time_scope == "Forecast Only":
        filtered_data = filtered_data[filtered_data['Year'] > 2023]
    elif time_scope == "Custom Range":
        years = sorted(distributed_market['Year'].unique())
        year_range = st.slider(
            "Select Year Range",
            min_value=int(years[0]),
            max_value=int(years[-1]),
            value=(int(years[0]), int(years[-1])),
            help="Choose custom year range",
            key="enhanced_viz_year_range"
        )
        filtered_data = filtered_data[
            (filtered_data['Year'] >= year_range[0]) & 
            (filtered_data['Year'] <= year_range[1])
        ]
    
    # Country filtering
    latest_year = filtered_data['Year'].max()
    latest_data = filtered_data[filtered_data['Year'] == latest_year]
    
    if countries_to_show == "Top 10":
        top_countries = latest_data.nlargest(10, 'Value')[country_col].tolist()
    elif countries_to_show == "Top 15":
        top_countries = latest_data.nlargest(15, 'Value')[country_col].tolist()
    elif countries_to_show == "Top 20":
        top_countries = latest_data.nlargest(20, 'Value')[country_col].tolist()
    elif countries_to_show == "Custom Selection":
        all_countries = sorted(filtered_data[country_col].unique())
        top_countries = st.multiselect(
            "Select Countries",
            all_countries,
            default=latest_data.nlargest(10, 'Value')[country_col].tolist(),
            help="Choose specific countries to analyze",
            key="enhanced_viz_custom_countries"
        )
    else:
        top_countries = filtered_data[country_col].unique().tolist()
    
    if not top_countries:
        st.warning("Please select at least one country to visualize.")
        return
    
    display_data = filtered_data[filtered_data[country_col].isin(top_countries)]
    colors = COLOR_SCHEMES[color_theme.lower()]
    
    # Create visualization based on chart type
    if chart_type == "Area Chart":
        # Stacked area chart
        fig = create_stacked_area_chart(display_data, country_col, colors)
    elif chart_type == "Line Chart":
        # Multi-line chart
        fig = create_multi_line_chart(display_data, country_col, colors)
    elif chart_type == "Bar Chart":
        # Animated bar chart
        fig = create_animated_bar_chart(display_data, country_col, colors)
    elif chart_type == "Animated Bubble":
        # Animated bubble chart
        fig = create_animated_bubble_chart(display_data, country_col, colors)
    else:  # Waterfall
        # Waterfall chart for growth analysis
        fig = create_waterfall_chart(display_data, country_col, colors)
    
    st.plotly_chart(fig, use_container_width=True, key="market_size_visualization_main")
    
    # Additional insights
    st.markdown("### 📊 Market Size Insights")
    
    # Calculate insights
    total_market = display_data.groupby('Year')['Value'].sum()
    if len(total_market) > 1 and total_market.iloc[0] > 0:
        growth_rate = ((total_market.iloc[-1] / total_market.iloc[0]) ** (1/(len(total_market)-1)) - 1) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Format start and end values appropriately
            start_value = total_market.iloc[0]
            end_value = total_market.iloc[-1]
            
            create_insight_box(
                f"📈 <b>Market Growth:</b> The selected market segment is growing at {growth_rate:.1f}% CAGR "
                f"over the analyzed period, from {format_value(start_value)} to {format_value(end_value)}.",
                "info"
            )
        
        with col2:
            # Market leader analysis
            leader_data = display_data[display_data['Year'] == latest_year].nlargest(1, 'Value')
            if not leader_data.empty and len(total_market) > 0 and total_market.iloc[-1] > 0:
                leader = leader_data.iloc[0][country_col]
                leader_share = leader_data.iloc[0]['Value'] / total_market.iloc[-1] * 100
                
                leader_value = leader_data.iloc[0]['Value']
                create_insight_box(
                    f"👑 <b>Market Leader:</b> {leader} dominates with {leader_share:.1f}% market share "
                    f"({format_value(leader_value)}), establishing clear market leadership.",
                    "success"
                )

def create_stacked_area_chart(data: pd.DataFrame, country_col: str, colors: List[str]) -> go.Figure:
    """Create an enhanced stacked area chart"""
    fig = go.Figure()
    
    years = sorted(data['Year'].unique())
    countries = data[country_col].unique()
    
    for i, country in enumerate(countries):
        country_data = data[data[country_col] == country].sort_values('Year')
        
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Value'],
            mode='lines',
            stackgroup='one',
            name=country,
            line=dict(color=colors[i % len(colors)], width=0),
            fillcolor=colors[i % len(colors)],
            hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Market Size Evolution - Stacked Area View',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Year',
        yaxis_title='Market Size (USD)',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    return fig

def create_multi_line_chart(data: pd.DataFrame, country_col: str, colors: List[str]) -> go.Figure:
    """Create an enhanced multi-line chart"""
    fig = go.Figure()
    
    countries = data[country_col].unique()
    
    for i, country in enumerate(countries):
        country_data = data[data[country_col] == country].sort_values('Year')
        
        # Plot all data as a single continuous line
        fig.add_trace(go.Scatter(
            x=country_data['Year'],
            y=country_data['Value'],
            mode='lines+markers',
            name=country,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=6),
            hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Market Size Trends by Country',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Year',
        yaxis_title='Market Size (USD)',
        template='plotly_white',
        height=600,
        hovermode='x unified'
    )
    
    return fig

def create_animated_bar_chart(data: pd.DataFrame, country_col: str, colors: List[str]) -> go.Figure:
    """Create an animated bar chart showing market evolution"""
    # Prepare data for animation
    years = sorted(data['Year'].unique())
    
    # Get top countries for consistent display
    latest_data = data[data['Year'] == years[-1]]
    top_countries = latest_data.nlargest(15, 'Value')[country_col].tolist()
    
    animation_data = []
    for year in years:
        year_data = data[data['Year'] == year]
        year_data = year_data[year_data[country_col].isin(top_countries)]
        year_data = year_data.sort_values('Value', ascending=True)  # For horizontal bar
        
        animation_data.append(year_data)
    
    # Create animated bar chart
    fig = px.bar(
        animation_data[0],
        x='Value',
        y=country_col,
        orientation='h',
        color=country_col,
        color_discrete_sequence=colors,
        title='Market Size Evolution - Animated View'
    )
    
    # Add animation frames
    frames = []
    for i, year_data in enumerate(animation_data):
        frame = go.Frame(
            data=[go.Bar(
                x=year_data['Value'],
                y=year_data[country_col],
                orientation='h',
                marker_color=colors[:len(year_data)]
            )],
            name=str(years[i])
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True}]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                }
            ]
        }],
        sliders=[{
            'steps': [
                {
                    'args': [[f.name], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                    'label': f.name,
                    'method': 'animate'
                } for f in frames
            ],
            'active': 0,
            'currentvalue': {'prefix': 'Year: '}
        }],
        height=600,
        template='plotly_white'
    )
    
    return fig

def create_animated_bubble_chart(data: pd.DataFrame, country_col: str, colors: List[str]) -> go.Figure:
    """Create an animated bubble chart"""
    # Calculate market share and growth rate for bubble chart
    yearly_totals = data.groupby('Year')['Value'].sum()
    
    bubble_data = []
    years = sorted(data['Year'].unique())
    
    for year in years:
        year_data = data[data['Year'] == year].copy()
        year_total = yearly_totals[year]
        year_data['market_share'] = year_data['Value'] / year_total * 100
        
        # Calculate growth rate if possible
        if year > years[0]:
            prev_year_data = data[data['Year'] == year - 1]
            merged = pd.merge(year_data, prev_year_data, on=country_col, suffixes=('', '_prev'))
            merged['growth_rate'] = ((merged['Value'] - merged['Value_prev']) / merged['Value_prev'] * 100).fillna(0)
            year_data = merged[[country_col, 'Value', 'market_share', 'growth_rate']].copy()
        else:
            year_data['growth_rate'] = 0
        
        year_data['Year'] = year
        bubble_data.append(year_data)
    
    bubble_df = pd.concat(bubble_data, ignore_index=True)
    
    # Create animated bubble chart
    fig = px.scatter(
        bubble_df,
        x='market_share',
        y='growth_rate',
        size='Value',
        color=country_col,
        animation_frame='Year',
        hover_name=country_col,
        title='Market Dynamics - Size vs Growth vs Market Share',
        labels={
            'market_share': 'Market Share (%)',
            'growth_rate': 'Growth Rate (%)',
            'Value': 'Market Size'
        },
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white',
        xaxis_title='Market Share (%)',
        yaxis_title='Growth Rate (%)'
    )
    
    return fig

def create_waterfall_chart(data: pd.DataFrame, country_col: str, colors: List[str]) -> go.Figure:
    """Create a waterfall chart for growth analysis"""
    # Calculate year-over-year changes
    years = sorted(data['Year'].unique())
    if len(years) < 2:
        # Not enough data for waterfall
        return create_multi_line_chart(data, country_col, colors)
    
    # Get total market for first and last year
    first_year_total = data[data['Year'] == years[0]]['Value'].sum()
    last_year_total = data[data['Year'] == years[-1]]['Value'].sum()
    
    # Calculate contributions by country
    first_year_data = data[data['Year'] == years[0]]
    last_year_data = data[data['Year'] == years[-1]]
    
    # Get top contributors
    top_countries = last_year_data.nlargest(10, 'Value')[country_col].tolist()
    
    contributions = []
    cumulative = first_year_total
    
    for country in top_countries:
        first_value = first_year_data[first_year_data[country_col] == country]['Value'].sum()
        last_value = last_year_data[last_year_data[country_col] == country]['Value'].sum()
        contribution = last_value - first_value
        contributions.append((country, contribution))
        cumulative += contribution
    
    # Prepare waterfall data
    x_labels = [f'{years[0]} Total'] + [f'{country}\nContribution' for country, _ in contributions] + [f'{years[-1]} Total']
    y_values = [first_year_total] + [contrib for _, contrib in contributions] + [last_year_total]
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Market Growth Analysis",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(contributions) + ["total"],
        x=x_labels,
        y=y_values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#E76F51"}},
        increasing={"marker": {"color": "#2A9D8F"}},
        totals={"marker": {"color": "#264653"}}
    ))
    
    fig.update_layout(
        title={
            'text': f'Market Growth Waterfall Analysis ({years[0]} - {years[-1]})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        yaxis_title='Market Size (USD)',
        template='plotly_white',
        height=600
    )
    
    return fig

def render_interactive_world_map(distributed_market: pd.DataFrame, config_manager) -> None:
    """
    Render interactive world map visualization
    """
    st.title("🌍 Interactive World Map")
    st.markdown("*Explore global market distribution with interactive country selection*")
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        country_col = country_mapping.get('country_column', 'Country')
    except:
        country_col = 'Country'
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        years = sorted(distributed_market['Year'].unique())
        selected_year = st.selectbox("Select Year", years, index=len(years)-1, key="map_year")
    
    with col2:
        color_scale = st.selectbox(
            "Color Scheme", 
            ["Viridis", "Blues", "Reds", "Greens", "Plasma", "Turbo"],
            key="map_colorscale"
        )
    
    # Create choropleth map
    year_data = distributed_market[distributed_market['Year'] == selected_year]
    
    # Create map
    fig = go.Figure(data=go.Choropleth(
        locations=year_data[country_col],
        z=year_data['Value'],
        locationmode='country names',
        colorscale=color_scale,
        autocolorscale=False,
        text=year_data[country_col],
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title="Market Value (USD)",
        hovertemplate='<b>%{text}</b><br>' +
                     'Market Value: $%{z:,.0f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'Global Market Distribution ({selected_year})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="world_map_visualization")
    
    # Country comparison section
    st.markdown("### 🔍 Country Deep Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Country selection
        countries = sorted(distributed_market[country_col].unique())
        selected_countries = st.multiselect(
            "Select Countries to Compare",
            countries,
            default=year_data.nlargest(3, 'Value')[country_col].tolist(),
            help="Choose countries for detailed comparison",
            key="world_map_country_selection"
        )
    
    with col2:
        comparison_metric = st.selectbox(
            "Comparison Metric",
            ["Market Size", "Market Share", "Growth Rate", "CAGR"],
            help="Choose what to compare between countries",
            key="world_map_comparison_metric"
        )
    
    if selected_countries:
        render_country_comparison(distributed_market, selected_countries, comparison_metric, country_col, context="world_map")

def render_country_comparison(data: pd.DataFrame, countries: List[str], metric: str, country_col: str, context: str = "default") -> None:
    """
    Render detailed country comparison
    
    Args:
        data: DataFrame with market data
        countries: List of countries to compare
        metric: Metric to compare
        country_col: Name of country column
        context: Context identifier for unique keys
    """
    # Filter data for selected countries
    comparison_data = data[data[country_col].isin(countries)]
    
    if comparison_data.empty:
        st.warning("No data available for selected countries.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time series comparison
        if metric == "Market Size":
            fig = px.line(
                comparison_data,
                x='Year',
                y='Value',
                color=country_col,
                title=f'{metric} Comparison Over Time',
                markers=True,
                color_discrete_sequence=COLOR_SCHEMES['professional']
            )
            fig.update_layout(
                yaxis_title='Market Size (USD)',
                template='plotly_white',
                height=400
            )
            
        elif metric == "Market Share":
            # Calculate market share
            yearly_totals = data.groupby('Year')['Value'].sum()
            comparison_data = comparison_data.copy()
            comparison_data['Market_Share'] = comparison_data.apply(
                lambda row: (row['Value'] / yearly_totals[row['Year']]) * 100, axis=1
            )
            
            fig = px.line(
                comparison_data,
                x='Year',
                y='Market_Share',
                color=country_col,
                title=f'{metric} Comparison Over Time',
                markers=True,
                color_discrete_sequence=COLOR_SCHEMES['professional']
            )
            fig.update_layout(
                yaxis_title='Market Share (%)',
                template='plotly_white',
                height=400
            )
            
        elif metric == "Growth Rate":
            # Calculate year-over-year growth
            growth_data = []
            for country in countries:
                country_data = comparison_data[comparison_data[country_col] == country].sort_values('Year')
                country_data['Growth_Rate'] = country_data['Value'].pct_change() * 100
                growth_data.append(country_data.dropna())
            
            if growth_data:
                growth_df = pd.concat(growth_data)
                
                fig = px.line(
                    growth_df,
                    x='Year',
                    y='Growth_Rate',
                    color=country_col,
                    title=f'{metric} Comparison Over Time',
                    markers=True,
                    color_discrete_sequence=COLOR_SCHEMES['professional']
                )
                fig.update_layout(
                    yaxis_title='Growth Rate (%)',
                    template='plotly_white',
                    height=400
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            else:
                fig = go.Figure()
                fig.add_annotation(text="No growth data available", x=0.5, y=0.5, showarrow=False)
        
        else:  # CAGR
            # Calculate CAGR for each country
            cagr_data = []
            years = sorted(comparison_data['Year'].unique())
            
            for country in countries:
                country_data = comparison_data[comparison_data[country_col] == country].sort_values('Year')
                if len(country_data) > 1:
                    first_value = country_data.iloc[0]['Value']
                    last_value = country_data.iloc[-1]['Value']
                    years_diff = len(country_data) - 1
                    
                    if first_value > 0 and years_diff > 0:
                        cagr = ((last_value / first_value) ** (1 / years_diff) - 1) * 100
                        cagr_data.append({country_col: country, 'CAGR': cagr})
            
            if cagr_data:
                cagr_df = pd.DataFrame(cagr_data)
                
                fig = px.bar(
                    cagr_df,
                    x=country_col,
                    y='CAGR',
                    title=f'{metric} Comparison',
                    color='CAGR',
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0
                )
                fig.update_layout(
                    yaxis_title='CAGR (%)',
                    template='plotly_white',
                    height=400
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            else:
                fig = go.Figure()
                fig.add_annotation(text="No CAGR data available", x=0.5, y=0.5, showarrow=False)
        
        st.plotly_chart(fig, use_container_width=True, key=f"country_comparison_{context}_{metric}_chart")
    
    with col2:
        # Country metrics table
        st.markdown("#### 📊 Country Metrics")
        
        metrics_data = []
        latest_year = data['Year'].max()
        earliest_year = data['Year'].min()
        
        for country in countries:
            country_data = comparison_data[comparison_data[country_col] == country]
            
            if not country_data.empty:
                latest_value = country_data[country_data['Year'] == latest_year]['Value']
                latest_value = latest_value.iloc[0] if not latest_value.empty else 0
                
                # Calculate rank correctly
                latest_data_all = data[data['Year'] == latest_year].sort_values('Value', ascending=False).reset_index(drop=True)
                if country in latest_data_all[country_col].values:
                    # Find the position of the country in the sorted data
                    rank = latest_data_all[latest_data_all[country_col] == country].index[0] + 1
                else:
                    rank = "N/A"
                
                # Calculate market share
                total_market = data[data['Year'] == latest_year]['Value'].sum()
                market_share = (latest_value / total_market) * 100 if total_market > 0 else 0
                
                # Calculate CAGR
                if len(country_data) > 1:
                    first_value = country_data.iloc[0]['Value']
                    years_diff = len(country_data) - 1
                    if first_value > 0 and years_diff > 0:
                        cagr = ((latest_value / first_value) ** (1 / years_diff) - 1) * 100
                    else:
                        cagr = 0
                else:
                    cagr = 0
                
                # Format market value properly
                value_display = format_value(latest_value)
                
                metrics_data.append({
                    'Country': country,
                    'Market Value': value_display,
                    'Global Rank': rank,
                    'Market Share': f"{market_share:.1f}%",
                    'CAGR': f"{cagr:.1f}%"
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Additional insights
        st.markdown("#### 💡 Key Insights")
        
        if len(countries) >= 2:
            # Find best and worst performers
            latest_values = {}
            for country in countries:
                country_data = comparison_data[comparison_data[country_col] == country]
                if not country_data.empty:
                    latest_value = country_data[country_data['Year'] == latest_year]['Value']
                    latest_values[country] = latest_value.iloc[0] if not latest_value.empty else 0
            
            if latest_values:
                best_performer = max(latest_values.keys(), key=lambda k: latest_values[k])
                best_value = latest_values[best_performer]
                
                # Format the best value properly
                best_value_display = format_value(best_value)
                
                create_insight_box(
                    f"🏆 <b>Top Performer:</b> {best_performer} leads with "
                    f"{best_value_display} market value among selected countries.",
                    "success"
                )
                
                # Growth leader
                growth_leader = None
                max_cagr = -float('inf')
                
                for country in countries:
                    country_data = comparison_data[comparison_data[country_col] == country]
                    if len(country_data) > 1:
                        first_value = country_data.iloc[0]['Value']
                        last_value = country_data.iloc[-1]['Value']
                        years_diff = len(country_data) - 1
                        
                        if first_value > 0 and years_diff > 0:
                            cagr = ((last_value / first_value) ** (1 / years_diff) - 1) * 100
                            if cagr > max_cagr:
                                max_cagr = cagr
                                growth_leader = country
                
                if growth_leader:
                    create_insight_box(
                        f"📈 <b>Growth Leader:</b> {growth_leader} shows the highest growth rate "
                        f"at {max_cagr:.1f}% CAGR among selected countries.",
                        "info"
                    )

def render_enhanced_visualization_interface(config_manager) -> None:
    """
    Main enhanced visualization interface with all new features
    """
    set_page_style()
    
    st.title("🚀 Enhanced Market Visualizations")
    st.markdown("*Professional, interactive visualizations with advanced analytics*")
    
    # Check if we have data
    if 'distributed_market' not in st.session_state or st.session_state.distributed_market is None:
        st.warning("⚠️ No market data available. Please run market distribution first.")
        
        if st.button("🔄 Go to Market Distribution", type="primary"):
            st.session_state.active_page = "Market Distribution"
            st.rerun()
        return
    
    # Get data
    distributed_market = st.session_state.distributed_market
    
    # Create navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Dashboard",
        "📈 Enhanced Market Analysis", 
        "🌍 Interactive World Map",
        "🔍 Country Comparison",
        "📅 YoY Change Analysis"
    ])
    
    with tab1:
        render_executive_dashboard(distributed_market, config_manager)
    
    with tab2:
        render_enhanced_market_size_visualization(distributed_market, config_manager)
    
    with tab3:
        render_interactive_world_map(distributed_market, config_manager)
    
    with tab4:
        # Get column mappings
        try:
            country_mapping = config_manager.get_column_mapping('country_historical')
            country_col = country_mapping.get('country_column', 'Country')
        except:
            country_col = 'Country'
        
        st.title("🔍 Advanced Country Comparison")
        st.markdown("*Compare countries across multiple dimensions with detailed analytics*")
        
        # Country selection
        countries = sorted(distributed_market[country_col].unique())
        selected_countries = st.multiselect(
            "Select Countries to Compare",
            countries,
            default=distributed_market[distributed_market['Year'] == distributed_market['Year'].max()].nlargest(3, 'Value')[country_col].tolist(),
            help="Choose countries for detailed comparison",
            key="country_comparison_selection"
        )
        
        if selected_countries:
            comparison_metric = st.selectbox(
                "Primary Comparison Metric",
                ["Market Size", "Market Share", "Growth Rate", "CAGR"],
                help="Choose the main metric for comparison",
                key="country_comparison_metric"
            )
            
            render_country_comparison(distributed_market, selected_countries, comparison_metric, country_col, context="country_analysis")
        else:
            st.info("👆 Please select at least one country to begin comparison analysis.")
    
    with tab5:
        # Render YoY Change Analysis
        try:
            yoy_visualizer = YoYChangeVisualizer(distributed_market)
            yoy_visualizer.render_visualization()
        except Exception as e:
            logger.error(f"Error rendering YoY visualization: {str(e)}")
            st.error(f"Error loading YoY Change Analysis: {str(e)}")