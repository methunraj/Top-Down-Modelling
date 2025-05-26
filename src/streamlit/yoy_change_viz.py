"""
Year-over-Year Change Visualization Module

This module provides interactive visualizations for comparing YoY changes
across multiple countries with support for single and multi-country selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Tuple
import logging
from src.utils.math_utils import calculate_growth_rate, safe_divide

logger = logging.getLogger(__name__)


class YoYChangeVisualizer:
    """
    Provides interactive year-over-year change visualizations for market data
    """
    
    def __init__(self, market_data: pd.DataFrame):
        """
        Initialize the YoY Change Visualizer
        
        Args:
            market_data: DataFrame containing market data with columns
                        [idGeo, Country, Year, Value, market_share]
        """
        self.market_data = market_data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for YoY analysis"""
        # Ensure required columns exist
        required_cols = ['Country', 'Year', 'Value']
        missing_cols = [col for col in required_cols if col not in self.market_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure Year column is numeric
        self.market_data['Year'] = pd.to_numeric(self.market_data['Year'], errors='coerce')
        self.market_data = self.market_data.dropna(subset=['Year'])
        
        # Ensure data is sorted by country and year
        self.market_data = self.market_data.sort_values(['Country', 'Year'])
        
        # Calculate YoY changes for each country with safe calculation
        self.market_data['value_yoy_change'] = self.market_data.groupby('Country')['Value'].pct_change() * 100
        
        # Only calculate share YoY if market_share column exists
        if 'market_share' in self.market_data.columns:
            self.market_data['share_yoy_change'] = self.market_data.groupby('Country')['market_share'].pct_change() * 100
            self.market_data['share_abs_change'] = self.market_data.groupby('Country')['market_share'].diff()
        else:
            # Create dummy columns to avoid errors
            self.market_data['share_yoy_change'] = np.nan
            self.market_data['share_abs_change'] = np.nan
        
        # Calculate absolute changes as well
        self.market_data['value_abs_change'] = self.market_data.groupby('Country')['Value'].diff()
        
        # Only calculate share absolute change if market_share column exists
        if 'market_share' in self.market_data.columns:
            self.market_data['share_abs_change'] = self.market_data.groupby('Country')['market_share'].diff()
        else:
            self.market_data['share_abs_change'] = np.nan
        
        # Get list of available countries
        self.countries = sorted(self.market_data['Country'].unique())
        self.years = sorted(self.market_data['Year'].unique())
    
    def render_visualization(self):
        """Render the YoY change visualization interface"""
        st.header("Year-over-Year Change Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Single Country Analysis", 
            "Multi-Country Comparison", 
            "Ranking Analysis",
            "Growth Patterns"
        ])
        
        with tab1:
            self._render_single_country_analysis()
        
        with tab2:
            self._render_multi_country_comparison()
            
        with tab3:
            self._render_ranking_analysis()
            
        with tab4:
            self._render_growth_patterns()
    
    def _render_single_country_analysis(self):
        """Render single country YoY analysis"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Country selection
            selected_country = st.selectbox(
                "Select Country",
                self.countries,
                key="single_country_select"
            )
            
            # Metric selection
            metric_type = st.radio(
                "Metric",
                ["Market Value", "Market Share"],
                key="single_metric"
            )
            
            # Display type
            display_type = st.radio(
                "Display Type",
                ["Percentage Change", "Absolute Change"],
                key="single_display"
            )
        
        # Filter data for selected country
        country_data = self.market_data[self.market_data['Country'] == selected_country].copy()
        
        with col2:
            # Create visualization
            fig = go.Figure()
            
            if metric_type == "Market Value":
                if display_type == "Percentage Change":
                    y_values = country_data['value_yoy_change']
                    y_title = "YoY % Change in Market Value"
                else:
                    y_values = country_data['value_abs_change']
                    y_title = "YoY Absolute Change in Market Value"
            else:
                if display_type == "Percentage Change":
                    y_values = country_data['share_yoy_change']
                    y_title = "YoY % Change in Market Share"
                else:
                    y_values = country_data['share_abs_change']
                    y_title = "YoY Absolute Change in Market Share (percentage points)"
            
            # Add bar chart
            colors = ['green' if x > 0 else 'red' for x in y_values.fillna(0)]
            
            fig.add_trace(go.Bar(
                x=country_data['Year'],
                y=y_values,
                marker_color=colors,
                text=[f"{v:.1f}%" if display_type == "Percentage Change" else f"{v:,.0f}" 
                      for v in y_values.fillna(0)],
                textposition='outside',
                name=selected_country
            ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=f"{selected_country}: {y_title}",
                xaxis_title="Year",
                yaxis_title=y_title,
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed statistics
        with st.expander("Detailed Statistics"):
            stats_data = []
            for _, row in country_data.iterrows():
                stats_data.append({
                    'Year': int(row['Year']),
                    'Market Value': f"{row['Value']:,.0f}",
                    'Value YoY %': f"{row['value_yoy_change']:.1f}%" if pd.notna(row['value_yoy_change']) else "N/A",
                    'Market Share': f"{row['market_share']:.2f}%",
                    'Share YoY %': f"{row['share_yoy_change']:.1f}%" if pd.notna(row['share_yoy_change']) else "N/A"
                })
            
            st.dataframe(pd.DataFrame(stats_data), hide_index=True)
    
    def _render_multi_country_comparison(self):
        """Render multi-country YoY comparison"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Country selection (multiple)
            selected_countries = st.multiselect(
                "Select Countries (max 10)",
                self.countries,
                default=self.countries[:5] if len(self.countries) >= 5 else self.countries,
                max_selections=10,
                key="multi_country_select"
            )
            
            if not selected_countries:
                st.warning("Please select at least one country")
                return
            
            # Metric selection
            metric_type = st.radio(
                "Metric",
                ["Market Value", "Market Share"],
                key="multi_metric"
            )
            
            # Chart type
            chart_type = st.radio(
                "Chart Type",
                ["Line Chart", "Grouped Bar Chart", "Heatmap"],
                key="multi_chart"
            )
        
        with col2:
            # Filter data for selected countries
            filtered_data = self.market_data[self.market_data['Country'].isin(selected_countries)].copy()
            
            if chart_type == "Line Chart":
                fig = go.Figure()
                
                for country in selected_countries:
                    country_data = filtered_data[filtered_data['Country'] == country].copy()
                    # Ensure data is sorted by year for proper line connection
                    country_data = country_data.sort_values('Year')
                    
                    if metric_type == "Market Value":
                        y_values = country_data['value_yoy_change']
                        y_title = "YoY % Change in Market Value"
                    else:
                        y_values = country_data['share_yoy_change']
                        y_title = "YoY % Change in Market Share"
                    
                    fig.add_trace(go.Scatter(
                        x=country_data['Year'],
                        y=y_values,
                        mode='lines+markers',
                        name=country,
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title=f"YoY Change Comparison: {metric_type}",
                    xaxis_title="Year",
                    yaxis_title=y_title,
                    height=500,
                    hovermode='x unified'
                )
                
            elif chart_type == "Grouped Bar Chart":
                # Prepare data for grouped bar chart
                if metric_type == "Market Value":
                    value_col = 'value_yoy_change'
                    y_title = "YoY % Change in Market Value"
                else:
                    value_col = 'share_yoy_change'
                    y_title = "YoY % Change in Market Share"
                
                # Pivot data for grouped bars
                pivot_data = filtered_data.pivot(
                    index='Year',
                    columns='Country',
                    values=value_col
                )
                
                fig = go.Figure()
                
                for country in selected_countries:
                    if country in pivot_data.columns:
                        fig.add_trace(go.Bar(
                            x=pivot_data.index,
                            y=pivot_data[country],
                            name=country,
                            text=[f"{v:.1f}%" if pd.notna(v) else "" for v in pivot_data[country]],
                            textposition='outside'
                        ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title=f"YoY Change Comparison: {metric_type}",
                    xaxis_title="Year",
                    yaxis_title=y_title,
                    barmode='group',
                    height=500
                )
                
            else:  # Heatmap
                # Prepare data for heatmap
                if metric_type == "Market Value":
                    value_col = 'value_yoy_change'
                    title = "YoY % Change in Market Value"
                else:
                    value_col = 'share_yoy_change'
                    title = "YoY % Change in Market Share"
                
                # Pivot data
                heatmap_data = filtered_data.pivot(
                    index='Country',
                    columns='Year',
                    values=value_col
                )
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=[str(year) for year in heatmap_data.columns],
                    y=heatmap_data.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] 
                          for row in heatmap_data.values],
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="YoY %")
                ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title="Year",
                    yaxis_title="Country",
                    height=max(400, len(selected_countries) * 40)
                )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_ranking_analysis(self):
        """Render YoY growth ranking analysis"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Year selection
            selected_year = st.selectbox(
                "Select Year",
                [year for year in self.years if year > self.years[0]],
                key="ranking_year"
            )
            
            # Metric selection
            metric_type = st.radio(
                "Metric",
                ["Market Value", "Market Share"],
                key="ranking_metric"
            )
            
            # Top/Bottom selection
            ranking_type = st.radio(
                "Show",
                ["Top 10 Gainers", "Bottom 10 Losers", "All Countries"],
                key="ranking_type"
            )
        
        with col2:
            # Filter data for selected year
            year_data = self.market_data[self.market_data['Year'] == selected_year].copy()
            
            if metric_type == "Market Value":
                value_col = 'value_yoy_change'
                title_suffix = "Market Value YoY Change"
            else:
                value_col = 'share_yoy_change'
                title_suffix = "Market Share YoY Change"
            
            # Sort and filter based on ranking type
            year_data = year_data.dropna(subset=[value_col])
            year_data = year_data.sort_values(value_col, ascending=False)
            
            if ranking_type == "Top 10 Gainers":
                plot_data = year_data.head(10)
                title = f"Top 10 Countries by {title_suffix} ({selected_year})"
            elif ranking_type == "Bottom 10 Losers":
                plot_data = year_data.tail(10)
                title = f"Bottom 10 Countries by {title_suffix} ({selected_year})"
            else:
                plot_data = year_data
                title = f"All Countries by {title_suffix} ({selected_year})"
            
            # Create horizontal bar chart
            colors = ['green' if x > 0 else 'red' for x in plot_data[value_col]]
            
            fig = go.Figure(go.Bar(
                x=plot_data[value_col],
                y=plot_data['Country'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:.1f}%" for v in plot_data[value_col]],
                textposition='outside'
            ))
            
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title=title,
                xaxis_title="YoY % Change",
                yaxis_title="Country",
                height=max(400, len(plot_data) * 30),
                yaxis={'categoryorder': 'total ascending' if ranking_type == "Bottom 10 Losers" else 'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_growth_patterns(self):
        """Render growth pattern analysis"""
        st.subheader("Growth Pattern Analysis")
        
        # Calculate growth statistics for each country
        growth_stats = []
        
        for country in self.countries:
            country_data = self.market_data[self.market_data['Country'] == country]
            
            # Calculate average YoY growth (excluding NaN values)
            avg_value_growth = country_data['value_yoy_change'].mean(skipna=True)
            avg_share_growth = country_data['share_yoy_change'].mean(skipna=True)
            
            # Calculate volatility (std dev)
            value_volatility = country_data['value_yoy_change'].std(skipna=True)
            share_volatility = country_data['share_yoy_change'].std(skipna=True)
            
            # Count positive growth years (excluding NaN)
            valid_value_data = country_data['value_yoy_change'].dropna()
            valid_share_data = country_data['share_yoy_change'].dropna()
            
            positive_value_years = (valid_value_data > 0).sum()
            positive_share_years = (valid_share_data > 0).sum()
            
            # Use valid data length for calculation
            total_value_years = len(valid_value_data)
            total_share_years = len(valid_share_data)
            
            if total_value_years > 0:
                growth_stats.append({
                    'Country': country,
                    'Avg Value Growth': avg_value_growth if pd.notna(avg_value_growth) else 0.0,
                    'Value Volatility': value_volatility if pd.notna(value_volatility) else 0.0,
                    'Positive Value Years': (positive_value_years / total_value_years * 100) if total_value_years > 0 else 0.0,
                    'Avg Share Growth': avg_share_growth if pd.notna(avg_share_growth) else 0.0,
                    'Share Volatility': share_volatility if pd.notna(share_volatility) else 0.0,
                    'Positive Share Years': (positive_share_years / total_share_years * 100) if total_share_years > 0 else 0.0
                })
        
        growth_df = pd.DataFrame(growth_stats)
        
        # Check if we have valid data
        if growth_df.empty:
            st.warning("No growth data available for analysis.")
            return
        
        # Remove any rows with missing data for plotting
        growth_df_clean = growth_df.dropna(subset=['Avg Value Growth', 'Value Volatility', 
                                                   'Avg Share Growth', 'Share Volatility'])
        
        if growth_df_clean.empty:
            st.warning("Insufficient data for growth pattern analysis.")
            return
        
        # Create scatter plot: Average Growth vs Volatility
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.scatter(
                growth_df_clean,
                x='Avg Value Growth',
                y='Value Volatility',
                hover_data=['Country'],
                title='Market Value: Growth vs Volatility',
                labels={
                    'Avg Value Growth': 'Average YoY Growth (%)',
                    'Value Volatility': 'Growth Volatility (Std Dev %)'
                }
            )
            
            # Add quadrant lines
            if len(growth_df_clean) > 0:
                fig1.add_hline(y=growth_df_clean['Value Volatility'].median(), line_dash="dash", line_color="gray", opacity=0.5)
            fig1.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Annotate quadrants
            fig1.add_annotation(x=15, y=30, text="High Growth<br>High Volatility", showarrow=False, opacity=0.5)
            fig1.add_annotation(x=15, y=5, text="High Growth<br>Low Volatility", showarrow=False, opacity=0.5)
            fig1.add_annotation(x=-15, y=30, text="Low Growth<br>High Volatility", showarrow=False, opacity=0.5)
            fig1.add_annotation(x=-15, y=5, text="Low Growth<br>Low Volatility", showarrow=False, opacity=0.5)
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.scatter(
                growth_df_clean,
                x='Avg Share Growth',
                y='Share Volatility',
                hover_data=['Country'],
                title='Market Share: Growth vs Volatility',
                labels={
                    'Avg Share Growth': 'Average YoY Growth (%)',
                    'Share Volatility': 'Growth Volatility (Std Dev %)'
                }
            )
            
            # Add quadrant lines
            if len(growth_df_clean) > 0:
                fig2.add_hline(y=growth_df_clean['Share Volatility'].median(), line_dash="dash", line_color="gray", opacity=0.5)
            fig2.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Show growth consistency table
        st.subheader("Growth Consistency Analysis")
        
        # Prepare data for display
        consistency_df = growth_df_clean.copy()
        # Handle NaN values in string formatting
        consistency_df['Value Growth Score'] = consistency_df['Avg Value Growth'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        consistency_df['Share Growth Score'] = consistency_df['Avg Share Growth'].apply(
            lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
        )
        consistency_df['Value Consistency'] = consistency_df['Positive Value Years'].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"
        )
        consistency_df['Share Consistency'] = consistency_df['Positive Share Years'].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"
        )
        
        # Select and sort columns for display
        display_df = consistency_df[[
            'Country', 
            'Value Growth Score', 
            'Value Consistency',
            'Share Growth Score',
            'Share Consistency'
        ]].copy()
        
        # Sort by the original column before renaming
        if len(consistency_df) > 0:
            display_df = display_df.loc[consistency_df.sort_values('Avg Value Growth', ascending=False).index]
        else:
            st.warning("No data available for consistency analysis.")
        
        st.dataframe(display_df, hide_index=True, height=400)
        
        # Export functionality
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Growth Analysis as CSV",
            data=csv,
            file_name="yoy_growth_analysis.csv",
            mime="text/csv"
        )