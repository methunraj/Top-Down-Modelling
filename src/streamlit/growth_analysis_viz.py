"""
Advanced Growth Analysis Visualizations

This module provides sophisticated growth analysis visualizations including
CAGR analysis, growth trajectories, volatility analysis, and growth forecasting.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Professional color schemes
COLOR_SCHEMES = {
    'growth': ['#27AE60', '#E74C3C', '#3498DB', '#F39C12', '#9B59B6', '#1ABC9C'],
    'performance': ['#2ECC71', '#E67E22', '#3498DB', '#E74C3C', '#9B59B6', '#F1C40F'],
    'trend': ['#16A085', '#E74C3C', '#2980B9', '#F39C12', '#8E44AD', '#95A5A6']
}

def render_growth_analysis_dashboard(distributed_market: pd.DataFrame, config_manager) -> None:
    """
    Render comprehensive growth analysis dashboard
    """
    st.title("📈 Advanced Growth Analysis Dashboard")
    
    # Get column mappings
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        country_col = country_mapping.get('country_column', 'Country')
    except:
        country_col = 'Country'
    
    # Calculate growth metrics
    growth_data = calculate_comprehensive_growth_metrics(distributed_market, country_col)
    
    if growth_data.empty:
        st.warning("Insufficient data for growth analysis. Need at least 2 years of data.")
        return
    
    # Growth analysis controls
    st.markdown("### 🎛️ Analysis Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["CAGR Analysis", "YoY Growth", "Volatility Analysis", "Growth Trajectory", "Growth Forecasting"],
            help="Select the type of growth analysis"
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period",
            ["All Years", "Last 3 Years", "Last 5 Years", "Custom Period"],
            help="Choose analysis time period"
        )
    
    with col3:
        countries_scope = st.selectbox(
            "Country Scope",
            ["Top 10", "Top 15", "All Countries", "Custom Selection"],
            help="Select countries to analyze"
        )
    
    with col4:
        comparison_basis = st.selectbox(
            "Comparison Basis",
            ["Market Size", "Growth Rate", "Market Share", "Volatility"],
            help="Choose comparison metric"
        )
    
    # Filter data based on selections
    filtered_growth_data = filter_growth_data(
        growth_data, time_period, countries_scope, country_col, distributed_market
    )
    
    # Render analysis based on type
    if analysis_type == "CAGR Analysis":
        render_cagr_analysis(filtered_growth_data, country_col)
    elif analysis_type == "YoY Growth":
        render_yoy_growth_analysis(filtered_growth_data, country_col)
    elif analysis_type == "Volatility Analysis":
        render_volatility_analysis(filtered_growth_data, country_col)
    elif analysis_type == "Growth Trajectory":
        render_growth_trajectory_analysis(filtered_growth_data, country_col)
    else:  # Growth Forecasting
        render_growth_forecasting(filtered_growth_data, country_col)

def calculate_comprehensive_growth_metrics(data: pd.DataFrame, country_col: str) -> pd.DataFrame:
    """
    Calculate comprehensive growth metrics for all countries
    """
    growth_metrics = []
    
    countries = data[country_col].unique()
    years = sorted(data['Year'].unique())
    
    if len(years) < 2:
        return pd.DataFrame()
    
    for country in countries:
        country_data = data[data[country_col] == country].sort_values('Year')
        
        if len(country_data) < 2:
            continue
        
        # Calculate various growth metrics
        values = country_data['Value'].values
        years_country = country_data['Year'].values
        
        # CAGR
        if values[0] > 0 and values[-1] > 0:
            years_diff = years_country[-1] - years_country[0]
            if years_diff > 0:
                cagr = ((values[-1] / values[0]) ** (1 / years_diff) - 1) * 100
            else:
                cagr = 0
        else:
            cagr = 0
        
        # Year-over-year growth rates
        yoy_rates = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                yoy_rate = ((values[i] / values[i-1]) - 1) * 100
                yoy_rates.append(yoy_rate)
        
        # Volatility (standard deviation of YoY rates)
        volatility = np.std(yoy_rates) if yoy_rates else 0
        
        # Average growth rate
        avg_growth = np.mean(yoy_rates) if yoy_rates else 0
        
        # Growth acceleration (trend in growth rates)
        if len(yoy_rates) > 2:
            x = np.arange(len(yoy_rates)).reshape(-1, 1)
            reg = LinearRegression().fit(x, yoy_rates)
            acceleration = reg.coef_[0]
        else:
            acceleration = 0
        
        # Market share trajectory
        total_market_by_year = data.groupby('Year')['Value'].sum()
        market_shares = []
        for _, row in country_data.iterrows():
            year_total = total_market_by_year[row['Year']]
            share = (row['Value'] / year_total) * 100 if year_total > 0 else 0
            market_shares.append(share)
        
        # Market share growth
        if len(market_shares) > 1 and years_diff > 0:
            share_growth = ((market_shares[-1] / market_shares[0]) ** (1 / years_diff) - 1) * 100 if market_shares[0] > 0 else 0
        else:
            share_growth = 0
        
        growth_metrics.append({
            country_col: country,
            'CAGR': cagr,
            'Avg_YoY_Growth': avg_growth,
            'Volatility': volatility,
            'Growth_Acceleration': acceleration,
            'Latest_Value': values[-1],
            'Initial_Value': values[0],
            'Market_Share_Growth': share_growth,
            'Latest_Market_Share': market_shares[-1] if market_shares else 0,
            'Years_Analyzed': len(country_data),
            'YoY_Rates': yoy_rates,
            'Market_Shares': market_shares,
            'Years': years_country.tolist(),
            'Values': values.tolist()
        })
    
    return pd.DataFrame(growth_metrics)

def filter_growth_data(growth_data: pd.DataFrame, time_period: str, countries_scope: str, 
                      country_col: str, original_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter growth data based on user selections
    """
    filtered_data = growth_data.copy()
    
    # Time period filtering would be applied during recalculation if needed
    # For now, we'll work with the full growth data
    
    # Country scope filtering
    if countries_scope == "Top 10":
        filtered_data = filtered_data.nlargest(10, 'Latest_Value')
    elif countries_scope == "Top 15":
        filtered_data = filtered_data.nlargest(15, 'Latest_Value')
    elif countries_scope == "Custom Selection":
        available_countries = filtered_data[country_col].tolist()
        selected_countries = st.multiselect(
            "Select Countries for Analysis",
            available_countries,
            default=filtered_data.nlargest(10, 'Latest_Value')[country_col].tolist(),
            help="Choose specific countries to analyze"
        )
        filtered_data = filtered_data[filtered_data[country_col].isin(selected_countries)]
    
    return filtered_data

def render_cagr_analysis(growth_data: pd.DataFrame, country_col: str) -> None:
    """
    Render CAGR analysis with multiple visualizations
    """
    st.markdown("### 📊 Compound Annual Growth Rate (CAGR) Analysis")
    
    # CAGR distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # CAGR bar chart
        fig = px.bar(
            growth_data.sort_values('CAGR', ascending=True),
            x='CAGR',
            y=country_col,
            orientation='h',
            color='CAGR',
            color_continuous_scale='RdYlGn',
            title='CAGR by Country',
            labels={'CAGR': 'CAGR (%)'}
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="No Growth")
        fig.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # CAGR vs Market Size scatter
        fig = px.scatter(
            growth_data,
            x='Latest_Value',
            y='CAGR',
            size='Latest_Value',
            color='Volatility',
            hover_name=country_col,
            title='CAGR vs Market Size',
            labels={
                'Latest_Value': 'Market Size (USD)',
                'CAGR': 'CAGR (%)',
                'Volatility': 'Volatility'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=500, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # CAGR insights
    st.markdown("### 💡 CAGR Insights")
    
    # Top and bottom performers
    top_growth = growth_data.nlargest(3, 'CAGR')
    bottom_growth = growth_data.nsmallest(3, 'CAGR')
    avg_cagr = growth_data['CAGR'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🚀 Top Growth Markets")
        for _, row in top_growth.iterrows():
            st.markdown(f"**{row[country_col]}**: {row['CAGR']:.1f}%")
    
    with col2:
        st.markdown("#### 📉 Challenging Markets")
        for _, row in bottom_growth.iterrows():
            st.markdown(f"**{row[country_col]}**: {row['CAGR']:.1f}%")
    
    with col3:
        st.markdown("#### 📈 Market Overview")
        st.markdown(f"**Average CAGR**: {avg_cagr:.1f}%")
        st.markdown(f"**High Growth (>10%)**: {len(growth_data[growth_data['CAGR'] > 10])} countries")
        st.markdown(f"**Declining (<0%)**: {len(growth_data[growth_data['CAGR'] < 0])} countries")

def render_yoy_growth_analysis(growth_data: pd.DataFrame, country_col: str) -> None:
    """
    Render year-over-year growth analysis
    """
    st.markdown("### 📈 Year-over-Year Growth Analysis")
    
    # YoY growth heatmap
    countries_with_yoy = growth_data[growth_data['YoY_Rates'].apply(len) > 0]
    
    if countries_with_yoy.empty:
        st.warning("No year-over-year data available for analysis.")
        return
    
    # Create heatmap data
    max_years = max(len(rates) for rates in countries_with_yoy['YoY_Rates'])
    heatmap_data = []
    
    for _, row in countries_with_yoy.iterrows():
        rates = row['YoY_Rates']
        # Pad with NaN to match max length
        padded_rates = rates + [np.nan] * (max_years - len(rates))
        heatmap_data.append([row[country_col]] + padded_rates)
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df.columns = ['Country'] + [f'Year {i+1}' for i in range(max_years)]
        
        # Create heatmap
        fig = px.imshow(
            heatmap_df.set_index('Country'),
            aspect='auto',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title='Year-over-Year Growth Rate Heatmap (%)'
        )
        
        fig.update_layout(height=max(400, len(heatmap_df) * 25), template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Growth volatility analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility vs Growth scatter
        fig = px.scatter(
            growth_data,
            x='Avg_YoY_Growth',
            y='Volatility',
            size='Latest_Value',
            color='CAGR',
            hover_name=country_col,
            title='Growth vs Volatility Analysis',
            labels={
                'Avg_YoY_Growth': 'Average YoY Growth (%)',
                'Volatility': 'Growth Volatility (Std Dev)',
                'CAGR': 'CAGR (%)'
            },
            color_continuous_scale='Spectral'
        )
        
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Growth acceleration
        fig = px.bar(
            growth_data.sort_values('Growth_Acceleration', ascending=True),
            x='Growth_Acceleration',
            y=country_col,
            orientation='h',
            color='Growth_Acceleration',
            color_continuous_scale='RdYlGn',
            title='Growth Acceleration Trend',
            labels={'Growth_Acceleration': 'Growth Acceleration (% change/year)'}
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

def render_volatility_analysis(growth_data: pd.DataFrame, country_col: str) -> None:
    """
    Render volatility analysis
    """
    st.markdown("### 🌊 Growth Volatility Analysis")
    
    # Volatility metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_volatility = growth_data['Volatility'].mean()
        st.metric("Average Volatility", f"{avg_volatility:.1f}%")
    
    with col2:
        low_vol_count = len(growth_data[growth_data['Volatility'] < avg_volatility * 0.5])
        st.metric("Low Volatility Markets", f"{low_vol_count}")
    
    with col3:
        high_vol_count = len(growth_data[growth_data['Volatility'] > avg_volatility * 1.5])
        st.metric("High Volatility Markets", f"{high_vol_count}")
    
    # Volatility distribution
    fig = px.histogram(
        growth_data,
        x='Volatility',
        nbins=20,
        title='Distribution of Growth Volatility',
        labels={'Volatility': 'Growth Volatility (%)'},
        color_discrete_sequence=['#3498DB']
    )
    
    fig.add_vline(x=avg_volatility, line_dash="dash", line_color="red", 
                  annotation_text="Average")
    fig.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk-Return matrix
    st.markdown("#### 🎯 Risk-Return Matrix")
    
    fig = go.Figure()
    
    # Quadrants
    avg_return = growth_data['CAGR'].mean()
    avg_risk = growth_data['Volatility'].mean()
    
    # Add quadrant backgrounds
    fig.add_shape(type="rect", x0=-50, y0=0, x1=avg_return, y1=avg_risk,
                  fillcolor="lightcoral", opacity=0.3, line_width=0)
    fig.add_shape(type="rect", x0=avg_return, y0=0, x1=100, y1=avg_risk,
                  fillcolor="lightgreen", opacity=0.3, line_width=0)
    fig.add_shape(type="rect", x0=-50, y0=avg_risk, x1=avg_return, y1=50,
                  fillcolor="lightyellow", opacity=0.3, line_width=0)
    fig.add_shape(type="rect", x0=avg_return, y0=avg_risk, x1=100, y1=50,
                  fillcolor="lightblue", opacity=0.3, line_width=0)
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=growth_data['CAGR'],
        y=growth_data['Volatility'],
        mode='markers+text',
        text=growth_data[country_col],
        textposition='top center',
        marker=dict(
            size=growth_data['Latest_Value'] / growth_data['Latest_Value'].max() * 30 + 10,
            color=growth_data['CAGR'],
            colorscale='RdYlGn',
            colorbar=dict(title="CAGR (%)")
        ),
        hovertemplate='<b>%{text}</b><br>CAGR: %{x:.1f}%<br>Volatility: %{y:.1f}%<extra></extra>'
    ))
    
    # Add quadrant lines
    fig.add_hline(y=avg_risk, line_dash="dash", line_color="gray")
    fig.add_vline(x=avg_return, line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig.add_annotation(x=avg_return/2, y=avg_risk/2, text="Low Return<br>Low Risk", 
                      showarrow=False, font=dict(size=12))
    fig.add_annotation(x=avg_return + (100-avg_return)/2, y=avg_risk/2, text="High Return<br>Low Risk", 
                      showarrow=False, font=dict(size=12))
    fig.add_annotation(x=avg_return/2, y=avg_risk + (50-avg_risk)/2, text="Low Return<br>High Risk", 
                      showarrow=False, font=dict(size=12))
    fig.add_annotation(x=avg_return + (100-avg_return)/2, y=avg_risk + (50-avg_risk)/2, 
                      text="High Return<br>High Risk", showarrow=False, font=dict(size=12))
    
    fig.update_layout(
        title='Risk-Return Analysis Matrix',
        xaxis_title='CAGR (%)',
        yaxis_title='Volatility (%)',
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_growth_trajectory_analysis(growth_data: pd.DataFrame, country_col: str) -> None:
    """
    Render growth trajectory analysis
    """
    st.markdown("### 🛤️ Growth Trajectory Analysis")
    
    # Select countries for trajectory analysis
    top_countries = growth_data.nlargest(8, 'Latest_Value')[country_col].tolist()
    selected_countries = st.multiselect(
        "Select Countries for Trajectory Analysis",
        growth_data[country_col].tolist(),
        default=top_countries[:5],
        help="Choose countries to compare growth trajectories"
    )
    
    if not selected_countries:
        st.warning("Please select at least one country for trajectory analysis.")
        return
    
    # Create trajectory plot
    fig = go.Figure()
    
    colors = COLOR_SCHEMES['trend']
    
    for i, country in enumerate(selected_countries):
        country_data = growth_data[growth_data[country_col] == country].iloc[0]
        years = country_data['Years']
        values = country_data['Values']
        
        # Normalize to base 100 for comparison
        normalized_values = [(v / values[0]) * 100 for v in values]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=normalized_values,
            mode='lines+markers',
            name=country,
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8),
            hovertemplate=f'<b>{country}</b><br>Year: %{{x}}<br>Index: %{{y:.1f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Growth Trajectory Comparison (Base Year = 100)',
        xaxis_title='Year',
        yaxis_title='Growth Index (Base Year = 100)',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth pattern analysis
    st.markdown("#### 📊 Growth Pattern Classification")
    
    # Classify growth patterns
    patterns = []
    for _, row in growth_data.iterrows():
        if len(row['YoY_Rates']) < 3:
            pattern = "Insufficient Data"
        else:
            rates = row['YoY_Rates']
            trend = np.polyfit(range(len(rates)), rates, 1)[0]
            volatility = row['Volatility']
            avg_growth = row['Avg_YoY_Growth']
            
            if avg_growth > 15 and volatility < 10:
                pattern = "Steady High Growth"
            elif avg_growth > 5 and trend > 0:
                pattern = "Accelerating Growth"
            elif avg_growth > 0 and volatility < 5:
                pattern = "Stable Growth"
            elif trend < -2:
                pattern = "Declining"
            elif volatility > 20:
                pattern = "Volatile"
            else:
                pattern = "Mixed Signals"
        
        patterns.append(pattern)
    
    growth_data['Growth_Pattern'] = patterns
    
    # Pattern distribution
    pattern_counts = growth_data['Growth_Pattern'].value_counts()
    
    fig = px.pie(
        values=pattern_counts.values,
        names=pattern_counts.index,
        title='Distribution of Growth Patterns',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

def render_growth_forecasting(growth_data: pd.DataFrame, country_col: str) -> None:
    """
    Render growth forecasting analysis
    """
    st.markdown("### 🔮 Growth Forecasting Analysis")
    
    # Forecasting controls
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_years = st.slider("Forecast Horizon (Years)", 1, 10, 5)
        
    with col2:
        forecast_method = st.selectbox(
            "Forecasting Method",
            ["Linear Trend", "Polynomial Trend", "Growth Rate Projection", "Conservative Estimate"]
        )
    
    # Select country for detailed forecasting
    selected_country = st.selectbox(
        "Select Country for Detailed Forecast",
        growth_data[country_col].tolist(),
        index=0
    )
    
    if selected_country:
        country_row = growth_data[growth_data[country_col] == selected_country].iloc[0]
        
        # Create forecast
        forecast_result = create_growth_forecast(
            country_row, forecast_years, forecast_method
        )
        
        # Plot historical and forecast
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=country_row['Years'],
            y=country_row['Values'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        
        # Forecast data
        if forecast_result:
            forecast_years_list = list(range(country_row['Years'][-1] + 1, 
                                           country_row['Years'][-1] + 1 + forecast_years))
            
            fig.add_trace(go.Scatter(
                x=forecast_years_list,
                y=forecast_result['values'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#E76F51', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Confidence intervals if available
            if 'confidence_upper' in forecast_result and 'confidence_lower' in forecast_result:
                fig.add_trace(go.Scatter(
                    x=forecast_years_list + forecast_years_list[::-1],
                    y=forecast_result['confidence_upper'] + forecast_result['confidence_lower'][::-1],
                    fill='toself',
                    fillcolor='rgba(231, 111, 81, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval',
                    showlegend=True
                ))
        
        fig.update_layout(
            title=f'Growth Forecast for {selected_country}',
            xaxis_title='Year',
            yaxis_title='Market Size (USD)',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        if forecast_result:
            st.markdown("#### 📈 Forecast Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_value = country_row['Values'][-1]
                forecast_value = forecast_result['values'][-1]
                if current_value > 0 and forecast_years > 0:
                    growth = ((forecast_value / current_value) ** (1 / forecast_years) - 1) * 100
                else:
                    growth = 0
                st.metric("Forecasted CAGR", f"{growth:.1f}%")
            
            with col2:
                st.metric("Final Year Value", f"${forecast_value/1e9:.1f}B" if forecast_value > 1e9 else f"${forecast_value/1e6:.0f}M")
            
            with col3:
                total_growth = ((forecast_value / current_value) - 1) * 100
                st.metric("Total Growth", f"{total_growth:.1f}%")

def create_growth_forecast(country_row: pd.Series, forecast_years: int, method: str) -> Dict:
    """
    Create growth forecast for a country
    """
    years = np.array(country_row['Years'])
    values = np.array(country_row['Values'])
    
    if len(years) < 2:
        return None
    
    try:
        if method == "Linear Trend":
            # Linear regression
            X = years.reshape(-1, 1)
            reg = LinearRegression().fit(X, values)
            
            forecast_years_array = np.arange(years[-1] + 1, years[-1] + 1 + forecast_years)
            forecast_values = reg.predict(forecast_years_array.reshape(-1, 1))
            
            # Simple confidence interval (±20% of trend)
            trend_slope = reg.coef_[0]
            confidence_range = abs(trend_slope) * 0.2 * np.arange(1, forecast_years + 1)
            
            return {
                'values': forecast_values.tolist(),
                'confidence_upper': (forecast_values + confidence_range).tolist(),
                'confidence_lower': (forecast_values - confidence_range).tolist()
            }
            
        elif method == "Polynomial Trend":
            # Polynomial regression (degree 2)
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(years.reshape(-1, 1))
            reg = LinearRegression().fit(X_poly, values)
            
            forecast_years_array = np.arange(years[-1] + 1, years[-1] + 1 + forecast_years)
            X_forecast = poly_features.transform(forecast_years_array.reshape(-1, 1))
            forecast_values = reg.predict(X_forecast)
            
            return {
                'values': forecast_values.tolist()
            }
            
        elif method == "Growth Rate Projection":
            # Use historical CAGR
            cagr = country_row['CAGR'] / 100
            current_value = values[-1]
            
            forecast_values = [current_value * ((1 + cagr) ** i) for i in range(1, forecast_years + 1)]
            
            return {
                'values': forecast_values
            }
            
        else:  # Conservative Estimate
            # Use 75% of historical CAGR
            conservative_cagr = country_row['CAGR'] * 0.75 / 100
            current_value = values[-1]
            
            forecast_values = [current_value * ((1 + conservative_cagr) ** i) for i in range(1, forecast_years + 1)]
            
            return {
                'values': forecast_values
            }
            
    except Exception as e:
        logger.error(f"Error creating forecast: {str(e)}")
        return None