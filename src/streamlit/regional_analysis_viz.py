"""
Comprehensive Regional Analysis Visualizations
Advanced geographical market analysis with interactive maps and regional insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class RegionalAnalysisVisualizer:
    """Comprehensive regional analysis with advanced geographical visualizations"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#7209B7',
            'background': '#f8f9fa',
            'text': '#2c3e50'
        }
        
        # Regional color schemes
        self.regional_colors = {
            'North America': '#1f77b4',
            'Europe': '#ff7f0e',
            'Asia Pacific': '#2ca02c',
            'Latin America': '#d62728',
            'Middle East & Africa': '#9467bd',
            'Others': '#8c564b'
        }
        
        # Country to region mapping (expandable)
        self.country_to_region = {
            'USA': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
            'UK': 'Europe', 'Germany': 'Europe', 'France': 'Europe', 'Italy': 'Europe',
            'Spain': 'Europe', 'Netherlands': 'Europe', 'Sweden': 'Europe', 'Norway': 'Europe',
            'China': 'Asia Pacific', 'Japan': 'Asia Pacific', 'India': 'Asia Pacific',
            'South Korea': 'Asia Pacific', 'Australia': 'Asia Pacific', 'Singapore': 'Asia Pacific',
            'Thailand': 'Asia Pacific', 'Indonesia': 'Asia Pacific', 'Malaysia': 'Asia Pacific',
            'Brazil': 'Latin America', 'Argentina': 'Latin America', 'Chile': 'Latin America',
            'Colombia': 'Latin America', 'Peru': 'Latin America',
            'Saudi Arabia': 'Middle East & Africa', 'UAE': 'Middle East & Africa',
            'South Africa': 'Middle East & Africa', 'Nigeria': 'Middle East & Africa',
            'Egypt': 'Middle East & Africa', 'Israel': 'Middle East & Africa'
        }
    
    def apply_professional_styling(self):
        """Apply professional CSS styling for regional analysis"""
        st.markdown("""
        <style>
        .regional-header {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .regional-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .regional-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .region-metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        .region-metric-card h3 {
            margin: 0 0 1rem 0;
            font-size: 1.3rem;
        }
        
        .region-metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        .region-metric-card .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .growth-indicator {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        
        .growth-high { background: #d4edda; color: #155724; }
        .growth-medium { background: #fff3cd; color: #856404; }
        .growth-low { background: #f8d7da; color: #721c24; }
        
        .regional-insight {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 1.2rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #ff6b35;
        }
        
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .regional-stats {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .country-tile {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .country-tile:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_header(self):
        """Create professional header for regional analysis"""
        st.markdown("""
        <div class="regional-header">
            <h1>🌍 Regional Market Analysis</h1>
            <p>Comprehensive geographical insights and regional market dynamics</p>
        </div>
        """, unsafe_allow_html=True)
    
    def add_regional_mapping(self, data: pd.DataFrame, country_col: str) -> pd.DataFrame:
        """Add regional mapping to data"""
        data_with_regions = data.copy()
        data_with_regions['Region'] = data_with_regions[country_col].map(
            lambda x: self.country_to_region.get(x, 'Others')
        )
        return data_with_regions
    
    def create_world_choropleth(self, data: pd.DataFrame, country_col: str, 
                               value_col: str, title: str) -> go.Figure:
        """Create interactive world choropleth map"""
        fig = px.choropleth(
            data,
            locations=country_col,
            color=value_col,
            hover_name=country_col,
            hover_data={value_col: ':,.0f'},
            color_continuous_scale='Viridis',
            locationmode='country names',
            title=title
        )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            height=600,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        return fig
    
    def create_regional_sunburst(self, data: pd.DataFrame, country_col: str, 
                                value_col: str, title: str) -> go.Figure:
        """Create regional sunburst chart showing hierarchy"""
        data_with_regions = self.add_regional_mapping(data, country_col)
        
        # Aggregate by region
        regional_data = data_with_regions.groupby('Region')[value_col].sum().reset_index()
        
        # Create hierarchical data
        ids = []
        labels = []
        parents = []
        values = []
        
        # Add regions
        for _, row in regional_data.iterrows():
            ids.append(row['Region'])
            labels.append(row['Region'])
            parents.append("")
            values.append(row[value_col])
        
        # Add countries
        for _, row in data_with_regions.iterrows():
            ids.append(f"{row['Region']}-{row[country_col]}")
            labels.append(row[country_col])
            parents.append(row['Region'])
            values.append(row[value_col])
        
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percentParent}<extra></extra>',
            maxdepth=2
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            font=dict(size=12),
            height=600,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        return fig
    
    def create_regional_comparison_radar(self, data: pd.DataFrame, metrics: List[str]) -> go.Figure:
        """Create radar chart comparing regions across multiple metrics"""
        data_with_regions = self.add_regional_mapping(data, data.columns[0])
        
        # Aggregate metrics by region
        regional_metrics = data_with_regions.groupby('Region')[metrics].mean()
        
        # Normalize metrics to 0-100 scale
        normalized_metrics = pd.DataFrame()
        for metric in metrics:
            min_val = regional_metrics[metric].min()
            max_val = regional_metrics[metric].max()
            if max_val > min_val:
                normalized_metrics[metric] = ((regional_metrics[metric] - min_val) / (max_val - min_val)) * 100
            else:
                normalized_metrics[metric] = 50
        
        fig = go.Figure()
        
        colors = list(self.regional_colors.values())
        for i, (region, values) in enumerate(normalized_metrics.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=values.tolist() + [values.iloc[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=region,
                line_color=colors[i % len(colors)],
                fillcolor=f"rgba{(*tuple(int(colors[i % len(colors)][1:][j:j+2], 16) for j in (0, 2, 4)), 0.3)}"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title={
                'text': 'Regional Performance Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            showlegend=True,
            height=600,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        return fig
    
    def create_regional_growth_heatmap(self, data: pd.DataFrame, country_col: str, 
                                     year_col: str, value_col: str) -> go.Figure:
        """Create heatmap showing regional growth patterns"""
        data_with_regions = self.add_regional_mapping(data, country_col)
        
        # Calculate year-over-year growth by region
        regional_yearly = data_with_regions.groupby(['Region', year_col])[value_col].sum().reset_index()
        
        # Calculate growth rates
        growth_data = []
        for region in regional_yearly['Region'].unique():
            region_data = regional_yearly[regional_yearly['Region'] == region].sort_values(year_col)
            if len(region_data) > 1:
                region_data['growth_rate'] = region_data[value_col].pct_change() * 100
                growth_data.append(region_data)
        
        if not growth_data:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for growth analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        combined_growth = pd.concat(growth_data, ignore_index=True)
        combined_growth = combined_growth.dropna(subset=['growth_rate'])
        
        # Create pivot table for heatmap
        heatmap_data = combined_growth.pivot(index='Region', columns=year_col, values='growth_rate')
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Year", y="Region", color="Growth Rate (%)"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlBu_r',
            aspect="auto"
        )
        
        fig.update_layout(
            title={
                'text': 'Regional Growth Rate Heatmap',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            height=400,
            margin=dict(t=80, b=40, l=120, r=40)
        )
        
        return fig
    
    def create_regional_bubble_chart(self, data: pd.DataFrame, country_col: str,
                                   x_col: str, y_col: str, size_col: str) -> go.Figure:
        """Create bubble chart for regional analysis"""
        data_with_regions = self.add_regional_mapping(data, country_col)
        
        fig = go.Figure()
        
        for region in data_with_regions['Region'].unique():
            region_data = data_with_regions[data_with_regions['Region'] == region]
            
            fig.add_trace(go.Scatter(
                x=region_data[x_col],
                y=region_data[y_col],
                mode='markers',
                marker=dict(
                    size=region_data[size_col] / region_data[size_col].max() * 50 + 10,
                    color=self.regional_colors.get(region, '#8c564b'),
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name=region,
                text=region_data[country_col],
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_col}: %{{x}}<br>' +
                             f'{y_col}: %{{y}}<br>' +
                             f'{size_col}: %{{marker.size}}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': f'Regional Analysis: {y_col} vs {x_col}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=600,
            margin=dict(t=80, b=60, l=60, r=40),
            showlegend=True
        )
        
        return fig
    
    def create_regional_bar_race(self, data: pd.DataFrame, country_col: str,
                               year_col: str, value_col: str) -> go.Figure:
        """Create animated bar chart showing regional evolution"""
        data_with_regions = self.add_regional_mapping(data, country_col)
        
        # Aggregate by region and year
        regional_data = data_with_regions.groupby(['Region', year_col])[value_col].sum().reset_index()
        
        fig = px.bar(
            regional_data,
            x='Region',
            y=value_col,
            animation_frame=year_col,
            color='Region',
            color_discrete_map=self.regional_colors,
            title='Regional Market Evolution Over Time'
        )
        
        fig.update_layout(
            title={
                'text': 'Regional Market Evolution Over Time',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title='Region',
            yaxis_title=value_col,
            height=500,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        return fig
    
    def calculate_regional_metrics(self, data: pd.DataFrame, country_col: str, 
                                 value_col: str) -> Dict[str, Any]:
        """Calculate comprehensive regional metrics"""
        data_with_regions = self.add_regional_mapping(data, country_col)
        
        regional_summary = data_with_regions.groupby('Region').agg({
            value_col: ['sum', 'mean', 'count', 'std']
        }).round(2)
        
        regional_summary.columns = ['Total', 'Average', 'Count', 'StdDev']
        regional_summary = regional_summary.reset_index()
        
        # Calculate additional metrics
        total_market = data[value_col].sum()
        regional_summary['Market_Share'] = (regional_summary['Total'] / total_market * 100).round(2)
        regional_summary['Coefficient_of_Variation'] = (regional_summary['StdDev'] / regional_summary['Average'] * 100).round(2)
        
        # Identify dominant region
        dominant_region = regional_summary.loc[regional_summary['Total'].idxmax(), 'Region']
        
        # Calculate regional concentration
        hhi = ((regional_summary['Market_Share'] / 100) ** 2).sum() * 10000
        
        return {
            'summary': regional_summary,
            'dominant_region': dominant_region,
            'total_regions': len(regional_summary),
            'hhi': hhi,
            'concentration_level': 'High' if hhi > 2500 else 'Moderate' if hhi > 1500 else 'Low'
        }
    
    def display_regional_metrics(self, metrics: Dict[str, Any]):
        """Display regional metrics in cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="region-metric-card">
                <h3>Dominant Region</h3>
                <div class="metric-value">{metrics['dominant_region']}</div>
                <div class="metric-label">Leading Market</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="region-metric-card">
                <h3>Total Regions</h3>
                <div class="metric-value">{metrics['total_regions']}</div>
                <div class="metric-label">Active Markets</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="region-metric-card">
                <h3>Concentration</h3>
                <div class="metric-value">{metrics['concentration_level']}</div>
                <div class="metric-label">HHI: {metrics['hhi']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            top_region_share = metrics['summary']['Market_Share'].max()
            st.markdown(f"""
            <div class="region-metric-card">
                <h3>Top Share</h3>
                <div class="metric-value">{top_region_share:.1f}%</div>
                <div class="metric-label">Regional Dominance</div>
            </div>
            """, unsafe_allow_html=True)
    
    def generate_regional_insights(self, data: pd.DataFrame, country_col: str, 
                                 value_col: str, year_col: str = None) -> List[str]:
        """Generate actionable regional insights"""
        data_with_regions = self.add_regional_mapping(data, country_col)
        metrics = self.calculate_regional_metrics(data, country_col, value_col)
        
        insights = []
        
        # Regional dominance insights
        dominant_region = metrics['dominant_region']
        dominant_share = metrics['summary'].loc[
            metrics['summary']['Region'] == dominant_region, 'Market_Share'
        ].iloc[0]
        
        insights.append(f"🏆 {dominant_region} dominates with {dominant_share:.1f}% of total market")
        
        # Market concentration insights
        if metrics['concentration_level'] == 'High':
            insights.append("⚠️ High regional concentration - consider diversification strategies")
        elif metrics['concentration_level'] == 'Low':
            insights.append("🌟 Well-distributed regional presence - balanced market approach")
        
        # Regional diversity insights
        active_regions = len(metrics['summary'])
        if active_regions >= 4:
            insights.append(f"🌍 Strong global presence across {active_regions} regions")
        elif active_regions == 3:
            insights.append("📍 Multi-regional strategy with room for expansion")
        else:
            insights.append("🎯 Focused regional approach - expansion opportunities available")
        
        # Performance variance insights
        cv_values = metrics['summary']['Coefficient_of_Variation'].dropna()
        if len(cv_values) > 0:
            avg_cv = cv_values.mean()
            if avg_cv > 100:
                insights.append("📊 High regional performance variance - review market strategies")
            elif avg_cv < 50:
                insights.append("⚖️ Consistent regional performance across markets")
        
        # Growth insights (if year data available)
        if year_col and year_col in data.columns:
            years = sorted(data[year_col].unique())
            if len(years) > 1:
                latest_year = max(years)
                previous_year = years[-2]
                
                latest_regional = data_with_regions[data_with_regions[year_col] == latest_year].groupby('Region')[value_col].sum()
                previous_regional = data_with_regions[data_with_regions[year_col] == previous_year].groupby('Region')[value_col].sum()
                
                growth_regions = []
                for region in latest_regional.index:
                    if region in previous_regional.index:
                        growth = ((latest_regional[region] / previous_regional[region]) - 1) * 100
                        if growth > 10:
                            growth_regions.append(region)
                
                if growth_regions:
                    insights.append(f"📈 Strong growth in {', '.join(growth_regions[:2])} region(s)")
        
        return insights
    
    def render_regional_dashboard(self, data: pd.DataFrame, country_col: str, 
                                value_col: str, year_col: str = None):
        """Render complete regional analysis dashboard"""
        self.apply_professional_styling()
        self.create_header()
        
        if data.empty:
            st.error("No data available for regional analysis")
            return
        
        # Add regional mapping
        data_with_regions = self.add_regional_mapping(data, country_col)
        
        # Sidebar controls
        st.sidebar.header("🌍 Regional Controls")
        
        # Region filter
        available_regions = sorted(data_with_regions['Region'].unique())
        selected_regions = st.sidebar.multiselect(
            "Select Regions",
            available_regions,
            default=available_regions
        )
        
        filtered_data = data_with_regions[data_with_regions['Region'].isin(selected_regions)]
        
        # Year filter if available
        if year_col and year_col in data.columns:
            years = sorted(filtered_data[year_col].unique())
            selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)
            year_filtered_data = filtered_data[filtered_data[year_col] == selected_year]
        else:
            year_filtered_data = filtered_data
            selected_year = "All Years"
        
        # Chart selection
        chart_types = st.sidebar.multiselect(
            "Select Visualizations",
            ["World Map", "Regional Sunburst", "Growth Heatmap", "Bubble Chart", "Bar Race"],
            default=["World Map", "Regional Sunburst"]
        )
        
        # Calculate and display metrics
        st.subheader(f"📊 Regional Overview - {selected_year}")
        metrics = self.calculate_regional_metrics(year_filtered_data, country_col, value_col)
        self.display_regional_metrics(metrics)
        
        # Generate insights
        insights = self.generate_regional_insights(filtered_data, country_col, value_col, year_col)
        if insights:
            st.markdown("### 💡 Regional Insights")
            for insight in insights:
                st.markdown(f"""
                <div class="regional-insight">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Display charts
        if "World Map" in chart_types:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = self.create_world_choropleth(
                year_filtered_data, country_col, value_col,
                f"Global Market Distribution - {selected_year}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if "Regional Sunburst" in chart_types:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = self.create_regional_sunburst(
                year_filtered_data, country_col, value_col,
                f"Regional Market Hierarchy - {selected_year}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if "Growth Heatmap" in chart_types and year_col and year_col in data.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = self.create_regional_growth_heatmap(filtered_data, country_col, year_col, value_col)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if "Bubble Chart" in chart_types and len(year_filtered_data.columns) >= 4:
            numeric_cols = year_filtered_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = self.create_regional_bubble_chart(
                    year_filtered_data, country_col,
                    numeric_cols[0], numeric_cols[1], value_col
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        if "Bar Race" in chart_types and year_col and year_col in data.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = self.create_regional_bar_race(filtered_data, country_col, year_col, value_col)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Regional summary table
        st.subheader("📋 Regional Performance Summary")
        st.markdown('<div class="regional-stats">', unsafe_allow_html=True)
        summary_display = metrics['summary'].round(2)
        st.dataframe(summary_display, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Country details by region
        if st.expander("🏳️ Country Details by Region"):
            for region in selected_regions:
                region_countries = year_filtered_data[year_filtered_data['Region'] == region]
                if not region_countries.empty:
                    st.markdown(f"**{region}**")
                    country_summary = region_countries[[country_col, value_col]].sort_values(value_col, ascending=False)
                    
                    cols = st.columns(min(4, len(country_summary)))
                    for i, (_, row) in enumerate(country_summary.iterrows()):
                        with cols[i % 4]:
                            st.markdown(f"""
                            <div class="country-tile">
                                <strong>{row[country_col]}</strong><br>
                                {row[value_col]:,.0f}
                            </div>
                            """, unsafe_allow_html=True)


def main():
    """Main function for testing"""
    visualizer = RegionalAnalysisVisualizer()
    
    # Sample data
    sample_data = pd.DataFrame({
        'Country': ['USA', 'China', 'Germany', 'Japan', 'UK', 'France', 'India', 'Brazil', 'Canada', 'Australia'],
        'Market_Size': [1000, 800, 400, 350, 300, 250, 200, 150, 120, 100],
        'Year': [2023] * 10
    })
    
    visualizer.render_regional_dashboard(sample_data, 'Country', 'Market_Size', 'Year')

if __name__ == "__main__":
    main()