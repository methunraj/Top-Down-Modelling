"""
Enhanced Market Share Analysis Visualizations
Professional dashboard for comprehensive market share analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MarketShareVisualizer:
    """Enhanced market share analysis with professional visualizations"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'background': '#f8f9fa',
            'text': '#2c3e50'
        }
        
        self.extended_colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
    
    def apply_professional_styling(self):
        """Apply professional CSS styling"""
        st.markdown("""
        <style>
        .market-share-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .market-share-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .market-share-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .metric-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        
        .concentration-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .share-evolution-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .competitive-card {
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
        }
        
        .trend-indicator {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        
        .trend-up { background: #d4edda; color: #155724; }
        .trend-down { background: #f8d7da; color: #721c24; }
        .trend-stable { background: #fff3cd; color: #856404; }
        
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .insight-box {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #ff6b35;
        }
        
        .stSelectbox > div > div {
            background-color: white;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_header(self):
        """Create professional header"""
        st.markdown("""
        <div class="market-share-header">
            <h1>📊 Market Share Analytics</h1>
            <p>Comprehensive analysis of market dynamics and competitive positioning</p>
        </div>
        """, unsafe_allow_html=True)
    
    def calculate_market_concentration(self, data: pd.DataFrame, value_col: str) -> Dict[str, float]:
        """Calculate market concentration metrics"""
        shares = data[value_col].values
        total = shares.sum()
        
        if total <= 0:
            return {
                'hhi': 0,
                'cr4': 0,
                'cr8': 0,
                'gini': 0,
                'entropy': 0
            }
        
        # Normalize shares
        normalized_shares = shares / total
        sorted_shares = np.sort(normalized_shares)[::-1]
        
        # Herfindahl-Hirschman Index
        hhi = np.sum(normalized_shares ** 2) * 10000
        
        # Concentration ratios
        cr4 = np.sum(sorted_shares[:min(4, len(sorted_shares))]) * 100
        cr8 = np.sum(sorted_shares[:min(8, len(sorted_shares))]) * 100
        
        # Gini coefficient
        def gini_coefficient(shares):
            if len(shares) == 0:
                return 0
            sorted_shares = np.sort(shares)
            n = len(shares)
            index = np.arange(1, n + 1)
            return (2 * np.sum(index * sorted_shares)) / (n * np.sum(sorted_shares)) - (n + 1) / n
        
        gini = gini_coefficient(normalized_shares)
        
        # Shannon entropy
        entropy = -np.sum(normalized_shares * np.log2(normalized_shares + 1e-10))
        
        return {
            'hhi': hhi,
            'cr4': cr4,
            'cr8': cr8,
            'gini': gini,
            'entropy': entropy
        }
    
    def create_interactive_pie_chart(self, data: pd.DataFrame, names_col: str, 
                                   values_col: str, title: str) -> go.Figure:
        """Create interactive pie chart with hover details"""
        # Sort data by values
        data_sorted = data.sort_values(values_col, ascending=False)
        
        # Group small segments
        threshold = data_sorted[values_col].sum() * 0.02  # 2% threshold
        large_segments = data_sorted[data_sorted[values_col] >= threshold]
        small_segments = data_sorted[data_sorted[values_col] < threshold]
        
        if len(small_segments) > 0:
            others_row = pd.DataFrame({
                names_col: ['Others'],
                values_col: [small_segments[values_col].sum()]
            })
            plot_data = pd.concat([large_segments, others_row], ignore_index=True)
        else:
            plot_data = large_segments
        
        fig = go.Figure(data=[go.Pie(
            labels=plot_data[names_col],
            values=plot_data[values_col],
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>' +
                         'Value: %{value:,.0f}<br>' +
                         'Share: %{percent}<br>' +
                         '<extra></extra>',
            marker=dict(
                colors=self.extended_colors[:len(plot_data)],
                line=dict(color='white', width=2)
            ),
            pull=[0.1 if i == 0 else 0 for i in range(len(plot_data))]
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05
            ),
            margin=dict(t=80, b=40, l=40, r=160),
            height=500
        )
        
        return fig
    
    def create_treemap_visualization(self, data: pd.DataFrame, names_col: str, 
                                   values_col: str, title: str) -> go.Figure:
        """Create treemap for hierarchical market share view"""
        # Sort data
        data_sorted = data.sort_values(values_col, ascending=False)
        
        fig = go.Figure(go.Treemap(
            labels=data_sorted[names_col],
            values=data_sorted[values_col],
            parents=[""] * len(data_sorted),
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>' +
                         'Value: %{value:,.0f}<br>' +
                         'Share: %{percentParent}<br>' +
                         '<extra></extra>',
            marker=dict(
                colors=self.extended_colors[:len(data_sorted)],
                line=dict(width=2, color='white')
            )
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            font=dict(size=12),
            margin=dict(t=80, b=40, l=40, r=40),
            height=500
        )
        
        return fig
    
    def create_share_evolution_chart(self, data: pd.DataFrame, entity_col: str, 
                                   year_col: str, value_col: str) -> go.Figure:
        """Create market share evolution over time"""
        # Ensure year column is numeric for proper sorting
        data = data.copy()
        data[year_col] = pd.to_numeric(data[year_col], errors='coerce')
        data = data.dropna(subset=[year_col])
        
        # Calculate shares by year
        yearly_totals = data.groupby(year_col)[value_col].sum()
        data_with_shares = data.copy()
        data_with_shares['share'] = data_with_shares.apply(
            lambda row: (row[value_col] / yearly_totals[row[year_col]] * 100) if yearly_totals[row[year_col]] > 0 else 0, axis=1
        )
        
        # Get top entities by average share
        avg_shares = data_with_shares.groupby(entity_col)['share'].mean().sort_values(ascending=False)
        top_entities = avg_shares.head(10).index.tolist()
        
        fig = go.Figure()
        
        for i, entity in enumerate(top_entities):
            entity_data = data_with_shares[data_with_shares[entity_col] == entity]
            entity_data = entity_data.sort_values(year_col)
            
            fig.add_trace(go.Scatter(
                x=entity_data[year_col],
                y=entity_data['share'],
                mode='lines+markers',
                name=entity,
                line=dict(
                    color=self.extended_colors[i % len(self.extended_colors)],
                    width=3
                ),
                marker=dict(
                    size=8,
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Year: %{x}<br>' +
                             'Share: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': 'Market Share Evolution Over Time',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title='Year',
            yaxis_title='Market Share (%)',
            font=dict(size=12),
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(t=80, b=60, l=60, r=150),
            height=500
        )
        
        return fig
    
    def create_competitive_positioning_matrix(self, data: pd.DataFrame, entity_col: str,
                                            size_col: str, growth_col: str) -> go.Figure:
        """Create competitive positioning matrix (size vs growth)"""
        # Calculate growth rates if not provided
        if growth_col not in data.columns:
            data = data.copy()
            data = data.sort_values(['entity', 'year'])
            data['growth_rate'] = data.groupby(entity_col)[size_col].pct_change() * 100
            growth_col = 'growth_rate'
        
        # Get latest data for each entity
        latest_data = data.groupby(entity_col).last().reset_index()
        
        # Remove entities with missing data
        latest_data = latest_data.dropna(subset=[size_col, growth_col])
        
        if len(latest_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for competitive positioning",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate bubble sizes (normalize)
        max_size = latest_data[size_col].max()
        latest_data['bubble_size'] = (latest_data[size_col] / max_size) * 50 + 10
        
        # Define quadrants
        median_size = latest_data[size_col].median()
        median_growth = latest_data[growth_col].median()
        
        fig = go.Figure()
        
        # Add quadrant backgrounds
        fig.add_shape(
            type="rect",
            x0=latest_data[size_col].min(), x1=median_size,
            y0=median_growth, y1=latest_data[growth_col].max(),
            fillcolor="rgba(255, 182, 193, 0.3)",
            line=dict(width=0),
            layer="below"
        )
        
        fig.add_shape(
            type="rect",
            x0=median_size, x1=latest_data[size_col].max(),
            y0=median_growth, y1=latest_data[growth_col].max(),
            fillcolor="rgba(144, 238, 144, 0.3)",
            line=dict(width=0),
            layer="below"
        )
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=latest_data[size_col],
            y=latest_data[growth_col],
            mode='markers+text',
            text=latest_data[entity_col],
            textposition='top center',
            marker=dict(
                size=latest_data['bubble_size'],
                color=self.extended_colors[:len(latest_data)],
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         f'{size_col}: %{{x:,.0f}}<br>' +
                         f'{growth_col}: %{{y:.1f}}%<br>' +
                         '<extra></extra>',
            showlegend=False
        ))
        
        # Add median lines
        fig.add_hline(y=median_growth, line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_vline(x=median_size, line_dash="dash", line_color="gray", opacity=0.7)
        
        # Add quadrant labels
        fig.add_annotation(
            x=median_size * 0.5, y=latest_data[growth_col].max() * 0.9,
            text="Rising Stars", showarrow=False,
            font=dict(size=14, color="red")
        )
        
        fig.add_annotation(
            x=latest_data[size_col].max() * 0.9, y=latest_data[growth_col].max() * 0.9,
            text="Market Leaders", showarrow=False,
            font=dict(size=14, color="green")
        )
        
        fig.update_layout(
            title={
                'text': 'Competitive Positioning Matrix',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title=f'Market Size ({size_col})',
            yaxis_title=f'Growth Rate (%)',
            font=dict(size=12),
            height=600,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        return fig
    
    def create_waterfall_chart(self, data: pd.DataFrame, categories: List[str], 
                              values: List[float], title: str) -> go.Figure:
        """Create waterfall chart for market share changes"""
        fig = go.Figure(go.Waterfall(
            name="Market Share Changes",
            orientation="v",
            measure=["relative"] * (len(categories) - 2) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"{v:+.1f}%" if v != 0 else "0%" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": self.color_palette['warning']}},
            increasing={"marker": {"color": self.color_palette['success']}},
            totals={"marker": {"color": self.color_palette['primary']}}
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title='Period',
            yaxis_title='Market Share Change (%)',
            font=dict(size=12),
            height=500,
            margin=dict(t=80, b=60, l=60, r=40)
        )
        
        return fig
    
    def display_market_share_metrics(self, data: pd.DataFrame, value_col: str):
        """Display market concentration and distribution metrics"""
        metrics = self.calculate_market_concentration(data, value_col)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="concentration-card">
                <h3>Market Concentration</h3>
                <p><strong>HHI:</strong> {metrics['hhi']:.0f}</p>
                <p><strong>CR4:</strong> {metrics['cr4']:.1f}%</p>
                <p><strong>CR8:</strong> {metrics['cr8']:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="share-evolution-card">
                <h3>Distribution Metrics</h3>
                <p><strong>Gini Coefficient:</strong> {metrics['gini']:.3f}</p>
                <p><strong>Shannon Entropy:</strong> {metrics['entropy']:.2f}</p>
                <p><strong>Effective Competitors:</strong> {2**metrics['entropy']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            concentration_level = "High" if metrics['hhi'] > 2500 else "Moderate" if metrics['hhi'] > 1500 else "Low"
            competition_level = "Intense" if metrics['entropy'] > 3 else "Moderate" if metrics['entropy'] > 2 else "Limited"
            
            st.markdown(f"""
            <div class="competitive-card">
                <h3>Market Structure</h3>
                <p><strong>Concentration:</strong> {concentration_level}</p>
                <p><strong>Competition:</strong> {competition_level}</p>
                <p><strong>Players:</strong> {len(data)}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def generate_insights(self, data: pd.DataFrame, entity_col: str, 
                         value_col: str, year_col: str = None) -> List[str]:
        """Generate actionable insights from market share data"""
        insights = []
        
        # Market dominance insights
        total_market = data[value_col].sum()
        top_player = data.loc[data[value_col].idxmax()]
        top_share = (top_player[value_col] / total_market) * 100
        
        if top_share > 50:
            insights.append(f"🏆 {top_player[entity_col]} dominates with {top_share:.1f}% market share")
        elif top_share > 30:
            insights.append(f"📈 {top_player[entity_col]} leads the market with {top_share:.1f}% share")
        
        # Competition insights
        top_3_share = data.nlargest(3, value_col)[value_col].sum() / total_market * 100
        if top_3_share > 80:
            insights.append("🎯 Market is highly concentrated among top 3 players")
        elif top_3_share < 50:
            insights.append("🌟 Market shows healthy competition with distributed shares")
        
        # Size distribution insights
        small_players = len(data[data[value_col] < total_market * 0.01])
        if small_players > len(data) * 0.5:
            insights.append(f"🔍 {small_players} niche players with <1% market share each")
        
        # Growth insights (if year data available)
        if year_col and year_col in data.columns:
            years = sorted(data[year_col].unique())
            if len(years) > 1:
                latest_year = max(years)
                previous_year = years[-2] if len(years) >= 2 else years[0]
                
                latest_data = data[data[year_col] == latest_year]
                previous_data = data[data[year_col] == previous_year]
                
                if len(latest_data) > 0 and len(previous_data) > 0:
                    latest_total = latest_data[value_col].sum()
                    previous_total = previous_data[value_col].sum()
                    
                    if latest_total > previous_total:
                        growth = ((latest_total / previous_total) - 1) * 100
                        insights.append(f"📊 Overall market grew {growth:.1f}% from {previous_year} to {latest_year}")
        
        return insights
    
    def render_market_share_dashboard(self, data: pd.DataFrame, entity_col: str, 
                                    value_col: str, year_col: str = None):
        """Render complete market share dashboard"""
        self.apply_professional_styling()
        self.create_header()
        
        if data.empty:
            st.error("No data available for market share analysis")
            return
        
        # Sidebar controls
        st.sidebar.header("📊 Analysis Controls")
        
        # Year filter if available
        if year_col and year_col in data.columns:
            years = sorted(data[year_col].unique())
            selected_year = st.sidebar.selectbox("Select Year", years, index=len(years)-1)
            filtered_data = data[data[year_col] == selected_year]
        else:
            filtered_data = data
            selected_year = "All Years"
        
        # Entity filter
        entities = sorted(filtered_data[entity_col].unique())
        if len(entities) > 20:
            top_n = st.sidebar.slider("Show Top N Entities", 5, min(50, len(entities)), 15)
            top_entities = filtered_data.nlargest(top_n, value_col)[entity_col].tolist()
            display_data = filtered_data[filtered_data[entity_col].isin(top_entities)]
        else:
            display_data = filtered_data
        
        # Chart type selection
        chart_types = st.sidebar.multiselect(
            "Select Chart Types",
            ["Pie Chart", "Treemap", "Share Evolution", "Competitive Matrix", "Waterfall"],
            default=["Pie Chart", "Treemap"]
        )
        
        # Display metrics
        st.subheader(f"📈 Market Overview - {selected_year}")
        self.display_market_share_metrics(display_data, value_col)
        
        # Generate insights
        insights = self.generate_insights(display_data, entity_col, value_col, year_col)
        if insights:
            st.markdown("### 💡 Key Insights")
            for insight in insights:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Display charts
        chart_cols = st.columns(2)
        chart_idx = 0
        
        if "Pie Chart" in chart_types:
            with chart_cols[chart_idx % 2]:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = self.create_interactive_pie_chart(
                    display_data, entity_col, value_col, 
                    f"Market Share Distribution - {selected_year}"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                chart_idx += 1
        
        if "Treemap" in chart_types:
            with chart_cols[chart_idx % 2]:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = self.create_treemap_visualization(
                    display_data, entity_col, value_col,
                    f"Market Share Treemap - {selected_year}"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                chart_idx += 1
        
        if "Share Evolution" in chart_types and year_col and year_col in data.columns:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = self.create_share_evolution_chart(data, entity_col, year_col, value_col)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if "Competitive Matrix" in chart_types and len(display_data) > 1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig = self.create_competitive_positioning_matrix(
                data if year_col in data.columns else display_data,
                entity_col, value_col, 'growth_rate'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary table
        st.subheader("📋 Detailed Breakdown")
        summary_data = display_data.copy()
        total = summary_data[value_col].sum()
        summary_data['Market Share (%)'] = (summary_data[value_col] / total * 100).round(2)
        summary_data['Cumulative Share (%)'] = summary_data['Market Share (%)'].cumsum().round(2)
        
        summary_display = summary_data[[entity_col, value_col, 'Market Share (%)', 'Cumulative Share (%)']].sort_values(value_col, ascending=False)
        st.dataframe(summary_display, use_container_width=True)


def main():
    """Main function for testing"""
    visualizer = MarketShareVisualizer()
    
    # Sample data
    sample_data = pd.DataFrame({
        'Country': ['USA', 'China', 'Germany', 'Japan', 'UK', 'France', 'India', 'Brazil', 'Canada', 'Australia'],
        'Market_Size': [1000, 800, 400, 350, 300, 250, 200, 150, 120, 100],
        'Year': [2023] * 10
    })
    
    visualizer.render_market_share_dashboard(sample_data, 'Country', 'Market_Size', 'Year')

if __name__ == "__main__":
    main()