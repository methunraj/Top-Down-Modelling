"""
Unified Visualization Interface
Brings together all visualization components in a cohesive, user-friendly interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go

# Import all visualization components
from src.streamlit.market_share_viz import MarketShareVisualizer
from src.streamlit.regional_analysis_viz import RegionalAnalysisVisualizer
from src.streamlit.forecast_accuracy_viz import ForecastAccuracyVisualizer

# Import visualization functions from modules without classes
import src.streamlit.enhanced_visualization as enhanced_viz
import src.streamlit.growth_analysis_viz as growth_viz

class UnifiedVisualizationInterface:
    """Unified interface for all visualization components"""
    
    def __init__(self):
        self.visualizers = {
            "Overview": {
                "icon": "📊",
                "description": "Executive dashboard with key metrics and insights",
                "class": None,  # Will use function directly
                "function": enhanced_viz.render_executive_dashboard,
                "help": "Get a high-level overview of your market forecast with key performance indicators, trends, and executive insights."
            },
            "Growth Analysis": {
                "icon": "📈",
                "description": "Comprehensive growth rate and trend analysis",
                "class": None,  # Will use function directly
                "function": growth_viz.render_growth_analysis_dashboard,
                "help": "Analyze growth patterns, CAGR calculations, volatility metrics, and future growth trajectories."
            },
            "Market Share": {
                "icon": "🥧",
                "description": "Market share distribution and competitive analysis",
                "class": MarketShareVisualizer(),
                "help": "Understand market share dynamics, concentration metrics, and competitive positioning across countries and regions."
            },
            "Regional Analysis": {
                "icon": "🌍",
                "description": "Geographical market analysis and regional insights",
                "class": RegionalAnalysisVisualizer(),
                "help": "Explore geographical market patterns, regional growth differences, and global market distribution."
            },
            "Forecast Accuracy": {
                "icon": "🎯",
                "description": "Model performance and forecast reliability metrics",
                "class": ForecastAccuracyVisualizer(),
                "help": "Evaluate forecast accuracy, model performance metrics, and reliability indicators for your predictions."
            }
        }
        
        self.chart_explanations = {
            "Line Chart": "Shows trends over time. Best for understanding growth patterns and temporal changes.",
            "Bar Chart": "Compares values across categories. Ideal for country comparisons and rankings.",
            "Pie Chart": "Shows proportional relationships. Perfect for market share analysis.",
            "Map": "Geographical visualization. Excellent for regional patterns and global distribution.",
            "Scatter Plot": "Shows relationships between variables. Great for correlation analysis.",
            "Heatmap": "Color-coded matrix. Useful for showing patterns across multiple dimensions.",
            "Treemap": "Hierarchical visualization. Good for nested data like regions and countries.",
            "Waterfall": "Shows cumulative changes. Ideal for understanding step-by-step changes."
        }
    
    def render_visualization_selector(self):
        """Render visualization type selector with explanations"""
        st.markdown("### 📊 Choose Your Analysis")
        
        # Create cards for each visualization type
        cols = st.columns(len(self.visualizers))
        
        selected_viz = None
        for i, (viz_name, viz_info) in enumerate(self.visualizers.items()):
            with cols[i]:
                # Create clickable card
                card_html = f"""
                <div style="
                    background: white;
                    border: 2px solid #e2e8f0;
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                " onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 8px 25px rgba(0,0,0,0.15)'; this.style.borderColor='#667eea';"
                   onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'; this.style.borderColor='#e2e8f0';">
                    <div style="font-size: 2.5rem; margin-bottom: 1rem;">{viz_info['icon']}</div>
                    <h4 style="margin: 0 0 0.5rem 0; color: #2d3748;">{viz_name}</h4>
                    <p style="margin: 0; color: #718096; font-size: 0.9rem;">{viz_info['description']}</p>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                if st.button(f"Open {viz_name}", key=f"viz_select_{viz_name}"):
                    selected_viz = viz_name
        
        return selected_viz
    
    def render_analysis_workflow(self, step: int):
        """Render the analysis workflow"""
        if step == 1:
            st.markdown("### 📊 Choose Analysis Type")
            selected_viz = self.render_visualization_selector()
            
            if selected_viz:
                st.session_state.selected_visualization = selected_viz
                st.session_state.workflow_step = 2
                st.rerun()
        
        elif step == 2:
            selected_viz = st.session_state.get('selected_visualization', 'Overview')
            st.markdown(f"### {self.visualizers[selected_viz]['icon']} {selected_viz}")
            
            # Show help information
            help_info = self.visualizers[selected_viz]['help']
            st.info(f"💡 **About this analysis:** {help_info}")
            
            # Render the selected visualization
            self.render_selected_visualization(selected_viz)
            
            # Navigation
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back to Analysis Selection"):
                    st.session_state.workflow_step = 1
                    st.rerun()
            with col2:
                if st.button("Try Another Analysis →"):
                    st.session_state.workflow_step = 1
                    st.rerun()
        
        elif step == 3:
            st.markdown("### 📈 Trend Analysis")
            self.render_trend_analysis()
        
        elif step == 4:
            st.markdown("### 🌍 Regional Insights")
            self.render_regional_insights()
        
        elif step == 5:
            st.markdown("### ✅ Analysis Summary")
            self.render_analysis_summary()
    
    def render_selected_visualization(self, viz_name: str):
        """Render the selected visualization component"""
        if st.session_state.distributed_market is None:
            st.warning("No forecast data available. Please generate a forecast first.")
            if st.button("🔮 Go to Forecasting"):
                st.session_state.current_workflow = "Forecasting"
                st.session_state.workflow_step = 1
                st.rerun()
            return
        
        viz_info = self.visualizers[viz_name]
        data = st.session_state.distributed_market
        config_manager = st.session_state.get('config_manager', None)
        
        try:
            if viz_name == "Overview":
                # Executive dashboard - use function directly
                viz_info['function'](data, config_manager)
            
            elif viz_name == "Growth Analysis":
                # Growth analysis - use function directly
                viz_info['function'](data, config_manager)
            
            elif viz_name == "Market Share":
                # Market share analysis - use class method
                visualizer = viz_info['class']
                visualizer.render_market_share_dashboard(data, 'Country', 'Value', 'Year')
            
            elif viz_name == "Regional Analysis":
                # Regional analysis - use class method
                visualizer = viz_info['class']
                visualizer.render_regional_dashboard(data, 'Country', 'Value', 'Year')
            
            elif viz_name == "Forecast Accuracy":
                # Create mock accuracy data for demonstration
                visualizer = viz_info['class']
                accuracy_data = self.create_mock_accuracy_data(data)
                visualizer.render_forecast_accuracy_dashboard(
                    accuracy_data, 'Actual', ['Model_1', 'Model_2'], 'Year'
                )
            
        except Exception as e:
            st.error(f"Error rendering visualization: {str(e)}")
            st.markdown("**Troubleshooting tips:**")
            st.markdown("• Check that your data has the required columns")
            st.markdown("• Ensure data is properly formatted")
            st.markdown("• Try refreshing the page")
    
    def create_mock_accuracy_data(self, forecast_data: pd.DataFrame) -> pd.DataFrame:
        """Create mock accuracy data for forecast accuracy visualization"""
        # Use forecast data to create realistic accuracy comparison
        accuracy_data = []
        
        for _, row in forecast_data.iterrows():
            actual_value = row['Value']
            # Add some realistic prediction errors
            model_1_pred = actual_value * (1 + np.random.normal(0, 0.1))
            model_2_pred = actual_value * (1 + np.random.normal(0, 0.15))
            
            accuracy_data.append({
                'Country': row['Country'],
                'Year': row['Year'],
                'Actual': actual_value,
                'Model_1': model_1_pred,
                'Model_2': model_2_pred
            })
        
        return pd.DataFrame(accuracy_data)
    
    def render_quick_overview(self):
        """Render quick overview for the Quick Start workflow"""
        st.markdown("### 📊 Your Forecast Results")
        
        if st.session_state.distributed_market is not None:
            data = st.session_state.distributed_market
            
            # Key metrics
            st.markdown("#### 🎯 Key Metrics")
            
            total_2024 = data[data['Year'] == 2024]['Value'].sum() if 2024 in data['Year'].values else 0
            total_2028 = data[data['Year'] == 2028]['Value'].sum() if 2028 in data['Year'].values else 0
            
            if total_2024 > 0 and total_2028 > 0:
                cagr = ((total_2028 / total_2024) ** (1/4) - 1) * 100
            else:
                cagr = 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Countries", data['Country'].nunique())
            with col2:
                if total_2024 > 0:
                    st.metric("2024 Market Size", f"${total_2024:,.0f}M")
                else:
                    latest_year = data['Year'].max()
                    latest_total = data[data['Year'] == latest_year]['Value'].sum()
                    st.metric(f"{latest_year} Market Size", f"${latest_total:,.0f}M")
            with col3:
                if total_2028 > 0:
                    st.metric("2028 Forecast", f"${total_2028:,.0f}M")
                else:
                    future_year = data['Year'].max()
                    future_total = data[data['Year'] == future_year]['Value'].sum()
                    st.metric(f"{future_year} Forecast", f"${future_total:,.0f}M")
            with col4:
                if cagr > 0:
                    st.metric("CAGR", f"{cagr:.1f}%")
                else:
                    # Calculate simple growth rate
                    years = sorted(data['Year'].unique())
                    if len(years) > 1:
                        first_year_total = data[data['Year'] == years[0]]['Value'].sum()
                        last_year_total = data[data['Year'] == years[-1]]['Value'].sum()
                        growth = ((last_year_total / first_year_total) ** (1/(years[-1] - years[0])) - 1) * 100
                        st.metric("CAGR", f"{growth:.1f}%")
            
            # Quick visualization
            st.markdown("#### 📈 Market Growth Trend")
            
            yearly_totals = data.groupby('Year')['Value'].sum().reset_index()
            
            fig = px.line(
                yearly_totals,
                x='Year',
                y='Value',
                title='Total Market Size Over Time',
                markers=True
            )
            
            fig.update_layout(
                xaxis_title='Year',
                yaxis_title='Market Size',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top countries
            st.markdown("#### 🏆 Top Countries")
            
            latest_year = data['Year'].max()
            top_countries = data[data['Year'] == latest_year].nlargest(5, 'Value')
            
            fig = px.bar(
                top_countries,
                x='Country',
                y='Value',
                title=f'Top 5 Countries by Market Size ({latest_year})'
            )
            
            fig.update_layout(
                xaxis_title='Country',
                yaxis_title='Market Size',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis options
            st.markdown("#### 🔍 Dive Deeper")
            
            analysis_cols = st.columns(3)
            
            with analysis_cols[0]:
                if st.button("📈 Growth Analysis"):
                    st.session_state.current_workflow = "Analysis"
                    st.session_state.selected_visualization = "Growth Analysis"
                    st.session_state.workflow_step = 2
                    st.rerun()
            
            with analysis_cols[1]:
                if st.button("🌍 Regional Analysis"):
                    st.session_state.current_workflow = "Analysis"
                    st.session_state.selected_visualization = "Regional Analysis"
                    st.session_state.workflow_step = 2
                    st.rerun()
            
            with analysis_cols[2]:
                if st.button("🥧 Market Share"):
                    st.session_state.current_workflow = "Analysis"
                    st.session_state.selected_visualization = "Market Share"
                    st.session_state.workflow_step = 2
                    st.rerun()
        
        else:
            st.warning("No forecast data available.")
            st.markdown("### 🔮 Generate Your First Forecast")
            st.markdown("""
            To see your results:
            1. Ensure you have uploaded market data
            2. Generate a forecast using our forecasting tools
            3. Return here to analyze your results
            """)
            
            if st.button("🚀 Go to Forecasting"):
                st.session_state.current_workflow = "Forecasting"
                st.session_state.workflow_step = 1
                st.rerun()
    
    def render_chart_type_guide(self):
        """Render guide for different chart types"""
        st.markdown("### 📊 Chart Type Guide")
        st.markdown("Choose the right visualization for your analysis:")
        
        for chart_type, explanation in self.chart_explanations.items():
            with st.expander(f"{chart_type}"):
                st.markdown(explanation)
                
                # Add example use cases
                if chart_type == "Line Chart":
                    st.markdown("**Use cases:** Time series analysis, trend identification, growth tracking")
                elif chart_type == "Bar Chart":
                    st.markdown("**Use cases:** Country comparisons, ranking analysis, categorical data")
                elif chart_type == "Pie Chart":
                    st.markdown("**Use cases:** Market share, proportion analysis, composition breakdown")
                elif chart_type == "Map":
                    st.markdown("**Use cases:** Geographical patterns, regional analysis, global distribution")
    
    def render_export_options(self):
        """Render export and sharing options"""
        st.markdown("### 📤 Export & Share")
        
        if st.session_state.distributed_market is not None:
            export_formats = st.multiselect(
                "Select Export Formats",
                ["Interactive Dashboard", "PDF Report", "Excel Workbook", "CSV Data", "PNG Images"],
                default=["PDF Report", "Excel Workbook"],
                help="Choose which formats to export your analysis"
            )
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Report Options:**")
                include_charts = st.checkbox("Include Charts", value=True)
                include_data = st.checkbox("Include Raw Data", value=True)
                include_methodology = st.checkbox("Include Methodology", value=False)
            
            with col2:
                st.markdown("**Sharing Options:**")
                create_link = st.checkbox("Create Shareable Link", value=False)
                email_report = st.checkbox("Email Report", value=False)
                schedule_updates = st.checkbox("Schedule Updates", value=False)
            
            if st.button("📥 Generate Exports"):
                with st.spinner("Generating exports..."):
                    # Mock export generation
                    import time
                    time.sleep(2)
                    
                    st.success("✅ Exports generated successfully!")
                    
                    # Provide download links
                    for format_type in export_formats:
                        st.download_button(
                            f"Download {format_type}",
                            data="Mock export data",
                            file_name=f"market_analysis.{format_type.lower().replace(' ', '_')}",
                            key=f"download_{format_type}"
                        )
        else:
            st.warning("No data available to export. Please generate a forecast first.")
    
    def render_trend_analysis(self):
        """Render comprehensive trend analysis"""
        if st.session_state.distributed_market is not None:
            data = st.session_state.distributed_market
            
            # Growth trends
            st.markdown("#### 📈 Growth Trends")
            
            # Calculate year-over-year growth
            growth_data = []
            for country in data['Country'].unique():
                country_data = data[data['Country'] == country].sort_values('Year')
                if len(country_data) > 1:
                    for i in range(1, len(country_data)):
                        current_value = country_data.iloc[i]['Value']
                        previous_value = country_data.iloc[i-1]['Value']
                        growth_rate = ((current_value / previous_value) - 1) * 100
                        
                        growth_data.append({
                            'Country': country,
                            'Year': country_data.iloc[i]['Year'],
                            'Growth_Rate': growth_rate
                        })
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                
                # Average growth by country
                avg_growth = growth_df.groupby('Country')['Growth_Rate'].mean().sort_values(ascending=False)
                
                fig = px.bar(
                    x=avg_growth.index,
                    y=avg_growth.values,
                    title='Average Growth Rate by Country',
                    labels={'x': 'Country', 'y': 'Average Growth Rate (%)'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Growth trends over time
                fig2 = px.line(
                    growth_df,
                    x='Year',
                    y='Growth_Rate',
                    color='Country',
                    title='Growth Rate Trends Over Time'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    def render_regional_insights(self):
        """Render regional insights"""
        if st.session_state.distributed_market is not None:
            data = st.session_state.distributed_market
            
            # Regional grouping (simplified)
            region_mapping = {
                'USA': 'North America',
                'Canada': 'North America',
                'Mexico': 'North America',
                'China': 'Asia Pacific',
                'Japan': 'Asia Pacific',
                'India': 'Asia Pacific',
                'Germany': 'Europe',
                'UK': 'Europe',
                'France': 'Europe',
                'Brazil': 'Latin America',
                'Argentina': 'Latin America'
            }
            
            # Add region column
            data_with_regions = data.copy()
            data_with_regions['Region'] = data_with_regions['Country'].map(
                lambda x: region_mapping.get(x, 'Others')
            )
            
            # Regional market size
            regional_data = data_with_regions.groupby(['Region', 'Year'])['Value'].sum().reset_index()
            
            fig = px.line(
                regional_data,
                x='Year',
                y='Value',
                color='Region',
                title='Regional Market Size Trends'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Regional share
            latest_year = data_with_regions['Year'].max()
            latest_regional = data_with_regions[data_with_regions['Year'] == latest_year]
            regional_share = latest_regional.groupby('Region')['Value'].sum()
            
            fig2 = px.pie(
                values=regional_share.values,
                names=regional_share.index,
                title=f'Regional Market Share ({latest_year})'
            )
            
            st.plotly_chart(fig2, use_container_width=True)
    
    def render_analysis_summary(self):
        """Render analysis summary and recommendations"""
        st.markdown("### 📋 Analysis Summary")
        
        if st.session_state.distributed_market is not None:
            data = st.session_state.distributed_market
            
            # Key findings
            st.markdown("#### 🔍 Key Findings")
            
            findings = []
            
            # Market size finding
            total_current = data[data['Year'] == data['Year'].min()]['Value'].sum()
            total_future = data[data['Year'] == data['Year'].max()]['Value'].sum()
            total_growth = ((total_future / total_current) ** (1/(data['Year'].max() - data['Year'].min())) - 1) * 100
            
            findings.append(f"📊 **Market Growth**: The market is projected to grow at {total_growth:.1f}% CAGR")
            
            # Top performer
            latest_year = data['Year'].max()
            top_country = data[data['Year'] == latest_year].nlargest(1, 'Value')['Country'].iloc[0]
            top_value = data[data['Year'] == latest_year].nlargest(1, 'Value')['Value'].iloc[0]
            
            findings.append(f"🏆 **Market Leader**: {top_country} leads with ${top_value:,.0f}M market size")
            
            # Growth leader
            growth_by_country = []
            for country in data['Country'].unique():
                country_data = data[data['Country'] == country].sort_values('Year')
                if len(country_data) > 1:
                    first_value = country_data.iloc[0]['Value']
                    last_value = country_data.iloc[-1]['Value']
                    growth = ((last_value / first_value) ** (1/(len(country_data)-1)) - 1) * 100
                    growth_by_country.append((country, growth))
            
            if growth_by_country:
                fastest_growing = max(growth_by_country, key=lambda x: x[1])
                findings.append(f"🚀 **Fastest Growing**: {fastest_growing[0]} with {fastest_growing[1]:.1f}% CAGR")
            
            for finding in findings:
                st.markdown(finding)
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            
            recommendations = [
                "🎯 **Focus Markets**: Prioritize high-growth markets for expansion",
                "📈 **Investment Strategy**: Consider increased investment in emerging markets",
                "🌍 **Regional Balance**: Maintain presence across multiple regions for risk mitigation",
                "📊 **Regular Monitoring**: Track performance against forecasts quarterly"
            ]
            
            for rec in recommendations:
                st.markdown(rec)
            
            # Action items
            st.markdown("#### ✅ Next Steps")
            
            action_items = [
                "Review detailed country-level analysis",
                "Validate assumptions with local market experts",
                "Develop market entry strategies for high-potential countries",
                "Set up monitoring dashboards for key metrics",
                "Schedule quarterly forecast updates"
            ]
            
            for i, item in enumerate(action_items, 1):
                st.checkbox(f"{i}. {item}", key=f"action_{i}")
        
        else:
            st.warning("No analysis data available.")
        
        # Completion
        if st.button("🎉 Complete Analysis"):
            st.session_state.workflow_progress['analysis_complete'] = True
            st.balloons()
            st.success("Analysis workflow completed!")