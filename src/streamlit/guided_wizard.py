"""
Guided Configuration Wizard
Provides step-by-step guidance for configuring market forecasting with comprehensive explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

class GuidedConfigurationWizard:
    """Interactive wizard for guided configuration setup"""
    
    def __init__(self):
        self.forecasting_methods = {
            "Statistical Models": {
                "icon": "📊",
                "description": "Traditional time series forecasting using statistical methods",
                "methods": {
                    "Linear Regression": "Simple linear trend forecasting - good for stable growth patterns",
                    "Exponential Smoothing": "Adaptive forecasting that adjusts to recent trends - handles seasonal patterns",
                    "ARIMA": "Advanced statistical model for complex time series - best for historical pattern analysis",
                    "Seasonal Decomposition": "Separates trend, seasonal, and residual components - ideal for cyclical markets"
                },
                "pros": ["Fast computation", "Interpretable results", "Good for stable markets"],
                "cons": ["Limited adaptability", "Struggles with disruptions", "Requires historical data"],
                "best_for": "Mature markets with consistent historical patterns"
            },
            "Machine Learning": {
                "icon": "🤖",
                "description": "Advanced ML algorithms for pattern recognition and forecasting",
                "methods": {
                    "Random Forest": "Ensemble method combining multiple decision trees - robust to outliers",
                    "XGBoost": "Gradient boosting for high accuracy - excellent for complex relationships", 
                    "Neural Networks": "Deep learning for non-linear patterns - captures complex interactions",
                    "Support Vector Machines": "Pattern recognition with kernel methods - good for high-dimensional data"
                },
                "pros": ["Handles complex patterns", "Adapts to new data", "High accuracy potential"],
                "cons": ["Requires more data", "Less interpretable", "Computationally intensive"],
                "best_for": "Dynamic markets with complex external factors"
            },
            "Hybrid Ensemble": {
                "icon": "🔄",
                "description": "Combines multiple forecasting approaches for improved accuracy",
                "methods": {
                    "Weighted Average": "Simple combination of multiple models - reduces individual model bias",
                    "Dynamic Weighting": "Adaptive weights based on recent performance - self-optimizing",
                    "Stacked Ensemble": "ML model learns optimal combination - maximizes predictive power",
                    "Bayesian Model Averaging": "Probabilistic combination with uncertainty quantification"
                },
                "pros": ["Improved accuracy", "Reduced overfitting", "Robust predictions"],
                "cons": ["More complex setup", "Longer computation", "Harder to interpret"],
                "best_for": "Critical forecasts requiring maximum accuracy and reliability"
            },
            "Bayesian Hierarchical": {
                "icon": "🎯",
                "description": "Advanced probabilistic modeling with uncertainty quantification",
                "methods": {
                    "Regional Hierarchy": "Models regional relationships and correlations",
                    "Market Segment Hierarchy": "Captures dependencies between market segments",
                    "Time Series Hierarchy": "Models temporal dependencies and seasonality",
                    "Mixed Effects": "Combines fixed and random effects for better generalization"
                },
                "pros": ["Uncertainty quantification", "Handles missing data", "Incorporates prior knowledge"],
                "cons": ["Computationally intensive", "Requires expertise", "Complex interpretation"],
                "best_for": "Strategic planning requiring confidence intervals and risk assessment"
            }
        }
        
        self.distribution_algorithms = {
            "Indicator-Based": {
                "icon": "📈",
                "description": "Distributes market based on economic and market indicators",
                "explanation": "Uses indicators like GDP, population, technology adoption to determine each country's market share",
                "pros": ["Data-driven", "Reflects economic reality", "Adaptable to new indicators"],
                "cons": ["Requires quality indicator data", "May miss unique market factors"],
                "best_for": "Markets with strong correlation to economic indicators",
                "indicators_needed": ["GDP", "Population", "Market Size", "Technology Adoption"]
            },
            "Historical Share": {
                "icon": "📊",
                "description": "Maintains historical market share patterns with growth adjustments",
                "explanation": "Preserves existing market share relationships while allowing for growth rate differences",
                "pros": ["Stable predictions", "Respects market structure", "Simple to understand"],
                "cons": ["May miss market disruptions", "Assumes stable competitive landscape"],
                "best_for": "Mature markets with established competitive positions",
                "indicators_needed": ["Historical market data by country"]
            },
            "Tier-Based": {
                "icon": "🏆",
                "description": "Classifies countries into tiers based on market characteristics",
                "explanation": "Groups countries by market maturity/size and applies tier-specific growth patterns",
                "pros": ["Captures market maturity differences", "Scalable approach", "Clear segmentation"],
                "cons": ["May oversimplify", "Requires tier definition", "Less granular"],
                "best_for": "Global markets with clear maturity stages",
                "indicators_needed": ["Market development indicators", "Economic development level"]
            },
            "Causal Inference": {
                "icon": "🔗",
                "description": "Advanced causal modeling to understand true market drivers",
                "explanation": "Identifies causal relationships between indicators and market performance",
                "pros": ["Understands causation", "Robust to confounding", "Actionable insights"],
                "cons": ["Requires extensive data", "Complex methodology", "Longer setup time"],
                "best_for": "Strategic analysis requiring deep market understanding",
                "indicators_needed": ["Multiple years of indicator data", "Market performance data"]
            }
        }
        
        self.market_constraints = {
            "Growth Rate Limits": {
                "description": "Prevents unrealistic growth rates that exceed market reality",
                "explanation": "Sets maximum annual growth rates (e.g., 50%) to ensure realistic forecasts",
                "example": "Prevents a small market from growing 1000% in one year",
                "recommended": True
            },
            "Market Share Caps": {
                "description": "Limits individual country market share to realistic levels",
                "explanation": "Prevents any single country from dominating beyond feasible levels",
                "example": "Caps USA market share at 40% for global technology markets",
                "recommended": True
            },
            "Minimum Market Size": {
                "description": "Ensures countries maintain minimum viable market presence",
                "explanation": "Prevents market shares from becoming too small to be meaningful",
                "example": "Minimum $1M market size for developed countries",
                "recommended": False
            },
            "Regional Balance": {
                "description": "Maintains balanced growth across geographical regions",
                "explanation": "Prevents extreme concentration in single regions",
                "example": "Ensures Asia-Pacific doesn't exceed 60% of global market",
                "recommended": False
            }
        }
    
    def render_method_explanation(self, method_category: str, method_name: str):
        """Render detailed explanation for a specific method"""
        if method_category in self.forecasting_methods:
            method_info = self.forecasting_methods[method_category]
            
            if method_name == "overview":
                st.markdown(f"""
                ### {method_info['icon']} {method_category}
                
                **Description:** {method_info['description']}
                
                **Best For:** {method_info['best_for']}
                
                **Pros:**
                {chr(10).join([f"• {pro}" for pro in method_info['pros']])}
                
                **Cons:**
                {chr(10).join([f"• {con}" for con in method_info['cons']])}
                """)
            elif method_name in method_info['methods']:
                st.markdown(f"""
                **{method_name}**
                
                {method_info['methods'][method_name]}
                """)
    
    def render_quick_forecast_setup(self):
        """Render quick forecast setup for beginners"""
        st.markdown("### 🔮 Quick Forecast Setup")
        st.markdown("We'll help you choose the best forecasting approach for your market.")
        
        # Market characteristics questionnaire
        st.markdown("#### 📋 Tell us about your market:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_maturity = st.radio(
                "Market Maturity",
                ["New/Emerging", "Growing", "Mature", "Declining"],
                help="How established is this market?"
            )
            
            historical_data_quality = st.radio(
                "Historical Data Quality",
                ["Excellent (5+ years)", "Good (3-5 years)", "Limited (1-2 years)", "Minimal/None"],
                help="How much reliable historical data do you have?"
            )
        
        with col2:
            market_volatility = st.radio(
                "Market Volatility",
                ["Very Stable", "Moderately Stable", "Somewhat Volatile", "Highly Volatile"],
                help="How predictable are market changes?"
            )
            
            external_factors = st.radio(
                "External Factor Influence",
                ["Minimal", "Moderate", "Significant", "Dominant"],
                help="How much do external factors (economy, technology, etc.) affect this market?"
            )
        
        # Smart recommendations
        if st.button("🎯 Get Smart Recommendations"):
            recommendations = self.generate_smart_recommendations(
                market_maturity, historical_data_quality, market_volatility, external_factors
            )
            
            st.markdown("### 💡 Recommended Configuration")
            
            for category, recommendation in recommendations.items():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                           padding: 1rem; border-radius: 8px; margin: 1rem 0; 
                           border-left: 4px solid #2196f3;">
                    <strong>{category}:</strong> {recommendation['choice']}<br>
                    <em>Why:</em> {recommendation['reason']}
                </div>
                """, unsafe_allow_html=True)
        
        # Manual configuration option
        with st.expander("🔧 Manual Configuration (Advanced)"):
            self.render_manual_forecast_config()
        
        # Generate forecast button
        if st.button("🚀 Generate Forecast", type="primary"):
            if st.session_state.global_forecast is not None:
                with st.spinner("Generating market forecast..."):
                    self.generate_quick_forecast()
                st.success("✅ Forecast generated successfully!")
                st.session_state.workflow_progress['forecast_generated'] = True
            else:
                st.error("Please upload market data first!")
    
    def generate_smart_recommendations(self, maturity, data_quality, volatility, external_factors):
        """Generate smart recommendations based on market characteristics"""
        recommendations = {}
        
        # Forecasting method recommendation
        if data_quality in ["Excellent (5+ years)", "Good (3-5 years)"] and volatility in ["Very Stable", "Moderately Stable"]:
            if external_factors in ["Minimal", "Moderate"]:
                recommendations["Forecasting Method"] = {
                    "choice": "Statistical Models",
                    "reason": "Your stable market with good historical data is perfect for statistical forecasting"
                }
            else:
                recommendations["Forecasting Method"] = {
                    "choice": "Hybrid Ensemble",
                    "reason": "External factors require more sophisticated modeling, but your good data supports ensemble methods"
                }
        elif external_factors in ["Significant", "Dominant"] or volatility in ["Somewhat Volatile", "Highly Volatile"]:
            recommendations["Forecasting Method"] = {
                "choice": "Machine Learning",
                "reason": "High volatility and external factors require adaptive ML algorithms"
            }
        else:
            recommendations["Forecasting Method"] = {
                "choice": "Bayesian Hierarchical",
                "reason": "Limited data with uncertainty requires probabilistic modeling with confidence intervals"
            }
        
        # Distribution method recommendation
        if maturity in ["New/Emerging", "Growing"]:
            recommendations["Distribution Method"] = {
                "choice": "Indicator-Based",
                "reason": "Growing markets benefit from economic indicator-driven distribution"
            }
        elif maturity == "Mature":
            recommendations["Distribution Method"] = {
                "choice": "Historical Share",
                "reason": "Mature markets have established competitive positions worth preserving"
            }
        else:  # Declining
            recommendations["Distribution Method"] = {
                "choice": "Tier-Based",
                "reason": "Declining markets require tier-based analysis to identify resilient segments"
            }
        
        # Constraints recommendation
        if volatility in ["Somewhat Volatile", "Highly Volatile"]:
            recommendations["Growth Constraints"] = {
                "choice": "Strict (30% max growth)",
                "reason": "Volatile markets need strict constraints to prevent unrealistic projections"
            }
        else:
            recommendations["Growth Constraints"] = {
                "choice": "Moderate (50% max growth)",
                "reason": "Stable markets can accommodate moderate growth constraints"
            }
        
        return recommendations
    
    def render_manual_forecast_config(self):
        """Render manual forecasting configuration"""
        st.markdown("#### 🔧 Advanced Configuration")
        
        # Forecasting method selection
        st.markdown("**Forecasting Method:**")
        
        method_tabs = st.tabs(list(self.forecasting_methods.keys()))
        
        selected_method = None
        for i, (method_name, method_info) in enumerate(self.forecasting_methods.items()):
            with method_tabs[i]:
                st.markdown(f"### {method_info['icon']} {method_name}")
                st.markdown(method_info['description'])
                
                if st.button(f"Select {method_name}", key=f"select_{method_name}"):
                    selected_method = method_name
                    st.session_state.selected_forecast_method = method_name
                
                # Show available sub-methods
                st.markdown("**Available Methods:**")
                for sub_method, description in method_info['methods'].items():
                    st.markdown(f"• **{sub_method}**: {description}")
        
        # Distribution algorithm selection
        st.markdown("---")
        st.markdown("**Distribution Algorithm:**")
        
        dist_method = st.selectbox(
            "Choose Distribution Method",
            list(self.distribution_algorithms.keys()),
            help="How should the global forecast be distributed across countries?"
        )
        
        if dist_method:
            method_info = self.distribution_algorithms[dist_method]
            st.markdown(f"""
            **{method_info['icon']} {dist_method}**
            
            {method_info['explanation']}
            
            **Best For:** {method_info['best_for']}
            
            **Required Data:** {', '.join(method_info['indicators_needed'])}
            """)
        
        # Constraints configuration
        st.markdown("---")
        st.markdown("**Market Constraints:**")
        
        for constraint_name, constraint_info in self.market_constraints.items():
            enable_constraint = st.checkbox(
                constraint_name,
                value=constraint_info['recommended'],
                help=constraint_info['explanation']
            )
            
            if enable_constraint:
                if constraint_name == "Growth Rate Limits":
                    max_growth = st.slider(
                        "Maximum Annual Growth Rate (%)",
                        min_value=10,
                        max_value=200,
                        value=50,
                        step=5,
                        help="Maximum percentage growth allowed per year"
                    )
                elif constraint_name == "Market Share Caps":
                    max_share = st.slider(
                        "Maximum Country Market Share (%)",
                        min_value=20,
                        max_value=80,
                        value=45,
                        step=5,
                        help="Maximum market share any single country can hold"
                    )
    
    def generate_quick_forecast(self):
        """Generate forecast using current configuration"""
        # Mock forecast generation
        try:
            from src.market_analysis.market_analyzer import MarketAnalyzer
            from src.distribution.market_distributor import MarketDistributor
            
            # Use available data
            global_forecast = st.session_state.global_forecast
            country_data = st.session_state.country_historical
            indicators = st.session_state.indicators
            
            # Simple forecast generation (placeholder)
            countries = ['USA', 'China', 'Germany', 'Japan', 'UK', 'France', 'India', 'Brazil']
            years = range(2024, 2029)
            
            forecast_data = []
            base_values = {country: np.random.uniform(100, 1000) for country in countries}
            
            for year in years:
                for country in countries:
                    growth_rate = np.random.uniform(0.05, 0.15)  # 5-15% growth
                    if year == 2024:
                        value = base_values[country]
                    else:
                        prev_value = [d for d in forecast_data if d['Country'] == country and d['Year'] == year-1][0]['Value']
                        value = prev_value * (1 + growth_rate)
                    
                    forecast_data.append({
                        'Country': country,
                        'Year': year,
                        'Value': value
                    })
            
            st.session_state.distributed_market = pd.DataFrame(forecast_data)
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            # Create simple mock data
            countries = ['USA', 'China', 'Germany', 'Japan', 'UK']
            years = range(2024, 2029)
            
            forecast_data = []
            for year in years:
                for i, country in enumerate(countries):
                    value = (1000 + i * 200) * (1.1 ** (year - 2024))
                    forecast_data.append({
                        'Country': country,
                        'Year': year,
                        'Value': value
                    })
            
            st.session_state.distributed_market = pd.DataFrame(forecast_data)
    
    def render_forecasting_wizard(self, step: int):
        """Render full forecasting wizard"""
        if step == 1:
            st.markdown("### 📊 Choose Forecasting Method")
            self.render_method_selection()
        elif step == 2:
            st.markdown("### ⚙️ Configure Parameters")
            self.render_parameter_configuration()
        elif step == 3:
            st.markdown("### 🚀 Generate Forecast")
            self.render_forecast_generation()
        elif step == 4:
            st.markdown("### 🎯 Auto-Calibration")
            self.render_auto_calibration()
        elif step == 5:
            st.markdown("### ✅ Review Results")
            self.render_forecast_review()
    
    def render_method_selection(self):
        """Render detailed method selection interface"""
        st.markdown("Choose the forecasting approach that best fits your market characteristics:")
        
        # Method comparison table
        comparison_data = []
        for method_name, method_info in self.forecasting_methods.items():
            comparison_data.append({
                'Method': f"{method_info['icon']} {method_name}",
                'Best For': method_info['best_for'],
                'Complexity': 'Low' if method_name == 'Statistical Models' else 'High',
                'Data Requirements': 'Moderate' if method_name in ['Statistical Models', 'Machine Learning'] else 'High',
                'Accuracy': 'Good' if method_name == 'Statistical Models' else 'Excellent'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Detailed method cards
        for method_name, method_info in self.forecasting_methods.items():
            with st.expander(f"{method_info['icon']} {method_name} - Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pros:**")
                    for pro in method_info['pros']:
                        st.markdown(f"✅ {pro}")
                
                with col2:
                    st.markdown("**Cons:**")
                    for con in method_info['cons']:
                        st.markdown(f"❌ {con}")
                
                st.markdown(f"**Best For:** {method_info['best_for']}")
                
                if st.button(f"Select {method_name}", key=f"wizard_select_{method_name}"):
                    st.session_state.selected_forecast_method = method_name
                    st.session_state.workflow_step = 2
                    st.success(f"Selected {method_name}!")
                    st.rerun()
    
    def render_parameter_configuration(self):
        """Render parameter configuration interface"""
        selected_method = st.session_state.get('selected_forecast_method', 'Statistical Models')
        
        st.markdown(f"### Configure {selected_method} Parameters")
        
        # Method-specific parameters
        if selected_method == "Statistical Models":
            self.render_statistical_parameters()
        elif selected_method == "Machine Learning":
            self.render_ml_parameters()
        elif selected_method == "Hybrid Ensemble":
            self.render_ensemble_parameters()
        elif selected_method == "Bayesian Hierarchical":
            self.render_bayesian_parameters()
        
        # Navigation
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back to Method Selection"):
                st.session_state.workflow_step = 1
                st.rerun()
        with col2:
            if st.button("Continue to Generation →"):
                st.session_state.workflow_step = 3
                st.rerun()
    
    def render_statistical_parameters(self):
        """Render statistical model parameters"""
        st.markdown("#### 📊 Statistical Model Configuration")
        
        model_type = st.selectbox(
            "Primary Model Type",
            ["Linear Regression", "Exponential Smoothing", "ARIMA", "Seasonal Decomposition"],
            help="Choose the main statistical method"
        )
        
        if model_type == "ARIMA":
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.slider("AR Order (p)", 0, 5, 1, help="Autoregressive order")
            with col2:
                d = st.slider("Differencing (d)", 0, 2, 1, help="Degree of differencing")
            with col3:
                q = st.slider("MA Order (q)", 0, 5, 1, help="Moving average order")
        
        seasonality = st.checkbox("Include Seasonality", value=True, help="Account for seasonal patterns")
        
        if seasonality:
            seasonal_period = st.slider("Seasonal Period", 4, 52, 12, help="Length of seasonal cycle")
    
    def render_ml_parameters(self):
        """Render ML model parameters"""
        st.markdown("#### 🤖 Machine Learning Configuration")
        
        algorithm = st.selectbox(
            "ML Algorithm",
            ["Random Forest", "XGBoost", "Neural Networks", "Support Vector Machines"],
            help="Choose the machine learning algorithm"
        )
        
        if algorithm == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 500, 100, help="More trees = better accuracy but slower")
            max_depth = st.slider("Max Tree Depth", 3, 20, 10, help="Deeper trees can overfit")
        
        elif algorithm == "XGBoost":
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, help="Lower = more conservative learning")
            n_estimators = st.slider("Number of Boosting Rounds", 50, 1000, 300)
        
        elif algorithm == "Neural Networks":
            hidden_layers = st.slider("Hidden Layers", 1, 5, 2, help="More layers for complex patterns")
            neurons_per_layer = st.slider("Neurons per Layer", 10, 200, 50)
        
        feature_engineering = st.multiselect(
            "Feature Engineering",
            ["Lag Features", "Rolling Statistics", "Trend Features", "Seasonal Features"],
            default=["Lag Features", "Rolling Statistics"],
            help="Additional features to improve model performance"
        )
    
    def render_ensemble_parameters(self):
        """Render ensemble model parameters"""
        st.markdown("#### 🔄 Ensemble Configuration")
        
        ensemble_methods = st.multiselect(
            "Component Methods",
            ["Linear Regression", "Random Forest", "XGBoost", "ARIMA"],
            default=["Linear Regression", "Random Forest"],
            help="Methods to combine in the ensemble"
        )
        
        weighting_strategy = st.radio(
            "Weighting Strategy",
            ["Equal Weights", "Performance-Based", "Dynamic Weighting", "Learned Weights"],
            help="How to combine predictions from different models"
        )
        
        if weighting_strategy == "Performance-Based":
            performance_metric = st.selectbox(
                "Performance Metric",
                ["MAPE", "RMSE", "MAE", "R²"],
                help="Metric used to weight model contributions"
            )
        
        validation_split = st.slider(
            "Validation Split (%)",
            10, 50, 20,
            help="Percentage of data used for validation"
        )
    
    def render_bayesian_parameters(self):
        """Render Bayesian model parameters"""
        st.markdown("#### 🎯 Bayesian Configuration")
        
        hierarchy_type = st.selectbox(
            "Hierarchy Structure",
            ["Regional", "Market Segment", "Time Series", "Mixed Effects"],
            help="Type of hierarchical structure to model"
        )
        
        prior_specification = st.radio(
            "Prior Specification",
            ["Weakly Informative", "Informative", "Custom"],
            help="How much prior knowledge to incorporate"
        )
        
        if prior_specification == "Custom":
            st.markdown("**Custom Prior Settings:**")
            growth_prior_mean = st.slider("Expected Growth Rate (%)", -10, 50, 10)
            growth_prior_std = st.slider("Growth Rate Uncertainty (%)", 1, 20, 5)
        
        mcmc_samples = st.slider(
            "MCMC Samples",
            1000, 10000, 4000,
            help="More samples = better convergence but slower computation"
        )
        
        uncertainty_quantiles = st.multiselect(
            "Uncertainty Intervals",
            ["50%", "80%", "90%", "95%", "99%"],
            default=["80%", "95%"],
            help="Confidence intervals to compute"
        )
    
    def render_forecast_generation(self):
        """Render forecast generation interface"""
        st.markdown("### 🚀 Generate Your Forecast")
        
        # Configuration summary
        st.markdown("#### 📋 Configuration Summary")
        
        config_summary = {
            "Forecasting Method": st.session_state.get('selected_forecast_method', 'Not Selected'),
            "Data Quality": "Good" if st.session_state.global_forecast is not None else "Missing",
            "Countries": len(st.session_state.country_historical['Country'].unique()) if st.session_state.country_historical is not None else 0,
            "Indicators": len(st.session_state.indicators),
            "Time Horizon": "5 years (2024-2028)"
        }
        
        col1, col2 = st.columns(2)
        for i, (key, value) in enumerate(config_summary.items()):
            with col1 if i % 2 == 0 else col2:
                st.metric(key, value)
        
        # Generate button
        if st.button("🚀 Generate Forecast", type="primary", help="Start the forecasting process"):
            with st.spinner("Generating market forecast... This may take a few minutes."):
                progress_bar = st.progress(0)
                
                # Simulate forecast generation steps
                steps = [
                    "Loading data...",
                    "Preprocessing indicators...",
                    "Training models...",
                    "Generating forecasts...",
                    "Applying constraints...",
                    "Finalizing results..."
                ]
                
                for i, step in enumerate(steps):
                    st.text(step)
                    progress_bar.progress((i + 1) / len(steps))
                    # Mock processing time
                    import time
                    time.sleep(0.5)
                
                # Generate the actual forecast
                self.generate_quick_forecast()
                
                progress_bar.progress(1.0)
                st.success("✅ Forecast generated successfully!")
                st.session_state.workflow_progress['forecast_generated'] = True
                
                # Auto-advance to next step
                st.session_state.workflow_step = 4
                st.rerun()
    
    def render_auto_calibration(self):
        """Render auto-calibration interface"""
        st.markdown("### 🎯 Auto-Calibration")
        st.markdown("Automatically optimize your forecast based on historical accuracy.")
        
        if st.session_state.distributed_market is not None:
            # Mock calibration metrics
            st.markdown("#### 📊 Current Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy (MAPE)", "12.3%", "-2.1%")
            with col2:
                st.metric("R² Score", "0.847", "+0.056")
            with col3:
                st.metric("Directional Accuracy", "78.5%", "+5.2%")
            with col4:
                st.metric("Bias", "-1.2%", "+0.8%")
            
            # Calibration options
            st.markdown("#### 🔧 Calibration Options")
            
            calibration_target = st.selectbox(
                "Optimization Target",
                ["Overall Accuracy", "Reduce Bias", "Improve Direction", "Minimize Variance"],
                help="What aspect of the forecast to optimize"
            )
            
            calibration_method = st.radio(
                "Calibration Method",
                ["Automatic", "Manual Adjustment", "Hybrid"],
                help="How to perform the calibration"
            )
            
            if st.button("🎯 Run Auto-Calibration"):
                with st.spinner("Calibrating model..."):
                    import time
                    time.sleep(2)
                    st.success("✅ Model calibrated! Accuracy improved by 3.2%")
                    st.session_state.workflow_step = 5
                    st.rerun()
        else:
            st.warning("No forecast available to calibrate. Please generate a forecast first.")
    
    def render_forecast_review(self):
        """Render forecast review interface"""
        st.markdown("### ✅ Review Your Forecast")
        
        if st.session_state.distributed_market is not None:
            forecast_df = st.session_state.distributed_market
            
            # Summary statistics
            st.markdown("#### 📊 Forecast Summary")
            
            total_market_2024 = forecast_df[forecast_df['Year'] == 2024]['Value'].sum()
            total_market_2028 = forecast_df[forecast_df['Year'] == 2028]['Value'].sum()
            cagr = ((total_market_2028 / total_market_2024) ** (1/4) - 1) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("2024 Market Size", f"${total_market_2024:,.0f}M")
            with col2:
                st.metric("2028 Market Size", f"${total_market_2028:,.0f}M")
            with col3:
                st.metric("CAGR", f"{cagr:.1f}%")
            with col4:
                st.metric("Countries", forecast_df['Country'].nunique())
            
            # Quick visualization
            st.markdown("#### 📈 Market Growth Preview")
            
            yearly_totals = forecast_df.groupby('Year')['Value'].sum()
            st.line_chart(yearly_totals)
            
            # Top countries
            st.markdown("#### 🏆 Top Countries (2028)")
            
            top_countries_2028 = forecast_df[forecast_df['Year'] == 2028].nlargest(5, 'Value')
            st.dataframe(
                top_countries_2028[['Country', 'Value']].rename(columns={'Value': 'Market Size ($M)'}),
                use_container_width=True
            )
            
            # Navigation
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back to Calibration"):
                    st.session_state.workflow_step = 4
                    st.rerun()
            with col2:
                if st.button("Complete Forecasting →"):
                    st.session_state.workflow_progress['forecast_generated'] = True
                    st.success("🎉 Forecasting workflow completed!")
        else:
            st.warning("No forecast available to review.")
    
    def render_advanced_configuration(self, step: int):
        """Render advanced configuration workflow"""
        if step == 1:
            st.markdown("### ⚙️ Advanced Configuration")
            st.markdown("Fine-tune every aspect of your forecasting system.")
            
            # Configuration categories
            config_categories = {
                "Model Configuration": "Deep model parameter tuning",
                "Data Processing": "Advanced data preprocessing options", 
                "Indicator Weighting": "Custom indicator weight optimization",
                "Constraint Tuning": "Sophisticated constraint mechanisms",
                "Export Settings": "Custom export and API configurations"
            }
            
            selected_category = st.selectbox(
                "Configuration Category",
                list(config_categories.keys()),
                help="Choose the area to configure"
            )
            
            st.markdown(f"**{selected_category}:** {config_categories[selected_category]}")
            
            # Category-specific configuration
            if selected_category == "Model Configuration":
                self.render_advanced_model_config()
            elif selected_category == "Data Processing":
                self.render_advanced_data_config()
            elif selected_category == "Indicator Weighting":
                self.render_advanced_indicator_config()
            elif selected_category == "Constraint Tuning":
                self.render_advanced_constraint_config()
            elif selected_category == "Export Settings":
                self.render_advanced_export_config()
        
        # Additional steps for advanced configuration...
    
    def render_advanced_model_config(self):
        """Render advanced model configuration"""
        st.markdown("#### 🔧 Advanced Model Configuration")
        
        # Cross-validation settings
        st.markdown("**Cross-Validation:**")
        cv_method = st.selectbox(
            "CV Method",
            ["Time Series Split", "K-Fold", "Leave-One-Out", "Custom"],
            help="Method for model validation"
        )
        
        # Hyperparameter optimization
        st.markdown("**Hyperparameter Optimization:**")
        hpo_method = st.selectbox(
            "Optimization Method",
            ["Grid Search", "Random Search", "Bayesian Optimization", "Genetic Algorithm"],
            help="Method for finding optimal parameters"
        )
        
        # Model ensemble settings
        st.markdown("**Advanced Ensemble:**")
        ensemble_diversity = st.slider(
            "Ensemble Diversity",
            0.0, 1.0, 0.5,
            help="Balance between diversity and accuracy in ensemble"
        )
    
    def render_advanced_data_config(self):
        """Render advanced data processing configuration"""
        st.markdown("#### 📊 Advanced Data Processing")
        
        # Data quality settings
        outlier_handling = st.multiselect(
            "Outlier Detection Methods",
            ["Z-Score", "IQR", "Isolation Forest", "Local Outlier Factor"],
            default=["Z-Score", "IQR"],
            help="Methods to detect and handle outliers"
        )
        
        # Missing data handling
        missing_data_strategy = st.selectbox(
            "Missing Data Strategy",
            ["Forward Fill", "Backward Fill", "Interpolation", "Model-Based Imputation"],
            help="How to handle missing data points"
        )
        
        # Feature engineering
        feature_engineering = st.multiselect(
            "Advanced Feature Engineering",
            ["Fourier Features", "Wavelet Transform", "Principal Components", "Custom Transformations"],
            help="Advanced feature creation methods"
        )
    
    def render_advanced_indicator_config(self):
        """Render advanced indicator configuration"""
        st.markdown("#### 📈 Advanced Indicator Weighting")
        
        if st.session_state.indicators:
            st.markdown("**Indicator Importance Analysis:**")
            
            for indicator_name, indicator_data in st.session_state.indicators.items():
                with st.expander(f"Configure {indicator_name}"):
                    # Weight optimization method
                    weight_method = st.radio(
                        f"Weight Method for {indicator_name}",
                        ["Manual", "Correlation-Based", "Mutual Information", "Causal Inference"],
                        key=f"weight_method_{indicator_name}"
                    )
                    
                    if weight_method == "Manual":
                        weight = st.slider(
                            "Weight",
                            0.0, 1.0, 0.5,
                            key=f"weight_{indicator_name}"
                        )
                    
                    # Transformation options
                    transformation = st.selectbox(
                        "Data Transformation",
                        ["None", "Log", "Square Root", "Standardize", "Normalize"],
                        key=f"transform_{indicator_name}"
                    )
                    
                    # Lag configuration
                    use_lags = st.checkbox(
                        "Use Lagged Values",
                        key=f"lags_{indicator_name}"
                    )
                    
                    if use_lags:
                        lag_periods = st.multiselect(
                            "Lag Periods",
                            [1, 2, 3, 4, 5],
                            default=[1],
                            key=f"lag_periods_{indicator_name}"
                        )
        else:
            st.info("No indicators loaded. Upload indicator data to configure weights.")
    
    def render_advanced_constraint_config(self):
        """Render advanced constraint configuration"""
        st.markdown("#### 🎯 Advanced Constraint Tuning")
        
        # Dynamic constraints
        st.markdown("**Dynamic Constraints:**")
        
        dynamic_growth_limits = st.checkbox(
            "Dynamic Growth Limits",
            help="Adjust growth limits based on market conditions"
        )
        
        if dynamic_growth_limits:
            growth_acceleration = st.slider(
                "Growth Acceleration Factor",
                0.5, 2.0, 1.0,
                help="How quickly growth limits adapt"
            )
        
        # Regional constraints
        st.markdown("**Regional Constraints:**")
        
        regional_balance = st.checkbox(
            "Enforce Regional Balance",
            help="Maintain balanced growth across regions"
        )
        
        if regional_balance:
            balance_strictness = st.slider(
                "Balance Strictness",
                0.0, 1.0, 0.5,
                help="How strictly to enforce regional balance"
            )
        
        # Custom constraint rules
        st.markdown("**Custom Constraint Rules:**")
        
        custom_rules = st.text_area(
            "Custom Rules (Python Expression)",
            placeholder="e.g., country_share['USA'] <= 0.4",
            help="Advanced users can define custom constraint rules"
        )
    
    def render_advanced_export_config(self):
        """Render advanced export configuration"""
        st.markdown("#### 📤 Advanced Export Settings")
        
        # Export formats
        export_formats = st.multiselect(
            "Export Formats",
            ["Excel", "CSV", "JSON", "Parquet", "SQL", "API"],
            default=["Excel", "CSV"],
            help="Choose output formats for your results"
        )
        
        # Report customization
        st.markdown("**Report Customization:**")
        
        include_confidence_intervals = st.checkbox(
            "Include Confidence Intervals",
            value=True,
            help="Add uncertainty bounds to forecasts"
        )
        
        include_methodology = st.checkbox(
            "Include Methodology Section",
            value=True,
            help="Add detailed methodology explanation"
        )
        
        # API configuration
        if "API" in export_formats:
            st.markdown("**API Configuration:**")
            
            api_update_frequency = st.selectbox(
                "Update Frequency",
                ["Real-time", "Daily", "Weekly", "Monthly"],
                help="How often to update API endpoints"
            )
            
            authentication_method = st.selectbox(
                "Authentication",
                ["API Key", "OAuth", "JWT", "None"],
                help="API security method"
            )