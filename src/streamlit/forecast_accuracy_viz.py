"""
Forecast Accuracy and Calibration Dashboard
Advanced visualization for model performance analysis and calibration insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ForecastAccuracyVisualizer:
    """Comprehensive forecast accuracy and calibration visualization suite"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#3498db',
            'secondary': '#e74c3c', 
            'success': '#2ecc71',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'background': '#ecf0f1',
            'text': '#2c3e50',
            'accuracy_high': '#27ae60',
            'accuracy_medium': '#f39c12',
            'accuracy_low': '#e74c3c'
        }
        
        self.model_colors = {
            'Actual': '#2c3e50',
            'Statistical': '#3498db',
            'ML': '#e74c3c',
            'Ensemble': '#2ecc71',
            'Hybrid': '#9b59b6',
            'Bayesian': '#f39c12'
        }
    
    def apply_professional_styling(self):
        """Apply professional CSS styling for forecast analysis"""
        st.markdown("""
        <style>
        .forecast-header {
            background: linear-gradient(135deg, #3498db 0%, #9b59b6 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .forecast-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .forecast-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .accuracy-metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        .accuracy-metric-card h3 {
            margin: 0 0 1rem 0;
            font-size: 1.2rem;
        }
        
        .accuracy-metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        
        .accuracy-metric-card .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .model-performance-card {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .accuracy-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        
        .accuracy-excellent { background: #d4edda; color: #155724; }
        .accuracy-good { background: #d1ecf1; color: #0c5460; }
        .accuracy-fair { background: #fff3cd; color: #856404; }
        .accuracy-poor { background: #f8d7da; color: #721c24; }
        
        .calibration-insight {
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
        
        .error-analysis {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .forecast-summary {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def create_header(self):
        """Create professional header for forecast analysis"""
        st.markdown("""
        <div class="forecast-header">
            <h1>📊 Forecast Accuracy Dashboard</h1>
            <p>Comprehensive model performance analysis and calibration insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    def calculate_accuracy_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive accuracy metrics"""
        if len(actual) == 0 or len(predicted) == 0:
            return {}
        
        # Remove NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {}
        
        # Basic metrics
        mae = mean_absolute_error(actual_clean, predicted_clean)
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        
        # Percentage errors
        mape = np.mean(np.abs((actual_clean - predicted_clean) / np.maximum(actual_clean, 1e-10))) * 100
        
        # R-squared
        r2 = r2_score(actual_clean, predicted_clean)
        
        # Directional accuracy
        actual_direction = np.diff(actual_clean) > 0
        predicted_direction = np.diff(predicted_clean) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100 if len(actual_direction) > 0 else 0
        
        # Bias
        bias = np.mean(predicted_clean - actual_clean)
        bias_percentage = (bias / np.mean(actual_clean)) * 100 if np.mean(actual_clean) != 0 else 0
        
        # Theil's U statistic
        def theil_u(actual, predicted):
            if len(actual) <= 1:
                return np.nan
            naive_forecast = actual[:-1]  # Previous period forecast
            actual_test = actual[1:]
            predicted_test = predicted[1:] if len(predicted) > 1 else predicted
            
            if len(predicted_test) != len(actual_test):
                min_len = min(len(predicted_test), len(actual_test))
                predicted_test = predicted_test[:min_len]
                actual_test = actual_test[:min_len]
                naive_forecast = naive_forecast[:min_len]
            
            if len(actual_test) == 0:
                return np.nan
                
            mse_model = np.mean((predicted_test - actual_test) ** 2)
            mse_naive = np.mean((naive_forecast - actual_test) ** 2)
            
            return np.sqrt(mse_model / mse_naive) if mse_naive > 0 else np.nan
        
        theil_u_stat = theil_u(actual_clean, predicted_clean)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'bias': bias,
            'bias_percentage': bias_percentage,
            'theil_u': theil_u_stat
        }
    
    def create_accuracy_vs_actual_plot(self, actual: np.ndarray, predicted: np.ndarray, 
                                     model_name: str = "Model") -> go.Figure:
        """Create scatter plot of predicted vs actual values"""
        fig = go.Figure()
        
        # Perfect prediction line
        min_val = min(np.min(actual), np.min(predicted))
        max_val = max(np.max(actual), np.max(predicted))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2),
            hoverinfo='skip'
        ))
        
        # Actual vs predicted scatter
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name=f'{model_name} Predictions',
            marker=dict(
                color=self.color_palette['primary'],
                size=8,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
        ))
        
        # Calculate R² for annotation
        r2 = r2_score(actual, predicted) if len(actual) > 1 else 0
        
        fig.add_annotation(
            x=0.05, y=0.95,
            xref='paper', yref='paper',
            text=f'R² = {r2:.3f}',
            showarrow=False,
            font=dict(size=14, color=self.color_palette['text']),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
        
        fig.update_layout(
            title={
                'text': f'Predicted vs Actual Values - {model_name}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=500,
            margin=dict(t=80, b=60, l=60, r=40),
            showlegend=True
        )
        
        return fig
    
    def create_residual_analysis_plot(self, actual: np.ndarray, predicted: np.ndarray,
                                    model_name: str = "Model") -> go.Figure:
        """Create residual analysis plot"""
        residuals = predicted - actual
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Residuals vs Predicted',
                'Residuals Distribution',
                'Q-Q Plot',
                'Residuals Over Time'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(
                x=predicted,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color=self.color_palette['secondary'], size=6, opacity=0.7),
                hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Residuals histogram
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Distribution',
                marker_color=self.color_palette['info'],
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=2
        )
        
        # Q-Q plot
        qq_data = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[0][1],
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=self.color_palette['success'], size=6)
            ),
            row=2, col=1
        )
        
        # Q-Q line
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0],
                y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                mode='lines',
                name='Normal Line',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Residuals over time
        fig.add_trace(
            go.Scatter(
                x=list(range(len(residuals))),
                y=residuals,
                mode='lines+markers',
                name='Time Series',
                line=dict(color=self.color_palette['warning']),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        fig.update_layout(
            title={
                'text': f'Residual Analysis - {model_name}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            height=600,
            margin=dict(t=100, b=60, l=60, r=40),
            showlegend=False
        )
        
        return fig
    
    def create_model_comparison_chart(self, models_data: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create model comparison chart"""
        metrics = ['mae', 'rmse', 'mape', 'r2', 'directional_accuracy']
        model_names = list(models_data.keys())
        
        fig = go.Figure()
        
        colors = list(self.model_colors.values())
        
        for i, model in enumerate(model_names):
            metric_values = [models_data[model].get(metric, 0) for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=metric_values,
                theta=metrics,
                fill='toself',
                name=model,
                line_color=colors[i % len(colors)],
                fillcolor=f"rgba{(*[int(colors[i % len(colors)][j:j+2], 16) for j in (1, 3, 5)], 0.3)}"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(100, max([max(v.values()) for v in models_data.values()]))]
                )
            ),
            title={
                'text': 'Model Performance Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            showlegend=True,
            height=600,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        return fig
    
    def create_forecast_horizon_analysis(self, data: pd.DataFrame, actual_col: str,
                                       forecast_cols: List[str], horizon_col: str) -> go.Figure:
        """Create forecast accuracy by horizon analysis"""
        fig = go.Figure()
        
        horizons = sorted(data[horizon_col].unique())
        
        for i, col in enumerate(forecast_cols):
            horizon_accuracies = []
            horizon_labels = []
            
            for horizon in horizons:
                horizon_data = data[data[horizon_col] == horizon]
                if len(horizon_data) > 0:
                    actual_vals = horizon_data[actual_col].values
                    predicted_vals = horizon_data[col].values
                    
                    metrics = self.calculate_accuracy_metrics(actual_vals, predicted_vals)
                    mape = metrics.get('mape', np.nan)
                    
                    if not np.isnan(mape):
                        horizon_accuracies.append(mape)
                        horizon_labels.append(horizon)
            
            if horizon_accuracies:
                fig.add_trace(go.Scatter(
                    x=horizon_labels,
                    y=horizon_accuracies,
                    mode='lines+markers',
                    name=col,
                    line=dict(
                        color=list(self.model_colors.values())[i % len(self.model_colors)],
                        width=3
                    ),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title={
                'text': 'Forecast Accuracy by Horizon',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title='Forecast Horizon',
            yaxis_title='MAPE (%)',
            height=500,
            margin=dict(t=80, b=60, l=60, r=40),
            showlegend=True
        )
        
        return fig
    
    def create_calibration_plot(self, actual: np.ndarray, predicted: np.ndarray,
                               confidence_intervals: Optional[np.ndarray] = None) -> go.Figure:
        """Create calibration plot for forecast intervals"""
        fig = go.Figure()
        
        # Sort by predicted values for better visualization
        sort_idx = np.argsort(predicted)
        actual_sorted = actual[sort_idx]
        predicted_sorted = predicted[sort_idx]
        
        # Actual values
        fig.add_trace(go.Scatter(
            x=list(range(len(actual_sorted))),
            y=actual_sorted,
            mode='lines+markers',
            name='Actual',
            line=dict(color=self.color_palette['text'], width=3),
            marker=dict(size=6)
        ))
        
        # Predicted values
        fig.add_trace(go.Scatter(
            x=list(range(len(predicted_sorted))),
            y=predicted_sorted,
            mode='lines+markers',
            name='Predicted',
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=6)
        ))
        
        # Confidence intervals if provided
        if confidence_intervals is not None and len(confidence_intervals) == len(predicted):
            ci_sorted = confidence_intervals[sort_idx]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(predicted_sorted))),
                y=predicted_sorted + ci_sorted,
                mode='lines',
                name='Upper CI',
                line=dict(color=self.color_palette['primary'], width=1, dash='dash'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(predicted_sorted))),
                y=predicted_sorted - ci_sorted,
                mode='lines',
                name='Lower CI',
                line=dict(color=self.color_palette['primary'], width=1, dash='dash'),
                fill='tonexty',
                fillcolor=f"rgba{(*[int(self.color_palette['primary'][j:j+2], 16) for j in (1, 3, 5)], 0.2)}",
                showlegend=True
            ))
        
        fig.update_layout(
            title={
                'text': 'Forecast Calibration Plot',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': self.color_palette['text']}
            },
            xaxis_title='Observation Index',
            yaxis_title='Value',
            height=500,
            margin=dict(t=80, b=60, l=60, r=40),
            showlegend=True
        )
        
        return fig
    
    def display_accuracy_metrics(self, metrics: Dict[str, float], model_name: str = "Model"):
        """Display accuracy metrics in cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Determine accuracy level
        mape = metrics.get('mape', float('inf'))
        if mape < 10:
            accuracy_level = "Excellent"
            accuracy_class = "accuracy-excellent"
        elif mape < 20:
            accuracy_level = "Good"
            accuracy_class = "accuracy-good"
        elif mape < 30:
            accuracy_level = "Fair"
            accuracy_class = "accuracy-fair"
        else:
            accuracy_level = "Poor"
            accuracy_class = "accuracy-poor"
        
        with col1:
            st.markdown(f"""
            <div class="accuracy-metric-card">
                <h3>MAPE</h3>
                <div class="metric-value">{metrics.get('mape', 0):.2f}%</div>
                <div class="metric-label">Mean Absolute Percentage Error</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="accuracy-metric-card">
                <h3>R²</h3>
                <div class="metric-value">{metrics.get('r2', 0):.3f}</div>
                <div class="metric-label">Coefficient of Determination</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="accuracy-metric-card">
                <h3>Direction</h3>
                <div class="metric-value">{metrics.get('directional_accuracy', 0):.1f}%</div>
                <div class="metric-label">Directional Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="accuracy-metric-card">
                <h3>Overall</h3>
                <div class="metric-value">{accuracy_level}</div>
                <div class="metric-label">Model Performance</div>
            </div>
            """, unsafe_allow_html=True)
    
    def generate_forecast_insights(self, metrics: Dict[str, float], model_name: str = "Model") -> List[str]:
        """Generate actionable forecast insights"""
        insights = []
        
        mape = metrics.get('mape', float('inf'))
        r2 = metrics.get('r2', 0)
        bias_pct = metrics.get('bias_percentage', 0)
        dir_acc = metrics.get('directional_accuracy', 0)
        theil_u = metrics.get('theil_u', float('inf'))
        
        # Overall performance
        if mape < 10:
            insights.append(f"🎯 {model_name} shows excellent accuracy with MAPE of {mape:.1f}%")
        elif mape < 20:
            insights.append(f"✅ {model_name} demonstrates good performance with MAPE of {mape:.1f}%")
        elif mape < 30:
            insights.append(f"⚠️ {model_name} shows fair accuracy with MAPE of {mape:.1f}% - room for improvement")
        else:
            insights.append(f"❌ {model_name} has poor accuracy with MAPE of {mape:.1f}% - requires optimization")
        
        # R-squared insights
        if r2 > 0.9:
            insights.append(f"📊 Strong explanatory power with R² of {r2:.3f}")
        elif r2 > 0.7:
            insights.append(f"📈 Good model fit with R² of {r2:.3f}")
        elif r2 > 0.5:
            insights.append(f"📉 Moderate model fit with R² of {r2:.3f}")
        else:
            insights.append(f"⚡ Weak model fit with R² of {r2:.3f} - consider alternative approaches")
        
        # Bias insights
        if abs(bias_pct) < 5:
            insights.append("⚖️ Model shows minimal bias - well-calibrated predictions")
        elif bias_pct > 5:
            insights.append(f"📈 Model tends to overestimate by {bias_pct:.1f}% on average")
        elif bias_pct < -5:
            insights.append(f"📉 Model tends to underestimate by {abs(bias_pct):.1f}% on average")
        
        # Directional accuracy
        if dir_acc > 80:
            insights.append(f"🎯 Excellent directional accuracy at {dir_acc:.1f}%")
        elif dir_acc > 70:
            insights.append(f"✅ Good directional accuracy at {dir_acc:.1f}%")
        elif dir_acc > 60:
            insights.append(f"⚠️ Fair directional accuracy at {dir_acc:.1f}%")
        else:
            insights.append(f"❌ Poor directional accuracy at {dir_acc:.1f}%")
        
        # Theil's U statistic
        if not np.isnan(theil_u):
            if theil_u < 1:
                insights.append("🏆 Model outperforms naive forecast (Theil's U < 1)")
            elif theil_u > 1:
                insights.append("⚠️ Model underperforms naive forecast (Theil's U > 1)")
        
        return insights
    
    def render_forecast_accuracy_dashboard(self, data: pd.DataFrame, actual_col: str,
                                         predicted_cols: List[str], time_col: str = None):
        """Render complete forecast accuracy dashboard"""
        self.apply_professional_styling()
        self.create_header()
        
        if data.empty:
            st.error("No data available for forecast accuracy analysis")
            return
        
        # Sidebar controls
        st.sidebar.header("📊 Analysis Controls")
        
        # Model selection
        selected_models = st.sidebar.multiselect(
            "Select Models to Analyze",
            predicted_cols,
            default=predicted_cols[:3] if len(predicted_cols) > 3 else predicted_cols
        )
        
        if not selected_models:
            st.warning("Please select at least one model to analyze")
            return
        
        # Analysis options
        analysis_types = st.sidebar.multiselect(
            "Select Analysis Types",
            ["Accuracy Metrics", "Residual Analysis", "Model Comparison", "Calibration Plot"],
            default=["Accuracy Metrics", "Model Comparison"]
        )
        
        # Calculate metrics for all models
        models_metrics = {}
        for model in selected_models:
            if model in data.columns:
                actual_vals = data[actual_col].dropna().values
                predicted_vals = data[model].dropna().values
                
                # Align arrays
                min_len = min(len(actual_vals), len(predicted_vals))
                actual_vals = actual_vals[:min_len]
                predicted_vals = predicted_vals[:min_len]
                
                if len(actual_vals) > 0:
                    models_metrics[model] = self.calculate_accuracy_metrics(actual_vals, predicted_vals)
        
        if not models_metrics:
            st.error("No valid data found for selected models")
            return
        
        # Display metrics for each model
        if "Accuracy Metrics" in analysis_types:
            for model in selected_models:
                if model in models_metrics:
                    st.subheader(f"📈 {model} Performance")
                    self.display_accuracy_metrics(models_metrics[model], model)
                    
                    # Generate insights
                    insights = self.generate_forecast_insights(models_metrics[model], model)
                    if insights:
                        st.markdown("### 💡 Key Insights")
                        for insight in insights:
                            st.markdown(f"""
                            <div class="calibration-insight">
                                {insight}
                            </div>
                            """, unsafe_allow_html=True)
        
        # Accuracy vs Actual plots
        if len(selected_models) <= 2:
            chart_cols = st.columns(len(selected_models))
            for i, model in enumerate(selected_models):
                if model in data.columns:
                    with chart_cols[i]:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        actual_vals = data[actual_col].dropna().values
                        predicted_vals = data[model].dropna().values
                        min_len = min(len(actual_vals), len(predicted_vals))
                        
                        fig = self.create_accuracy_vs_actual_plot(
                            actual_vals[:min_len], predicted_vals[:min_len], model
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Residual analysis
        if "Residual Analysis" in analysis_types and len(selected_models) > 0:
            selected_model_residual = st.selectbox("Select Model for Residual Analysis", selected_models)
            if selected_model_residual in data.columns:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                actual_vals = data[actual_col].dropna().values
                predicted_vals = data[selected_model_residual].dropna().values
                min_len = min(len(actual_vals), len(predicted_vals))
                
                fig = self.create_residual_analysis_plot(
                    actual_vals[:min_len], predicted_vals[:min_len], selected_model_residual
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Model comparison
        if "Model Comparison" in analysis_types and len(models_metrics) > 1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Normalize metrics for radar chart
            normalized_metrics = {}
            for model, metrics in models_metrics.items():
                normalized_metrics[model] = {
                    'mae': 100 - min(100, metrics.get('mae', 0)),  # Invert so higher is better
                    'rmse': 100 - min(100, metrics.get('rmse', 0)),
                    'mape': 100 - min(100, metrics.get('mape', 0)),
                    'r2': max(0, metrics.get('r2', 0)) * 100,
                    'directional_accuracy': metrics.get('directional_accuracy', 0)
                }
            
            fig = self.create_model_comparison_chart(normalized_metrics)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Calibration plot
        if "Calibration Plot" in analysis_types and len(selected_models) > 0:
            selected_model_cal = st.selectbox("Select Model for Calibration Analysis", selected_models)
            if selected_model_cal in data.columns:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                actual_vals = data[actual_col].dropna().values
                predicted_vals = data[selected_model_cal].dropna().values
                min_len = min(len(actual_vals), len(predicted_vals))
                
                fig = self.create_calibration_plot(actual_vals[:min_len], predicted_vals[:min_len])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Summary table
        st.subheader("📋 Model Performance Summary")
        summary_data = []
        for model, metrics in models_metrics.items():
            summary_data.append({
                'Model': model,
                'MAPE (%)': f"{metrics.get('mape', 0):.2f}",
                'R²': f"{metrics.get('r2', 0):.3f}",
                'RMSE': f"{metrics.get('rmse', 0):.2f}",
                'Directional Accuracy (%)': f"{metrics.get('directional_accuracy', 0):.1f}",
                'Bias (%)': f"{metrics.get('bias_percentage', 0):.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)


def main():
    """Main function for testing"""
    visualizer = ForecastAccuracyVisualizer()
    
    # Sample data
    np.random.seed(42)
    n_points = 100
    actual = np.cumsum(np.random.randn(n_points)) + 100
    predicted_1 = actual + np.random.randn(n_points) * 2
    predicted_2 = actual + np.random.randn(n_points) * 3 + 1
    
    sample_data = pd.DataFrame({
        'Actual': actual,
        'Model_1': predicted_1,
        'Model_2': predicted_2,
        'Time': range(n_points)
    })
    
    visualizer.render_forecast_accuracy_dashboard(
        sample_data, 'Actual', ['Model_1', 'Model_2'], 'Time'
    )

if __name__ == "__main__":
    main()