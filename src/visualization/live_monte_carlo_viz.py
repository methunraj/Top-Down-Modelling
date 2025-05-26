"""
Real-time Monte Carlo Visualization System
Provides live charts and progress monitoring for Monte Carlo simulations
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import threading
import time
import queue
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from collections import deque
import logging

@dataclass
class MonteCarloProgress:
    """Data structure for Monte Carlo progress tracking"""
    iteration: int
    total_iterations: int
    convergence_metric: float
    confidence_interval_width: float
    current_estimate: float
    variance: float
    time_elapsed: float
    eta: float  # Estimated time to completion
    regime_probabilities: Optional[Dict[str, float]] = None
    uncertainty_decomposition: Optional[Dict[str, float]] = None

class LiveMonteCarloVisualizer:
    """Real-time visualization system for Monte Carlo simulations"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.progress_history = deque(maxlen=max_history)
        self.estimates_history = deque(maxlen=max_history)
        self.convergence_history = deque(maxlen=max_history)
        self.is_running = False
        self.start_time = None
        self.progress_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start the monitoring system"""
        self.is_running = True
        self.start_time = time.time()
        self.progress_history.clear()
        self.estimates_history.clear()
        self.convergence_history.clear()
        
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        
    def update_progress(self, progress: MonteCarloProgress):
        """Update progress data"""
        if self.is_running:
            self.progress_history.append(progress)
            self.progress_queue.put(progress)
            
    def create_live_dashboard(self) -> Dict[str, go.Figure]:
        """Create live dashboard with multiple charts"""
        if not self.progress_history:
            return self._create_empty_dashboard()
            
        latest_progress = self.progress_history[-1]
        
        # Main convergence chart
        convergence_fig = self._create_convergence_chart()
        
        # Progress and ETA chart
        progress_fig = self._create_progress_chart()
        
        # Distribution evolution chart
        distribution_fig = self._create_distribution_chart()
        
        # Uncertainty decomposition chart
        uncertainty_fig = self._create_uncertainty_chart()
        
        # Real-time metrics
        metrics_fig = self._create_metrics_dashboard(latest_progress)
        
        return {
            'convergence': convergence_fig,
            'progress': progress_fig,
            'distribution': distribution_fig,
            'uncertainty': uncertainty_fig,
            'metrics': metrics_fig
        }
        
    def _create_convergence_chart(self) -> go.Figure:
        """Create convergence monitoring chart"""
        if not self.progress_history:
            return go.Figure()
            
        iterations = [p.iteration for p in self.progress_history]
        estimates = [p.current_estimate for p in self.progress_history]
        ci_widths = [p.confidence_interval_width for p in self.progress_history]
        convergence_metrics = [p.convergence_metric for p in self.progress_history]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Estimate Convergence', 'Convergence Metric'),
            vertical_spacing=0.1
        )
        
        # Estimate with confidence interval
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=estimates,
                mode='lines',
                name='Estimate',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Confidence interval bands
        upper_ci = [est + ci/2 for est, ci in zip(estimates, ci_widths)]
        lower_ci = [est - ci/2 for est, ci in zip(estimates, ci_widths)]
        
        fig.add_trace(
            go.Scatter(
                x=iterations + iterations[::-1],
                y=upper_ci + lower_ci[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                name='Confidence Interval'
            ),
            row=1, col=1
        )
        
        # Convergence metric
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=convergence_metrics,
                mode='lines',
                name='Convergence',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Monte Carlo Convergence Monitoring",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Estimate Value", row=1, col=1)
        fig.update_yaxes(title_text="Convergence Metric", row=2, col=1)
        
        return fig
        
    def _create_progress_chart(self) -> go.Figure:
        """Create progress and ETA chart"""
        if not self.progress_history:
            return go.Figure()
            
        latest_progress = self.progress_history[-1]
        
        # Progress bar
        progress_pct = (latest_progress.iteration / latest_progress.total_iterations) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = progress_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Progress ({latest_progress.iteration:,}/{latest_progress.total_iterations:,})"},
            delta = {'reference': 100},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Add ETA information
        eta_minutes = latest_progress.eta / 60
        fig.add_annotation(
            x=0.5, y=0.1,
            text=f"ETA: {eta_minutes:.1f} minutes<br>Elapsed: {latest_progress.time_elapsed:.1f}s",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            title="Simulation Progress",
            height=400
        )
        
        return fig
        
    def _create_distribution_chart(self) -> go.Figure:
        """Create distribution evolution chart"""
        if len(self.progress_history) < 10:
            return go.Figure()
            
        # Sample recent estimates for distribution
        recent_estimates = [p.current_estimate for p in list(self.progress_history)[-100:]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=recent_estimates,
            nbinsx=30,
            name='Estimate Distribution',
            opacity=0.7
        ))
        
        # Add mean line
        mean_estimate = np.mean(recent_estimates)
        fig.add_vline(
            x=mean_estimate,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_estimate:.4f}"
        )
        
        fig.update_layout(
            title="Current Estimate Distribution (Last 100 iterations)",
            xaxis_title="Estimate Value",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig
        
    def _create_uncertainty_chart(self) -> go.Figure:
        """Create uncertainty decomposition chart"""
        if not self.progress_history:
            return go.Figure()
            
        latest_progress = self.progress_history[-1]
        
        if not latest_progress.uncertainty_decomposition:
            return go.Figure().add_annotation(
                text="Uncertainty decomposition not available",
                x=0.5, y=0.5, showarrow=False
            )
            
        uncertainty_data = latest_progress.uncertainty_decomposition
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(uncertainty_data.keys()),
                y=list(uncertainty_data.values()),
                marker_color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
            )
        ])
        
        fig.update_layout(
            title="Uncertainty Decomposition",
            xaxis_title="Uncertainty Source",
            yaxis_title="Contribution",
            height=400
        )
        
        return fig
        
    def _create_metrics_dashboard(self, progress: MonteCarloProgress) -> go.Figure:
        """Create real-time metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Current Estimate', 'Variance', 'Convergence', 'Efficiency'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Current estimate
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=progress.current_estimate,
            title={"text": "Current Estimate"},
            delta={'reference': progress.current_estimate * 0.95}
        ), row=1, col=1)
        
        # Variance
        fig.add_trace(go.Indicator(
            mode="number",
            value=progress.variance,
            title={"text": "Variance"},
            number={'valueformat': '.6f'}
        ), row=1, col=2)
        
        # Convergence
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=progress.convergence_metric,
            title={'text': "Convergence"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 1], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 0.9}}
        ), row=2, col=1)
        
        # Efficiency (iterations per second)
        iterations_per_second = progress.iteration / progress.time_elapsed if progress.time_elapsed > 0 else 0
        fig.add_trace(go.Indicator(
            mode="number",
            value=iterations_per_second,
            title={"text": "Iterations/sec"},
            number={'valueformat': '.1f'}
        ), row=2, col=2)
        
        fig.update_layout(
            title="Real-time Metrics",
            height=500
        )
        
        return fig
        
    def _create_empty_dashboard(self) -> Dict[str, go.Figure]:
        """Create empty dashboard when no data is available"""
        empty_fig = go.Figure().add_annotation(
            text="Waiting for Monte Carlo data...",
            x=0.5, y=0.5, showarrow=False, font_size=16
        )
        
        return {
            'convergence': empty_fig,
            'progress': empty_fig,
            'distribution': empty_fig,
            'uncertainty': empty_fig,
            'metrics': empty_fig
        }
        
    def create_regime_chart(self) -> go.Figure:
        """Create regime probability evolution chart"""
        if not self.progress_history:
            return go.Figure()
            
        regime_data = {}
        iterations = []
        
        for progress in self.progress_history:
            if progress.regime_probabilities:
                iterations.append(progress.iteration)
                for regime, prob in progress.regime_probabilities.items():
                    if regime not in regime_data:
                        regime_data[regime] = []
                    regime_data[regime].append(prob)
                    
        if not regime_data:
            return go.Figure().add_annotation(
                text="No regime data available",
                x=0.5, y=0.5, showarrow=False
            )
            
        fig = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (regime, probs) in enumerate(regime_data.items()):
            fig.add_trace(go.Scatter(
                x=iterations,
                y=probs,
                mode='lines',
                name=f'Regime {regime}',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
            
        fig.update_layout(
            title="Market Regime Probabilities Over Time",
            xaxis_title="Iteration",
            yaxis_title="Probability",
            height=400,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
        
    def export_progress_data(self) -> pd.DataFrame:
        """Export progress data to DataFrame"""
        if not self.progress_history:
            return pd.DataFrame()
            
        data = []
        for progress in self.progress_history:
            row = asdict(progress)
            if row['regime_probabilities']:
                for regime, prob in row['regime_probabilities'].items():
                    row[f'regime_{regime}'] = prob
            if row['uncertainty_decomposition']:
                for source, value in row['uncertainty_decomposition'].items():
                    row[f'uncertainty_{source}'] = value
            data.append(row)
            
        return pd.DataFrame(data)

class StreamlitMonteCarloInterface:
    """Streamlit interface for live Monte Carlo monitoring"""
    
    def __init__(self, visualizer: LiveMonteCarloVisualizer):
        self.visualizer = visualizer
        
    def render_live_dashboard(self):
        """Render the live dashboard in Streamlit"""
        st.title("🎯 Live Monte Carlo Monitoring")
        
        if not self.visualizer.is_running and not self.visualizer.progress_history:
            st.info("No active Monte Carlo simulation. Start a simulation to see live charts.")
            return
            
        # Auto-refresh setup
        placeholder = st.empty()
        
        with placeholder.container():
            # Status indicators
            col1, col2, col3, col4 = st.columns(4)
            
            if self.visualizer.progress_history:
                latest = self.visualizer.progress_history[-1]
                
                with col1:
                    st.metric(
                        "Current Estimate", 
                        f"{latest.current_estimate:.6f}",
                        delta=f"±{latest.confidence_interval_width/2:.6f}"
                    )
                    
                with col2:
                    progress_pct = (latest.iteration / latest.total_iterations) * 100
                    st.metric("Progress", f"{progress_pct:.1f}%", f"{latest.iteration:,} iterations")
                    
                with col3:
                    st.metric("Convergence", f"{latest.convergence_metric:.4f}")
                    
                with col4:
                    eta_min = latest.eta / 60
                    st.metric("ETA", f"{eta_min:.1f} min")
            
            # Main charts
            dashboard = self.visualizer.create_live_dashboard()
            
            # Convergence chart
            st.plotly_chart(dashboard['convergence'], use_container_width=True)
            
            # Progress and metrics in columns
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(dashboard['progress'], use_container_width=True)
            with col2:
                st.plotly_chart(dashboard['metrics'], use_container_width=True)
                
            # Distribution and uncertainty
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(dashboard['distribution'], use_container_width=True)
            with col2:
                st.plotly_chart(dashboard['uncertainty'], use_container_width=True)
                
            # Regime analysis if available
            regime_fig = self.visualizer.create_regime_chart()
            if regime_fig.data:
                st.plotly_chart(regime_fig, use_container_width=True)
                
        # Export functionality
        if st.button("📊 Export Progress Data"):
            df = self.visualizer.export_progress_data()
            if not df.empty:
                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    "monte_carlo_progress.csv",
                    "text/csv"
                )
                
        # Auto-refresh control
        if st.checkbox("Auto-refresh (every 2 seconds)", value=True):
            time.sleep(2)
            st.rerun()

def create_monte_carlo_callback(visualizer: LiveMonteCarloVisualizer, 
                              progress_bar=None, status_text=None, monitoring_refs=None) -> Callable:
    """Create callback function for Monte Carlo engines to report progress"""
    
    def progress_callback(iteration: int, total_iterations: int, current_estimate: float, 
                         convergence_metric: float, confidence_interval_width: float,
                         variance: float, elapsed_time: float, eta: float,
                         regime_probabilities: Optional[Dict[str, float]] = None,
                         uncertainty_decomposition: Optional[Dict[str, float]] = None):
        """Progress callback for Monte Carlo engines"""
        
        progress = MonteCarloProgress(
            iteration=iteration,
            total_iterations=total_iterations,
            convergence_metric=convergence_metric,
            confidence_interval_width=confidence_interval_width,
            current_estimate=current_estimate,
            variance=variance,
            time_elapsed=elapsed_time,
            eta=eta,
            regime_probabilities=regime_probabilities,
            uncertainty_decomposition=uncertainty_decomposition
        )
        
        visualizer.update_progress(progress)
        
        # Update Streamlit progress bar and status
        if progress_bar is not None:
            progress_pct = iteration / total_iterations
            progress_bar.progress(progress_pct)
            
        if status_text is not None:
            eta_min = eta / 60 if eta > 0 else 0
            status_text.text(f"Iteration {iteration:,}/{total_iterations:,} | "
                           f"Estimate: {current_estimate:.4f} | "
                           f"Convergence: {convergence_metric:.3f} | "
                           f"ETA: {eta_min:.1f}min")
        
        # Update live monitoring interface
        if monitoring_refs is not None:
            try:
                # Update metrics
                progress_pct = iteration / total_iterations
                eta_min = eta / 60 if eta > 0 else 0
                
                monitoring_refs['progress_metric'].metric(
                    "Progress", 
                    f"{progress_pct*100:.1f}%", 
                    f"{iteration:,}/{total_iterations:,}"
                )
                
                monitoring_refs['estimate_metric'].metric(
                    "Current Estimate", 
                    f"{current_estimate:.4f}",
                    f"±{confidence_interval_width/2:.4f}"
                )
                
                monitoring_refs['convergence_metric'].metric(
                    "Convergence", 
                    f"{convergence_metric:.3f}",
                    "Higher is better"
                )
                
                monitoring_refs['eta_metric'].metric(
                    "ETA", 
                    f"{eta_min:.1f}min",
                    f"Elapsed: {elapsed_time:.1f}s"
                )
                
                # Update charts if we have enough data
                if len(visualizer.progress_history) > 5:
                    dashboard = visualizer.create_live_dashboard()
                    
                    # Update convergence chart
                    if 'convergence' in dashboard and dashboard['convergence'].data:
                        monitoring_refs['convergence_chart'].plotly_chart(
                            dashboard['convergence'], 
                            use_container_width=True,
                            key=f"conv_{iteration}"
                        )
                    
                    # Update progress chart
                    if 'progress' in dashboard and dashboard['progress'].data:
                        monitoring_refs['progress_chart'].plotly_chart(
                            dashboard['progress'], 
                            use_container_width=True,
                            key=f"prog_{iteration}"
                        )
                    
                    # Update distribution chart
                    if 'distribution' in dashboard and dashboard['distribution'].data:
                        monitoring_refs['distribution_chart'].plotly_chart(
                            dashboard['distribution'], 
                            use_container_width=True,
                            key=f"dist_{iteration}"
                        )
                
            except Exception as e:
                # Don't let UI updates crash the simulation
                pass
        
    return progress_callback