"""
Streamlit Live Monte Carlo Interface
Advanced real-time monitoring dashboard for Monte Carlo simulations
"""

import streamlit as st
import threading
import time
import queue
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

from src.visualization.live_monte_carlo_viz import (
    LiveMonteCarloVisualizer, 
    StreamlitMonteCarloInterface,
    MonteCarloProgress,
    create_monte_carlo_callback
)
from src.advanced_forecasting.enhanced_monte_carlo_framework import EnhancedMonteCarloFramework

class LiveMonteCarloApp:
    """Main application class for live Monte Carlo monitoring"""
    
    def __init__(self):
        self.visualizer = LiveMonteCarloVisualizer()
        self.interface = StreamlitMonteCarloInterface(self.visualizer)
        self.monte_carlo_framework = None
        self.simulation_thread = None
        
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = self.visualizer
            
    def render_main_interface(self):
        """Render the main Streamlit interface"""
        # Note: st.set_page_config is already called by main app, so we skip it here
        
        self.initialize_session_state()
        
        st.title("🎯 Live Monte Carlo Simulation Monitor")
        st.markdown("Real-time visualization and monitoring of advanced Monte Carlo simulations")
        
        # Sidebar controls
        self.render_sidebar()
        
        # Main dashboard
        if st.session_state.simulation_running or self.visualizer.progress_history:
            self.interface.render_live_dashboard()
        else:
            self.render_welcome_screen()
            
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ Simulation Controls")
        
        # Simulation parameters
        st.sidebar.subheader("Parameters")
        n_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, 100)
        confidence_level = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            use_quasi_mc = st.checkbox("Use Quasi-Monte Carlo", value=True)
            use_variance_reduction = st.checkbox("Use Variance Reduction", value=True)
            use_regime_switching = st.checkbox("Enable Regime Switching", value=True)
            adaptive_sampling = st.checkbox("Adaptive Sampling", value=True)
            
        # Simulation control buttons
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("▶️ Start", disabled=st.session_state.simulation_running):
                self.start_simulation({
                    'n_simulations': n_simulations,
                    'confidence_level': confidence_level,
                    'use_quasi_mc': use_quasi_mc,
                    'use_variance_reduction': use_variance_reduction,
                    'use_regime_switching': use_regime_switching,
                    'adaptive_sampling': adaptive_sampling
                })
                
        with col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.simulation_running):
                self.stop_simulation()
                
        # Status indicator
        if st.session_state.simulation_running:
            st.sidebar.success("🟢 Simulation Running")
        else:
            st.sidebar.info("🔴 Simulation Stopped")
            
        # Progress summary
        if self.visualizer.progress_history:
            latest = self.visualizer.progress_history[-1]
            st.sidebar.metric("Current Iteration", f"{latest.iteration:,}")
            st.sidebar.metric("Convergence", f"{latest.convergence_metric:.4f}")
            
            progress_pct = (latest.iteration / latest.total_iterations) * 100
            st.sidebar.progress(progress_pct / 100)
            
    def render_welcome_screen(self):
        """Render welcome screen when no simulation is running"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## 🚀 Welcome to Live Monte Carlo Monitor
            
            This advanced monitoring system provides real-time visualization of Monte Carlo simulations with:
            
            ### ✨ Features
            - **Real-time convergence monitoring** with confidence intervals
            - **Live progress tracking** with ETA estimation
            - **Distribution evolution** visualization
            - **Uncertainty decomposition** analysis
            - **Market regime detection** (when enabled)
            - **Interactive dashboards** with auto-refresh
            
            ### 🎯 Advanced Capabilities
            - Quasi-Monte Carlo sampling with Sobol sequences
            - Variance reduction techniques
            - Copula-based parameter dependencies
            - Regime-switching Monte Carlo
            - Adaptive sampling optimization
            
            ### 🎮 Getting Started
            1. Configure simulation parameters in the sidebar
            2. Click "▶️ Start" to begin simulation
            3. Watch live charts update in real-time
            4. Export results when complete
            
            **Configure your simulation parameters in the sidebar and click Start to begin!**
            """)
            
    def start_simulation(self, params: Dict[str, Any]):
        """Start Monte Carlo simulation with live monitoring"""
        try:
            st.session_state.simulation_running = True
            self.visualizer.start_monitoring()
            
            # Create progress callback
            progress_callback = create_monte_carlo_callback(self.visualizer)
            
            # Start simulation in background thread
            self.simulation_thread = threading.Thread(
                target=self._run_simulation_thread,
                args=(params, progress_callback),
                daemon=True
            )
            self.simulation_thread.start()
            
            st.success("🚀 Simulation started! Watch the live charts below.")
            st.rerun()
            
        except Exception as e:
            st.error(f"Failed to start simulation: {str(e)}")
            st.session_state.simulation_running = False
            
    def stop_simulation(self):
        """Stop the current simulation"""
        st.session_state.simulation_running = False
        self.visualizer.stop_monitoring()
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            # Note: Cannot directly stop thread, but the simulation will check the flag
            pass
            
        st.info("⏹️ Simulation stopped.")
        st.rerun()
        
    def _run_simulation_thread(self, params: Dict[str, Any], progress_callback):
        """Run simulation in background thread"""
        try:
            # Initialize framework if needed
            if not self.monte_carlo_framework:
                self.monte_carlo_framework = EnhancedMonteCarloFramework()
                
            # Create sample model function for demonstration
            def sample_market_model(parameters):
                """Sample market forecasting model"""
                volatility = parameters.get('volatility', 0.2)
                drift = parameters.get('drift', 0.05)
                
                # Simple geometric Brownian motion
                dt = 1/252  # Daily time step
                T = 1  # 1 year
                n_steps = int(T / dt)
                
                returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_steps)
                price_path = 100 * np.exp(np.cumsum(returns))
                
                return price_path[-1]  # Final price
                
            # Parameter definitions
            parameter_definitions = {
                'volatility': {
                    'type': 'normal',
                    'mean': 0.2,
                    'std': 0.05,
                    'bounds': [0.05, 0.5]
                },
                'drift': {
                    'type': 'normal', 
                    'mean': 0.05,
                    'std': 0.02,
                    'bounds': [-0.1, 0.2]
                }
            }
            
            # Analysis configuration
            analysis_config = {
                'n_simulations': params['n_simulations'],
                'confidence_level': params['confidence_level'],
                'quasi_monte_carlo': params['use_quasi_mc'],
                'variance_reduction': params['use_variance_reduction'],
                'regime_switching': params['use_regime_switching'],
                'adaptive_sampling': params['adaptive_sampling'],
                'progress_callback': progress_callback,
                'max_time_minutes': 30  # Safety timeout
            }
            
            # Run comprehensive analysis
            results = self.monte_carlo_framework.run_comprehensive_analysis(
                model_function=sample_market_model,
                parameter_definitions=parameter_definitions,
                analysis_config=analysis_config
            )
            
            # Store results
            st.session_state.simulation_results = results
            st.session_state.simulation_running = False
            
        except Exception as e:
            st.session_state.simulation_running = False
            # Log error but don't crash the app
            print(f"Simulation error: {str(e)}")

def create_enhanced_dashboard():
    """Create enhanced dashboard with additional analytics"""
    st.header("📊 Enhanced Analytics Dashboard")
    
    if not st.session_state.get('simulation_results'):
        st.info("Complete a simulation to see enhanced analytics.")
        return
        
    results = st.session_state.simulation_results
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Final Estimate",
            f"{results.get('estimate', 0):.4f}",
            delta=f"±{results.get('confidence_interval_width', 0)/2:.4f}"
        )
        
    with col2:
        st.metric(
            "Convergence Score",
            f"{results.get('convergence_metric', 0):.4f}"
        )
        
    with col3:
        st.metric(
            "Total Iterations",
            f"{results.get('total_iterations', 0):,}"
        )
        
    with col4:
        st.metric(
            "Simulation Time",
            f"{results.get('total_time', 0):.1f}s"
        )
        
    # Detailed results
    if 'sobol_indices' in results:
        st.subheader("🔍 Sensitivity Analysis")
        sobol_data = results['sobol_indices']
        
        # Create Sobol indices chart
        fig = go.Figure()
        parameters = list(sobol_data.get('first_order', {}).keys())
        first_order = list(sobol_data.get('first_order', {}).values())
        total_order = list(sobol_data.get('total_order', {}).values())
        
        fig.add_trace(go.Bar(
            x=parameters,
            y=first_order,
            name='First Order',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=parameters,
            y=total_order,
            name='Total Order',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Sobol Sensitivity Indices",
            xaxis_title="Parameters",
            yaxis_title="Sensitivity Index",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    # Export results
    if st.button("📥 Export Complete Results"):
        results_json = json.dumps(results, indent=2, default=str)
        st.download_button(
            "Download Results JSON",
            results_json,
            "monte_carlo_results.json",
            "application/json"
        )

def main():
    """Main application entry point"""
    app = LiveMonteCarloApp()
    
    # Navigation
    tab1, tab2 = st.tabs(["🎯 Live Monitor", "📊 Enhanced Analytics"])
    
    with tab1:
        app.render_main_interface()
        
    with tab2:
        create_enhanced_dashboard()

if __name__ == "__main__":
    main()