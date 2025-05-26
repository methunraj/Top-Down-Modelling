"""
Universal Market Forecasting Framework - Redesigned Streamlit Application

A completely redesigned, intuitive interface with guided workflows, 
comprehensive explanations, and professional UX design.
"""

import os
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import framework components
from src.config.config_manager import ConfigurationManager
from src.market_analysis.market_analyzer import MarketAnalyzer
from src.distribution.market_distributor import MarketDistributor
from src.data_processing.data_loader import DataLoader

# Import new interface components
from src.streamlit.guided_wizard import GuidedConfigurationWizard
from src.streamlit.smart_data_interface import SmartDataInterface
from src.streamlit.unified_visualization import UnifiedVisualizationInterface
from src.streamlit.help_system import HelpSystem

# Import and apply enhanced visualization fix
try:
    from src.streamlit.enhanced_visualization_fix import fix_enhanced_visualization
    fix_enhanced_visualization()
    logger.info("Applied enhanced visualization column naming fix")
except Exception as e:
    logger.warning(f"Could not apply enhanced visualization fix: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration with improved settings
st.set_page_config(
    page_title="Market Forecasting Framework",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",  # Start with clean interface
    menu_items={
        'Get Help': 'https://github.com/your-repo/docs',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Universal Market Forecasting Framework v2.0 - Professional market analysis and forecasting platform"
    }
)

class MarketForecastingApp:
    """Main application class with redesigned UX"""
    
    def __init__(self):
        self.help_system = HelpSystem()
        self.wizard = GuidedConfigurationWizard()
        self.data_interface = SmartDataInterface()
        self.viz_interface = UnifiedVisualizationInterface()
        
        # Application state
        self.workflows = {
            "Quick Start": {"icon": "⚡", "desc": "Get started in 3 easy steps"},
            "Data Setup": {"icon": "📊", "desc": "Import and configure your data"},
            "Forecasting": {"icon": "🔮", "desc": "Generate market forecasts"},
            "Analysis": {"icon": "📈", "desc": "Analyze results and insights"},
            "Advanced": {"icon": "🔧", "desc": "Advanced configuration and tuning"}
        }
        
        self.initialize_session_state()
        self.apply_custom_styling()
    
    def initialize_session_state(self):
        """Initialize enhanced session state with workflow tracking"""
        defaults = {
            'current_workflow': 'Quick Start',
            'workflow_step': 1,
            'config': None,
            'global_forecast': None,
            'country_historical': None,
            'indicators': {},
            'distributed_market': None,
            'project_settings': {
                'name': 'Untitled Project',
                'description': '',
                'market_type': 'General',
                'created_date': datetime.now().strftime('%Y-%m-%d'),
                'last_modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'user_preferences': {
                'show_help': True,
                'auto_save': True,
                'theme': 'Professional',
                'tutorial_completed': False
            },
            'workflow_progress': {
                'data_loaded': False,
                'config_set': False,
                'forecast_generated': False,
                'analysis_complete': False
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def apply_custom_styling(self):
        """Apply professional custom styling"""
        st.markdown("""
        <style>
        /* Import modern fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styling */
        .main {
            font-family: 'Inter', sans-serif;
        }
        
        /* Header styling */
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .app-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        
        .app-header .subtitle {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 400;
        }
        
        /* Navigation styling */
        .workflow-nav {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .workflow-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid transparent;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .workflow-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }
        
        .workflow-card.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }
        
        .workflow-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        
        .workflow-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .workflow-desc {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        /* Progress bar styling */
        .progress-container {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .progress-step {
            display: inline-block;
            width: 20%;
            text-align: center;
            position: relative;
        }
        
        .progress-step .step-number {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #e9ecef;
            color: #6c757d;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 0.5rem;
            font-weight: 600;
            border: 2px solid #e9ecef;
        }
        
        .progress-step.completed .step-number {
            background: #28a745;
            color: white;
            border-color: #28a745;
        }
        
        .progress-step.current .step-number {
            background: #667eea;
            color: white;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
        }
        
        .progress-step .step-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: #6c757d;
        }
        
        .progress-step.completed .step-label,
        .progress-step.current .step-label {
            color: #495057;
        }
        
        /* Help system styling */
        .help-tooltip {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        
        .help-tooltip .help-title {
            font-weight: 600;
            color: #495057;
            margin-bottom: 0.5rem;
        }
        
        .help-tooltip .help-content {
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        /* Card styling */
        .info-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .success-card {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 1px solid #c3e6cb;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .warning-card {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 1px solid #ffeaa7;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }
        
        /* Metrics styling */
        .metric-container {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            margin: 1rem 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            font-weight: 500;
        }
        
        /* Sidebar improvements */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Hide default streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Responsive design */
        @media (max-width: 768px) {
            .workflow-card {
                margin: 0.25rem;
                padding: 1rem;
            }
            
            .app-header h1 {
                font-size: 2rem;
            }
            
            .progress-step {
                width: 100%;
                margin-bottom: 1rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the application header with project info"""
        project_name = st.session_state.project_settings['name']
        st.markdown(f"""
        <div class="app-header">
            <h1>🚀 Market Forecasting Framework</h1>
            <div class="subtitle">Professional Market Analysis & Forecasting Platform</div>
            <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
                Project: {project_name} | Last Modified: {st.session_state.project_settings['last_modified']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_workflow_navigation(self):
        """Render the main workflow navigation"""
        st.markdown('<div class="workflow-nav">', unsafe_allow_html=True)
        st.markdown("### Choose Your Workflow")
        
        cols = st.columns(len(self.workflows))
        
        for i, (workflow_id, workflow_info) in enumerate(self.workflows.items()):
            with cols[i]:
                is_active = st.session_state.current_workflow == workflow_id
                card_class = "workflow-card active" if is_active else "workflow-card"
                
                if st.button(
                    f"{workflow_info['icon']}\n{workflow_id}\n{workflow_info['desc']}", 
                    key=f"workflow_{workflow_id}",
                    help=f"Switch to {workflow_id} workflow"
                ):
                    st.session_state.current_workflow = workflow_id
                    st.session_state.workflow_step = 1
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_progress_tracker(self):
        """Render workflow progress tracker"""
        current_workflow = st.session_state.current_workflow
        
        if current_workflow == "Quick Start":
            steps = ["Setup", "Data", "Forecast", "Analyze", "Export"]
        elif current_workflow == "Data Setup":
            steps = ["Import", "Validate", "Configure", "Map", "Save"]
        elif current_workflow == "Forecasting":
            steps = ["Method", "Parameters", "Generate", "Calibrate", "Review"]
        elif current_workflow == "Analysis":
            steps = ["Overview", "Trends", "Regions", "Accuracy", "Insights"]
        else:  # Advanced
            steps = ["Config", "Models", "Indicators", "Tuning", "Export"]
        
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        st.markdown(f"**{current_workflow} Progress**")
        
        progress_html = ""
        for i, step in enumerate(steps, 1):
            current_step = st.session_state.workflow_step
            
            # Fixed: Improve step tracking logic with bounds checking
            completed_steps = getattr(st.session_state, 'completed_steps', set())
            
            if i in completed_steps or i < current_step:
                step_class = "progress-step completed"
                step_icon = "✓"
            elif i == current_step and i <= len(steps):  # Bounds check
                step_class = "progress-step current"
                step_icon = str(i)
            else:
                step_class = "progress-step"
                step_icon = str(i) if i <= len(steps) else "?"
            
            progress_html += f"""
            <div class="{step_class}">
                <div class="step-number">{step_icon}</div>
                <div class="step-label">{step}</div>
            </div>
            """
        
        st.markdown(progress_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_help_panel(self):
        """Render contextual help panel"""
        if st.session_state.user_preferences['show_help']:
            current_workflow = st.session_state.current_workflow
            current_step = st.session_state.workflow_step
            
            help_content = self.help_system.get_contextual_help(current_workflow, current_step)
            
            if help_content:
                st.markdown(f"""
                <div class="help-tooltip">
                    <div class="help-title">💡 {help_content['title']}</div>
                    <div class="help-content">{help_content['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_project_status(self):
        """Render project status overview"""
        progress = st.session_state.workflow_progress
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅" if progress['data_loaded'] else "⏳"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{status}</div>
                <div class="metric-label">Data Loaded</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status = "✅" if progress['config_set'] else "⏳"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{status}</div>
                <div class="metric-label">Configuration</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status = "✅" if progress['forecast_generated'] else "⏳"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{status}</div>
                <div class="metric-label">Forecast Ready</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            status = "✅" if progress['analysis_complete'] else "⏳"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{status}</div>
                <div class="metric-label">Analysis Done</div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_workflow_content(self):
        """Render content based on current workflow and step"""
        workflow = st.session_state.current_workflow
        step = st.session_state.workflow_step
        
        if workflow == "Quick Start":
            self.render_quick_start_workflow(step)
        elif workflow == "Data Setup":
            self.render_data_setup_workflow(step)
        elif workflow == "Forecasting":
            self.render_forecasting_workflow(step)
        elif workflow == "Analysis":
            self.render_analysis_workflow(step)
        elif workflow == "Advanced":
            self.render_advanced_workflow(step)
    
    def render_quick_start_workflow(self, step: int):
        """Render Quick Start workflow"""
        if step == 1:
            st.markdown("## 🚀 Welcome to Quick Start!")
            st.markdown("""
            This guided workflow will help you get started with market forecasting in just a few minutes.
            We'll walk you through the essential steps to generate your first forecast.
            """)
            
            # Project setup
            st.markdown("### 📋 Project Setup")
            
            col1, col2 = st.columns(2)
            with col1:
                project_name = st.text_input(
                    "Project Name",
                    value=st.session_state.project_settings['name'],
                    help="Give your forecasting project a descriptive name"
                )
                
                market_type = st.selectbox(
                    "Market Type",
                    ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "General"],
                    help="Select the type of market you're analyzing"
                )
            
            with col2:
                description = st.text_area(
                    "Project Description",
                    value=st.session_state.project_settings['description'],
                    help="Brief description of what you're forecasting"
                )
            
            # Sample data option
            st.markdown("### 📊 Get Started with Sample Data")
            
            if st.button("🎯 Load Sample Data & Continue", help="Load pre-configured sample data to explore the platform"):
                # Load sample data
                from src.streamlit.test_data import generate_all_test_data
                test_data = generate_all_test_data()
                
                st.session_state.global_forecast = test_data['global_forecast']
                st.session_state.country_historical = test_data['country_historical']
                st.session_state.indicators = {}
                
                for name, df in test_data['indicators'].items():
                    indicator_type = 'rank' if 'Rank' in name else 'value'
                    st.session_state.indicators[name] = {
                        'data': df,
                        'meta': {
                            'name': name,
                            'type': indicator_type,
                            'weight': 'auto'
                        }
                    }
                
                # Update project settings
                st.session_state.project_settings.update({
                    'name': project_name,
                    'description': description,
                    'market_type': market_type,
                    'last_modified': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Fixed: Update progress consistently
                st.session_state.workflow_progress['data_loaded'] = True
                st.session_state.workflow_progress['config_set'] = True  # Also mark config as set
                st.session_state.workflow_step = 3  # Skip to forecasting
                
                # Track that we've completed steps 1 and 2
                if 'completed_steps' not in st.session_state:
                    st.session_state.completed_steps = set()
                st.session_state.completed_steps.update([1, 2])
                
                st.success("✅ Sample data loaded! Moving to forecasting step...")
                st.rerun()
            
            # Navigation
            col1, col2 = st.columns([1, 1])
            with col2:
                if st.button("Continue with Own Data →"):
                    st.session_state.workflow_step = 2
                    st.rerun()
        
        elif step == 2:
            st.markdown("## 📊 Data Upload")
            
            # Fixed: Add error handling for missing interface components
            try:
                self.data_interface.render_quick_upload()
            except Exception as e:
                st.error(f"Data upload interface not available: {e}")
                st.info("Please use the basic data input functionality instead.")
                if st.button("Use Basic Data Input"):
                    st.session_state.active_page = "Data Input"
                    st.rerun()
            
            # Navigation
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back"):
                    st.session_state.workflow_step = 1
                    st.rerun()
            with col2:
                if st.session_state.global_forecast is not None:
                    if st.button("Continue to Forecasting →"):
                        st.session_state.workflow_step = 3
                        st.rerun()
        
        elif step == 3:
            st.markdown("## 🔮 Generate Forecast")
            
            # Fixed: Add error handling for missing wizard components
            try:
                self.wizard.render_quick_forecast_setup()
            except Exception as e:
                st.error(f"Forecast setup interface not available: {e}")
                st.info("Please use the manual forecasting setup.")
                if st.button("Use Manual Forecasting"):
                    st.session_state.active_page = "Global Forecasting"
                    st.rerun()
            
            # Navigation
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back"):
                    st.session_state.workflow_step = 2
                    st.rerun()
            with col2:
                if st.session_state.distributed_market is not None:
                    if st.button("View Results →"):
                        st.session_state.workflow_step = 4
                        st.rerun()
        
        elif step == 4:
            st.markdown("## 📈 Analyze Results")
            
            # Fixed: Add error handling for missing visualization components
            try:
                self.viz_interface.render_quick_overview()
            except Exception as e:
                st.error(f"Visualization interface not available: {e}")
                st.info("Please use the manual visualization features.")
                if st.button("Use Manual Visualization"):
                    st.session_state.active_page = "Visualization"
                    st.rerun()
            
            # Navigation
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back to Forecasting"):
                    st.session_state.workflow_step = 3
                    st.rerun()
            with col2:
                if st.button("Export Results →"):
                    st.session_state.workflow_step = 5
                    st.rerun()
        
        elif step == 5:
            st.markdown("## 📤 Export & Share")
            self.render_export_interface()
            
            # Navigation
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back to Analysis"):
                    st.session_state.workflow_step = 4
                    st.rerun()
            with col2:
                if st.button("🎉 Complete!"):
                    st.session_state.workflow_progress['analysis_complete'] = True
                    st.balloons()
                    st.success("Congratulations! You've completed your first forecast!")
    
    def render_data_setup_workflow(self, step: int):
        """Render Data Setup workflow"""
        st.markdown("## 📊 Data Setup Workflow")
        self.data_interface.render_comprehensive_interface(step)
    
    def render_forecasting_workflow(self, step: int):
        """Render Forecasting workflow"""
        st.markdown("## 🔮 Forecasting Workflow")
        self.wizard.render_forecasting_wizard(step)
    
    def render_analysis_workflow(self, step: int):
        """Render Analysis workflow"""
        st.markdown("## 📈 Analysis Workflow")
        self.viz_interface.render_analysis_workflow(step)
    
    def render_advanced_workflow(self, step: int):
        """Render Advanced workflow"""
        st.markdown("## 🔧 Advanced Configuration")
        self.wizard.render_advanced_configuration(step)
    
    def render_export_interface(self):
        """Render export and sharing interface"""
        st.markdown("### 📤 Export Your Results")
        
        if st.session_state.distributed_market is not None:
            export_options = st.multiselect(
                "Select Export Formats",
                ["Excel Report", "PDF Summary", "CSV Data", "Interactive Dashboard", "API Endpoint"],
                default=["Excel Report"],
                help="Choose which formats to export your analysis results"
            )
            
            if st.button("Generate Exports"):
                with st.spinner("Generating exports..."):
                    # Mock export generation
                    import time
                    time.sleep(2)
                    st.success("✅ Exports generated successfully!")
                    
                    for format_type in export_options:
                        st.download_button(
                            f"Download {format_type}",
                            data="Mock export data",
                            file_name=f"forecast_results.{format_type.lower().replace(' ', '_')}",
                            key=f"download_{format_type}"
                        )
        else:
            st.warning("No forecast results available to export. Please complete the forecasting step first.")
    
    def render_sidebar_controls(self):
        """Render sidebar with controls and settings"""
        with st.sidebar:
            st.markdown("### ⚙️ Settings")
            
            # User preferences
            st.session_state.user_preferences['show_help'] = st.checkbox(
                "Show Help Tips", 
                value=st.session_state.user_preferences['show_help']
            )
            
            st.session_state.user_preferences['auto_save'] = st.checkbox(
                "Auto Save", 
                value=st.session_state.user_preferences['auto_save']
            )
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Reset Project"):
                for key in ['global_forecast', 'country_historical', 'indicators', 'distributed_market']:
                    st.session_state[key] = None if key != 'indicators' else {}
                st.session_state.workflow_progress = {
                    'data_loaded': False,
                    'config_set': False,
                    'forecast_generated': False,
                    'analysis_complete': False
                }
                st.success("Project reset!")
                st.rerun()
            
            if st.button("💾 Save Project"):
                # Mock save functionality
                st.success("Project saved!")
            
            if st.button("📁 Load Project"):
                # Mock load functionality
                st.info("Load project functionality coming soon!")
            
            st.markdown("---")
            
            # Project info
            st.markdown("### 📋 Project Info")
            st.text(f"Name: {st.session_state.project_settings['name']}")
            st.text(f"Type: {st.session_state.project_settings['market_type']}")
            st.text(f"Created: {st.session_state.project_settings['created_date']}")
    
    def run(self):
        """Main application entry point"""
        # Render header
        self.render_header()
        
        # Render workflow navigation
        self.render_workflow_navigation()
        
        # Render progress tracker
        self.render_progress_tracker()
        
        # Render project status
        self.render_project_status()
        
        # Render help panel
        self.render_help_panel()
        
        # Render main content based on workflow
        self.render_workflow_content()
        
        # Render sidebar
        self.render_sidebar_controls()


# Create supporting interface components
def create_interface_components():
    """Create the supporting interface component files"""
    
    # This is a placeholder - we'll create the actual component files next
    pass


# Main application execution
if __name__ == "__main__":
    app = MarketForecastingApp()
    app.run()