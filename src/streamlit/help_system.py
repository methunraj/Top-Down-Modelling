"""
Comprehensive Help System
Provides contextual help, tutorials, and explanations throughout the application
"""

import streamlit as st
from typing import Dict, List, Any, Optional

class HelpSystem:
    """Comprehensive help system with contextual guidance"""
    
    def __init__(self):
        self.help_content = {
            "Quick Start": {
                1: {
                    "title": "Welcome to Quick Start!",
                    "content": """
                    This workflow helps you generate your first market forecast in minutes.
                    
                    **What you'll do:**
                    • Set up your project basics
                    • Load sample data or upload your own
                    • Generate a forecast automatically
                    • View and analyze results
                    
                    **Tip:** Use sample data to explore the platform, then return with your own data.
                    """
                },
                2: {
                    "title": "Data Upload Made Simple",
                    "content": """
                    Upload your market data in CSV or Excel format.
                    
                    **Required data:**
                    • Global market size by year
                    • Country-level data (optional but recommended)
                    
                    **Supported formats:**
                    • CSV files (comma-separated)
                    • Excel spreadsheets (.xlsx)
                    
                    **Need help?** Download our data template to see the expected format.
                    """
                },
                3: {
                    "title": "Automatic Forecast Generation",
                    "content": """
                    We'll automatically choose the best forecasting method for your data.
                    
                    **What happens:**
                    • Data quality analysis
                    • Method selection based on your market
                    • Forecast generation with constraints
                    • Distribution across countries
                    
                    **Pro tip:** Review the method selection to understand why we chose this approach.
                    """
                },
                4: {
                    "title": "Understanding Your Results",
                    "content": """
                    Your forecast results include key metrics and visualizations.
                    
                    **Key metrics to watch:**
                    • CAGR (Compound Annual Growth Rate)
                    • Market size projections
                    • Country rankings
                    • Regional distribution
                    
                    **Next steps:** Explore detailed analysis or export your results.
                    """
                }
            },
            "Data Setup": {
                1: {
                    "title": "Importing Your Data",
                    "content": """
                    Choose the right data template for your upload.
                    
                    **Data types:**
                    • **Global Market Data**: Overall market size by year
                    • **Country Historical**: Market performance by country
                    • **Economic Indicators**: GDP, population, etc.
                    
                    **File formats:** We support CSV and Excel files with automatic format detection.
                    """
                },
                2: {
                    "title": "Data Quality Validation",
                    "content": """
                    We automatically check your data quality and suggest improvements.
                    
                    **Quality checks:**
                    • Missing values detection
                    • Data type validation
                    • Range and consistency checks
                    • Duplicate detection
                    
                    **Quality score:** Aim for 60+ for reliable forecasts. Higher scores = better results.
                    """
                },
                3: {
                    "title": "Column Mapping",
                    "content": """
                    Map your data columns to our expected format.
                    
                    **Smart mapping:** We automatically suggest mappings based on column names.
                    
                    **Required columns vary by data type:**
                    • Global data: Year, Market_Size
                    • Country data: Country, Year, Value
                    • Indicators: Country, Year, Value
                    
                    **Tip:** Consistent column naming across files speeds up future uploads.
                    """
                }
            },
            "Forecasting": {
                1: {
                    "title": "Choosing the Right Method",
                    "content": """
                    Different forecasting methods work better for different markets.
                    
                    **Statistical Models:** Best for stable markets with good historical data
                    **Machine Learning:** Ideal for complex markets with many variables
                    **Hybrid Ensemble:** Combines multiple methods for maximum accuracy
                    **Bayesian Hierarchical:** Provides uncertainty estimates for strategic planning
                    
                    **Need guidance?** Use our smart recommendations based on your market characteristics.
                    """
                },
                2: {
                    "title": "Parameter Configuration",
                    "content": """
                    Fine-tune your model parameters for optimal performance.
                    
                    **Key parameters:**
                    • Model complexity (higher = more flexible, risk of overfitting)
                    • Validation split (20% is usually good)
                    • Feature engineering (improves accuracy)
                    
                    **Default settings** work well for most cases. Advanced users can customize everything.
                    """
                },
                3: {
                    "title": "Forecast Generation Process",
                    "content": """
                    Your forecast is generated through multiple steps:
                    
                    1. **Data preprocessing**: Clean and prepare your data
                    2. **Model training**: Learn patterns from historical data  
                    3. **Forecast generation**: Project future values
                    4. **Constraint application**: Ensure realistic bounds
                    5. **Country distribution**: Allocate global forecast
                    
                    **Time required:** 1-5 minutes depending on data size and complexity.
                    """
                },
                4: {
                    "title": "Auto-Calibration Benefits",
                    "content": """
                    Auto-calibration improves forecast accuracy automatically.
                    
                    **What it does:**
                    • Analyzes historical prediction errors
                    • Adjusts model parameters
                    • Reduces systematic bias
                    • Optimizes performance metrics
                    
                    **When to use:** Always recommended for production forecasts. Skip only for quick exploratory analysis.
                    """
                }
            },
            "Analysis": {
                1: {
                    "title": "Comprehensive Analysis Options",
                    "content": """
                    Choose the analysis that matches your goals:
                    
                    **Overview:** Executive dashboard with key insights
                    **Growth Analysis:** Deep dive into growth patterns and drivers
                    **Market Share:** Competitive positioning and concentration
                    **Regional Analysis:** Geographical patterns and opportunities
                    **Forecast Accuracy:** Model performance and reliability
                    
                    **Tip:** Start with Overview, then drill down into specific areas of interest.
                    """
                },
                2: {
                    "title": "Interpreting Visualizations",
                    "content": """
                    Each chart type reveals different insights:
                    
                    **Line charts:** Show trends and patterns over time
                    **Bar charts:** Compare values across categories
                    **Maps:** Reveal geographical patterns
                    **Pie charts:** Show proportional relationships
                    
                    **Interactive features:** Hover for details, click to filter, zoom to focus on specific periods.
                    """
                }
            },
            "Advanced": {
                1: {
                    "title": "Advanced Configuration Options",
                    "content": """
                    Fine-tune every aspect of your forecasting system.
                    
                    **Model Configuration:** Deep parameter tuning for experts
                    **Data Processing:** Custom preprocessing pipelines
                    **Indicator Weighting:** Optimize indicator contributions
                    **Constraint Tuning:** Custom business rules and limits
                    **Export Settings:** API endpoints and custom reports
                    
                    **Warning:** Advanced settings require technical expertise. Default settings work well for most users.
                    """
                }
            }
        }
        
        self.tutorials = {
            "First Time User": {
                "title": "Complete Beginner's Guide",
                "description": "Step-by-step tutorial for first-time users",
                "steps": [
                    {
                        "title": "Welcome & Overview",
                        "content": "Learn about the platform capabilities and main workflows",
                        "action": "Start Quick Start workflow"
                    },
                    {
                        "title": "Load Sample Data",
                        "content": "Use our pre-loaded sample data to explore features",
                        "action": "Click 'Load Sample Data' on the home page"
                    },
                    {
                        "title": "Generate Your First Forecast",
                        "content": "Create a market forecast using automated settings",
                        "action": "Follow the Quick Start workflow"
                    },
                    {
                        "title": "Explore Visualizations",
                        "content": "View different analysis types and charts",
                        "action": "Try each visualization in the Analysis workflow"
                    },
                    {
                        "title": "Export Results",
                        "content": "Download your analysis in various formats",
                        "action": "Use the Export functionality"
                    }
                ]
            },
            "Data Professional": {
                "title": "Data Professional Quick Start",
                "description": "Fast track for users familiar with data analysis",
                "steps": [
                    {
                        "title": "Upload Your Data",
                        "content": "Use the comprehensive Data Setup workflow for full control",
                        "action": "Go to Data Setup workflow"
                    },
                    {
                        "title": "Choose Advanced Methods",
                        "content": "Select ML or ensemble methods for sophisticated modeling",
                        "action": "Use Forecasting workflow with custom settings"
                    },
                    {
                        "title": "Model Validation",
                        "content": "Review forecast accuracy and model performance",
                        "action": "Check Forecast Accuracy analysis"
                    },
                    {
                        "title": "Advanced Analytics",
                        "content": "Explore regional analysis and growth patterns",
                        "action": "Use all visualization components"
                    }
                ]
            },
            "Business User": {
                "title": "Business User Guide",
                "description": "Focus on insights and business implications",
                "steps": [
                    {
                        "title": "Quick Setup",
                        "content": "Get started quickly with minimal technical details",
                        "action": "Use Quick Start with sample data"
                    },
                    {
                        "title": "Understanding Results",
                        "content": "Learn to interpret forecasts and key metrics",
                        "action": "Focus on Overview and Growth Analysis"
                    },
                    {
                        "title": "Market Insights",
                        "content": "Identify opportunities and trends",
                        "action": "Use Regional and Market Share analysis"
                    },
                    {
                        "title": "Business Reporting",
                        "content": "Export professional reports for stakeholders",
                        "action": "Generate PDF and Excel reports"
                    }
                ]
            }
        }
        
        self.faqs = {
            "Getting Started": [
                {
                    "question": "What data do I need to get started?",
                    "answer": """
                    **Minimum required:** Global market size data by year (at least 3-5 years)
                    
                    **Recommended additions:**
                    • Country-level historical data
                    • Economic indicators (GDP, population, etc.)
                    • Market-specific indicators
                    
                    **Formats:** CSV or Excel files with consistent column naming
                    """
                },
                {
                    "question": "How accurate are the forecasts?",
                    "answer": """
                    Forecast accuracy depends on several factors:
                    
                    **Typical accuracy ranges:**
                    • Mature markets: 80-95% accuracy (MAPE 5-20%)
                    • Emerging markets: 70-85% accuracy (MAPE 15-30%)
                    • Highly volatile markets: 60-75% accuracy (MAPE 25-40%)
                    
                    **Improving accuracy:**
                    • Use more historical data
                    • Include relevant indicators
                    • Apply auto-calibration
                    • Use ensemble methods
                    """
                },
                {
                    "question": "Can I forecast any type of market?",
                    "answer": """
                    Yes! Our framework is market-agnostic and works for:
                    
                    **Technology markets:** Software, hardware, telecommunications
                    **Healthcare:** Medical devices, pharmaceuticals, services  
                    **Financial services:** Banking, insurance, fintech
                    **Retail & Consumer:** Products, services, e-commerce
                    **Industrial:** Manufacturing, energy, materials
                    
                    **The key:** Having relevant historical data and understanding market drivers
                    """
                }
            ],
            "Data & Upload": [
                {
                    "question": "What file formats are supported?",
                    "answer": """
                    **Supported formats:**
                    • CSV files (comma-separated values)
                    • Excel files (.xlsx, .xls)
                    
                    **Requirements:**
                    • UTF-8 encoding for international characters
                    • Consistent column headers
                    • No merged cells in Excel files
                    • Maximum file size: 50MB
                    """
                },
                {
                    "question": "How do I structure my data correctly?",
                    "answer": """
                    **Global Market Data format:**
                    ```
                    Year, Market_Size
                    2020, 1000
                    2021, 1100
                    2022, 1250
                    ```
                    
                    **Country Data format:**
                    ```
                    Country, Year, Value
                    USA, 2020, 400
                    China, 2020, 300
                    ```
                    
                    **Tips:**
                    • Use consistent country names
                    • Include all years for each country
                    • Use numeric values only (no currency symbols)
                    """
                },
                {
                    "question": "What if I have missing data?",
                    "answer": """
                    **Missing data handling:**
                    • <5% missing: Automatically interpolated
                    • 5-20% missing: Warning shown, still processable
                    • >20% missing: Requires manual review
                    
                    **Solutions:**
                    • Use forward/backward fill for time series gaps
                    • Interpolate missing values
                    • Remove incomplete countries/years
                    • Use proxy data from similar markets
                    
                    **Best practice:** Address missing data before upload for best results
                    """
                }
            ],
            "Forecasting Methods": [
                {
                    "question": "Which forecasting method should I choose?",
                    "answer": """
                    **Quick guide:**
                    
                    **Statistical Models** - Choose if:
                    • Stable, mature market
                    • Good historical data (5+ years)
                    • Predictable patterns
                    • Want fast, interpretable results
                    
                    **Machine Learning** - Choose if:
                    • Dynamic, complex market
                    • Many external factors
                    • Large amounts of data
                    • Need high accuracy
                    
                    **Hybrid Ensemble** - Choose if:
                    • Critical business decisions
                    • Want maximum accuracy
                    • Have quality data
                    • Can afford longer computation time
                    
                    **Still unsure?** Use our smart recommendations feature!
                    """
                },
                {
                    "question": "What is auto-calibration and should I use it?",
                    "answer": """
                    **Auto-calibration** automatically optimizes your forecast based on historical accuracy.
                    
                    **What it does:**
                    • Analyzes prediction errors
                    • Adjusts model parameters
                    • Reduces systematic bias
                    • Improves accuracy metrics
                    
                    **When to use:**
                    ✅ Production forecasts
                    ✅ Business planning
                    ✅ Strategic decisions
                    
                    **When to skip:**
                    ❌ Quick exploratory analysis
                    ❌ Very limited historical data
                    ❌ Proof-of-concept work
                    """
                }
            ],
            "Analysis & Visualization": [
                {
                    "question": "How do I interpret the visualizations?",
                    "answer": """
                    **Key visualization types:**
                    
                    **Line Charts:**
                    • Show trends over time
                    • Rising line = growth, falling = decline
                    • Steeper slope = faster change
                    
                    **Bar Charts:**
                    • Compare values across categories
                    • Longer bars = higher values
                    • Use for country/region comparisons
                    
                    **Maps:**
                    • Show geographical patterns
                    • Color intensity = value magnitude
                    • Identify regional clusters
                    
                    **Pie Charts:**
                    • Show proportional relationships
                    • Larger slices = bigger share
                    • Good for market share analysis
                    
                    **Interactive features:** Hover for details, click to filter, zoom for focus
                    """
                },
                {
                    "question": "What do the accuracy metrics mean?",
                    "answer": """
                    **Key accuracy metrics explained:**
                    
                    **MAPE (Mean Absolute Percentage Error):**
                    • <10% = Excellent accuracy
                    • 10-20% = Good accuracy  
                    • 20-30% = Acceptable accuracy
                    • >30% = Poor accuracy
                    
                    **R² (R-squared):**
                    • >0.9 = Excellent model fit
                    • 0.7-0.9 = Good model fit
                    • 0.5-0.7 = Moderate model fit
                    • <0.5 = Poor model fit
                    
                    **Directional Accuracy:**
                    • >80% = Excellent trend prediction
                    • 70-80% = Good trend prediction
                    • 60-70% = Fair trend prediction
                    • <60% = Poor trend prediction
                    """
                }
            ]
        }
        
        self.glossary = {
            "CAGR": "Compound Annual Growth Rate - the rate of return that would be required for an investment to grow from its beginning balance to its ending balance over time",
            "MAPE": "Mean Absolute Percentage Error - a measure of prediction accuracy expressed as a percentage",
            "R²": "R-squared - statistical measure representing the proportion of variance in the dependent variable explained by the model",
            "Ensemble Method": "Combining multiple forecasting models to improve overall prediction accuracy",
            "Auto-calibration": "Automatic optimization of model parameters based on historical performance",
            "Market Share": "The percentage of total sales in a market captured by a particular company or country",
            "Indicator": "Economic or market variable used to predict or explain market behavior",
            "Hierarchical Model": "Statistical model that accounts for data structure at multiple levels (e.g., countries within regions)",
            "Cross-validation": "Method for assessing model performance by training on part of data and testing on another part",
            "Confidence Interval": "Range of values that likely contains the true value with a certain level of confidence"
        }
    
    def get_contextual_help(self, workflow: str, step: int) -> Optional[Dict[str, str]]:
        """Get contextual help for current workflow and step"""
        if workflow in self.help_content and step in self.help_content[workflow]:
            return self.help_content[workflow][step]
        return None
    
    def render_help_sidebar(self):
        """Render help sidebar with quick access to common help topics"""
        st.sidebar.markdown("### 🆘 Quick Help")
        
        # Quick help topics
        help_topics = [
            "Getting Started",
            "Data Upload Tips", 
            "Choosing Methods",
            "Understanding Results"
        ]
        
        selected_topic = st.sidebar.selectbox("Help Topic", help_topics)
        
        if selected_topic == "Getting Started":
            st.sidebar.markdown("""
            **Quick Start Guide:**
            1. Load sample data or upload your own
            2. Choose forecasting method (or use auto)
            3. Generate forecast
            4. Analyze results
            5. Export findings
            """)
        
        elif selected_topic == "Data Upload Tips":
            st.sidebar.markdown("""
            **Data Requirements:**
            • CSV or Excel format
            • Consistent column names
            • No missing critical data
            • At least 3 years of history
            
            **Need help?** Download our template!
            """)
        
        elif selected_topic == "Choosing Methods":
            st.sidebar.markdown("""
            **Quick Selection:**
            • **Stable market** → Statistical
            • **Complex market** → Machine Learning  
            • **Critical decisions** → Ensemble
            • **Uncertainty needed** → Bayesian
            
            **Not sure?** Use smart recommendations!
            """)
        
        elif selected_topic == "Understanding Results":
            st.sidebar.markdown("""
            **Key Metrics:**
            • **CAGR**: Growth rate
            • **MAPE**: Accuracy (lower = better)
            • **R²**: Model fit (higher = better)
            
            **Charts:** Hover for details, click to interact
            """)
    
    def render_comprehensive_help(self):
        """Render comprehensive help interface"""
        st.markdown("# 🆘 Help Center")
        st.markdown("Find answers, tutorials, and guidance for using the Market Forecasting Framework")
        
        # Help navigation
        help_tabs = st.tabs(["📚 Tutorials", "❓ FAQ", "📖 Glossary", "💡 Tips & Tricks"])
        
        with help_tabs[0]:
            self.render_tutorials()
        
        with help_tabs[1]:
            self.render_faqs()
        
        with help_tabs[2]:
            self.render_glossary()
        
        with help_tabs[3]:
            self.render_tips_tricks()
    
    def render_tutorials(self):
        """Render interactive tutorials"""
        st.markdown("## 📚 Interactive Tutorials")
        st.markdown("Choose a tutorial based on your role and experience level:")
        
        # Tutorial selection
        tutorial_cols = st.columns(len(self.tutorials))
        
        for i, (tutorial_id, tutorial_info) in enumerate(self.tutorials.items()):
            with tutorial_cols[i]:
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                           border: 2px solid #e2e8f0; height: 250px;">
                    <h4>{tutorial_info['title']}</h4>
                    <p>{tutorial_info['description']}</p>
                    <p><strong>Steps:</strong> {len(tutorial_info['steps'])}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Start {tutorial_id} Tutorial", key=f"tutorial_{tutorial_id}"):
                    st.session_state.active_tutorial = tutorial_id
                    st.session_state.tutorial_step = 0
                    st.rerun()
        
        # Render active tutorial
        if 'active_tutorial' in st.session_state:
            self.render_active_tutorial()
    
    def render_active_tutorial(self):
        """Render the currently active tutorial"""
        tutorial_id = st.session_state.active_tutorial
        tutorial = self.tutorials[tutorial_id]
        current_step = st.session_state.get('tutorial_step', 0)
        
        st.markdown(f"### 🎓 {tutorial['title']}")
        
        # Progress bar
        progress = (current_step + 1) / len(tutorial['steps'])
        st.progress(progress)
        st.markdown(f"Step {current_step + 1} of {len(tutorial['steps'])}")
        
        # Current step content
        if current_step < len(tutorial['steps']):
            step = tutorial['steps'][current_step]
            
            st.markdown(f"#### {step['title']}")
            st.markdown(step['content'])
            
            if 'action' in step:
                st.info(f"**Action:** {step['action']}")
            
            # Navigation
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if current_step > 0:
                    if st.button("← Previous"):
                        st.session_state.tutorial_step = current_step - 1
                        st.rerun()
            
            with col2:
                if st.button("Exit Tutorial"):
                    del st.session_state.active_tutorial
                    del st.session_state.tutorial_step
                    st.rerun()
            
            with col3:
                if current_step < len(tutorial['steps']) - 1:
                    if st.button("Next →"):
                        st.session_state.tutorial_step = current_step + 1
                        st.rerun()
                else:
                    if st.button("Complete Tutorial"):
                        st.success("🎉 Tutorial completed!")
                        del st.session_state.active_tutorial
                        del st.session_state.tutorial_step
                        st.rerun()
    
    def render_faqs(self):
        """Render frequently asked questions"""
        st.markdown("## ❓ Frequently Asked Questions")
        
        # FAQ categories
        for category, questions in self.faqs.items():
            st.markdown(f"### {category}")
            
            for i, faq in enumerate(questions):
                with st.expander(faq['question']):
                    st.markdown(faq['answer'])
    
    def render_glossary(self):
        """Render glossary of terms"""
        st.markdown("## 📖 Glossary")
        st.markdown("Key terms and concepts used in market forecasting:")
        
        # Search functionality
        search_term = st.text_input("🔍 Search glossary", placeholder="Enter term to search...")
        
        for term, definition in self.glossary.items():
            if not search_term or search_term.lower() in term.lower() or search_term.lower() in definition.lower():
                with st.expander(f"**{term}**"):
                    st.markdown(definition)
    
    def render_tips_tricks(self):
        """Render tips and tricks for better forecasting"""
        st.markdown("## 💡 Tips & Tricks")
        
        tips_categories = {
            "Data Quality": [
                "Use at least 5 years of historical data for stable forecasts",
                "Ensure consistent time periods (monthly, quarterly, annual)",
                "Clean outliers before uploading - they can skew forecasts",
                "Include relevant economic indicators for better distribution",
                "Validate data sources and check for inconsistencies"
            ],
            "Method Selection": [
                "Start with statistical methods for stable, predictable markets",
                "Use machine learning for complex markets with many variables",
                "Ensemble methods provide the best accuracy for critical forecasts",
                "Consider Bayesian methods when uncertainty quantification is important",
                "Test multiple methods and compare results"
            ],
            "Improving Accuracy": [
                "Always use auto-calibration for production forecasts",
                "Include more indicators for better country distribution",
                "Set realistic growth constraints based on market knowledge",
                "Regularly update forecasts with new data",
                "Validate results with market experts"
            ],
            "Visualization": [
                "Start with Overview dashboard for executive summaries",
                "Use Regional Analysis to identify geographic opportunities", 
                "Growth Analysis helps understand market dynamics",
                "Market Share analysis reveals competitive landscapes",
                "Export charts as PNG for presentations"
            ],
            "Performance": [
                "Larger datasets require more processing time",
                "Ensemble methods take longer but provide better accuracy",
                "Use sampling for very large datasets (>100K rows)",
                "Cache results to avoid re-computation",
                "Consider data reduction techniques for faster processing"
            ]
        }
        
        for category, tips in tips_categories.items():
            st.markdown(f"### {category}")
            
            for tip in tips:
                st.markdown(f"💡 {tip}")
            
            st.markdown("---")
    
    def render_contextual_tooltip(self, element_id: str, content: str):
        """Render contextual tooltip for specific UI elements"""
        st.markdown(f"""
        <div class="tooltip" data-element="{element_id}">
            ❓
            <span class="tooltiptext">{content}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def get_method_recommendation_help(self, market_characteristics: Dict[str, str]) -> str:
        """Provide method recommendation based on market characteristics"""
        maturity = market_characteristics.get('maturity', '')
        data_quality = market_characteristics.get('data_quality', '')
        volatility = market_characteristics.get('volatility', '')
        external_factors = market_characteristics.get('external_factors', '')
        
        if volatility in ['Very Stable', 'Moderately Stable'] and data_quality in ['Excellent', 'Good']:
            return """
            **Recommended: Statistical Models**
            
            Your stable market with good historical data is perfect for statistical forecasting methods.
            These models are fast, interpretable, and work well for predictable markets.
            
            **Why this choice:**
            • Low volatility suggests predictable patterns
            • Good data quality enables reliable model training
            • Statistical methods excel in stable environments
            """
        
        elif external_factors in ['Significant', 'Dominant'] or volatility in ['Somewhat Volatile', 'Highly Volatile']:
            return """
            **Recommended: Machine Learning**
            
            Your dynamic market with external influences requires adaptive ML algorithms
            that can capture complex relationships and adjust to changing conditions.
            
            **Why this choice:**
            • High volatility needs adaptive modeling
            • External factors require complex pattern recognition
            • ML methods excel with multiple variables
            """
        
        else:
            return """
            **Recommended: Hybrid Ensemble**
            
            Your market characteristics suggest a balanced approach combining multiple
            methods for robust predictions across different scenarios.
            
            **Why this choice:**
            • Mixed characteristics benefit from multiple approaches
            • Ensemble methods reduce individual model bias
            • Provides robust predictions across scenarios
            """