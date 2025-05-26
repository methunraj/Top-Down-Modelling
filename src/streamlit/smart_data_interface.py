"""
Smart Data Interface
Intelligent data upload and configuration with comprehensive validation and help
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import io
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

class SmartDataInterface:
    """Intelligent data interface with guided upload and validation"""
    
    def __init__(self):
        self.data_templates = {
            "Global Market Data": {
                "description": "Historical global market size data",
                "required_columns": ["Year", "Market_Size"],
                "optional_columns": ["Region", "Segment", "Currency"],
                "example_data": {
                    "Year": [2020, 2021, 2022, 2023],
                    "Market_Size": [1000, 1100, 1250, 1400],
                    "Currency": ["USD", "USD", "USD", "USD"]
                },
                "help_text": "Upload your global market size data by year. This is the foundation for all forecasting."
            },
            "Country Historical Data": {
                "description": "Historical market data by country",
                "required_columns": ["Country", "Year", "Value"],
                "optional_columns": ["Country_Code", "Region", "Currency"],
                "example_data": {
                    "Country": ["USA", "China", "Germany", "Japan"],
                    "Year": [2023, 2023, 2023, 2023],
                    "Value": [400, 300, 150, 120],
                    "Country_Code": ["US", "CN", "DE", "JP"]
                },
                "help_text": "Historical market performance by country helps understand distribution patterns."
            },
            "Economic Indicators": {
                "description": "Economic indicators for market distribution",
                "required_columns": ["Country", "Year", "Value"],
                "optional_columns": ["Indicator_Name", "Source", "Units"],
                "example_data": {
                    "Country": ["USA", "China", "Germany", "Japan"],
                    "Year": [2023, 2023, 2023, 2023],
                    "Value": [26900, 17700, 4259, 4937],
                    "Indicator_Name": ["GDP", "GDP", "GDP", "GDP"],
                    "Units": ["Billion USD", "Billion USD", "Billion USD", "Billion USD"]
                },
                "help_text": "Economic indicators like GDP, population, technology adoption that influence market distribution."
            }
        }
        
        self.validation_rules = {
            "Year": {
                "type": "numeric",
                "min_value": 1900,
                "max_value": 2050,
                "description": "Year should be between 1900 and 2050"
            },
            "Market_Size": {
                "type": "numeric",
                "min_value": 0,
                "description": "Market size must be positive"
            },
            "Value": {
                "type": "numeric", 
                "min_value": 0,
                "description": "Values must be positive"
            },
            "Country": {
                "type": "text",
                "min_length": 2,
                "description": "Country names should be at least 2 characters"
            }
        }
        
        self.file_format_help = {
            "CSV": {
                "description": "Comma-separated values file",
                "pros": ["Universal compatibility", "Human readable", "Lightweight"],
                "cons": ["No data types", "Encoding issues possible"],
                "best_for": "Simple data exchange"
            },
            "Excel": {
                "description": "Microsoft Excel spreadsheet",
                "pros": ["Preserves formatting", "Multiple sheets", "Data types"],
                "cons": ["Larger file size", "Proprietary format"],
                "best_for": "Complex data with multiple tables"
            }
        }
    
    def render_data_upload_help(self):
        """Render comprehensive data upload help"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%); 
                   padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                   border-left: 4px solid #0ea5e9;">
            <h4>📚 Data Upload Guide</h4>
            <p>Follow these steps to ensure successful data upload:</p>
            <ol>
                <li><strong>Prepare your data:</strong> Ensure consistent formatting and no missing critical values</li>
                <li><strong>Choose the right format:</strong> CSV for simple data, Excel for complex datasets</li>
                <li><strong>Map columns correctly:</strong> Match your data columns to our expected format</li>
                <li><strong>Validate data quality:</strong> Review our automated quality checks</li>
                <li><strong>Preview results:</strong> Check the data preview before proceeding</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    def render_file_format_guidance(self):
        """Render file format selection guidance"""
        st.markdown("### 📁 Choose File Format")
        
        format_cols = st.columns(2)
        
        for i, (format_name, format_info) in enumerate(self.file_format_help.items()):
            with format_cols[i]:
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                           border: 2px solid #e2e8f0; height: 200px;">
                    <h4>{format_name}</h4>
                    <p><em>{format_info['description']}</em></p>
                    <p><strong>Best for:</strong> {format_info['best_for']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_data_template_selector(self):
        """Render data template selector with examples"""
        st.markdown("### 📋 Data Template Guide")
        
        template_choice = st.selectbox(
            "What type of data are you uploading?",
            list(self.data_templates.keys()),
            help="Select the type of data to see format requirements and examples"
        )
        
        if template_choice:
            template = self.data_templates[template_choice]
            
            # Template information
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <h4>{template_choice}</h4>
                <p>{template['description']}</p>
                <p><em>{template['help_text']}</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Column requirements
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Required Columns:**")
                for col in template['required_columns']:
                    st.markdown(f"✅ `{col}`")
            
            with col2:
                st.markdown("**Optional Columns:**")
                for col in template['optional_columns']:
                    st.markdown(f"➕ `{col}`")
            
            # Example data
            st.markdown("**Example Data:**")
            example_df = pd.DataFrame(template['example_data'])
            st.dataframe(example_df, use_container_width=True)
            
            # Download template
            if st.button(f"📥 Download {template_choice} Template"):
                csv = example_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Template",
                    data=csv,
                    file_name=f"{template_choice.lower().replace(' ', '_')}_template.csv",
                    mime="text/csv"
                )
        
        return template_choice
    
    def validate_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Comprehensive data quality validation"""
        validation_results = {
            "overall_score": 0,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "statistics": {}
        }
        
        # Basic statistics
        validation_results["statistics"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum()
        }
        
        # Check required columns
        if data_type in self.data_templates:
            template = self.data_templates[data_type]
            required_cols = template['required_columns']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                validation_results["issues"].append(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                validation_results["overall_score"] += 30
        
        # Data type validation
        for col in df.columns:
            if col in self.validation_rules:
                rule = self.validation_rules[col]
                
                if rule["type"] == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        validation_results["issues"].append(f"Column '{col}' should be numeric")
                    else:
                        # Check value ranges
                        if "min_value" in rule:
                            invalid_values = (df[col] < rule["min_value"]).sum()
                            if invalid_values > 0:
                                validation_results["warnings"].append(
                                    f"Column '{col}' has {invalid_values} values below minimum ({rule['min_value']})"
                                )
                        
                        if "max_value" in rule:
                            invalid_values = (df[col] > rule["max_value"]).sum()
                            if invalid_values > 0:
                                validation_results["warnings"].append(
                                    f"Column '{col}' has {invalid_values} values above maximum ({rule['max_value']})"
                                )
                
                elif rule["type"] == "text":
                    if "min_length" in rule:
                        short_values = (df[col].str.len() < rule["min_length"]).sum()
                        if short_values > 0:
                            validation_results["warnings"].append(
                                f"Column '{col}' has {short_values} values shorter than {rule['min_length']} characters"
                            )
        
        # Missing data analysis
        missing_pct = (df.isnull().sum() / len(df)) * 100
        for col, pct in missing_pct.items():
            if pct > 20:
                validation_results["issues"].append(f"Column '{col}' has {pct:.1f}% missing values")
            elif pct > 5:
                validation_results["warnings"].append(f"Column '{col}' has {pct:.1f}% missing values")
        
        # Duplicate analysis
        if validation_results["statistics"]["duplicate_rows"] > 0:
            validation_results["warnings"].append(
                f"Found {validation_results['statistics']['duplicate_rows']} duplicate rows"
            )
        
        # Calculate overall score
        if len(validation_results["issues"]) == 0:
            validation_results["overall_score"] += 40
        
        if len(validation_results["warnings"]) <= 2:
            validation_results["overall_score"] += 30
        
        # Generate suggestions
        if validation_results["statistics"]["missing_values"] > 0:
            validation_results["suggestions"].append("Consider filling missing values or removing incomplete rows")
        
        if validation_results["statistics"]["duplicate_rows"] > 0:
            validation_results["suggestions"].append("Remove duplicate rows to avoid skewing results")
        
        return validation_results
    
    def render_validation_results(self, validation_results: Dict[str, Any]):
        """Render comprehensive validation results"""
        st.markdown("### 🔍 Data Quality Assessment")
        
        # Overall score
        score = validation_results["overall_score"]
        if score >= 80:
            score_color = "#28a745"
            score_label = "Excellent"
        elif score >= 60:
            score_color = "#ffc107"
            score_label = "Good"
        elif score >= 40:
            score_color = "#fd7e14"
            score_label = "Needs Improvement"
        else:
            score_color = "#dc3545"
            score_label = "Poor"
        
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                   border: 2px solid {score_color}; margin: 1rem 0;">
            <div style="text-align: center;">
                <h3 style="color: {score_color}; margin: 0;">Quality Score: {score}/100</h3>
                <p style="margin: 0.5rem 0 0 0; color: {score_color};"><strong>{score_label}</strong></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistics
        stats = validation_results["statistics"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{stats['rows']:,}")
        with col2:
            st.metric("Columns", stats['columns'])
        with col3:
            st.metric("Missing Values", stats['missing_values'])
        with col4:
            st.metric("Duplicates", stats['duplicate_rows'])
        
        # Issues and warnings
        if validation_results["issues"]:
            st.markdown("#### ❌ Critical Issues")
            for issue in validation_results["issues"]:
                st.error(issue)
        
        if validation_results["warnings"]:
            st.markdown("#### ⚠️ Warnings")
            for warning in validation_results["warnings"]:
                st.warning(warning)
        
        if validation_results["suggestions"]:
            st.markdown("#### 💡 Suggestions")
            for suggestion in validation_results["suggestions"]:
                st.info(suggestion)
        
        return score >= 60  # Return whether data is acceptable
    
    def render_column_mapping(self, df: pd.DataFrame, data_type: str) -> Dict[str, str]:
        """Render intelligent column mapping interface"""
        st.markdown("### 🔗 Column Mapping")
        st.markdown("Map your data columns to our expected format:")
        
        if data_type in self.data_templates:
            template = self.data_templates[data_type]
            required_cols = template['required_columns']
            optional_cols = template['optional_columns']
            
            column_mapping = {}
            
            # Auto-suggest mappings
            auto_mappings = self.auto_suggest_mappings(df.columns.tolist(), required_cols + optional_cols)
            
            # Required columns
            st.markdown("#### ✅ Required Columns")
            for req_col in required_cols:
                suggested = auto_mappings.get(req_col, None)
                default_index = 0
                
                if suggested and suggested in df.columns:
                    default_index = df.columns.tolist().index(suggested)
                
                mapped_col = st.selectbox(
                    f"Map '{req_col}' to:",
                    ["<Not Mapped>"] + df.columns.tolist(),
                    index=default_index + 1 if suggested else 0,
                    key=f"map_{req_col}",
                    help=f"Select which column contains {req_col} data"
                )
                
                if mapped_col != "<Not Mapped>":
                    column_mapping[req_col] = mapped_col
            
            # Optional columns
            if optional_cols:
                st.markdown("#### ➕ Optional Columns")
                for opt_col in optional_cols:
                    suggested = auto_mappings.get(opt_col, None)
                    default_index = 0
                    
                    if suggested and suggested in df.columns:
                        default_index = df.columns.tolist().index(suggested)
                    
                    mapped_col = st.selectbox(
                        f"Map '{opt_col}' to:",
                        ["<Not Mapped>"] + df.columns.tolist(),
                        index=default_index + 1 if suggested else 0,
                        key=f"map_{opt_col}",
                        help=f"Optional: Select which column contains {opt_col} data"
                    )
                    
                    if mapped_col != "<Not Mapped>":
                        column_mapping[opt_col] = mapped_col
            
            return column_mapping
        
        return {}
    
    def auto_suggest_mappings(self, data_columns: List[str], expected_columns: List[str]) -> Dict[str, str]:
        """Automatically suggest column mappings based on name similarity"""
        mappings = {}
        
        # Common variations for standard columns
        column_variations = {
            "Year": ["year", "yr", "date", "time", "period"],
            "Country": ["country", "nation", "state", "region", "geo"],
            "Value": ["value", "amount", "size", "total", "sum"],
            "Market_Size": ["market_size", "market", "size", "volume", "revenue"]
        }
        
        for expected_col in expected_columns:
            best_match = None
            best_score = 0
            
            # Check exact matches first
            for data_col in data_columns:
                if data_col.lower() == expected_col.lower():
                    mappings[expected_col] = data_col
                    break
            else:
                # Check partial matches
                variations = column_variations.get(expected_col, [expected_col.lower()])
                
                for data_col in data_columns:
                    data_col_lower = data_col.lower()
                    
                    for variation in variations:
                        if variation in data_col_lower or data_col_lower in variation:
                            score = len(variation) / max(len(data_col_lower), len(variation))
                            if score > best_score:
                                best_score = score
                                best_match = data_col
                
                if best_match and best_score > 0.5:
                    mappings[expected_col] = best_match
        
        return mappings
    
    def render_data_preview(self, df: pd.DataFrame, title: str = "Data Preview"):
        """Render enhanced data preview with statistics"""
        st.markdown(f"### 👀 {title}")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Columns", numeric_cols)
        
        # Data preview
        st.markdown("**First 10 rows:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data types
        st.markdown("**Column Information:**")
        column_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(column_info, use_container_width=True)
        
        # Quick visualization for numeric data
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            st.markdown("**Quick Visualization:**")
            
            viz_col = st.selectbox(
                "Select column to visualize:",
                numeric_columns,
                help="Choose a numeric column to see its distribution"
            )
            
            if viz_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df, x=viz_col, 
                        title=f"Distribution of {viz_col}",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Summary statistics
                    st.markdown(f"**{viz_col} Statistics:**")
                    stats = df[viz_col].describe()
                    for stat, value in stats.items():
                        st.metric(stat.capitalize(), f"{value:.2f}")
    
    def render_quick_upload(self):
        """Render quick upload interface for the Quick Start workflow"""
        st.markdown("### 📊 Upload Your Data")
        
        # Help section
        with st.expander("📚 Need Help? Click here for upload guidance"):
            self.render_data_upload_help()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose your market data file",
            type=["csv", "xlsx"],
            help="Upload CSV or Excel files containing your market data"
        )
        
        if uploaded_file:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                
                # Determine data type
                data_type = self.guess_data_type(df)
                st.info(f"🤔 This looks like: **{data_type}**")
                
                # Data preview
                self.render_data_preview(df, f"{data_type} Preview")
                
                # Quick validation
                validation_results = self.validate_data_quality(df, data_type)
                is_valid = self.render_validation_results(validation_results)
                
                if is_valid:
                    # Store data
                    if data_type == "Global Market Data":
                        st.session_state.global_forecast = df
                        st.session_state.workflow_progress['data_loaded'] = True
                    elif data_type == "Country Historical Data":
                        st.session_state.country_historical = df
                    
                    st.success("✅ Data saved successfully!")
                else:
                    st.error("❌ Please fix the data quality issues before proceeding.")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.markdown("**Common solutions:**")
                st.markdown("• Check file format (CSV/Excel)")
                st.markdown("• Ensure proper encoding (UTF-8)")
                st.markdown("• Verify no special characters in headers")
    
    def guess_data_type(self, df: pd.DataFrame) -> str:
        """Guess the type of data based on column names"""
        columns = [col.lower() for col in df.columns]
        
        # Check for global market data patterns
        if any(word in ' '.join(columns) for word in ['market', 'size', 'global', 'total']):
            if 'year' in ' '.join(columns):
                return "Global Market Data"
        
        # Check for country data patterns
        if any(word in ' '.join(columns) for word in ['country', 'nation', 'state']):
            return "Country Historical Data"
        
        # Check for indicator patterns
        if any(word in ' '.join(columns) for word in ['gdp', 'population', 'indicator', 'index']):
            return "Economic Indicators"
        
        return "Global Market Data"  # Default
    
    def render_comprehensive_interface(self, step: int):
        """Render comprehensive data setup interface"""
        if step == 1:
            st.markdown("### 📥 Import Data")
            
            # Data type selection
            data_type = self.render_data_template_selector()
            
            # File format guidance
            self.render_file_format_guidance()
            
            # File upload
            uploaded_file = st.file_uploader(
                f"Upload {data_type}",
                type=["csv", "xlsx"],
                help=f"Upload your {data_type.lower()} file"
            )
            
            if uploaded_file:
                try:
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.temp_upload = {
                        'data': df,
                        'type': data_type,
                        'filename': uploaded_file.name
                    }
                    
                    st.success(f"✅ File loaded: {uploaded_file.name}")
                    
                    if st.button("Continue to Validation →"):
                        st.session_state.workflow_step = 2
                        st.rerun()
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        elif step == 2:
            st.markdown("### ✅ Validate Data")
            
            if 'temp_upload' in st.session_state:
                df = st.session_state.temp_upload['data']
                data_type = st.session_state.temp_upload['type']
                
                # Data preview
                self.render_data_preview(df)
                
                # Validation
                validation_results = self.validate_data_quality(df, data_type)
                is_valid = self.render_validation_results(validation_results)
                
                # Navigation
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("← Back to Import"):
                        st.session_state.workflow_step = 1
                        st.rerun()
                with col2:
                    if is_valid and st.button("Continue to Configuration →"):
                        st.session_state.workflow_step = 3
                        st.rerun()
        
        elif step == 3:
            st.markdown("### 🔗 Configure Data")
            
            if 'temp_upload' in st.session_state:
                df = st.session_state.temp_upload['data']
                data_type = st.session_state.temp_upload['type']
                
                # Column mapping
                column_mapping = self.render_column_mapping(df, data_type)
                
                # Preview mapped data
                if column_mapping:
                    st.markdown("#### 👀 Mapped Data Preview")
                    mapped_df = df.rename(columns={v: k for k, v in column_mapping.items()})
                    st.dataframe(mapped_df.head(), use_container_width=True)
                
                # Navigation
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("← Back to Validation"):
                        st.session_state.workflow_step = 2
                        st.rerun()
                with col2:
                    if column_mapping and st.button("Continue to Mapping →"):
                        st.session_state.temp_mapping = column_mapping
                        st.session_state.workflow_step = 4
                        st.rerun()
        
        elif step == 4:
            st.markdown("### 🗺️ Final Mapping")
            
            if 'temp_upload' in st.session_state and 'temp_mapping' in st.session_state:
                df = st.session_state.temp_upload['data']
                data_type = st.session_state.temp_upload['type']
                mapping = st.session_state.temp_mapping
                
                # Apply mapping
                mapped_df = df.rename(columns={v: k for k, v in mapping.items()})
                
                # Final preview
                st.markdown("#### ✅ Final Data Preview")
                self.render_data_preview(mapped_df, "Final Mapped Data")
                
                # Save options
                st.markdown("#### 💾 Save Configuration")
                
                save_config = st.checkbox(
                    "Save column mapping for future uploads",
                    value=True,
                    help="Remember this mapping for similar files"
                )
                
                # Navigation
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("← Back to Configuration"):
                        st.session_state.workflow_step = 3
                        st.rerun()
                with col2:
                    if st.button("Save Data →"):
                        # Save to session state
                        if data_type == "Global Market Data":
                            st.session_state.global_forecast = mapped_df
                        elif data_type == "Country Historical Data":
                            st.session_state.country_historical = mapped_df
                        elif data_type == "Economic Indicators":
                            indicator_name = st.session_state.temp_upload['filename'].split('.')[0]
                            st.session_state.indicators[indicator_name] = {
                                'data': mapped_df,
                                'meta': {
                                    'name': indicator_name,
                                    'type': 'value',
                                    'weight': 'auto'
                                }
                            }
                        
                        # Update progress
                        st.session_state.workflow_progress['data_loaded'] = True
                        st.session_state.workflow_step = 5
                        
                        # Clean up temp data
                        if 'temp_upload' in st.session_state:
                            del st.session_state.temp_upload
                        if 'temp_mapping' in st.session_state:
                            del st.session_state.temp_mapping
                        
                        st.success("✅ Data saved successfully!")
                        st.rerun()
        
        elif step == 5:
            st.markdown("### ✅ Data Setup Complete")
            
            st.success("🎉 Data setup completed successfully!")
            
            # Summary of loaded data
            data_summary = []
            
            if st.session_state.global_forecast is not None:
                data_summary.append({
                    "Data Type": "Global Market Data",
                    "Status": "✅ Loaded",
                    "Rows": len(st.session_state.global_forecast),
                    "Columns": len(st.session_state.global_forecast.columns)
                })
            
            if st.session_state.country_historical is not None:
                data_summary.append({
                    "Data Type": "Country Historical Data",
                    "Status": "✅ Loaded",
                    "Rows": len(st.session_state.country_historical),
                    "Columns": len(st.session_state.country_historical.columns)
                })
            
            for indicator_name in st.session_state.indicators:
                data_summary.append({
                    "Data Type": f"Indicator: {indicator_name}",
                    "Status": "✅ Loaded",
                    "Rows": len(st.session_state.indicators[indicator_name]['data']),
                    "Columns": len(st.session_state.indicators[indicator_name]['data'].columns)
                })
            
            if data_summary:
                summary_df = pd.DataFrame(data_summary)
                st.dataframe(summary_df, use_container_width=True)
            
            # Next steps
            st.markdown("### 🚀 What's Next?")
            st.markdown("""
            - **Forecasting**: Generate market forecasts using your data
            - **Analysis**: Explore insights and visualizations
            - **Export**: Download results and reports
            """)
            
            if st.button("🎯 Continue to Forecasting"):
                st.session_state.current_workflow = "Forecasting"
                st.session_state.workflow_step = 1
                st.rerun()