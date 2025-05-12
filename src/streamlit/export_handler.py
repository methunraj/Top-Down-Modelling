"""
Export Handler Module - Universal Market Forecasting Framework

This module provides functionality for exporting market forecasts, visualizations,
and reports in various formats.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import io
import base64
import plotly.graph_objects as go
from datetime import datetime

# Configure logger
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_market_data(distributed_market: pd.DataFrame,
                     output_dir: str,
                     export_formats: List[str],
                     include_market_values: bool = True,
                     include_market_shares: bool = True,
                     include_growth_rates: bool = True,
                     include_metadata: bool = True,
                     year_range: Tuple[int, int] = None) -> Dict[str, str]:
    """
    Export market data to various formats

    Args:
        distributed_market: DataFrame with distributed market data
        output_dir: Directory to save exported files
        export_formats: List of formats to export (Excel, CSV, JSON)
        include_market_values: Whether to include market values
        include_market_shares: Whether to include market shares
        include_growth_rates: Whether to include growth rates
        include_metadata: Whether to include forecast metadata
        year_range: Tuple of (start_year, end_year) to filter data

    Returns:
        Dictionary with paths to exported files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Make a copy of the dataframe
    df = distributed_market.copy()

    # Check if we have the required columns
    if 'Year' not in df.columns and 'year' in df.columns:
        df = df.rename(columns={'year': 'Year'})

    # Get country column name
    country_col = 'Country'
    if country_col not in df.columns and 'country' in df.columns:
        country_col = 'country'
        df = df.rename(columns={'country': 'Country'})
    elif country_col not in df.columns and 'Name' in df.columns:
        country_col = 'Name'
        df = df.rename(columns={'Name': 'Country'})

    # Get value column name
    value_col = 'Value'
    if value_col not in df.columns:
        if 'value' in df.columns:
            value_col = 'value'
            df = df.rename(columns={'value': 'Value'})
        elif 'market_value' in df.columns:
            value_col = 'market_value'
            df = df.rename(columns={'market_value': 'Value'})

    # Filter data by year range if specified
    if year_range and 'Year' in df.columns:
        start_year, end_year = year_range
        df = df[df['Year'].between(start_year, end_year)]

    # Calculate market share if not present and requested
    if include_market_shares and 'market_share' not in df.columns and 'Value' in df.columns:
        # Calculate market share by year
        for year in df['Year'].unique():
            year_mask = df['Year'] == year
            year_total = df.loc[year_mask, 'Value'].sum()
            if year_total > 0:
                df.loc[year_mask, 'market_share'] = (
                    df.loc[year_mask, 'Value'] / year_total * 100
                )

    # Calculate growth rates if requested
    if include_growth_rates and 'growth_rate' not in df.columns and 'Value' in df.columns:
        # Sort by Country and Year
        df = df.sort_values(['Country', 'Year'])

        # Calculate year-over-year growth rates
        df['growth_rate'] = df.groupby('Country')['Value'].pct_change() * 100

    # Add metadata if requested
    if include_metadata:
        df['export_date'] = datetime.now().strftime('%Y-%m-%d')
        df['data_type'] = 'Market Forecast'

    # Create a subfolder for data exports
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Dictionary to store export file paths
    exported_files = {}

    # Export in each format
    for fmt in export_formats:
        if "Excel" in fmt:
            # Export to Excel with multiple sheets
            excel_path = os.path.join(data_dir, 'market_forecast.xlsx')

            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    # Main data sheet
                    df.to_excel(writer, sheet_name='Market Data', index=False)

                    # Create pivot table with years as columns (wide format)
                    if 'Year' in df.columns and 'Country' in df.columns and 'Value' in df.columns:
                        wide_data = df.pivot_table(
                            index='Country',
                            columns='Year',
                            values='Value',
                            aggfunc='sum'
                        )
                        wide_data.to_excel(writer, sheet_name='Market Size (Wide)')

                    # Market share sheet if included
                    if include_market_shares and 'market_share' in df.columns:
                        # Get the latest year for market share
                        latest_year = df['Year'].max()
                        share_data = df[df['Year'] == latest_year][['Country', 'market_share']]
                        share_data = share_data.sort_values('market_share', ascending=False)
                        share_data.to_excel(writer, sheet_name='Market Share', index=False)

                    # Growth rates sheet if included
                    if include_growth_rates and 'growth_rate' in df.columns:
                        growth_data = df[['Country', 'Year', 'growth_rate']].dropna()
                        growth_data.to_excel(writer, sheet_name='Growth Rates', index=False)

                exported_files['excel'] = excel_path
                logger.info(f"Exported market data to Excel: {excel_path}")

            except Exception as e:
                logger.error(f"Error exporting to Excel: {str(e)}")

        if "CSV" in fmt:
            # Export to CSV - separate files for different aspects
            try:
                # Main data CSV
                csv_path = os.path.join(data_dir, 'market_values.csv')
                df.to_csv(csv_path, index=False)
                exported_files['csv_values'] = csv_path

                # Market share CSV if included
                if include_market_shares and 'market_share' in df.columns:
                    share_path = os.path.join(data_dir, 'market_shares.csv')
                    share_data = df[['Country', 'Year', 'market_share']]
                    share_data.to_csv(share_path, index=False)
                    exported_files['csv_shares'] = share_path

                # Growth rates CSV if included
                if include_growth_rates and 'growth_rate' in df.columns:
                    growth_path = os.path.join(data_dir, 'growth_rates.csv')
                    growth_data = df[['Country', 'Year', 'growth_rate']].dropna()
                    growth_data.to_csv(growth_path, index=False)
                    exported_files['csv_growth'] = growth_path

                logger.info(f"Exported market data to CSV: {csv_path}")

            except Exception as e:
                logger.error(f"Error exporting to CSV: {str(e)}")

        if "JSON" in fmt:
            # Export to JSON
            try:
                json_path = os.path.join(data_dir, 'forecast_data.json')
                df.to_json(json_path, orient='records', date_format='iso')
                exported_files['json'] = json_path
                logger.info(f"Exported market data to JSON: {json_path}")
            except Exception as e:
                logger.error(f"Error exporting to JSON: {str(e)}")

    return exported_files


def export_visualizations(distributed_market: pd.DataFrame,
                        output_dir: str,
                        chart_types: List[str],
                        chart_format: str,
                        dpi: int = 300) -> Dict[str, str]:
    """
    Export visualizations as image files
    
    Args:
        distributed_market: DataFrame with distributed market data
        output_dir: Directory to save exported files
        chart_types: List of chart types to export
        chart_format: Format to export charts (PNG, PDF, SVG, HTML)
        dpi: Image resolution (for raster formats)
        
    Returns:
        Dictionary with paths to exported chart files
    """
    # Create charts directory
    charts_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_dir, exist_ok=True)
    
    # Dictionary to store chart file paths
    chart_files = {}
    
    # Determine file extension from format
    if chart_format == "PNG":
        file_ext = "png"
    elif chart_format == "PDF":
        file_ext = "pdf"
    elif chart_format == "SVG":
        file_ext = "svg"
    else:  # HTML
        file_ext = "html"
    
    try:
        # Process each requested chart type
        for chart_type in chart_types:
            if chart_type == "Market Size Chart":
                # Create market size chart
                fig = create_market_size_chart(distributed_market)
                chart_path = os.path.join(charts_dir, f"market_size.{file_ext}")
                save_figure(fig, chart_path, chart_format, dpi)
                chart_files['market_size'] = chart_path
            
            elif chart_type == "Growth Rate Chart":
                # Create growth rate chart
                fig = create_growth_rate_chart(distributed_market)
                chart_path = os.path.join(charts_dir, f"growth_rates.{file_ext}")
                save_figure(fig, chart_path, chart_format, dpi)
                chart_files['growth_rates'] = chart_path
            
            elif chart_type == "Market Share Chart":
                # Create market share chart
                fig = create_market_share_chart(distributed_market)
                chart_path = os.path.join(charts_dir, f"market_share.{file_ext}")
                save_figure(fig, chart_path, chart_format, dpi)
                chart_files['market_share'] = chart_path
            
            elif chart_type == "Regional Analysis Chart":
                # Create regional analysis chart
                fig = create_regional_analysis_chart(distributed_market)
                chart_path = os.path.join(charts_dir, f"regional_analysis.{file_ext}")
                save_figure(fig, chart_path, chart_format, dpi)
                chart_files['regional_analysis'] = chart_path
    
    except Exception as e:
        logger.error(f"Error exporting visualizations: {str(e)}")
    
    return chart_files


def export_report(distributed_market: pd.DataFrame,
                output_dir: str,
                report_format: str,
                report_content: Dict[str, bool]) -> Dict[str, str]:
    """
    Generate and export a formatted report
    
    Args:
        distributed_market: DataFrame with distributed market data
        output_dir: Directory to save exported files
        report_format: Format of the report (PDF, PowerPoint, HTML)
        report_content: Dictionary of content sections to include
        
    Returns:
        Dictionary with paths to exported report files
    """
    # Create reports directory
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Dictionary to store report file paths
    report_files = {}
    
    try:
        if report_format == "PDF":
            # Generate PDF report
            report_path = os.path.join(reports_dir, "market_forecast_report.pdf")
            
            # This would normally use a PDF generation library like ReportLab
            # For now, just creating a placeholder file
            with open(report_path, "w") as f:
                f.write("Market Forecast Report")
            
            report_files['pdf'] = report_path
            logger.info(f"Exported PDF report: {report_path}")
        
        elif report_format == "PowerPoint":
            # Generate PowerPoint presentation
            report_path = os.path.join(reports_dir, "market_forecast_presentation.pptx")
            
            # This would normally use a library like python-pptx
            # For now, just creating a placeholder file
            with open(report_path, "w") as f:
                f.write("Market Forecast Presentation")
            
            report_files['pptx'] = report_path
            logger.info(f"Exported PowerPoint presentation: {report_path}")
        
        elif report_format == "HTML":
            # Generate HTML report
            report_path = os.path.join(reports_dir, "market_forecast_report.html")
            
            # Simple HTML template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Market Forecast Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #336699; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>Market Forecast Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                {'<h2>Executive Summary</h2><p>This report provides a forecast for the market.</p>' if report_content.get('Executive Summary', False) else ''}
                
                {'<h2>Methodology Description</h2><p>The forecast was generated using advanced statistical methods.</p>' if report_content.get('Methodology Description', False) else ''}
                
                {'<h2>Detailed Market Analysis</h2><p>Market analysis details would appear here.</p>' if report_content.get('Detailed Market Analysis', False) else ''}
                
                {'<h2>Country Profiles</h2><p>Individual country profiles would appear here.</p>' if report_content.get('Country Profiles', False) else ''}
                
                {'<h2>Appendix with Raw Data</h2><p>Raw data tables would appear here.</p>' if report_content.get('Appendix with Raw Data', False) else ''}
            </body>
            </html>
            """
            
            with open(report_path, "w") as f:
                f.write(html_content)
            
            report_files['html'] = report_path
            logger.info(f"Exported HTML report: {report_path}")
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
    
    return report_files


def create_market_size_chart(distributed_market: pd.DataFrame) -> go.Figure:
    """
    Create a market size chart using Plotly

    Args:
        distributed_market: DataFrame with distributed market data

    Returns:
        Plotly figure object
    """
    import plotly.express as px

    # Process data for visualization
    df = distributed_market.copy()

    # Check if we have the required columns
    if 'Year' not in df.columns and 'year' in df.columns:
        df = df.rename(columns={'year': 'Year'})

    # Get country column name
    country_col = 'Country'
    if country_col not in df.columns and 'country' in df.columns:
        country_col = 'country'
        df = df.rename(columns={'country': 'Country'})
    elif country_col not in df.columns and 'Name' in df.columns:
        country_col = 'Name'
        df = df.rename(columns={'Name': 'Country'})

    # Get value column name
    value_col = 'Value'
    if value_col not in df.columns:
        if 'value' in df.columns:
            value_col = 'value'
            df = df.rename(columns={'value': 'Value'})
        elif 'market_value' in df.columns:
            value_col = 'market_value'
            df = df.rename(columns={'market_value': 'Value'})

    # Get top 10 countries by value in the latest year
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    top_countries = latest_data.sort_values('Value', ascending=False).head(10)['Country'].unique().tolist()

    # Filter to include only top countries and add "Other" for the rest
    filtered_data = []
    for year in sorted(df['Year'].unique()):
        year_data = df[df['Year'] == year]

        # Top countries
        top_data = year_data[year_data['Country'].isin(top_countries)]

        # Other countries
        other_data = year_data[~year_data['Country'].isin(top_countries)]
        other_value = other_data['Value'].sum()

        # Add top countries
        for _, row in top_data.iterrows():
            filtered_data.append({
                'Year': row['Year'],
                'Country': row['Country'],
                'Value': row['Value']
            })

        # Add other countries
        filtered_data.append({
            'Year': year,
            'Country': 'Other Countries',
            'Value': other_value
        })

    # Convert to DataFrame
    filtered_df = pd.DataFrame(filtered_data)

    # Create stacked area chart
    fig = px.area(
        filtered_df,
        x='Year',
        y='Value',
        color='Country',
        title='Market Size by Country',
        labels={'Value': 'Market Size', 'Year': 'Year', 'Country': 'Country'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Update layout
    fig.update_layout(
        width=1200,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_growth_rate_chart(distributed_market: pd.DataFrame) -> go.Figure:
    """
    Create a growth rate chart using Plotly

    Args:
        distributed_market: DataFrame with distributed market data

    Returns:
        Plotly figure object
    """
    import plotly.express as px

    # Process data for visualization
    df = distributed_market.copy()

    # Check if we have the required columns
    if 'Year' not in df.columns and 'year' in df.columns:
        df = df.rename(columns={'year': 'Year'})

    # Get country column name
    country_col = 'Country'
    if country_col not in df.columns and 'country' in df.columns:
        country_col = 'country'
        df = df.rename(columns={'country': 'Country'})
    elif country_col not in df.columns and 'Name' in df.columns:
        country_col = 'Name'
        df = df.rename(columns={'Name': 'Country'})

    # Get value column name
    value_col = 'Value'
    if value_col not in df.columns:
        if 'value' in df.columns:
            value_col = 'value'
            df = df.rename(columns={'value': 'Value'})
        elif 'market_value' in df.columns:
            value_col = 'market_value'
            df = df.rename(columns={'market_value': 'Value'})

    # Sort by Country and Year
    df = df.sort_values(['Country', 'Year'])

    # Calculate year-over-year growth rates
    df['Growth Rate'] = df.groupby('Country')['Value'].pct_change() * 100

    # Filter out NaN values
    df = df.dropna(subset=['Growth Rate'])

    # Get top 10 countries by value in the latest year
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]
    top_countries = latest_data.sort_values('Value', ascending=False).head(10)['Country'].unique().tolist()

    # Filter to include only top countries
    filtered_df = df[df['Country'].isin(top_countries)]

    # Create bar chart for average growth rates
    avg_growth = filtered_df.groupby('Country')['Growth Rate'].mean().reset_index()
    avg_growth = avg_growth.sort_values('Growth Rate', ascending=False)

    fig = px.bar(
        avg_growth,
        x='Country',
        y='Growth Rate',
        title='Average Growth Rate by Country',
        labels={'Growth Rate': 'Growth Rate (%)', 'Country': 'Country'},
        color='Growth Rate',
        color_continuous_scale=px.colors.diverging.RdBu_r,
        color_continuous_midpoint=0
    )

    # Update layout
    fig.update_layout(
        width=1200,
        height=800,
        xaxis_tickangle=-45
    )

    return fig


def create_market_share_chart(distributed_market: pd.DataFrame) -> go.Figure:
    """
    Create a market share chart using Plotly

    Args:
        distributed_market: DataFrame with distributed market data

    Returns:
        Plotly figure object
    """
    import plotly.express as px

    # Process data for visualization
    df = distributed_market.copy()

    # Check if we have the required columns
    if 'Year' not in df.columns:
        print("Warning: 'Year' column not found in data")
        df['Year'] = 2023  # Default to current year

    # Get country column name
    country_col = 'Country'
    if country_col not in df.columns and 'country' in df.columns:
        country_col = 'country'
    elif country_col not in df.columns and 'Name' in df.columns:
        country_col = 'Name'

    # Get value column name
    value_col = 'Value'
    if value_col not in df.columns and 'value' in df.columns:
        value_col = 'value'
    elif value_col not in df.columns and 'market_value' in df.columns:
        value_col = 'market_value'

    # Get data for the latest year
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].copy()

    # Calculate market share
    latest_data['Market_Share'] = latest_data[value_col] / latest_data[value_col].sum() * 100

    # Get top 10 countries and combine the rest as "Other"
    top10 = latest_data.sort_values('Market_Share', ascending=False).head(10)
    other_share = 100 - top10['Market_Share'].sum()

    # Add "Other" category if there are more than 10 countries
    if len(latest_data) > 10:
        other_row = pd.DataFrame([{
            country_col: 'Other Countries',
            'Market_Share': other_share,
            value_col: latest_data[value_col].sum() - top10[value_col].sum()
        }])
        chart_data = pd.concat([top10[[country_col, 'Market_Share', value_col]], other_row])
    else:
        chart_data = top10[[country_col, 'Market_Share', value_col]]

    # Create pie chart
    fig = px.pie(
        chart_data,
        values='Market_Share',
        names=country_col,
        title=f'Market Share by Country ({latest_year})',
        hover_data=[value_col],
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # Update layout
    fig.update_layout(
        width=1200,
        height=800
    )

    # Update traces to show percentages
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Market Share: %{value:.1f}%<br>Value: %{customdata[0]:,.0f}'
    )

    return fig


def create_regional_analysis_chart(distributed_market: pd.DataFrame) -> go.Figure:
    """
    Create a regional analysis chart using Plotly

    Args:
        distributed_market: DataFrame with distributed market data

    Returns:
        Plotly figure object
    """
    import plotly.express as px

    # Process data for visualization
    df = distributed_market.copy()

    # Check if we have the required columns
    if 'Year' not in df.columns and 'year' in df.columns:
        df = df.rename(columns={'year': 'Year'})

    # Get country column name
    country_col = 'Country'
    if country_col not in df.columns and 'country' in df.columns:
        country_col = 'country'
        df = df.rename(columns={'country': 'Country'})
    elif country_col not in df.columns and 'Name' in df.columns:
        country_col = 'Name'
        df = df.rename(columns={'Name': 'Country'})

    # Get value column name
    value_col = 'Value'
    if value_col not in df.columns:
        if 'value' in df.columns:
            value_col = 'value'
            df = df.rename(columns={'value': 'Value'})
        elif 'market_value' in df.columns:
            value_col = 'market_value'
            df = df.rename(columns={'market_value': 'Value'})

    # Define regions (simplified example - would normally come from config)
    regions = {
        "North America": ["United States", "Canada", "Mexico"],
        "Europe": ["Germany", "United Kingdom", "France", "Italy", "Spain"],
        "Asia Pacific": ["China", "Japan", "India", "South Korea", "Australia"],
        "Latin America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru"],
        "Middle East & Africa": ["South Africa", "United Arab Emirates", "Saudi Arabia"]
    }

    # Assign regions
    def assign_region(country):
        for region, countries in regions.items():
            if country in countries:
                return region
        return "Other Regions"

    # Add region column
    df['Region'] = df['Country'].apply(assign_region)

    # Get latest year data
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year]

    # Aggregate by region
    region_data = latest_data.groupby('Region')['Value'].sum().reset_index()
    region_data['Market_Share'] = region_data['Value'] / region_data['Value'].sum() * 100
    region_data = region_data.sort_values('Market_Share', ascending=False)

    # Create bar chart
    fig = px.bar(
        region_data,
        x='Region',
        y='Market_Share',
        title=f'Market Share by Region ({latest_year})',
        labels={'Market_Share': 'Market Share (%)', 'Region': 'Region'},
        color='Market_Share',
        text='Market_Share',
        color_continuous_scale=px.colors.sequential.Blues
    )

    # Update layout
    fig.update_layout(
        width=1200,
        height=800,
        xaxis_tickangle=0
    )

    # Add value labels
    fig.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside'
    )

    return fig


def save_figure(fig: go.Figure, file_path: str, format_type: str, dpi: int = 300) -> str:
    """
    Save a Plotly figure to a file

    Args:
        fig: Plotly figure object
        file_path: Path to save the figure
        format_type: Format to save (PNG, PDF, SVG, HTML)
        dpi: Resolution for raster formats

    Returns:
        Path to the saved file
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if format_type in ["PNG", "PDF", "SVG"]:
            try:
                # Try to import kaleido for static image export
                import kaleido

                if format_type == "PNG":
                    fig.write_image(file_path, scale=dpi/100)
                else:
                    fig.write_image(file_path)

            except ImportError:
                # If kaleido is not available, save as HTML and add a message
                logger.warning(f"Kaleido package not found. Saving {format_type} as HTML instead.")
                logger.warning("Install kaleido with: pip install -U kaleido")

                # Change file extension to HTML
                file_path = os.path.splitext(file_path)[0] + ".html"

                # Add a message to the figure
                fig.add_annotation(
                    text=f"Note: {format_type} export requires the kaleido package. Install with: pip install -U kaleido",
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(color="red", size=12),
                    bgcolor="yellow"
                )

                # Save as HTML
                fig.write_html(file_path)
        else:  # HTML
            fig.write_html(file_path)

        logger.info(f"Saved figure to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")

        # Even if there's an error, try to save as HTML as a fallback
        try:
            # Change file extension to HTML
            html_file_path = os.path.splitext(file_path)[0] + ".html"

            # Add error message to the figure
            fig.add_annotation(
                text=f"Error saving as {format_type}: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=1.05,
                showarrow=False,
                font=dict(color="red", size=12),
                bgcolor="yellow"
            )

            # Save as HTML
            fig.write_html(html_file_path)
            logger.info(f"Saved figure as HTML instead: {html_file_path}")
            return html_file_path

        except Exception as html_err:
            logger.error(f"Also failed to save as HTML: {str(html_err)}")
            return ""