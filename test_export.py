"""
Test script for the export functionality
"""

import os
import pandas as pd
import numpy as np
from src.streamlit.export_handler import export_market_data, export_visualizations, export_report

# Create some dummy market data
def create_dummy_data():
    # Countries
    countries = ["United States", "China", "Japan", "Germany", "United Kingdom", 
                "France", "India", "Brazil", "Canada", "South Korea",
                "Italy", "Russia", "Australia", "Spain", "Mexico"]
    
    # Years
    years = list(range(2020, 2026))
    
    # Generate dummy data
    data = []
    for country in countries:
        # Base value for each country (varied to create differentiation)
        base_value = np.random.uniform(100, 1000)
        
        # Growth trajectory varied by country
        growth_rate = np.random.uniform(0.05, 0.20)
        
        for year in years:
            # Apply growth each year with some randomness
            year_factor = (year - 2020) * growth_rate
            random_factor = np.random.uniform(0.9, 1.1)
            value = base_value * (1 + year_factor) * random_factor
            
            data.append({
                "Country": country,
                "Year": year,
                "Value": value
            })
    
    # Convert to DataFrame
    return pd.DataFrame(data)

# Main test function
def test_export():
    # Create dummy data
    market_data = create_dummy_data()
    
    # Create a test output directory
    output_dir = "/Users/methunraj/Desktop/example/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export data files
    print("Exporting data files...")
    data_files = export_market_data(
        distributed_market=market_data,
        output_dir=output_dir,
        export_formats=["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"],
        include_market_values=True,
        include_market_shares=True,
        include_growth_rates=True,
        include_metadata=True,
        year_range=(2020, 2025)
    )
    
    # Export visualizations
    print("Exporting visualizations...")
    chart_files = export_visualizations(
        distributed_market=market_data,
        output_dir=output_dir,
        chart_types=["Market Size Chart", "Growth Rate Chart", "Market Share Chart"],
        chart_format="PNG",
        dpi=300
    )
    
    # Export report
    print("Exporting report...")
    report_content = {
        "Executive Summary": True,
        "Methodology Description": True,
        "Detailed Market Analysis": True,
        "Country Profiles": False,
        "Appendix with Raw Data": False
    }
    
    report_files = export_report(
        distributed_market=market_data,
        output_dir=output_dir,
        report_format="HTML",
        report_content=report_content
    )
    
    # Show all exported files
    print("\nExported Files:")
    for file_type, file_path in {**data_files, **chart_files, **report_files}.items():
        print(f"- {file_type}: {file_path}")

if __name__ == "__main__":
    test_export()