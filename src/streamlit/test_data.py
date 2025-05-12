"""
Test Data Module - Generate sample data for demonstration

This module provides functions to generate sample data for testing and demonstration
of the Universal Market Forecasting Framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_global_forecast() -> pd.DataFrame:
    """
    Generate sample global forecast data.
    
    Returns:
        DataFrame with global forecast data
    """
    # Define years
    years = list(range(2018, 2031))
    historical_years = [y for y in years if y <= 2023]
    forecast_years = [y for y in years if y > 2023]
    
    # Create historical data with realistic pattern
    start_value = 100000
    growth_rate = 0.15  # 15% annual growth
    
    historical_values = []
    current_value = start_value
    
    for _ in range(len(historical_years)):
        # Add some random variation to growth
        random_factor = np.random.normal(1.0, 0.03)  # Normal distribution around 1.0 with 3% std dev
        growth_factor = 1 + (growth_rate * random_factor)
        current_value = current_value * growth_factor
        historical_values.append(round(current_value, 0))
    
    # Create forecast data with growth trend
    base = historical_values[-1]
    forecast_values = []
    current_value = base
    
    for i in range(len(forecast_years)):
        # Slightly higher growth rate for forecast (optimistic bias)
        random_factor = np.random.normal(1.0, 0.02)  # Less variation in forecast
        adjusted_growth = growth_rate * (1.05 - i * 0.01)  # Slightly decreasing growth over time
        growth_factor = 1 + (adjusted_growth * random_factor)
        current_value = current_value * growth_factor
        forecast_values.append(round(current_value, 0))
    
    # Combine into DataFrame
    global_data = []
    
    for year, value in zip(historical_years, historical_values):
        global_data.append({
            'Year': year,
            'Value': value,
            'Type': 'Historical'
        })
    
    for year, value in zip(forecast_years, forecast_values):
        global_data.append({
            'Year': year,
            'Value': value,
            'Type': 'Forecast'
        })
    
    return pd.DataFrame(global_data)


def generate_country_historical() -> pd.DataFrame:
    """
    Generate sample country historical data.
    
    Returns:
        DataFrame with country historical data
    """
    # Define countries
    countries = [
        {"id": 1, "name": "United States"},
        {"id": 2, "name": "China"},
        {"id": 3, "name": "Japan"},
        {"id": 4, "name": "Germany"},
        {"id": 5, "name": "United Kingdom"},
        {"id": 6, "name": "France"},
        {"id": 7, "name": "India"},
        {"id": 8, "name": "Canada"},
        {"id": 9, "name": "South Korea"},
        {"id": 10, "name": "Brazil"}
    ]
    
    # Define years
    historical_years = list(range(2018, 2024))
    
    # Get global historical data for total market size
    global_data = generate_global_forecast()
    global_historical = global_data[global_data['Type'] == 'Historical']
    
    # Initial market share distribution (approximately realistic)
    initial_shares = {
        1: 0.32,  # US
        2: 0.25,  # China
        3: 0.11,  # Japan
        4: 0.07,  # Germany
        5: 0.06,  # UK
        6: 0.04,  # France
        7: 0.05,  # India
        8: 0.03,  # Canada
        9: 0.04,  # South Korea
        10: 0.03  # Brazil
    }
    
    # Share trends (annual change factors)
    share_trends = {
        1: 0.98,   # US: slight decline
        2: 1.04,   # China: moderate growth
        3: 0.97,   # Japan: slight decline
        4: 0.99,   # Germany: very slight decline
        5: 0.995,  # UK: very slight decline
        6: 0.99,   # France: very slight decline
        7: 1.09,   # India: strong growth
        8: 1.00,   # Canada: stable
        9: 1.02,   # South Korea: slight growth
        10: 1.03   # Brazil: slight growth
    }
    
    # Generate country data
    country_data = []
    
    for year_idx, year in enumerate(historical_years):
        # Get total market for this year
        year_market = global_historical[global_historical['Year'] == year]['Value'].values[0]
        
        # Keep track of total share to normalize
        total_share = 0
        preliminary_values = {}
        
        # Calculate preliminary values based on trend
        for country in countries:
            country_id = country["id"]
            
            # Apply trend to market share
            if year_idx == 0:
                # First year: use initial shares
                share = initial_shares[country_id]
            else:
                # Subsequent years: apply trend
                share = country_data[(year_idx - 1) * len(countries) + country_id - 1]['share'] * share_trends[country_id]
                
                # Add some random variation
                share_variation = np.random.normal(0, 0.01 * share)  # 1% random variation
                share = max(0.01, share + share_variation)  # Ensure minimum share
            
            total_share += share
            preliminary_values[country_id] = share
        
        # Normalize shares to sum to 1
        for country in countries:
            country_id = country["id"]
            
            # Get normalized share
            normalized_share = preliminary_values[country_id] / total_share
            
            # Calculate country value
            value = year_market * normalized_share
            
            # Add to data
            country_data.append({
                'idGeo': country_id,
                'Country': country["name"],
                'Year': year,
                'Value': round(value, 0),
                'share': normalized_share
            })
    
    # Convert to DataFrame and drop internal 'share' column
    result = pd.DataFrame(country_data)
    result = result.drop(columns=['share'])
    
    return result


def generate_indicators() -> Dict[str, pd.DataFrame]:
    """
    Generate sample indicator data.
    
    Returns:
        Dictionary of indicator DataFrames
    """
    # Define countries
    countries = [
        {"id": 1, "name": "United States"},
        {"id": 2, "name": "China"},
        {"id": 3, "name": "Japan"},
        {"id": 4, "name": "Germany"},
        {"id": 5, "name": "United Kingdom"},
        {"id": 6, "name": "France"},
        {"id": 7, "name": "India"},
        {"id": 8, "name": "Canada"},
        {"id": 9, "name": "South Korea"},
        {"id": 10, "name": "Brazil"}
    ]
    
    # Define years
    years = list(range(2018, 2024))
    
    # GDP Indicator (value-based)
    gdp_base = {
        1: 21400,  # US
        2: 16800,  # China
        3: 5200,   # Japan
        4: 4100,   # Germany
        5: 3200,   # UK
        6: 2800,   # France
        7: 3100,   # India
        8: 1900,   # Canada
        9: 1700,   # South Korea
        10: 2000   # Brazil
    }
    
    # GDP growth rates
    gdp_growth = {
        1: 0.022,  # US
        2: 0.055,  # China
        3: 0.012,  # Japan
        4: 0.015,  # Germany
        5: 0.017,  # UK
        6: 0.018,  # France
        7: 0.067,  # India
        8: 0.020,  # Canada
        9: 0.029,  # South Korea
        10: 0.026  # Brazil
    }
    
    # Generate GDP data
    gdp_data = []
    
    for year in years:
        for country in countries:
            country_id = country["id"]
            
            # Calculate GDP with growth and random variation
            if year == 2018:
                gdp = gdp_base[country_id]
            else:
                prev_gdp = next(item["GDP"] for item in gdp_data 
                              if item["Year"] == year - 1 and item["idGeo"] == country_id)
                
                growth_variation = np.random.normal(0, 0.004)  # Random variation
                effective_growth = gdp_growth[country_id] + growth_variation
                gdp = prev_gdp * (1 + effective_growth)
            
            gdp_data.append({
                "idGeo": country_id,
                "Country": country["name"],
                "Year": year,
                "GDP": round(gdp, 0)
            })
    
    # Technology Adoption Indicator (rank-based)
    # Initial ranks (1 is best)
    tech_ranks = {
        1: 1,   # US
        2: 3,   # China
        3: 2,   # Japan
        4: 4,   # Germany
        5: 5,   # UK
        6: 7,   # France
        7: 9,   # India
        8: 6,   # Canada
        9: 8,   # South Korea
        10: 10  # Brazil
    }
    
    # Rank trends (can change by this amount each year)
    rank_trends = {
        1: 0,     # US: stable
        2: -0.4,  # China: improving (negative means better rank)
        3: 0.1,   # Japan: slightly worsening
        4: -0.1,  # Germany: slightly improving
        5: 0,     # UK: stable
        6: -0.1,  # France: slightly improving
        7: -0.3,  # India: improving
        8: 0,     # Canada: stable
        9: -0.2,  # South Korea: improving
        10: -0.1  # Brazil: slightly improving
    }
    
    # Generate technology adoption data
    tech_data = []
    
    for year_idx, year in enumerate(years):
        # Keep track of actual ranks
        year_ranks = {}
        
        # Calculate preliminary ranks
        for country in countries:
            country_id = country["id"]
            
            # Apply trend to rank
            if year_idx == 0:
                # First year: use initial ranks
                rank_value = tech_ranks[country_id]
            else:
                # Subsequent years: apply trend
                rank_value = next(item["RankValue"] for item in tech_data 
                                if item["Year"] == year - 1 and item["idGeo"] == country_id)
                
                # Apply trend with random variation
                rank_variation = np.random.normal(0, 0.2)  # Random variation
                rank_value += rank_trends[country_id] + rank_variation
                
                # Ensure rank value stays within reasonable bounds
                rank_value = max(0.5, min(10.5, rank_value))
            
            year_ranks[country_id] = rank_value
        
        # Assign actual integer ranks based on sorted rank values
        sorted_countries = sorted(year_ranks.items(), key=lambda x: x[1])
        
        for rank, (country_id, rank_value) in enumerate(sorted_countries, 1):
            country_name = next(c["name"] for c in countries if c["id"] == country_id)
            
            tech_data.append({
                "idGeo": country_id,
                "Country": country_name,
                "Year": year,
                "TechRank": rank,
                "RankValue": rank_value  # Store for trend calculation
            })
    
    # Remove internal RankValue column
    tech_df = pd.DataFrame(tech_data)
    tech_df = tech_df.drop(columns=['RankValue'])
    
    # Return both indicators
    return {
        "GDP": pd.DataFrame(gdp_data),
        "TechnologyAdoption": tech_df
    }


def generate_all_test_data() -> Dict[str, Any]:
    """
    Generate all sample data for testing.
    
    Returns:
        Dictionary with all test data
    """
    global_forecast = generate_global_forecast()
    country_historical = generate_country_historical()
    indicators = generate_indicators()
    
    return {
        'global_forecast': global_forecast,
        'country_historical': country_historical,
        'indicators': indicators
    }