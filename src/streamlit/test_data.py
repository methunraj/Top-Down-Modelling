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


def generate_indicators(indicator_configs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate sample indicator data based on configuration.
    
    Args:
        indicator_configs: Optional list of indicator configurations.
                          If None, generates example indicators for demonstration.
    
    Returns:
        Dictionary of indicator DataFrames
    """
    # If no indicators configured, return empty dict
    if indicator_configs is not None and len(indicator_configs) == 0:
        return {}
    
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
    
    # If no specific indicators requested, generate examples for demonstration
    if indicator_configs is None:
        # Generate example economic indicator (value-based)
        economic_base = {
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
        
        # Economic growth rates
        economic_growth = {
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
        
        # Generate sample economic indicator data for demonstration
        economic_data = []
        
        for year in years:
            for country in countries:
                country_id = country["id"]
                
                # Calculate economic value with growth and random variation
                if year == 2018:
                    value = economic_base[country_id]
                else:
                    prev_value = next(item["EconomicIndicator"] for item in economic_data 
                                    if item["Year"] == year - 1 and item["idGeo"] == country_id)
                    
                    growth_variation = np.random.normal(0, 0.004)  # Random variation
                    effective_growth = economic_growth[country_id] + growth_variation
                    value = prev_value * (1 + effective_growth)
                
                economic_data.append({
                    "idGeo": country_id,
                    "Country": country["name"],
                    "Year": year,
                    "EconomicIndicator": round(value, 0)
                })
        
        # Generate sample ranking indicator for demonstration
        rank_data = []
        initial_ranks = {1: 1, 2: 3, 3: 2, 4: 4, 5: 5, 6: 7, 7: 9, 8: 6, 9: 8, 10: 10}
        rank_trends = {1: 0, 2: -0.4, 3: 0.1, 4: -0.1, 5: 0, 6: -0.1, 7: -0.3, 8: 0, 9: -0.2, 10: -0.1}
        
        for year_idx, year in enumerate(years):
            year_ranks = {}
            
            for country in countries:
                country_id = country["id"]
                
                if year_idx == 0:
                    rank_value = initial_ranks[country_id]
                else:
                    prev_rank_value = next(item["RankValue"] for item in rank_data 
                                         if item["Year"] == year - 1 and item["idGeo"] == country_id)
                    rank_variation = np.random.normal(0, 0.2)
                    rank_value = prev_rank_value + rank_trends[country_id] + rank_variation
                    rank_value = max(0.5, min(10.5, rank_value))
                
                year_ranks[country_id] = rank_value
            
            # Assign actual ranks
            sorted_countries = sorted(year_ranks.items(), key=lambda x: x[1])
            for rank, (country_id, rank_value) in enumerate(sorted_countries, 1):
                country_name = next(c["name"] for c in countries if c["id"] == country_id)
                rank_data.append({
                    "idGeo": country_id,
                    "Country": country_name,
                    "Year": year,
                    "MarketReadiness": rank,
                    "RankValue": rank_value
                })
        
        # Clean up rank data
        rank_df = pd.DataFrame(rank_data).drop(columns=['RankValue'])
        
        # Return example indicators for demonstration
        return {
            "Sample_Economic_Indicator": pd.DataFrame(economic_data),
            "Sample_Market_Readiness_Rank": rank_df
        }
    
    # If specific indicators are configured, generate data based on their configuration
    result_indicators = {}
    
    for indicator_config in indicator_configs:
        indicator_name = indicator_config.get('name', 'Custom_Indicator')
        indicator_type = indicator_config.get('type', 'value')
        
        # Generate sample data based on type
        if indicator_type == 'rank':
            # Generate rank-based indicator
            rank_data = []
            initial_ranks = np.random.permutation(range(1, len(countries) + 1))
            
            for year_idx, year in enumerate(years):
                year_ranks = {}
                
                for idx, country in enumerate(countries):
                    country_id = country["id"]
                    
                    if year_idx == 0:
                        rank_value = initial_ranks[idx]
                    else:
                        prev_rank_value = next(item["RankValue"] for item in rank_data 
                                             if item["Year"] == year - 1 and item["idGeo"] == country_id)
                        rank_variation = np.random.normal(0, 0.3)
                        rank_value = prev_rank_value + rank_variation
                        rank_value = max(0.5, min(len(countries) + 0.5, rank_value))
                    
                    year_ranks[country_id] = rank_value
                
                # Assign actual ranks
                sorted_countries = sorted(year_ranks.items(), key=lambda x: x[1])
                for rank, (country_id, rank_value) in enumerate(sorted_countries, 1):
                    country_name = next(c["name"] for c in countries if c["id"] == country_id)
                    rank_data.append({
                        "idGeo": country_id,
                        "Country": country_name,
                        "Year": year,
                        indicator_name: rank,
                        "RankValue": rank_value
                    })
            
            result_indicators[indicator_name] = pd.DataFrame(rank_data).drop(columns=['RankValue'])
            
        else:
            # Generate value-based indicator
            value_data = []
            base_values = np.random.uniform(1000, 50000, len(countries))
            growth_rates = np.random.uniform(-0.02, 0.08, len(countries))
            
            for year in years:
                for idx, country in enumerate(countries):
                    country_id = country["id"]
                    
                    if year == years[0]:
                        value = base_values[idx]
                    else:
                        prev_value = next(item[indicator_name] for item in value_data 
                                        if item["Year"] == year - 1 and item["idGeo"] == country_id)
                        growth_variation = np.random.normal(0, 0.01)
                        effective_growth = growth_rates[idx] + growth_variation
                        value = prev_value * (1 + effective_growth)
                    
                    value_data.append({
                        "idGeo": country_id,
                        "Country": country["name"],
                        "Year": year,
                        indicator_name: round(value, 2)
                    })
            
            result_indicators[indicator_name] = pd.DataFrame(value_data)
    
    return result_indicators


def generate_all_test_data(use_sample_indicators: bool = True, 
                          custom_indicators: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Generate all sample data for testing.
    
    Args:
        use_sample_indicators: Whether to generate sample indicators for demonstration
        custom_indicators: Optional list of custom indicator configurations
    
    Returns:
        Dictionary with all test data
    """
    global_forecast = generate_global_forecast()
    country_historical = generate_country_historical()
    
    # Generate indicators based on configuration
    if custom_indicators is not None:
        # Use custom indicator configuration
        indicators = generate_indicators(custom_indicators)
    elif use_sample_indicators:
        # Generate sample indicators for demonstration
        indicators = generate_indicators(None)
    else:
        # No indicators
        indicators = {}
    
    return {
        'global_forecast': global_forecast,
        'country_historical': country_historical,
        'indicators': indicators
    }