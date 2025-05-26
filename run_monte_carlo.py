#!/usr/bin/env python3
"""
Monte Carlo Simulation Runner - Activate Monte Carlo forecasting
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from advanced_forecasting.monte_carlo_engine import MonteCarloDistributor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_monte_carlo_simulation():
    """Run Monte Carlo simulation with sample data"""
    
    print("🎯 Starting Monte Carlo Simulation...")
    
    # Create sample data (replace with your actual data)
    country_data = pd.DataFrame({
        'idGeo': [1, 2, 3, 1, 2, 3],
        'Country': ['USA', 'UK', 'Germany', 'USA', 'UK', 'Germany'],
        'Year': [2020, 2020, 2020, 2021, 2021, 2021],
        'Value': [100.0, 80.0, 90.0, 110.0, 85.0, 95.0],
        'nameVertical': ['Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech']
    })
    
    global_forecast = pd.DataFrame({
        'Year': [2022, 2023, 2024, 2025],
        'Value': [350.0, 380.0, 410.0, 440.0],
        'Type': ['Forecast', 'Forecast', 'Forecast', 'Forecast']
    })
    
    # Initialize Monte Carlo distributor
    mc_distributor = MonteCarloDistributor()
    
    # Define forecast years
    forecast_years = [2024, 2025, 2026]
    
    print(f"🔬 Running simulations for years: {forecast_years}")
    
    # Run Monte Carlo simulation
    results = mc_distributor.simulate_market_scenarios(
        country_data=country_data,
        forecast_years=forecast_years,
        global_forecast=global_forecast
    )
    
    print("\n✅ Monte Carlo Simulation Completed!")
    print(f"📊 Results Summary:")
    print(f"   - Number of scenarios: {len(results.get('scenarios', []))}")
    print(f"   - Statistics available: {list(results.get('statistics', {}).keys())}")
    print(f"   - Confidence intervals: {list(results.get('confidence_intervals', {}).keys())}")
    
    # Display sample statistics
    if 'statistics' in results:
        stats = results['statistics']
        print(f"\n📈 Sample Statistics:")
        for country in ['USA', 'UK', 'Germany']:
            if country in stats.get('mean', {}):
                mean_val = stats['mean'][country]
                std_val = stats['std'][country]
                print(f"   {country}: Mean = {mean_val:.2f}, Std = {std_val:.2f}")
    
    # Display confidence intervals
    if 'confidence_intervals' in results:
        ci = results['confidence_intervals']
        print(f"\n🎯 Confidence Intervals (95%):")
        if 0.95 in ci:
            for country in ['USA', 'UK', 'Germany']:
                if country in ci[0.95]:
                    lower, upper = ci[0.95][country]
                    print(f"   {country}: [{lower:.2f}, {upper:.2f}]")
    
    return results

if __name__ == "__main__":
    try:
        results = run_monte_carlo_simulation()
        print("\n🎉 Monte Carlo simulation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error running Monte Carlo simulation: {e}")
        sys.exit(1)