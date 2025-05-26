#!/usr/bin/env python3
"""
Activate Monte Carlo Simulation in Market Modelling Pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config.config_manager import ConfigurationManager
from data_processing.data_loader import DataLoader
from advanced_forecasting.monte_carlo_engine import MonteCarloDistributor
import pandas as pd

def activate_monte_carlo_with_real_data():
    """Activate Monte Carlo simulation with your actual data files"""
    
    print("🎯 Activating Monte Carlo Simulation with Real Data...")
    
    try:
        # Create configuration for existing data files
        config = {
            "project": {"name": "Monte Carlo Market Analysis"},
            "data_sources": {
                "global_forecast": {"path": "data/global_forecast.xlsx"},
                "country_historical": {"path": "data/country_data.xlsx"}
            }
        }
        
        # Create config manager
        config_manager = ConfigurationManager()
        config_manager.config = config
        
        # Load your actual data
        data_loader = DataLoader(config_manager)
        
        print("📁 Loading your data files...")
        
        # Load actual country data (this works with your existing data structure)
        country_data = pd.read_excel("data/country_data.xlsx")
        print(f"   ✅ Country data loaded: {country_data.shape}")
        
        # Load actual global forecast 
        global_forecast = pd.read_excel("data/global_forecast.xlsx")
        # Add Type column if missing (required for Monte Carlo)
        if 'Type' not in global_forecast.columns:
            global_forecast['Type'] = 'Forecast'
        print(f"   ✅ Global forecast loaded: {global_forecast.shape}")
        
        # Convert country data to long format for Monte Carlo
        country_long = pd.melt(
            country_data, 
            id_vars=['kpiKey', 'idGeo', 'Country', 'nameVertical'],
            value_vars=[col for col in country_data.columns if col.isdigit()],
            var_name='Year', 
            value_name='Value'
        )
        country_long['Year'] = pd.to_numeric(country_long['Year'])
        country_long = country_long.dropna(subset=['Value'])
        
        print(f"   ✅ Country data converted to long format: {country_long.shape}")
        
        # Initialize Monte Carlo
        print("🔬 Initializing Monte Carlo Distributor...")
        mc_distributor = MonteCarloDistributor()
        
        # Define forecast years
        forecast_years = [2024, 2025, 2026]
        print(f"🎯 Running Monte Carlo for years: {forecast_years}")
        
        # Run Monte Carlo simulation
        print("⏳ Running Monte Carlo simulations...")
        results = mc_distributor.simulate_market_scenarios(
            country_data=country_long,
            forecast_years=forecast_years,
            global_forecast=global_forecast
        )
        
        print("\n✅ Monte Carlo Simulation Completed Successfully!")
        print(f"📊 Generated {len(results.get('scenarios', []))} probabilistic scenarios")
        
        # Save results
        output_file = "data/output/monte_carlo_results.json"
        import json
        with open(output_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: v.tolist() if hasattr(v, 'tolist') else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = value.tolist() if hasattr(value, 'tolist') else value
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try running with sample data first: python3 run_monte_carlo.py")
        return None

if __name__ == "__main__":
    results = activate_monte_carlo_with_real_data()
    if results:
        print("\n🎉 Monte Carlo activation successful!")
        print("📋 Next steps:")
        print("   1. Check data/output/monte_carlo_results.json for detailed results")
        print("   2. Integrate with your Streamlit app for visualization")
        print("   3. Use results for enhanced forecasting and risk analysis")
    else:
        print("\n⚠️  Activation failed. Check error messages above.")