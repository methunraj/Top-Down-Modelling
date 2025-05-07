"""
Main Script - Universal Market Forecasting Framework

This script provides a simple way to run the Universal Market Forecasting Framework
from a Python script rather than the command line.
"""

import os
import sys
from src.market_analysis.market_analyzer import MarketAnalyzer


def main():
    """Main function to run the Universal Market Forecasting Framework"""
    # Print welcome message
    print("=" * 80)
    print("Universal Market Forecasting Framework")
    print("=" * 80)
    
    # Check for configuration file
    config_path = 'config/market_config.yaml'
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Creating default configuration file...")
        
        # Create config directory if it doesn't exist
        os.makedirs('config', exist_ok=True)
        
        # Create analyzer and generate config
        analyzer = MarketAnalyzer()
        analyzer.create_default_config_file(config_path)
        
        print(f"Created default configuration file: {config_path}")
        print("\nPlease edit the configuration file to match your dataset and run this script again.")
        return 0
    
    # Create analyzer with configuration
    print(f"Using configuration file: {config_path}")
    analyzer = MarketAnalyzer(config_path)
    
    # Run analysis
    print("\nAnalyzing market data...")
    try:
        distributed_market = analyzer.analyze_market()
        
        # Save results
        print("\nSaving results...")
        saved_files = analyzer.save_results(['xlsx', 'csv'])
        
        # Print summary
        analyzer.print_summary()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualization_files = analyzer.generate_visualizations()
        print(f"Generated {len(visualization_files)} visualization files")
        
        print("\nAnalysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 