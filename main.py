"""
Main Script - Universal Market Forecasting Framework

This script provides a simple way to run the Universal Market Forecasting Framework
from a Python script rather than the command line.

The framework now includes an expanded set of forecasting models:
- Statistical: CAGR, Moving Average, Exponential Smoothing, ARIMA, SARIMA, Regression
- Technology: Bass Diffusion, Gompertz Curve, Technology S-Curve
- Ensemble: Simple Average, Weighted Ensemble
"""

import os
import sys
import argparse
from src.market_analysis.market_analyzer import MarketAnalyzer
from src.global_forecasting import get_available_forecasters


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Universal Market Forecasting Framework")
    
    # Configuration file
    parser.add_argument("--config", "-c", default="config/market_config.yaml",
                        help="Path to configuration file (default: config/market_config.yaml)")
    
    # Forecasting model
    forecaster_categories = get_available_forecasters()
    all_forecasters = []
    for category, forecasters in forecaster_categories.items():
        all_forecasters.extend(forecasters)
    
    parser.add_argument("--model", "-m", choices=all_forecasters, default=None,
                        help="Forecasting model to use (default: use model from config file)")
    
    # Output formats
    parser.add_argument("--formats", "-f", nargs="+", default=["xlsx", "csv"],
                        choices=["xlsx", "csv", "json", "pickle"],
                        help="Output file formats (default: xlsx csv)")
    
    # Visualizations
    parser.add_argument("--visualize", "-v", action="store_true", default=True,
                        help="Generate visualizations (default: True)")
    
    # List available models
    parser.add_argument("--list-models", "-l", action="store_true",
                        help="List available forecasting models and exit")
    
    return parser.parse_args()


def list_available_models():
    """List all available forecasting models by category"""
    forecaster_categories = get_available_forecasters()
    
    print("\nAvailable forecasting models:")
    print("=" * 80)
    
    for category, forecasters in forecaster_categories.items():
        print(f"\n{category}:")
        for i, forecaster in enumerate(forecasters, 1):
            print(f"  {i}. {forecaster}")
    
    print("\nUse with: python main.py --model MODEL_NAME")


def main():
    """Main function to run the Universal Market Forecasting Framework"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check if user wants to list available models
    if args.list_models:
        list_available_models()
        return 0
    
    # Print welcome message
    print("=" * 80)
    print("Universal Market Forecasting Framework")
    print("=" * 80)
    
    # Check for configuration file
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Creating default configuration file...")
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Create analyzer and generate config
        analyzer = MarketAnalyzer()
        analyzer.create_default_config_file(config_path)
        
        print(f"Created default configuration file: {config_path}")
        print("\nPlease edit the configuration file to match your dataset and run this script again.")
        return 0
    
    # Create analyzer with configuration
    print(f"Using configuration file: {config_path}")
    analyzer = MarketAnalyzer(config_path)
    
    # Override forecasting model if specified
    if args.model:
        print(f"Using forecasting model: {args.model}")
        analyzer.set_forecasting_model(args.model)
    
    # Run analysis
    print("\nAnalyzing market data...")
    try:
        distributed_market = analyzer.analyze_market()
        
        # Save results
        print("\nSaving results...")
        saved_files = analyzer.save_results(args.formats)
        
        # Print summary
        analyzer.print_summary()
        
        # Generate visualizations
        if args.visualize:
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