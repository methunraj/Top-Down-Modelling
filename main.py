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
import traceback
from pathlib import Path
from src.market_analysis.market_analyzer import MarketAnalyzer
from src.global_forecasting import get_available_forecasters


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Universal Market Forecasting Framework")
    
    # Configuration file
    parser.add_argument("--config", "-c", default="config/market_config.yaml",
                        help="Path to configuration file (default: config/market_config.yaml)")
    
    # Forecasting model
    try:
        forecaster_categories = get_available_forecasters()
        all_forecasters = []
        for category, forecasters in forecaster_categories.items():
            all_forecasters.extend(forecasters)
    except Exception as e:
        print(f"Warning: Could not load forecaster list: {e}")
        all_forecasters = []
    
    parser.add_argument("--model", "-m", choices=all_forecasters if all_forecasters else None, 
                        default=None,
                        help="Forecasting model to use (default: use model from config file)")
    
    # Output formats
    parser.add_argument("--formats", "-f", nargs="+", default=["xlsx", "csv"],
                        choices=["xlsx", "csv", "json", "pickle"],
                        help="Output file formats (default: xlsx csv)")
    
    # Visualizations - Fixed logic issue
    parser.add_argument("--visualize", "-v", action="store_true", default=False,
                        help="Generate visualizations (default: False)")
    parser.add_argument("--no-visualize", "-nv", dest="visualize", action="store_false",
                        help="Skip visualization generation")
    
    # List available models
    parser.add_argument("--list-models", "-l", action="store_true",
                        help="List available forecasting models and exit")
    
    return parser.parse_args()


def validate_model_choice(model_name):
    """Validate that the chosen model is available"""
    try:
        forecaster_categories = get_available_forecasters()
        all_forecasters = []
        for category, forecasters in forecaster_categories.items():
            all_forecasters.extend(forecasters)
        
        if model_name not in all_forecasters:
            print(f"Error: Model '{model_name}' is not available.")
            print("Available models:")
            list_available_models()
            return False
        return True
    except Exception as e:
        print(f"Error validating model: {e}")
        return False


def list_available_models():
    """List all available forecasting models by category"""
    try:
        forecaster_categories = get_available_forecasters()
        
        print("\nAvailable forecasting models:")
        print("=" * 80)
        
        for category, forecasters in forecaster_categories.items():
            print(f"\n{category}:")
            for i, forecaster in enumerate(forecasters, 1):
                print(f"  {i}. {forecaster}")
        
        print("\nUse with: python main.py --model MODEL_NAME")
    except Exception as e:
        print(f"Error listing models: {e}")


def create_config_directory_safely(config_path):
    """Safely create configuration directory"""
    config_dir = os.path.dirname(config_path)
    
    # Handle case where config_path has no directory component
    if config_dir and config_dir != "":
        try:
            os.makedirs(config_dir, exist_ok=True)
            return True
        except PermissionError:
            print(f"Error: Permission denied creating directory: {config_dir}")
            return False
        except Exception as e:
            print(f"Error creating directory {config_dir}: {e}")
            return False
    elif not config_dir:
        # config_path is just a filename, use current directory
        return True
    else:
        print(f"Error: Invalid configuration path: {config_path}")
        return False


def main():
    """Main function to run the Universal Market Forecasting Framework"""
    # Parse command-line arguments
    try:
        args = parse_arguments()
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        return 1
    
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
        
        # Create config directory if it doesn't exist - Fixed directory creation issue
        if not create_config_directory_safely(config_path):
            return 1
        
        # Create analyzer and generate config
        try:
            analyzer = MarketAnalyzer()
            analyzer.create_default_config_file(config_path)
            
            print(f"Created default configuration file: {config_path}")
            print("\nPlease edit the configuration file to match your dataset and run this script again.")
            return 0
        except Exception as e:
            print(f"Error creating default configuration: {e}")
            return 1
    
    # Create analyzer with configuration
    print(f"Using configuration file: {config_path}")
    try:
        analyzer = MarketAnalyzer(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Override forecasting model if specified - Added validation
    if args.model:
        if not validate_model_choice(args.model):
            return 1
        
        print(f"Using forecasting model: {args.model}")
        try:
            analyzer.set_forecasting_model(args.model)
        except Exception as e:
            print(f"Error setting forecasting model: {e}")
            return 1
    
    # Run analysis
    print("\nAnalyzing market data...")
    try:
        distributed_market = analyzer.analyze_market()
        
        if distributed_market is None:
            print("Warning: No distributed market data was generated")
            return 1
        
        # Save results - Fixed unused variable by using the return value
        print("\nSaving results...")
        saved_files = analyzer.save_results(args.formats)
        
        if saved_files:
            print(f"Results saved to {len(saved_files)} files:")
            for file_path in saved_files:
                print(f"  - {file_path}")
        else:
            print("Warning: No files were saved")
        
        # Print summary
        analyzer.print_summary()
        
        # Generate visualizations
        if args.visualize:
            print("\nGenerating visualizations...")
            try:
                visualization_files = analyzer.generate_visualizations()
                if visualization_files:
                    print(f"Generated {len(visualization_files)} visualization files:")
                    for file_path in visualization_files:
                        print(f"  - {file_path}")
                else:
                    print("Warning: No visualization files were generated")
            except Exception as e:
                print(f"Error generating visualizations: {e}")
                # Don't fail the entire process for visualization errors
        
        print("\nAnalysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())