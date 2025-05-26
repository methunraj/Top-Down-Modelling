"""
Command-Line Interface - Universal Market Forecasting Framework

This module provides a command-line interface to run the Universal Market Forecasting 
Framework, making it easy to generate market forecasts from the command line.
"""

import argparse
import os
import logging
import sys
from typing import List, Optional

from src.market_analysis.market_analyzer import MarketAnalyzer


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Universal Market Forecasting Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main arguments
    parser.add_argument('-c', '--config', 
                        help='Path to configuration file (YAML or JSON)',
                        default=None)
    
    parser.add_argument('-o', '--output', 
                        help='Directory to save output files',
                        default='data/output/')
    
    # Commands (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument('--create-config', 
                      help='Create a default configuration file',
                      action='store_true')
    
    group.add_argument('--analyze', 
                      help='Analyze market and generate forecast',
                      action='store_true')
    
    # Additional options
    parser.add_argument('--visualize', 
                        help='Generate visualizations',
                        action='store_true')
    
    parser.add_argument('--formats', 
                        help='Output formats (comma-separated list)',
                        default='xlsx')
    
    parser.add_argument('--verbose', 
                        help='Enable verbose logging',
                        action='store_true')
    
    # Parse arguments
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the command-line interface
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    parsed_args = parse_args(args)
    
    # Set up logging
    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get logger
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration file
        if parsed_args.create_config:
            # Determine config path
            config_path = parsed_args.config
            if not config_path:
                config_path = 'config/market_config.yaml'
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            # Create analyzer and generate config
            analyzer = MarketAnalyzer()
            analyzer.create_default_config_file(config_path)
            
            logger.info(f"Created default configuration file: {config_path}")
            print(f"Created default configuration file: {config_path}")
            return 0
        
        # Analyze market
        if parsed_args.analyze:
            # Create analyzer
            analyzer = MarketAnalyzer(parsed_args.config)
            
            # Run analysis
            distributed_market = analyzer.analyze_market()
            
            # Save results
            output_formats = [f.strip() for f in parsed_args.formats.split(',')]
            saved_files = analyzer.save_results(output_formats)
            
            # Print summary
            analyzer.print_summary()
            
            # Generate visualizations if requested
            if parsed_args.visualize:
                visualization_files = analyzer.generate_visualizations()
                logger.info(f"Generated {len(visualization_files)} visualization files")
            
            return 0
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if parsed_args.verbose:
            # Print traceback in verbose mode
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main()) 