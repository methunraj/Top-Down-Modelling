# Universal Market Forecasting Framework

## Overview

The Universal Market Forecasting Framework is a comprehensive tool for top-down market forecasting and distribution. It provides a market-agnostic approach to distributing global market values across countries using dynamic, data-driven algorithms that adapt to any market structure.

The enhanced version adds:

1. **Interactive Web Interface** - A complete Streamlit-based web interface for all functionality
2. **Global Market Forecasting** - Multiple forecasting methods for global market prediction
3. **Technology-Specific Methods** - Specialized models for technology markets
4. **Enhanced Visualization** - Interactive charts and regional analysis
5. **Integrated Workflow** - End-to-end process from data input to visualization

## Key Features

- **Global Market Forecasting**:
  - Statistical methods (CAGR, Moving Average, Exponential Smoothing)
  - Technology-specific models (Bass Diffusion, Gompertz, S-Curve)
  - Ensemble forecasting techniques

- **Market Distribution**:
  - Dynamic tier-based distribution
  - Data-driven growth constraints
  - Historical data preservation
  - Indicator-based adjustments
  - Advanced smoothing algorithms

- **Interactive Visualization**:
  - Market size analysis
  - Growth rate comparisons
  - Market share evolution
  - Regional analysis
  - Interactive data tables

- **Configuration System**:
  - Web-based configuration editor
  - Comprehensive parameter settings
  - Configuration validation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/universal-market-forecasting.git
   cd universal-market-forecasting
   ```

2. Install dependencies:
   ```bash
   pip install -r enhanced_requirements.txt
   ```

3. Run the application:
   ```bash
   ./run_app.sh
   ```
   or
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

The application provides a step-by-step workflow:

1. **Configuration**: Set up your project settings and parameters
2. **Data Input**: Upload global market data, country historical data, and indicators
3. **Global Forecasting**: Generate global market forecasts using multiple methods
4. **Market Distribution**: Distribute global values across countries with advanced algorithms
5. **Visualization**: Explore results with interactive charts and analyses
6. **Export**: Download results in various formats

## Data Format Requirements

### Global Market Data
- Must include columns for Year, Value, and Type (Historical/Forecast)
- Supported formats: Excel (.xlsx) or CSV

### Country Historical Data
- Can be in wide format (years as columns) or long format (year column)
- Must include country ID and name columns
- Supported formats: Excel (.xlsx) or CSV

### Indicator Data
- Can be value-based or rank-based indicators
- Must include country ID column for linking to country data
- Supported formats: Excel (.xlsx) or CSV

## Technology Market Specialization

The framework provides specialized features for technology markets:

- **Bass Diffusion Model**: Models technology adoption with innovation and imitation parameters
- **Gompertz Curve**: Models asymmetric growth common in technology markets
- **Technology S-Curve**: Models multi-phase technology adoption lifecycles
- **Tier-Based Analysis**: Different parameters for market leaders, established players, and emerging markets

## Advanced Configuration

The system provides extensive configuration options:

- **Tier Determination**: Automatic (K-means clustering) or manual threshold-based
- **Growth Constraints**: Dynamic constraints based on historical patterns or manual settings
- **Smoothing**: Tier-specific smoothing parameters for realistic growth patterns
- **Indicator Weights**: Automatic correlation-based or manually specified weights
- **Redistribution**: Option to preserve historical data before a specific year

## License

[Specify your license information here]

## Acknowledgments

- Original market distribution algorithm by [Your Name]
- Enhanced with global forecasting and Streamlit interface by [Your Name]