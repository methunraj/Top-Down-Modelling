# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Web interface (recommended)
streamlit run streamlit_app_redesigned.py

# Classic web interface
streamlit run streamlit_app.py

# Command line interface
python3 main.py --config config/market_config.yaml

# List available forecasting models
python3 main.py --list-models

# Generate forecast with specific model
python3 main.py --model Prophet --visualize

# Run with different output formats
python3 main.py --formats xlsx csv json --visualize
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Check application startup
python3 main.py --help

# Run shell script (includes fixes)
./run_app.sh
```

### Testing and Development
- No formal test suite is present (pytest not configured)
- Test data generation is available via `src/streamlit/test_data.py`
- Framework includes sample configurations in `config/` directory

## Architecture Overview

This is a **Universal Market Forecasting Framework** - a comprehensive system for global market forecasting and country-level distribution. The architecture follows a modular design with clear separation of concerns:

### Core Pipeline
1. **Data Processing** (`src/data_processing/`) - Load and validate market data
2. **Global Forecasting** (`src/global_forecasting/`) - Generate global market forecasts using 25+ methods
3. **Market Distribution** (`src/distribution/`) - Distribute global forecasts to country level
4. **Visualization** (`src/visualization/`, `src/streamlit/`) - Generate charts and interactive interfaces

### Key Components

**Forecasting Engine** (`src/global_forecasting/`):
- 25+ forecasting models: Statistical (ARIMA, ETS), ML (XGBoost, LSTM), Technology-specific (Bass Diffusion)
- Ensemble methods with auto-calibration
- Base forecaster pattern with pluggable models

**Market Distribution** (`src/distribution/`):
- Intelligent country-level allocation using historical patterns
- Tier classification system (market leaders, established, emerging)
- Gradient harmonization for smooth transitions
- Market dynamics engine for phase modeling

**Configuration System** (`src/config/`):
- YAML-based configuration with validation
- Multiple sample configs for different use cases
- Dynamic column mapping and data source configuration

**Indicator System** (`src/indicators/`):
- Optional indicators (GDP, technology adoption, etc.)
- Causal inference capabilities
- Adaptive weighting system

**Auto-Calibration** (`src/global_forecasting/auto_calibration.py`):
- Continuous model performance monitoring
- Automatic parameter adjustment
- Confidence-based recalibration strategies

### Web Interface Architecture
Two Streamlit applications:
- `streamlit_app_redesigned.py` - Modern interface with guided workflows
- `streamlit_app.py` - Classic interface with component-based navigation

Interface components are modularized in `src/streamlit/`:
- `config_interface.py` - Configuration management
- `data_interface.py` - Data upload and validation
- `distribution_interface.py` - Market distribution controls
- `enhanced_visualization.py` - Charts and analytics
- `guided_wizard.py` - Step-by-step workflows

### Data Flow
1. **Input**: Global forecast data (Excel/CSV) + Country historical data + Optional indicators
2. **Processing**: Data validation, column mapping, indicator analysis
3. **Forecasting**: Global forecast generation using selected method(s)
4. **Distribution**: Country-level allocation with tier classification and constraints
5. **Output**: Market values, growth rates, visualizations in multiple formats

### Configuration Patterns
- Framework is designed to work with ANY market type (technology, healthcare, energy, etc.)
- Indicators are completely optional - uses historical patterns if no indicators provided
- Supports both wide and long data formats
- Auto-detects optimal forecasting parameters when not specified

### Important Implementation Notes
- All forecasting models inherit from `BaseForecaster` in `src/global_forecasting/base_forecaster.py`
- Market distribution uses tier-based growth constraints to ensure realistic projections
- Gradient harmonization prevents unrealistic jumps between historical and forecast data
- Session state management in Streamlit maintains data across interface interactions
- Export functionality supports multiple formats (Excel, CSV, JSON, PDF reports)

### Extension Points
- Add new forecasting models by extending `BaseForecaster`
- Create custom indicators in `src/indicators/`
- Add visualization types in `src/streamlit/`
- Extend export formats in `src/streamlit/export_handler.py`