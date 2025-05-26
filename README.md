# Universal Market Forecasting Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.24+-red.svg)](https://streamlit.io/)

**A comprehensive, production-ready framework for global market forecasting and country-level distribution with advanced machine learning and statistical modeling capabilities.**

## 🚀 Overview

The Universal Market Forecasting Framework is a state-of-the-art system that transforms global market forecasts into detailed country-level projections. It combines multiple forecasting methodologies, advanced statistical techniques, and machine learning models to provide accurate, reliable market predictions across any industry vertical.

### Key Features

✨ **25+ Forecasting Methods** - Statistical, ML, and domain-specific models  
🎯 **Auto-Calibrating System** - Continuous model improvement and optimization  
🌍 **Global Distribution** - Intelligent country-level market allocation  
📊 **Interactive Web Interface** - Professional Streamlit-based UI  
⚙️ **Fully Configurable** - Works with ANY indicators or NO indicators at all  
🔄 **Production Ready** - Robust error handling and comprehensive testing  

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Web Interface](#-web-interface)
- [Data Requirements](#-data-requirements)
- [Advanced Features](#-advanced-features)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd "Market Modelling"

# Install required packages
pip install -r requirements.txt
```

### Verify Installation

```bash
# Test the installation
python main.py --help

# Launch web interface
streamlit run streamlit_app.py
```

## ⚡ Quick Start

### 1. Command Line Interface

```bash
# Create a sample configuration
python main.py --create-config --output config/my_config.yaml

# Run market analysis with your configuration
python main.py --analyze --config config/my_config.yaml --output results/
```

### 2. Web Interface

```bash
# Launch the redesigned interface (recommended)
streamlit run streamlit_app_redesigned.py

# Or use the classic interface
streamlit run streamlit_app.py
```

### 3. Python API

```python
from src.market_analysis.market_analyzer import MarketAnalyzer

# Initialize with configuration
analyzer = MarketAnalyzer('config/market_config.yaml')

# Run complete analysis
results = analyzer.analyze_market()

# Generate visualizations
analyzer.generate_visualizations()
```

## 🏗 Architecture

The framework consists of several core modules:

### Core Components

```
├── src/
│   ├── config/                    # Configuration management
│   ├── data_processing/           # Data loading and preprocessing
│   ├── global_forecasting/        # 25+ forecasting models
│   ├── distribution/              # Market distribution algorithms
│   ├── indicators/                # Causal indicator analysis
│   ├── market_analysis/           # Core analysis engine
│   ├── visualization/             # Chart and report generation
│   └── streamlit/                 # Web interface components
```

### Forecasting Models

**Statistical Models:**
- ARIMA, Seasonal ARIMA
- Exponential Smoothing (ETS)
- State Space Models
- TBATS, BATS

**Machine Learning Models:**
- Random Forest, XGBoost, LightGBM
- Neural Networks (MLP, LSTM, GRU)
- Support Vector Regression
- Gaussian Process Regression

**Advanced Models:**
- Prophet (Facebook)
- Temporal Fusion Transformer
- DeepAR (Amazon)
- Bayesian Structural Time Series

**Ensemble Methods:**
- Weighted ensembles
- Stacked ensembles
- Dynamic ensemble selection

## 📖 Usage

### Configuration-Based Analysis

1. **Create Configuration File**
```yaml
project:
  name: "Technology Market Forecast"
  market_type: "Technology"

data_sources:
  global_forecast:
    path: "data/global_forecast.xlsx"
    sheet_name: "Sheet1"
  
  country_historical:
    path: "data/country_data.xlsx"
    sheet_name: "Sheet1"

market_distribution:
  tier_determination: "auto"
  calibration:
    enabled: true
    learning_rate: 0.15
```

2. **Run Analysis**
```python
from src.config.config_manager import ConfigurationManager
from src.market_analysis.market_analyzer import MarketAnalyzer

# Load configuration
config = ConfigurationManager('config/my_config.yaml')

# Initialize analyzer
analyzer = MarketAnalyzer(config.config_path)

# Execute analysis pipeline
results = analyzer.analyze_market()
print(f"Analysis complete. Results saved to: {results['output_path']}")
```

### Programmatic Usage

```python
import pandas as pd
from src.global_forecasting.ensemble import EnsembleForecaster
from src.distribution.market_distributor import MarketDistributor

# Load your data
global_data = pd.read_excel('data/global_forecast.xlsx')
country_data = pd.read_excel('data/country_data.xlsx')

# Create ensemble forecaster
forecaster = EnsembleForecaster(methods=['prophet', 'arima', 'xgboost'])

# Generate forecast
forecast = forecaster.forecast(global_data, horizon=5)

# Distribute to countries
distributor = MarketDistributor(config)
distributed_results = distributor.distribute_forecast(forecast, country_data)
```

## ⚙️ Configuration

### Configuration File Structure

```yaml
project:
  name: "Project Name"
  version: "1.0"
  market_type: "Technology"

data_sources:
  global_forecast:
    path: "path/to/global_data.xlsx"
    identifier:
      value_column: "Value"
      year_column: "Year"
  
  country_historical:
    path: "path/to/country_data.xlsx"
    format: "wide"  # or "long"
  
  indicators:
    - name: "GDP"
      path: "path/to/gdp_data.xlsx"
      weight: "auto"
      type: "value"  # or "rank"

market_distribution:
  tier_determination: "auto"  # or "manual"
  
  calibration:
    enabled: true
    learning_rate: 0.15
    recalibration_strategy: "gradual"
    confidence_thresholds:
      high: 0.85
      medium: 0.7
      low: 0.5

output:
  save_path: "results/"
  formats: ["xlsx", "csv", "json"]
  
  visualizations:
    enabled: true
    types:
      - name: "market_size"
        title: "Market Size by Country"
        top_n_countries: 10
```

### Sample Configurations

The framework includes several sample configurations:

- `config/sample_config.yaml` - Basic configuration template
- `config/sample_config_advanced.yaml` - Advanced features enabled
- `config/sample_config_with_calibration.yaml` - Auto-calibration enabled
- `config/sample_config_with_causal.yaml` - Causal indicator analysis
- `config/comprehensive_regions.yaml` - Complex regional hierarchies

## 🌐 Web Interface

### Redesigned Interface (Recommended)

```bash
streamlit run streamlit_app_redesigned.py
```

**Features:**
- 🎯 **Guided Workflows** - Step-by-step processes for different use cases
- 📊 **Smart Data Interface** - Intelligent data upload and validation
- 🔮 **Wizard-Based Setup** - Easy configuration without manual file editing
- 📈 **Unified Visualizations** - Professional charts and analytics
- 💡 **Contextual Help** - Built-in help system and tooltips

### Classic Interface

```bash
streamlit run streamlit_app.py
```

**Features:**
- Configuration file upload and editing
- Data upload and preview
- Global forecasting interface
- Market distribution controls
- Visualization gallery
- Export functionality

### Interface Modules

- **Config Interface** (`src/streamlit/config_interface.py`)
- **Data Interface** (`src/streamlit/data_interface.py`) 
- **Distribution Interface** (`src/streamlit/distribution_interface.py`)
- **Visualization Interface** (`src/streamlit/visualization_interface.py`)
- **Guided Wizard** (`src/streamlit/guided_wizard.py`)

## 📊 Data Requirements

### Global Forecast Data

Required columns:
- `Year` - Forecast years
- `Value` - Market values
- `Type` - Market type/category (optional)

Example:
```csv
Year,Value,Type
2023,1000000000,Technology Market
2024,1200000000,Technology Market
2025,1400000000,Technology Market
```

### Country Historical Data

**Wide Format (Recommended):**
```csv
idGeo,Country,nameVertical,2020,2021,2022,2023
1,United States,Technology,500000000,550000000,600000000,650000000
2,Germany,Technology,100000000,110000000,120000000,130000000
```

**Long Format:**
```csv
idGeo,Country,nameVertical,Year,Value
1,United States,Technology,2020,500000000
1,United States,Technology,2021,550000000
```

### Indicator Data (OPTIONAL)

**Indicators are completely optional!** The framework works perfectly without any indicators, using only historical market patterns for distribution.

When you do want to use indicators, you can add any indicators relevant to your market:

**Healthcare Market Examples:**
```csv
idGeo,Country,Hospital_Density,Healthcare_Spending,Aging_Population
1,United States,2.9,4000,16.5
2,Germany,8.0,5500,22.1
```

**Energy Market Examples:**
```csv
idGeo,Country,Renewable_Capacity,Grid_Infrastructure,Energy_Security
1,United States,300000,8.5,7.2
2,Germany,120000,9.1,6.8
```

**Any Custom Indicators:**
```csv
idGeo,Country,Your_Custom_Metric,Market_Readiness_Rank
1,United States,1000.5,1
2,Germany,800.2,3
```

The framework automatically adapts to whatever indicators you provide (or none at all).

## 🔬 Advanced Features

### Auto-Calibration System

The framework includes an intelligent auto-calibration system that:

- **Continuously monitors** forecast accuracy
- **Identifies error sources** through component analysis
- **Automatically adjusts** model parameters
- **Applies confidence-based** recalibration strategies
- **Maintains calibration history** for improvement tracking

```python
# Enable auto-calibration
config['market_distribution']['calibration']['enabled'] = True

# Run analysis with calibration
analyzer = MarketAnalyzer(config_path)
results = analyzer.analyze_market()

# View calibration report
calibration_report = analyzer.get_calibration_report()
```

### Causal Indicator Integration

Advanced causal analysis capabilities:

- **Granger Causality Testing** - Time series causality analysis
- **Conditional Independence Tests** - Statistical independence testing
- **Causal Network Visualization** - Interactive causal relationship maps
- **Causal Impact Quantification** - Measure indicator influence

### Gradient Harmonization

Sophisticated smoothing algorithm that:
- Ensures realistic transitions between historical and forecast data
- Preserves important market inflection points
- Applies tier-specific smoothing parameters
- Maintains mathematical consistency

### Market Dynamics Engine

Intelligent market phase modeling:
- **Introduction Phase** - High growth potential, high uncertainty
- **Growth Phase** - Rapid expansion, competitive dynamics
- **Maturity Phase** - Stable growth, market saturation
- **Decline Phase** - Negative growth, market contraction

## 📚 API Reference

### Core Classes

#### `MarketAnalyzer`
Main analysis engine for market forecasting and distribution.

```python
class MarketAnalyzer:
    def __init__(self, config_path: str)
    def analyze_market(self) -> Dict[str, Any]
    def evaluate_forecast_accuracy(self) -> Dict[str, float]
    def calibrate_models(self) -> Dict[str, Any]
    def generate_visualizations(self) -> None
    def save_results(self) -> str
```

#### `ConfigurationManager`
Handles configuration file loading and validation.

```python
class ConfigurationManager:
    def __init__(self, config_path: str)
    def load_config(self, config_path: str) -> None
    def validate_config(self) -> List[str]
    def get_project_info(self) -> Dict[str, Any]
```

#### `MarketDistributor`
Distributes global forecasts to country level.

```python
class MarketDistributor:
    def __init__(self, config: Dict[str, Any])
    def distribute_forecast(self, forecast: pd.DataFrame, 
                          country_data: pd.DataFrame) -> pd.DataFrame
    def apply_tier_classification(self) -> None
    def calculate_distribution_weights(self) -> pd.DataFrame
```

### Forecasting Models

All forecasting models inherit from `BaseForecaster`:

```python
from src.global_forecasting.base_forecaster import BaseForecaster

class CustomModel(BaseForecaster):
    def fit(self, data: pd.DataFrame) -> None
    def forecast(self, horizon: int) -> pd.DataFrame
    def get_model_info(self) -> Dict[str, Any]
```

## 🛠 Development

### Project Structure

```
Market Modelling/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   ├── data_processing/          # Data loading and preprocessing
│   ├── global_forecasting/       # Forecasting models
│   ├── distribution/             # Distribution algorithms
│   ├── indicators/               # Indicator analysis
│   ├── market_analysis/          # Core analysis
│   ├── visualization/            # Visualization
│   └── streamlit/                # Web interface
├── config/                       # Configuration files
├── data/                         # Sample data
│   ├── indicators/               # Economic indicators
│   └── output/                   # Generated results
├── main.py                       # Command-line interface
├── streamlit_app.py              # Classic web interface
├── streamlit_app_redesigned.py   # Modern web interface
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_market_analyzer.py

# Run with coverage
pytest --cov=src
```

### Code Quality

```bash
# Format code
black src/

# Sort imports  
isort src/

# Lint code
flake8 src/
```

### Adding New Forecasting Models

1. Create a new class inheriting from `BaseForecaster`
2. Implement required methods: `fit()`, `forecast()`, `get_model_info()`
3. Add the model to the ensemble registry
4. Update configuration documentation

Example:
```python
from src.global_forecasting.base_forecaster import BaseForecaster

class MyCustomModel(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
    
    def fit(self, data: pd.DataFrame) -> None:
        # Implement model training
        pass
    
    def forecast(self, horizon: int) -> pd.DataFrame:
        # Implement forecasting
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': 'MyCustomModel',
            'type': 'statistical',
            'parameters': self.parameters
        }
```

## 🤝 Contributing

We welcome contributions to the Universal Market Forecasting Framework!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings to all functions and classes
- Include unit tests for new features
- Update documentation for API changes
- Ensure backward compatibility when possible

### Issues and Feature Requests

Please use the GitHub issue tracker to:
- Report bugs
- Request new features
- Ask questions about usage
- Suggest improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Prophet** - Facebook's forecasting library
- **Streamlit** - For the amazing web application framework
- **Scikit-learn** - For machine learning tools
- **Plotly** - For interactive visualizations
- **Pandas** - For data manipulation and analysis

## 📞 Support

For support and questions:

- 📖 Check the documentation in this README
- 🐛 Report bugs via GitHub Issues
- 💡 Request features via GitHub Issues
- 📧 Contact the development team

---

**Made with ❤️ for the global forecasting community**