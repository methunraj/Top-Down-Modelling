# Forecasting Models Documentation

This document provides comprehensive information about all forecasting models available in the Market Modelling system, their parameters, use cases, and configuration options.

## Table of Contents

1. [Overview](#overview)
2. [Statistical Models](#statistical-models)
3. [Machine Learning Models](#machine-learning-models)
4. [Time Series Models](#time-series-models)
5. [Ensemble Models](#ensemble-models)
6. [Hybrid Models](#hybrid-models)
7. [Probabilistic Models](#probabilistic-models)
8. [Technology-Specific Models](#technology-specific-models)
9. [Model Selection Guide](#model-selection-guide)
10. [Parameter Tuning](#parameter-tuning)

---

## Overview

The Market Modelling system provides multiple forecasting approaches to handle different data characteristics, market conditions, and forecasting horizons. Each model has specific strengths and is suitable for different scenarios.

### Model Categories

| Category | Models | Best For |
|----------|--------|----------|
| **Statistical** | Linear Regression, Polynomial, Exponential | Simple trends, baseline forecasts |
| **Machine Learning** | Random Forest, XGBoost, Neural Networks | Complex patterns, non-linear relationships |
| **Time Series** | ARIMA, SARIMA, Exponential Smoothing | Temporal patterns, seasonality |
| **Ensemble** | Adaptive Ensemble, Advanced Ensemble | Robust predictions, uncertainty quantification |
| **Hybrid** | Combined approaches | Leveraging multiple model strengths |
| **Probabilistic** | Bayesian, Monte Carlo | Uncertainty estimation, risk analysis |

---

## Statistical Models

### 1. Linear Regression

**Description**: Simple linear relationship modeling between time and market values.

**Parameters**:
```yaml
linear_regression:
  fit_intercept: true          # Whether to calculate intercept
  normalize: false             # Whether to normalize features
  regularization:
    type: "none"              # Options: "none", "ridge", "lasso", "elastic_net"
    alpha: 1.0                # Regularization strength
    l1_ratio: 0.5             # Elastic net mixing parameter
```

**Use Cases**:
- Simple trend analysis
- Baseline forecasting
- Quick estimates for stable markets

**Strengths**: Simple, interpretable, fast
**Weaknesses**: Limited to linear relationships

### 2. Polynomial Regression

**Description**: Captures non-linear trends using polynomial features.

**Parameters**:
```yaml
polynomial_regression:
  degree: 2                    # Polynomial degree (1-5 recommended)
  include_bias: true           # Whether to include bias term
  interaction_only: false      # Only interaction features
  regularization:
    type: "ridge"             # Regularization to prevent overfitting
    alpha: 1.0
```

**Use Cases**:
- Markets with curved growth patterns
- S-curve adoption modeling
- Non-linear trend fitting

**Strengths**: Captures non-linear patterns, still interpretable
**Weaknesses**: Can overfit with high degrees

### 3. Exponential Growth

**Description**: Models exponential growth or decay patterns.

**Parameters**:
```yaml
exponential_growth:
  growth_rate: "auto"          # Fixed rate or "auto" for estimation
  base_value: "auto"           # Starting value or "auto"
  saturation_point: null       # Optional market saturation limit
  decay_factor: 1.0            # For modeling market decay
```

**Use Cases**:
- Technology adoption curves
- Early-stage market growth
- Population growth modeling

**Strengths**: Good for growth scenarios, simple parameterization
**Weaknesses**: May not capture market saturation

---

## Machine Learning Models

### 1. Random Forest

**Description**: Ensemble of decision trees for robust predictions.

**Parameters**:
```yaml
random_forest:
  n_estimators: 100           # Number of trees
  max_depth: null             # Maximum tree depth (null = unlimited)
  min_samples_split: 2        # Minimum samples to split node
  min_samples_leaf: 1         # Minimum samples in leaf
  max_features: "sqrt"        # Features to consider per split
  bootstrap: true             # Whether to bootstrap samples
  random_state: 42            # For reproducibility
  n_jobs: -1                  # Parallel processing (-1 = all cores)
```

**Feature Engineering**:
```yaml
features:
  temporal:
    - year
    - year_squared
    - years_since_start
  lagged_values:
    - lag_1                   # Previous year value
    - lag_2                   # Two years ago
    - rolling_mean_3          # 3-year rolling average
  external:
    - gdp_growth
    - technology_indicators
    - market_indicators
```

**Use Cases**:
- Complex market patterns
- Multiple influencing factors
- Non-linear relationships

**Strengths**: Handles non-linearity, robust to outliers, feature importance
**Weaknesses**: Less interpretable, can overfit with small datasets

### 2. XGBoost (Gradient Boosting)

**Description**: Advanced gradient boosting for high-performance forecasting.

**Parameters**:
```yaml
xgboost:
  n_estimators: 100           # Number of boosting rounds
  learning_rate: 0.1          # Step size shrinkage
  max_depth: 6                # Maximum tree depth
  subsample: 0.8              # Fraction of samples for each tree
  colsample_bytree: 0.8       # Fraction of features for each tree
  reg_alpha: 0                # L1 regularization
  reg_lambda: 1               # L2 regularization
  gamma: 0                    # Minimum loss reduction for split
  random_state: 42
```

**Advanced Parameters**:
```yaml
xgboost_advanced:
  objective: "reg:squarederror"  # Loss function
  eval_metric: "rmse"            # Evaluation metric
  early_stopping_rounds: 10      # Stop if no improvement
  tree_method: "auto"            # Tree construction algorithm
  verbosity: 0                   # Logging level
```

**Use Cases**:
- High-accuracy forecasting
- Competition-grade predictions
- Complex feature interactions

**Strengths**: State-of-the-art accuracy, handles missing values, fast training
**Weaknesses**: Requires parameter tuning, can overfit

### 3. Neural Networks

**Description**: Deep learning models for complex pattern recognition.

**Parameters**:
```yaml
neural_network:
  architecture:
    hidden_layers: [64, 32, 16]  # Hidden layer sizes
    activation: "relu"           # Activation function
    dropout_rate: 0.2            # Dropout for regularization
    batch_normalization: true    # Normalize layer inputs
  
  training:
    epochs: 100                  # Training iterations
    batch_size: 32               # Samples per batch
    learning_rate: 0.001         # Adam optimizer learning rate
    validation_split: 0.2        # Fraction for validation
    early_stopping:
      patience: 10               # Epochs to wait for improvement
      monitor: "val_loss"        # Metric to monitor
  
  regularization:
    l1: 0.0                     # L1 regularization
    l2: 0.01                    # L2 regularization
```

**Use Cases**:
- Very complex patterns
- Large datasets
- Non-linear feature interactions

**Strengths**: Can model very complex relationships, automatic feature learning
**Weaknesses**: Requires large datasets, black box, computationally intensive

---

## Time Series Models

### 1. ARIMA (AutoRegressive Integrated Moving Average)

**Description**: Classic time series model for temporal dependencies.

**Parameters**:
```yaml
arima:
  order: [1, 1, 1]            # (p, d, q) - AR, differencing, MA orders
  seasonal_order: [0, 0, 0, 0] # (P, D, Q, s) - seasonal parameters
  trend: "c"                   # Trend component: "n", "c", "t", "ct"
  method: "lbfgs"             # Optimization method
  maxiter: 50                 # Maximum iterations
  auto_order_selection:
    enabled: true             # Auto-select optimal orders
    max_p: 5                  # Maximum AR order to test
    max_d: 2                  # Maximum differencing
    max_q: 5                  # Maximum MA order
    information_criterion: "aic" # "aic", "bic", "hqic"
```

**Use Cases**:
- Data with clear temporal patterns
- Stationary time series
- Short to medium-term forecasting

**Strengths**: Well-established theory, good for temporal patterns
**Weaknesses**: Assumes stationarity, limited with trends

### 2. SARIMA (Seasonal ARIMA)

**Description**: ARIMA extended for seasonal patterns.

**Parameters**:
```yaml
sarima:
  order: [1, 1, 1]            # Non-seasonal (p, d, q)
  seasonal_order: [1, 1, 1, 12] # Seasonal (P, D, Q, s)
  seasonality_period: 12       # Season length (months/quarters)
  enforce_stationarity: true   # Ensure model stationarity
  enforce_invertibility: true  # Ensure model invertibility
```

**Use Cases**:
- Data with seasonal patterns
- Annual/quarterly cycles
- Retail and consumer markets

**Strengths**: Handles seasonality well, interpretable parameters
**Weaknesses**: Requires seasonal data, complex parameter selection

### 3. Exponential Smoothing (ETS)

**Description**: Weighted averages with exponentially decreasing weights.

**Parameters**:
```yaml
exponential_smoothing:
  trend: "add"                # "add", "mul", null
  seasonal: "add"             # "add", "mul", null
  seasonal_periods: 12        # Length of seasonal cycle
  damped_trend: false         # Whether to damp trend
  smoothing_level: null       # Alpha (auto if null)
  smoothing_trend: null       # Beta (auto if null)
  smoothing_seasonal: null    # Gamma (auto if null)
  damping_trend: null         # Phi (auto if null)
  
  optimization:
    use_boxcox: false         # Box-Cox transformation
    remove_bias: false        # Bias correction
    method: "L-BFGS-B"       # Optimization method
```

**Use Cases**:
- Smooth trend forecasting
- Simple seasonal patterns
- Robust to outliers

**Strengths**: Simple, robust, good baseline performance
**Weaknesses**: Limited complexity, assumes exponential decay

---

## Ensemble Models

### 1. Adaptive Ensemble

**Description**: Dynamically weighted combination of multiple models.

**Parameters**:
```yaml
adaptive_ensemble:
  base_models:
    - name: "linear_regression"
      weight: 0.3
      parameters: {...}
    - name: "random_forest"
      weight: 0.4
      parameters: {...}
    - name: "xgboost"
      weight: 0.3
      parameters: {...}
  
  weighting_strategy:
    method: "performance_based"  # "equal", "performance_based", "dynamic"
    performance_metric: "mape"   # Metric for weight calculation
    adaptation_rate: 0.1         # How quickly weights adapt
    lookback_period: 12          # Periods for performance evaluation
  
  combination_method: "weighted_average"  # "weighted_average", "median", "trimmed_mean"
  trim_percentage: 0.1           # For trimmed mean (remove extreme predictions)
```

**Advanced Configuration**:
```yaml
adaptive_ensemble_advanced:
  cross_validation:
    enabled: true
    folds: 5
    strategy: "time_series_split"
  
  uncertainty_estimation:
    enabled: true
    method: "prediction_intervals"  # "bootstrap", "prediction_intervals"
    confidence_level: 0.95
  
  model_selection:
    dynamic_inclusion: true      # Include/exclude models based on performance
    min_models: 2               # Minimum models to include
    performance_threshold: 0.1   # Relative performance threshold
```

**Use Cases**:
- Robust forecasting across different market conditions
- Combining strengths of multiple approaches
- Uncertainty quantification

**Strengths**: Robust, adaptable, reduces overfitting
**Weaknesses**: More complex, requires tuning multiple models

### 2. Advanced Ensemble

**Description**: Sophisticated ensemble with meta-learning capabilities.

**Parameters**:
```yaml
advanced_ensemble:
  meta_learner:
    type: "linear_regression"    # "linear_regression", "xgboost", "neural_network"
    parameters: {...}
  
  base_models:
    diversity_enforcement: true  # Ensure model diversity
    correlation_threshold: 0.8   # Maximum correlation between models
    
  stacking:
    enabled: true
    cv_folds: 5                 # Cross-validation folds for meta-features
    feature_engineering:
      include_predictions: true  # Raw predictions as features
      include_residuals: true    # Prediction errors as features
      include_confidence: true   # Confidence scores as features
  
  blending:
    enabled: false
    holdout_fraction: 0.2       # Fraction for blending
```

**Use Cases**:
- Maximum accuracy requirements
- Complex market dynamics
- Research and development

**Strengths**: State-of-the-art performance, sophisticated uncertainty handling
**Weaknesses**: Very complex, requires significant computational resources

---

## Hybrid Models

### 1. Statistical-ML Hybrid

**Description**: Combines statistical models for trends with ML for residuals.

**Parameters**:
```yaml
statistical_ml_hybrid:
  trend_model:
    type: "linear_regression"    # Primary trend model
    parameters: {...}
  
  residual_model:
    type: "random_forest"       # Model for residuals
    parameters: {...}
  
  decomposition:
    method: "additive"          # "additive", "multiplicative"
    seasonal_component: true    # Include seasonal decomposition
```

**Use Cases**:
- Markets with clear trends plus complex noise
- Combining interpretability with accuracy
- Hierarchical forecasting

### 2. Time Series-ML Hybrid

**Description**: Time series models enhanced with external features via ML.

**Parameters**:
```yaml
ts_ml_hybrid:
  time_series_component:
    type: "sarima"
    parameters: {...}
  
  ml_component:
    type: "xgboost"
    external_features:
      - gdp_growth
      - technology_indicators
      - competitive_analysis
    parameters: {...}
  
  combination:
    method: "weighted_sum"      # "weighted_sum", "multiplicative", "learned"
    ts_weight: 0.7             # Weight for time series component
```

---

## Probabilistic Models

### 1. Bayesian Hierarchical

**Description**: Bayesian approach with hierarchical structure for related markets.

**Parameters**:
```yaml
bayesian_hierarchical:
  hierarchy_levels:
    - global                    # Global market level
    - region                    # Regional level
    - country                   # Country level
  
  priors:
    growth_rate:
      distribution: "normal"
      mean: 0.1
      std: 0.05
    seasonality:
      distribution: "normal"
      mean: 0.0
      std: 0.1
  
  mcmc:
    iterations: 2000
    warmup: 1000
    chains: 4
    thin: 1
  
  shrinkage:
    enabled: true              # Shrink country estimates toward region
    shrinkage_factor: 0.3      # Amount of shrinkage
```

**Use Cases**:
- Related markets (countries, regions)
- Limited data scenarios
- Uncertainty quantification

**Strengths**: Principled uncertainty, handles related markets, works with limited data
**Weaknesses**: Computationally intensive, requires domain knowledge for priors

### 2. Monte Carlo Methods

**Description**: Simulation-based approaches for uncertainty estimation.

**Parameters**:
```yaml
monte_carlo:
  base_model:
    type: "random_forest"
    parameters: {...}
  
  simulation:
    n_simulations: 1000
    sampling_method: "bootstrap"  # "bootstrap", "gaussian_noise", "residual_bootstrap"
    confidence_levels: [0.05, 0.25, 0.75, 0.95]
  
  uncertainty_sources:
    parameter_uncertainty: true  # Model parameter uncertainty
    data_uncertainty: true      # Data noise uncertainty
    model_uncertainty: true     # Model selection uncertainty
  
  variance_reduction:
    antithetic_variates: true   # Reduce sampling variance
    control_variates: false     # Use control variates
    stratified_sampling: false  # Stratify sampling space
```

---

## Technology-Specific Models

### 1. Technology Adoption Model

**Description**: Models based on technology adoption theories (Bass model, S-curves).

**Parameters**:
```yaml
technology_adoption:
  model_type: "bass"           # "bass", "logistic", "gompertz"
  
  bass_model:
    innovation_coefficient: "auto"  # p parameter (innovators)
    imitation_coefficient: "auto"   # q parameter (imitators)
    market_potential: "auto"        # m parameter (total market)
    
  constraints:
    max_market_size: null      # Optional market size limit
    adoption_rate_limits:
      min: 0.01               # Minimum adoption rate
      max: 0.5                # Maximum adoption rate
  
  external_factors:
    price_elasticity: -0.5     # Price impact on adoption
    competition_effect: -0.2   # Competition impact
    regulation_impact: 0.1     # Regulatory impact
```

**Use Cases**:
- New technology forecasting
- Product lifecycle modeling
- Innovation diffusion

### 2. Market Maturity Model

**Description**: Accounts for different market maturity stages.

**Parameters**:
```yaml
market_maturity:
  maturity_stages:
    emerging:
      growth_rate: "high"      # High variability, exponential growth
      volatility: "high"
    growth:
      growth_rate: "medium"    # Steady growth
      volatility: "medium"
    mature:
      growth_rate: "low"       # Slow, steady growth
      volatility: "low"
    decline:
      growth_rate: "negative"  # Negative growth
      volatility: "medium"
  
  stage_detection:
    method: "automatic"        # "automatic", "manual", "hybrid"
    indicators:
      - growth_rate_stability
      - market_size_relative
      - competition_level
  
  transition_modeling:
    enabled: true
    transition_probabilities: "auto"  # Model stage transitions
```

---

## Model Selection Guide

### Decision Matrix

| Scenario | Recommended Models | Rationale |
|----------|-------------------|-----------|
| **Small Dataset (<50 points)** | Linear Regression, Exponential Smoothing | Simple models avoid overfitting |
| **Large Dataset (>500 points)** | XGBoost, Neural Networks, Ensemble | Can leverage complex patterns |
| **Clear Seasonality** | SARIMA, Exponential Smoothing | Designed for seasonal patterns |
| **Multiple Related Markets** | Bayesian Hierarchical | Shares information across markets |
| **High Accuracy Required** | Advanced Ensemble, XGBoost | Maximum predictive performance |
| **Interpretability Required** | Linear Regression, ARIMA | Transparent, explainable models |
| **Uncertainty Critical** | Bayesian, Monte Carlo, Ensemble | Provides confidence intervals |
| **Real-time Forecasting** | Simple models, cached ensembles | Fast prediction required |
| **New Technology** | Technology Adoption Models | Domain-specific patterns |

### Model Complexity vs Performance Trade-off

```
High Performance    │ ◆ Advanced Ensemble
                   │ ◆ Neural Networks
                   │ ◆ XGBoost
Medium Performance │ ◆ Random Forest
                   │ ◆ SARIMA
                   │ ◆ Adaptive Ensemble
Low Performance    │ ◆ Linear Regression
                   │ ◆ Exponential Smoothing
                   │ ◆ Simple ARIMA
                   └─────────────────────────
                   Low ←→ Medium ←→ High
                        Complexity
```

---

## Parameter Tuning

### Automated Parameter Selection

```yaml
auto_tuning:
  enabled: true
  method: "bayesian_optimization"  # "grid_search", "random_search", "bayesian_optimization"
  
  search_space:
    random_forest:
      n_estimators: [50, 100, 200, 500]
      max_depth: [null, 5, 10, 20]
      min_samples_split: [2, 5, 10]
    
    xgboost:
      learning_rate: [0.01, 0.1, 0.2]
      max_depth: [3, 6, 9]
      n_estimators: [50, 100, 200]
  
  optimization:
    metric: "mape"              # Optimization target
    cv_folds: 5                 # Cross-validation folds
    n_iterations: 50            # Maximum tuning iterations
    early_stopping: 10          # Stop if no improvement
```

### Manual Parameter Guidelines

#### For Linear Models:
- Start with simple linear regression
- Add polynomial features if residuals show curvature
- Use regularization (Ridge/Lasso) if overfitting

#### For Tree-Based Models:
- Start with default parameters
- Increase `n_estimators` until performance plateaus
- Tune `max_depth` and `min_samples_split` for overfitting control
- Use cross-validation for final parameter selection

#### For Time Series Models:
- Use auto-selection for ARIMA orders
- Check residuals for model adequacy
- Consider seasonal components if data shows seasonality

#### For Ensemble Models:
- Start with diverse base models
- Use cross-validation for meta-model training
- Balance model diversity vs individual performance

---

## Best Practices

### 1. Data Preparation
- Handle missing values appropriately for each model type
- Scale features for distance-based algorithms
- Create relevant time-based features
- Check for data leakage

### 2. Model Validation
- Use time-series specific cross-validation
- Test on out-of-sample data
- Monitor for overfitting
- Validate assumptions (stationarity for time series)

### 3. Performance Monitoring
- Track multiple metrics (MAPE, RMSE, MAE)
- Monitor prediction intervals coverage
- Check for concept drift
- Regular model retraining

### 4. Production Considerations
- Model serialization and versioning
- Prediction latency requirements
- Memory and computational constraints
- Fallback mechanisms for model failures

---

## Configuration Examples

### Simple Baseline Configuration
```yaml
forecasting:
  model: "linear_regression"
  parameters:
    fit_intercept: true
  validation:
    method: "time_series_split"
    test_size: 0.2
```

### Production-Ready Configuration
```yaml
forecasting:
  model: "adaptive_ensemble"
  parameters:
    base_models:
      - name: "linear_regression"
        weight: 0.2
      - name: "random_forest"
        weight: 0.4
        parameters:
          n_estimators: 100
          max_depth: 10
      - name: "xgboost"
        weight: 0.4
        parameters:
          n_estimators: 100
          learning_rate: 0.1
    
    weighting_strategy:
      method: "performance_based"
      adaptation_rate: 0.1
  
  validation:
    method: "walk_forward"
    initial_window: 36
    step_size: 1
  
  uncertainty:
    enabled: true
    confidence_levels: [0.05, 0.95]
```

### Research Configuration
```yaml
forecasting:
  model: "advanced_ensemble"
  parameters:
    meta_learner:
      type: "xgboost"
    base_models:
      diversity_enforcement: true
    stacking:
      enabled: true
      cv_folds: 10
  
  hyperparameter_tuning:
    enabled: true
    method: "bayesian_optimization"
    n_iterations: 100
  
  uncertainty:
    method: "monte_carlo"
    n_simulations: 1000
```

---

*This documentation is maintained by the Market Modelling team. For questions or contributions, please refer to the project repository.*