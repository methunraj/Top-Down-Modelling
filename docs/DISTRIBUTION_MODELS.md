# Distribution Models Documentation

This document provides comprehensive information about all distribution models available in the Market Modelling system, their parameters, algorithms, and configuration options for distributing global forecasts across countries and regions.

## Table of Contents

1. [Overview](#overview)
2. [Tier-Based Distribution](#tier-based-distribution)
3. [Indicator-Based Distribution](#indicator-based-distribution)
4. [Regional Distribution Models](#regional-distribution-models)
5. [Growth Constraint Models](#growth-constraint-models)
6. [Dynamic Distribution](#dynamic-distribution)
7. [Causal Distribution Models](#causal-distribution-models)
8. [Redistribution Algorithms](#redistribution-algorithms)
9. [Distribution Validation](#distribution-validation)
10. [Configuration Examples](#configuration-examples)

---

## Overview

Distribution models determine how global market forecasts are allocated across countries, regions, and market segments. The system provides multiple distribution strategies that can be combined and customized based on market characteristics, data availability, and business requirements.

### Distribution Philosophy

The distribution system follows these principles:
- **Hierarchical Consistency**: Country-level distributions sum to regional totals
- **Growth Constraints**: Realistic growth limits based on market maturity
- **Indicator Integration**: Leverage economic and technology indicators
- **Temporal Stability**: Smooth transitions between forecast periods
- **Uncertainty Propagation**: Maintain uncertainty through distribution process

### Core Distribution Types

| Type | Description | Use Cases |
|------|-------------|-----------|
| **Tier-Based** | Classify countries into tiers with different allocation rules | Market maturity-based distribution |
| **Indicator-Based** | Use economic/tech indicators for proportional allocation | Data-driven distribution |
| **Regional** | Geographic region-based distribution with spillovers | Regional market analysis |
| **Growth-Constrained** | Apply realistic growth limits and market saturation | Mature market modeling |
| **Dynamic** | Time-varying distribution patterns | Evolving market dynamics |
| **Causal** | Cause-effect relationships between markets | Interconnected markets |

---

## Tier-Based Distribution

### Overview

Tier-based distribution classifies countries into development tiers and applies different allocation strategies for each tier. This approach recognizes that market development patterns vary significantly across countries.

### Tier Classification

#### Automatic Tier Assignment
```yaml
tier_classification:
  method: "automatic"
  
  criteria:
    gdp_per_capita:
      weight: 0.4
      thresholds: [15000, 40000, 80000]  # USD per capita
    
    technology_readiness:
      weight: 0.3
      indicators:
        - internet_penetration
        - mobile_penetration
        - digital_infrastructure
    
    market_maturity:
      weight: 0.3
      indicators:
        - market_size_per_capita
        - growth_stability
        - competitive_landscape

  tiers:
    tier_1:  # Advanced markets
      description: "Developed, mature markets"
      characteristics:
        - high_gdp_per_capita: ">40000"
        - high_tech_readiness: ">0.8"
        - mature_market: true
    
    tier_2:  # Emerging markets
      description: "Developing markets with growth potential"
      characteristics:
        - medium_gdp_per_capita: "15000-40000"
        - medium_tech_readiness: "0.5-0.8"
        - growth_market: true
    
    tier_3:  # Developing markets
      description: "Early-stage markets"
      characteristics:
        - low_gdp_per_capita: "<15000"
        - low_tech_readiness: "<0.5"
        - emerging_market: true
```

#### Manual Tier Assignment
```yaml
tier_classification:
  method: "manual"
  
  assignments:
    tier_1: ["USA", "Germany", "Japan", "UK", "France", "Canada", "Australia"]
    tier_2: ["China", "India", "Brazil", "Mexico", "Russia", "South Korea"]
    tier_3: ["Vietnam", "Bangladesh", "Nigeria", "Kenya", "Indonesia"]
  
  custom_tiers:
    tier_special:  # Custom tier for specific markets
      countries: ["Singapore", "UAE", "Israel"]
      allocation_method: "hybrid"
```

### Tier-Specific Distribution Strategies

#### Tier 1 (Advanced Markets)
```yaml
tier_1_distribution:
  allocation_method: "stability_based"
  
  parameters:
    base_allocation: "historical_share"  # Start with historical market share
    stability_factor: 0.8               # High stability preference
    growth_cap: 0.15                    # Maximum 15% annual growth
    
    allocation_factors:
      gdp_weight: 0.6                   # GDP-based allocation
      population_weight: 0.2            # Population consideration
      tech_readiness_weight: 0.2        # Technology readiness
    
    adjustments:
      market_saturation: true           # Apply saturation effects
      competitive_pressure: 0.1         # Account for competition
      regulatory_impact: 0.05           # Regulatory considerations
```

#### Tier 2 (Emerging Markets)
```yaml
tier_2_distribution:
  allocation_method: "growth_potential_based"
  
  parameters:
    base_allocation: "potential_weighted"
    growth_orientation: 0.7             # High growth preference
    volatility_tolerance: 0.3           # Medium volatility acceptance
    
    allocation_factors:
      gdp_growth_weight: 0.4            # GDP growth rate
      population_growth_weight: 0.2     # Population dynamics
      infrastructure_weight: 0.2        # Infrastructure development
      market_potential_weight: 0.2      # Untapped market potential
    
    growth_constraints:
      min_growth: 0.05                  # Minimum 5% growth
      max_growth: 0.50                  # Maximum 50% growth
      volatility_cap: 0.3               # Control volatility
```

#### Tier 3 (Developing Markets)
```yaml
tier_3_distribution:
  allocation_method: "opportunity_based"
  
  parameters:
    base_allocation: "opportunity_score"
    risk_tolerance: 0.5                 # Medium-high risk tolerance
    growth_potential: 0.9               # Very high growth potential
    
    allocation_factors:
      development_indicators: 0.3       # Development level
      demographic_dividend: 0.3         # Young population benefit
      resource_availability: 0.2        # Natural/human resources
      policy_environment: 0.2           # Government policies
    
    special_considerations:
      leapfrog_potential: true          # Technology leapfrogging
      infrastructure_gaps: true         # Infrastructure constraints
      institutional_quality: 0.15       # Governance quality impact
```

### Cross-Tier Dynamics

```yaml
cross_tier_effects:
  technology_spillovers:
    enabled: true
    direction: "tier1_to_tier3"         # Technology flows from T1 to T3
    spillover_rate: 0.1                 # 10% spillover effect
    delay_periods: 2                    # 2-year delay
  
  market_graduation:
    enabled: true
    graduation_criteria:
      gdp_threshold: 25000              # GDP per capita threshold
      stability_period: 3               # Years of stable growth
    rebalancing_method: "gradual"       # Smooth transition
  
  competitive_effects:
    tier1_competition: 0.05             # T1 markets compete slightly
    tier2_competition: 0.15             # T2 markets compete more
    tier3_cooperation: 0.1              # T3 markets may cooperate
```

---

## Indicator-Based Distribution

### Economic Indicators

#### GDP-Based Distribution
```yaml
gdp_distribution:
  method: "gdp_weighted"
  
  gdp_metrics:
    nominal_gdp:
      weight: 0.4
      adjustment_factor: 1.0
    
    gdp_per_capita:
      weight: 0.3
      adjustment_factor: 0.8            # Slightly reduce per-capita influence
    
    gdp_growth_rate:
      weight: 0.3
      smoothing_window: 3               # 3-year average growth
      outlier_cap: 0.2                  # Cap extreme growth rates
  
  purchasing_power_adjustment:
    enabled: true
    ppp_weight: 0.5                     # PPP vs nominal GDP weight
    base_year: 2020                     # PPP base year
```

#### Population-Based Distribution
```yaml
population_distribution:
  method: "demographic_weighted"
  
  demographic_factors:
    total_population:
      weight: 0.4
      urban_rural_split: true           # Consider urban/rural divide
    
    working_age_population:
      weight: 0.4
      age_brackets: [15, 64]            # Working age definition
    
    population_growth:
      weight: 0.2
      projection_horizon: 10            # Years to project
  
  demographic_dividend:
    enabled: true
    young_population_bonus: 0.1         # Bonus for young populations
    aging_population_penalty: -0.05     # Penalty for aging societies
```

### Technology Indicators

#### Digital Infrastructure
```yaml
digital_infrastructure:
  indicators:
    internet_penetration:
      weight: 0.3
      threshold_effects:
        basic_access: 0.2               # 20% penetration threshold
        widespread_adoption: 0.6        # 60% penetration threshold
        saturation: 0.9                 # 90% saturation point
    
    mobile_penetration:
      weight: 0.25
      smartphone_weight: 0.7            # Smartphone vs basic mobile
    
    broadband_quality:
      weight: 0.25
      speed_weight: 0.6                 # Connection speed importance
      reliability_weight: 0.4           # Connection reliability
    
    digital_skills:
      weight: 0.2
      education_proxy: "tertiary_education_rate"
      training_programs: "ict_training_participation"
```

#### Innovation Ecosystem
```yaml
innovation_ecosystem:
  rd_investment:
    weight: 0.4
    public_private_split: [0.3, 0.7]    # Public vs private R&D
    intensity_measurement: "percentage_of_gdp"
  
  patent_activity:
    weight: 0.3
    patent_types:
      - technology_patents
      - utility_models
      - design_patents
    quality_adjustment: true            # Adjust for patent quality
  
  startup_ecosystem:
    weight: 0.2
    metrics:
      - venture_capital_investment
      - startup_density
      - unicorn_companies
  
  human_capital:
    weight: 0.1
    metrics:
      - stem_graduates
      - researcher_density
      - brain_gain_index
```

### Market-Specific Indicators

#### Technology Adoption Indicators
```yaml
technology_adoption:
  early_adopters:
    identification_method: "diffusion_curve"
    adoption_threshold: 0.1             # 10% early adopter threshold
    weight_multiplier: 1.5              # Boost for early adopters
  
  adoption_barriers:
    cost_barriers:
      weight: 0.4
      affordability_index: "price_to_income_ratio"
    
    infrastructure_barriers:
      weight: 0.3
      infrastructure_readiness: "composite_score"
    
    regulatory_barriers:
      weight: 0.2
      regulatory_friendliness: "ease_of_business_score"
    
    cultural_barriers:
      weight: 0.1
      technology_acceptance: "cultural_survey_data"
```

#### Competitive Landscape
```yaml
competitive_landscape:
  market_concentration:
    measurement: "herfindahl_index"
    competitive_intensity_adjustment: true
  
  local_vs_global_players:
    local_player_advantage: 0.1         # Advantage for local players
    global_player_efficiency: 0.15      # Efficiency of global players
  
  market_entry_barriers:
    regulatory_barriers: "business_entry_score"
    capital_requirements: "startup_capital_index"
    technical_barriers: "technical_complexity_score"
```

---

## Regional Distribution Models

### Geographic Regions

#### Regional Hierarchy
```yaml
regional_hierarchy:
  level_1_regions:
    north_america: ["USA", "Canada", "Mexico"]
    europe: ["Germany", "France", "UK", "Italy", "Spain", "Netherlands", "Others"]
    asia_pacific: ["China", "Japan", "India", "South Korea", "Australia", "Others"]
    latin_america: ["Brazil", "Argentina", "Chile", "Colombia", "Others"]
    middle_east_africa: ["UAE", "Saudi Arabia", "South Africa", "Nigeria", "Others"]
  
  sub_regions:
    western_europe: ["Germany", "France", "UK", "Netherlands", "Belgium"]
    eastern_europe: ["Poland", "Czech Republic", "Hungary", "Others"]
    southeast_asia: ["Singapore", "Thailand", "Malaysia", "Indonesia", "Vietnam"]
    east_asia: ["China", "Japan", "South Korea", "Taiwan"]
```

#### Regional Allocation Models

##### Hub-and-Spoke Model
```yaml
hub_and_spoke:
  regional_hubs:
    europe:
      hub_country: "Germany"
      hub_allocation: 0.3               # 30% to hub
      spoke_distribution: "gdp_weighted"
      spillover_effect: 0.15            # 15% spillover to spokes
    
    asia_pacific:
      hub_country: "China"
      hub_allocation: 0.4
      secondary_hubs: ["Japan", "India"]
      hub_competition: 0.1              # Competition between hubs
  
  spillover_mechanisms:
    technology_transfer: 0.1
    trade_relationships: 0.05
    cultural_similarity: 0.03
```

##### Gravity Model
```yaml
gravity_model:
  distance_effects:
    physical_distance:
      weight: 0.3
      decay_function: "inverse_square"   # Distance decay
      max_distance: 10000               # km
    
    economic_distance:
      weight: 0.4
      measurement: "gdp_per_capita_ratio"
      similarity_bonus: 0.1
    
    cultural_distance:
      weight: 0.3
      language_similarity: 0.15
      institutional_similarity: 0.15
  
  mass_effects:
    economic_mass: "gdp"
    population_mass: "population"
    market_mass: "market_size"
    
  interaction_strength:
    trade_intensity: 0.2
    investment_flows: 0.15
    technology_collaboration: 0.1
```

### Regional Dynamics

#### Cross-Regional Effects
```yaml
cross_regional_effects:
  technology_diffusion:
    direction_matrix:
      north_america_to_europe: 0.15
      north_america_to_asia: 0.12
      europe_to_asia: 0.10
      asia_to_latin_america: 0.08
    
    diffusion_speed:
      fast_technologies: 1.0            # 1 year lag
      medium_technologies: 2.0          # 2 year lag
      slow_technologies: 5.0            # 5 year lag
  
  economic_contagion:
    crisis_transmission: 0.2            # Economic crisis spillover
    growth_momentum: 0.1                # Growth spillover
    
  trade_relationships:
    bilateral_trade_impact: 0.05
    supply_chain_dependencies: 0.1
```

---

## Growth Constraint Models

### Market Saturation

#### Saturation Curves
```yaml
market_saturation:
  saturation_model: "logistic"          # "logistic", "gompertz", "bass"
  
  logistic_parameters:
    carrying_capacity: "auto"           # Auto-estimate or manual
    growth_rate: "auto"
    inflection_point: "auto"
  
  saturation_indicators:
    penetration_rate:
      threshold_warning: 0.7            # Warning at 70% penetration
      saturation_point: 0.95            # Saturation at 95%
    
    market_size_per_capita:
      comparison_method: "peer_countries"
      saturation_multiple: 1.2          # 120% of peer average
  
  saturation_effects:
    growth_slowdown:
      start_threshold: 0.6              # Slowdown starts at 60%
      slowdown_rate: 0.8                # 20% reduction per 10% increase
    
    substitution_effects:
      enabled: true
      substitution_rate: 0.1            # Rate of technology substitution
```

#### Maturity-Based Constraints
```yaml
maturity_constraints:
  market_lifecycle_stages:
    introduction:
      max_growth_rate: 2.0              # 200% max growth
      volatility_tolerance: 0.5
      market_share_ceiling: 0.05
    
    growth:
      max_growth_rate: 0.8              # 80% max growth
      volatility_tolerance: 0.3
      market_share_ceiling: 0.3
    
    maturity:
      max_growth_rate: 0.2              # 20% max growth
      volatility_tolerance: 0.1
      market_share_ceiling: 0.8
    
    decline:
      max_growth_rate: -0.05            # -5% max growth (decline)
      volatility_tolerance: 0.2
      market_share_ceiling: 1.0
  
  stage_transition_criteria:
    growth_to_maturity:
      market_penetration: 0.5
      growth_stability: 3               # 3 years stable growth
    
    maturity_to_decline:
      negative_growth_periods: 2        # 2 consecutive decline periods
      technological_disruption: true
```

### Economic Constraints

#### GDP-Based Limits
```yaml
gdp_constraints:
  market_to_gdp_ratio:
    historical_maximum: "auto"          # Learn from historical data
    country_specific_limits: true       # Different limits per country
    
    ratio_calculations:
      conservative: 0.02                # 2% of GDP
      moderate: 0.05                    # 5% of GDP
      aggressive: 0.10                  # 10% of GDP
  
  affordability_constraints:
    price_to_income_ratio:
      affordable_threshold: 0.1         # 10% of income
      luxury_threshold: 0.3             # 30% of income
    
    market_access:
      income_distribution_impact: true
      middle_class_size_weight: 0.6
```

#### Resource Constraints
```yaml
resource_constraints:
  human_capital:
    skilled_workforce_availability:
      stem_graduates_per_capita: "threshold_based"
      training_capacity: "infrastructure_limited"
    
    brain_drain_effects:
      emigration_rate_impact: -0.1
      immigration_bonus: 0.05
  
  infrastructure_limitations:
    physical_infrastructure:
      transportation: "logistics_performance_index"
      utilities: "electricity_reliability_index"
    
    digital_infrastructure:
      bandwidth_capacity: "internet_speed_index"
      coverage_gaps: "rural_urban_divide"
  
  regulatory_capacity:
    institutional_quality: "governance_indicators"
    regulatory_efficiency: "ease_of_business_score"
    enforcement_capability: "rule_of_law_index"
```

---

## Dynamic Distribution

### Time-Varying Patterns

#### Temporal Distribution Evolution
```yaml
temporal_evolution:
  distribution_trends:
    convergence_patterns:
      enabled: true
      convergence_rate: 0.05            # 5% convergence per year
      target_distribution: "equilibrium" # Converge to equilibrium
    
    divergence_patterns:
      technology_leaders_pull_ahead: 0.1
      laggards_fall_behind: -0.05
    
    cyclical_patterns:
      business_cycle_correlation: 0.2
      seasonal_adjustments: false       # Usually not applicable
  
  shock_propagation:
    economic_shocks:
      recession_impact: -0.15           # 15% reduction during recession
      recovery_bounce: 0.1              # 10% bounce during recovery
    
    technology_shocks:
      breakthrough_acceleration: 0.25    # 25% boost for breakthroughs
      disruption_displacement: -0.1     # 10% reduction for disruption
```

#### Adaptive Distribution Weights
```yaml
adaptive_weights:
  weight_updating:
    update_frequency: "quarterly"       # How often to update weights
    learning_rate: 0.1                  # How quickly to adapt
    stability_preference: 0.7           # Preference for stable weights
  
  performance_feedback:
    forecast_accuracy_weight: 0.4       # Weight based on forecast accuracy
    market_reality_alignment: 0.6       # Alignment with observed outcomes
  
  external_signal_integration:
    early_warning_indicators:
      - economic_leading_indicators
      - technology_adoption_signals
      - policy_change_indicators
    
    signal_processing:
      noise_filtering: true
      trend_extraction: "hodrick_prescott"
      signal_confidence_weighting: true
```

### Scenario-Based Distribution

#### Multi-Scenario Framework
```yaml
scenario_distribution:
  scenarios:
    baseline:
      probability: 0.6
      description: "Expected outcome based on current trends"
      distribution_adjustments: {}      # No adjustments
    
    optimistic:
      probability: 0.2
      description: "Accelerated growth scenario"
      distribution_adjustments:
        growth_multiplier: 1.3
        emerging_market_boost: 0.1
    
    pessimistic:
      probability: 0.2
      description: "Constrained growth scenario"
      distribution_adjustments:
        growth_multiplier: 0.7
        developed_market_preference: 0.1
  
  scenario_weights:
    dynamic_weighting: true
    weight_update_triggers:
      - economic_indicator_changes
      - geopolitical_events
      - technology_breakthroughs
```

---

## Causal Distribution Models

### Causal Relationships

#### Market Interdependencies
```yaml
causal_relationships:
  technology_spillovers:
    source_markets: ["USA", "Germany", "Japan"]
    target_markets: ["emerging_economies"]
    spillover_mechanism: "innovation_diffusion"
    
    spillover_parameters:
      spillover_rate: 0.15              # 15% spillover effect
      time_lag: 2                       # 2-year lag
      decay_rate: 0.8                   # 80% retention per year
  
  economic_linkages:
    trade_dependencies:
      bilateral_trade_impact: 0.1
      supply_chain_multiplier: 1.2
    
    investment_flows:
      fdi_impact: 0.05
      portfolio_investment_impact: 0.02
  
  competitive_effects:
    direct_competition:
      substitute_markets: ["similar_tech_level"]
      competition_intensity: 0.1
    
    complementary_effects:
      complementary_markets: ["different_tech_segments"]
      synergy_bonus: 0.05
```

#### Causal Inference Methods
```yaml
causal_inference:
  identification_strategy:
    method: "instrumental_variables"     # "IV", "RDD", "diff_in_diff"
    
    instruments:
      technology_shocks: "patent_breakthroughs"
      policy_shocks: "regulatory_changes"
      natural_experiments: "geography_based"
  
  model_specification:
    structural_model: "VAR"             # Vector Autoregression
    lag_structure: 4                    # 4-period lags
    cointegration_testing: true
  
  validation:
    placebo_tests: true
    robustness_checks: ["alternative_instruments", "different_time_periods"]
    sensitivity_analysis: true
```

---

## Redistribution Algorithms

### Rebalancing Mechanisms

#### Proportional Redistribution
```yaml
proportional_redistribution:
  trigger_conditions:
    imbalance_threshold: 0.05           # 5% deviation triggers rebalancing
    consistency_check_frequency: "quarterly"
  
  rebalancing_method:
    proportional_scaling: true
    preserve_rankings: true             # Maintain country rankings
    smooth_transitions: true
  
  constraints:
    minimum_allocation: 0.001           # 0.1% minimum per country
    maximum_allocation: 0.5             # 50% maximum per country
    tier_constraints: true              # Respect tier limitations
```

#### Optimization-Based Redistribution
```yaml
optimization_redistribution:
  objective_function:
    primary_objective: "minimize_forecast_error"
    secondary_objectives:
      - "maximize_consistency"
      - "minimize_volatility"
    
    objective_weights:
      forecast_accuracy: 0.6
      consistency: 0.25
      stability: 0.15
  
  constraints:
    equality_constraints:
      - "sum_to_global_total"
      - "respect_regional_totals"
    
    inequality_constraints:
      - "positive_allocations"
      - "growth_rate_bounds"
      - "market_size_bounds"
  
  optimization_algorithm:
    method: "sequential_quadratic_programming"
    tolerance: 1e-6
    max_iterations: 1000
```

### Hierarchical Consistency

#### Top-Down Consistency
```yaml
top_down_consistency:
  hierarchy_levels:
    1: "global"
    2: "regions"
    3: "countries"
  
  consistency_enforcement:
    method: "proportional_scaling"
    
    scaling_priorities:
      1: "preserve_global_total"        # Highest priority
      2: "preserve_regional_shares"     # Medium priority
      3: "preserve_country_rankings"    # Lowest priority
  
  reconciliation_algorithm:
    iterative_proportional_fitting: true
    convergence_tolerance: 0.001
    max_iterations: 100
```

#### Bottom-Up Aggregation
```yaml
bottom_up_aggregation:
  aggregation_method: "weighted_sum"
  
  country_weights:
    base_weights: "gdp_based"
    weight_adjustments:
      reliability_factor: 0.1           # Adjust for data reliability
      uncertainty_penalty: -0.05       # Penalize high uncertainty
  
  consistency_checks:
    cross_validation: true
    sanity_checks:
      - "growth_rate_reasonableness"
      - "market_size_plausibility"
      - "trend_consistency"
```

---

## Distribution Validation

### Validation Metrics

#### Statistical Validation
```yaml
statistical_validation:
  accuracy_metrics:
    - "mean_absolute_percentage_error"
    - "root_mean_square_error"
    - "mean_absolute_error"
  
  consistency_metrics:
    hierarchical_consistency: "sum_squared_deviations"
    temporal_consistency: "volatility_index"
    cross_sectional_consistency: "coefficient_of_variation"
  
  uncertainty_metrics:
    coverage_probability: "prediction_interval_coverage"
    interval_width: "average_interval_width"
    calibration: "probability_integral_transform"
```

#### Economic Validation
```yaml
economic_validation:
  plausibility_checks:
    growth_rate_bounds:
      min_annual_growth: -0.5          # -50% minimum
      max_annual_growth: 2.0           # 200% maximum
    
    market_size_bounds:
      min_market_size: 0               # Non-negative
      max_gdp_ratio: 0.2               # 20% of GDP maximum
  
  cross_country_comparisons:
    peer_group_analysis: true
    outlier_detection: "isolation_forest"
    benchmark_comparisons: "similar_economies"
  
  temporal_validation:
    trend_reasonableness: true
    cyclical_pattern_detection: true
    structural_break_testing: true
```

### Model Diagnostics

#### Residual Analysis
```yaml
residual_analysis:
  residual_tests:
    normality: "shapiro_wilk"
    heteroscedasticity: "breusch_pagan"
    autocorrelation: "ljung_box"
    cross_correlation: "portmanteau"
  
  residual_plots:
    qq_plots: true
    residual_vs_fitted: true
    residual_vs_time: true
    spatial_residual_maps: true
```

#### Model Comparison
```yaml
model_comparison:
  comparison_metrics:
    - "aic"  # Akaike Information Criterion
    - "bic"  # Bayesian Information Criterion
    - "cross_validation_score"
  
  backtesting:
    rolling_window_validation: true
    expanding_window_validation: true
    time_series_cross_validation: true
  
  ensemble_validation:
    individual_model_performance: true
    ensemble_combination_effectiveness: true
    weight_stability_analysis: true
```

---

## Configuration Examples

### Simple Country Distribution
```yaml
distribution:
  method: "gdp_weighted"
  
  parameters:
    gdp_data_source: "world_bank"
    base_year: 2020
    adjustment_factors:
      purchasing_power_parity: true
      population_adjustment: false
  
  countries: ["USA", "China", "Germany", "Japan", "UK"]
  
  validation:
    sum_check: true
    growth_bounds: [-0.3, 1.0]
```

### Advanced Multi-Tier Distribution
```yaml
distribution:
  method: "multi_tier_adaptive"
  
  tier_configuration:
    automatic_classification: true
    classification_indicators:
      - gdp_per_capita
      - technology_readiness
      - market_maturity
    
    tier_strategies:
      tier_1:
        method: "stability_based"
        growth_constraints: [0.0, 0.2]
        
      tier_2:
        method: "growth_potential"
        growth_constraints: [0.05, 0.5]
        
      tier_3:
        method: "opportunity_based"
        growth_constraints: [0.1, 1.0]
  
  cross_tier_effects:
    spillovers: true
    graduation_modeling: true
  
  temporal_dynamics:
    adaptive_weights: true
    scenario_modeling: true
  
  validation:
    hierarchical_consistency: true
    economic_plausibility: true
    uncertainty_quantification: true
```

### Research Configuration with Causal Modeling
```yaml
distribution:
  method: "causal_ensemble"
  
  causal_structure:
    identification_strategy: "instrumental_variables"
    structural_model: "VAR"
    
    causal_relationships:
      technology_spillovers: true
      trade_dependencies: true
      competitive_effects: true
  
  ensemble_components:
    - tier_based_distribution
    - indicator_based_distribution
    - regional_gravity_model
    - growth_constrained_model
  
  optimization:
    objective: "multi_objective"
    objectives:
      - forecast_accuracy
      - causal_consistency
      - uncertainty_minimization
  
  uncertainty_quantification:
    method: "bayesian_model_averaging"
    monte_carlo_simulations: 10000
    credible_intervals: [0.05, 0.95]
  
  validation:
    causal_validation: true
    counterfactual_analysis: true
    robustness_testing: true
```

---

## Best Practices

### 1. Model Selection
- Start with simple GDP-weighted distribution for baseline
- Add complexity based on data availability and requirements
- Consider market characteristics when choosing distribution method
- Validate assumptions with domain experts

### 2. Parameter Tuning
- Use cross-validation for parameter selection
- Consider economic reasonableness in addition to statistical fit
- Monitor parameter stability over time
- Document parameter choices and rationale

### 3. Validation and Monitoring
- Implement comprehensive validation framework
- Monitor distribution performance continuously
- Check for economic plausibility regularly
- Maintain audit trail of model changes

### 4. Uncertainty Management
- Quantify and propagate uncertainty throughout process
- Provide confidence intervals for all distributions
- Consider multiple scenarios for robust planning
- Communicate uncertainty to end users

### 5. Production Considerations
- Implement automated consistency checks
- Design for scalability and performance
- Plan for model versioning and rollback
- Establish clear governance and approval processes

---

*This documentation is maintained by the Market Modelling team. For questions or contributions, please refer to the project repository.*