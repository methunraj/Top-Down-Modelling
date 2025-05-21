# Market Smoothing Algorithm Documentation

The Universal Market Forecasting Framework includes a sophisticated adaptive smoothing algorithm that ensures realistic growth patterns in market forecasts. This document explains how the smoothing works and how to configure it for different market types.

## Overview of the Smoothing Algorithm

The smoothing algorithm addresses a common challenge in market forecasting: ensuring that growth rates follow realistic patterns without introducing artificial distortions. The implementation is found in the `_apply_smoothing` method of the `MarketDistributor` class in `src/distribution/market_distributor.py`.

### Key Features

1. **Tier-Based Smoothing**: Different market tiers receive different smoothing parameters
2. **Multi-Stage Smoothing Process**: Combines multiple statistical techniques
3. **Convergence to Target Growth**: Forecast years gradually trend toward sustainable growth rates
4. **Extreme Value Handling**: Pre-processes extreme growth rates before smoothing
5. **Total Market Preservation**: Maintains global market totals while adjusting individual country values

## How the Algorithm Works

The smoothing process follows these steps:

1. **Tier Determination**: Countries are classified into tiers based on market share (leaders, established, emerging)
2. **Growth Rate Calculation**: Year-over-year growth rates are calculated for each country
3. **Extreme Value Capping**: Growth rates are capped to prevent outliers from skewing results
4. **Multi-Stage Smoothing**:
   - First pass: Rolling window average to handle outliers
   - Second pass: Exponential weighted moving average for trend smoothing
5. **Growth Constraints**: Tier-specific constraints applied to ensure realistic bounds
6. **Progressive Convergence**: For forecast years, growth rates gradually converge toward sustainable rates
7. **Scaling**: Values are scaled to preserve global market totals
8. **Recalculation**: Market shares and growth rates are recalculated for consistency

## Configuring the Smoothing Algorithm

### Modifying Market Tier Parameters

To adjust tier-based smoothing parameters, modify the `tier_smoothing_params` dictionary in the `_apply_smoothing` method. This controls how aggressively smoothing is applied to each market tier.

```python
# Found in src/distribution/market_distributor.py (around line 765)
tier_smoothing_params = {
    1: {'window': 3, 'min_periods': 1, 'center': True, 'max_growth': 35, 'min_growth': -15},  # Tier 1 (leaders)
    2: {'window': 3, 'min_periods': 1, 'center': True, 'max_growth': 40, 'min_growth': -20},  # Tier 2 (established)
    3: {'window': 5, 'min_periods': 1, 'center': True, 'max_growth': 45, 'min_growth': -25},  # Tier 3 (emerging)
    None: {'window': 4, 'min_periods': 1, 'center': True, 'max_growth': 40, 'min_growth': -20}  # Default
}
```

#### Parameters explained:

- **window**: Size of the rolling window for initial smoothing (larger = more smoothing)
- **min_periods**: Minimum number of observations in window required to have a value
- **center**: Whether to set the window labels at the center
- **max_growth**: Maximum allowed growth rate (%) for this tier
- **min_growth**: Minimum allowed growth rate (%) for this tier

### Adjusting Growth Rate Caps

To modify the initial caps applied to extreme growth rates:

```python
# Found in src/distribution/market_distributor.py (around line 780)
extreme_cap = 80.0    # Cap extreme growth rates (upper bound)
extreme_floor = -40.0  # Floor for extreme negative growth
```

### Modifying Target Growth Rates

To change the long-term target growth rates that the algorithm converges toward:

```python
# Found in src/distribution/market_distributor.py (around line 810)
# Adjust target growth rates to be more realistic for each tier
target_growth = 15.0 if tier == 1 else 20.0 if tier == 2 else 25.0
```

### Adjusting Convergence Speed

To modify how quickly the algorithm converges toward target growth rates:

```python
# Found in src/distribution/market_distributor.py (around line 811)
convergence_rate = 0.25  # How quickly to converge to target growth
# and
weight = min(0.9, convergence_rate * years_into_future)  # Max weight is 0.9
```

## Advanced Customization

### Changing the Tier Determination Logic

If you want to adjust how countries are assigned to tiers, modify the `_determine_tiers` or `_assign_country_tier` methods in the `MarketDistributor` class.

```python
# Found in src/distribution/market_distributor.py
def _determine_tiers(self, historical_shares: pd.DataFrame, latest_year: int) -> None:
    # This method uses KMeans clustering to determine tiers automatically
    # ...
```

or

```python
# Found in src/distribution/market_distributor.py
def _assign_country_tier(self, share: float) -> int:
    # Manual tier assignment based on market share thresholds
    # ...
```

### Manual Tier Assignment

To manually assign countries to tiers, modify the `market_distribution` section in your configuration file:

```yaml
market_distribution:
  tier_determination: "manual"  # Set to "manual" instead of "auto"
  tier_thresholds: [10, 1, 0.1] # Market share % thresholds for tiers
  
  # Optional: Manually specify tiers for specific countries
  manual_tiers:
    United States: 1
    China: 1
    Germany: 2
    # ...
```

## Troubleshooting

### Excessive Volatility

If growth rates still show excessive volatility:

1. Increase the `window` parameter for problematic tiers
2. Reduce the `max_growth` and increase the `min_growth` values
3. Lower the `convergence_rate` to more gradually approach target growth

### Too Much Smoothing

If growth rates are too flat or unrealistic:

1. Decrease the `window` parameter
2. Increase the spread between `max_growth` and `min_growth`
3. Increase the `convergence_rate` for faster adaptation

## Performance Considerations

The smoothing algorithm's complexity scales linearly with the number of countries and years. For very large datasets (hundreds of countries, dozens of years), performance optimizations may be necessary. 