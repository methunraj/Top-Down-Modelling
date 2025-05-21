# Configuring Market Tiers and Leaders

This guide explains how to configure market tiers and manually designate market leaders in the Universal Market Forecasting Framework.

## Understanding Market Tiers

The framework uses a tiered approach to classify countries in the market:

- **Tier 1**: Market leaders with significant market share (typically >10%)
- **Tier 2**: Established markets with moderate market share (typically 1-10%)
- **Tier 3**: Emerging markets with smaller market share (typically <1%)

These tiers influence how growth rates are calculated, constrained, and smoothed for each country.

## Tier Determination Methods

### Automatic Tier Determination (Default)

By default, the system uses KMeans clustering to automatically classify countries into tiers based on their market share in the most recent historical year. This approach works well for most markets without requiring manual configuration.

To use automatic tier determination, set in your configuration file:

```yaml
market_distribution:
  tier_determination: "auto"
```

### Manual Tier Determination

For more control, you can manually define how countries are classified into tiers:

```yaml
market_distribution:
  tier_determination: "manual"
  
  # Define thresholds for tier classification based on market share percentages
  tier_thresholds: [10, 1, 0.1]  # Tier 1: >10%, Tier 2: 1-10%, Tier 3: 0.1-1%
  
  # Optional: Explicitly assign specific countries to tiers
  manual_tiers:
    United States: 1  # Assign US to Tier 1
    China: 1          # Assign China to Tier 1
    Germany: 2        # Assign Germany to Tier 2
    Brazil: 2         # Assign Brazil to Tier 2
    Vietnam: 3        # Assign Vietnam to Tier 3
```

## Configuring Market Leaders

### Via Manual Tier Assignment

The simplest way to designate market leaders is to assign them to Tier 1 using the `manual_tiers` configuration as shown above.

### Via Custom Tier Thresholds

You can also define what constitutes a "leader" by adjusting the tier thresholds:

```yaml
market_distribution:
  tier_determination: "manual"
  tier_thresholds: [15, 2, 0.2]  # Makes Tier 1 more exclusive (>15% market share)
```

## Growth Pattern Settings by Tier

Each tier has specific growth pattern settings that you can customize. These are defined in the `_apply_smoothing` method in `src/distribution/market_distributor.py`:

```python
tier_smoothing_params = {
    1: {'window': 3, 'min_periods': 1, 'center': True, 'max_growth': 35, 'min_growth': -15},  # Tier 1 (leaders)
    2: {'window': 3, 'min_periods': 1, 'center': True, 'max_growth': 40, 'min_growth': -20},  # Tier 2 (established)
    3: {'window': 5, 'min_periods': 1, 'center': True, 'max_growth': 45, 'min_growth': -25},  # Tier 3 (emerging)
    None: {'window': 4, 'min_periods': 1, 'center': True, 'max_growth': 40, 'min_growth': -20}  # Default
}
```

You can modify these parameters to change how growth patterns are handled for each tier:

- **Market Leaders (Tier 1)**: Typically have more stable growth with lower maximum growth rates
- **Established Markets (Tier 2)**: Moderate growth constraints
- **Emerging Markets (Tier 3)**: Allow higher growth rates and more volatility

## Long-Term Growth Rate Targets

The algorithm gradually converges growth rates toward sustainable long-term targets for each tier:

```python
target_growth = 15.0 if tier == 1 else 20.0 if tier == 2 else 25.0
```

These values (15% for Tier 1, 20% for Tier 2, 25% for Tier 3) can be adjusted based on the specific market dynamics:

- For high-growth technology markets, you might increase these values
- For mature, stable markets, you might reduce these values

## Programmatic Configuration

For advanced users who want to modify the tier determination logic programmatically:

1. Update the `_determine_tiers` method in `MarketDistributor` to change how automatic tier detection works
2. Modify the `_assign_country_tier` method to change the logic for assigning a tier to a specific country
3. Extend the `_load_manual_tiers` method to implement custom tier loading logic

## When to Adjust Tier Settings

Consider adjusting tier settings when:

1. Your market has unusual concentration patterns (very concentrated or very fragmented)
2. Your market has specific leaders that should follow different growth patterns
3. The automatic tier detection produces unexpected results
4. Specific countries should be treated differently based on external factors

## Example: Mature Technology Market

```yaml
market_distribution:
  tier_determination: "manual"
  tier_thresholds: [15, 3, 0.5]  # Higher thresholds for mature market
  
  manual_tiers:
    United States: 1
    China: 1
    Japan: 1
    Germany: 2
    United Kingdom: 2
    France: 2
    # (other countries will be auto-assigned based on thresholds)
  
  growth_constraints:
    determination_method: "manual"
    max_growth: 30.0  # Lower max growth for mature tech market
    min_growth: -15.0
```

## Example: Emerging Market with Few Leaders

```yaml
market_distribution:
  tier_determination: "manual"
  tier_thresholds: [20, 5, 1]  # Higher thresholds for highly concentrated market
  
  manual_tiers:
    United States: 1
    China: 1
    # (other countries will auto-assign)
  
  growth_constraints:
    determination_method: "manual"
    max_growth: 50.0  # Higher max growth for emerging market
    min_growth: -25.0
```

## Performance Impact

Note that using manual tier assignment instead of automatic detection has negligible performance impact, but it gives you more control over the forecast behavior for specific countries. 