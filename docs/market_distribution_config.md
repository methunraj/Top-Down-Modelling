# Market Distribution Configuration

This document explains the market distribution configuration options, including the new `redistribution_start_year` feature.

## Overview

The market distribution module is responsible for distributing global market values across countries. It uses historical data to calculate market shares and trends, then projects these shares into the future.

## Configuration Options

### Basic Configuration

```yaml
market_distribution:
  tier_determination: "auto"  # "auto" or "manual"
  redistribution_start_year: 2020  # Year to start redistribution from (null for all years)
  
  # Other settings...
```

### Redistribution Start Year

The `redistribution_start_year` setting allows you to preserve historical data exactly as it is up to a certain year, and only apply redistribution and smoothing from that year forward.

- When set to `null` (default): The system will redistribute and smooth all years, including historical years.
- When set to a specific year (e.g., `2020`): The system will:
  - Preserve historical data exactly as is for all years before 2020
  - Only apply redistribution and smoothing for 2020 and later years

This is useful when:
1. You have high-quality historical data that you want to preserve exactly
2. You only want to apply the distribution algorithm to forecast years
3. You want to start redistribution from a specific historical year (e.g., the COVID-19 pandemic)

### Example

```yaml
market_distribution:
  tier_determination: "auto"
  redistribution_start_year: 2020  # Only redistribute from 2020 onward
  
  # Other settings...
```

In this example, all data before 2020 will be preserved exactly as it appears in your input data. Only data from 2020 onward will be subject to the distribution and smoothing algorithms.

## Implementation Details

When `redistribution_start_year` is set:

1. The system loads all historical data normally
2. During the smoothing process, it splits each country's data into:
   - Historical data (before the specified year)
   - Forecast data (from the specified year onward)
3. It only applies smoothing to the forecast portion
4. When recalculating market shares and scaling values to preserve global totals, it only modifies years from the specified year onward

This ensures that your historical data remains untouched while still allowing the system to apply its sophisticated distribution and smoothing algorithms to forecast years. 