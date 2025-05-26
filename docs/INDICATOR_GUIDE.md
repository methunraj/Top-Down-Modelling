# Indicator Configuration Guide

## 🎯 **Key Point: Indicators are 100% Optional**

The Universal Market Forecasting Framework is designed to work perfectly **without any indicators**. You can:

1. **Use NO indicators** - Framework uses historical market patterns only
2. **Use sample indicators** - For demonstration and testing
3. **Use custom indicators** - Add any indicators relevant to your market

## 🚀 **Quick Start Options**

### Option 1: No Indicators (Simplest)
```yaml
data_sources:
  indicators: []  # Empty list = no indicators
```

The framework will use only historical market data patterns for distribution. This works great for most use cases!

### Option 2: Custom Indicators (Advanced)
```yaml
data_sources:
  indicators:
    - name: "your_market_indicator"
      path: "data/your_indicator.xlsx"
      type: "value"  # or "rank"
      weight: "auto"
```

## 📊 **Market-Specific Examples**

### Healthcare Market
```yaml
indicators:
  - name: "hospital_density"
    path: "data/hospitals.xlsx"
    type: "value"
    weight: "auto"
  
  - name: "healthcare_spending_per_capita"
    path: "data/healthcare_spending.xlsx"
    type: "value"
    weight: "auto"
    
  - name: "health_system_ranking"
    path: "data/health_rankings.xlsx"
    type: "rank"  # 1 = best, higher = worse
    weight: 0.3
```

### Energy Market
```yaml
indicators:
  - name: "renewable_energy_capacity"
    path: "data/renewable_capacity.xlsx"
    type: "value"
    weight: "auto"
    
  - name: "grid_infrastructure_quality"
    path: "data/grid_quality.xlsx"
    type: "value"
    weight: "auto"
    
  - name: "environmental_regulations"
    path: "data/env_regulations.xlsx"
    type: "rank"
    weight: 0.4
```

### Financial Services Market
```yaml
indicators:
  - name: "banking_penetration"
    path: "data/banking_penetration.xlsx"
    type: "value"
    weight: "auto"
    
  - name: "fintech_adoption_rate"
    path: "data/fintech_adoption.xlsx"
    type: "value"
    weight: "auto"
    
  - name: "regulatory_environment_rank"
    path: "data/regulatory_rank.xlsx"
    type: "rank"
    weight: 0.2
```

### Technology Market
```yaml
indicators:
  - name: "digital_infrastructure_index"
    path: "data/digital_infrastructure.xlsx"
    type: "value"
    weight: "auto"
    
  - name: "tech_talent_availability"
    path: "data/tech_talent.xlsx"
    type: "value"
    weight: "auto"
    
  - name: "innovation_ranking"
    path: "data/innovation_rank.xlsx"
    type: "rank"
    weight: 0.3
```

## ⚙️ **Configuration Details**

### Indicator Types
- **`value`**: Numerical data (GDP, spending, population, capacity, etc.)
- **`rank`**: Ranking data where 1 = best, higher numbers = worse performance

### Weight Options
- **`"auto"`**: System automatically calculates optimal weight based on correlation with market data
- **`0.0 to 1.0`**: Manual weight (higher = more influence on distribution)

### Data Formats
- **`"wide"`**: Years as columns (2020, 2021, 2022, ...)
- **`"long"`**: Separate Year and Value columns

### Best Practices
1. **Start simple** - Try without indicators first
2. **Use 2-5 indicators** - More isn't always better
3. **Choose relevant indicators** - Must logically influence your market
4. **Start with auto weights** - Fine-tune manually if needed
5. **Mix types** - Combine value and rank indicators when possible

## 🔧 **Template Files**

Use these template files to get started:

- `config/template_no_indicators.yaml` - Framework without any indicators
- `config/template_custom_indicators.yaml` - Framework with custom indicators
- `config/sample_config.yaml` - General template with examples

## 📝 **Data File Format**

Your indicator data files should have these columns:

### Required Columns
- `idGeo` - Country ID (must match your country historical data)
- `Country` - Country name

### Data Columns (choose format)

**Wide Format:**
```csv
idGeo,Country,2020,2021,2022,2023
1,United States,1000,1100,1200,1300
2,Germany,800,850,900,950
```

**Long Format:**
```csv
idGeo,Country,Year,Value
1,United States,2020,1000
1,United States,2021,1100
2,Germany,2020,800
2,Germany,2021,850
```

## 🎉 **Remember**

The framework is designed to be **market-agnostic** and **user-driven**. There are no hardcoded assumptions about which indicators you should use. Choose what makes sense for YOUR market, or use none at all!