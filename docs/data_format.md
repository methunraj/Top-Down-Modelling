# Detailed Input Data Structure for Universal Market Estimation Framework

## Global Market Forecast Data Structure

**File Type:** Excel (.xlsx) or CSV (.csv)

**Required Structure:**
```
Year | Value           | Type
-----|-----------------|------------
2020 | 51963860168.127 | Historical
2021 | 126656404051.53 | Historical
2022 | 67189149798.755 | Historical
2023 | 53339629642.902 | Historical
2024 | 77160954396.945 | Historical
2025 | 103182643919.19 | Forecast
...  | ...             | ...
2031 | 446970682157.38 | Forecast
```

**Key Fields:**
- `Year`: The year of the forecast/historical data (numeric or string)
- `Value`: The global market value (numeric)
- `Type`: Indicates whether the data is historical or forecasted (string)

**Optional Fields:**
- `Growth_Rate`: Year-over-year growth rate (can be calculated if not provided)
- `Confidence`: Confidence level for forecast values

## Country Historical Data Structure

**File Type:** Excel (.xlsx) or CSV (.csv)

**Required Structure:**
```
idGeo | Country   | nameVertical           | 2020          | 2021          | 2022          | 2023
------|-----------|------------------------|---------------|---------------|---------------|---------------
357   | Algeria   | Machine Learning revenue| 101182011.931 | 246620434.748 | 130828105.044 | 103860856.862
114   | Argentina | Machine Learning revenue| 327109524.329 | 797294811.212 | 422951850.761 | 335769914.397
107   | Australia | Machine Learning revenue| 883151001.810 | 2152587004.93 | 1141912182.11 | 906532871.170
...   | ...       | ...                    | ...           | ...           | ...           | ...
```

**Key Fields:**
- `idGeo`: Unique country identifier (numeric)
- `Country`: Country name (string)
- `nameVertical`: Market/vertical name (string)
- Year columns (`2020`, `2021`, etc.): Market values for each country and year (numeric)

**Alternative Structure (Accepted):**
```
idGeo | Country   | Year | Value           | nameVertical
------|-----------|------|-----------------|------------------------
357   | Algeria   | 2020 | 101182011.931   | Machine Learning revenue
357   | Algeria   | 2021 | 246620434.748   | Machine Learning revenue
357   | Algeria   | 2022 | 130828105.044   | Machine Learning revenue
...   | ...       | ...  | ...             | ...
```

## Indicator Data Structure

Each indicator file should follow one of these formats:

### Wide Format (Years as Columns)

**Example: Secure Internet Servers**
```
idGeo | Country   | 2019  | 2020  | 2021  | 2022  | 2023  | 2024    | ...  | 2031
------|-----------|-------|-------|-------|-------|-------|---------|------|-------
357   | Algeria   | 2154  | 2106  | 2780  | 3414  | 4630  | 4680.4  | ...  | 6163.7
114   | Argentina | 135647| 167248| 231834| 240087| 248239| 275497.2| ...  | 386016.4
...   | ...       | ...   | ...   | ...   | ...   | ...   | ...     | ...  | ...
```

### Long Format (Year in a Column)

**Example: R&D Expenditure**
```
idGeo | Country   | Year | Value
------|-----------|------|---------------
357   | Algeria   | 2020 | 939211800
357   | Algeria   | 2021 | 1115304000
357   | Algeria   | 2022 | 1421475300
...   | ...       | ...  | ...
```

### Single Value Format (e.g., Rankings)

**Example: AI Ranking**
```
idGeo | Country       | 2024
------|---------------|-----
109   | United States | 1
361   | Mainland China| 2
124   | Singapore     | 3
...   | ...           | ...
```

**Required Fields for All Indicator Files:**
- `idGeo`: Must match country identifiers in the country historical data
- `Country`: Country name (for readability)
- Year values: Either as columns (wide format) or as a column (long format)

## Country Reference Data

**Purpose:** Provides common reference information about countries for consistency

**Recommended Structure:**
```
idGeo | Country     | Region          | Development_Level | ISO_Code | Population
------|-------------|-----------------|-------------------|----------|------------
357   | Algeria     | Middle East/Africa | Developing     | DZ       | 44700000
114   | Argentina   | Latin America   | Developing        | AR       | 45810000
107   | Australia   | Asia Pacific    | Developed         | AU       | 25690000
...   | ...         | ...             | ...               | ...      | ...
```

## Configuration File Structure

**File Type:** YAML or JSON

**Required Structure:**
```yaml
# Paths to input files
data_sources:
  global_forecast:
    path: "path/to/global_forecast.xlsx"
    sheet_name: "Sheet1"  # Optional
    
  country_historical:
    path: "path/to/country_historical.xlsx"
    sheet_name: "Sheet1"  # Optional
    
  indicators:
    - name: "secure_servers"
      path: "path/to/secure_servers.xlsx"
      sheet_name: "Sheet1"  # Optional
      weight: 0.05  # Optional, will be calculated if not provided
      
    - name: "rd_expenditure"
      path: "path/to/rd_expenditure.xlsx"
      sheet_name: "Sheet1"
      weight: 0.25
      
    # Additional indicators...

# Column mapping for flexibility
column_mapping:
  global_forecast:
    year_column: "Year"
    value_column: "Value"
    type_column: "Type"
    
  country_historical:
    id_column: "idGeo"
    country_column: "Country"
    vertical_column: "nameVertical"
    
  indicators:
    # Common mapping for all indicators, can be overridden in individual indicator configs
    id_column: "idGeo"
    country_column: "Country"
    # For long format indicators
    year_column: "Year"
    value_column: "Value"

# Additional configuration parameters...
```

## File Format Requirements

### Excel Files
- First row must contain column headers
- No merged cells in data area
- No hidden rows/columns containing data
- No formulas in critical data fields (values should be actual numbers)

### CSV Files
- First row must contain column headers
- Standard CSV format (comma-separated)
- UTF-8 encoding recommended
- No text qualifiers around numeric values

## Data Quality Requirements

1. **Country Consistency**:
   - Same set of countries across all files (ideally)
   - Same country identifiers (idGeo) used consistently
   - Same country names used consistently

2. **Numeric Data**:
   - Market values and indicators should be numeric
   - No text in numeric fields
   - Consistent units across years (e.g., all in USD)

3. **Time Periods**:
   - Consistent year format (numeric, YYYY)
   - No gaps in critical time series
   - Historical data properly labeled/distinguished from forecasts

4. **Nulls/Missing Values**:
   - Missing historical data clearly marked (empty cell, NULL, or N/A)
   - No missing values for key identifiers (idGeo, Country)

This detailed input structure ensures the system can process various market data types consistently while remaining flexible enough to handle different formatting conventions. The configuration-driven approach allows for adapting to different input structures without changing the core code. 