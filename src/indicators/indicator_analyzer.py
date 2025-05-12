"""
Indicator Analyzer Module - Dynamic indicator evaluation for market forecasting

This module provides functionality to analyze, evaluate, and weight indicators
for market forecasting in a market-agnostic way, making it universally applicable
to any market type.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeightTransformation:
    """Class to handle different weight transformation methods"""
    
    @staticmethod
    def squared(correlation: float) -> float:
        """Traditional squared correlation transformation"""
        return correlation * correlation
    
    @staticmethod
    def log_transform(correlation: float, base: float = 2.0, scale: float = 1.0) -> float:
        """
        Logarithmic transformation with adjustable base and scale
        
        Args:
            correlation: Correlation value to transform
            base: Base of the logarithm (default: 2.0)
            scale: Scaling factor for the transformation (default: 1.0)
        """
        abs_corr = abs(correlation)
        # Using log(1 + x) to ensure output is between 0 and 1
        transformed = np.log(1 + abs_corr) / np.log(base)
        return transformed * scale

    @staticmethod
    def sigmoid(correlation: float, steepness: float = 5.0) -> float:
        """
        Sigmoid transformation with adjustable steepness
        
        Args:
            correlation: Correlation value to transform
            steepness: Controls how quickly the function transitions (default: 5.0)
        """
        abs_corr = abs(correlation)
        return 1 / (1 + np.exp(-steepness * (abs_corr - 0.5)))

class SignificanceAdjustment:
    """Class to handle different significance adjustment methods"""
    
    @staticmethod
    def stepped(p_value: float) -> float:
        """Traditional stepped significance adjustment"""
        if p_value <= 0.01:
            return 1.0
        elif p_value <= 0.05:
            return 0.7
        elif p_value <= 0.1:
            return 0.4
        else:
            return 0.2
    
    @staticmethod
    def continuous(p_value: float) -> float:
        """Continuous significance adjustment using exponential decay"""
        return np.exp(-5 * p_value)  # Decay rate of 5 gives good spread

class IndicatorAnalyzer:
    """
    Universal indicator analyzer for market forecasting
    
    This class provides functionality to analyze, evaluate, and weight indicators
    for market forecasting in a market-agnostic way.
    """
    
    def __init__(self, config_manager, data_loader):
        """
        Initialize the IndicatorAnalyzer
        
        Args:
            config_manager: Configuration manager instance for accessing settings
            data_loader: Data loader instance for accessing market data
        """
        self.config_manager = config_manager
        self.data_loader = data_loader
        self.indicator_weights = {}
        self.indicator_correlations = {}
        self.indicator_scores = {}
        
        # Weight calculation parameters
        self.weight_params = {
            'transformation': 'log',  # 'log', 'squared', or 'sigmoid'
            'log_base': 2.0,
            'log_scale': 1.0,
            'sigmoid_steepness': 5.0,
            'significance_method': 'continuous'  # 'stepped' or 'continuous'
        }
        
        # Store both old and new weights for comparison
        self.old_weights = {}
        self.new_weights = {}
    
    def _handle_rank_indicator(self, indicator_data: pd.DataFrame, country_year_data: pd.DataFrame, 
                             indicator_name: str, id_col: str) -> Dict[str, float]:
        """
        Handle rank-based indicators differently from continuous value indicators
        
        Args:
            indicator_data: DataFrame containing the rank indicator data
            country_year_data: DataFrame containing the country market data
            indicator_name: Name of the indicator
            id_col: Name of the ID column
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            # Merge country data with indicator data
            merged_data = pd.merge(
                country_year_data,
                indicator_data,
                on=[id_col],
                how='inner',
                suffixes=('_country', '_indicator')
            )
            
            if merged_data.empty:
                logger.warning(f"No matching data found between country data and rank indicator '{indicator_name}'")
                return {'correlation': 0.0, 'p_value': 1.0, 'weight': 0.0}
            
            # Invert ranks (1 becomes highest value)
            max_rank = merged_data['Value_indicator'].max()
            merged_data['inverted_rank'] = max_rank - merged_data['Value_indicator'] + 1
            
            # Normalize inverted ranks to [0,1] range
            scaler = MinMaxScaler()
            normalized_ranks = scaler.fit_transform(merged_data[['inverted_rank']])
            merged_data['normalized_rank'] = normalized_ranks
            
            # Calculate correlation with market values
            correlation, p_value = self._calculate_correlation(
                merged_data['Value_country'], 
                merged_data['normalized_rank']
            )
            
            # For rank indicators, use logarithmic transformation and coverage
            coverage_ratio = len(merged_data) / len(country_year_data)
            transformed_corr = WeightTransformation.log_transform(correlation)
            base_weight = transformed_corr * coverage_ratio
            
            # Apply significance factor with less severe penalties for rank indicators
            if p_value <= 0.05:
                weight = base_weight
            elif p_value <= 0.1:
                weight = base_weight * 0.7
            else:
                weight = base_weight * 0.4
                
            return {
                'correlation': correlation,
                'p_value': p_value,
                'weight': weight,
                'countries_covered': len(merged_data),
                'data_completeness': coverage_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing rank indicator '{indicator_name}': {str(e)}")
            return {'correlation': 0.0, 'p_value': 1.0, 'weight': 0.0}

    def analyze_indicators(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze all configured indicators and calculate their weights
        
        Returns:
            Dictionary with indicator analysis results, including weights and correlations
        """
        # Load country historical data and indicators
        country_data = self.data_loader.load_country_historical()
        all_indicators = self.data_loader.load_all_indicators()
        
        if all_indicators.empty:
            logger.warning("No indicator data available for analysis")
            return {}
        
        # Get indicator list from configuration
        indicator_configs = self.config_manager.get_indicators()
        
        # Get column names from mapping
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = column_mapping.get('id_column', 'idGeo')
        
        # Get the most recent common year for analysis
        country_years = set(country_data['Year'].unique())
        indicator_years = set(all_indicators['Year'].unique())
        common_years = sorted(country_years.intersection(indicator_years), reverse=True)
        
        if not common_years:
            logger.warning("No common years found between country data and indicators")
            return {}
        
        # Use the most recent common year for correlation analysis
        analysis_year = common_years[0]
        logger.info(f"Using year {analysis_year} for indicator analysis")
        
        # Filter data for the analysis year
        country_year_data = country_data[country_data['Year'] == analysis_year]
        
        # Calculate correlations and weights for each indicator
        results = {}
        self.old_weights = {}
        self.new_weights = {}
        
        # Debug logging
        logger.info(f"Processing {len(indicator_configs)} indicators")
        
        for indicator_config in indicator_configs:
            indicator_name = indicator_config.get('name')
            is_rank_indicator = indicator_config.get('type') == 'rank'
            
            logger.info(f"Processing indicator: {indicator_name}")
            
            try:
                # Get indicator data for the specific indicator
                indicator_data = all_indicators[all_indicators['Indicator'] == indicator_name]
                
                if indicator_data.empty:
                    logger.warning(f"No data found for indicator '{indicator_name}'")
                    continue
                
                logger.info(f"Found {len(indicator_data)} rows for {indicator_name}")
                
                # For rank indicators, we don't filter by year since they're single-year
                if not is_rank_indicator:
                    indicator_data = indicator_data[indicator_data['Year'] == analysis_year]
                
                if indicator_data.empty:
                    logger.warning(f"No data for year {analysis_year} found for indicator '{indicator_name}'")
                    continue
                
                # Handle rank indicators differently
                if is_rank_indicator:
                    indicator_results = self._handle_rank_indicator(
                        indicator_data, country_year_data, indicator_name, id_col
                    )
                else:
                    # Original correlation-based analysis for continuous indicators
                    merged_data = pd.merge(
                        country_year_data,
                        indicator_data,
                        on=[id_col, 'Year'] if not is_rank_indicator else [id_col],
                        how='inner',
                        suffixes=('_country', '_indicator')
                    )
                    
                    if merged_data.empty:
                        logger.warning(f"No matching data found between country data and indicator '{indicator_name}'")
                        continue
                    
                    correlation, p_value = self._calculate_correlation(
                        merged_data['Value_country'], 
                        merged_data['Value_indicator']
                    )
                    
                    # Calculate both old and new weights
                    old_weight, new_weight = self._calculate_weight(correlation, p_value)
                    
                    indicator_results = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'weight': new_weight,
                        'countries_covered': len(merged_data),
                        'data_completeness': len(merged_data) / len(country_year_data)
                    }
                
                # Store results
                results[indicator_name] = indicator_results
                self.indicator_correlations[indicator_name] = indicator_results['correlation']
                self.old_weights[indicator_name] = indicator_results.get('old_weight', old_weight)
                self.new_weights[indicator_name] = indicator_results.get('weight', new_weight)
                
                logger.info(f"Successfully processed {indicator_name}")
                
            except Exception as e:
                logger.error(f"Error analyzing indicator '{indicator_name}': {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Normalize both old and new weights
        total_old = sum(self.old_weights.values())
        total_new = sum(self.new_weights.values())
        
        if total_old > 0:
            self.old_weights = {k: v/total_old for k, v in self.old_weights.items()}
        if total_new > 0:
            self.new_weights = {k: v/total_new for k, v in self.new_weights.items()}
            self.indicator_weights = self.new_weights.copy()
        
        # Log final weights
        logger.info("Final indicator weights:")
        for name, weight in self.indicator_weights.items():
            logger.info(f"{name}: {weight:.4f}")
        
        # Generate visualizations
        self.visualize_weight_comparison()
        
        return results
    
    def _calculate_correlation(self, market_values: pd.Series, indicator_values: pd.Series) -> Tuple[float, float]:
        """
        Calculate correlation between market values and indicator values
        
        Args:
            market_values: Series of market values
            indicator_values: Series of indicator values
            
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        # Handle non-numeric values
        market_values = pd.to_numeric(market_values, errors='coerce')
        indicator_values = pd.to_numeric(indicator_values, errors='coerce')
        
        # Drop NaN values
        valid_data = pd.DataFrame({
            'market': market_values,
            'indicator': indicator_values
        }).dropna()
        
        if len(valid_data) < 3:
            logger.warning("Not enough valid data points for correlation calculation")
            return 0.0, 1.0
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(valid_data['market'], valid_data['indicator'])
        
        # Handle NaN correlation (can happen with constant values)
        if np.isnan(correlation):
            correlation = 0.0
            p_value = 1.0
        
        return correlation, p_value
    
    def _calculate_weight(self, correlation: float, p_value: float) -> Tuple[float, float]:
        """
        Calculate both old and new weights for comparison
        
        Returns:
            Tuple of (old_weight, new_weight)
        """
        # Calculate old weight (squared correlation)
        old_weight = WeightTransformation.squared(correlation)
        if p_value > 0.05:
            old_weight *= 0.5
        if p_value > 0.1:
            old_weight *= 0.4
            
        # Calculate new weight based on selected transformation
        if self.weight_params['transformation'] == 'log':
            new_weight = WeightTransformation.log_transform(
                correlation,
                base=self.weight_params['log_base'],
                scale=self.weight_params['log_scale']
            )
        elif self.weight_params['transformation'] == 'sigmoid':
            new_weight = WeightTransformation.sigmoid(
                correlation,
                steepness=self.weight_params['sigmoid_steepness']
            )
        else:  # squared
            new_weight = WeightTransformation.squared(correlation)
            
        # Apply significance adjustment
        if self.weight_params['significance_method'] == 'continuous':
            significance = SignificanceAdjustment.continuous(p_value)
        else:
            significance = SignificanceAdjustment.stepped(p_value)
            
        new_weight *= significance
        
        return old_weight, new_weight
    
    def _update_indicator_weights_in_config(self) -> None:
        """
        Update indicator weights in configuration if set to "auto"
        """
        indicators = self.config_manager.get_indicators()
        updated_indicators = False
        
        for indicator in indicators:
            indicator_name = indicator.get('name')
            if indicator_name in self.indicator_weights and indicator.get('weight') == 'auto':
                weight = self.indicator_weights[indicator_name]
                indicator['weight'] = weight
                updated_indicators = True
        
        if updated_indicators:
            # Apply updates to the configuration
            data_sources = self.config_manager.config.get('data_sources', {})
            data_sources['indicators'] = indicators
    
    def apply_indicator_adjustments(self, country_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply indicator-based adjustments to country market shares
        
        Args:
            country_df: DataFrame with country market shares
            
        Returns:
            DataFrame with adjusted market shares
        """
        # Check if we need to analyze indicators first
        if not self.indicator_weights:
            self.analyze_indicators()
        
        if not self.indicator_weights:
            logger.warning("No indicator weights available, skipping adjustments")
            return country_df
        
        # Make a copy of the input data to avoid modifying the original
        adjusted_df = country_df.copy()
        
        # Get column names from mapping
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = column_mapping.get('id_column', 'idGeo')
        
        # Load all indicators
        all_indicators = self.data_loader.load_all_indicators()
        
        if all_indicators.empty:
            logger.warning("No indicator data available for adjustments")
            return adjusted_df
        
        # Get the most recent year in the data
        most_recent_year = adjusted_df['Year'].max() if 'Year' in adjusted_df.columns else None
        
        # If there's no year column, we assume it's a single year dataframe
        if 'Year' not in adjusted_df.columns:
            # Create an adjustment score for each country
            adjustment_scores = pd.DataFrame({id_col: adjusted_df[id_col].unique()})
            adjustment_scores['score'] = 1.0  # Start with neutral score
            
            # Apply each indicator's adjustment
            for indicator_name, weight in self.indicator_weights.items():
                if weight <= 0:
                    continue
                
                # Get the most recent indicator data
                indicator_df = all_indicators[all_indicators['Indicator'] == indicator_name]
                if indicator_df.empty:
                    continue
                
                indicator_latest_year = indicator_df['Year'].max()
                latest_indicator = indicator_df[indicator_df['Year'] == indicator_latest_year]
                
                # Calculate min/max for normalization
                min_val = latest_indicator['Value'].min()
                max_val = latest_indicator['Value'].max()
                range_val = max_val - min_val
                
                if range_val <= 0:
                    continue
                
                # Merge indicator data with adjustment scores
                merged = pd.merge(
                    adjustment_scores,
                    latest_indicator[[id_col, 'Value']],
                    on=id_col,
                    how='left'
                )
                
                # Normalize indicator values to 0-2 range (0.5-1.5 after applying factor)
                merged['normalized'] = (merged['Value'] - min_val) / range_val
                merged['normalized'] = merged['normalized'] * 1.0 + 0.5
                
                # Apply weighted adjustment
                factor = 1.0 + (weight * (merged['normalized'] - 1.0))
                merged['score'] = merged['score'] * factor
                
                # Update scores
                adjustment_scores = merged[[id_col, 'score']]
            
            # Merge adjustment scores back to the original data
            adjusted_df = pd.merge(
                adjusted_df,
                adjustment_scores,
                on=id_col,
                how='left'
            )
            
            # Fill missing scores with 1 (neutral)
            adjusted_df['score'] = adjusted_df['score'].fillna(1.0)
            
            # Apply score to market share
            if 'market_share' in adjusted_df.columns:
                # Store original share for reference
                adjusted_df['original_share'] = adjusted_df['market_share']
                
                # Apply adjustment
                adjusted_df['market_share'] = adjusted_df['market_share'] * adjusted_df['score']
                
                # Normalize shares to sum to 100%
                total_share = adjusted_df['market_share'].sum()
                if total_share > 0:
                    adjusted_df['market_share'] = adjusted_df['market_share'] / total_share * 100
            
            # Apply score to value if it exists
            if 'Value' in adjusted_df.columns:
                # Store original value for reference
                adjusted_df['original_value'] = adjusted_df['Value']
                
                # Apply adjustment
                adjusted_df['adjusted_value'] = adjusted_df['Value'] * adjusted_df['score']
        
        # If we have a Year column, process each year separately
        else:
            # Initialize empty DataFrame for all adjustment scores
            adjustment_scores_all = pd.DataFrame()
            
            for year in adjusted_df['Year'].unique():
                # Create an adjustment score for each country in this year
                year_df = adjusted_df[adjusted_df['Year'] == year].copy()
                adjustment_scores = pd.DataFrame({id_col: year_df[id_col].unique()})
                adjustment_scores['score'] = 1.0  # Start with neutral score
                adjustment_scores['Year'] = year  # Add Year column immediately
                
                # Apply each indicator's adjustment
                for indicator_name, weight in self.indicator_weights.items():
                    if weight <= 0:
                        continue
                    
                    # Get indicator data for this year or the closest previous year
                    indicator_df = all_indicators[all_indicators['Indicator'] == indicator_name]
                    if indicator_df.empty:
                        continue
                    
                    # Find closest year that's not after the target year
                    available_years = sorted(indicator_df['Year'].unique())
                    closest_year = None
                    for indicator_year in available_years:
                        if indicator_year <= year:
                            closest_year = indicator_year
                    
                    if closest_year is None:
                        continue
                    
                    year_indicator = indicator_df[indicator_df['Year'] == closest_year]
                    
                    # Calculate min/max for normalization
                    min_val = year_indicator['Value'].min()
                    max_val = year_indicator['Value'].max()
                    range_val = max_val - min_val
                    
                    if range_val <= 0:
                        continue
                    
                    # Merge indicator data with adjustment scores
                    merged = pd.merge(
                        adjustment_scores,
                        year_indicator[[id_col, 'Value']],
                        on=id_col,
                        how='left'
                    )
                    
                    # Normalize indicator values to 0-2 range (0.5-1.5 after applying factor)
                    merged['normalized'] = (merged['Value'] - min_val) / range_val
                    merged['normalized'] = merged['normalized'] * 1.0 + 0.5
                    
                    # Apply weighted adjustment
                    factor = 1.0 + (weight * (merged['normalized'] - 1.0))
                    merged['score'] = merged['score'] * factor
                    
                    # Update scores
                    adjustment_scores = merged[[id_col, 'Year', 'score']]
                
                # Append scores for this year to the all scores DataFrame
                adjustment_scores_all = pd.concat([adjustment_scores_all, adjustment_scores], ignore_index=True)
            
            # Merge adjustment scores back to the original data
            adjusted_df = pd.merge(
                adjusted_df,
                adjustment_scores_all,
                on=[id_col, 'Year'],
                how='left'
            )
            
            # Fill missing scores with 1 (neutral)
            adjusted_df['score'] = adjusted_df['score'].fillna(1.0)
            
            # Apply score to market share if it exists
            if 'market_share' in adjusted_df.columns:
                # Store original share for reference
                adjusted_df['original_share'] = adjusted_df['market_share']
                
                # Apply adjustment
                adjusted_df['market_share'] = adjusted_df['market_share'] * adjusted_df['score']
                
                # Normalize shares to sum to 100% for each year if Year column exists
                if 'Year' in adjusted_df.columns:
                    for year in adjusted_df['Year'].unique():
                        year_mask = adjusted_df['Year'] == year
                        year_total = adjusted_df.loc[year_mask, 'market_share'].sum()
                        if year_total > 0:
                            adjusted_df.loc[year_mask, 'market_share'] = (
                                adjusted_df.loc[year_mask, 'market_share'] / year_total * 100
                            )
                else:
                    # Normalize shares to sum to 100% for single year data
                    total_share = adjusted_df['market_share'].sum()
                    if total_share > 0:
                        adjusted_df['market_share'] = adjusted_df['market_share'] / total_share * 100
            
            # Apply score to value if it exists
            if 'Value' in adjusted_df.columns:
                # Store original value for reference
                adjusted_df['original_value'] = adjusted_df['Value']
                
                # Apply adjustment
                adjusted_df['adjusted_value'] = adjusted_df['Value'] * adjusted_df['score']
        
        return adjusted_df
    
    def get_indicator_weights(self) -> Dict[str, float]:
        """
        Get calculated indicator weights
        
        Returns:
            Dictionary of indicator names to weights
        """
        # Calculate weights if not already done
        if not self.indicator_weights:
            self.analyze_indicators()
        
        return self.indicator_weights
    
    def get_indicator_correlations(self) -> Dict[str, float]:
        """
        Get calculated indicator correlations
        
        Returns:
            Dictionary of indicator names to correlation coefficients
        """
        # Calculate correlations if not already done
        if not self.indicator_correlations:
            self.analyze_indicators()
        
        return self.indicator_correlations
    
    def save_indicator_analysis(self) -> str:
        """
        Save indicator analysis results to an Excel file
        
        Returns:
            Path to the saved Excel file
        """
        # Ensure we have analyzed indicators
        if not self.indicator_weights:
            self.analyze_indicators()
            
        # Create a DataFrame with all indicator analysis results
        analysis_data = []
        
        for indicator_name in self.indicator_weights.keys():
            row = {
                'Indicator': indicator_name,
                'Weight': self.indicator_weights.get(indicator_name, 0),
                'Correlation': self.indicator_correlations.get(indicator_name, 0),
                'Type': self._get_indicator_type(indicator_name)
            }
            analysis_data.append(row)
            
        # Convert to DataFrame
        df = pd.DataFrame(analysis_data)
        
        # Sort by weight in descending order
        df = df.sort_values('Weight', ascending=False)
        
        # Format the numbers
        df['Weight'] = df['Weight'].apply(lambda x: f"{x:.4f}")
        df['Correlation'] = df['Correlation'].apply(lambda x: f"{x:.4f}")
        
        # Get the output directory from config
        output_dir = self.config_manager.config.get('output', {}).get('save_path', '')
        if not output_dir:
            output_dir = 'output'
            
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the output file path
        output_file = os.path.join(output_dir, 'indicator_weights_analysis.xlsx')
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # Write the main analysis sheet
            df.to_excel(writer, sheet_name='Indicator Weights', index=False)
            
            # Get the workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Indicator Weights']
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D9E1F2',
                'border': 1
            })
            
            cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })
            
            # Apply formats
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Set column widths
            worksheet.set_column('A:A', 20)  # Indicator name
            worksheet.set_column('B:C', 12)  # Weight and Correlation
            worksheet.set_column('D:D', 15)  # Type
            
            # Add a title
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'center'
            })
            worksheet.merge_range('A1:D1', 'Indicator Analysis Results', title_format)
            
            # Write the data starting from row 2
            df.to_excel(writer, sheet_name='Indicator Weights', index=False, startrow=1)
            
            # Add explanatory notes
            notes = [
                "Notes:",
                "- Weights are normalized to sum to 1",
                "- Correlation shows the relationship with market values",
                "- Type indicates whether the indicator is rank-based or continuous"
            ]
            
            row_offset = len(df) + 4
            for i, note in enumerate(notes):
                worksheet.write(row_offset + i, 0, note)
        
        logger.info(f"Saved indicator analysis to: {output_file}")
        return output_file
    
    def _get_indicator_type(self, indicator_name: str) -> str:
        """
        Get the type of an indicator from the configuration
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            String indicating the indicator type ('rank' or 'continuous')
        """
        indicators = self.config_manager.get_indicators()
        for indicator in indicators:
            if indicator.get('name') == indicator_name:
                return indicator.get('type', 'continuous')
        return 'continuous'

    def visualize_weight_comparison(self, output_dir: str = None) -> str:
        """
        Create visualizations comparing old and new weight calculations

        Args:
            output_dir: Directory to save visualizations (default: config output dir)

        Returns:
            Path to the saved visualization file
        """
        if not output_dir:
            output_dir = self.config_manager.config.get('output', {}).get('save_path', 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Create comparison DataFrame
        comparison_data = []

        # Debug logging
        logger.info("Creating visualization with the following indicators:")
        logger.info(f"Old weights: {self.old_weights}")
        logger.info(f"New weights: {self.new_weights}")

        # Ensure we have all indicators
        all_indicators = set(self.old_weights.keys()) | set(self.new_weights.keys())

        for indicator in all_indicators:
            comparison_data.append({
                'Indicator': indicator,
                'Old Weight': self.old_weights.get(indicator, 0),
                'New Weight': self.new_weights.get(indicator, 0),
                'Correlation': self.indicator_correlations.get(indicator, 0),
                'Type': self._get_indicator_type(indicator)
            })

        df = pd.DataFrame(comparison_data)

        # Sort by new weights for better visualization
        df = df.sort_values('New Weight', ascending=False)

        # Create visualization
        plt.figure(figsize=(15, 10))

        # Plot 1: Weight comparison
        plt.subplot(2, 2, 1)
        x = range(len(df))
        width = 0.35

        # Create bar chart
        plt.bar([i - width/2 for i in x], df['Old Weight'], width, label='Old Weights', alpha=0.6)
        plt.bar([i + width/2 for i in x], df['New Weight'], width, label='New Weights', alpha=0.6)

        # Customize x-axis
        plt.xticks([i for i in x], df['Indicator'], rotation=45, ha='right')
        plt.title('Weight Comparison: Old vs New')
        plt.legend()

        # Add value labels on top of bars
        for i, v in enumerate(df['Old Weight']):
            plt.text(i - width/2, v, f'{v:.3f}', ha='center', va='bottom')
        for i, v in enumerate(df['New Weight']):
            plt.text(i + width/2, v, f'{v:.3f}', ha='center', va='bottom')

        # Plot 2: Correlation vs Weights scatter
        plt.subplot(2, 2, 2)
        plt.scatter(df['Correlation'], df['Old Weight'], alpha=0.6, label='Old Weights')
        plt.scatter(df['Correlation'], df['New Weight'], alpha=0.6, label='New Weights')

        # Add indicator labels to scatter plot
        for i, txt in enumerate(df['Indicator']):
            plt.annotate(txt, (df['Correlation'].iloc[i], df['Old Weight'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
            plt.annotate(txt, (df['Correlation'].iloc[i], df['New Weight'].iloc[i]),
                        xytext=(5, -5), textcoords='offset points', fontsize=8)

        plt.xlabel('Correlation')
        plt.ylabel('Weight')
        plt.title('Correlation vs Weights')
        plt.legend()

        # Plot 3: Weight distribution
        plt.subplot(2, 2, 3)
        sns.kdeplot(data=df[['Old Weight', 'New Weight']], fill=True, alpha=0.5)
        plt.title('Weight Distribution')

        # Plot 4: Type-based comparison
        plt.subplot(2, 2, 4)
        df_melted = pd.melt(df,
                            id_vars=['Indicator', 'Type'],
                            value_vars=['Old Weight', 'New Weight'],
                            var_name='Method', value_name='Weight')
        sns.boxplot(data=df_melted, x='Type', y='Weight', hue='Method')
        plt.title('Weight Distribution by Indicator Type')

        plt.tight_layout()

        # Save the figure
        output_file = os.path.join(output_dir, 'weight_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Also save the comparison data to Excel with more detail
        excel_file = os.path.join(output_dir, 'weight_comparison.xlsx')
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            # Add detailed comparison sheet
            df_detailed = df.copy()
            df_detailed['Weight Change'] = df_detailed['New Weight'] - df_detailed['Old Weight']
            df_detailed['Change %'] = ((df_detailed['New Weight'] / df_detailed['Old Weight'] - 1) * 100).fillna(0)

            df_detailed.to_excel(writer, sheet_name='Weight Comparison', index=False)

            # Add formatting
            workbook = writer.book
            worksheet = writer.sheets['Weight Comparison']

            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D9E1F2',
                'border': 1
            })

            # Apply formats
            for col_num, value in enumerate(df_detailed.columns.values):
                worksheet.write(0, col_num, value, header_format)
                worksheet.set_column(col_num, col_num, 15)

            # Add transformation parameters
            param_data = pd.DataFrame([{
                'Parameter': 'Transformation Method',
                'Value': self.weight_params['transformation']
            }, {
                'Parameter': 'Log Base',
                'Value': self.weight_params['log_base']
            }, {
                'Parameter': 'Log Scale',
                'Value': self.weight_params['log_scale']
            }, {
                'Parameter': 'Sigmoid Steepness',
                'Value': self.weight_params['sigmoid_steepness']
            }, {
                'Parameter': 'Significance Method',
                'Value': self.weight_params['significance_method']
            }])

            param_data.to_excel(writer, sheet_name='Parameters', index=False, startrow=1)

        return output_file

    def add_indicator(self, indicator_name: str, indicator_data: pd.DataFrame, indicator_type: str = 'continuous', weight: Union[float, str] = 'auto') -> bool:
        """
        Add a new indicator to the system from the Streamlit interface

        Args:
            indicator_name: Name of the indicator
            indicator_data: DataFrame containing the indicator data
            indicator_type: Type of indicator ('continuous' or 'rank')
            weight: Weight to assign to the indicator (numeric or 'auto')

        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Adding indicator '{indicator_name}' with type '{indicator_type}'")

            # Validate the data format
            required_columns = ['idGeo', 'Year', 'Value']
            for col in required_columns:
                if col not in indicator_data.columns:
                    logger.error(f"Missing required column '{col}' in indicator data")
                    return False

            # Add indicator to configuration
            existing_indicators = self.config_manager.get_indicators()

            # Check if indicator already exists
            for indicator in existing_indicators:
                if indicator.get('name') == indicator_name:
                    logger.info(f"Indicator '{indicator_name}' already exists, updating")
                    indicator['type'] = indicator_type
                    indicator['weight'] = weight
                    break
            else:
                # Add new indicator
                new_indicator = {
                    'name': indicator_name,
                    'type': indicator_type,
                    'weight': weight
                }
                existing_indicators.append(new_indicator)

            # Update configuration
            data_sources = self.config_manager.config.setdefault('data_sources', {})
            data_sources['indicators'] = existing_indicators

            # Save indicator data to file
            self._save_indicator_data(indicator_name, indicator_data)

            # Reset weights to trigger recalculation
            self.indicator_weights = {}
            self.indicator_correlations = {}

            logger.info(f"Successfully added indicator '{indicator_name}'")
            return True

        except Exception as e:
            logger.error(f"Error adding indicator '{indicator_name}': {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _save_indicator_data(self, indicator_name: str, indicator_data: pd.DataFrame) -> str:
        """
        Save indicator data to the appropriate file

        Args:
            indicator_name: Name of the indicator
            indicator_data: DataFrame containing the indicator data

        Returns:
            Path to the saved file
        """
        # Get the data directory
        data_dir = self.config_manager.config.get('data_sources', {}).get('path', 'data')
        indicators_dir = os.path.join(data_dir, 'indicators')

        # Create directory if it doesn't exist
        os.makedirs(indicators_dir, exist_ok=True)

        # Clean up the indicator name for the filename
        safe_name = indicator_name.replace(' ', '_').lower()

        # Save to CSV
        output_file = os.path.join(indicators_dir, f"{safe_name}.csv")
        indicator_data.to_csv(output_file, index=False)

        logger.info(f"Saved indicator data to {output_file}")
        return output_file

    def set_indicator_weight(self, indicator_name: str, weight: float) -> bool:
        """
        Set the weight for a specific indicator

        Args:
            indicator_name: Name of the indicator
            weight: Weight value to set (between 0 and 1)

        Returns:
            Boolean indicating success
        """
        try:
            # Normalize weight to be between 0 and 1
            weight = max(0, min(1, float(weight)))

            # Update in-memory weights
            self.indicator_weights[indicator_name] = weight

            # Update configuration
            existing_indicators = self.config_manager.get_indicators()
            for indicator in existing_indicators:
                if indicator.get('name') == indicator_name:
                    indicator['weight'] = weight
                    break
            else:
                logger.warning(f"Indicator '{indicator_name}' not found in configuration")
                return False

            # Update configuration
            data_sources = self.config_manager.config.setdefault('data_sources', {})
            data_sources['indicators'] = existing_indicators

            logger.info(f"Set weight for indicator '{indicator_name}' to {weight}")
            return True

        except Exception as e:
            logger.error(f"Error setting weight for indicator '{indicator_name}': {str(e)}")
            return False 