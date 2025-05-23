"""
Causal Indicator Integration Module - Advanced causal analysis for market indicators

This module implements a sophisticated causal inference framework that goes beyond
correlation analysis to identify and quantify causal relationships between
indicators and market outcomes. It supports both model-based approaches and
advanced non-parametric causal inference methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import networkx as nx
import warnings

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CausalIndicatorIntegration:
    """
    Advanced causal indicator integration for market forecasting
    
    This class provides functionality to identify, analyze, and integrate causal
    relationships between indicators and market outcomes, going beyond simple
    correlation to establish more robust causal linkages.
    """
    
    def __init__(self, config_manager, data_loader, indicator_analyzer):
        """
        Initialize the CausalIndicatorIntegration
        
        Args:
            config_manager: Configuration manager instance for accessing settings
            data_loader: Data loader instance for accessing market data
            indicator_analyzer: Indicator analyzer instance for accessing base indicator analysis
        """
        self.config_manager = config_manager
        self.data_loader = data_loader
        self.indicator_analyzer = indicator_analyzer
        
        # Store causal analysis results
        self.causal_strengths = {}
        self.causal_graph = None
        self.feature_importances = {}
        self.lagged_effects = {}
        self.conditional_effects = {}
        
        # Analysis parameters
        self.causal_analysis_params = {
            'method': 'ensemble',  # 'ensemble', 'granger', 'intervention', 'structural'
            'lags': 1,  # Number of lags to consider for time series analysis
            'alpha': 0.05,  # Significance level
            'bootstrap_samples': 100,  # Number of bootstrap samples for robustness
            'regularization': 'elastic_net',  # 'lasso', 'elastic_net', 'none'
            'interaction_detection': True,  # Whether to detect interactions
            'enable_granger': True,  # Enable Granger causality testing
            'enable_conditional': True,  # Enable conditional independence testing
            'enable_feature_importance': True  # Enable feature importance analysis
        }
        
        # Load causal analysis parameters from config if available
        self._load_config_params()
        
        # Prepare causal graph
        self.causal_graph = nx.DiGraph()
    
    def _load_config_params(self) -> None:
        """
        Load causal analysis parameters from configuration
        """
        causal_config = self.config_manager.get_value('indicators.causal_analysis', {})
        
        # Update parameters from config
        if causal_config:
            for key, value in causal_config.items():
                if key in self.causal_analysis_params:
                    self.causal_analysis_params[key] = value
                    
            logger.info(f"Loaded causal analysis parameters from configuration")
    
    def analyze_causal_relationships(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze causal relationships between indicators and market outcomes
        
        This method uses various causal inference techniques to estimate the causal
        effect of each indicator on market outcomes, beyond simple correlation.
        
        Returns:
            Dictionary with causal analysis results for each indicator
        """
        # Load country historical data and indicators
        country_data = self.data_loader.load_country_historical()
        all_indicators = self.data_loader.load_all_indicators()
        
        if all_indicators.empty:
            logger.warning("No indicator data available for causal analysis")
            return {}
        
        # Get indicator list from configuration
        indicator_configs = self.config_manager.get_indicators()
        
        # Get column names from mapping
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = column_mapping.get('id_column', 'idGeo')
        value_col = column_mapping.get('value_column', 'Value')
        
        # Get common years for analysis
        country_years = set(country_data['Year'].unique())
        indicator_years = set(all_indicators['Year'].unique())
        common_years = sorted(country_years.intersection(indicator_years))
        
        if not common_years:
            logger.warning("No common years found between country data and indicators")
            return {}
        
        # Prepare data for causal analysis
        analysis_data = self._prepare_panel_data(country_data, all_indicators, common_years, id_col)
        
        # Analyze based on selected method
        method = self.causal_analysis_params['method']
        
        if method == 'ensemble':
            results = self._ensemble_causal_analysis(analysis_data, indicator_configs)
        elif method == 'granger':
            results = self._granger_causality_analysis(analysis_data, indicator_configs)
        elif method == 'intervention':
            results = self._intervention_analysis(analysis_data, indicator_configs)
        elif method == 'structural':
            results = self._structural_equation_analysis(analysis_data, indicator_configs)
        else:
            logger.warning(f"Unknown causal analysis method: {method}, using ensemble")
            results = self._ensemble_causal_analysis(analysis_data, indicator_configs)
        
        # Build causal graph
        self._build_causal_graph(results)
        
        # Store results
        self.causal_strengths = {ind: res.get('causal_strength', 0) for ind, res in results.items()}
        
        # Visualize causal relationships
        self.visualize_causal_graph()
        
        # Generate comprehensive report
        self._generate_causal_report(results)
        
        return results
    
    def _prepare_panel_data(self, country_data: pd.DataFrame, indicators_data: pd.DataFrame, 
                           years: List[int], id_col: str) -> pd.DataFrame:
        """
        Prepare panel data for causal analysis
        
        Args:
            country_data: DataFrame with country market data
            indicators_data: DataFrame with indicator data
            years: List of years to include
            id_col: Name of country ID column
            
        Returns:
            Panel data DataFrame with country, year, market value, and indicators
        """
        # Filter country data for selected years
        filtered_country = country_data[country_data['Year'].isin(years)].copy()
        
        # Prepare base panel data with country, year, and market value
        panel_data = filtered_country[[id_col, 'Country', 'Year', 'Value']].copy()
        panel_data.rename(columns={'Value': 'market_value'}, inplace=True)
        
        # Add indicators one by one
        for indicator_name in indicators_data['Indicator'].unique():
            indicator_df = indicators_data[indicators_data['Indicator'] == indicator_name]
            
            # Only keep years we're analyzing
            indicator_df = indicator_df[indicator_df['Year'].isin(years)]
            
            if indicator_df.empty:
                continue
            
            # Merge with panel data
            panel_data = pd.merge(
                panel_data,
                indicator_df[[id_col, 'Year', 'Value']],
                on=[id_col, 'Year'],
                how='left',
                suffixes=('', f'_{indicator_name}')
            )
            
            # Rename the indicator column
            panel_data.rename(columns={'Value': indicator_name}, inplace=True)
        
        # Handle missing values
        # For indicators, we use forward and backward fill by country
        indicator_cols = [col for col in panel_data.columns 
                         if col not in [id_col, 'Country', 'Year', 'market_value']]
        
        # Group by country and fill missing values
        for country_id in panel_data[id_col].unique():
            country_mask = panel_data[id_col] == country_id
            
            # Forward fill then backward fill for each indicator
            for ind_col in indicator_cols:
                panel_data.loc[country_mask, ind_col] = (
                    panel_data.loc[country_mask, ind_col].ffill().bfill()
                )
        
        # Add lagged variables if configured
        lags = self.causal_analysis_params['lags']
        if lags > 0:
            # Add lagged values for market_value
            panel_data = self._add_lagged_variables(panel_data, id_col, 'market_value', lags)
            
            # Add lagged values for each indicator
            for ind_col in indicator_cols:
                panel_data = self._add_lagged_variables(panel_data, id_col, ind_col, lags)
        
        # Add interaction terms if configured
        if self.causal_analysis_params['interaction_detection'] and len(indicator_cols) > 1:
            panel_data = self._add_interaction_terms(panel_data, indicator_cols)
        
        # Drop rows with missing values after all preprocessing
        panel_data = panel_data.dropna(subset=['market_value'])
        
        return panel_data
    
    def _add_lagged_variables(self, df: pd.DataFrame, id_col: str, 
                             var_name: str, lags: int) -> pd.DataFrame:
        """
        Add lagged variables to the panel data
        
        Args:
            df: Panel data DataFrame
            id_col: ID column name
            var_name: Variable to lag
            lags: Number of lags
            
        Returns:
            DataFrame with added lagged variables
        """
        result_df = df.copy()
        
        # Add lags for each country
        for country_id in df[id_col].unique():
            country_mask = df[id_col] == country_id
            country_data = df.loc[country_mask].sort_values('Year')
            
            # Add lagged values
            for lag in range(1, lags + 1):
                lag_col = f"{var_name}_lag{lag}"
                country_data[lag_col] = country_data[var_name].shift(lag)
                
                # Update in the result DataFrame
                result_df.loc[country_mask, lag_col] = country_data[lag_col]
        
        return result_df
    
    def _add_interaction_terms(self, df: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
        """
        Add interaction terms between indicators
        
        Args:
            df: Panel data DataFrame
            indicator_cols: List of indicator column names
            
        Returns:
            DataFrame with added interaction terms
        """
        result_df = df.copy()
        
        # Limit to a reasonable number of interactions to avoid explosion
        max_interactions = min(10, len(indicator_cols) * (len(indicator_cols) - 1) // 2)
        
        # Get correlation with target to prioritize interactions
        correlations = {}
        for col in indicator_cols:
            corr = df[col].corr(df['market_value'])
            correlations[col] = abs(corr) if not np.isnan(corr) else 0
        
        # Sort indicators by correlation
        sorted_indicators = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # Use top indicators for interactions
        top_indicators = [item[0] for item in sorted_indicators[:5]]
        
        # Create interaction terms
        interactions_added = 0
        for i, ind1 in enumerate(top_indicators):
            for ind2 in top_indicators[i+1:]:
                if interactions_added >= max_interactions:
                    break
                
                interaction_name = f"{ind1}_{ind2}_interaction"
                result_df[interaction_name] = result_df[ind1] * result_df[ind2]
                interactions_added += 1
        
        return result_df
    
    def _ensemble_causal_analysis(self, analysis_data: pd.DataFrame, 
                                 indicator_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Perform ensemble causal analysis using multiple methods
        
        This approach combines multiple causal inference methods to obtain more
        robust estimates of causal effects.
        
        Args:
            analysis_data: Panel data DataFrame
            indicator_configs: List of indicator configurations
            
        Returns:
            Dictionary with causal analysis results
        """
        logger.info("Performing ensemble causal analysis")
        
        results = {}
        indicator_names = [ind['name'] for ind in indicator_configs]
        
        # Filter to only include configured indicators and their lags
        indicator_cols = []
        for ind_name in indicator_names:
            if ind_name in analysis_data.columns:
                indicator_cols.append(ind_name)
                # Add lag columns if they exist
                for lag in range(1, self.causal_analysis_params['lags'] + 1):
                    lag_col = f"{ind_name}_lag{lag}"
                    if lag_col in analysis_data.columns:
                        indicator_cols.append(lag_col)
        
        # Add interaction terms if they exist
        interaction_cols = [col for col in analysis_data.columns 
                          if '_interaction' in col and any(ind in col for ind in indicator_names)]
        feature_cols = indicator_cols + interaction_cols
        
        if not feature_cols:
            logger.warning("No indicator columns found in analysis data")
            return {}
        
        # Feature importance from tree-based models
        if self.causal_analysis_params['enable_feature_importance']:
            logger.info("Computing feature importance from ensemble models")
            importances = self._compute_feature_importance(analysis_data, feature_cols)
            self.feature_importances = importances
        else:
            importances = {}
        
        # Granger causality
        if self.causal_analysis_params['enable_granger']:
            logger.info("Testing Granger causality")
            granger_results = self._granger_causality_tests(analysis_data, indicator_names)
        else:
            granger_results = {}
        
        # Conditional independence
        if self.causal_analysis_params['enable_conditional']:
            logger.info("Testing conditional independence")
            conditional_results = self._conditional_independence_tests(analysis_data, indicator_names)
        else:
            conditional_results = {}
        
        # Regularized regression for feature selection
        logger.info("Performing regularized regression")
        regularization_results = self._regularized_regression(analysis_data, feature_cols)
        
        # Combine all analysis approaches
        for ind_name in indicator_names:
            if ind_name not in analysis_data.columns:
                logger.warning(f"Indicator {ind_name} not found in analysis data")
                continue
            
            # Initialize results for this indicator
            results[ind_name] = {}
            
            # Get base correlation
            correlation = analysis_data[ind_name].corr(analysis_data['market_value'])
            results[ind_name]['correlation'] = correlation if not np.isnan(correlation) else 0
            
            # Get feature importance
            importance = importances.get(ind_name, 0)
            results[ind_name]['importance'] = importance
            
            # Get Granger causality
            granger = granger_results.get(ind_name, {'p_value': 1.0, 'f_stat': 0.0})
            results[ind_name]['granger_p_value'] = granger['p_value']
            results[ind_name]['granger_f_stat'] = granger['f_stat']
            
            # Get conditional independence
            conditional = conditional_results.get(ind_name, {'p_value': 1.0, 'partial_corr': 0.0})
            results[ind_name]['conditional_p_value'] = conditional['p_value']
            results[ind_name]['partial_correlation'] = conditional['partial_corr']
            
            # Get regularized regression coefficient
            reg_coef = regularization_results.get(ind_name, 0)
            results[ind_name]['reg_coefficient'] = reg_coef
            
            # Calculate lagged effects if lags > 0
            if self.causal_analysis_params['lags'] > 0:
                lagged_effects = {}
                for lag in range(1, self.causal_analysis_params['lags'] + 1):
                    lag_col = f"{ind_name}_lag{lag}"
                    if lag_col in analysis_data.columns:
                        lag_coef = regularization_results.get(lag_col, 0)
                        lagged_effects[lag] = lag_coef
                results[ind_name]['lagged_effects'] = lagged_effects
                self.lagged_effects[ind_name] = lagged_effects
            
            # Calculate interaction effects
            interaction_effects = {}
            for col in interaction_cols:
                if ind_name in col:
                    int_coef = regularization_results.get(col, 0)
                    # Extract interacting indicator
                    parts = col.split('_interaction')[0].split('_')
                    other_ind = parts[1] if parts[0] == ind_name else parts[0]
                    interaction_effects[other_ind] = int_coef
            
            results[ind_name]['interaction_effects'] = interaction_effects
            
            # Compute composite causal strength
            # Weight different components
            causal_strength = 0.0
            
            # Base weight from correlation/regression
            corr_component = abs(correlation) * 0.3 if not np.isnan(correlation) else 0
            
            # Granger causality component (higher F-stat means stronger causality)
            granger_component = min(1.0, granger['f_stat'] / 10.0) * 0.2 if granger['p_value'] < 0.05 else 0
            
            # Conditional independence component
            cond_component = abs(conditional['partial_corr']) * 0.2 if conditional['p_value'] < 0.05 else 0
            
            # Feature importance component
            importance_component = importance * 0.2
            
            # Regularized regression component
            reg_component = abs(reg_coef) * 0.1
            
            # Combine all components
            causal_strength = corr_component + granger_component + cond_component + importance_component + reg_component
            
            # Scale to 0-1 range
            causal_strength = min(1.0, causal_strength)
            
            results[ind_name]['causal_strength'] = causal_strength
            
            logger.info(f"Indicator {ind_name}: Causal strength = {causal_strength:.4f}")
        
        return results
    
    def _compute_feature_importance(self, data: pd.DataFrame, 
                                   feature_cols: List[str]) -> Dict[str, float]:
        """
        Compute feature importance using tree-based models
        
        Args:
            data: Panel data DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Handle missing values
        X = data[feature_cols].copy()
        y = data['market_value'].copy()
        
        # Drop rows with missing values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            logger.warning("Not enough data for feature importance calculation")
            return {col: 0.0 for col in feature_cols}
        
        # Use Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importances = dict(zip(feature_cols, rf.feature_importances_))
        except Exception as e:
            logger.warning(f"Error in Random Forest: {str(e)}")
            rf_importances = {col: 0.0 for col in feature_cols}
        
        # Use Gradient Boosting
        try:
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X, y)
            gb_importances = dict(zip(feature_cols, gb.feature_importances_))
        except Exception as e:
            logger.warning(f"Error in Gradient Boosting: {str(e)}")
            gb_importances = {col: 0.0 for col in feature_cols}
        
        # Combine importances (average)
        combined_importances = {}
        for col in feature_cols:
            rf_imp = rf_importances.get(col, 0.0)
            gb_imp = gb_importances.get(col, 0.0)
            combined_importances[col] = (rf_imp + gb_imp) / 2.0
        
        # Normalize to sum to 1.0
        total_importance = sum(combined_importances.values())
        if total_importance > 0:
            combined_importances = {k: v/total_importance for k, v in combined_importances.items()}
        
        return combined_importances
    
    def _granger_causality_tests(self, data: pd.DataFrame, 
                               indicator_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Perform Granger causality tests between indicators and market value
        
        Args:
            data: Panel data DataFrame
            indicator_names: List of indicator names
            
        Returns:
            Dictionary mapping indicator names to Granger test results
        """
        # Need at least 2 time points per country for Granger causality
        if self.causal_analysis_params['lags'] < 1:
            logger.warning("Granger causality requires lags > 0")
            return {}
        
        results = {}
        
        # Test for each country separately, then aggregate
        id_col = self.config_manager.get_column_mapping('country_historical').get('id_column', 'idGeo')
        
        for ind_name in indicator_names:
            if ind_name not in data.columns:
                continue
            
            # Collect results for each country
            country_results = []
            
            for country_id in data[id_col].unique():
                country_mask = data[id_col] == country_id
                country_data = data.loc[country_mask].sort_values('Year')
                
                # Need enough data points
                if len(country_data) <= self.causal_analysis_params['lags'] + 2:
                    continue
                
                # Check for sufficient variance
                if np.var(country_data[ind_name]) < 1e-8 or np.var(country_data['market_value']) < 1e-8:
                    continue
                
                # Prepare data for Granger test
                y = country_data['market_value'].values
                x = country_data[ind_name].values
                
                # Perform Granger test
                try:
                    from statsmodels.tsa.stattools import grangercausalitytests
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gc_res = grangercausalitytests(
                            np.column_stack([y, x]), 
                            maxlag=self.causal_analysis_params['lags'], 
                            verbose=False
                        )
                    
                    # Extract results for the specified lag
                    lag_results = gc_res[self.causal_analysis_params['lags']][0]
                    f_stat = lag_results['ssr_ftest'][0]
                    p_value = lag_results['ssr_ftest'][1]
                    
                    country_results.append({
                        'country_id': country_id,
                        'f_stat': f_stat,
                        'p_value': p_value
                    })
                except Exception as e:
                    logger.warning(f"Error in Granger test for {country_id}, {ind_name}: {str(e)}")
            
            # Aggregate results across countries
            if country_results:
                # Use median p-value and f-statistic
                p_values = [res['p_value'] for res in country_results]
                f_stats = [res['f_stat'] for res in country_results]
                
                median_p = np.median(p_values)
                median_f = np.median(f_stats)
                
                # Calculate proportion of significant results
                sig_proportion = sum(1 for p in p_values if p < 0.05) / len(p_values)
                
                results[ind_name] = {
                    'p_value': median_p,
                    'f_stat': median_f,
                    'sig_proportion': sig_proportion,
                    'n_countries': len(country_results)
                }
            else:
                results[ind_name] = {
                    'p_value': 1.0,
                    'f_stat': 0.0,
                    'sig_proportion': 0.0,
                    'n_countries': 0
                }
        
        return results
    
    def _conditional_independence_tests(self, data: pd.DataFrame, 
                                      indicator_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Perform conditional independence tests between indicators and market value
        
        Args:
            data: Panel data DataFrame
            indicator_names: List of indicator names
            
        Returns:
            Dictionary mapping indicator names to conditional independence test results
        """
        results = {}
        
        # Get all indicators that are in the data
        valid_indicators = [ind for ind in indicator_names if ind in data.columns]
        
        if len(valid_indicators) < 2:
            logger.warning("Need at least 2 indicators for conditional independence tests")
            return {ind: {'p_value': 1.0, 'partial_corr': 0.0} for ind in valid_indicators}
        
        for target_ind in valid_indicators:
            # Conditioning set: all other indicators
            conditioning_set = [ind for ind in valid_indicators if ind != target_ind]
            
            # Prepare data
            X = data[conditioning_set + [target_ind]].copy()
            y = data['market_value'].copy()
            
            # Drop rows with missing values
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < len(conditioning_set) + 5:
                # Not enough data points relative to dimensionality
                results[target_ind] = {'p_value': 1.0, 'partial_corr': 0.0}
                continue
            
            try:
                # Calculate partial correlation
                from scipy.stats import pearsonr
                
                # First, regress out other indicators from target indicator
                X_others = X[conditioning_set]
                X_target = X[target_ind]
                
                # Add constant term
                X_others_const = np.column_stack([np.ones(len(X_others)), X_others])
                
                # Solve regression using least squares with proper error handling
                from scipy.linalg import lstsq
                try:
                    # Fixed: Add matrix conditioning check for numerical stability
                    condition_number = np.linalg.cond(X_others_const)
                    if condition_number > 1e12:
                        logger.warning(f"Poorly conditioned matrix (condition number: {condition_number:.2e})")
                        # Use pseudoinverse for better numerical stability
                        beta_target = np.linalg.pinv(X_others_const) @ X_target
                        residuals = None
                        rank = None
                        s = None
                    else:
                        beta_target, residuals, rank, s = lstsq(X_others_const, X_target)
                    
                    # Calculate residuals
                    target_residuals = X_target - X_others_const @ beta_target
                    
                    # Similarly, regress out other indicators from outcome
                    beta_outcome, residuals, rank, s = lstsq(X_others_const, y)
                    outcome_residuals = y - X_others_const @ beta_outcome
                    
                    # Check for invalid residuals
                    if np.any(np.isnan(target_residuals)) or np.any(np.isnan(outcome_residuals)):
                        raise ValueError("NaN values in residuals")
                    
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning(f"Linear algebra error in partial correlation for {target_ind}: {str(e)}")
                    results[target_ind] = {'p_value': 1.0, 'partial_corr': 0.0}
                    continue
                
                # Calculate correlation between residuals
                partial_corr, p_value = pearsonr(target_residuals, outcome_residuals)
                
                results[target_ind] = {
                    'p_value': p_value,
                    'partial_corr': partial_corr
                }
                
            except Exception as e:
                logger.warning(f"Error in conditional independence test for {target_ind}: {str(e)}")
                results[target_ind] = {'p_value': 1.0, 'partial_corr': 0.0}
        
        return results
    
    def _regularized_regression(self, data: pd.DataFrame, 
                              feature_cols: List[str]) -> Dict[str, float]:
        """
        Perform regularized regression to identify important features
        
        Args:
            data: Panel data DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Dictionary mapping feature names to regression coefficients
        """
        # Handle missing values
        X = data[feature_cols].copy()
        y = data['market_value'].copy()
        
        # Drop rows with missing values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            logger.warning("Not enough data for regularized regression")
            return {col: 0.0 for col in feature_cols}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Choose regularization method
        reg_method = self.causal_analysis_params['regularization']
        
        if reg_method == 'lasso':
            # LASSO regression
            model = LassoCV(cv=5, random_state=42)
        elif reg_method == 'elastic_net':
            # Elastic Net regression
            model = ElasticNetCV(cv=5, random_state=42)
        else:
            # OLS regression (using scikit-learn for consistency)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        
        # Fit model
        try:
            model.fit(X_scaled, y)
            
            # Get coefficients
            coefficients = model.coef_
            
            # Map coefficients to feature names
            coef_dict = dict(zip(feature_cols, coefficients))
            
            return coef_dict
            
        except Exception as e:
            logger.warning(f"Error in regularized regression: {str(e)}")
            return {col: 0.0 for col in feature_cols}
    
    def _granger_causality_analysis(self, analysis_data: pd.DataFrame, 
                                  indicator_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Perform Granger causality analysis
        
        This is a simplified version since we already implemented this in ensemble method.
        
        Args:
            analysis_data: Panel data DataFrame
            indicator_configs: List of indicator configurations
            
        Returns:
            Dictionary with Granger causality results
        """
        # Just call the granger tests from ensemble method
        indicator_names = [ind['name'] for ind in indicator_configs]
        granger_results = self._granger_causality_tests(analysis_data, indicator_names)
        
        # Format results
        results = {}
        for ind_name, granger in granger_results.items():
            # Calculate causal strength based on F-statistic and significance
            if granger['p_value'] < 0.05:
                # Normalize F-statistic to a 0-1 range (higher F = stronger causality)
                causal_strength = min(1.0, granger['f_stat'] / 10.0)
            else:
                causal_strength = 0.0
            
            results[ind_name] = {
                'p_value': granger['p_value'],
                'f_stat': granger['f_stat'],
                'causal_strength': causal_strength,
                'sig_proportion': granger.get('sig_proportion', 0.0),
                'n_countries': granger.get('n_countries', 0)
            }
        
        return results
    
    def _intervention_analysis(self, analysis_data: pd.DataFrame, 
                             indicator_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Perform intervention-based causal analysis
        
        This method uses changes in indicators as pseudo-interventions to estimate
        causal effects.
        
        Args:
            analysis_data: Panel data DataFrame
            indicator_configs: List of indicator configurations
            
        Returns:
            Dictionary with intervention analysis results
        """
        results = {}
        indicator_names = [ind['name'] for ind in indicator_configs]
        
        # Get ID column
        id_col = self.config_manager.get_column_mapping('country_historical').get('id_column', 'idGeo')
        
        for ind_name in indicator_names:
            if ind_name not in analysis_data.columns:
                continue
            
            # Find significant changes in the indicator (pseudo-interventions)
            interventions = []
            
            for country_id in analysis_data[id_col].unique():
                country_mask = analysis_data[id_col] == country_id
                country_data = analysis_data.loc[country_mask].sort_values('Year')
                
                if len(country_data) < 3:
                    continue
                
                # Calculate year-over-year changes
                country_data['ind_change'] = country_data[ind_name].pct_change()
                country_data['market_change'] = country_data['market_value'].pct_change()
                
                # Need at least 1 year pre and post intervention
                if len(country_data) < 3:
                    continue
                
                # Identify significant changes (top 33% of changes)
                threshold = np.nanpercentile(abs(country_data['ind_change']), 67)
                
                # Find intervention years
                for i in range(1, len(country_data) - 1):
                    if abs(country_data['ind_change'].iloc[i]) >= threshold:
                        # We found an intervention
                        
                        # Get pre-intervention year
                        pre_year = country_data['Year'].iloc[i-1]
                        int_year = country_data['Year'].iloc[i]
                        post_year = country_data['Year'].iloc[i+1]
                        
                        # Get market values
                        pre_market = country_data['market_value'].iloc[i-1]
                        int_market = country_data['market_value'].iloc[i]
                        post_market = country_data['market_value'].iloc[i+1]
                        
                        # Get indicator values
                        pre_ind = country_data[ind_name].iloc[i-1]
                        int_ind = country_data[ind_name].iloc[i]
                        post_ind = country_data[ind_name].iloc[i+1]
                        
                        # Calculate changes
                        ind_change = (int_ind - pre_ind) / pre_ind if pre_ind != 0 else 0
                        market_change_during = (int_market - pre_market) / pre_market if pre_market != 0 else 0
                        market_change_after = (post_market - int_market) / int_market if int_market != 0 else 0
                        
                        interventions.append({
                            'country_id': country_id,
                            'pre_year': pre_year,
                            'int_year': int_year,
                            'post_year': post_year,
                            'ind_change': ind_change,
                            'market_change_during': market_change_during,
                            'market_change_after': market_change_after
                        })
            
            # Analyze interventions
            if interventions:
                # Convert to DataFrame
                int_df = pd.DataFrame(interventions)
                
                # Calculate correlation between indicator change and market change
                during_corr = int_df['ind_change'].corr(int_df['market_change_during'])
                after_corr = int_df['ind_change'].corr(int_df['market_change_after'])
                
                # Calculate average treatment effect
                # Group by direction of indicator change
                int_df['ind_direction'] = np.sign(int_df['ind_change'])
                
                pos_vals = int_df[int_df['ind_direction'] > 0]['market_change_during']
                neg_vals = int_df[int_df['ind_direction'] < 0]['market_change_during']
                
                pos_effect = pos_vals.mean() if len(pos_vals) > 0 else 0.0
                neg_effect = neg_vals.mean() if len(neg_vals) > 0 else 0.0
                
                # Fixed: Calculate proper effect size using Cohen's d
                if len(pos_vals) > 0 and len(neg_vals) > 0:
                    # Calculate pooled standard deviation
                    pooled_std = np.sqrt(((len(pos_vals) - 1) * np.var(pos_vals, ddof=1) + 
                                         (len(neg_vals) - 1) * np.var(neg_vals, ddof=1)) / 
                                         (len(pos_vals) + len(neg_vals) - 2))
                    
                    # Cohen's d effect size
                    if pooled_std > 0:
                        effect_size = (pos_effect - neg_effect) / pooled_std
                    else:
                        effect_size = 0.0
                else:
                    # Fallback to simple difference
                    effect_size = (pos_effect - neg_effect) / 2
                
                # T-test between positive and negative intervention effects
                from scipy.stats import ttest_ind
                
                if len(pos_vals) > 0 and len(neg_vals) > 0:
                    t_stat, p_value = ttest_ind(pos_vals, neg_vals, equal_var=False)
                else:
                    t_stat, p_value = 0, 1.0
                
                # Calculate causal strength
                # Based on effect size and statistical significance
                if p_value < 0.05:
                    causal_strength = min(1.0, abs(effect_size))
                else:
                    # Weaker signal if not significant
                    causal_strength = min(0.3, abs(effect_size) * 0.5)
                
                results[ind_name] = {
                    'during_corr': during_corr,
                    'after_corr': after_corr,
                    'effect_size': effect_size,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'n_interventions': len(interventions),
                    'causal_strength': causal_strength
                }
            else:
                results[ind_name] = {
                    'during_corr': 0,
                    'after_corr': 0,
                    'effect_size': 0,
                    't_stat': 0,
                    'p_value': 1.0,
                    'n_interventions': 0,
                    'causal_strength': 0.0
                }
        
        return results
    
    def _structural_equation_analysis(self, analysis_data: pd.DataFrame, 
                                    indicator_configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Perform structural equation modeling for causal analysis
        
        This method uses structural equation modeling to estimate causal relationships.
        
        Args:
            analysis_data: Panel data DataFrame
            indicator_configs: List of indicator configurations
            
        Returns:
            Dictionary with structural equation analysis results
        """
        try:
            import statsmodels.api as sm
        except ImportError:
            logger.warning("statsmodels package required for structural equation modeling")
            return {}
        
        results = {}
        indicator_names = [ind['name'] for ind in indicator_configs]
        
        # Filter to include only configured indicators
        indicators_in_data = [ind for ind in indicator_names if ind in analysis_data.columns]
        
        if not indicators_in_data:
            logger.warning("No indicators found in analysis data")
            return {}
        
        # Prepare data for structural equation modeling
        X = analysis_data[indicators_in_data].copy()
        y = analysis_data['market_value'].copy()
        
        # Drop rows with missing values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Add constant term
        X = sm.add_constant(X)
        
        # Fit OLS model
        try:
            model = sm.OLS(y, X).fit()
            
            # Extract coefficients and p-values
            for i, ind_name in enumerate(indicators_in_data):
                # +1 because we added a constant term
                idx = i + 1
                coef = model.params[idx]
                p_value = model.pvalues[idx]
                std_error = model.bse[idx]
                t_stat = coef / std_error
                
                # Calculate standardized coefficient
                std_coef = coef * (X[ind_name].std() / y.std())
                
                # Calculate causal strength
                if p_value < 0.05:
                    causal_strength = min(1.0, abs(std_coef))
                else:
                    causal_strength = min(0.2, abs(std_coef) * 0.3)
                
                results[ind_name] = {
                    'coefficient': coef,
                    'std_coef': std_coef,
                    'p_value': p_value,
                    't_stat': t_stat,
                    'std_error': std_error,
                    'causal_strength': causal_strength
                }
            
        except Exception as e:
            logger.warning(f"Error in structural equation analysis: {str(e)}")
            # Return empty results
            for ind_name in indicators_in_data:
                results[ind_name] = {
                    'coefficient': 0,
                    'std_coef': 0,
                    'p_value': 1.0,
                    't_stat': 0,
                    'std_error': 0,
                    'causal_strength': 0.0
                }
        
        return results
    
    def _build_causal_graph(self, results: Dict[str, Dict[str, float]]) -> None:
        """
        Build a causal graph from analysis results
        
        Args:
            results: Dictionary with causal analysis results
        """
        # Create a new directed graph
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node('market_value', type='target')
        
        for ind_name, res in results.items():
            G.add_node(ind_name, type='indicator')
            
            # Add edge from indicator to market value
            causal_strength = res.get('causal_strength', 0)
            
            if causal_strength > 0:
                G.add_edge(ind_name, 'market_value', weight=causal_strength)
        
        # Add edges between indicators if interaction effects are available
        for ind_name, res in results.items():
            if 'interaction_effects' in res:
                for other_ind, effect in res['interaction_effects'].items():
                    if abs(effect) > 0.05 and other_ind in results:
                        # Add interaction edge
                        G.add_edge(ind_name, other_ind, weight=abs(effect), type='interaction')
        
        # Add lagged effects if available
        for ind_name, res in results.items():
            if 'lagged_effects' in res:
                for lag, effect in res['lagged_effects'].items():
                    if abs(effect) > 0.05:
                        lag_node = f"{ind_name}_lag{lag}"
                        G.add_node(lag_node, type='lag')
                        G.add_edge(ind_name, lag_node, weight=1.0, type='temporal')
                        G.add_edge(lag_node, 'market_value', weight=abs(effect))
        
        # Store the graph
        self.causal_graph = G
    
    def visualize_causal_graph(self) -> str:
        """
        Visualize the causal graph
        
        Returns:
            Path to the saved visualization file
        """
        if self.causal_graph is None or len(self.causal_graph.edges()) == 0:
            logger.warning("No causal graph available for visualization")
            return ""
        
        # Get output directory
        output_dir = self.config_manager.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'causal_graph.png')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Assign positions using a hierarchical layout
        pos = nx.spring_layout(self.causal_graph, seed=42)
        
        # Define node colors by type
        node_colors = []
        for node in self.causal_graph.nodes():
            node_type = self.causal_graph.nodes[node].get('type', '')
            if node_type == 'target':
                node_colors.append('red')
            elif node_type == 'indicator':
                node_colors.append('skyblue')
            elif node_type == 'lag':
                node_colors.append('lightgreen')
            else:
                node_colors.append('gray')
        
        # Get edge weights for thickness
        edge_weights = [self.causal_graph[u][v].get('weight', 0.1) * 3 for u, v in self.causal_graph.edges()]
        
        # Draw the graph
        nx.draw_networkx(
            self.causal_graph,
            pos=pos,
            with_labels=True,
            node_color=node_colors,
            node_size=1000,
            font_size=10,
            width=edge_weights,
            edge_color='gray',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='Market Value'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='Indicator'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Lagged Variable')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        plt.title('Causal Indicator Graph', size=16)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved causal graph visualization to {output_file}")
        return output_file
    
    def _generate_causal_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a comprehensive report of causal analysis results
        
        Args:
            results: Dictionary with causal analysis results
            
        Returns:
            Path to the saved report file
        """
        # Get output directory
        output_dir = self.config_manager.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Excel report
        excel_file = os.path.join(output_dir, 'causal_analysis_report.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                # Create summary DataFrame
                summary_data = []
                
                for ind_name, res in results.items():
                    row = {
                        'Indicator': ind_name,
                        'Causal Strength': res.get('causal_strength', 0),
                        'Correlation': res.get('correlation', 0),
                        'Importance': res.get('importance', 0),
                        'Coefficient': res.get('reg_coefficient', res.get('coefficient', 0)),
                        'P-value': res.get('p_value', 1.0)
                    }
                    
                    # Add Granger causality info if available
                    if 'granger_p_value' in res:
                        row['Granger P-value'] = res['granger_p_value']
                        row['Granger F-stat'] = res['granger_f_stat']
                    
                    summary_data.append(row)
                
                # Convert to DataFrame and sort by causal strength
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values('Causal Strength', ascending=False)
                
                # Write summary sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format the summary sheet
                workbook = writer.book
                summary_sheet = writer.sheets['Summary']
                
                # Add formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Apply header format
                for col_num, value in enumerate(summary_df.columns.values):
                    summary_sheet.write(0, col_num, value, header_format)
                
                # Set column widths
                summary_sheet.set_column('A:A', 20)  # Indicator
                summary_sheet.set_column('B:F', 12)  # Metrics
                
                # Add title
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 16,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                summary_sheet.merge_range('A1:F1', 'Causal Indicator Analysis Summary', title_format)
                summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=1)
                
                # Add detailed sheets for each analysis method
                self._add_detailed_analysis_sheets(writer, results)
                
                # Add lagged effects sheet if available
                if self.lagged_effects:
                    self._add_lagged_effects_sheet(writer)
                
                # Add interaction effects sheet if available
                interaction_effects = {}
                for ind_name, res in results.items():
                    if 'interaction_effects' in res and res['interaction_effects']:
                        interaction_effects[ind_name] = res['interaction_effects']
                
                if interaction_effects:
                    self._add_interaction_effects_sheet(writer, interaction_effects)
                
                # Add interpretation sheet
                self._add_interpretation_sheet(writer, summary_df)
                
            logger.info(f"Generated causal analysis report: {excel_file}")
            return excel_file
            
        except Exception as e:
            logger.error(f"Error generating causal report: {str(e)}")
            return ""
    
    def _add_detailed_analysis_sheets(self, writer, results: Dict[str, Dict[str, float]]) -> None:
        """
        Add detailed analysis sheets to the Excel report
        
        Args:
            writer: Excel writer object
            results: Dictionary with causal analysis results
        """
        workbook = writer.book
        
        # Format for headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Add Granger causality sheet if available
        granger_data = []
        for ind_name, res in results.items():
            if 'granger_p_value' in res:
                row = {
                    'Indicator': ind_name,
                    'P-value': res['granger_p_value'],
                    'F-statistic': res['granger_f_stat'],
                    'Significance': 'Yes' if res['granger_p_value'] < 0.05 else 'No'
                }
                
                if 'sig_proportion' in res:
                    row['Proportion Significant'] = res['sig_proportion']
                    row['Number of Countries'] = res['n_countries']
                
                granger_data.append(row)
        
        if granger_data:
            granger_df = pd.DataFrame(granger_data)
            granger_df = granger_df.sort_values('P-value')
            
            # Write to sheet
            granger_df.to_excel(writer, sheet_name='Granger Causality', index=False)
            
            # Format the sheet
            granger_sheet = writer.sheets['Granger Causality']
            
            # Apply header format
            for col_num, value in enumerate(granger_df.columns.values):
                granger_sheet.write(0, col_num, value, header_format)
            
            # Set column widths
            granger_sheet.set_column('A:A', 20)  # Indicator
            granger_sheet.set_column('B:F', 12)  # Metrics
        
        # Add conditional independence sheet if available
        conditional_data = []
        for ind_name, res in results.items():
            if 'conditional_p_value' in res:
                row = {
                    'Indicator': ind_name,
                    'P-value': res['conditional_p_value'],
                    'Partial Correlation': res['partial_correlation'],
                    'Significance': 'Yes' if res['conditional_p_value'] < 0.05 else 'No'
                }
                conditional_data.append(row)
        
        if conditional_data:
            conditional_df = pd.DataFrame(conditional_data)
            conditional_df = conditional_df.sort_values('P-value')
            
            # Write to sheet
            conditional_df.to_excel(writer, sheet_name='Conditional Independence', index=False)
            
            # Format the sheet
            conditional_sheet = writer.sheets['Conditional Independence']
            
            # Apply header format
            for col_num, value in enumerate(conditional_df.columns.values):
                conditional_sheet.write(0, col_num, value, header_format)
            
            # Set column widths
            conditional_sheet.set_column('A:A', 20)  # Indicator
            conditional_sheet.set_column('B:F', 15)  # Metrics
        
        # Add regression results sheet if available
        reg_data = []
        for ind_name, res in results.items():
            if 'reg_coefficient' in res:
                row = {
                    'Indicator': ind_name,
                    'Coefficient': res['reg_coefficient']
                }
                reg_data.append(row)
        
        if reg_data:
            reg_df = pd.DataFrame(reg_data)
            reg_df = reg_df.sort_values('Coefficient', key=abs, ascending=False)
            
            # Write to sheet
            reg_df.to_excel(writer, sheet_name='Regression Results', index=False)
            
            # Format the sheet
            reg_sheet = writer.sheets['Regression Results']
            
            # Apply header format
            for col_num, value in enumerate(reg_df.columns.values):
                reg_sheet.write(0, col_num, value, header_format)
            
            # Set column widths
            reg_sheet.set_column('A:A', 20)  # Indicator
            reg_sheet.set_column('B:B', 12)  # Coefficient
    
    def _add_lagged_effects_sheet(self, writer) -> None:
        """
        Add lagged effects sheet to the Excel report
        
        Args:
            writer: Excel writer object
        """
        workbook = writer.book
        
        # Format for headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Prepare lagged effects data
        lagged_data = []
        
        for ind_name, lagged_effects in self.lagged_effects.items():
            if not lagged_effects:
                continue
                
            for lag, effect in lagged_effects.items():
                row = {
                    'Indicator': ind_name,
                    'Lag': lag,
                    'Effect': effect
                }
                lagged_data.append(row)
        
        if lagged_data:
            lagged_df = pd.DataFrame(lagged_data)
            lagged_df = lagged_df.sort_values(['Indicator', 'Lag'])
            
            # Write to sheet
            lagged_df.to_excel(writer, sheet_name='Lagged Effects', index=False)
            
            # Format the sheet
            lagged_sheet = writer.sheets['Lagged Effects']
            
            # Apply header format
            for col_num, value in enumerate(lagged_df.columns.values):
                lagged_sheet.write(0, col_num, value, header_format)
            
            # Set column widths
            lagged_sheet.set_column('A:A', 20)  # Indicator
            lagged_sheet.set_column('B:C', 10)  # Lag and Effect
    
    def _add_interaction_effects_sheet(self, writer, interaction_effects: Dict[str, Dict[str, float]]) -> None:
        """
        Add interaction effects sheet to the Excel report
        
        Args:
            writer: Excel writer object
            interaction_effects: Dictionary with interaction effects
        """
        workbook = writer.book
        
        # Format for headers
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Prepare interaction effects data
        interaction_data = []
        
        for ind1, effects in interaction_effects.items():
            for ind2, effect in effects.items():
                row = {
                    'Indicator 1': ind1,
                    'Indicator 2': ind2,
                    'Interaction Effect': effect
                }
                interaction_data.append(row)
        
        if interaction_data:
            interaction_df = pd.DataFrame(interaction_data)
            interaction_df = interaction_df.sort_values('Interaction Effect', key=abs, ascending=False)
            
            # Write to sheet
            interaction_df.to_excel(writer, sheet_name='Interaction Effects', index=False)
            
            # Format the sheet
            interaction_sheet = writer.sheets['Interaction Effects']
            
            # Apply header format
            for col_num, value in enumerate(interaction_df.columns.values):
                interaction_sheet.write(0, col_num, value, header_format)
            
            # Set column widths
            interaction_sheet.set_column('A:B', 20)  # Indicators
            interaction_sheet.set_column('C:C', 15)  # Effect
    
    def _add_interpretation_sheet(self, writer, summary_df: pd.DataFrame) -> None:
        """
        Add interpretation sheet to the Excel report
        
        Args:
            writer: Excel writer object
            summary_df: DataFrame with summary results
        """
        workbook = writer.book
        interpretation_sheet = workbook.add_worksheet('Interpretation')
        
        # Title format
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 16,
            'align': 'center'
        })
        
        # Heading format
        heading_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'underline': True
        })
        
        # Normal text format
        text_format = workbook.add_format({
            'font_size': 11,
            'text_wrap': True
        })
        
        # Add title
        interpretation_sheet.merge_range('A1:G1', 'Interpreting Causal Analysis Results', title_format)
        
        # Add explanation
        row = 2
        interpretation_sheet.write(row, 0, 'Overview:', heading_format)
        row += 1
        interpretation_sheet.merge_range(f'A{row+1}:G{row+3}', 
            'This analysis identifies and quantifies causal relationships between indicators and market outcomes. '
            'Unlike simple correlation, causal analysis attempts to determine whether changes in an indicator '
            'actually cause changes in market values, not just that they happen to move together.', text_format)
        row += 4
        
        # Add key metrics explanation
        interpretation_sheet.write(row, 0, 'Key Metrics:', heading_format)
        row += 1
        
        metrics = [
            ('Causal Strength', 'A composite measure (0-1) of how strongly an indicator causally influences market values. '
                               'Higher values indicate stronger causal relationships.'),
            ('Correlation', 'The Pearson correlation coefficient between the indicator and market values. '
                          'Ranges from -1 to 1, with 0 indicating no correlation.'),
            ('Importance', 'Feature importance from tree-based models, indicating how useful the indicator is for predicting market values.'),
            ('Granger P-value', 'Tests whether past values of the indicator help predict future market values. '
                              'P-values < 0.05 suggest Granger causality.'),
            ('Granger F-stat', 'The F-statistic from Granger causality tests. Higher values indicate stronger causality.'),
            ('Partial Correlation', 'Correlation between an indicator and market values after controlling for other indicators. '
                                  'Helps identify direct causal effects.')
        ]
        
        for metric, description in metrics:
            interpretation_sheet.write(row, 0, metric, workbook.add_format({'bold': True}))
            interpretation_sheet.merge_range(f'B{row+1}:G{row+1}', description, text_format)
            row += 1
        
        row += 1
        
        # Add top indicators
        interpretation_sheet.write(row, 0, 'Top Causal Indicators:', heading_format)
        row += 1
        
        # Get top 5 indicators by causal strength
        top_indicators = summary_df.head(5)
        
        for _, ind_row in top_indicators.iterrows():
            ind_name = ind_row['Indicator']
            strength = ind_row['Causal Strength']
            
            interpretation_sheet.write(row, 0, ind_name, workbook.add_format({'bold': True}))
            interpretation_sheet.write(row, 1, f'Causal Strength: {strength:.4f}', text_format)
            row += 1
        
        row += 1
        
        # Add lagged effects explanation if available
        if self.lagged_effects:
            interpretation_sheet.write(row, 0, 'Lagged Effects:', heading_format)
            row += 1
            interpretation_sheet.merge_range(f'A{row+1}:G{row+2}', 
                'Lagged effects show how past values of indicators affect current market values. '
                'This helps understand the time delay between changes in indicators and their impact on the market.', text_format)
            row += 3
        
        # Add interaction effects explanation
        for ind_name, res in summary_df.iterrows():
            if 'interaction_effects' in res and res['interaction_effects']:
                interpretation_sheet.write(row, 0, 'Interaction Effects:', heading_format)
                row += 1
                interpretation_sheet.merge_range(f'A{row+1}:G{row+2}', 
                    'Interaction effects show how indicators work together to influence market values. '
                    'A significant interaction means the effect of one indicator depends on the value of another.', text_format)
                row += 3
                break
        
        # Set column widths
        interpretation_sheet.set_column('A:A', 20)
        interpretation_sheet.set_column('B:G', 15)
    
    def apply_causal_adjustments(self, country_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply causal-based adjustments to country market shares
        
        This method uses identified causal relationships to adjust market shares,
        giving more weight to causally significant indicators.
        
        Args:
            country_df: DataFrame with country market shares
            
        Returns:
            DataFrame with causally adjusted market shares
        """
        # Check if we have causal strengths
        if not self.causal_strengths:
            self.analyze_causal_relationships()
        
        if not self.causal_strengths:
            logger.warning("No causal strengths available, skipping causal adjustments")
            return country_df
        
        # Get existing indicator weights
        indicator_weights = self.indicator_analyzer.get_indicator_weights()
        
        # Update weights based on causal analysis
        causal_weights = {}
        
        # Combine current weights with causal strengths
        for ind_name, weight in indicator_weights.items():
            causal_strength = self.causal_strengths.get(ind_name, 0)
            
            # Calculate new weight as weighted average of correlation and causal strength
            causal_weights[ind_name] = 0.4 * weight + 0.6 * causal_strength
        
        # Normalize weights to sum to 1.0
        total_weight = sum(causal_weights.values())
        if total_weight > 0:
            causal_weights = {k: v/total_weight for k, v in causal_weights.items()}
        
        # Log weight changes
        logger.info("Updated indicator weights based on causal analysis:")
        for ind_name in sorted(causal_weights.keys()):
            old_weight = indicator_weights.get(ind_name, 0)
            new_weight = causal_weights[ind_name]
            logger.info(f"{ind_name}: {old_weight:.4f} -> {new_weight:.4f}")
        
        # Temporarily update indicator weights in the analyzer
        original_weights = self.indicator_analyzer.indicator_weights.copy()
        self.indicator_analyzer.indicator_weights = causal_weights
        
        # Apply the adjustments using the indicator analyzer's method
        adjusted_df = self.indicator_analyzer.apply_indicator_adjustments(country_df)
        
        # Restore original weights
        self.indicator_analyzer.indicator_weights = original_weights
        
        return adjusted_df
    
    def get_causal_strengths(self) -> Dict[str, float]:
        """
        Get calculated causal strengths
        
        Returns:
            Dictionary mapping indicator names to causal strengths
        """
        if not self.causal_strengths:
            logger.warning("No causal strengths available, call analyze_causal_relationships() first")
        
        return self.causal_strengths
    
    def save_causal_analysis(self) -> str:
        """
        Save causal analysis results to an Excel file
        
        Returns:
            Path to the saved Excel file
        """
        if not self.causal_strengths:
            logger.warning("No causal analysis available to save")
            return ""
        
        # Get output directory
        output_dir = self.config_manager.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Excel file
        output_file = os.path.join(output_dir, 'causal_indicator_analysis.xlsx')
        
        try:
            # Create DataFrame for causal strengths
            data = []
            
            for ind_name, strength in self.causal_strengths.items():
                row = {
                    'Indicator': ind_name,
                    'Causal Strength': strength
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df = df.sort_values('Causal Strength', ascending=False)
            
            # Save to Excel
            df.to_excel(output_file, index=False)
            
            logger.info(f"Saved causal analysis to: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving causal analysis: {str(e)}")
            return ""