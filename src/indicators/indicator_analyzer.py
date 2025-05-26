"""
Enhanced Indicator Analyzer Module - Advanced indicator evaluation with state-of-the-art methods

This module provides sophisticated functionality to analyze, evaluate, and weight indicators
using advanced statistical, machine learning, and causal inference techniques.
Features include dynamic time warping, SHAP values, information criteria weighting,
multicollinearity handling, and robust correlation analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV, BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.signal import correlate
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedCorrelationAnalyzer:
    """Advanced correlation analysis methods beyond simple Pearson correlation"""
    
    @staticmethod
    def dynamic_time_warping_correlation(x: np.ndarray, y: np.ndarray, window: int = None) -> float:
        """
        Calculate correlation using Dynamic Time Warping for phase-shifted relationships
        
        Args:
            x, y: Time series arrays
            window: Constraint window for DTW (None for unconstrained)
            
        Returns:
            DTW-based correlation coefficient
        """
        try:
            # Normalize series to [0,1] for DTW
            x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
            
            # Simple DTW implementation
            n, m = len(x_norm), len(y_norm)
            dtw_matrix = np.full((n, m), np.inf)
            dtw_matrix[0, 0] = abs(x_norm[0] - y_norm[0])
            
            # Fill DTW matrix
            for i in range(1, n):
                for j in range(max(1, i - window if window else 1), 
                             min(m, i + window + 1 if window else m)):
                    cost = abs(x_norm[i] - y_norm[j])
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],     # insertion
                        dtw_matrix[i, j-1],     # deletion
                        dtw_matrix[i-1, j-1]    # match
                    )
            
            # Convert DTW distance to correlation-like measure
            max_distance = n + m  # theoretical maximum
            dtw_distance = dtw_matrix[n-1, m-1]
            correlation = 1 - (dtw_distance / max_distance)
            
            return max(-1, min(1, correlation))
            
        except Exception as e:
            logger.warning(f"DTW correlation failed: {str(e)}")
            return 0.0
    
    @staticmethod
    def time_lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> Dict[int, float]:
        """
        Calculate correlations at different time lags
        
        Args:
            x, y: Time series arrays
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary mapping lag to correlation coefficient
        """
        correlations = {}
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr, _ = pearsonr(x, y)
            elif lag > 0:
                # y leads x by lag periods
                if len(x) > lag:
                    corr, _ = pearsonr(x[:-lag], y[lag:])
                else:
                    corr = 0.0
            else:  # lag < 0
                # x leads y by |lag| periods
                if len(y) > abs(lag):
                    corr, _ = pearsonr(x[abs(lag):], y[:lag])
                else:
                    corr = 0.0
            
            correlations[lag] = corr if not np.isnan(corr) else 0.0
        
        return correlations
    
    @staticmethod
    def rolling_correlation(x: pd.Series, y: pd.Series, window: int = 12) -> pd.Series:
        """
        Calculate rolling correlation to detect time-varying relationships
        
        Args:
            x, y: Time series
            window: Rolling window size
            
        Returns:
            Series of rolling correlations
        """
        return x.rolling(window=window).corr(y)

class SophisticatedWeighting:
    """Advanced weighting schemes for indicators"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def information_criteria_weights(self, X: pd.DataFrame, y: pd.Series, 
                                   criterion: str = 'aic') -> Dict[str, float]:
        """
        Calculate weights based on information criteria (AIC/BIC)
        
        Args:
            X: Feature matrix
            y: Target variable
            criterion: 'aic' or 'bic'
            
        Returns:
            Dictionary mapping feature names to IC-based weights
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        weights = {}
        n_samples = len(y)
        
        for col in X.columns:
            try:
                # Fit simple linear regression
                X_col = X[[col]].values
                model = LinearRegression()
                model.fit(X_col, y)
                y_pred = model.predict(X_col)
                
                # Calculate MSE
                mse = mean_squared_error(y, y_pred)
                
                # Calculate log-likelihood (assuming normal errors)
                log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * mse) - (n_samples / 2)
                
                # Calculate AIC or BIC
                k = 2  # number of parameters (intercept + slope)
                if criterion == 'aic':
                    ic = 2 * k - 2 * log_likelihood
                elif criterion == 'bic':
                    ic = k * np.log(n_samples) - 2 * log_likelihood
                else:
                    raise ValueError("Criterion must be 'aic' or 'bic'")
                
                # Convert to weight (lower IC = higher weight)
                weights[col] = 1 / (1 + ic)
                
            except Exception as e:
                logger.warning(f"IC calculation failed for {col}: {str(e)}")
                weights[col] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def shap_based_weights(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate weights using SHAP (SHapley Additive exPlanations) values
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary mapping feature names to SHAP-based weights
        """
        try:
            import shap
            
            # Use a simple model for SHAP analysis
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP values for each feature
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create weights dictionary
            weights = dict(zip(X.columns, mean_shap))
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            return weights
            
        except ImportError:
            logger.warning("SHAP package not available, falling back to feature importance")
            return self._feature_importance_weights(X, y)
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {str(e)}")
            return self._feature_importance_weights(X, y)
    
    def _feature_importance_weights(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Fallback to standard feature importance"""
        model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        model.fit(X, y)
        
        weights = dict(zip(X.columns, model.feature_importances_))
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def meta_learner_weights(self, X: pd.DataFrame, y: pd.Series, 
                           context_features: pd.DataFrame = None) -> Dict[str, float]:
        """
        Use a meta-learner to predict optimal weights based on context
        
        Args:
            X: Feature matrix
            y: Target variable
            context_features: Additional context (market conditions, etc.)
            
        Returns:
            Dictionary mapping feature names to meta-learned weights
        """
        # For now, implement a simple heuristic-based approach
        # In practice, this would be trained on historical performance data
        
        weights = {}
        
        for col in X.columns:
            try:
                # Calculate multiple performance metrics
                corr = X[col].corr(y)
                
                # Mutual information
                mi = mutual_info_regression(X[[col]], y)[0]
                
                # Stability (inverse of coefficient of variation)
                stability = 1 / (1 + X[col].std() / (abs(X[col].mean()) + 1e-8))
                
                # Combine metrics
                weight = abs(corr) * 0.4 + mi * 0.3 + stability * 0.3
                weights[col] = weight
                
            except Exception as e:
                logger.warning(f"Meta-learner weight calculation failed for {col}: {str(e)}")
                weights[col] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights

class FeatureEngineering:
    """Advanced feature engineering for indicators"""
    
    @staticmethod
    def ensure_stationarity(series: pd.Series, method: str = 'diff') -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Make time series stationary
        
        Args:
            series: Time series to make stationary
            method: 'diff', 'log_diff', or 'detrend'
            
        Returns:
            Tuple of (stationary_series, transformation_info)
        """
        info = {'method': method, 'transformations': []}
        
        if method == 'diff':
            # First difference
            stationary = series.diff().dropna()
            info['transformations'].append('first_difference')
            
            # Test stationarity
            if len(stationary) > 10:
                adf_stat, adf_p = adfuller(stationary.dropna())[:2]
                if adf_p > 0.05:
                    # Second difference if needed
                    stationary = stationary.diff().dropna()
                    info['transformations'].append('second_difference')
        
        elif method == 'log_diff':
            # Log transformation then difference
            if (series > 0).all():
                log_series = np.log(series)
                stationary = log_series.diff().dropna()
                info['transformations'].extend(['log', 'first_difference'])
            else:
                # Fallback to regular differencing
                stationary = series.diff().dropna()
                info['transformations'].append('first_difference')
        
        elif method == 'detrend':
            # Remove linear trend
            from scipy import signal
            detrended = signal.detrend(series.values)
            stationary = pd.Series(detrended, index=series.index)
            info['transformations'].append('detrend')
        
        else:
            stationary = series
        
        return stationary, info
    
    @staticmethod
    def create_advanced_features(df: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
        """
        Create advanced features from indicators
        
        Args:
            df: DataFrame with indicators
            indicator_cols: List of indicator column names
            
        Returns:
            DataFrame with additional engineered features
        """
        result_df = df.copy()
        
        for col in indicator_cols:
            if col not in df.columns:
                continue
                
            # Moving averages
            for window in [3, 6, 12]:
                if len(df) >= window:
                    result_df[f"{col}_ma{window}"] = df[col].rolling(window=window).mean()
            
            # Volatility (rolling standard deviation)
            for window in [6, 12]:
                if len(df) >= window:
                    result_df[f"{col}_vol{window}"] = df[col].rolling(window=window).std()
            
            # Acceleration (second derivative)
            first_diff = df[col].diff()
            result_df[f"{col}_acceleration"] = first_diff.diff()
            
            # Momentum (rate of change)
            for period in [3, 6]:
                if len(df) > period:
                    result_df[f"{col}_momentum{period}"] = df[col].pct_change(periods=period)
            
            # Z-score (standardized values)
            result_df[f"{col}_zscore"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
            
            # Quantile rank
            result_df[f"{col}_rank"] = df[col].rank(pct=True)
        
        return result_df
    
    @staticmethod
    def handle_multicollinearity(X: pd.DataFrame, threshold: float = 10.0) -> Tuple[pd.DataFrame, List[str]]:
        """
        Handle multicollinearity using Variance Inflation Factor
        
        Args:
            X: Feature matrix
            threshold: VIF threshold (typically 5-10)
            
        Returns:
            Tuple of (reduced_features, removed_features)
        """
        removed_features = []
        remaining_features = list(X.columns)
        
        while True:
            # Calculate VIF for remaining features
            X_subset = X[remaining_features]
            
            if len(remaining_features) <= 1:
                break
            
            try:
                vif_data = pd.DataFrame()
                vif_data["Feature"] = remaining_features
                vif_data["VIF"] = [variance_inflation_factor(X_subset.values, i) 
                                  for i in range(len(remaining_features))]
                
                # Find highest VIF
                max_vif_idx = vif_data["VIF"].idxmax()
                max_vif = vif_data.loc[max_vif_idx, "VIF"]
                
                if max_vif > threshold:
                    # Remove feature with highest VIF
                    feature_to_remove = vif_data.loc[max_vif_idx, "Feature"]
                    remaining_features.remove(feature_to_remove)
                    removed_features.append(feature_to_remove)
                    logger.info(f"Removed {feature_to_remove} due to high VIF: {max_vif:.2f}")
                else:
                    break
                    
            except Exception as e:
                logger.warning(f"VIF calculation failed: {str(e)}")
                break
        
        return X[remaining_features], removed_features

class IndicatorAnalyzer:
    """
    Enhanced indicator analyzer with advanced statistical and ML techniques
    """
    
    def __init__(self, config_manager, data_loader):
        """
        Initialize the EnhancedIndicatorAnalyzer
        
        Args:
            config_manager: Configuration manager instance
            data_loader: Data loader instance
        """
        self.config_manager = config_manager
        self.data_loader = data_loader
        self.indicator_weights = {}
        self.indicator_correlations = {}
        self.indicator_scores = {}
        
        # Advanced analysis components
        self.correlation_analyzer = AdvancedCorrelationAnalyzer()
        self.weighting_engine = SophisticatedWeighting()
        self.feature_engineer = FeatureEngineering()
        
        # Enhanced parameters
        self.analysis_params = {
            'correlation_methods': ['pearson', 'spearman', 'kendall', 'dtw', 'lagged'],
            'weighting_method': 'ensemble',  # 'shap', 'ic', 'meta_learner', 'ensemble'
            'feature_engineering': True,
            'handle_multicollinearity': True,
            'multicollinearity_threshold': 10.0,
            'stationarity_test': True,
            'rolling_analysis': True,
            'bootstrap_samples': 100,
            'cross_validation_folds': 5
        }
        
        # Load parameters from config if available
        self._load_config_params()
        
        # Storage for advanced analysis results
        self.correlation_results = {}
        self.feature_importance_results = {}
        self.robustness_results = {}
        self.temporal_analysis_results = {}
    
    def _load_config_params(self) -> None:
        """Load enhanced analysis parameters from configuration"""
        enhanced_config = self.config_manager.get_value('indicators.enhanced_analysis', {})
        
        if enhanced_config:
            for key, value in enhanced_config.items():
                if key in self.analysis_params:
                    self.analysis_params[key] = value
                    
            logger.info("Loaded enhanced analysis parameters from configuration")
    
    def analyze_indicators_advanced(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform advanced indicator analysis using multiple sophisticated methods
        
        Returns:
            Comprehensive dictionary with analysis results
        """
        logger.info("🔬 Starting advanced indicator analysis")
        
        # Load data
        country_data = self.data_loader.load_country_historical()
        all_indicators = self.data_loader.load_all_indicators()
        
        if all_indicators.empty:
            logger.warning("No indicator data available for analysis")
            return {}
        
        # Get configuration
        indicator_configs = self.config_manager.get_indicators()
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        id_col = column_mapping.get('id_column', 'idGeo')
        
        # Prepare panel data for analysis
        analysis_data = self._prepare_advanced_panel_data(
            country_data, all_indicators, indicator_configs, id_col
        )
        
        if analysis_data.empty:
            logger.warning("No valid analysis data after preprocessing")
            return {}
        
        # Extract indicator columns
        indicator_names = [ind['name'] for ind in indicator_configs]
        indicator_cols = [col for col in analysis_data.columns 
                         if any(ind_name in col for ind_name in indicator_names)]
        
        if not indicator_cols:
            logger.warning("No indicator columns found in analysis data")
            return {}
        
        # Feature engineering if enabled
        if self.analysis_params['feature_engineering']:
            logger.info("🔧 Performing feature engineering")
            analysis_data = self.feature_engineer.create_advanced_features(
                analysis_data, indicator_cols
            )
            # Update indicator columns list
            indicator_cols = [col for col in analysis_data.columns 
                             if any(ind_name in col for ind_name in indicator_names)]
        
        # Handle multicollinearity if enabled
        if self.analysis_params['handle_multicollinearity']:
            logger.info("📊 Handling multicollinearity")
            feature_matrix = analysis_data[indicator_cols].select_dtypes(include=[np.number])
            if not feature_matrix.empty:
                reduced_features, removed_features = self.feature_engineer.handle_multicollinearity(
                    feature_matrix, self.analysis_params['multicollinearity_threshold']
                )
                indicator_cols = list(reduced_features.columns)
                logger.info(f"Removed {len(removed_features)} features due to multicollinearity")
        
        # Perform comprehensive correlation analysis
        logger.info("🔍 Performing advanced correlation analysis")
        correlation_results = self._advanced_correlation_analysis(
            analysis_data, indicator_cols
        )
        
        # Calculate sophisticated weights
        logger.info("⚖️ Calculating sophisticated weights")
        weight_results = self._calculate_sophisticated_weights(
            analysis_data, indicator_cols
        )
        
        # Temporal analysis
        logger.info("📈 Performing temporal analysis")
        temporal_results = self._temporal_analysis(
            analysis_data, indicator_cols
        )
        
        # Robustness analysis
        logger.info("🛡️ Performing robustness analysis")
        robustness_results = self._robustness_analysis(
            analysis_data, indicator_cols
        )
        
        # Combine all results
        final_results = {}
        
        for ind_name in indicator_names:
            # Find columns related to this indicator
            related_cols = [col for col in indicator_cols if ind_name in col]
            
            if not related_cols:
                continue
            
            # Use the main indicator column (without suffixes) as primary
            main_col = ind_name if ind_name in indicator_cols else related_cols[0]
            
            final_results[ind_name] = {
                'correlation': correlation_results.get(main_col, {}),
                'weights': weight_results.get(main_col, {}),
                'temporal': temporal_results.get(main_col, {}),
                'robustness': robustness_results.get(main_col, {}),
                'related_features': related_cols,
                'final_weight': weight_results.get(main_col, {}).get('ensemble_weight', 0.0),
                'confidence_score': robustness_results.get(main_col, {}).get('confidence', 0.0)
            }
        
        # Store results
        self.correlation_results = correlation_results
        self.feature_importance_results = weight_results
        self.temporal_analysis_results = temporal_results
        self.robustness_results = robustness_results
        
        # Update indicator weights
        self.indicator_weights = {
            ind_name: results['final_weight'] 
            for ind_name, results in final_results.items()
        }
        
        # Generate comprehensive visualizations
        self._generate_advanced_visualizations()
        
        # Generate detailed report
        self._generate_comprehensive_report(final_results)
        
        logger.info("✅ Advanced indicator analysis completed")
        return final_results
    
    def _prepare_advanced_panel_data(self, country_data: pd.DataFrame, 
                                   indicators_data: pd.DataFrame, 
                                   indicator_configs: List[Dict[str, Any]], 
                                   id_col: str) -> pd.DataFrame:
        """
        Prepare advanced panel data with proper handling and preprocessing
        """
        # Get common years
        country_years = set(country_data['Year'].unique())
        indicator_years = set(indicators_data['Year'].unique())
        common_years = sorted(country_years.intersection(indicator_years))
        
        if not common_years:
            logger.warning("No common years found")
            return pd.DataFrame()
        
        # Filter for common years
        country_filtered = country_data[country_data['Year'].isin(common_years)].copy()
        
        # Start with country data
        panel_data = country_filtered[[id_col, 'Year', 'Value']].copy()
        panel_data.rename(columns={'Value': 'market_value'}, inplace=True)
        
        # Add each indicator
        for indicator_config in indicator_configs:
            indicator_name = indicator_config.get('name')
            indicator_type = indicator_config.get('type', 'continuous')
            
            indicator_df = indicators_data[indicators_data['Indicator'] == indicator_name]
            
            if indicator_df.empty:
                continue
            
            # Handle rank indicators (no year filtering)
            if indicator_type == 'rank':
                merge_cols = [id_col]
                merge_data = indicator_df[[id_col, 'Value']].copy()
            else:
                merge_cols = [id_col, 'Year']
                indicator_df = indicator_df[indicator_df['Year'].isin(common_years)]
                merge_data = indicator_df[[id_col, 'Year', 'Value']].copy()
            
            if merge_data.empty:
                continue
            
            # Merge with panel data
            panel_data = pd.merge(
                panel_data,
                merge_data,
                on=merge_cols,
                how='left',
                suffixes=('', f'_{indicator_name}')
            )
            
            # Rename indicator column
            panel_data.rename(columns={'Value': indicator_name}, inplace=True)
        
        # Handle missing values
        indicator_cols = [col for col in panel_data.columns 
                         if col not in [id_col, 'Year', 'market_value']]
        
        # Forward fill and backward fill by country
        for country_id in panel_data[id_col].unique():
            country_mask = panel_data[id_col] == country_id
            for col in indicator_cols:
                panel_data.loc[country_mask, col] = (
                    panel_data.loc[country_mask, col].ffill().bfill()
                )
        
        # Remove rows with missing market values
        panel_data = panel_data.dropna(subset=['market_value'])
        
        return panel_data
    
    def _advanced_correlation_analysis(self, data: pd.DataFrame, 
                                     indicator_cols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Perform advanced correlation analysis using multiple methods
        """
        results = {}
        
        for col in indicator_cols:
            if col not in data.columns:
                continue
            
            col_results = {}
            
            # Get valid data
            valid_mask = ~(data[col].isna() | data['market_value'].isna())
            x = data.loc[valid_mask, col].values
            y = data.loc[valid_mask, 'market_value'].values
            
            if len(x) < 10:
                results[col] = {'pearson': 0.0, 'error': 'insufficient_data'}
                continue
            
            # Traditional correlations
            try:
                pearson_r, pearson_p = pearsonr(x, y)
                col_results['pearson'] = pearson_r if not np.isnan(pearson_r) else 0.0
                col_results['pearson_p'] = pearson_p if not np.isnan(pearson_p) else 1.0
            except:
                col_results['pearson'] = 0.0
                col_results['pearson_p'] = 1.0
            
            try:
                spearman_r, spearman_p = spearmanr(x, y)
                col_results['spearman'] = spearman_r if not np.isnan(spearman_r) else 0.0
                col_results['spearman_p'] = spearman_p if not np.isnan(spearman_p) else 1.0
            except:
                col_results['spearman'] = 0.0
                col_results['spearman_p'] = 1.0
            
            try:
                kendall_r, kendall_p = kendalltau(x, y)
                col_results['kendall'] = kendall_r if not np.isnan(kendall_r) else 0.0
                col_results['kendall_p'] = kendall_p if not np.isnan(kendall_p) else 1.0
            except:
                col_results['kendall'] = 0.0
                col_results['kendall_p'] = 1.0
            
            # DTW correlation for time series
            if 'dtw' in self.analysis_params['correlation_methods']:
                dtw_corr = self.correlation_analyzer.dynamic_time_warping_correlation(x, y)
                col_results['dtw'] = dtw_corr
            
            # Time-lagged correlations
            if 'lagged' in self.analysis_params['correlation_methods']:
                lagged_corrs = self.correlation_analyzer.time_lagged_correlation(x, y)
                col_results['lagged_correlations'] = lagged_corrs
                col_results['max_lagged_corr'] = max(lagged_corrs.values(), key=abs)
                col_results['optimal_lag'] = max(lagged_corrs.items(), key=lambda x: abs(x[1]))[0]
            
            # Rolling correlation if enough data
            if len(x) > 24:  # At least 2 years of monthly data
                x_series = pd.Series(x)
                y_series = pd.Series(y)
                rolling_corr = self.correlation_analyzer.rolling_correlation(x_series, y_series, window=12)
                col_results['rolling_corr_mean'] = rolling_corr.mean() if not rolling_corr.empty else 0.0
                col_results['rolling_corr_std'] = rolling_corr.std() if not rolling_corr.empty else 0.0
            
            results[col] = col_results
        
        return results
    
    def _calculate_sophisticated_weights(self, data: pd.DataFrame, 
                                       indicator_cols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate weights using sophisticated methods
        """
        results = {}
        
        # Prepare feature matrix
        X = data[indicator_cols].select_dtypes(include=[np.number])
        y = data['market_value']
        
        # Remove rows with missing values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 10:
            logger.warning("Insufficient data for weight calculation")
            return {col: {'ensemble_weight': 0.0} for col in indicator_cols}
        
        # Method 1: Information Criteria weights
        ic_weights = self.weighting_engine.information_criteria_weights(X_clean, y_clean, 'aic')
        
        # Method 2: SHAP-based weights
        shap_weights = self.weighting_engine.shap_based_weights(X_clean, y_clean)
        
        # Method 3: Meta-learner weights
        meta_weights = self.weighting_engine.meta_learner_weights(X_clean, y_clean)
        
        # Combine weights based on method
        for col in indicator_cols:
            col_results = {}
            
            col_results['ic_weight'] = ic_weights.get(col, 0.0)
            col_results['shap_weight'] = shap_weights.get(col, 0.0)
            col_results['meta_weight'] = meta_weights.get(col, 0.0)
            
            # Ensemble weight (weighted average)
            if self.analysis_params['weighting_method'] == 'ensemble':
                ensemble_weight = (
                    0.3 * col_results['ic_weight'] +
                    0.4 * col_results['shap_weight'] +
                    0.3 * col_results['meta_weight']
                )
            elif self.analysis_params['weighting_method'] == 'shap':
                ensemble_weight = col_results['shap_weight']
            elif self.analysis_params['weighting_method'] == 'ic':
                ensemble_weight = col_results['ic_weight']
            elif self.analysis_params['weighting_method'] == 'meta_learner':
                ensemble_weight = col_results['meta_weight']
            else:
                ensemble_weight = col_results['shap_weight']  # default
            
            col_results['ensemble_weight'] = ensemble_weight
            results[col] = col_results
        
        return results
    
    def _temporal_analysis(self, data: pd.DataFrame, 
                         indicator_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Perform temporal analysis of indicators
        """
        results = {}
        
        for col in indicator_cols:
            if col not in data.columns:
                continue
            
            col_results = {}
            
            # Get time series data
            col_data = data[[col, 'market_value', 'Year']].dropna()
            
            if len(col_data) < 10:
                results[col] = {'error': 'insufficient_data'}
                continue
            
            # Stationarity test
            if self.analysis_params['stationarity_test']:
                try:
                    series = col_data[col]
                    stationary_series, transformation_info = self.feature_engineer.ensure_stationarity(series)
                    
                    # ADF test
                    adf_stat, adf_p = adfuller(stationary_series.dropna())[:2]
                    col_results['adf_statistic'] = adf_stat
                    col_results['adf_p_value'] = adf_p
                    col_results['is_stationary'] = adf_p < 0.05
                    col_results['transformations'] = transformation_info['transformations']
                    
                except Exception as e:
                    logger.warning(f"Stationarity test failed for {col}: {str(e)}")
                    col_results['is_stationary'] = False
            
            # Time-varying correlation analysis
            if self.analysis_params['rolling_analysis'] and len(col_data) > 24:
                try:
                    rolling_corr = col_data[col].rolling(12).corr(col_data['market_value'])
                    col_results['correlation_stability'] = 1 - (rolling_corr.std() / (abs(rolling_corr.mean()) + 1e-8))
                    col_results['correlation_trend'] = np.polyfit(range(len(rolling_corr.dropna())), rolling_corr.dropna(), 1)[0]
                except:
                    col_results['correlation_stability'] = 0.0
                    col_results['correlation_trend'] = 0.0
            
            results[col] = col_results
        
        return results
    
    def _robustness_analysis(self, data: pd.DataFrame, 
                           indicator_cols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Perform robustness analysis using bootstrap and cross-validation
        """
        results = {}
        
        X = data[indicator_cols].select_dtypes(include=[np.number])
        y = data['market_value']
        
        # Remove missing values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 20:
            return {col: {'confidence': 0.0} for col in indicator_cols}
        
        for col in indicator_cols:
            if col not in X_clean.columns:
                continue
            
            col_results = {}
            
            # Bootstrap analysis
            bootstrap_corrs = []
            n_samples = len(X_clean)
            
            for _ in range(min(self.analysis_params['bootstrap_samples'], 50)):  # Limit for performance
                # Bootstrap sample
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_clean.iloc[indices]
                y_boot = y_clean.iloc[indices]
                
                # Calculate correlation
                try:
                    corr = X_boot[col].corr(y_boot)
                    if not np.isnan(corr):
                        bootstrap_corrs.append(corr)
                except:
                    continue
            
            if bootstrap_corrs:
                col_results['bootstrap_mean'] = np.mean(bootstrap_corrs)
                col_results['bootstrap_std'] = np.std(bootstrap_corrs)
                col_results['bootstrap_ci_lower'] = np.percentile(bootstrap_corrs, 2.5)
                col_results['bootstrap_ci_upper'] = np.percentile(bootstrap_corrs, 97.5)
                
                # Confidence score based on bootstrap stability
                col_results['confidence'] = 1 - (col_results['bootstrap_std'] / (abs(col_results['bootstrap_mean']) + 1e-8))
                col_results['confidence'] = max(0, min(1, col_results['confidence']))
            else:
                col_results['confidence'] = 0.0
            
            results[col] = col_results
        
        return results
    
    def _generate_advanced_visualizations(self) -> str:
        """
        Generate comprehensive visualizations of analysis results
        """
        output_dir = self.config_manager.config.get('output', {}).get('save_path', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Enhanced Indicator Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Weight comparison across methods
        if self.feature_importance_results:
            weight_data = []
            for indicator, results in self.feature_importance_results.items():
                weight_data.append({
                    'Indicator': indicator,
                    'IC Weight': results.get('ic_weight', 0),
                    'SHAP Weight': results.get('shap_weight', 0),
                    'Meta Weight': results.get('meta_weight', 0),
                    'Ensemble Weight': results.get('ensemble_weight', 0)
                })
            
            if weight_data:
                weight_df = pd.DataFrame(weight_data)
                weight_df.set_index('Indicator')[['IC Weight', 'SHAP Weight', 'Meta Weight', 'Ensemble Weight']].plot(
                    kind='bar', ax=axes[0, 0], rot=45
                )
                axes[0, 0].set_title('Weight Comparison Across Methods')
                axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Correlation methods comparison
        if self.correlation_results:
            corr_methods = ['pearson', 'spearman', 'kendall', 'dtw']
            corr_data = []
            for indicator, results in self.correlation_results.items():
                row = {'Indicator': indicator}
                for method in corr_methods:
                    row[method.title()] = results.get(method, 0)
                corr_data.append(row)
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                corr_df.set_index('Indicator')[['Pearson', 'Spearman', 'Kendall', 'Dtw']].plot(
                    kind='bar', ax=axes[0, 1], rot=45
                )
                axes[0, 1].set_title('Correlation Methods Comparison')
                axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 3: Confidence scores
        if self.robustness_results:
            confidence_data = [(ind, res.get('confidence', 0)) 
                             for ind, res in self.robustness_results.items()]
            if confidence_data:
                indicators, confidences = zip(*confidence_data)
                axes[1, 0].bar(range(len(indicators)), confidences)
                axes[1, 0].set_xticks(range(len(indicators)))
                axes[1, 0].set_xticklabels(indicators, rotation=45)
                axes[1, 0].set_title('Robustness Confidence Scores')
                axes[1, 0].set_ylabel('Confidence')
        
        # Plot 4: Bootstrap confidence intervals
        if self.robustness_results:
            boot_data = []
            for indicator, results in self.robustness_results.items():
                if 'bootstrap_ci_lower' in results and 'bootstrap_ci_upper' in results:
                    boot_data.append({
                        'Indicator': indicator,
                        'Mean': results.get('bootstrap_mean', 0),
                        'CI_Lower': results.get('bootstrap_ci_lower', 0),
                        'CI_Upper': results.get('bootstrap_ci_upper', 0)
                    })
            
            if boot_data:
                boot_df = pd.DataFrame(boot_data)
                x_pos = range(len(boot_df))
                axes[1, 1].errorbar(x_pos, boot_df['Mean'], 
                                  yerr=[boot_df['Mean'] - boot_df['CI_Lower'], 
                                        boot_df['CI_Upper'] - boot_df['Mean']], 
                                  fmt='o', capsize=5)
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(boot_df['Indicator'], rotation=45)
                axes[1, 1].set_title('Bootstrap Confidence Intervals')
                axes[1, 1].set_ylabel('Correlation')
        
        # Plot 5: Time-varying correlation example
        if self.temporal_analysis_results:
            stability_data = [(ind, res.get('correlation_stability', 0)) 
                            for ind, res in self.temporal_analysis_results.items() 
                            if 'correlation_stability' in res]
            if stability_data:
                indicators, stabilities = zip(*stability_data)
                axes[2, 0].bar(range(len(indicators)), stabilities)
                axes[2, 0].set_xticks(range(len(indicators)))
                axes[2, 0].set_xticklabels(indicators, rotation=45)
                axes[2, 0].set_title('Correlation Stability Over Time')
                axes[2, 0].set_ylabel('Stability Score')
        
        # Plot 6: Final ensemble weights
        if self.indicator_weights:
            final_weights = list(self.indicator_weights.values())
            final_indicators = list(self.indicator_weights.keys())
            axes[2, 1].pie(final_weights, labels=final_indicators, autopct='%1.1f%%')
            axes[2, 1].set_title('Final Ensemble Weights Distribution')
        
        plt.tight_layout()
        
        # Save visualization
        output_file = os.path.join(output_dir, 'enhanced_indicator_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved enhanced visualizations to: {output_file}")
        return output_file
    
    def _generate_comprehensive_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate comprehensive Excel report of all analysis results
        """
        output_dir = self.config_manager.config.get('output', {}).get('save_path', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        excel_file = os.path.join(output_dir, 'enhanced_indicator_analysis_report.xlsx')
        
        try:
            with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_data = []
                for ind_name, res in results.items():
                    summary_data.append({
                        'Indicator': ind_name,
                        'Final Weight': res['final_weight'],
                        'Confidence Score': res['confidence_score'],
                        'Pearson Correlation': res['correlation'].get('pearson', 0),
                        'SHAP Weight': res['weights'].get('shap_weight', 0),
                        'Is Stationary': res['temporal'].get('is_stationary', False),
                        'Related Features': len(res['related_features'])
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values('Final Weight', ascending=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed correlation sheet
                corr_data = []
                for ind_name, res in results.items():
                    corr_res = res['correlation']
                    corr_data.append({
                        'Indicator': ind_name,
                        'Pearson': corr_res.get('pearson', 0),
                        'Pearson P-value': corr_res.get('pearson_p', 1),
                        'Spearman': corr_res.get('spearman', 0),
                        'Kendall': corr_res.get('kendall', 0),
                        'DTW': corr_res.get('dtw', 0),
                        'Max Lagged Corr': corr_res.get('max_lagged_corr', 0),
                        'Optimal Lag': corr_res.get('optimal_lag', 0)
                    })
                
                corr_df = pd.DataFrame(corr_data)
                corr_df.to_excel(writer, sheet_name='Correlations', index=False)
                
                # Weights comparison sheet
                weight_data = []
                for ind_name, res in results.items():
                    weight_res = res['weights']
                    weight_data.append({
                        'Indicator': ind_name,
                        'IC Weight': weight_res.get('ic_weight', 0),
                        'SHAP Weight': weight_res.get('shap_weight', 0),
                        'Meta Weight': weight_res.get('meta_weight', 0),
                        'Ensemble Weight': weight_res.get('ensemble_weight', 0)
                    })
                
                weight_df = pd.DataFrame(weight_data)
                weight_df.to_excel(writer, sheet_name='Weights', index=False)
                
                # Robustness sheet
                robust_data = []
                for ind_name, res in results.items():
                    robust_res = res['robustness']
                    robust_data.append({
                        'Indicator': ind_name,
                        'Confidence': robust_res.get('confidence', 0),
                        'Bootstrap Mean': robust_res.get('bootstrap_mean', 0),
                        'Bootstrap Std': robust_res.get('bootstrap_std', 0),
                        'CI Lower': robust_res.get('bootstrap_ci_lower', 0),
                        'CI Upper': robust_res.get('bootstrap_ci_upper', 0)
                    })
                
                robust_df = pd.DataFrame(robust_data)
                robust_df.to_excel(writer, sheet_name='Robustness', index=False)
                
                # Temporal analysis sheet
                temporal_data = []
                for ind_name, res in results.items():
                    temporal_res = res['temporal']
                    temporal_data.append({
                        'Indicator': ind_name,
                        'Is Stationary': temporal_res.get('is_stationary', False),
                        'ADF P-value': temporal_res.get('adf_p_value', 1),
                        'Correlation Stability': temporal_res.get('correlation_stability', 0),
                        'Correlation Trend': temporal_res.get('correlation_trend', 0),
                        'Transformations': ', '.join(temporal_res.get('transformations', []))
                    })
                
                temporal_df = pd.DataFrame(temporal_data)
                temporal_df.to_excel(writer, sheet_name='Temporal Analysis', index=False)
                
            logger.info(f"Generated comprehensive report: {excel_file}")
            return excel_file
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return ""
    
    def get_indicator_weights(self) -> Dict[str, float]:
        """Get final indicator weights"""
        return self.indicator_weights
    
    def apply_indicator_adjustments(self, country_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply enhanced indicator-based adjustments to country market shares
        """
        if not self.indicator_weights:
            logger.warning("No enhanced weights available, running analysis first")
            self.analyze_indicators_advanced()
        
        if not self.indicator_weights:
            logger.warning("No indicator weights available after analysis")
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
        
        # Apply the adjustments using the enhanced weights (simplified implementation)
        logger.info("Applying enhanced indicator-based adjustments")
        
        # Create adjustment scores for each country
        if id_col in adjusted_df.columns:
            for country_id in adjusted_df[id_col].unique():
                country_mask = adjusted_df[id_col] == country_id
                
                # Calculate composite adjustment score
                adjustment_score = 1.0  # neutral
                
                for indicator_name, weight in self.indicator_weights.items():
                    if weight <= 0:
                        continue
                        
                    # Get latest indicator data for this country
                    indicator_data = all_indicators[
                        (all_indicators['Indicator'] == indicator_name) &
                        (all_indicators[id_col] == country_id)
                    ]
                    
                    if not indicator_data.empty:
                        latest_value = indicator_data['Value'].iloc[-1]
                        # Normalize to 0.5-1.5 range (simple normalization)
                        normalized_value = 0.75 + (latest_value / (latest_value + 1)) * 0.5
                        adjustment_score *= (1 + weight * (normalized_value - 1))
                
                # Apply bounded adjustment to market shares
                if 'market_share' in adjusted_df.columns:
                    bounded_score = max(0.5, min(1.5, adjustment_score))
                    adjusted_df.loc[country_mask, 'market_share'] *= bounded_score
        
        # Normalize market shares to sum to 100%
        if 'market_share' in adjusted_df.columns:
            total_share = adjusted_df['market_share'].sum()
            if total_share > 0:
                adjusted_df['market_share'] = adjusted_df['market_share'] / total_share * 100
        
        return adjusted_df
    
    def analyze_indicators(self) -> Dict[str, Dict[str, Any]]:
        """
        Legacy method name - calls the advanced analysis
        """
        return self.analyze_indicators_advanced()
    
    def get_indicator_correlations(self) -> Dict[str, float]:
        """
        Get calculated indicator correlations for backwards compatibility
        
        Returns:
            Dictionary mapping indicator names to correlation coefficients
        """
        if not self.correlation_results:
            logger.warning("No correlation results available, running analysis first")
            self.analyze_indicators_advanced()
        
        # Extract Pearson correlations from the correlation results
        correlations = {}
        for indicator, results in self.correlation_results.items():
            correlations[indicator] = results.get('pearson', 0.0)
        
        return correlations
    
    def get_indicator_scores(self) -> Dict[str, float]:
        """
        Get indicator scores for backwards compatibility
        
        Returns:
            Dictionary mapping indicator names to confidence scores
        """
        if not self.robustness_results:
            logger.warning("No robustness results available, running analysis first")
            self.analyze_indicators_advanced()
        
        # Extract confidence scores from robustness results
        scores = {}
        for indicator, results in self.robustness_results.items():
            scores[indicator] = results.get('confidence', 0.0)
        
        return scores