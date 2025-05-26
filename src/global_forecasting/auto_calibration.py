"""
Auto-calibrating Learning System Module - Adaptive model optimization

This module provides functionality to continuously evaluate and automatically
recalibrate forecasting models based on historical accuracy, ensuring that
the system learns from its past performance and adapts to changing market
conditions and data patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import pearsonr
from scipy.optimize import minimize

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoCalibrator:
    """
    Auto-calibrating Learning System for market forecasting.
    
    This class provides functionality to evaluate forecast accuracy, learn from past
    performance, and automatically recalibrate forecasting models and parameters to
    optimize future forecasts.
    """
    
    def __init__(self, config_manager, data_loader):
        """
        Initialize the AutoCalibrator
        
        Args:
            config_manager: Configuration manager instance for accessing settings
            data_loader: Data loader instance for accessing market data
        """
        self.config_manager = config_manager
        self.data_loader = data_loader
        
        # Get calibration settings from configuration
        self.calibration_settings = self.config_manager.get_value(
            'market_distribution.calibration', {}
        )
        
        # Default settings if not specified in config
        self.default_settings = {
            'enabled': True,
            'accuracy_metrics': ['mape', 'rmse', 'r2'],
            'evaluation_periods': [1, 3, 5],  # Years to evaluate
            'component_weights': {
                'tier_classification': 0.2,
                'causal_integration': 0.2,
                'gradient_harmonization': 0.15,
                'distribution_method': 0.3,
                'regional_aggregation': 0.15
            },
            'recalibration_strategy': 'adaptive',  # Options: adaptive, gradual, aggressive
            'auto_adjust_weights': True,
            'learning_rate': 0.15,  # How quickly to adapt weights
            'memory_length': 5,  # How many past calibrations to consider
            'apply_country_specific_adjustments': True,
            'confidence_thresholds': {
                'high': 0.85,
                'medium': 0.7,
                'low': 0.5
            },
            'save_calibration_history': True,
            'backup_frequency': 5  # Backup every N calibrations
        }
        
        # Merge config settings with defaults
        self._initialize_settings()
        
        # Initialize tracking variables
        self.forecast_history = {}
        self.calibration_history = []
        self.component_performance = {}
        self.country_performance = {}
        self.calibration_count = 0
        self.latest_calibration_date = None
        
        # Load existing history if available
        self._load_history()
        
        # Initialize component parameters and weights
        self.component_params = {}
        self.optimal_weights = {}
        
        # Initialize performance metrics
        self.current_metrics = {}
    
    def _initialize_settings(self) -> None:
        """
        Initialize settings by merging configuration with defaults
        """
        self.settings = self.default_settings.copy()
        
        # Update with values from configuration
        for key, value in self.calibration_settings.items():
            if key in self.settings:
                # Handle nested dictionaries
                if isinstance(value, dict) and isinstance(self.settings[key], dict):
                    self.settings[key].update(value)
                else:
                    self.settings[key] = value
        
        # Log settings
        logger.info(f"Auto-calibration initialized with strategy: {self.settings['recalibration_strategy']}")
        if not self.settings['enabled']:
            logger.info("Auto-calibration is disabled")
    
    def _load_history(self) -> None:
        """
        Load calibration history from storage if available
        """
        try:
            # Get output directory from config
            output_dir = self.config_manager.get_output_directory()
            history_file = os.path.join(output_dir, 'calibration_history.json')
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                self.calibration_history = history_data.get('calibration_history', [])
                self.component_performance = history_data.get('component_performance', {})
                self.country_performance = history_data.get('country_performance', {})
                self.calibration_count = history_data.get('calibration_count', 0)
                self.latest_calibration_date = history_data.get('latest_calibration_date')
                
                logger.info(f"Loaded calibration history with {len(self.calibration_history)} past calibrations")
            else:
                logger.info("No previous calibration history found, starting fresh")
        except Exception as e:
            logger.warning(f"Error loading calibration history: {str(e)}")
    
    def _save_history(self) -> None:
        """
        Save calibration history to storage
        """
        if not self.settings['save_calibration_history']:
            return
            
        try:
            # Get output directory from config
            output_dir = self.config_manager.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            
            history_file = os.path.join(output_dir, 'calibration_history.json')
            
            # Create history data structure
            history_data = {
                'calibration_history': self.calibration_history[-self.settings['memory_length']:],
                'component_performance': self.component_performance,
                'country_performance': self.country_performance,
                'calibration_count': self.calibration_count,
                'latest_calibration_date': datetime.now().isoformat()
            }
            
            # Save to file
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Create a backup if needed
            if self.calibration_count % self.settings['backup_frequency'] == 0:
                backup_file = os.path.join(output_dir, f'calibration_history_backup_{self.calibration_count}.json')
                with open(backup_file, 'w') as f:
                    json.dump(history_data, f, indent=2)
            
            logger.info(f"Saved calibration history to {history_file}")
        except Exception as e:
            logger.warning(f"Error saving calibration history: {str(e)}")
    
    def evaluate_forecast_accuracy(self, 
                                  historical_data: pd.DataFrame,
                                  forecast_data: pd.DataFrame,
                                  actual_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate the accuracy of previous forecasts against actual data
        
        Args:
            historical_data: DataFrame with historical market data used for forecasting
            forecast_data: DataFrame with forecast market data
            actual_data: DataFrame with actual market data to compare against forecasts
                         (if None, will try to load from data_loader)
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.settings['enabled']:
            logger.info("Auto-calibration is disabled, skipping evaluation")
            return {}
        
        logger.info("Evaluating forecast accuracy")
        
        # If actual data not provided, try to load from data_loader
        if actual_data is None:
            try:
                actual_data = self.data_loader.load_country_historical()
                logger.info(f"Loaded actual data with {len(actual_data)} records")
            except Exception as e:
                logger.warning(f"Could not load actual data: {str(e)}")
                return {}
        
        # Get column mappings
        id_col = 'idGeo'  # Default
        country_col = 'Country'  # Default
        
        # Get column mappings from configuration if available
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        if column_mapping:
            id_col = column_mapping.get('id_column', id_col)
            country_col = column_mapping.get('country_column', country_col)
        
        # Find common years between forecast and actual data
        forecast_years = sorted(forecast_data['Year'].unique())
        actual_years = sorted(actual_data['Year'].unique())
        
        # Determine evaluation periods based on available data
        evaluation_periods = []
        for period in self.settings['evaluation_periods']:
            cutoff_year = max(actual_years) - period
            if cutoff_year in forecast_years:
                evaluation_periods.append(period)
        
        if not evaluation_periods:
            logger.warning("No suitable evaluation periods found")
            return {}
        
        logger.info(f"Using evaluation periods: {evaluation_periods} years")
        
        # Calculate accuracy metrics for each country and period
        metrics = {}
        
        for period in evaluation_periods:
            period_metrics = {
                'mape': [],  # Mean Absolute Percentage Error
                'rmse': [],  # Root Mean Squared Error
                'r2': [],    # R-squared
                'bias': []   # Systematic bias
            }
            
            # Get cutoff year for this period
            cutoff_year = max(actual_years) - period
            
            # Filter data for evaluation period
            eval_forecast = forecast_data[forecast_data['Year'] > cutoff_year].copy()
            eval_actual = actual_data[actual_data['Year'] > cutoff_year].copy()
            
            # Calculate metrics for each country
            country_metrics = {}
            
            # Process all countries in common
            common_countries = set(eval_forecast[id_col].unique()).intersection(
                set(eval_actual[id_col].unique())
            )
            
            for country_id in common_countries:
                # Get country name
                country_rows = eval_actual[eval_actual[id_col] == country_id]
                if country_rows.empty:
                    continue
                country_name = country_rows[country_col].iloc[0]
                
                # Get forecast and actual values for this country
                country_forecast = eval_forecast[eval_forecast[id_col] == country_id].copy()
                country_actual = eval_actual[eval_actual[id_col] == country_id].copy()
                
                # Find common years
                common_years = set(country_forecast['Year']).intersection(set(country_actual['Year']))
                
                if len(common_years) < 2:
                    # Not enough data points for meaningful comparison
                    continue
                
                # Align data on common years
                aligned_forecast = []
                aligned_actual = []
                
                for year in sorted(common_years):
                    # Fixed: Add validation that f_val and a_val are single values
                    f_data = country_forecast[country_forecast['Year'] == year]['Value']
                    a_data = country_actual[country_actual['Year'] == year]['Value']
                    
                    if len(f_data) == 0 or len(a_data) == 0:
                        continue  # Skip if no data for this year
                    
                    # Take first value if multiple matches exist
                    f_val = f_data.iloc[0] if len(f_data) > 0 else None
                    a_val = a_data.iloc[0] if len(a_data) > 0 else None
                    
                    if f_val is not None and a_val is not None:
                        aligned_forecast.append(f_val)
                        aligned_actual.append(a_val)
                
                # Calculate metrics
                try:
                    # Calculate metrics only if we have enough data and non-zero actual values
                    if len(aligned_actual) >= 2 and sum(aligned_actual) > 0:
                        mape = mean_absolute_percentage_error(aligned_actual, aligned_forecast) * 100
                        rmse = np.sqrt(mean_squared_error(aligned_actual, aligned_forecast))
                        r2 = r2_score(aligned_actual, aligned_forecast)
                        
                        # Calculate bias (average percentage difference)
                        pct_diff = [(f - a) / a * 100 if a != 0 else 0 
                                   for f, a in zip(aligned_forecast, aligned_actual)]
                        bias = np.mean(pct_diff)
                        
                        # Store metrics for this country
                        country_metrics[country_id] = {
                            'name': country_name,
                            'mape': mape,
                            'rmse': rmse,
                            'r2': r2,
                            'bias': bias,
                            'sample_size': len(aligned_actual)
                        }
                        
                        # Add to period metrics
                        period_metrics['mape'].append(mape)
                        period_metrics['rmse'].append(rmse)
                        period_metrics['r2'].append(r2)
                        period_metrics['bias'].append(bias)
                except Exception as e:
                    logger.warning(f"Error calculating metrics for {country_name}: {str(e)}")
            
            # Calculate aggregate metrics for this period
            agg_metrics = {}
            
            for metric, values in period_metrics.items():
                if values:
                    if metric == 'r2':
                        # For R-squared, use median (more robust to outliers)
                        agg_metrics[metric] = np.median(values)
                    else:
                        # For error metrics, use weighted mean based on country size
                        weights = []
                        weighted_values = []
                        
                        for country_id, c_metrics in country_metrics.items():
                            country_forecast = eval_forecast[eval_forecast[id_col] == country_id]
                            if not country_forecast.empty:
                                weight = country_forecast['Value'].mean()
                                if weight > 0 and not np.isnan(weight) and not np.isinf(weight):
                                    weights.append(weight)
                                    weighted_values.append(c_metrics[metric] * weight)
                                else:
                                    # Use equal weighting if invalid weight
                                    weights.append(1.0)
                                    weighted_values.append(c_metrics[metric])
                        
                        # Calculate weighted average
                        if sum(weights) > 0:
                            agg_metrics[metric] = sum(weighted_values) / sum(weights)
                        else:
                            agg_metrics[metric] = np.mean(values)
                else:
                    agg_metrics[metric] = None
            
            # Store metrics for this period
            metrics[f"{period}yr"] = {
                'aggregate': agg_metrics,
                'countries': country_metrics,
                'evaluation_years': sorted(common_years)
            }
            
            # Update country performance tracking
            for country_id, c_metrics in country_metrics.items():
                if country_id not in self.country_performance:
                    self.country_performance[country_id] = {
                        'name': c_metrics['name'],
                        'history': []
                    }
                
                # Add to history
                self.country_performance[country_id]['history'].append({
                    'period': f"{period}yr",
                    'mape': c_metrics['mape'],
                    'rmse': c_metrics['rmse'],
                    'r2': c_metrics['r2'],
                    'bias': c_metrics['bias'],
                    'calibration_id': self.calibration_count + 1
                })
                
                # Limit history length
                if len(self.country_performance[country_id]['history']) > self.settings['memory_length']:
                    self.country_performance[country_id]['history'] = \
                        self.country_performance[country_id]['history'][-self.settings['memory_length']:]
        
        # Calculate overall metrics across all periods
        overall_metrics = {}
        
        for metric in ['mape', 'rmse', 'r2', 'bias']:
            values = []
            for period_data in metrics.values():
                value = period_data['aggregate'].get(metric)
                if value is not None:
                    values.append(value)
            
            if values:
                overall_metrics[metric] = np.mean(values)
        
        # Add overall metrics to results
        metrics['overall'] = overall_metrics
        
        # Store current metrics
        self.current_metrics = metrics
        
        # Add to calibration history
        self.calibration_history.append({
            'calibration_id': self.calibration_count + 1,
            'date': datetime.now().isoformat(),
            'metrics': metrics,
            'evaluation_periods': evaluation_periods
        })
        
        # Limit history length
        if len(self.calibration_history) > self.settings['memory_length']:
            self.calibration_history = self.calibration_history[-self.settings['memory_length']:]
        
        logger.info(f"Overall MAPE: {overall_metrics.get('mape', 'N/A'):.2f}%, "
                   f"RMSE: {overall_metrics.get('rmse', 'N/A'):.2f}, "
                   f"R²: {overall_metrics.get('r2', 'N/A'):.4f}")
        
        return metrics
    
    def calibrate_models(self, market_analyzer=None) -> Dict[str, Any]:
        """
        Recalibrate forecasting models based on accuracy evaluation
        
        Args:
            market_analyzer: Optional market analyzer instance to access components
            
        Returns:
            Dictionary with calibration changes
        """
        if not self.settings['enabled']:
            logger.info("Auto-calibration is disabled, skipping calibration")
            return {}
        
        if not self.calibration_history:
            logger.warning("No calibration history available, skipping calibration")
            return {}
        
        logger.info("Performing model calibration")
        
        # Get latest metrics
        latest_metrics = self.calibration_history[-1]['metrics']
        overall_metrics = latest_metrics.get('overall', {})
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(overall_metrics)
        logger.info(f"Current confidence score: {confidence_score:.2f}")
        
        # Determine calibration approach based on confidence
        calibration_approach = self._determine_calibration_approach(confidence_score)
        logger.info(f"Using {calibration_approach} calibration approach")
        
        # Analyze component contributions to error
        component_impacts = self._analyze_component_impacts(
            latest_metrics, market_analyzer
        )
        
        # Update component parameters based on impacts
        parameter_changes = self._update_component_parameters(
            component_impacts, calibration_approach, market_analyzer
        )
        
        # Calculate optimal component weights
        weight_changes = {}
        if self.settings['auto_adjust_weights'] and len(self.calibration_history) >= 2:
            weight_changes = self._optimize_component_weights(market_analyzer)
        
        # Apply country-specific adjustments if enabled
        country_adjustments = {}
        if self.settings['apply_country_specific_adjustments']:
            country_adjustments = self._apply_country_specific_adjustments(
                latest_metrics, calibration_approach
            )
        
        # Update calibration count and save history
        self.calibration_count += 1
        self.latest_calibration_date = datetime.now().isoformat()
        self._save_history()
        
        # Generate calibration report
        calibration_report = {
            'calibration_id': self.calibration_count,
            'date': self.latest_calibration_date,
            'confidence_score': confidence_score,
            'approach': calibration_approach,
            'parameter_changes': parameter_changes,
            'weight_changes': weight_changes,
            'country_adjustments': country_adjustments,
            'overall_metrics': overall_metrics
        }
        
        # Generate visualizations if enabled
        try:
            self._generate_calibration_visualizations(calibration_report)
        except Exception as e:
            logger.warning(f"Error generating calibration visualizations: {str(e)}")
        
        return calibration_report
    
    def _calculate_confidence_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate confidence score based on forecast metrics
        
        Args:
            metrics: Dictionary with forecast accuracy metrics
            
        Returns:
            Confidence score between 0 and 1
        """
        # Define metric weights for confidence calculation
        metric_weights = {
            'mape': 0.4,  # Higher weight for MAPE
            'r2': 0.4,    # Higher weight for R²
            'rmse': 0.1,  # Lower weight for RMSE (can be scale-dependent)
            'bias': 0.1   # Lower weight for bias
        }
        
        score_components = []
        
        # Calculate score component for MAPE
        if 'mape' in metrics and metrics['mape'] is not None:
            # Convert MAPE to a 0-1 score (lower MAPE = higher score)
            # MAPE below 5% is excellent (score near 1)
            # MAPE above 50% is poor (score near 0)
            mape_score = max(0, min(1, 1 - (metrics['mape'] / 50)))
            score_components.append(mape_score * metric_weights['mape'])
        
        # Calculate score component for R²
        if 'r2' in metrics and metrics['r2'] is not None:
            # R² is already in 0-1 range (higher is better)
            r2_score = max(0, min(1, metrics['r2']))
            score_components.append(r2_score * metric_weights['r2'])
        
        # Calculate score component for RMSE
        # This is trickier since RMSE is scale-dependent
        # We'll normalize it based on the mean value
        if 'rmse' in metrics and metrics['rmse'] is not None:
            # Use a relative RMSE approach
            # Assuming RMSE / mean_value < 0.5 is good
            # This would require additional data, for simplicity we'll use a fixed scale
            rmse_score = max(0, min(1, 1 - (metrics['rmse'] / 1000)))
            score_components.append(rmse_score * metric_weights['rmse'])
        
        # Calculate score component for bias
        if 'bias' in metrics and metrics['bias'] is not None:
            # Convert bias to a 0-1 score (lower absolute bias = higher score)
            # Bias below 5% is excellent (score near 1)
            # Bias above 25% is poor (score near 0)
            bias_score = max(0, min(1, 1 - (abs(metrics['bias']) / 25)))
            score_components.append(bias_score * metric_weights['bias'])
        
        # Calculate overall confidence score
        if score_components:
            # Normalize to account for missing metrics
            total_weight = sum(weight for metric, weight in metric_weights.items() 
                              if metric in metrics and metrics[metric] is not None)
            
            if total_weight > 0:
                return sum(score_components) / total_weight
        
        # Default if no components could be calculated
        return 0.5
    
    def _determine_calibration_approach(self, confidence_score: float) -> str:
        """
        Determine calibration approach based on confidence score
        
        Args:
            confidence_score: Confidence score between 0 and 1
            
        Returns:
            Calibration approach (conservative, moderate, aggressive)
        """
        # Get thresholds from settings
        thresholds = self.settings['confidence_thresholds']
        
        if confidence_score >= thresholds['high']:
            # High confidence - conservative approach with minor adjustments
            return 'conservative'
        elif confidence_score >= thresholds['medium']:
            # Medium confidence - moderate approach with balanced adjustments
            return 'moderate'
        elif confidence_score >= thresholds['low']:
            # Low confidence - more aggressive adjustments
            return 'aggressive'
        else:
            # Very low confidence - most aggressive adjustments
            return 'very_aggressive'
    
    def _analyze_component_impacts(self, metrics: Dict[str, Any], 
                                  market_analyzer=None) -> Dict[str, float]:
        """
        Analyze which components are contributing most to forecast error
        
        Args:
            metrics: Dictionary with forecast accuracy metrics
            market_analyzer: Optional market analyzer instance to access components
            
        Returns:
            Dictionary with component impact scores (higher = more contribution to error)
        """
        # Default impact values based on component weights
        component_weights = self.settings['component_weights']
        
        # Start with equal impacts
        impacts = {component: 0.5 for component in component_weights.keys()}
        
        # If we have market analyzer and sufficient history, perform more detailed analysis
        if market_analyzer and len(self.calibration_history) >= 2:
            # Get available components from market analyzer
            components = {}
            
            # Try to access various components
            if hasattr(market_analyzer, 'market_distributor'):
                distributor = market_analyzer.market_distributor
                components['distribution_method'] = distributor
                
                # Check for tier classification
                if hasattr(distributor, 'tiers'):
                    components['tier_classification'] = {'tiers': distributor.tiers}
                
                # Check for gradient harmonization
                if hasattr(distributor, 'gradient_harmonizer'):
                    components['gradient_harmonization'] = distributor.gradient_harmonizer
                
                # Check for regional aggregation
                if hasattr(distributor, 'regional_aggregator'):
                    components['regional_aggregation'] = distributor.regional_aggregator
            
            # Check for causal integration
            if hasattr(market_analyzer, 'causal_integration'):
                components['causal_integration'] = market_analyzer.causal_integration
            
            # Analyze errors in relation to component settings
            if components and 'countries' in metrics.get('overall', {}):
                # Get country-level metrics to analyze correlations with component errors
                country_metrics = metrics['overall']['countries']
                
                # For each component, analyze patterns in errors that might indicate issues
                for component_name, component in components.items():
                    impact_score = 0.5  # Default moderate impact
                    
                    # Analyze based on component type
                    if component_name == 'tier_classification':
                        # Check if errors correlate with country tier
                        tier_errors = {}
                        
                        for country_id, c_metrics in country_metrics.items():
                            # Try to get country tier
                            tier = None
                            if hasattr(component, 'tier_thresholds') and c_metrics.get('market_share'):
                                # Estimate tier based on market share
                                share = c_metrics['market_share']
                                tier_thresholds = component['tier_thresholds']
                                
                                for i, threshold in enumerate(tier_thresholds):
                                    if share >= threshold:
                                        tier = i
                                        break
                            
                            if tier is not None:
                                if tier not in tier_errors:
                                    tier_errors[tier] = []
                                tier_errors[tier].append(c_metrics['mape'])
                        
                        # Check for patterns in tier errors
                        if tier_errors:
                            tier_mapes = {tier: np.mean(errors) for tier, errors in tier_errors.items()}
                            
                            # If errors vary significantly by tier, might indicate tier issues
                            if len(tier_mapes) > 1:
                                max_tier_diff = max(tier_mapes.values()) - min(tier_mapes.values())
                                if max_tier_diff > 15:  # >15 percentage points difference
                                    impact_score = 0.7  # Higher impact
                    
                    elif component_name == 'causal_integration':
                        # Check if errors correlate with causal strengths
                        if hasattr(component, 'causal_strengths'):
                            causal_strengths = component.causal_strengths
                            
                            # Check correlation between errors and causal strength
                            strength_pairs = []
                            error_pairs = []
                            
                            for country_id, c_metrics in country_metrics.items():
                                country_name = c_metrics.get('name', '')
                                
                                # Find relevant indicators for this country
                                country_indicators = [ind for ind, strength in causal_strengths.items()
                                                    if country_name.lower() in ind.lower()]
                                
                                if country_indicators:
                                    avg_strength = np.mean([causal_strengths[ind] for ind in country_indicators])
                                    error = c_metrics.get('mape', 0)
                                    
                                    strength_pairs.append(avg_strength)
                                    error_pairs.append(error)
                            
                            # Check correlation
                            if len(strength_pairs) > 5:
                                try:
                                    corr, p_value = pearsonr(strength_pairs, error_pairs)
                                    
                                    if p_value < 0.1:
                                        # Significant correlation
                                        if corr < -0.3:
                                            # Strong negative correlation (higher strength = lower error)
                                            # This is good, lower impact
                                            impact_score = 0.3
                                        elif corr > 0.3:
                                            # Strong positive correlation (higher strength = higher error)
                                            # This is bad, higher impact
                                            impact_score = 0.7
                                except Exception:
                                    pass
                    
                    elif component_name == 'gradient_harmonization':
                        # Check if errors correlate with growth rates
                        growth_errors = []
                        stable_errors = []
                        
                        for country_id, c_metrics in country_metrics.items():
                            # Get growth rate for this country
                            try:
                                country_data = self.data_loader.load_country_historical()
                                country_data = country_data[country_data[id_col] == country_id]
                                
                                growth_rates = country_data.sort_values('Year')['Value'].pct_change() * 100
                                avg_growth = np.nanmean(growth_rates)
                                volatility = np.nanstd(growth_rates)
                                
                                error = c_metrics.get('mape', 0)
                                
                                if volatility > 15:
                                    growth_errors.append(error)
                                else:
                                    stable_errors.append(error)
                            except Exception:
                                pass
                        
                        # Compare errors for high-growth vs stable countries
                        if growth_errors and stable_errors:
                            avg_growth_error = np.mean(growth_errors)
                            avg_stable_error = np.mean(stable_errors)
                            
                            # If high-growth countries have much higher errors,
                            # might indicate harmonization issues
                            if avg_growth_error > 1.5 * avg_stable_error:
                                impact_score = 0.7  # Higher impact
                            elif avg_growth_error < 1.1 * avg_stable_error:
                                impact_score = 0.3  # Lower impact
                    
                    # Store component impact score
                    impacts[component_name] = impact_score
        
        # Normalize impacts
        total_impact = sum(impacts.values())
        if total_impact > 0:
            impacts = {k: v / total_impact for k, v in impacts.items()}
        
        return impacts
    
    def _update_component_parameters(self, component_impacts: Dict[str, float],
                                    calibration_approach: str,
                                    market_analyzer=None) -> Dict[str, Dict[str, Any]]:
        """
        Update component parameters based on impact analysis
        
        Args:
            component_impacts: Dictionary with component impact scores
            calibration_approach: Calibration approach (conservative, moderate, aggressive)
            market_analyzer: Optional market analyzer instance to access components
            
        Returns:
            Dictionary with parameter changes
        """
        # Define adjustment factors based on calibration approach
        adjustment_factors = {
            'conservative': 0.05,   # 5% adjustment
            'moderate': 0.1,       # 10% adjustment
            'aggressive': 0.2,     # 20% adjustment
            'very_aggressive': 0.3  # 30% adjustment
        }
        
        factor = adjustment_factors.get(calibration_approach, 0.1)
        
        # Initialize parameter changes
        parameter_changes = {}
        
        # If we have market analyzer, adjust component parameters
        if market_analyzer:
            # Update tier classification parameters
            tier_impact = component_impacts.get('tier_classification', 0.5)
            if tier_impact > 0.6 and hasattr(market_analyzer, 'market_distributor'):
                distributor = market_analyzer.market_distributor
                
                parameter_changes['tier_classification'] = {}
                
                # Adjust tier thresholds if available
                if hasattr(distributor, 'tier_thresholds') and distributor.tier_thresholds:
                    old_thresholds = distributor.tier_thresholds
                    
                    # Analyze metrics for tier-specific patterns
                    country_metrics = {}
                    for calibration in self.calibration_history:
                        for period_metrics in calibration['metrics'].values():
                            if 'countries' in period_metrics:
                                for country_id, c_metrics in period_metrics['countries'].items():
                                    if country_id not in country_metrics:
                                        country_metrics[country_id] = []
                                    country_metrics[country_id].append(c_metrics)
                    
                    # Calculate average metrics by country
                    avg_metrics = {}
                    for country_id, metrics_list in country_metrics.items():
                        mapes = [m.get('mape', 0) for m in metrics_list if 'mape' in m]
                        if mapes:
                            avg_metrics[country_id] = np.mean(mapes)
                    
                    # Group countries by tier and analyze errors
                    tier_errors = {}
                    
                    # We need market_share for tier assignment
                    shares_available = 'market_share' in distributor.distributed_market.columns
                    
                    if shares_available:
                        for country_id, error in avg_metrics.items():
                            # Get country's market share
                            country_data = distributor.distributed_market[
                                distributor.distributed_market['idGeo'] == country_id
                            ]
                            
                            if country_data.empty:
                                continue
                                
                            share = country_data['market_share'].mean()
                            
                            # Determine tier
                            tier = 0
                            for i, threshold in enumerate(old_thresholds):
                                if share < threshold:
                                    tier = i + 1
                                    break
                            
                            if tier not in tier_errors:
                                tier_errors[tier] = []
                            tier_errors[tier].append(error)
                        
                        # Adjust thresholds based on tier errors
                        if tier_errors and len(old_thresholds) > 0:
                            new_thresholds = old_thresholds.copy()
                            
                            # Adjust boundaries between tiers with highest error differentials
                            tier_avg_errors = {tier: np.mean(errors) for tier, errors in tier_errors.items()}
                            
                            # Find tier boundaries with highest error differentials
                            error_diffs = []
                            for i in range(len(old_thresholds) - 1):
                                if i in tier_avg_errors and i+1 in tier_avg_errors:
                                    error_diff = abs(tier_avg_errors[i] - tier_avg_errors[i+1])
                                    error_diffs.append((i, error_diff))
                            
                            # Sort by error differential
                            error_diffs.sort(key=lambda x: x[1], reverse=True)
                            
                            # Adjust top boundaries with highest error differentials
                            for tier_idx, _ in error_diffs[:2]:  # Adjust top 2 boundaries
                                # Determine direction of adjustment
                                if tier_idx in tier_avg_errors and tier_idx+1 in tier_avg_errors:
                                    if tier_avg_errors[tier_idx] > tier_avg_errors[tier_idx+1]:
                                        # Lower tier has higher error, move boundary up
                                        new_thresholds[tier_idx] = old_thresholds[tier_idx] * (1 + factor)
                                    else:
                                        # Upper tier has higher error, move boundary down
                                        new_thresholds[tier_idx] = old_thresholds[tier_idx] * (1 - factor)
                            
                            # Apply changes
                            distributor.tier_thresholds = new_thresholds
                            parameter_changes['tier_classification']['thresholds'] = {
                                'old': old_thresholds,
                                'new': new_thresholds
                            }
            
            # Update causal integration parameters
            causal_impact = component_impacts.get('causal_integration', 0.5)
            if causal_impact > 0.6 and hasattr(market_analyzer, 'causal_integration'):
                causal = market_analyzer.causal_integration
                
                parameter_changes['causal_integration'] = {}
                
                # Adjust causal parameters if available
                if hasattr(causal, 'causal_analysis_params'):
                    old_params = causal.causal_analysis_params.copy()
                    new_params = old_params.copy()
                    
                    # Adjust parameters based on performance
                    if 'alpha' in new_params:
                        # Adjust significance level
                        if calibration_approach in ['aggressive', 'very_aggressive']:
                            # Make more strict
                            new_params['alpha'] = max(0.01, old_params['alpha'] * (1 - factor))
                        else:
                            # Make less strict
                            new_params['alpha'] = min(0.1, old_params['alpha'] * (1 + factor))
                    
                    # Update parameters
                    causal.causal_analysis_params = new_params
                    parameter_changes['causal_integration']['params'] = {
                        'old': old_params,
                        'new': new_params
                    }
            
            # Update gradient harmonization parameters
            harm_impact = component_impacts.get('gradient_harmonization', 0.5)
            if harm_impact > 0.6 and hasattr(market_analyzer, 'market_distributor'):
                distributor = market_analyzer.market_distributor
                
                if hasattr(distributor, 'gradient_harmonizer'):
                    harmonizer = distributor.gradient_harmonizer
                    
                    parameter_changes['gradient_harmonization'] = {}
                    
                    # Adjust harmonization parameters if available
                    if hasattr(harmonizer, 'settings'):
                        old_settings = harmonizer.settings.copy()
                        new_settings = old_settings.copy()
                        
                        # Adjust smoothing strength
                        if 'smoothing_strength' in new_settings:
                            # Determine if we need more or less smoothing
                            if calibration_approach in ['aggressive', 'very_aggressive']:
                                # Increase smoothing for more aggressive approach
                                new_settings['smoothing_strength'] = min(
                                    0.9, old_settings['smoothing_strength'] * (1 + factor)
                                )
                            else:
                                # Decrease smoothing for more conservative approach
                                new_settings['smoothing_strength'] = max(
                                    0.1, old_settings['smoothing_strength'] * (1 - factor)
                                )
                        
                        # Adjust tier-specific settings if available
                        if 'tier_specific_settings' in new_settings:
                            tier_settings = new_settings['tier_specific_settings']
                            
                            for tier, settings in tier_settings.items():
                                if 'smoothing_strength' in settings:
                                    # Adjust smoothing strength for this tier
                                    old_strength = settings['smoothing_strength']
                                    
                                    if calibration_approach in ['aggressive', 'very_aggressive']:
                                        # Increase smoothing
                                        new_strength = min(0.9, old_strength * (1 + factor))
                                    else:
                                        # Decrease smoothing
                                        new_strength = max(0.1, old_strength * (1 - factor))
                                        
                                    tier_settings[tier]['smoothing_strength'] = new_strength
                        
                        # Update settings
                        harmonizer.settings = new_settings
                        parameter_changes['gradient_harmonization']['settings'] = {
                            'old': old_settings,
                            'new': new_settings
                        }
        
        return parameter_changes
    
    def _optimize_component_weights(self, market_analyzer=None) -> Dict[str, Dict[str, float]]:
        """
        Optimize component weights based on historical performance
        
        Args:
            market_analyzer: Optional market analyzer instance to access components
            
        Returns:
            Dictionary with weight changes
        """
        # Initialize weight changes
        weight_changes = {}
        
        # If we have insufficient history, return empty changes
        if len(self.calibration_history) < 2:
            return weight_changes
        
        # Check if we have component performance data
        if not self.component_performance:
            # Initialize component performance with default weights
            self.component_performance = {
                component: {'weight': weight, 'performance': []}
                for component, weight in self.settings['component_weights'].items()
            }
        
        # Update component performance with latest metrics
        latest_metrics = self.calibration_history[-1]['metrics']['overall']
        
        # We'll use MAPE as our primary metric for optimization
        if 'mape' in latest_metrics:
            mape = latest_metrics['mape']
            
            # Component impacts from most recent analysis
            for component, impact in self.component_impacts.items():
                if component in self.component_performance:
                    # Higher impact = higher contribution to error = worse performance
                    performance = 1 - impact
                    
                    # Add to performance history
                    self.component_performance[component]['performance'].append(performance)
                    
                    # Limit history length
                    if len(self.component_performance[component]['performance']) > self.settings['memory_length']:
                        self.component_performance[component]['performance'] = \
                            self.component_performance[component]['performance'][-self.settings['memory_length']:]
        
        # Optimize weights if we have enough data
        for component, data in self.component_performance.items():
            if len(data['performance']) >= 2:
                # Calculate average performance
                avg_performance = np.mean(data['performance'])
                
                # Fixed: Validate avg_performance range and handle edge cases
                if not np.isfinite(avg_performance):
                    logger.warning(f"Invalid average performance for component {component}: {avg_performance}")
                    continue
                
                # Clamp avg_performance to reasonable range to prevent extreme sigmoid values
                avg_performance = np.clip(avg_performance, 0.0, 1.0)
                
                # Calculate weight adjustment
                old_weight = data['weight']
                
                # Use a sigmoid function to map performance to weight adjustment
                # Performance of 0.5 = no change
                # Performance < 0.5 = decrease weight
                # Performance > 0.5 = increase weight
                try:
                    performance_factor = 1 / (1 + np.exp(-10 * (avg_performance - 0.5)))
                    adjustment = (performance_factor - 0.5) * 2 * self.settings['learning_rate']
                except (OverflowError, ZeroDivisionError) as e:
                    logger.warning(f"Error calculating performance factor for component {component}: {e}")
                    adjustment = 0.0  # No adjustment if calculation fails
                
                # Apply adjustment
                new_weight = max(0.05, min(0.5, old_weight * (1 + adjustment)))
                
                # Update weight
                self.component_performance[component]['weight'] = new_weight
                
                # Add to weight changes
                weight_changes[component] = {
                    'old': old_weight,
                    'new': new_weight,
                    'change': adjustment
                }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(data['weight'] for data in self.component_performance.values())
        
        if total_weight > 0:
            for component in self.component_performance:
                self.component_performance[component]['weight'] /= total_weight
        
        # Return weight changes
        return weight_changes
    
    def _apply_country_specific_adjustments(self, metrics: Dict[str, Any],
                                          calibration_approach: str) -> Dict[str, Dict[str, Any]]:
        """
        Apply country-specific adjustments based on performance
        
        Args:
            metrics: Dictionary with forecast accuracy metrics
            calibration_approach: Calibration approach (conservative, moderate, aggressive)
            
        Returns:
            Dictionary with country-specific adjustments
        """
        # Define adjustment factors based on calibration approach
        adjustment_factors = {
            'conservative': 0.05,   # 5% adjustment
            'moderate': 0.1,       # 10% adjustment
            'aggressive': 0.2,     # 20% adjustment
            'very_aggressive': 0.3  # 30% adjustment
        }
        
        factor = adjustment_factors.get(calibration_approach, 0.1)
        
        # Initialize country adjustments
        country_adjustments = {}
        
        # Get country-level metrics
        country_metrics = {}
        for period_key, period_data in metrics.items():
            if period_key != 'overall' and 'countries' in period_data:
                for country_id, c_metrics in period_data['countries'].items():
                    if country_id not in country_metrics:
                        country_metrics[country_id] = []
                    country_metrics[country_id].append(c_metrics)
        
        # Process countries with high errors
        high_error_countries = []
        
        for country_id, metrics_list in country_metrics.items():
            # Calculate average MAPE
            mapes = [m.get('mape', 0) for m in metrics_list if 'mape' in m]
            if mapes:
                avg_mape = np.mean(mapes)
                country_name = metrics_list[0].get('name', 'Unknown')
                
                # Check if error is significantly high
                if avg_mape > 30:  # >30% MAPE is considered high
                    high_error_countries.append({
                        'id': country_id,
                        'name': country_name,
                        'mape': avg_mape
                    })
        
        # Sort by error (highest first)
        high_error_countries.sort(key=lambda x: x['mape'], reverse=True)
        
        # Apply adjustments to top high-error countries
        for country in high_error_countries[:10]:  # Adjust top 10 high-error countries
            country_id = country['id']
            country_name = country['name']
            mape = country['mape']
            
            # Calculate adjustment strength based on error magnitude
            # Higher error = stronger adjustment
            adjustment_strength = min(0.5, mape / 100) * factor
            
            # Create adjustment parameters
            adjustment = {
                'name': country_name,
                'error': mape,
                'adjustment_strength': adjustment_strength,
                'parameters': {}
            }
            
            # Parameters to adjust:
            # 1. Growth constraints
            # 2. Smoothing parameters
            # 3. Tier classification
            
            # Determine growth constraint adjustments
            bias = 0
            for metrics_list in country_metrics[country_id]:
                if 'bias' in metrics_list:
                    bias += metrics_list['bias']
            
            if len(country_metrics[country_id]) > 0:
                avg_bias = bias / len(country_metrics[country_id])
                
                # If positive bias (forecast > actual), reduce growth constraints
                # If negative bias (forecast < actual), increase growth constraints
                growth_adjustment = -avg_bias / 100 * adjustment_strength
                
                adjustment['parameters']['growth_constraints'] = {
                    'bias': avg_bias,
                    'adjustment': growth_adjustment
                }
            
            # Add to country adjustments
            country_adjustments[country_id] = adjustment
        
        return country_adjustments
    
    def _generate_calibration_visualizations(self, calibration_report: Dict[str, Any]) -> List[str]:
        """
        Generate visualizations of calibration results
        
        Args:
            calibration_report: Dictionary with calibration report
            
        Returns:
            List of paths to generated visualization files
        """
        # Get output directory
        output_dir = self.config_manager.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_files = []
        
        # Generate accuracy trends visualization
        if len(self.calibration_history) >= 2:
            # Extract metrics from history
            mapes = []
            rmses = []
            r2s = []
            calibration_ids = []
            
            for calibration in self.calibration_history:
                overall_metrics = calibration['metrics'].get('overall', {})
                
                if 'mape' in overall_metrics:
                    mapes.append(overall_metrics['mape'])
                    calibration_ids.append(calibration['calibration_id'])
                
                if 'rmse' in overall_metrics:
                    rmses.append(overall_metrics['rmse'])
                
                if 'r2' in overall_metrics:
                    r2s.append(overall_metrics['r2'])
            
            # Plot accuracy trends
            if mapes:
                plt.figure(figsize=(12, 8))
                
                # Plot MAPE trend
                plt.subplot(3, 1, 1)
                plt.plot(calibration_ids, mapes, 'b-o', label='MAPE (%)')
                plt.title('Forecast MAPE Trend')
                plt.xlabel('Calibration ID')
                plt.ylabel('MAPE (%)')
                plt.grid(True, alpha=0.3)
                
                # Plot RMSE trend if available
                if rmses:
                    plt.subplot(3, 1, 2)
                    plt.plot(calibration_ids, rmses, 'r-o', label='RMSE')
                    plt.title('Forecast RMSE Trend')
                    plt.xlabel('Calibration ID')
                    plt.ylabel('RMSE')
                    plt.grid(True, alpha=0.3)
                
                # Plot R² trend if available
                if r2s:
                    plt.subplot(3, 1, 3)
                    plt.plot(calibration_ids, r2s, 'g-o', label='R²')
                    plt.title('Forecast R² Trend')
                    plt.xlabel('Calibration ID')
                    plt.ylabel('R²')
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save visualization
                accuracy_file = os.path.join(output_dir, 'accuracy_trends.png')
                plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_files.append(accuracy_file)
        
        # Generate component weights visualization
        if self.component_performance:
            plt.figure(figsize=(10, 6))
            
            # Extract components and weights
            components = []
            weights = []
            
            for component, data in self.component_performance.items():
                components.append(component)
                weights.append(data['weight'])
            
            # Sort by weight
            sorted_indices = np.argsort(weights)[::-1]
            components = [components[i] for i in sorted_indices]
            weights = [weights[i] for i in sorted_indices]
            
            # Plot weights
            plt.barh(components, weights, color='skyblue')
            plt.xlabel('Weight')
            plt.title('Component Weights')
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save visualization
            weights_file = os.path.join(output_dir, 'component_weights.png')
            plt.savefig(weights_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_files.append(weights_file)
        
        # Generate country performance visualization
        if self.country_performance:
            # Find top 10 countries with highest average MAPE
            country_mapes = {}
            
            for country_id, data in self.country_performance.items():
                mapes = [h.get('mape', 0) for h in data['history'] if 'mape' in h]
                if mapes:
                    country_mapes[country_id] = {
                        'name': data['name'],
                        'mape': np.mean(mapes)
                    }
            
            # Sort by MAPE and get top 10
            top_countries = sorted(
                country_mapes.items(), 
                key=lambda x: x[1]['mape'], 
                reverse=True
            )[:10]
            
            if top_countries:
                plt.figure(figsize=(12, 8))
                
                # Extract country names and MAPEs
                country_names = [c[1]['name'] for c in top_countries]
                country_errors = [c[1]['mape'] for c in top_countries]
                
                # Plot country MAPEs
                plt.barh(country_names, country_errors, color='salmon')
                plt.xlabel('Average MAPE (%)')
                plt.title('Top 10 Countries with Highest Forecast Error')
                plt.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                
                # Save visualization
                country_file = os.path.join(output_dir, 'country_performance.png')
                plt.savefig(country_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_files.append(country_file)
        
        return visualization_files
    
    def apply_auto_calibration(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply auto-calibration adjustments to market data
        
        Args:
            market_data: DataFrame with market forecast data
            
        Returns:
            DataFrame with auto-calibrated market data
        """
        if not self.settings['enabled'] or not self.calibration_history:
            logger.info("Auto-calibration is disabled or no calibration history available")
            return market_data
        
        # Create a copy to avoid modifying the original
        calibrated_data = market_data.copy()
        
        logger.info("Applying auto-calibration adjustments to market data")
        
        # Get latest calibration report
        latest_calibration = self.calibration_history[-1]
        
        # Get country-specific adjustments
        country_adjustments = {}
        latest_metrics = latest_calibration['metrics']
        
        # Extract country-level metrics from all periods
        for period_key, period_data in latest_metrics.items():
            if period_key != 'overall' and 'countries' in period_data:
                for country_id, c_metrics in period_data['countries'].items():
                    if country_id not in country_adjustments:
                        country_adjustments[country_id] = c_metrics
        
        # Apply country-specific adjustments
        id_col = 'idGeo'  # Default
        
        # Get column mappings from configuration if available
        column_mapping = self.config_manager.get_column_mapping('country_historical')
        if column_mapping:
            id_col = column_mapping.get('id_column', id_col)
        
        # For each country with adjustments
        for country_id, adjustments in country_adjustments.items():
            country_mask = calibrated_data[id_col] == country_id
            country_rows = calibrated_data[country_mask]
            
            if country_rows.empty:
                continue
            
            # Get country forecast years only
            forecast_years = []
            if 'is_forecast' in calibrated_data.columns:
                forecast_mask = calibrated_data['is_forecast'] == True
                forecast_years = calibrated_data[forecast_mask]['Year'].unique()
            else:
                # Assume later years are forecast
                all_years = sorted(calibrated_data['Year'].unique())
                forecast_years = all_years[len(all_years)//2:]
            
            # Filter for forecast years only
            forecast_mask = country_mask & calibrated_data['Year'].isin(forecast_years)
            
            # Get bias adjustment if available
            bias = adjustments.get('bias', 0)
            
            if abs(bias) > 10:  # Only adjust for significant bias
                # Calculate adjustment factor (negative bias = increase values)
                adjustment_factor = 1.0 - (bias / 100) * 0.5  # 50% correction of bias
                
                # Apply adjustment to value
                if 'Value' in calibrated_data.columns:
                    calibrated_data.loc[forecast_mask, 'Value'] *= adjustment_factor
                
                # Apply adjustment to market_share if available
                if 'market_share' in calibrated_data.columns:
                    calibrated_data.loc[forecast_mask, 'market_share'] *= adjustment_factor
        
        # Re-normalize market shares if needed
        if 'market_share' in calibrated_data.columns:
            # Normalize shares to sum to 100% for each year
            for year in calibrated_data['Year'].unique():
                year_mask = calibrated_data['Year'] == year
                year_total = calibrated_data.loc[year_mask, 'market_share'].sum()
                
                if year_total > 0:
                    calibrated_data.loc[year_mask, 'market_share'] = (
                        calibrated_data.loc[year_mask, 'market_share'] / year_total * 100
                    )
        
        # Re-calculate values from market shares if needed
        if 'market_share' in calibrated_data.columns and 'Value' in calibrated_data.columns:
            for year in calibrated_data['Year'].unique():
                year_mask = calibrated_data['Year'] == year
                
                # Get total market value for this year
                total_value = calibrated_data.loc[year_mask, 'Value'].sum()
                
                # Re-calculate values from shares
                calibrated_data.loc[year_mask, 'Value'] = (
                    calibrated_data.loc[year_mask, 'market_share'] * total_value / 100
                )
        
        logger.info("Completed auto-calibration adjustments")
        return calibrated_data
    
    def save_calibration_model(self, file_path: Optional[str] = None) -> str:
        """
        Save the calibration model to a file
        
        Args:
            file_path: Path to save the model (if None, use default path)
            
        Returns:
            Path to the saved model file
        """
        if not self.settings['enabled']:
            logger.info("Auto-calibration is disabled, not saving model")
            return ""
        
        # Get output directory from config if file_path not provided
        if file_path is None:
            output_dir = self.config_manager.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, 'calibration_model.pkl')
        
        try:
            # Create model data structure
            model_data = {
                'calibration_history': self.calibration_history,
                'component_performance': self.component_performance,
                'country_performance': self.country_performance,
                'calibration_count': self.calibration_count,
                'current_metrics': self.current_metrics,
                'optimal_weights': self.optimal_weights,
                'component_params': self.component_params,
                'settings': self.settings,
                'latest_calibration_date': datetime.now().isoformat()
            }
            
            # Save model to file
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved calibration model to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving calibration model: {str(e)}")
            return ""
    
    def load_calibration_model(self, file_path: str) -> bool:
        """
        Load a calibration model from a file
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Boolean indicating success
        """
        if not os.path.exists(file_path):
            logger.error(f"Calibration model file not found: {file_path}")
            return False
        
        try:
            # Load model from file
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Update instance variables
            self.calibration_history = model_data.get('calibration_history', [])
            self.component_performance = model_data.get('component_performance', {})
            self.country_performance = model_data.get('country_performance', {})
            self.calibration_count = model_data.get('calibration_count', 0)
            self.current_metrics = model_data.get('current_metrics', {})
            self.optimal_weights = model_data.get('optimal_weights', {})
            self.component_params = model_data.get('component_params', {})
            
            # Update settings if available
            if 'settings' in model_data:
                # Merge with default settings to ensure all required settings are present
                for key, value in model_data['settings'].items():
                    if key in self.settings:
                        # Handle nested dictionaries
                        if isinstance(value, dict) and isinstance(self.settings[key], dict):
                            self.settings[key].update(value)
                        else:
                            self.settings[key] = value
            
            logger.info(f"Loaded calibration model from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration model: {str(e)}")
            return False
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """
        Get the latest calibration metrics
        
        Returns:
            Dictionary with calibration metrics
        """
        if not self.calibration_history:
            return {}
        
        # Get most recent calibration metrics
        latest_calibration = self.calibration_history[-1]
        return latest_calibration['metrics']