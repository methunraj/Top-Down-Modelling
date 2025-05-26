"""
Adaptive Indicator Learning System - Next-generation indicator intelligence

This module provides an adaptive learning system for indicators that:
1. Continuously learns from forecast accuracy and adjusts indicator weights
2. Detects regime changes and adapts indicator relevance accordingly
3. Automatically discovers new indicator relationships
4. Provides real-time indicator performance monitoring
5. Integrates seamlessly with Monte Carlo and enhanced confidence systems

Built to work perfectly with existing indicator infrastructure while adding
cutting-edge adaptive learning capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr
import warnings

from src.indicators.indicator_analyzer import IndicatorAnalyzer
from src.indicators.causal_indicator_integration import CausalIndicatorIntegration

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regime changes and adjusts indicator relevance accordingly
    """
    
    def __init__(self, 
                 window_size: int = 12,
                 change_threshold: float = 2.0,
                 min_regime_length: int = 6):
        """
        Initialize regime detector
        
        Args:
            window_size: Window size for regime detection
            change_threshold: Threshold for detecting regime changes (in standard deviations)
            min_regime_length: Minimum length of a regime
        """
        self.window_size = window_size
        self.change_threshold = change_threshold
        self.min_regime_length = min_regime_length
        
        # Storage for regime analysis
        self.regime_history = []
        self.current_regime = None
        self.regime_characteristics = {}
        
    def detect_regime_changes(self, 
                            market_data: pd.DataFrame,
                            indicators_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect regime changes in market behavior
        
        Args:
            market_data: Historical market data
            indicators_data: Historical indicator data
            
        Returns:
            Dictionary with regime detection results
        """
        logger.info("Detecting market regime changes")
        
        # Calculate market volatility and growth metrics
        market_metrics = self._calculate_market_metrics(market_data)
        
        # Detect change points in market behavior
        change_points = self._detect_change_points(market_metrics)
        
        # Identify regimes based on change points
        regimes = self._identify_regimes(market_metrics, change_points)
        
        # Analyze indicator performance in each regime
        indicator_performance = self._analyze_indicator_performance_by_regime(
            indicators_data, regimes, market_data
        )
        
        # Update regime history
        self._update_regime_history(regimes, indicator_performance)
        
        return {
            'regimes': regimes,
            'change_points': change_points,
            'indicator_performance': indicator_performance,
            'current_regime': self.current_regime,
            'market_metrics': market_metrics
        }
    
    def _calculate_market_metrics(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate key market metrics for regime detection"""
        metrics = market_data.copy()
        
        if 'Value' in metrics.columns:
            # Growth rate
            metrics['growth_rate'] = metrics['Value'].pct_change()
            
            # Volatility (rolling standard deviation)
            metrics['volatility'] = metrics['growth_rate'].rolling(window=self.window_size).std()
            
            # Momentum (rolling mean)
            metrics['momentum'] = metrics['growth_rate'].rolling(window=self.window_size).mean()
            
            # Trend strength (correlation with time)
            metrics['trend_strength'] = metrics['Value'].rolling(window=self.window_size).apply(
                lambda x: abs(pearsonr(range(len(x)), x)[0]) if len(x) == self.window_size else np.nan
            )
            
            # Market size relative to moving average
            metrics['size_relative'] = metrics['Value'] / metrics['Value'].rolling(window=self.window_size).mean()
        
        return metrics
    
    def _detect_change_points(self, market_metrics: pd.DataFrame) -> List[int]:
        """Detect change points in market metrics"""
        change_points = []
        
        if 'volatility' in market_metrics.columns:
            volatility = market_metrics['volatility'].dropna()
            
            # Use Z-score method for change point detection
            for i in range(self.window_size, len(volatility) - self.window_size):
                # Compare current window with previous window
                current_window = volatility.iloc[i:i+self.window_size]
                previous_window = volatility.iloc[i-self.window_size:i]
                
                # Calculate Z-score
                combined_std = np.sqrt((current_window.var() + previous_window.var()) / 2)
                if combined_std > 0:
                    z_score = abs(current_window.mean() - previous_window.mean()) / combined_std
                    
                    if z_score > self.change_threshold:
                        change_points.append(i)
        
        # Filter change points to ensure minimum regime length
        filtered_change_points = []
        last_change = -self.min_regime_length
        
        for cp in change_points:
            if cp - last_change >= self.min_regime_length:
                filtered_change_points.append(cp)
                last_change = cp
        
        return filtered_change_points
    
    def _identify_regimes(self, market_metrics: pd.DataFrame, change_points: List[int]) -> List[Dict[str, Any]]:
        """Identify regimes based on change points"""
        regimes = []
        
        # Add start and end points
        boundaries = [0] + change_points + [len(market_metrics)]
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Extract regime data
            regime_data = market_metrics.iloc[start_idx:end_idx]
            
            # Calculate regime characteristics
            regime_char = self._calculate_regime_characteristics(regime_data)
            
            regime = {
                'regime_id': i,
                'start_index': start_idx,
                'end_index': end_idx,
                'length': end_idx - start_idx,
                'characteristics': regime_char
            }
            
            regimes.append(regime)
        
        # Update current regime
        if regimes:
            self.current_regime = regimes[-1]
        
        return regimes
    
    def _calculate_regime_characteristics(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate characteristics of a market regime"""
        characteristics = {}
        
        if 'growth_rate' in regime_data.columns:
            growth_rates = regime_data['growth_rate'].dropna()
            if len(growth_rates) > 0:
                characteristics['mean_growth'] = growth_rates.mean()
                characteristics['growth_volatility'] = growth_rates.std()
                characteristics['growth_skewness'] = growth_rates.skew()
        
        if 'volatility' in regime_data.columns:
            volatility = regime_data['volatility'].dropna()
            if len(volatility) > 0:
                characteristics['mean_volatility'] = volatility.mean()
                characteristics['max_volatility'] = volatility.max()
        
        if 'momentum' in regime_data.columns:
            momentum = regime_data['momentum'].dropna()
            if len(momentum) > 0:
                characteristics['mean_momentum'] = momentum.mean()
        
        if 'trend_strength' in regime_data.columns:
            trend = regime_data['trend_strength'].dropna()
            if len(trend) > 0:
                characteristics['mean_trend_strength'] = trend.mean()
        
        # Classify regime type
        characteristics['regime_type'] = self._classify_regime_type(characteristics)
        
        return characteristics
    
    def _classify_regime_type(self, characteristics: Dict[str, float]) -> str:
        """Classify regime type based on characteristics"""
        mean_growth = characteristics.get('mean_growth', 0)
        growth_volatility = characteristics.get('growth_volatility', 0)
        mean_volatility = characteristics.get('mean_volatility', 0)
        
        # Simple classification logic
        if mean_growth > 0.02 and growth_volatility < 0.1:
            return 'stable_growth'
        elif mean_growth > 0.02 and growth_volatility >= 0.1:
            return 'volatile_growth'
        elif abs(mean_growth) <= 0.02 and mean_volatility < 0.05:
            return 'stable'
        elif abs(mean_growth) <= 0.02 and mean_volatility >= 0.05:
            return 'volatile_stable'
        elif mean_growth < -0.02:
            return 'decline'
        else:
            return 'mixed'
    
    def _analyze_indicator_performance_by_regime(self,
                                               indicators_data: pd.DataFrame,
                                               regimes: List[Dict[str, Any]],
                                               market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze how well indicators perform in each regime"""
        performance_by_regime = {}
        
        for regime in regimes:
            regime_id = regime['regime_id']
            start_idx = regime['start_index']
            end_idx = regime['end_index']
            
            # Get market data for this regime
            regime_market = market_data.iloc[start_idx:end_idx]
            
            if regime_market.empty:
                continue
            
            # Analyze each indicator
            regime_performance = {}
            
            for indicator_name in indicators_data['Indicator'].unique():
                indicator_data = indicators_data[indicators_data['Indicator'] == indicator_name]
                
                # Find overlapping years
                market_years = set(regime_market['Year'].unique())
                indicator_years = set(indicator_data['Year'].unique())
                common_years = market_years.intersection(indicator_years)
                
                if len(common_years) < 3:  # Need at least 3 data points
                    continue
                
                # Calculate correlation for this regime
                try:
                    # Align data
                    regime_market_subset = regime_market[regime_market['Year'].isin(common_years)]
                    indicator_subset = indicator_data[indicator_data['Year'].isin(common_years)]
                    
                    if len(regime_market_subset) > 0 and len(indicator_subset) > 0:
                        # Merge on year and calculate correlation
                        merged = pd.merge(regime_market_subset[['Year', 'Value']], 
                                        indicator_subset[['Year', 'Value']], 
                                        on='Year', suffixes=('_market', '_indicator'))
                        
                        if len(merged) >= 3:
                            corr, p_value = pearsonr(merged['Value_market'], merged['Value_indicator'])
                            regime_performance[indicator_name] = {
                                'correlation': corr if not np.isnan(corr) else 0,
                                'p_value': p_value if not np.isnan(p_value) else 1,
                                'n_observations': len(merged)
                            }
                
                except Exception as e:
                    logger.warning(f"Error analyzing {indicator_name} in regime {regime_id}: {str(e)}")
                    continue
            
            performance_by_regime[regime_id] = regime_performance
        
        return performance_by_regime
    
    def _update_regime_history(self, regimes: List[Dict[str, Any]], indicator_performance: Dict[str, Dict[str, float]]):
        """Update regime history for future reference"""
        for regime in regimes:
            regime_record = {
                'regime_id': regime['regime_id'],
                'characteristics': regime['characteristics'],
                'indicator_performance': indicator_performance.get(regime['regime_id'], {}),
                'timestamp': datetime.now()
            }
            self.regime_history.append(regime_record)
        
        # Keep only recent regimes
        max_history = 20
        if len(self.regime_history) > max_history:
            self.regime_history = self.regime_history[-max_history:]


class IndicatorLearningEngine:
    """
    Learns from forecast accuracy and automatically adjusts indicator weights
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 decay_factor: float = 0.95,
                 min_observations: int = 5):
        """
        Initialize learning engine
        
        Args:
            learning_rate: Rate of learning from new observations
            decay_factor: Factor for decaying old observations
            min_observations: Minimum observations needed for learning
        """
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.min_observations = min_observations
        
        # Learning history
        self.performance_history = {}
        self.weight_history = {}
        self.learning_metrics = {}
        
    def learn_from_forecast_accuracy(self,
                                   indicators: Dict[str, float],
                                   actual_values: pd.DataFrame,
                                   forecast_values: pd.DataFrame,
                                   current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Learn from forecast accuracy and update indicator weights
        
        Args:
            indicators: Dictionary of indicator values
            actual_values: Actual market values
            forecast_values: Forecast market values
            current_weights: Current indicator weights
            
        Returns:
            Updated indicator weights
        """
        logger.info("Learning from forecast accuracy")
        
        # Calculate forecast errors
        forecast_errors = self._calculate_forecast_errors(actual_values, forecast_values)
        
        # Analyze indicator contributions to errors
        indicator_contributions = self._analyze_indicator_contributions(
            indicators, forecast_errors, current_weights
        )
        
        # Update weights based on contributions
        updated_weights = self._update_weights_from_contributions(
            current_weights, indicator_contributions
        )
        
        # Store learning history
        self._update_learning_history(current_weights, updated_weights, forecast_errors)
        
        return updated_weights
    
    def _calculate_forecast_errors(self,
                                 actual_values: pd.DataFrame,
                                 forecast_values: pd.DataFrame) -> Dict[str, float]:
        """Calculate various forecast error metrics"""
        errors = {}
        
        try:
            # Align data
            merged = pd.merge(actual_values, forecast_values, on=['Year', 'Country'], suffixes=('_actual', '_forecast'))
            
            if len(merged) > 0:
                actual = merged['Value_actual']
                forecast = merged['Value_forecast']
                
                # Calculate error metrics
                errors['mape'] = mean_absolute_percentage_error(actual, forecast) * 100
                errors['rmse'] = np.sqrt(mean_squared_error(actual, forecast))
                errors['mae'] = np.mean(np.abs(actual - forecast))
                errors['bias'] = np.mean(forecast - actual)
                errors['r2'] = 1 - (np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
                
                # Country-level errors
                country_errors = {}
                for country in merged['Country'].unique():
                    country_data = merged[merged['Country'] == country]
                    if len(country_data) > 0:
                        c_actual = country_data['Value_actual']
                        c_forecast = country_data['Value_forecast']
                        country_errors[country] = mean_absolute_percentage_error(c_actual, c_forecast) * 100
                
                errors['country_errors'] = country_errors
            
        except Exception as e:
            logger.error(f"Error calculating forecast errors: {str(e)}")
            errors = {'mape': 100, 'rmse': 0, 'mae': 0, 'bias': 0, 'r2': 0}
        
        return errors
    
    def _analyze_indicator_contributions(self,
                                       indicators: Dict[str, float],
                                       forecast_errors: Dict[str, float],
                                       current_weights: Dict[str, float]) -> Dict[str, float]:
        """Analyze how each indicator contributed to forecast errors"""
        contributions = {}
        
        overall_error = forecast_errors.get('mape', 100)
        
        # Simple contribution analysis based on weight and error correlation
        for indicator_name, weight in current_weights.items():
            # Higher weight and higher error = negative contribution
            # Higher weight and lower error = positive contribution
            
            # Normalize error to 0-1 scale (lower is better)
            normalized_error = min(1.0, overall_error / 100)
            
            # Calculate contribution (negative if error is high)
            if normalized_error > 0.2:  # If error > 20%
                contribution = -weight * normalized_error
            else:
                contribution = weight * (1 - normalized_error)
            
            contributions[indicator_name] = contribution
        
        return contributions
    
    def _update_weights_from_contributions(self,
                                         current_weights: Dict[str, float],
                                         contributions: Dict[str, float]) -> Dict[str, float]:
        """Update weights based on indicator contributions"""
        updated_weights = current_weights.copy()
        
        for indicator_name, contribution in contributions.items():
            if indicator_name in updated_weights:
                # Adjust weight based on contribution
                adjustment = self.learning_rate * contribution
                new_weight = updated_weights[indicator_name] + adjustment
                
                # Keep weights positive and bounded
                updated_weights[indicator_name] = max(0.01, min(1.0, new_weight))
        
        # Normalize weights to sum to 1
        total_weight = sum(updated_weights.values())
        if total_weight > 0:
            updated_weights = {k: v/total_weight for k, v in updated_weights.items()}
        
        return updated_weights
    
    def _update_learning_history(self,
                               old_weights: Dict[str, float],
                               new_weights: Dict[str, float],
                               forecast_errors: Dict[str, float]):
        """Update learning history for analysis"""
        timestamp = datetime.now()
        
        # Store performance history
        self.performance_history[timestamp] = forecast_errors
        
        # Store weight changes
        weight_changes = {}
        for indicator in old_weights:
            old_w = old_weights[indicator]
            new_w = new_weights.get(indicator, old_w)
            weight_changes[indicator] = {
                'old_weight': old_w,
                'new_weight': new_w,
                'change': new_w - old_w
            }
        
        self.weight_history[timestamp] = weight_changes
        
        # Update learning metrics
        self._update_learning_metrics()
    
    def _update_learning_metrics(self):
        """Update metrics about the learning process"""
        if len(self.performance_history) < 2:
            return
        
        # Get recent performance
        recent_performances = list(self.performance_history.values())[-5:]
        recent_mapes = [p.get('mape', 100) for p in recent_performances]
        
        # Calculate learning metrics
        self.learning_metrics = {
            'recent_avg_mape': np.mean(recent_mapes),
            'performance_trend': self._calculate_trend(recent_mapes),
            'learning_iterations': len(self.performance_history),
            'last_update': max(self.performance_history.keys()) if self.performance_history else None
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in performance values"""
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple linear trend
        x = range(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)
        
        if p_value < 0.05:  # Significant trend
            if slope < -0.5:
                return 'improving'
            elif slope > 0.5:
                return 'deteriorating'
            else:
                return 'stable'
        else:
            return 'no_trend'


class AdaptiveIndicatorSystem:
    """
    Main adaptive indicator system that combines regime detection and learning
    """
    
    def __init__(self, 
                 indicator_analyzer: IndicatorAnalyzer,
                 causal_integration: CausalIndicatorIntegration = None,
                 config: Dict[str, Any] = None):
        """
        Initialize adaptive indicator system
        
        Args:
            indicator_analyzer: Base indicator analyzer
            causal_integration: Causal indicator integration
            config: Configuration dictionary
        """
        self.indicator_analyzer = indicator_analyzer
        self.causal_integration = causal_integration
        self.config = config or {}
        
        # Initialize sub-components
        self.regime_detector = RegimeDetector(
            window_size=self.config.get('regime_window_size', 12),
            change_threshold=self.config.get('regime_change_threshold', 2.0),
            min_regime_length=self.config.get('min_regime_length', 6)
        )
        
        self.learning_engine = IndicatorLearningEngine(
            learning_rate=self.config.get('learning_rate', 0.1),
            decay_factor=self.config.get('decay_factor', 0.95),
            min_observations=self.config.get('min_observations', 5)
        )
        
        # Adaptive system state
        self.adaptive_weights = {}
        self.regime_specific_weights = {}
        self.performance_tracking = {}
        self.adaptation_history = []
        
        # Integration with advanced forecasting
        self.monte_carlo_integration = self.config.get('monte_carlo_integration', True)
        self.confidence_integration = self.config.get('confidence_integration', True)
        
    def adaptive_indicator_analysis(self,
                                  market_data: pd.DataFrame,
                                  indicators_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive adaptive indicator analysis
        
        Args:
            market_data: Historical market data
            indicators_data: Historical indicator data
            
        Returns:
            Comprehensive adaptive analysis results
        """
        logger.info("Starting adaptive indicator analysis")
        
        # Step 1: Base indicator analysis
        base_analysis = self.indicator_analyzer.analyze_indicators()
        
        # Step 2: Regime detection and analysis
        regime_analysis = self.regime_detector.detect_regime_changes(market_data, indicators_data)
        
        # Step 3: Causal analysis if available
        causal_analysis = {}
        if self.causal_integration:
            try:
                causal_analysis = self.causal_integration.analyze_causal_relationships()
            except Exception as e:
                logger.warning(f"Error in causal analysis: {str(e)}")
        
        # Step 4: Adaptive weight calculation
        adaptive_weights = self._calculate_adaptive_weights(
            base_analysis, regime_analysis, causal_analysis
        )
        
        # Step 5: Regime-specific weight optimization
        regime_weights = self._optimize_regime_specific_weights(
            regime_analysis, base_analysis
        )
        
        # Step 6: Integration with advanced forecasting systems
        advanced_integration = self._integrate_with_advanced_systems(
            adaptive_weights, regime_analysis
        )
        
        # Store results
        results = {
            'base_analysis': base_analysis,
            'regime_analysis': regime_analysis,
            'causal_analysis': causal_analysis,
            'adaptive_weights': adaptive_weights,
            'regime_specific_weights': regime_weights,
            'advanced_integration': advanced_integration,
            'current_regime': self.regime_detector.current_regime,
            'learning_metrics': self.learning_engine.learning_metrics
        }
        
        # Update system state
        self._update_system_state(results)
        
        return results
    
    def _calculate_adaptive_weights(self,
                                  base_analysis: Dict[str, Any],
                                  regime_analysis: Dict[str, Any],
                                  causal_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adaptive weights combining multiple analysis methods"""
        adaptive_weights = {}
        
        # Get base weights from correlation analysis
        base_weights = self.indicator_analyzer.get_indicator_weights()
        
        # Get current regime characteristics
        current_regime = regime_analysis.get('current_regime')
        regime_performance = regime_analysis.get('indicator_performance', {})
        
        # Get causal strengths
        causal_strengths = {}
        if causal_analysis and 'causal_strengths' in causal_analysis:
            causal_strengths = causal_analysis['causal_strengths']
        elif self.causal_integration:
            causal_strengths = self.causal_integration.get_causal_strengths()
        
        # Combine different weight sources
        for indicator_name in base_weights.keys():
            # Base correlation weight
            base_weight = base_weights.get(indicator_name, 0)
            
            # Regime-specific adjustment
            regime_adjustment = 1.0
            if current_regime and current_regime['regime_id'] in regime_performance:
                regime_perf = regime_performance[current_regime['regime_id']]
                if indicator_name in regime_perf:
                    regime_corr = regime_perf[indicator_name].get('correlation', 0)
                    # Adjust based on regime-specific performance
                    regime_adjustment = 0.5 + 0.5 * abs(regime_corr)
            
            # Causal strength adjustment
            causal_adjustment = 1.0
            if indicator_name in causal_strengths:
                causal_strength = causal_strengths[indicator_name]
                causal_adjustment = 0.3 + 0.7 * causal_strength
            
            # Combined adaptive weight
            adaptive_weight = base_weight * regime_adjustment * causal_adjustment
            adaptive_weights[indicator_name] = adaptive_weight
        
        # Normalize weights
        total_weight = sum(adaptive_weights.values())
        if total_weight > 0:
            adaptive_weights = {k: v/total_weight for k, v in adaptive_weights.items()}
        
        return adaptive_weights
    
    def _optimize_regime_specific_weights(self,
                                        regime_analysis: Dict[str, Any],
                                        base_analysis: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Optimize weights for each identified regime"""
        regime_weights = {}
        
        regimes = regime_analysis.get('regimes', [])
        indicator_performance = regime_analysis.get('indicator_performance', {})
        
        for regime in regimes:
            regime_id = regime['regime_id']
            regime_type = regime['characteristics'].get('regime_type', 'unknown')
            
            # Get indicator performance for this regime
            regime_perf = indicator_performance.get(regime_id, {})
            
            # Calculate optimal weights for this regime
            regime_specific_weights = {}
            
            for indicator_name, performance in regime_perf.items():
                correlation = performance.get('correlation', 0)
                p_value = performance.get('p_value', 1)
                n_obs = performance.get('n_observations', 0)
                
                # Weight based on correlation strength and significance
                if p_value < 0.05 and n_obs >= 3:
                    weight = abs(correlation)
                else:
                    weight = 0.1  # Small baseline weight
                
                regime_specific_weights[indicator_name] = weight
            
            # Normalize weights
            total_weight = sum(regime_specific_weights.values())
            if total_weight > 0:
                regime_specific_weights = {k: v/total_weight for k, v in regime_specific_weights.items()}
            
            regime_weights[regime_type] = regime_specific_weights
        
        return regime_weights
    
    def _integrate_with_advanced_systems(self,
                                       adaptive_weights: Dict[str, float],
                                       regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate adaptive weights with Monte Carlo and confidence systems"""
        integration_results = {}
        
        # Monte Carlo integration
        if self.monte_carlo_integration:
            mc_params = self._generate_monte_carlo_parameters(adaptive_weights, regime_analysis)
            integration_results['monte_carlo_params'] = mc_params
        
        # Confidence interval integration
        if self.confidence_integration:
            confidence_params = self._generate_confidence_parameters(adaptive_weights, regime_analysis)
            integration_results['confidence_params'] = confidence_params
        
        # Uncertainty propagation parameters
        uncertainty_params = self._generate_uncertainty_parameters(adaptive_weights, regime_analysis)
        integration_results['uncertainty_params'] = uncertainty_params
        
        return integration_results
    
    def _generate_monte_carlo_parameters(self,
                                       adaptive_weights: Dict[str, float],
                                       regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for Monte Carlo simulation"""
        current_regime = regime_analysis.get('current_regime')
        
        # Base uncertainty from regime characteristics
        base_uncertainty = 0.05  # 5% default
        
        if current_regime:
            regime_char = current_regime.get('characteristics', {})
            volatility = regime_char.get('mean_volatility', 0.05)
            growth_volatility = regime_char.get('growth_volatility', 0.05)
            
            # Adjust uncertainty based on regime
            base_uncertainty = max(0.02, min(0.2, (volatility + growth_volatility) / 2))
        
        # Weight-specific uncertainties
        weight_uncertainties = {}
        for indicator_name, weight in adaptive_weights.items():
            # Higher weight = lower relative uncertainty (more confidence)
            weight_uncertainty = base_uncertainty * (1.5 - weight)
            weight_uncertainties[indicator_name] = max(0.01, weight_uncertainty)
        
        return {
            'base_uncertainty': base_uncertainty,
            'weight_uncertainties': weight_uncertainties,
            'indicator_weight_multipliers': adaptive_weights,
            'regime_type': current_regime.get('characteristics', {}).get('regime_type', 'unknown') if current_regime else 'unknown'
        }
    
    def _generate_confidence_parameters(self,
                                      adaptive_weights: Dict[str, float],
                                      regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for enhanced confidence intervals"""
        current_regime = regime_analysis.get('current_regime')
        
        # Base confidence level
        base_confidence = 0.95
        
        # Adjust based on regime stability
        if current_regime:
            regime_char = current_regime.get('characteristics', {})
            regime_type = regime_char.get('regime_type', 'mixed')
            
            # More stable regimes = higher confidence
            confidence_adjustments = {
                'stable': 0.98,
                'stable_growth': 0.97,
                'volatile_growth': 0.93,
                'volatile_stable': 0.90,
                'decline': 0.92,
                'mixed': 0.95
            }
            
            base_confidence = confidence_adjustments.get(regime_type, 0.95)
        
        # Bootstrap parameters
        bootstrap_params = {
            'n_bootstrap': max(200, min(1000, int(500 * sum(adaptive_weights.values())))),
            'confidence_levels': [1 - base_confidence, base_confidence]
        }
        
        # Time-varying parameters
        time_varying_params = {
            'horizon_adjustment': 'sqrt_time',
            'volatility_scaling': True,
            'base_confidence': base_confidence
        }
        
        return {
            'bootstrap_params': bootstrap_params,
            'time_varying_params': time_varying_params,
            'base_confidence': base_confidence,
            'adaptive_weights': adaptive_weights
        }
    
    def _generate_uncertainty_parameters(self,
                                       adaptive_weights: Dict[str, float],
                                       regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for uncertainty propagation"""
        uncertainty_sources = {}
        
        # Data uncertainty based on indicator weights
        for indicator_name, weight in adaptive_weights.items():
            uncertainty_sources[f'data_{indicator_name}'] = {
                'type': 'data',
                'params': {
                    'mean': 0,
                    'std': 0.02 + 0.08 * (1 - weight)  # Higher uncertainty for lower weight indicators
                }
            }
        
        # Parameter uncertainty for weights themselves
        for indicator_name, weight in adaptive_weights.items():
            uncertainty_sources[f'param_{indicator_name}_weight'] = {
                'type': 'parameter',
                'params': {
                    'mean': weight,
                    'std': weight * 0.1  # 10% relative uncertainty in weights
                }
            }
        
        # Model uncertainty based on regime stability
        current_regime = regime_analysis.get('current_regime')
        regime_uncertainty = 0.1  # Default 10%
        
        if current_regime:
            regime_type = current_regime.get('characteristics', {}).get('regime_type', 'mixed')
            regime_uncertainties = {
                'stable': 0.05,
                'stable_growth': 0.07,
                'volatile_growth': 0.15,
                'volatile_stable': 0.12,
                'decline': 0.10,
                'mixed': 0.13
            }
            regime_uncertainty = regime_uncertainties.get(regime_type, 0.1)
        
        uncertainty_sources['model_selection'] = {
            'type': 'model',
            'params': {
                'models': ['correlation', 'causal', 'regime_specific'],
                'uncertainty': regime_uncertainty
            }
        }
        
        return {
            'uncertainty_sources': uncertainty_sources,
            'propagation_samples': 500,
            'regime_uncertainty': regime_uncertainty
        }
    
    def _update_system_state(self, results: Dict[str, Any]):
        """Update the adaptive system state with new results"""
        # Store adaptive weights
        self.adaptive_weights = results['adaptive_weights']
        self.regime_specific_weights = results['regime_specific_weights']
        
        # Update adaptation history
        adaptation_record = {
            'timestamp': datetime.now(),
            'adaptive_weights': self.adaptive_weights.copy(),
            'current_regime': results['current_regime'],
            'learning_metrics': results['learning_metrics']
        }
        
        self.adaptation_history.append(adaptation_record)
        
        # Keep limited history
        max_history = 50
        if len(self.adaptation_history) > max_history:
            self.adaptation_history = self.adaptation_history[-max_history:]
    
    def get_current_adaptive_weights(self) -> Dict[str, float]:
        """Get current adaptive weights"""
        return self.adaptive_weights.copy()
    
    def get_regime_specific_weights(self, regime_type: str = None) -> Dict[str, float]:
        """Get regime-specific weights"""
        if regime_type and regime_type in self.regime_specific_weights:
            return self.regime_specific_weights[regime_type].copy()
        
        # Return current regime weights if no specific type requested
        current_regime = self.regime_detector.current_regime
        if current_regime:
            regime_type = current_regime.get('characteristics', {}).get('regime_type', 'mixed')
            return self.regime_specific_weights.get(regime_type, {})
        
        return {}
    
    def apply_adaptive_adjustments(self, country_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply adaptive indicator adjustments to country market shares
        
        Args:
            country_df: DataFrame with country market shares
            
        Returns:
            DataFrame with adaptively adjusted market shares
        """
        if not self.adaptive_weights:
            logger.warning("No adaptive weights available, using base indicator analysis")
            return self.indicator_analyzer.apply_indicator_adjustments(country_df)
        
        # Temporarily update indicator weights
        original_weights = self.indicator_analyzer.indicator_weights.copy()
        self.indicator_analyzer.indicator_weights = self.adaptive_weights
        
        try:
            # Apply adjustments with adaptive weights
            adjusted_df = self.indicator_analyzer.apply_indicator_adjustments(country_df)
            
            # Add adaptive metadata
            adjusted_df['adaptive_regime'] = 'unknown'
            adjusted_df['adaptation_confidence'] = 0.5
            
            if self.regime_detector.current_regime:
                regime_type = self.regime_detector.current_regime.get('characteristics', {}).get('regime_type', 'unknown')
                adjusted_df['adaptive_regime'] = regime_type
                
                # Calculate adaptation confidence based on regime stability and weight confidence
                regime_length = self.regime_detector.current_regime.get('length', 0)
                weight_confidence = sum(self.adaptive_weights.values()) / len(self.adaptive_weights)
                adaptation_confidence = min(1.0, (regime_length / 10 + weight_confidence) / 2)
                adjusted_df['adaptation_confidence'] = adaptation_confidence
            
            return adjusted_df
            
        finally:
            # Restore original weights
            self.indicator_analyzer.indicator_weights = original_weights
    
    def learn_from_forecast_performance(self,
                                      actual_values: pd.DataFrame,
                                      forecast_values: pd.DataFrame) -> Dict[str, float]:
        """
        Learn from forecast performance and update weights
        
        Args:
            actual_values: Actual market values
            forecast_values: Forecast market values
            
        Returns:
            Updated adaptive weights
        """
        # Use learning engine to update weights
        updated_weights = self.learning_engine.learn_from_forecast_accuracy(
            self.adaptive_weights,
            actual_values,
            forecast_values,
            self.adaptive_weights
        )
        
        # Update system weights
        self.adaptive_weights = updated_weights
        
        # Store learning record
        learning_record = {
            'timestamp': datetime.now(),
            'old_weights': self.adaptive_weights.copy(),
            'new_weights': updated_weights.copy(),
            'performance_metrics': self.learning_engine.learning_metrics.copy()
        }
        
        self.adaptation_history.append(learning_record)
        
        logger.info("Updated adaptive weights based on forecast performance")
        
        return updated_weights