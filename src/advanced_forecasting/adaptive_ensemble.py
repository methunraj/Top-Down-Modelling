"""
Adaptive Ensemble Manager - Intelligence-driven ensemble optimization

This module provides adaptive ensemble management that:
1. Automatically selects optimal model combinations based on market conditions
2. Adjusts ensemble weights based on recent performance and regime changes
3. Provides volatility-aware ensemble strategies
4. Integrates seamlessly with Monte Carlo and confidence systems
5. Learns from forecast accuracy to improve future ensemble decisions

Built to work perfectly with existing ensemble infrastructure while adding
next-generation adaptive intelligence.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
import warnings

from src.global_forecasting.base_forecaster import BaseForecaster
from src.global_forecasting.ensemble import WeightedEnsembleForecaster

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketConditionAnalyzer:
    """
    Analyzes current market conditions to inform ensemble selection
    """
    
    def __init__(self, lookback_periods: int = 12):
        """
        Initialize market condition analyzer
        
        Args:
            lookback_periods: Number of periods to analyze for conditions
        """
        self.lookback_periods = lookback_periods
        self.condition_history = []
        
    def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current market conditions
        
        Args:
            data: Historical market data
            
        Returns:
            Dictionary with market condition analysis
        """
        conditions = {}
        
        if 'value' not in data.columns:
            logger.warning("No 'value' column found in data")
            return conditions
        
        # Get recent data
        recent_data = data.tail(self.lookback_periods)
        
        if len(recent_data) < 3:
            logger.warning("Insufficient data for market condition analysis")
            return conditions
        
        # Calculate basic statistics
        values = recent_data['value'].values
        returns = np.diff(values) / values[:-1]
        
        # Volatility analysis
        conditions['volatility'] = np.std(returns)
        conditions['volatility_percentile'] = self._calculate_historical_percentile(
            data, 'volatility', conditions['volatility']
        )
        
        # Trend analysis
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)
        conditions['trend_slope'] = slope
        conditions['trend_strength'] = abs(r_value)
        conditions['trend_significance'] = p_value
        
        # Momentum analysis
        short_ma = np.mean(values[-5:]) if len(values) >= 5 else np.mean(values)
        long_ma = np.mean(values)
        conditions['momentum'] = (short_ma - long_ma) / long_ma
        
        # Regime classification
        conditions['regime'] = self._classify_regime(conditions)
        
        # Market phase
        conditions['phase'] = self._identify_market_phase(values, returns)
        
        # Stability measures
        conditions['stability'] = self._calculate_stability(returns)
        
        # Predictability index
        conditions['predictability'] = self._calculate_predictability(values)
        
        # Store in history
        condition_record = conditions.copy()
        condition_record['timestamp'] = datetime.now()
        self.condition_history.append(condition_record)
        
        # Keep limited history
        if len(self.condition_history) > 100:
            self.condition_history = self.condition_history[-100:]
        
        return conditions
    
    def _calculate_historical_percentile(self, 
                                       data: pd.DataFrame, 
                                       metric: str, 
                                       current_value: float) -> float:
        """Calculate percentile of current value in historical context"""
        if len(data) < self.lookback_periods:
            return 0.5
        
        # Calculate historical values
        historical_values = []
        for i in range(self.lookback_periods, len(data)):
            window = data.iloc[i-self.lookback_periods:i]['value'].values
            if metric == 'volatility':
                returns = np.diff(window) / window[:-1]
                historical_values.append(np.std(returns))
        
        if not historical_values:
            return 0.5
        
        # Calculate percentile
        return stats.percentileofscore(historical_values, current_value) / 100
    
    def _classify_regime(self, conditions: Dict[str, Any]) -> str:
        """Classify current market regime"""
        volatility = conditions.get('volatility', 0)
        trend_strength = conditions.get('trend_strength', 0)
        momentum = conditions.get('momentum', 0)
        
        if volatility > 0.1:  # High volatility
            if abs(momentum) > 0.05:
                return 'volatile_trending'
            else:
                return 'volatile_sideways'
        else:  # Low volatility
            if trend_strength > 0.7 and abs(momentum) > 0.02:
                return 'stable_trending'
            else:
                return 'stable_sideways'
    
    def _identify_market_phase(self, values: np.ndarray, returns: np.ndarray) -> str:
        """Identify current market phase"""
        if len(values) < 5:
            return 'unknown'
        
        # Recent performance
        recent_return = (values[-1] - values[-5]) / values[-5] if len(values) >= 5 else 0
        recent_volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0
        
        # Classify phase
        if recent_return > 0.1 and recent_volatility < 0.05:
            return 'growth'
        elif recent_return > 0.05 and recent_volatility > 0.05:
            return 'expansion'
        elif abs(recent_return) < 0.02 and recent_volatility < 0.03:
            return 'maturity'
        elif recent_return < -0.05:
            return 'decline'
        else:
            return 'transition'
    
    def _calculate_stability(self, returns: np.ndarray) -> float:
        """Calculate market stability index"""
        if len(returns) < 3:
            return 0.5
        
        # Stability based on consistency of returns
        return 1.0 / (1.0 + np.std(returns))
    
    def _calculate_predictability(self, values: np.ndarray) -> float:
        """Calculate market predictability index"""
        if len(values) < 10:
            return 0.5
        
        # Simple AR(1) model fit quality as predictability measure
        lagged_values = values[:-1]
        current_values = values[1:]
        
        correlation = np.corrcoef(lagged_values, current_values)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0


class EnsembleOptimizer:
    """
    Optimizes ensemble weights based on performance and market conditions
    """
    
    def __init__(self, 
                 optimization_method: str = 'adaptive',
                 performance_window: int = 20,
                 min_weight: float = 0.01):
        """
        Initialize ensemble optimizer
        
        Args:
            optimization_method: Method for optimization ('adaptive', 'performance', 'diversified')
            performance_window: Window for performance evaluation
            min_weight: Minimum weight for any forecaster
        """
        self.optimization_method = optimization_method
        self.performance_window = performance_window
        self.min_weight = min_weight
        
        self.performance_history = {}
        self.optimization_history = []
        
    def optimize_weights(self, 
                        forecasters: List[BaseForecaster],
                        historical_data: pd.DataFrame,
                        market_conditions: Dict[str, Any],
                        validation_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Optimize ensemble weights based on forecaster performance and market conditions
        
        Args:
            forecasters: List of forecasters in ensemble
            historical_data: Historical data for evaluation
            market_conditions: Current market conditions
            validation_data: Optional validation data
            
        Returns:
            Optimized weights dictionary
        """
        logger.info(f"Optimizing ensemble weights using {self.optimization_method} method")
        
        # Evaluate individual forecaster performance
        performance_metrics = self._evaluate_forecaster_performance(
            forecasters, historical_data, validation_data
        )
        
        # Apply optimization method
        if self.optimization_method == 'adaptive':
            weights = self._adaptive_optimization(performance_metrics, market_conditions)
        elif self.optimization_method == 'performance':
            weights = self._performance_based_optimization(performance_metrics)
        elif self.optimization_method == 'diversified':
            weights = self._diversified_optimization(performance_metrics, market_conditions)
        else:
            # Default to equal weights
            weights = {f.name: 1.0/len(forecasters) for f in forecasters}
        
        # Store optimization history
        optimization_record = {
            'timestamp': datetime.now(),
            'method': self.optimization_method,
            'market_conditions': market_conditions,
            'performance_metrics': performance_metrics,
            'optimized_weights': weights.copy()
        }
        self.optimization_history.append(optimization_record)
        
        return weights
    
    def _evaluate_forecaster_performance(self,
                                       forecasters: List[BaseForecaster],
                                       historical_data: pd.DataFrame,
                                       validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate performance of individual forecasters"""
        performance_metrics = {}
        
        # Use validation data if available, otherwise use historical data
        eval_data = validation_data if validation_data is not None else historical_data
        
        if len(eval_data) < self.performance_window:
            logger.warning("Insufficient data for performance evaluation")
            return {}
        
        # Split data for evaluation
        train_data = eval_data.iloc[:-self.performance_window]
        test_data = eval_data.iloc[-self.performance_window:]
        
        for forecaster in forecasters:
            try:
                # Fit forecaster on training data
                forecaster.fit(train_data)
                
                # Generate forecast for test period
                forecast = forecaster.forecast(len(test_data))
                
                # Calculate performance metrics
                if len(forecast) == len(test_data):
                    actual_values = test_data['value'].values
                    forecast_values = forecast['value'].values
                    
                    mape = mean_absolute_percentage_error(actual_values, forecast_values) * 100
                    rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
                    
                    # Bias calculation
                    bias = np.mean(forecast_values - actual_values)
                    
                    # Consistency (inverse of error variance)
                    errors = forecast_values - actual_values
                    consistency = 1.0 / (1.0 + np.var(errors))
                    
                    # Directional accuracy
                    actual_directions = np.sign(np.diff(actual_values))
                    forecast_directions = np.sign(np.diff(forecast_values))
                    directional_accuracy = np.mean(actual_directions == forecast_directions)
                    
                    performance_metrics[forecaster.name] = {
                        'mape': mape,
                        'rmse': rmse,
                        'bias': bias,
                        'consistency': consistency,
                        'directional_accuracy': directional_accuracy,
                        'composite_score': self._calculate_composite_score(mape, rmse, consistency, directional_accuracy)
                    }
                
            except Exception as e:
                logger.warning(f"Error evaluating {forecaster.name}: {str(e)}")
                performance_metrics[forecaster.name] = {
                    'mape': 100.0,
                    'rmse': float('inf'),
                    'bias': 0.0,
                    'consistency': 0.0,
                    'directional_accuracy': 0.5,
                    'composite_score': 0.0
                }
        
        return performance_metrics
    
    def _calculate_composite_score(self, 
                                 mape: float, 
                                 rmse: float, 
                                 consistency: float, 
                                 directional_accuracy: float) -> float:
        """Calculate composite performance score"""
        # Normalize MAPE (lower is better)
        mape_score = max(0, 1 - mape / 100)
        
        # Normalize RMSE (lower is better, use relative measure)
        rmse_score = 1.0 / (1.0 + rmse)
        
        # Consistency and directional accuracy are already in [0,1] range
        
        # Weighted combination
        composite = (0.4 * mape_score + 
                    0.3 * rmse_score + 
                    0.2 * consistency + 
                    0.1 * directional_accuracy)
        
        return composite
    
    def _adaptive_optimization(self, 
                             performance_metrics: Dict[str, Dict[str, float]],
                             market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Adaptive optimization based on market conditions and performance"""
        weights = {}
        
        # Get market regime and volatility
        regime = market_conditions.get('regime', 'unknown')
        volatility = market_conditions.get('volatility', 0.05)
        predictability = market_conditions.get('predictability', 0.5)
        
        # Adjust optimization strategy based on market conditions
        if regime in ['volatile_trending', 'volatile_sideways']:
            # In volatile markets, favor consistent forecasters
            for name, metrics in performance_metrics.items():
                consistency_weight = metrics.get('consistency', 0.5)
                directional_weight = metrics.get('directional_accuracy', 0.5)
                weights[name] = 0.6 * consistency_weight + 0.4 * directional_weight
        
        elif regime in ['stable_trending', 'stable_sideways']:
            # In stable markets, favor accurate forecasters
            for name, metrics in performance_metrics.items():
                accuracy_weight = 1 - (metrics.get('mape', 50) / 100)
                bias_penalty = 1 - abs(metrics.get('bias', 0)) / 10
                weights[name] = 0.7 * accuracy_weight + 0.3 * bias_penalty
        
        else:
            # Default: use composite scores
            for name, metrics in performance_metrics.items():
                weights[name] = metrics.get('composite_score', 0.5)
        
        # Adjust for predictability
        if predictability < 0.3:  # Low predictability
            # Increase diversification
            weights = self._increase_diversification(weights)
        
        # Normalize and apply minimum weights
        weights = self._normalize_weights(weights)
        
        return weights
    
    def _performance_based_optimization(self, 
                                      performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Pure performance-based optimization"""
        weights = {}
        
        for name, metrics in performance_metrics.items():
            # Use composite score as primary weight
            weights[name] = metrics.get('composite_score', 0.5)
        
        return self._normalize_weights(weights)
    
    def _diversified_optimization(self, 
                                performance_metrics: Dict[str, Dict[str, float]],
                                market_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Diversified optimization balancing performance and diversity"""
        weights = {}
        
        # Base weights from performance
        performance_weights = self._performance_based_optimization(performance_metrics)
        
        # Calculate diversity bonus
        n_forecasters = len(performance_metrics)
        diversity_bonus = 1.0 / n_forecasters  # Equal diversity bonus
        
        # Combine performance and diversity
        for name in performance_metrics.keys():
            performance_weight = performance_weights.get(name, 0)
            weights[name] = 0.7 * performance_weight + 0.3 * diversity_bonus
        
        return self._normalize_weights(weights)
    
    def _increase_diversification(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Increase diversification by reducing weight concentration"""
        # Calculate concentration
        weight_values = list(weights.values())
        max_weight = max(weight_values) if weight_values else 0
        
        # If too concentrated, redistribute
        if max_weight > 0.5:
            n_forecasters = len(weights)
            target_max = 0.4
            
            # Reduce top weights and redistribute
            adjusted_weights = {}
            excess_weight = 0
            
            for name, weight in weights.items():
                if weight > target_max:
                    excess_weight += weight - target_max
                    adjusted_weights[name] = target_max
                else:
                    adjusted_weights[name] = weight
            
            # Redistribute excess weight
            if excess_weight > 0:
                redistribution = excess_weight / len([w for w in adjusted_weights.values() if w < target_max])
                for name, weight in adjusted_weights.items():
                    if weight < target_max:
                        adjusted_weights[name] += redistribution
            
            return adjusted_weights
        
        return weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights and apply minimum constraints"""
        # Apply minimum weights
        for name in weights:
            weights[name] = max(self.min_weight, weights[name])
        
        # Normalize to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight/total_weight for name, weight in weights.items()}
        
        return weights


class VolatilityAwareEnsemble(WeightedEnsembleForecaster):
    """
    Volatility-aware ensemble that adapts to market volatility
    """
    
    def __init__(self, config=None):
        """Initialize volatility-aware ensemble"""
        super().__init__(config)
        self.name = "Volatility-Aware Ensemble"
        
        # Volatility parameters
        self.volatility_window = self.config.get('volatility_window', 12)
        self.volatility_threshold_low = self.config.get('volatility_threshold_low', 0.02)
        self.volatility_threshold_high = self.config.get('volatility_threshold_high', 0.1)
        
        # Adaptive parameters
        self.volatility_adjustment_factor = self.config.get('volatility_adjustment_factor', 0.5)
        
        # Storage
        self.volatility_history = []
        self.volatility_weights = {}
        
    def fit(self, data: pd.DataFrame) -> 'VolatilityAwareEnsemble':
        """Fit the volatility-aware ensemble"""
        # Calculate historical volatility
        self._calculate_volatility_profile(data)
        
        # Fit base ensemble
        super().fit(data)
        
        # Adjust weights based on volatility
        self._adjust_weights_for_volatility()
        
        return self
    
    def _calculate_volatility_profile(self, data: pd.DataFrame):
        """Calculate volatility profile of the data"""
        if 'value' not in data.columns:
            return
        
        values = data['value'].values
        returns = np.diff(values) / values[:-1]
        
        # Rolling volatility
        volatilities = []
        for i in range(self.volatility_window, len(returns)):
            window_returns = returns[i-self.volatility_window:i]
            volatility = np.std(window_returns)
            volatilities.append(volatility)
        
        # Store volatility statistics
        if volatilities:
            self.volatility_profile = {
                'mean_volatility': np.mean(volatilities),
                'volatility_std': np.std(volatilities),
                'current_volatility': volatilities[-1] if volatilities else 0,
                'volatility_percentile': self._calculate_volatility_percentile(volatilities)
            }
        else:
            self.volatility_profile = {
                'mean_volatility': 0.05,
                'volatility_std': 0.02,
                'current_volatility': 0.05,
                'volatility_percentile': 0.5
            }
    
    def _calculate_volatility_percentile(self, volatilities: List[float]) -> float:
        """Calculate current volatility percentile"""
        if not volatilities:
            return 0.5
        
        current_vol = volatilities[-1]
        return stats.percentileofscore(volatilities, current_vol) / 100
    
    def _adjust_weights_for_volatility(self):
        """Adjust ensemble weights based on volatility"""
        if not hasattr(self, 'volatility_profile'):
            return
        
        current_volatility = self.volatility_profile['current_volatility']
        volatility_percentile = self.volatility_profile['volatility_percentile']
        
        # Determine volatility regime
        if current_volatility < self.volatility_threshold_low:
            volatility_regime = 'low'
        elif current_volatility > self.volatility_threshold_high:
            volatility_regime = 'high'
        else:
            volatility_regime = 'medium'
        
        # Adjust weights based on regime
        adjusted_weights = {}
        
        for forecaster in self.forecasters:
            original_weight = self.weights.get(forecaster.name, 1.0/len(self.forecasters))
            
            # Get forecaster characteristics
            forecaster_volatility_preference = self._get_forecaster_volatility_preference(forecaster)
            
            # Calculate adjustment
            if volatility_regime == 'low':
                # Favor trend-following models in low volatility
                if forecaster_volatility_preference == 'trend':
                    adjustment = 1.0 + self.volatility_adjustment_factor * 0.2
                else:
                    adjustment = 1.0
            elif volatility_regime == 'high':
                # Favor mean-reverting or robust models in high volatility
                if forecaster_volatility_preference == 'robust':
                    adjustment = 1.0 + self.volatility_adjustment_factor * 0.3
                elif forecaster_volatility_preference == 'mean_revert':
                    adjustment = 1.0 + self.volatility_adjustment_factor * 0.2
                else:
                    adjustment = 1.0 - self.volatility_adjustment_factor * 0.1
            else:
                # Medium volatility - balanced approach
                adjustment = 1.0
            
            adjusted_weights[forecaster.name] = original_weight * adjustment
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            self.weights = {name: weight/total_weight for name, weight in adjusted_weights.items()}
        
        # Store volatility-adjusted weights
        self.volatility_weights = self.weights.copy()
    
    def _get_forecaster_volatility_preference(self, forecaster: BaseForecaster) -> str:
        """Determine forecaster's volatility preference"""
        # Simple heuristic based on forecaster name/type
        forecaster_name = forecaster.name.lower()
        
        if 'arima' in forecaster_name or 'moving_average' in forecaster_name:
            return 'trend'
        elif 'exponential' in forecaster_name or 'prophet' in forecaster_name:
            return 'trend'
        elif 'random_forest' in forecaster_name or 'xgboost' in forecaster_name:
            return 'robust'
        elif 'bass' in forecaster_name or 'gompertz' in forecaster_name:
            return 'mean_revert'
        else:
            return 'balanced'


class AdaptiveEnsembleManager:
    """
    Main adaptive ensemble manager that coordinates all adaptive ensemble capabilities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize adaptive ensemble manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.market_analyzer = MarketConditionAnalyzer(
            lookback_periods=self.config.get('lookback_periods', 12)
        )
        
        self.ensemble_optimizer = EnsembleOptimizer(
            optimization_method=self.config.get('optimization_method', 'adaptive'),
            performance_window=self.config.get('performance_window', 20),
            min_weight=self.config.get('min_weight', 0.01)
        )
        
        # Ensemble storage
        self.ensembles = {}
        self.current_ensemble = None
        self.adaptation_history = []
        
        # Performance tracking
        self.performance_tracker = {}
        
    def create_adaptive_ensemble(self, 
                                forecasters: List[BaseForecaster],
                                ensemble_type: str = 'volatility_aware',
                                data: pd.DataFrame = None) -> Union[VolatilityAwareEnsemble, WeightedEnsembleForecaster]:
        """
        Create an adaptive ensemble
        
        Args:
            forecasters: List of base forecasters
            ensemble_type: Type of ensemble ('volatility_aware', 'weighted')
            data: Historical data for analysis
            
        Returns:
            Configured adaptive ensemble
        """
        logger.info(f"Creating {ensemble_type} ensemble with {len(forecasters)} forecasters")
        
        # Analyze market conditions if data provided
        market_conditions = {}
        if data is not None:
            market_conditions = self.market_analyzer.analyze_market_conditions(data)
        
        # Create appropriate ensemble
        if ensemble_type == 'volatility_aware':
            ensemble = VolatilityAwareEnsemble(self.config)
        else:
            ensemble = WeightedEnsembleForecaster(self.config)
        
        # Add forecasters
        for forecaster in forecasters:
            ensemble.add_forecaster(forecaster)
        
        # Optimize weights if data available
        if data is not None:
            optimized_weights = self.ensemble_optimizer.optimize_weights(
                forecasters, data, market_conditions
            )
            
            # Update ensemble weights
            for name, weight in optimized_weights.items():
                ensemble.weights[name] = weight
        
        # Store ensemble
        ensemble_id = f"{ensemble_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.ensembles[ensemble_id] = {
            'ensemble': ensemble,
            'forecasters': forecasters,
            'market_conditions': market_conditions,
            'creation_time': datetime.now()
        }
        
        self.current_ensemble = ensemble
        
        return ensemble
    
    def adapt_ensemble_weights(self, 
                             ensemble_id: str = None,
                             new_data: pd.DataFrame = None,
                             performance_data: Dict[str, float] = None) -> Dict[str, float]:
        """
        Adapt ensemble weights based on new data or performance
        
        Args:
            ensemble_id: ID of ensemble to adapt (uses current if None)
            new_data: New data for analysis
            performance_data: Performance feedback
            
        Returns:
            Updated weights
        """
        # Get ensemble
        if ensemble_id and ensemble_id in self.ensembles:
            ensemble_info = self.ensembles[ensemble_id]
            ensemble = ensemble_info['ensemble']
            forecasters = ensemble_info['forecasters']
        elif self.current_ensemble:
            ensemble = self.current_ensemble
            forecasters = [f for f in ensemble.forecasters]
        else:
            logger.error("No ensemble available for adaptation")
            return {}
        
        # Analyze new market conditions
        market_conditions = {}
        if new_data is not None:
            market_conditions = self.market_analyzer.analyze_market_conditions(new_data)
        
        # Re-optimize weights
        updated_weights = self.ensemble_optimizer.optimize_weights(
            forecasters, new_data or pd.DataFrame(), market_conditions
        )
        
        # Update ensemble
        ensemble.weights = updated_weights
        
        # Store adaptation record
        adaptation_record = {
            'timestamp': datetime.now(),
            'ensemble_id': ensemble_id,
            'market_conditions': market_conditions,
            'old_weights': ensemble.weights.copy(),
            'new_weights': updated_weights.copy(),
            'performance_data': performance_data
        }
        self.adaptation_history.append(adaptation_record)
        
        logger.info(f"Adapted ensemble weights: {updated_weights}")
        
        return updated_weights
    
    def get_ensemble_recommendations(self, 
                                   market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ensemble configuration recommendations based on market conditions
        
        Args:
            market_conditions: Current market conditions
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {}
        
        regime = market_conditions.get('regime', 'unknown')
        volatility = market_conditions.get('volatility', 0.05)
        predictability = market_conditions.get('predictability', 0.5)
        
        # Recommend ensemble type
        if volatility > 0.1:
            recommendations['ensemble_type'] = 'volatility_aware'
            recommendations['reason'] = 'High volatility detected'
        else:
            recommendations['ensemble_type'] = 'weighted'
            recommendations['reason'] = 'Stable conditions suitable for weighted ensemble'
        
        # Recommend optimization method
        if predictability < 0.3:
            recommendations['optimization_method'] = 'diversified'
            recommendations['optimization_reason'] = 'Low predictability favors diversification'
        elif volatility > 0.08:
            recommendations['optimization_method'] = 'adaptive'
            recommendations['optimization_reason'] = 'High volatility requires adaptive approach'
        else:
            recommendations['optimization_method'] = 'performance'
            recommendations['optimization_reason'] = 'Stable conditions allow performance-based optimization'
        
        # Recommend forecaster types
        forecaster_recommendations = []
        
        if regime in ['volatile_trending', 'volatile_sideways']:
            forecaster_recommendations.extend(['RandomForest', 'ExponentialSmoothing', 'SimpleAverage'])
        elif regime in ['stable_trending']:
            forecaster_recommendations.extend(['ARIMA', 'Prophet', 'LinearRegression'])
        elif regime in ['stable_sideways']:
            forecaster_recommendations.extend(['MovingAverage', 'ExponentialSmoothing', 'ARIMA'])
        else:
            forecaster_recommendations.extend(['Prophet', 'ARIMA', 'ExponentialSmoothing'])
        
        recommendations['recommended_forecasters'] = forecaster_recommendations
        
        # Recommend ensemble parameters
        recommendations['parameters'] = {
            'volatility_window': min(24, max(6, int(12 / volatility))),
            'performance_window': min(30, max(10, int(20 / volatility))),
            'min_weight': max(0.01, min(0.1, volatility)),
            'learning_rate': max(0.05, min(0.3, volatility * 2))
        }
        
        return recommendations
    
    def monitor_ensemble_performance(self, 
                                   actual_values: pd.DataFrame,
                                   forecast_values: pd.DataFrame,
                                   ensemble_id: str = None) -> Dict[str, Any]:
        """
        Monitor and analyze ensemble performance
        
        Args:
            actual_values: Actual values
            forecast_values: Forecast values
            ensemble_id: Ensemble ID to monitor
            
        Returns:
            Performance analysis
        """
        # Calculate performance metrics
        try:
            merged = pd.merge(actual_values, forecast_values, 
                            on=['Year'] if 'Year' in actual_values.columns else None,
                            suffixes=('_actual', '_forecast'))
            
            if len(merged) > 0:
                actual = merged['value_actual'] if 'value_actual' in merged.columns else merged['Value_actual']
                forecast = merged['value_forecast'] if 'value_forecast' in merged.columns else merged['Value_forecast']
                
                performance = {
                    'mape': mean_absolute_percentage_error(actual, forecast) * 100,
                    'rmse': np.sqrt(mean_squared_error(actual, forecast)),
                    'bias': np.mean(forecast - actual),
                    'directional_accuracy': np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(forecast))),
                    'r2': 1 - np.sum((actual - forecast) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
                }
                
                # Store performance
                performance_record = {
                    'timestamp': datetime.now(),
                    'ensemble_id': ensemble_id,
                    'performance': performance
                }
                
                if ensemble_id not in self.performance_tracker:
                    self.performance_tracker[ensemble_id] = []
                
                self.performance_tracker[ensemble_id].append(performance_record)
                
                # Analyze performance trends
                performance['trend'] = self._analyze_performance_trend(ensemble_id)
                
                return performance
        
        except Exception as e:
            logger.error(f"Error monitoring ensemble performance: {str(e)}")
        
        return {}
    
    def _analyze_performance_trend(self, ensemble_id: str) -> str:
        """Analyze performance trend for an ensemble"""
        if ensemble_id not in self.performance_tracker:
            return 'insufficient_data'
        
        records = self.performance_tracker[ensemble_id]
        
        if len(records) < 3:
            return 'insufficient_data'
        
        # Get recent MAPEs
        recent_mapes = [r['performance']['mape'] for r in records[-5:]]
        
        # Simple trend analysis
        if len(recent_mapes) >= 3:
            slope, _, _, p_value, _ = stats.linregress(range(len(recent_mapes)), recent_mapes)
            
            if p_value < 0.05:  # Significant trend
                if slope < -1:
                    return 'improving'
                elif slope > 1:
                    return 'deteriorating'
                else:
                    return 'stable'
            else:
                return 'no_trend'
        
        return 'stable'