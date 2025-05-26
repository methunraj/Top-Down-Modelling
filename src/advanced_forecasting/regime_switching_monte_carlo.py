"""
Regime-Switching Monte Carlo - Market-state dependent parameter distributions

This module implements regime-switching Monte Carlo methods where parameter
distributions change based on market regimes (bull/bear, volatile/stable, etc.).

Key capabilities:
1. Markov regime-switching models
2. Market regime detection and classification
3. Regime-dependent parameter distributions
4. Transition probability estimation
5. Regime path simulation
6. Stress testing across regimes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from scipy import stats
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import warnings

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Market regime detection and classification
    
    Identifies market regimes using various methods including
    HMM, threshold models, and volatility-based classification.
    """
    
    def __init__(self, 
                 method: str = 'hmm',
                 n_regimes: int = 3,
                 lookback_window: int = 20):
        """
        Initialize market regime detector
        
        Args:
            method: Detection method ('hmm', 'threshold', 'volatility', 'mixture')
            n_regimes: Number of regimes to detect
            lookback_window: Window for regime indicators
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        
        # Regime models
        self.hmm_model = None
        self.threshold_model = None
        self.mixture_model = None
        
        # Regime history
        self.regime_history = []
        self.regime_indicators = {}
        
        # Regime definitions
        self.regime_definitions = self._initialize_regime_definitions()
        
        logger.info(f"Initialized regime detector with {method} method for {n_regimes} regimes")
    
    def _initialize_regime_definitions(self) -> Dict[int, Dict[str, Any]]:
        """Initialize standard regime definitions"""
        if self.n_regimes == 2:
            return {
                0: {'name': 'low_volatility', 'description': 'Stable market conditions'},
                1: {'name': 'high_volatility', 'description': 'Volatile market conditions'}
            }
        elif self.n_regimes == 3:
            return {
                0: {'name': 'bear_market', 'description': 'Declining market with high volatility'},
                1: {'name': 'stable_market', 'description': 'Stable market with low volatility'},
                2: {'name': 'bull_market', 'description': 'Growing market with moderate volatility'}
            }
        elif self.n_regimes == 4:
            return {
                0: {'name': 'crisis', 'description': 'Market crisis - high volatility, declining'},
                1: {'name': 'bear_market', 'description': 'Bear market - declining with high vol'},
                2: {'name': 'stable_market', 'description': 'Stable market - low volatility'},
                3: {'name': 'bull_market', 'description': 'Bull market - growing with low vol'}
            }
        else:
            return {i: {'name': f'regime_{i}', 'description': f'Market regime {i}'} 
                   for i in range(self.n_regimes)}
    
    def fit_regime_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit regime detection model to historical data
        
        Args:
            data: Historical market data with columns like 'value', 'return', 'volatility'
            
        Returns:
            Model fitting results
        """
        logger.info(f"Fitting {self.method} regime model")
        
        # Prepare features for regime detection
        features = self._prepare_regime_features(data)
        
        if self.method == 'hmm':
            return self._fit_hmm_model(features)
        elif self.method == 'threshold':
            return self._fit_threshold_model(features)
        elif self.method == 'volatility':
            return self._fit_volatility_model(features)
        elif self.method == 'mixture':
            return self._fit_mixture_model(features)
        else:
            raise ValueError(f"Unknown regime detection method: {self.method}")
    
    def _prepare_regime_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for regime detection"""
        features_list = []
        
        # Calculate returns if not present
        if 'return' not in data.columns and 'value' in data.columns:
            returns = data['value'].pct_change().dropna()
        else:
            returns = data['return'].dropna()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=self.lookback_window).std()
        
        # Calculate rolling mean return
        mean_return = returns.rolling(window=self.lookback_window).mean()
        
        # Calculate momentum
        momentum = data['value'].pct_change(periods=self.lookback_window) if 'value' in data.columns else returns
        
        # Combine features
        feature_df = pd.DataFrame({
            'return': returns,
            'volatility': volatility,
            'mean_return': mean_return,
            'momentum': momentum
        }).dropna()
        
        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(feature_df.values)
        
        self.feature_scaler = scaler
        self.feature_columns = feature_df.columns.tolist()
        
        return features
    
    def _fit_hmm_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Fit Hidden Markov Model for regime detection"""
        try:
            # Initialize HMM model
            self.hmm_model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            # Fit model
            self.hmm_model.fit(features)
            
            # Predict regime sequence
            regime_sequence = self.hmm_model.predict(features)
            
            # Calculate model statistics
            log_likelihood = self.hmm_model.score(features)
            aic = -2 * log_likelihood + 2 * self._count_hmm_parameters()
            
            results = {
                'method': 'hmm',
                'log_likelihood': log_likelihood,
                'aic': aic,
                'regime_sequence': regime_sequence,
                'transition_matrix': self.hmm_model.transmat_,
                'regime_means': self.hmm_model.means_,
                'regime_covariances': self.hmm_model.covars_
            }
            
            logger.info(f"HMM fitted: log-likelihood = {log_likelihood:.2f}, AIC = {aic:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting HMM model: {str(e)}")
            return {'method': 'hmm', 'error': str(e)}
    
    def _count_hmm_parameters(self) -> int:
        """Count number of parameters in HMM model"""
        n_states = self.n_regimes
        n_features = len(self.feature_columns)
        
        # Transition matrix parameters
        transition_params = n_states * (n_states - 1)
        
        # Mean parameters
        mean_params = n_states * n_features
        
        # Covariance parameters (full covariance)
        cov_params = n_states * n_features * (n_features + 1) // 2
        
        # Initial state probabilities
        initial_params = n_states - 1
        
        return transition_params + mean_params + cov_params + initial_params
    
    def _fit_threshold_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Fit threshold-based regime model"""
        # Use volatility as the main threshold variable
        volatility_idx = self.feature_columns.index('volatility') if 'volatility' in self.feature_columns else 0
        volatility = features[:, volatility_idx]
        
        # Determine thresholds based on percentiles
        if self.n_regimes == 2:
            threshold = np.percentile(volatility, 50)
            regime_sequence = (volatility > threshold).astype(int)
        elif self.n_regimes == 3:
            low_threshold = np.percentile(volatility, 33)
            high_threshold = np.percentile(volatility, 67)
            regime_sequence = np.zeros(len(volatility), dtype=int)
            regime_sequence[volatility > low_threshold] = 1
            regime_sequence[volatility > high_threshold] = 2
        else:
            # Multiple thresholds
            thresholds = [np.percentile(volatility, 100 * i / self.n_regimes) 
                         for i in range(1, self.n_regimes)]
            regime_sequence = np.digitize(volatility, thresholds)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regime_sequence)
        
        self.threshold_model = {
            'thresholds': thresholds if self.n_regimes > 2 else [threshold],
            'volatility_percentiles': [33, 67] if self.n_regimes == 3 else [50]
        }
        
        results = {
            'method': 'threshold',
            'regime_sequence': regime_sequence,
            'transition_matrix': transition_matrix,
            'thresholds': self.threshold_model['thresholds']
        }
        
        return results
    
    def _fit_volatility_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Fit volatility-based regime model"""
        # Similar to threshold but more sophisticated volatility analysis
        volatility_idx = self.feature_columns.index('volatility') if 'volatility' in self.feature_columns else 0
        return_idx = self.feature_columns.index('return') if 'return' in self.feature_columns else 1
        
        volatility = features[:, volatility_idx]
        returns = features[:, return_idx]
        
        # Classify regimes based on volatility and return characteristics
        high_vol_threshold = np.percentile(volatility, 75)
        low_vol_threshold = np.percentile(volatility, 25)
        
        positive_return_threshold = 0
        
        regime_sequence = np.zeros(len(volatility), dtype=int)
        
        if self.n_regimes == 3:
            # 0: Bear (high vol, negative returns)
            # 1: Stable (low vol)
            # 2: Bull (moderate vol, positive returns)
            
            bear_mask = (volatility > high_vol_threshold) & (returns < positive_return_threshold)
            bull_mask = (volatility <= high_vol_threshold) & (returns >= positive_return_threshold)
            stable_mask = volatility <= low_vol_threshold
            
            regime_sequence[bear_mask] = 0
            regime_sequence[bull_mask] = 2
            regime_sequence[stable_mask] = 1
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regime_sequence)
        
        results = {
            'method': 'volatility',
            'regime_sequence': regime_sequence,
            'transition_matrix': transition_matrix,
            'high_vol_threshold': high_vol_threshold,
            'low_vol_threshold': low_vol_threshold
        }
        
        return results
    
    def _fit_mixture_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Fit Gaussian mixture model for regime detection"""
        try:
            # Fit Gaussian Mixture Model
            self.mixture_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42
            )
            
            self.mixture_model.fit(features)
            
            # Predict regime sequence
            regime_sequence = self.mixture_model.predict(features)
            
            # Calculate model statistics
            log_likelihood = self.mixture_model.score(features) * len(features)
            aic = -2 * log_likelihood + 2 * self._count_mixture_parameters()
            
            # Calculate transition matrix
            transition_matrix = self._calculate_transition_matrix(regime_sequence)
            
            results = {
                'method': 'mixture',
                'log_likelihood': log_likelihood,
                'aic': aic,
                'regime_sequence': regime_sequence,
                'transition_matrix': transition_matrix,
                'regime_means': self.mixture_model.means_,
                'regime_covariances': self.mixture_model.covariances_,
                'regime_weights': self.mixture_model.weights_
            }
            
            logger.info(f"Mixture model fitted: log-likelihood = {log_likelihood:.2f}, AIC = {aic:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error fitting mixture model: {str(e)}")
            return {'method': 'mixture', 'error': str(e)}
    
    def _count_mixture_parameters(self) -> int:
        """Count number of parameters in mixture model"""
        n_components = self.n_regimes
        n_features = len(self.feature_columns)
        
        # Mean parameters
        mean_params = n_components * n_features
        
        # Covariance parameters
        cov_params = n_components * n_features * (n_features + 1) // 2
        
        # Mixture weights
        weight_params = n_components - 1
        
        return mean_params + cov_params + weight_params
    
    def _calculate_transition_matrix(self, regime_sequence: np.ndarray) -> np.ndarray:
        """Calculate empirical transition matrix from regime sequence"""
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regime_sequence) - 1):
            current_regime = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            transition_matrix[current_regime, next_regime] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1)
        for i in range(self.n_regimes):
            if row_sums[i] > 0:
                transition_matrix[i, :] /= row_sums[i]
        
        return transition_matrix
    
    def predict_current_regime(self, recent_data: pd.DataFrame) -> Tuple[int, float]:
        """
        Predict current market regime based on recent data
        
        Args:
            recent_data: Recent market data
            
        Returns:
            Tuple of (predicted_regime, confidence)
        """
        if self.method == 'hmm' and self.hmm_model is not None:
            features = self._prepare_regime_features(recent_data)
            if len(features) > 0:
                regime_probs = self.hmm_model.predict_proba(features[-1:])
                predicted_regime = np.argmax(regime_probs[0])
                confidence = np.max(regime_probs[0])
                return predicted_regime, confidence
        
        # Fallback: use simple volatility-based classification
        if 'value' in recent_data.columns:
            returns = recent_data['value'].pct_change().dropna()
            if len(returns) >= self.lookback_window:
                recent_volatility = returns.tail(self.lookback_window).std()
                
                # Simple classification
                if recent_volatility > returns.std() * 1.5:
                    return 0, 0.7  # High volatility regime
                else:
                    return 1, 0.7  # Low volatility regime
        
        return 0, 0.5  # Default regime with low confidence


class RegimeSwitchingDistribution:
    """
    Regime-switching parameter distribution
    
    Parameter distributions that change based on the current market regime.
    """
    
    def __init__(self, 
                 regime_distributions: Dict[int, Dict[str, Any]],
                 transition_matrix: np.ndarray,
                 initial_regime: int = 0):
        """
        Initialize regime-switching distribution
        
        Args:
            regime_distributions: Distribution parameters for each regime
            transition_matrix: Regime transition probabilities
            initial_regime: Starting regime
        """
        self.regime_distributions = regime_distributions
        self.transition_matrix = transition_matrix
        self.current_regime = initial_regime
        self.n_regimes = len(regime_distributions)
        
        # Validate transition matrix
        if transition_matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError("Transition matrix dimensions don't match number of regimes")
        
        # Create scipy distributions for each regime
        self.scipy_distributions = {}
        self._create_scipy_distributions()
        
        # Sampling history
        self.regime_history = [initial_regime]
        self.sampling_stats = {
            'total_samples': 0,
            'regime_counts': {i: 0 for i in range(self.n_regimes)},
            'regime_transitions': 0
        }
        
        logger.info(f"Initialized regime-switching distribution with {self.n_regimes} regimes")
    
    def _create_scipy_distributions(self):
        """Create scipy distribution objects for each regime"""
        for regime_id, dist_params in self.regime_distributions.items():
            dist_type = dist_params['type']
            params = dist_params['params']
            
            if dist_type == 'normal':
                self.scipy_distributions[regime_id] = stats.norm(
                    loc=params.get('mean', 0),
                    scale=params.get('std', 1)
                )
            elif dist_type == 'lognormal':
                self.scipy_distributions[regime_id] = stats.lognorm(
                    s=params.get('sigma', 1),
                    scale=np.exp(params.get('mu', 0))
                )
            elif dist_type == 'beta':
                self.scipy_distributions[regime_id] = stats.beta(
                    a=params.get('alpha', 1),
                    b=params.get('beta', 1)
                )
            elif dist_type == 'gamma':
                self.scipy_distributions[regime_id] = stats.gamma(
                    a=params.get('shape', 1),
                    scale=params.get('scale', 1)
                )
            elif dist_type == 'uniform':
                self.scipy_distributions[regime_id] = stats.uniform(
                    loc=params.get('low', 0),
                    scale=params.get('high', 1) - params.get('low', 0)
                )
            else:
                logger.warning(f"Unknown distribution type {dist_type} for regime {regime_id}")
                # Default to normal distribution
                self.scipy_distributions[regime_id] = stats.norm(0, 1)
    
    def sample(self, size: int = 1, regime_path: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample from regime-switching distribution
        
        Args:
            size: Number of samples
            regime_path: Optional predetermined regime path
            
        Returns:
            Array of samples
        """
        if regime_path is not None and len(regime_path) != size:
            raise ValueError("Regime path length must match sample size")
        
        samples = []
        current_regime = self.current_regime
        
        for i in range(size):
            # Determine regime for this sample
            if regime_path is not None:
                current_regime = regime_path[i]
            else:
                # Simulate regime transition
                if i > 0:  # After first sample
                    transition_probs = self.transition_matrix[current_regime]
                    current_regime = np.random.choice(self.n_regimes, p=transition_probs)
                    
                    if current_regime != self.regime_history[-1]:
                        self.sampling_stats['regime_transitions'] += 1
            
            # Sample from current regime distribution
            distribution = self.scipy_distributions[current_regime]
            sample = distribution.rvs()
            samples.append(sample)
            
            # Update statistics
            self.regime_history.append(current_regime)
            self.sampling_stats['regime_counts'][current_regime] += 1
        
        self.current_regime = current_regime
        self.sampling_stats['total_samples'] += size
        
        return np.array(samples)
    
    def simulate_regime_path(self, n_periods: int, current_regime: Optional[int] = None) -> np.ndarray:
        """
        Simulate future regime path
        
        Args:
            n_periods: Number of periods to simulate
            current_regime: Starting regime (uses current if None)
            
        Returns:
            Array of regime indices
        """
        if current_regime is None:
            current_regime = self.current_regime
        
        regime_path = [current_regime]
        
        for _ in range(n_periods - 1):
            transition_probs = self.transition_matrix[current_regime]
            next_regime = np.random.choice(self.n_regimes, p=transition_probs)
            regime_path.append(next_regime)
            current_regime = next_regime
        
        return np.array(regime_path)
    
    def calculate_stationary_distribution(self) -> np.ndarray:
        """Calculate stationary distribution of regimes"""
        # Find eigenvector corresponding to eigenvalue 1
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        
        # Find index of eigenvalue closest to 1
        stationary_idx = np.argmin(np.abs(eigenvals - 1))
        stationary_vec = np.real(eigenvecs[:, stationary_idx])
        
        # Normalize to get probabilities
        stationary_distribution = stationary_vec / np.sum(stationary_vec)
        
        return stationary_distribution
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics"""
        stationary_dist = self.calculate_stationary_distribution()
        
        # Calculate expected duration in each regime
        expected_durations = {}
        for i in range(self.n_regimes):
            if self.transition_matrix[i, i] < 1:
                expected_durations[i] = 1 / (1 - self.transition_matrix[i, i])
            else:
                expected_durations[i] = np.inf
        
        return {
            'current_regime': self.current_regime,
            'stationary_distribution': stationary_dist.tolist(),
            'expected_durations': expected_durations,
            'transition_matrix': self.transition_matrix.tolist(),
            'sampling_statistics': self.sampling_stats,
            'regime_definitions': {i: self.regime_distributions[i] for i in range(self.n_regimes)}
        }


class RegimeSwitchingMonteCarlo:
    """
    Main class for regime-switching Monte Carlo simulation
    
    Combines regime detection, regime-dependent distributions, and Monte Carlo sampling.
    """
    
    def __init__(self, 
                 regime_detector: MarketRegimeDetector,
                 parameter_distributions: Dict[str, RegimeSwitchingDistribution]):
        """
        Initialize regime-switching Monte Carlo
        
        Args:
            regime_detector: Market regime detection system
            parameter_distributions: Dictionary of regime-switching parameter distributions
        """
        self.regime_detector = regime_detector
        self.parameter_distributions = parameter_distributions
        self.parameter_names = list(parameter_distributions.keys())
        
        # Simulation results
        self.simulation_results = {}
        self.regime_scenarios = {}
        
        logger.info(f"Initialized regime-switching Monte Carlo with {len(self.parameter_names)} parameters")
    
    def run_regime_aware_simulation(self, 
                                  n_scenarios: int,
                                  n_periods: int,
                                  current_market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run regime-aware Monte Carlo simulation
        
        Args:
            n_scenarios: Number of scenarios to simulate
            n_periods: Number of time periods per scenario
            current_market_data: Current market data for regime detection
            
        Returns:
            Simulation results with regime information
        """
        logger.info(f"Running regime-aware simulation: {n_scenarios} scenarios, {n_periods} periods")
        
        # Detect current regime if market data provided
        current_regime = 0
        regime_confidence = 0.5
        
        if current_market_data is not None:
            try:
                current_regime, regime_confidence = self.regime_detector.predict_current_regime(current_market_data)
                logger.info(f"Detected current regime: {current_regime} (confidence: {regime_confidence:.2f})")
            except Exception as e:
                logger.warning(f"Could not detect current regime: {e}")
        
        # Run scenarios
        scenario_results = []
        regime_paths = []
        
        for scenario_idx in range(n_scenarios):
            # Simulate regime path for this scenario
            regime_path = self._simulate_scenario_regime_path(n_periods, current_regime)
            regime_paths.append(regime_path)
            
            # Simulate parameters for this scenario
            scenario_params = {}
            for param_name, regime_dist in self.parameter_distributions.items():
                # Sample parameters along the regime path
                param_samples = regime_dist.sample(n_periods, regime_path)
                scenario_params[param_name] = param_samples
            
            scenario_results.append({
                'scenario_id': scenario_idx,
                'regime_path': regime_path,
                'parameters': scenario_params
            })
        
        # Analyze results
        analysis_results = self._analyze_regime_simulation_results(scenario_results, regime_paths)
        
        # Store results
        simulation_results = {
            'scenarios': scenario_results,
            'regime_paths': regime_paths,
            'analysis': analysis_results,
            'current_regime': current_regime,
            'regime_confidence': regime_confidence,
            'simulation_metadata': {
                'n_scenarios': n_scenarios,
                'n_periods': n_periods,
                'n_parameters': len(self.parameter_names),
                'regime_detector_method': self.regime_detector.method
            }
        }
        
        self.simulation_results = simulation_results
        return simulation_results
    
    def _simulate_scenario_regime_path(self, n_periods: int, starting_regime: int) -> np.ndarray:
        """Simulate regime path for a single scenario"""
        # Get transition matrix from any parameter distribution (they should all have the same)
        if self.parameter_distributions:
            first_param = list(self.parameter_distributions.values())[0]
            regime_path = first_param.simulate_regime_path(n_periods, starting_regime)
        else:
            # Fallback: random regime path
            regime_path = np.random.choice(
                self.regime_detector.n_regimes, 
                size=n_periods
            )
        
        return regime_path
    
    def _analyze_regime_simulation_results(self, 
                                         scenario_results: List[Dict],
                                         regime_paths: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze simulation results across regimes"""
        analysis = {}
        
        # Regime frequency analysis
        all_regimes = np.concatenate(regime_paths)
        regime_frequencies = {}
        for regime in range(self.regime_detector.n_regimes):
            regime_frequencies[regime] = np.mean(all_regimes == regime)
        
        analysis['regime_frequencies'] = regime_frequencies
        
        # Regime duration analysis
        regime_durations = {regime: [] for regime in range(self.regime_detector.n_regimes)}
        
        for regime_path in regime_paths:
            current_regime = regime_path[0]
            duration = 1
            
            for i in range(1, len(regime_path)):
                if regime_path[i] == current_regime:
                    duration += 1
                else:
                    regime_durations[current_regime].append(duration)
                    current_regime = regime_path[i]
                    duration = 1
            
            # Add final duration
            regime_durations[current_regime].append(duration)
        
        # Calculate duration statistics
        duration_stats = {}
        for regime, durations in regime_durations.items():
            if durations:
                duration_stats[regime] = {
                    'mean_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'max_duration': np.max(durations),
                    'std_duration': np.std(durations)
                }
            else:
                duration_stats[regime] = {
                    'mean_duration': 0,
                    'median_duration': 0,
                    'max_duration': 0,
                    'std_duration': 0
                }
        
        analysis['regime_duration_stats'] = duration_stats
        
        # Parameter statistics by regime
        param_regime_stats = {}
        
        for param_name in self.parameter_names:
            param_regime_stats[param_name] = {}
            
            for regime in range(self.regime_detector.n_regimes):
                # Collect parameter values for this regime
                regime_values = []
                
                for scenario in scenario_results:
                    regime_path = scenario['regime_path']
                    param_values = scenario['parameters'][param_name]
                    
                    # Get values when in this regime
                    regime_mask = regime_path == regime
                    regime_values.extend(param_values[regime_mask])
                
                # Calculate statistics
                if regime_values:
                    param_regime_stats[param_name][regime] = {
                        'mean': np.mean(regime_values),
                        'std': np.std(regime_values),
                        'min': np.min(regime_values),
                        'max': np.max(regime_values),
                        'percentile_25': np.percentile(regime_values, 25),
                        'percentile_75': np.percentile(regime_values, 75),
                        'n_observations': len(regime_values)
                    }
                else:
                    param_regime_stats[param_name][regime] = {
                        'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                        'percentile_25': 0, 'percentile_75': 0,
                        'n_observations': 0
                    }
        
        analysis['parameter_regime_statistics'] = param_regime_stats
        
        # Transition analysis
        transition_counts = np.zeros((self.regime_detector.n_regimes, self.regime_detector.n_regimes))
        
        for regime_path in regime_paths:
            for i in range(len(regime_path) - 1):
                current = regime_path[i]
                next_regime = regime_path[i + 1]
                transition_counts[current, next_regime] += 1
        
        # Calculate empirical transition matrix
        empirical_transition_matrix = transition_counts.copy()
        row_sums = empirical_transition_matrix.sum(axis=1)
        for i in range(self.regime_detector.n_regimes):
            if row_sums[i] > 0:
                empirical_transition_matrix[i, :] /= row_sums[i]
        
        analysis['empirical_transition_matrix'] = empirical_transition_matrix.tolist()
        analysis['transition_counts'] = transition_counts.tolist()
        
        return analysis
    
    def stress_test_regimes(self, 
                          stress_regimes: List[int],
                          n_scenarios: int = 1000,
                          n_periods: int = 20) -> Dict[str, Any]:
        """
        Perform stress testing by forcing specific regime sequences
        
        Args:
            stress_regimes: List of regimes to stress test
            n_scenarios: Number of scenarios per stress regime
            n_periods: Number of periods per scenario
            
        Returns:
            Stress test results
        """
        logger.info(f"Running regime stress test for regimes: {stress_regimes}")
        
        stress_results = {}
        
        for stress_regime in stress_regimes:
            regime_name = self.regime_detector.regime_definitions[stress_regime]['name']
            logger.info(f"Stress testing regime {stress_regime} ({regime_name})")
            
            # Force all scenarios to stay in this regime
            forced_regime_paths = [np.full(n_periods, stress_regime) for _ in range(n_scenarios)]
            
            # Run simulation with forced regime paths
            scenario_results = []
            
            for scenario_idx in range(n_scenarios):
                regime_path = forced_regime_paths[scenario_idx]
                
                scenario_params = {}
                for param_name, regime_dist in self.parameter_distributions.items():
                    param_samples = regime_dist.sample(n_periods, regime_path)
                    scenario_params[param_name] = param_samples
                
                scenario_results.append({
                    'scenario_id': scenario_idx,
                    'regime_path': regime_path,
                    'parameters': scenario_params
                })
            
            # Analyze stress test results
            stress_analysis = self._analyze_regime_simulation_results(scenario_results, forced_regime_paths)
            
            stress_results[stress_regime] = {
                'regime_name': regime_name,
                'scenarios': scenario_results,
                'analysis': stress_analysis
            }
        
        return stress_results
    
    def export_regime_results(self, output_path: str) -> str:
        """Export regime simulation results to Excel"""
        if not self.simulation_results:
            logger.warning("No simulation results to export")
            return ""
        
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Regime frequency summary
                regime_freq = self.simulation_results['analysis']['regime_frequencies']
                freq_df = pd.DataFrame([regime_freq]).T
                freq_df.columns = ['Frequency']
                freq_df.index.name = 'Regime'
                freq_df.to_excel(writer, sheet_name='Regime_Frequencies')
                
                # Regime duration statistics
                duration_stats = self.simulation_results['analysis']['regime_duration_stats']
                duration_data = []
                for regime, stats in duration_stats.items():
                    row = {'Regime': regime}
                    row.update(stats)
                    duration_data.append(row)
                
                duration_df = pd.DataFrame(duration_data)
                duration_df.to_excel(writer, sheet_name='Regime_Durations', index=False)
                
                # Parameter statistics by regime
                param_stats = self.simulation_results['analysis']['parameter_regime_statistics']
                for param_name, regime_stats in param_stats.items():
                    param_data = []
                    for regime, stats in regime_stats.items():
                        row = {'Regime': regime}
                        row.update(stats)
                        param_data.append(row)
                    
                    param_df = pd.DataFrame(param_data)
                    sheet_name = f'Param_{param_name}'[:31]  # Excel sheet name limit
                    param_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Transition matrix
                transition_matrix = self.simulation_results['analysis']['empirical_transition_matrix']
                transition_df = pd.DataFrame(transition_matrix)
                transition_df.index.name = 'From_Regime'
                transition_df.columns.name = 'To_Regime'
                transition_df.to_excel(writer, sheet_name='Transition_Matrix')
            
            logger.info(f"Exported regime simulation results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting regime results: {str(e)}")
            return ""


# Integration and utility functions
def create_regime_switching_monte_carlo(historical_data: pd.DataFrame,
                                      parameter_configs: Dict[str, Dict],
                                      n_regimes: int = 3) -> RegimeSwitchingMonteCarlo:
    """
    Create regime-switching Monte Carlo system from historical data
    
    Args:
        historical_data: Historical market data
        parameter_configs: Parameter distribution configurations
        n_regimes: Number of regimes
        
    Returns:
        Configured regime-switching Monte Carlo system
    """
    # Initialize regime detector
    regime_detector = MarketRegimeDetector(method='hmm', n_regimes=n_regimes)
    
    # Fit regime model
    regime_model_results = regime_detector.fit_regime_model(historical_data)
    
    # Create regime-switching distributions for parameters
    parameter_distributions = {}
    
    for param_name, config in parameter_configs.items():
        # Create regime-specific distributions
        regime_distributions = {}
        transition_matrix = regime_model_results.get('transition_matrix', np.eye(n_regimes))
        
        for regime in range(n_regimes):
            # Modify distribution parameters based on regime
            regime_config = config.copy()
            
            # Example: modify parameters based on regime characteristics
            if regime == 0:  # High volatility regime
                if 'std' in regime_config.get('params', {}):
                    regime_config['params']['std'] *= 1.5
            elif regime == 2:  # Bull market regime
                if 'mean' in regime_config.get('params', {}):
                    regime_config['params']['mean'] *= 1.2
            
            regime_distributions[regime] = regime_config
        
        # Create regime-switching distribution
        rs_distribution = RegimeSwitchingDistribution(
            regime_distributions, 
            transition_matrix
        )
        
        parameter_distributions[param_name] = rs_distribution
    
    # Create regime-switching Monte Carlo system
    rs_monte_carlo = RegimeSwitchingMonteCarlo(regime_detector, parameter_distributions)
    
    return rs_monte_carlo


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Simulate regime-switching returns
    regime_changes = [0, 150, 300, 450]
    regimes = [0, 1, 0, 2]
    
    returns = []
    current_return = 0.001
    
    for i, date in enumerate(dates):
        # Determine current regime
        current_regime = 0
        for j, change_point in enumerate(regime_changes):
            if i >= change_point:
                current_regime = regimes[j]
        
        # Generate return based on regime
        if current_regime == 0:  # High volatility
            daily_return = np.random.normal(0, 0.02)
        elif current_regime == 1:  # Stable
            daily_return = np.random.normal(0.0005, 0.005)
        else:  # Bull market
            daily_return = np.random.normal(0.001, 0.01)
        
        returns.append(daily_return)
    
    # Create market data
    cumulative_returns = np.cumprod(1 + np.array(returns))
    market_data = pd.DataFrame({
        'date': dates,
        'value': 1000 * cumulative_returns,
        'return': returns
    })
    
    # Test regime detection
    regime_detector = MarketRegimeDetector(method='hmm', n_regimes=3)
    results = regime_detector.fit_regime_model(market_data)
    
    print("Regime Detection Results:")
    print(f"Method: {results['method']}")
    print(f"Log-likelihood: {results.get('log_likelihood', 'N/A')}")
    if 'transition_matrix' in results:
        print("Transition Matrix:")
        print(results['transition_matrix'])
    
    # Test regime-switching distribution
    regime_distributions = {
        0: {'type': 'normal', 'params': {'mean': 0.0, 'std': 0.02}},  # High vol
        1: {'type': 'normal', 'params': {'mean': 0.005, 'std': 0.005}},  # Stable
        2: {'type': 'normal', 'params': {'mean': 0.01, 'std': 0.01}}   # Bull
    }
    
    transition_matrix = results.get('transition_matrix', np.eye(3))
    rs_dist = RegimeSwitchingDistribution(regime_distributions, transition_matrix)
    
    # Generate samples
    samples = rs_dist.sample(100)
    regime_stats = rs_dist.get_regime_statistics()
    
    print(f"\nRegime-Switching Distribution:")
    print(f"Current regime: {regime_stats['current_regime']}")
    print(f"Stationary distribution: {regime_stats['stationary_distribution']}")
    print(f"Samples mean: {np.mean(samples):.4f}, std: {np.std(samples):.4f}")