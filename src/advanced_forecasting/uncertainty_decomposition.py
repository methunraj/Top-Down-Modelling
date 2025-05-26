"""
Uncertainty Decomposition - Epistemic vs Aleatory uncertainty separation

This module provides advanced uncertainty decomposition capabilities to separate:
1. Epistemic uncertainty (model/knowledge uncertainty)
2. Aleatory uncertainty (natural randomness/variability)
3. Parameter uncertainty
4. Structural uncertainty

Key capabilities:
- Bayesian uncertainty quantification
- Ensemble-based uncertainty decomposition
- Information-theoretic uncertainty measures
- Uncertainty propagation through forecast pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import warnings

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UncertaintyDecomposer:
    """
    Main uncertainty decomposition class
    
    Separates total uncertainty into different components and provides
    detailed analysis of uncertainty sources.
    """
    
    def __init__(self, 
                 ensemble_models: List[Any] = None,
                 n_bootstrap: int = 100):
        """
        Initialize uncertainty decomposer
        
        Args:
            ensemble_models: List of different forecasting models
            n_bootstrap: Number of bootstrap samples for uncertainty estimation
        """
        self.ensemble_models = ensemble_models or []
        self.n_bootstrap = n_bootstrap
        
        # Uncertainty components
        self.epistemic_uncertainty = {}
        self.aleatory_uncertainty = {}
        self.parameter_uncertainty = {}
        self.structural_uncertainty = {}
        
        # Analysis results
        self.decomposition_results = {}
        self.uncertainty_attribution = {}
        
        logger.info(f"Initialized uncertainty decomposer with {len(self.ensemble_models)} models")
    
    def decompose_forecast_uncertainty(self,
                                     forecast_ensemble: List[np.ndarray],
                                     data_samples: List[np.ndarray],
                                     parameter_samples: Optional[List[Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Decompose forecast uncertainty into components
        
        Args:
            forecast_ensemble: List of forecast arrays from different models/samples
            data_samples: List of data samples used for forecasting
            parameter_samples: Optional parameter uncertainty samples
            
        Returns:
            Dictionary with uncertainty decomposition
        """
        logger.info("Decomposing forecast uncertainty into components")
        
        # Convert to numpy arrays for easier manipulation
        forecast_matrix = np.array(forecast_ensemble)  # [n_models, n_periods]
        
        # Calculate total uncertainty
        total_uncertainty = self._calculate_total_uncertainty(forecast_matrix)
        
        # Epistemic uncertainty (model disagreement)
        epistemic_uncertainty = self._calculate_epistemic_uncertainty(forecast_matrix)
        
        # Aleatory uncertainty (inherent randomness)
        aleatory_uncertainty = self._calculate_aleatory_uncertainty(forecast_matrix, data_samples)
        
        # Parameter uncertainty (if provided)
        parameter_uncertainty = np.zeros_like(total_uncertainty)
        if parameter_samples:
            parameter_uncertainty = self._calculate_parameter_uncertainty(parameter_samples)
        
        # Structural uncertainty (model structure differences)
        structural_uncertainty = self._calculate_structural_uncertainty(forecast_ensemble)
        
        # Validate decomposition
        decomposition_check = self._validate_decomposition(
            total_uncertainty, epistemic_uncertainty, aleatory_uncertainty,
            parameter_uncertainty, structural_uncertainty
        )
        
        # Calculate uncertainty ratios
        uncertainty_ratios = self._calculate_uncertainty_ratios(
            total_uncertainty, epistemic_uncertainty, aleatory_uncertainty,
            parameter_uncertainty, structural_uncertainty
        )
        
        # Compile results
        decomposition_results = {
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatory_uncertainty': aleatory_uncertainty,
            'parameter_uncertainty': parameter_uncertainty,
            'structural_uncertainty': structural_uncertainty,
            'uncertainty_ratios': uncertainty_ratios,
            'decomposition_check': decomposition_check,
            'methodology': {
                'n_models': len(forecast_ensemble),
                'n_periods': len(total_uncertainty),
                'decomposition_method': 'ensemble_based'
            }
        }
        
        self.decomposition_results = decomposition_results
        return decomposition_results
    
    def _calculate_total_uncertainty(self, forecast_matrix: np.ndarray) -> np.ndarray:
        """Calculate total forecast uncertainty"""
        # Total uncertainty as variance across all forecasts
        return np.var(forecast_matrix, axis=0)
    
    def _calculate_epistemic_uncertainty(self, forecast_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate epistemic uncertainty (model disagreement)
        
        Epistemic uncertainty represents our lack of knowledge about the true model.
        It's measured by the disagreement between different models/approaches.
        """
        # Model disagreement: variance of model means
        model_means = np.mean(forecast_matrix, axis=1, keepdims=True)
        overall_mean = np.mean(forecast_matrix)
        
        # Epistemic uncertainty as variance of model predictions
        epistemic = np.var(np.mean(forecast_matrix, axis=1))
        
        # Broadcast to match forecast periods
        epistemic_uncertainty = np.full(forecast_matrix.shape[1], epistemic)
        
        return epistemic_uncertainty
    
    def _calculate_aleatory_uncertainty(self, 
                                      forecast_matrix: np.ndarray,
                                      data_samples: List[np.ndarray]) -> np.ndarray:
        """
        Calculate aleatory uncertainty (natural variability)
        
        Aleatory uncertainty represents the inherent randomness in the system
        that cannot be reduced by gathering more information.
        """
        # Average within-model variance
        within_model_variances = []
        
        # For each model, calculate variance across different data samples
        n_models = len(forecast_matrix)
        samples_per_model = len(data_samples) // n_models if data_samples else 1
        
        for i in range(n_models):
            start_idx = i * samples_per_model
            end_idx = start_idx + samples_per_model
            
            if end_idx <= len(data_samples):
                model_forecasts = forecast_matrix[start_idx:end_idx]
                if len(model_forecasts) > 1:
                    within_variance = np.var(model_forecasts, axis=0)
                    within_model_variances.append(within_variance)
        
        if within_model_variances:
            aleatory_uncertainty = np.mean(within_model_variances, axis=0)
        else:
            # Fallback: estimate from forecast residuals
            aleatory_uncertainty = np.var(forecast_matrix, axis=0) * 0.3  # Rough approximation
        
        return aleatory_uncertainty
    
    def _calculate_parameter_uncertainty(self, parameter_samples: List[Dict[str, float]]) -> np.ndarray:
        """
        Calculate uncertainty due to parameter estimation
        
        Parameter uncertainty arises from uncertainty in estimated model parameters.
        """
        if not parameter_samples:
            return np.array([0.0])
        
        # Convert parameter samples to matrix
        param_names = list(parameter_samples[0].keys())
        param_matrix = np.array([[sample[name] for name in param_names] 
                                for sample in parameter_samples])
        
        # Calculate parameter variance (simplified measure)
        param_variance = np.var(param_matrix, axis=0)
        parameter_uncertainty_magnitude = np.mean(param_variance)
        
        # Return scalar value (would need model-specific mapping for time series)
        return np.array([parameter_uncertainty_magnitude])
    
    def _calculate_structural_uncertainty(self, forecast_ensemble: List[np.ndarray]) -> np.ndarray:
        """
        Calculate structural uncertainty (model structure differences)
        
        Structural uncertainty arises from uncertainty about the correct model structure.
        """
        if len(forecast_ensemble) < 2:
            return np.zeros(len(forecast_ensemble[0]))
        
        forecast_matrix = np.array(forecast_ensemble)
        
        # Structural uncertainty as spread of model types
        # This is simplified - in practice would categorize models by structure
        model_medians = np.median(forecast_matrix, axis=1)
        structural_spread = np.var(model_medians)
        
        # Broadcast to forecast periods
        structural_uncertainty = np.full(forecast_matrix.shape[1], structural_spread)
        
        return structural_uncertainty
    
    def _validate_decomposition(self, 
                               total: np.ndarray,
                               epistemic: np.ndarray,
                               aleatory: np.ndarray,
                               parameter: np.ndarray,
                               structural: np.ndarray) -> Dict[str, Any]:
        """Validate uncertainty decomposition"""
        # Check if components sum approximately to total
        if len(parameter) == 1:
            parameter = np.full_like(total, parameter[0])
        if len(structural) != len(total):
            structural = np.full_like(total, np.mean(structural))
        
        component_sum = epistemic + aleatory + parameter + structural
        
        # Calculate relative error
        relative_error = np.abs(component_sum - total) / (total + 1e-8)
        
        return {
            'components_sum_to_total': np.allclose(component_sum, total, rtol=0.1),
            'max_relative_error': np.max(relative_error),
            'mean_relative_error': np.mean(relative_error),
            'decomposition_quality': 'good' if np.max(relative_error) < 0.2 else 'poor'
        }
    
    def _calculate_uncertainty_ratios(self,
                                    total: np.ndarray,
                                    epistemic: np.ndarray,
                                    aleatory: np.ndarray,
                                    parameter: np.ndarray,
                                    structural: np.ndarray) -> Dict[str, float]:
        """Calculate relative importance of uncertainty components"""
        # Ensure all arrays have same length
        if len(parameter) == 1:
            parameter = np.full_like(total, parameter[0])
        if len(structural) != len(total):
            structural = np.full_like(total, np.mean(structural))
        
        total_sum = np.mean(total)
        
        if total_sum > 0:
            ratios = {
                'epistemic_ratio': np.mean(epistemic) / total_sum,
                'aleatory_ratio': np.mean(aleatory) / total_sum,
                'parameter_ratio': np.mean(parameter) / total_sum,
                'structural_ratio': np.mean(structural) / total_sum
            }
        else:
            ratios = {
                'epistemic_ratio': 0.0,
                'aleatory_ratio': 0.0,
                'parameter_ratio': 0.0,
                'structural_ratio': 0.0
            }
        
        # Add interpretations
        ratios['dominant_uncertainty'] = max(ratios.keys(), key=ratios.get)
        ratios['reducible_uncertainty'] = ratios['epistemic_ratio'] + ratios['parameter_ratio']
        ratios['irreducible_uncertainty'] = ratios['aleatory_ratio']
        
        return ratios


class BayesianUncertaintyQuantifier:
    """
    Bayesian approach to uncertainty quantification
    
    Uses Bayesian inference to separate different types of uncertainty
    and provide credible intervals.
    """
    
    def __init__(self, 
                 prior_distributions: Dict[str, Any] = None,
                 n_samples: int = 1000):
        """
        Initialize Bayesian uncertainty quantifier
        
        Args:
            prior_distributions: Prior distributions for parameters
            n_samples: Number of MCMC samples
        """
        self.prior_distributions = prior_distributions or {}
        self.n_samples = n_samples
        
        # Posterior samples
        self.posterior_samples = {}
        self.predictive_samples = {}
        
        # Uncertainty estimates
        self.bayesian_uncertainty = {}
        
        logger.info(f"Initialized Bayesian uncertainty quantifier with {n_samples} samples")
    
    def estimate_bayesian_uncertainty(self,
                                    model_function: Callable,
                                    observed_data: np.ndarray,
                                    parameter_names: List[str]) -> Dict[str, Any]:
        """
        Estimate uncertainty using Bayesian inference
        
        Args:
            model_function: Model function to evaluate
            observed_data: Observed data for inference
            parameter_names: Names of parameters to infer
            
        Returns:
            Bayesian uncertainty estimates
        """
        logger.info("Estimating Bayesian uncertainty")
        
        # Simplified Bayesian inference (in practice, use PyMC, Stan, etc.)
        posterior_samples = self._sample_posterior(model_function, observed_data, parameter_names)
        
        # Calculate predictive uncertainty
        predictive_uncertainty = self._calculate_predictive_uncertainty(
            model_function, posterior_samples
        )
        
        # Decompose uncertainty
        epistemic_bayes = self._calculate_bayesian_epistemic_uncertainty(posterior_samples)
        aleatory_bayes = self._calculate_bayesian_aleatory_uncertainty(
            model_function, posterior_samples, observed_data
        )
        
        bayesian_results = {
            'posterior_samples': posterior_samples,
            'predictive_uncertainty': predictive_uncertainty,
            'epistemic_uncertainty': epistemic_bayes,
            'aleatory_uncertainty': aleatory_bayes,
            'credible_intervals': self._calculate_credible_intervals(posterior_samples),
            'uncertainty_reduction': self._calculate_uncertainty_reduction(
                self.prior_distributions, posterior_samples
            )
        }
        
        self.bayesian_uncertainty = bayesian_results
        return bayesian_results
    
    def _sample_posterior(self,
                         model_function: Callable,
                         observed_data: np.ndarray,
                         parameter_names: List[str]) -> Dict[str, np.ndarray]:
        """Sample from posterior distribution (simplified implementation)"""
        # This is a simplified implementation
        # In practice, use proper MCMC (PyMC, Stan, etc.)
        
        posterior_samples = {}
        
        for param_name in parameter_names:
            # Get prior distribution
            prior = self.prior_distributions.get(param_name, stats.norm(0, 1))
            
            # Simple random walk Metropolis (very simplified)
            samples = []
            current_param = prior.rvs()
            
            for _ in range(self.n_samples):
                # Propose new parameter
                proposal = current_param + np.random.normal(0, 0.1)
                
                # Calculate likelihood ratio (simplified)
                try:
                    current_likelihood = self._calculate_likelihood(
                        model_function, {param_name: current_param}, observed_data
                    )
                    proposal_likelihood = self._calculate_likelihood(
                        model_function, {param_name: proposal}, observed_data
                    )
                    
                    # Accept/reject
                    if np.random.random() < min(1, proposal_likelihood / current_likelihood):
                        current_param = proposal
                except:
                    pass  # Keep current parameter
                
                samples.append(current_param)
            
            posterior_samples[param_name] = np.array(samples)
        
        return posterior_samples
    
    def _calculate_likelihood(self,
                            model_function: Callable,
                            parameters: Dict[str, float],
                            observed_data: np.ndarray) -> float:
        """Calculate likelihood of parameters given data"""
        try:
            model_prediction = model_function(parameters)
            
            # Simple Gaussian likelihood
            residuals = observed_data - model_prediction
            likelihood = np.exp(-0.5 * np.sum(residuals**2))
            
            return likelihood
        except:
            return 1e-10  # Very small likelihood for invalid parameters
    
    def _calculate_predictive_uncertainty(self,
                                        model_function: Callable,
                                        posterior_samples: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate predictive uncertainty from posterior samples"""
        predictions = []
        
        n_samples = len(list(posterior_samples.values())[0])
        
        for i in range(min(100, n_samples)):  # Limit for computational efficiency
            param_sample = {name: samples[i] for name, samples in posterior_samples.items()}
            
            try:
                prediction = model_function(param_sample)
                predictions.append(prediction)
            except:
                continue
        
        if predictions:
            return np.var(predictions, axis=0)
        else:
            return np.array([0.0])
    
    def _calculate_bayesian_epistemic_uncertainty(self,
                                                posterior_samples: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate epistemic uncertainty from posterior parameter uncertainty"""
        # Epistemic uncertainty from parameter posterior variance
        param_variances = []
        
        for param_name, samples in posterior_samples.items():
            param_variances.append(np.var(samples))
        
        # Aggregate parameter uncertainty
        epistemic_magnitude = np.mean(param_variances)
        
        return np.array([epistemic_magnitude])
    
    def _calculate_bayesian_aleatory_uncertainty(self,
                                               model_function: Callable,
                                               posterior_samples: Dict[str, np.ndarray],
                                               observed_data: np.ndarray) -> np.ndarray:
        """Calculate aleatory uncertainty from observation noise"""
        # Estimate observation noise from residuals
        param_means = {name: np.mean(samples) for name, samples in posterior_samples.items()}
        
        try:
            mean_prediction = model_function(param_means)
            residuals = observed_data - mean_prediction
            aleatory_variance = np.var(residuals)
        except:
            aleatory_variance = 0.1  # Default estimate
        
        return np.array([aleatory_variance])
    
    def _calculate_credible_intervals(self,
                                    posterior_samples: Dict[str, np.ndarray],
                                    confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate credible intervals for parameters"""
        alpha = 1 - confidence_level
        
        credible_intervals = {}
        
        for param_name, samples in posterior_samples.items():
            lower = np.percentile(samples, 100 * alpha / 2)
            upper = np.percentile(samples, 100 * (1 - alpha / 2))
            credible_intervals[param_name] = (lower, upper)
        
        return credible_intervals
    
    def _calculate_uncertainty_reduction(self,
                                       priors: Dict[str, Any],
                                       posterior_samples: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate uncertainty reduction from prior to posterior"""
        uncertainty_reduction = {}
        
        for param_name, samples in posterior_samples.items():
            posterior_var = np.var(samples)
            
            if param_name in priors:
                prior = priors[param_name]
                if hasattr(prior, 'var'):
                    prior_var = prior.var()
                    reduction = 1 - posterior_var / prior_var
                    uncertainty_reduction[param_name] = max(0, reduction)
                else:
                    uncertainty_reduction[param_name] = 0.0
            else:
                uncertainty_reduction[param_name] = 0.0
        
        return uncertainty_reduction


class InformationTheoreticUncertainty:
    """
    Information-theoretic measures of uncertainty
    
    Uses entropy, mutual information, and other information measures
    to quantify different aspects of uncertainty.
    """
    
    def __init__(self):
        """Initialize information-theoretic uncertainty analyzer"""
        self.entropy_measures = {}
        self.mutual_information = {}
        
        logger.info("Initialized information-theoretic uncertainty analyzer")
    
    def calculate_entropy_measures(self,
                                 forecast_distributions: List[np.ndarray],
                                 parameter_distributions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate various entropy measures
        
        Args:
            forecast_distributions: List of forecast probability distributions
            parameter_distributions: Parameter uncertainty distributions
            
        Returns:
            Dictionary of entropy measures
        """
        logger.info("Calculating information-theoretic uncertainty measures")
        
        entropy_results = {}
        
        # Forecast entropy (predictive uncertainty)
        if forecast_distributions:
            forecast_entropy = self._calculate_differential_entropy(forecast_distributions)
            entropy_results['forecast_entropy'] = forecast_entropy
        
        # Parameter entropy
        parameter_entropies = {}
        for param_name, param_dist in parameter_distributions.items():
            param_entropy = self._calculate_differential_entropy([param_dist])
            parameter_entropies[param_name] = param_entropy
        
        entropy_results['parameter_entropies'] = parameter_entropies
        
        # Total entropy
        total_entropy = forecast_entropy + sum(parameter_entropies.values())
        entropy_results['total_entropy'] = total_entropy
        
        # Mutual information between parameters
        if len(parameter_distributions) > 1:
            param_names = list(parameter_distributions.keys())
            mutual_info = {}
            
            for i, param1 in enumerate(param_names):
                for j, param2 in enumerate(param_names[i+1:], i+1):
                    mi = self._calculate_mutual_information(
                        parameter_distributions[param1],
                        parameter_distributions[param2]
                    )
                    mutual_info[f"{param1}_{param2}"] = mi
            
            entropy_results['parameter_mutual_information'] = mutual_info
        
        # Information gain measures
        entropy_results['information_measures'] = self._calculate_information_measures(
            entropy_results
        )
        
        self.entropy_measures = entropy_results
        return entropy_results
    
    def _calculate_differential_entropy(self, distributions: List[np.ndarray]) -> float:
        """Calculate differential entropy for continuous distributions"""
        if not distributions:
            return 0.0
        
        # Combine all distributions
        combined_data = np.concatenate(distributions)
        
        # Estimate entropy using histogram
        n_bins = min(50, int(np.sqrt(len(combined_data))))
        hist, bin_edges = np.histogram(combined_data, bins=n_bins, density=True)
        
        # Calculate bin width
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Calculate differential entropy
        # H = -∫ p(x) log p(x) dx ≈ -Σ p(xi) log p(xi) * Δx
        entropy = 0.0
        for p in hist:
            if p > 0:
                entropy -= p * np.log(p) * bin_width
        
        return entropy
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate mutual information between two variables"""
        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
        
        # Discretize for mutual information calculation
        n_bins = int(np.sqrt(len(x)))
        
        # Create 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
        hist_2d = hist_2d + 1e-10  # Add small constant to avoid log(0)
        
        # Normalize to get joint probability
        p_xy = hist_2d / np.sum(hist_2d)
        
        # Calculate marginal probabilities
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)
        
        # Calculate mutual information
        mutual_info = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mutual_info += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mutual_info
    
    def _calculate_information_measures(self, entropy_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional information-theoretic measures"""
        measures = {}
        
        # Normalized entropy measures
        forecast_entropy = entropy_results.get('forecast_entropy', 0)
        total_entropy = entropy_results.get('total_entropy', 1)
        
        if total_entropy > 0:
            measures['normalized_forecast_entropy'] = forecast_entropy / total_entropy
        else:
            measures['normalized_forecast_entropy'] = 0.0
        
        # Parameter entropy contribution
        param_entropies = entropy_results.get('parameter_entropies', {})
        if param_entropies and total_entropy > 0:
            for param_name, entropy in param_entropies.items():
                measures[f'{param_name}_entropy_contribution'] = entropy / total_entropy
        
        # Information redundancy (from mutual information)
        mutual_info = entropy_results.get('parameter_mutual_information', {})
        if mutual_info:
            total_mutual_info = sum(mutual_info.values())
            measures['parameter_redundancy'] = total_mutual_info
        else:
            measures['parameter_redundancy'] = 0.0
        
        return measures


class UncertaintyPropagator:
    """
    Uncertainty propagation through forecasting pipeline
    
    Tracks how uncertainty flows through different stages of the forecasting process.
    """
    
    def __init__(self):
        """Initialize uncertainty propagator"""
        self.propagation_stages = {}
        self.uncertainty_flow = {}
        
        logger.info("Initialized uncertainty propagator")
    
    def trace_uncertainty_propagation(self,
                                    pipeline_stages: List[Callable],
                                    input_uncertainty: Dict[str, np.ndarray],
                                    stage_names: List[str]) -> Dict[str, Any]:
        """
        Trace uncertainty propagation through pipeline stages
        
        Args:
            pipeline_stages: List of pipeline stage functions
            input_uncertainty: Initial uncertainty at each input
            stage_names: Names of pipeline stages
            
        Returns:
            Uncertainty propagation analysis
        """
        logger.info(f"Tracing uncertainty through {len(pipeline_stages)} pipeline stages")
        
        propagation_results = {}
        current_uncertainty = input_uncertainty.copy()
        
        for i, (stage_func, stage_name) in enumerate(zip(pipeline_stages, stage_names)):
            logger.debug(f"Processing stage {i+1}: {stage_name}")
            
            # Propagate uncertainty through this stage
            stage_output_uncertainty = self._propagate_through_stage(
                stage_func, current_uncertainty
            )
            
            # Calculate uncertainty amplification/reduction
            amplification = self._calculate_uncertainty_amplification(
                current_uncertainty, stage_output_uncertainty
            )
            
            # Store stage results
            propagation_results[stage_name] = {
                'input_uncertainty': current_uncertainty.copy(),
                'output_uncertainty': stage_output_uncertainty,
                'amplification_factor': amplification,
                'uncertainty_change': self._calculate_uncertainty_change(
                    current_uncertainty, stage_output_uncertainty
                )
            }
            
            # Update current uncertainty for next stage
            current_uncertainty = stage_output_uncertainty
        
        # Calculate overall propagation statistics
        overall_stats = self._calculate_overall_propagation_stats(propagation_results)
        
        final_results = {
            'stage_by_stage': propagation_results,
            'overall_statistics': overall_stats,
            'uncertainty_path': self._extract_uncertainty_path(propagation_results),
            'critical_stages': self._identify_critical_stages(propagation_results)
        }
        
        self.uncertainty_flow = final_results
        return final_results
    
    def _propagate_through_stage(self,
                               stage_func: Callable,
                               input_uncertainty: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Propagate uncertainty through a single pipeline stage"""
        # Simplified uncertainty propagation
        # In practice, this would depend on the specific stage function
        
        output_uncertainty = {}
        
        for input_name, uncertainty_values in input_uncertainty.items():
            try:
                # Apply stage function to uncertainty (simplified)
                # This would need to be customized for each stage type
                if callable(stage_func):
                    # Assume linear propagation as approximation
                    output_uncertainty[input_name] = uncertainty_values * 1.1
                else:
                    output_uncertainty[input_name] = uncertainty_values
                    
            except Exception as e:
                logger.warning(f"Error propagating uncertainty for {input_name}: {e}")
                output_uncertainty[input_name] = uncertainty_values
        
        return output_uncertainty
    
    def _calculate_uncertainty_amplification(self,
                                           input_unc: Dict[str, np.ndarray],
                                           output_unc: Dict[str, np.ndarray]) -> float:
        """Calculate uncertainty amplification factor"""
        input_total = sum(np.mean(unc) for unc in input_unc.values())
        output_total = sum(np.mean(unc) for unc in output_unc.values())
        
        if input_total > 0:
            return output_total / input_total
        else:
            return 1.0
    
    def _calculate_uncertainty_change(self,
                                    input_unc: Dict[str, np.ndarray],
                                    output_unc: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate uncertainty change for each component"""
        changes = {}
        
        for name in input_unc.keys():
            if name in output_unc:
                input_mean = np.mean(input_unc[name])
                output_mean = np.mean(output_unc[name])
                
                if input_mean > 0:
                    change = (output_mean - input_mean) / input_mean
                else:
                    change = 0.0
                
                changes[name] = change
        
        return changes
    
    def _calculate_overall_propagation_stats(self, propagation_results: Dict) -> Dict[str, Any]:
        """Calculate overall propagation statistics"""
        amplification_factors = [stage_data['amplification_factor'] 
                               for stage_data in propagation_results.values()]
        
        return {
            'total_amplification': np.prod(amplification_factors),
            'average_amplification': np.mean(amplification_factors),
            'max_amplification': np.max(amplification_factors),
            'amplification_variance': np.var(amplification_factors),
            'most_amplifying_stage': max(propagation_results.keys(), 
                                       key=lambda k: propagation_results[k]['amplification_factor'])
        }
    
    def _extract_uncertainty_path(self, propagation_results: Dict) -> List[float]:
        """Extract uncertainty path through pipeline"""
        uncertainty_path = []
        
        for stage_name, stage_data in propagation_results.items():
            input_uncertainty = stage_data['input_uncertainty']
            total_input_unc = sum(np.mean(unc) for unc in input_uncertainty.values())
            uncertainty_path.append(total_input_unc)
        
        # Add final output uncertainty
        if propagation_results:
            final_stage = list(propagation_results.values())[-1]
            output_uncertainty = final_stage['output_uncertainty']
            total_output_unc = sum(np.mean(unc) for unc in output_uncertainty.values())
            uncertainty_path.append(total_output_unc)
        
        return uncertainty_path
    
    def _identify_critical_stages(self, propagation_results: Dict) -> List[str]:
        """Identify stages that contribute most to uncertainty"""
        stage_contributions = {}
        
        for stage_name, stage_data in propagation_results.items():
            amplification = stage_data['amplification_factor']
            stage_contributions[stage_name] = amplification
        
        # Sort by amplification factor
        sorted_stages = sorted(stage_contributions.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Return top contributing stages
        critical_stages = [stage for stage, _ in sorted_stages[:3]]
        
        return critical_stages


# Integration functions
def create_enhanced_uncertainty_analysis(forecast_ensemble: List[np.ndarray],
                                        parameter_samples: List[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create comprehensive uncertainty analysis combining all methods
    
    Args:
        forecast_ensemble: Ensemble of forecasts
        parameter_samples: Parameter uncertainty samples
        
    Returns:
        Combined uncertainty analysis
    """
    # Initialize analyzers
    decomposer = UncertaintyDecomposer()
    entropy_analyzer = InformationTheoreticUncertainty()
    
    # Perform uncertainty decomposition
    decomposition = decomposer.decompose_forecast_uncertainty(
        forecast_ensemble, [], parameter_samples
    )
    
    # Calculate information-theoretic measures
    if parameter_samples:
        param_distributions = {}
        for param_name in parameter_samples[0].keys():
            param_values = np.array([sample[param_name] for sample in parameter_samples])
            param_distributions[param_name] = param_values
        
        entropy_measures = entropy_analyzer.calculate_entropy_measures(
            forecast_ensemble, param_distributions
        )
    else:
        entropy_measures = {}
    
    # Combine results
    enhanced_analysis = {
        'uncertainty_decomposition': decomposition,
        'information_measures': entropy_measures,
        'summary': {
            'dominant_uncertainty_type': decomposition['uncertainty_ratios']['dominant_uncertainty'],
            'reducible_uncertainty_fraction': decomposition['uncertainty_ratios']['reducible_uncertainty'],
            'total_information_content': entropy_measures.get('total_entropy', 0),
            'analysis_quality': decomposition['decomposition_check']['decomposition_quality']
        }
    }
    
    return enhanced_analysis


# Example usage
if __name__ == "__main__":
    # Create example forecast ensemble
    np.random.seed(42)
    n_models = 5
    n_periods = 10
    
    forecast_ensemble = []
    for i in range(n_models):
        # Generate different forecast patterns
        base_trend = np.linspace(100, 120, n_periods)
        noise = np.random.normal(0, 2 + i, n_periods)
        model_bias = i * 0.5
        
        forecast = base_trend + noise + model_bias
        forecast_ensemble.append(forecast)
    
    # Create parameter samples
    parameter_samples = []
    for _ in range(100):
        sample = {
            'growth_rate': np.random.normal(0.05, 0.01),
            'volatility': np.random.gamma(2, 0.02),
            'trend_strength': np.random.beta(2, 2)
        }
        parameter_samples.append(sample)
    
    # Perform uncertainty analysis
    analysis = create_enhanced_uncertainty_analysis(forecast_ensemble, parameter_samples)
    
    print("Enhanced Uncertainty Analysis Results:")
    print("=" * 50)
    
    decomp = analysis['uncertainty_decomposition']
    print(f"Dominant uncertainty type: {analysis['summary']['dominant_uncertainty_type']}")
    print(f"Reducible uncertainty: {analysis['summary']['reducible_uncertainty_fraction']:.1%}")
    print(f"Decomposition quality: {analysis['summary']['analysis_quality']}")
    
    print("\nUncertainty Ratios:")
    for unc_type, ratio in decomp['uncertainty_ratios'].items():
        if unc_type.endswith('_ratio'):
            unc_name = unc_type.replace('_ratio', '').replace('_', ' ').title()
            print(f"  {unc_name}: {ratio:.1%}")
    
    if 'information_measures' in analysis and analysis['information_measures']:
        print(f"\nTotal Information Content: {analysis['summary']['total_information_content']:.2f}")