"""
Adaptive Monte Carlo Engine - Real-time learning and optimization

This module implements adaptive Monte Carlo methods that learn from forecast
accuracy and continuously improve their sampling strategies and parameter
distributions.

Key capabilities:
1. Online learning from forecast errors
2. Adaptive parameter distribution updates
3. Dynamic sample size optimization
4. Performance-based method selection
5. Real-time convergence monitoring
6. Intelligent restart mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from collections import deque
import warnings

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveSamplingOptimizer:
    """
    Adaptive sampling optimizer that learns optimal sampling strategies
    
    Uses reinforcement learning principles to optimize Monte Carlo sampling
    based on forecast performance feedback.
    """
    
    def __init__(self, 
                 initial_sample_size: int = 1000,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.2):
        """
        Initialize adaptive sampling optimizer
        
        Args:
            initial_sample_size: Starting number of samples
            learning_rate: Learning rate for parameter updates
            exploration_rate: Rate of exploration vs exploitation
        """
        self.current_sample_size = initial_sample_size
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.sample_size_history = deque(maxlen=100)
        self.strategy_rewards = {}
        
        # Adaptive parameters
        self.optimal_sample_size = initial_sample_size
        self.convergence_threshold = 0.001
        self.max_sample_size = 50000
        self.min_sample_size = 500
        
        # Strategy statistics
        self.strategy_stats = {
            'total_optimizations': 0,
            'successful_adaptations': 0,
            'average_performance_improvement': 0.0
        }
        
        logger.info(f"Initialized adaptive sampling optimizer with {initial_sample_size} initial samples")
    
    def optimize_sample_size(self, 
                           recent_performance: List[float],
                           computational_budget: float = 1.0) -> int:
        """
        Optimize sample size based on recent performance
        
        Args:
            recent_performance: Recent forecast accuracy measures
            computational_budget: Available computational budget (0-1 scale)
            
        Returns:
            Optimized sample size
        """
        if not recent_performance:
            return self.current_sample_size
        
        current_performance = np.mean(recent_performance[-5:])  # Recent average
        
        # Store performance and sample size
        self.performance_history.append(current_performance)
        self.sample_size_history.append(self.current_sample_size)
        
        # Calculate performance trend
        if len(self.performance_history) >= 3:
            performance_trend = self._calculate_performance_trend()
            
            # Adapt sample size based on performance and trend
            if performance_trend > 0:  # Performance improving
                # Check if we can reduce samples while maintaining performance
                if current_performance > np.percentile(list(self.performance_history), 75):
                    # Performance is good, try reducing samples for efficiency
                    new_sample_size = max(
                        self.min_sample_size,
                        int(self.current_sample_size * 0.9)
                    )
                else:
                    # Maintain current sample size
                    new_sample_size = self.current_sample_size
            else:  # Performance declining
                # Increase samples to improve accuracy
                new_sample_size = min(
                    self.max_sample_size,
                    int(self.current_sample_size * 1.2)
                )
        else:
            new_sample_size = self.current_sample_size
        
        # Apply computational budget constraint
        budget_adjusted_size = int(new_sample_size * computational_budget)
        budget_adjusted_size = max(self.min_sample_size, budget_adjusted_size)
        
        # Update current sample size
        self.current_sample_size = budget_adjusted_size
        
        # Update statistics
        self.strategy_stats['total_optimizations'] += 1
        
        logger.debug(f"Optimized sample size: {self.current_sample_size}")
        
        return self.current_sample_size
    
    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend"""
        if len(self.performance_history) < 3:
            return 0.0
        
        recent_performances = list(self.performance_history)[-5:]
        x = np.arange(len(recent_performances))
        
        # Simple linear trend
        try:
            slope = np.polyfit(x, recent_performances, 1)[0]
            return slope
        except:
            return 0.0
    
    def suggest_convergence_monitoring(self) -> Dict[str, Any]:
        """Suggest convergence monitoring parameters"""
        if len(self.performance_history) >= 10:
            performance_variance = np.var(list(self.performance_history)[-10:])
            
            # Adaptive convergence threshold
            if performance_variance < 0.001:
                convergence_threshold = 0.0005  # Tighter threshold for stable performance
            else:
                convergence_threshold = 0.002   # Looser threshold for volatile performance
        else:
            convergence_threshold = self.convergence_threshold
        
        return {
            'convergence_threshold': convergence_threshold,
            'monitoring_frequency': max(50, self.current_sample_size // 20),
            'early_stopping_patience': 5
        }


class ParameterLearningSystem:
    """
    System for learning and updating parameter distributions from forecast errors
    
    Continuously improves parameter distributions based on realized forecast accuracy.
    """
    
    def __init__(self, 
                 initial_distributions: Dict[str, Any],
                 learning_rate: float = 0.05):
        """
        Initialize parameter learning system
        
        Args:
            initial_distributions: Initial parameter distributions
            learning_rate: Rate of parameter adaptation
        """
        self.parameter_distributions = initial_distributions.copy()
        self.learning_rate = learning_rate
        
        # Learning history
        self.parameter_history = {name: [] for name in initial_distributions.keys()}
        self.performance_feedback = []
        self.update_history = []
        
        # Learned statistics
        self.optimal_parameters = {}
        self.parameter_correlations = {}
        
        logger.info(f"Initialized parameter learning system for {len(initial_distributions)} parameters")
    
    def update_from_forecast_error(self,
                                 forecast_error: float,
                                 used_parameters: Dict[str, float],
                                 error_attribution: Optional[Dict[str, float]] = None):
        """
        Update parameter distributions based on forecast error
        
        Args:
            forecast_error: Realized forecast error
            used_parameters: Parameters used in the forecast
            error_attribution: Attribution of error to specific parameters
        """
        # Store feedback
        self.performance_feedback.append(forecast_error)
        
        for param_name, param_value in used_parameters.items():
            if param_name in self.parameter_history:
                self.parameter_history[param_name].append(param_value)
        
        # Update distributions if we have enough data
        if len(self.performance_feedback) >= 10:
            self._update_parameter_distributions(error_attribution)
    
    def _update_parameter_distributions(self, error_attribution: Optional[Dict[str, float]]):
        """Update parameter distributions based on accumulated feedback"""
        recent_errors = self.performance_feedback[-20:]  # Use recent data
        recent_performance = np.mean(recent_errors)
        
        for param_name, dist_config in self.parameter_distributions.items():
            if param_name in self.parameter_history and self.parameter_history[param_name]:
                recent_values = self.parameter_history[param_name][-20:]
                
                # Calculate correlation between parameter values and performance
                if len(recent_values) >= 5:
                    correlation = self._calculate_param_performance_correlation(
                        recent_values, recent_errors[:len(recent_values)]
                    )
                    
                    # Update distribution parameters based on correlation
                    updated_config = self._adapt_distribution_config(
                        dist_config, recent_values, correlation, error_attribution
                    )
                    
                    self.parameter_distributions[param_name] = updated_config
                    
                    # Store update
                    self.update_history.append({
                        'parameter': param_name,
                        'old_config': dist_config,
                        'new_config': updated_config,
                        'correlation': correlation,
                        'sample_size': len(recent_values)
                    })
    
    def _calculate_param_performance_correlation(self, 
                                               param_values: List[float],
                                               error_values: List[float]) -> float:
        """Calculate correlation between parameter values and forecast errors"""
        if len(param_values) != len(error_values) or len(param_values) < 3:
            return 0.0
        
        try:
            correlation = np.corrcoef(param_values, error_values)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _adapt_distribution_config(self,
                                 current_config: Dict[str, Any],
                                 recent_values: List[float],
                                 correlation: float,
                                 error_attribution: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Adapt distribution configuration based on performance feedback"""
        updated_config = current_config.copy()
        
        # Calculate empirical statistics from recent values
        empirical_mean = np.mean(recent_values)
        empirical_std = np.std(recent_values)
        
        dist_type = current_config.get('type', 'normal')
        params = current_config.get('params', {})
        
        if dist_type == 'normal':
            current_mean = params.get('mean', 0)
            current_std = params.get('std', 1)
            
            # Adjust mean based on correlation with performance
            if abs(correlation) > 0.2:  # Significant correlation
                if correlation > 0:  # Higher values lead to higher errors
                    # Shift distribution towards lower values
                    new_mean = current_mean - self.learning_rate * abs(correlation) * current_std
                else:  # Higher values lead to lower errors
                    # Shift distribution towards higher values
                    new_mean = current_mean + self.learning_rate * abs(correlation) * current_std
            else:
                # Move towards empirical mean
                new_mean = (1 - self.learning_rate) * current_mean + self.learning_rate * empirical_mean
            
            # Adapt standard deviation based on empirical data
            new_std = (1 - self.learning_rate) * current_std + self.learning_rate * empirical_std
            
            updated_config['params'] = {
                'mean': new_mean,
                'std': max(0.001, new_std)  # Ensure positive std
            }
        
        elif dist_type == 'uniform':
            current_low = params.get('low', 0)
            current_high = params.get('high', 1)
            
            # Adapt bounds based on empirical data
            empirical_range = np.max(recent_values) - np.min(recent_values)
            range_expansion = 1.1  # 10% expansion
            
            new_low = empirical_mean - empirical_range * range_expansion / 2
            new_high = empirical_mean + empirical_range * range_expansion / 2
            
            # Blend with current bounds
            updated_config['params'] = {
                'low': (1 - self.learning_rate) * current_low + self.learning_rate * new_low,
                'high': (1 - self.learning_rate) * current_high + self.learning_rate * new_high
            }
        
        return updated_config
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        stats = {
            'total_updates': len(self.update_history),
            'parameters_learned': list(self.parameter_distributions.keys()),
            'performance_trend': self._calculate_performance_trend(),
            'parameter_adaptation_summary': {}
        }
        
        # Calculate adaptation summary for each parameter
        for param_name in self.parameter_distributions.keys():
            param_updates = [u for u in self.update_history if u['parameter'] == param_name]
            
            if param_updates:
                avg_correlation = np.mean([u['correlation'] for u in param_updates])
                stats['parameter_adaptation_summary'][param_name] = {
                    'n_updates': len(param_updates),
                    'average_correlation': avg_correlation,
                    'last_update': param_updates[-1] if param_updates else None
                }
        
        return stats
    
    def _calculate_performance_trend(self) -> float:
        """Calculate overall performance trend"""
        if len(self.performance_feedback) < 5:
            return 0.0
        
        recent_performance = self.performance_feedback[-10:]
        x = np.arange(len(recent_performance))
        
        try:
            slope = np.polyfit(x, recent_performance, 1)[0]
            return -slope  # Negative slope means improving (lower errors)
        except:
            return 0.0


class ConvergenceMonitor:
    """
    Real-time convergence monitoring with adaptive stopping criteria
    
    Monitors Monte Carlo convergence and provides intelligent stopping decisions.
    """
    
    def __init__(self, 
                 convergence_threshold: float = 0.001,
                 monitoring_frequency: int = 100,
                 patience: int = 5):
        """
        Initialize convergence monitor
        
        Args:
            convergence_threshold: Threshold for convergence detection
            monitoring_frequency: How often to check convergence
            patience: Number of checks to wait before early stopping
        """
        self.convergence_threshold = convergence_threshold
        self.monitoring_frequency = monitoring_frequency
        self.patience = patience
        
        # Monitoring state
        self.convergence_history = []
        self.running_estimates = []
        self.convergence_checks = 0
        self.patience_counter = 0
        self.is_converged = False
        
        # Statistics
        self.convergence_stats = {
            'final_convergence_value': None,
            'convergence_iteration': None,
            'early_stopped': False,
            'stability_measure': None
        }
        
        logger.info(f"Initialized convergence monitor with threshold {convergence_threshold}")
    
    def check_convergence(self, 
                        current_iteration: int,
                        current_estimate: float,
                        running_variance: Optional[float] = None) -> Dict[str, Any]:
        """
        Check convergence status
        
        Args:
            current_iteration: Current iteration number
            current_estimate: Current Monte Carlo estimate
            running_variance: Current variance estimate
            
        Returns:
            Convergence status and recommendations
        """
        self.running_estimates.append(current_estimate)
        
        # Check convergence at specified frequency
        if current_iteration % self.monitoring_frequency == 0:
            convergence_measure = self._calculate_convergence_measure()
            self.convergence_history.append(convergence_measure)
            self.convergence_checks += 1
            
            # Check if converged
            if convergence_measure < self.convergence_threshold:
                if not self.is_converged:
                    logger.info(f"Convergence detected at iteration {current_iteration}")
                    self.is_converged = True
                    self.convergence_stats['convergence_iteration'] = current_iteration
                    self.convergence_stats['final_convergence_value'] = convergence_measure
                
                self.patience_counter += 1
            else:
                self.patience_counter = 0
            
            # Early stopping decision
            should_stop = (self.is_converged and self.patience_counter >= self.patience)
            
            if should_stop:
                self.convergence_stats['early_stopped'] = True
                logger.info(f"Early stopping at iteration {current_iteration}")
            
            return {
                'is_converged': self.is_converged,
                'should_stop': should_stop,
                'convergence_measure': convergence_measure,
                'iterations_until_next_check': self.monitoring_frequency,
                'patience_remaining': max(0, self.patience - self.patience_counter)
            }
        
        return {
            'is_converged': self.is_converged,
            'should_stop': False,
            'convergence_measure': None,
            'iterations_until_next_check': self.monitoring_frequency - (current_iteration % self.monitoring_frequency),
            'patience_remaining': max(0, self.patience - self.patience_counter)
        }
    
    def _calculate_convergence_measure(self) -> float:
        """Calculate convergence measure from running estimates"""
        if len(self.running_estimates) < 10:
            return float('inf')
        
        # Use relative change in running mean as convergence measure
        recent_window = min(50, len(self.running_estimates) // 2)
        
        if len(self.running_estimates) >= recent_window * 2:
            older_mean = np.mean(self.running_estimates[-recent_window*2:-recent_window])
            recent_mean = np.mean(self.running_estimates[-recent_window:])
            
            if abs(older_mean) > 1e-10:
                convergence_measure = abs(recent_mean - older_mean) / abs(older_mean)
            else:
                convergence_measure = abs(recent_mean - older_mean)
        else:
            # Use variance-based measure for short sequences
            convergence_measure = np.std(self.running_estimates[-10:]) / (abs(np.mean(self.running_estimates[-10:])) + 1e-10)
        
        return convergence_measure
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive convergence diagnostics"""
        if not self.convergence_history:
            return {'status': 'insufficient_data'}
        
        diagnostics = {
            'convergence_achieved': self.is_converged,
            'convergence_history': self.convergence_history,
            'final_convergence_measure': self.convergence_history[-1] if self.convergence_history else None,
            'convergence_trend': self._calculate_convergence_trend(),
            'stability_assessment': self._assess_stability(),
            'recommendation': self._generate_recommendation()
        }
        
        return diagnostics
    
    def _calculate_convergence_trend(self) -> str:
        """Calculate trend in convergence measures"""
        if len(self.convergence_history) < 3:
            return 'insufficient_data'
        
        recent_trend = np.polyfit(range(len(self.convergence_history[-5:])), 
                                 self.convergence_history[-5:], 1)[0]
        
        if recent_trend < -0.001:
            return 'improving'
        elif recent_trend > 0.001:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _assess_stability(self) -> str:
        """Assess stability of convergence"""
        if len(self.convergence_history) < 5:
            return 'unknown'
        
        recent_variance = np.var(self.convergence_history[-5:])
        
        if recent_variance < self.convergence_threshold / 10:
            return 'very_stable'
        elif recent_variance < self.convergence_threshold / 2:
            return 'stable'
        else:
            return 'unstable'
    
    def _generate_recommendation(self) -> str:
        """Generate recommendation based on convergence analysis"""
        if not self.convergence_history:
            return 'continue_monitoring'
        
        trend = self._calculate_convergence_trend()
        stability = self._assess_stability()
        
        if self.is_converged and stability in ['stable', 'very_stable']:
            return 'can_stop'
        elif trend == 'improving':
            return 'continue_brief'
        elif trend == 'deteriorating':
            return 'increase_samples'
        else:
            return 'continue_monitoring'


class AdaptiveMonteCarloEngine:
    """
    Main adaptive Monte Carlo engine that integrates all adaptive components
    
    Provides a unified interface for adaptive Monte Carlo simulation with
    real-time learning and optimization.
    """
    
    def __init__(self, 
                 initial_distributions: Dict[str, Any],
                 config: Dict[str, Any] = None):
        """
        Initialize adaptive Monte Carlo engine
        
        Args:
            initial_distributions: Initial parameter distributions
            config: Engine configuration
        """
        self.config = config or {}
        
        # Initialize components
        self.sampling_optimizer = AdaptiveSamplingOptimizer(
            initial_sample_size=self.config.get('initial_sample_size', 1000),
            learning_rate=self.config.get('sampling_learning_rate', 0.1),
            exploration_rate=self.config.get('exploration_rate', 0.2)
        )
        
        self.parameter_learner = ParameterLearningSystem(
            initial_distributions,
            learning_rate=self.config.get('parameter_learning_rate', 0.05)
        )
        
        self.convergence_monitor = ConvergenceMonitor(
            convergence_threshold=self.config.get('convergence_threshold', 0.001),
            monitoring_frequency=self.config.get('monitoring_frequency', 100),
            patience=self.config.get('patience', 5)
        )
        
        # Engine state
        self.adaptation_history = []
        self.current_iteration = 0
        self.total_samples_used = 0
        
        # Performance tracking
        self.performance_metrics = {
            'adaptations_made': 0,
            'total_runtime': 0,
            'convergence_achieved': False,
            'final_accuracy': None
        }
        
        logger.info("Initialized adaptive Monte Carlo engine")
    
    def run_adaptive_simulation(self,
                              simulation_function: Callable,
                              target_accuracy: float = 0.01,
                              max_iterations: int = 50000,
                              performance_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run adaptive Monte Carlo simulation
        
        Args:
            simulation_function: Function to run Monte Carlo simulation
            target_accuracy: Target accuracy for stopping
            max_iterations: Maximum iterations
            performance_callback: Optional callback for performance feedback
            
        Returns:
            Simulation results with adaptation statistics
        """
        logger.info(f"Starting adaptive simulation with target accuracy {target_accuracy}")
        
        # Initialize simulation state
        simulation_results = []
        performance_history = []
        current_estimate = 0.0
        
        while self.current_iteration < max_iterations:
            # Get optimal sample size
            optimal_samples = self.sampling_optimizer.optimize_sample_size(
                performance_history,
                computational_budget=1.0  # Could be dynamic
            )
            
            # Run simulation with adaptive parameters
            current_distributions = self.parameter_learner.parameter_distributions
            
            try:
                # Run one iteration of simulation
                iteration_result = simulation_function(
                    n_samples=optimal_samples,
                    distributions=current_distributions
                )
                
                simulation_results.append(iteration_result)
                current_estimate = iteration_result.get('estimate', 0)
                
                # Check convergence
                convergence_status = self.convergence_monitor.check_convergence(
                    self.current_iteration,
                    current_estimate,
                    iteration_result.get('variance')
                )
                
                # Update performance history
                iteration_accuracy = iteration_result.get('accuracy', 0)
                performance_history.append(iteration_accuracy)
                
                # Provide feedback to parameter learner
                if len(performance_history) > 1:
                    forecast_error = abs(iteration_accuracy - np.mean(performance_history[-5:]))
                    used_parameters = iteration_result.get('parameters_used', {})
                    
                    self.parameter_learner.update_from_forecast_error(
                        forecast_error, used_parameters
                    )
                
                # Call performance callback if provided
                if performance_callback:
                    callback_data = {
                        'iteration': self.current_iteration,
                        'estimate': current_estimate,
                        'accuracy': iteration_accuracy,
                        'convergence_status': convergence_status,
                        'sample_size': optimal_samples
                    }
                    performance_callback(callback_data)
                
                # Check stopping criteria
                if convergence_status['should_stop']:
                    logger.info(f"Adaptive simulation converged after {self.current_iteration} iterations")
                    self.performance_metrics['convergence_achieved'] = True
                    break
                
                if iteration_accuracy <= target_accuracy:
                    logger.info(f"Target accuracy {target_accuracy} achieved")
                    break
                
                self.current_iteration += 1
                self.total_samples_used += optimal_samples
                
            except Exception as e:
                logger.error(f"Error in simulation iteration {self.current_iteration}: {str(e)}")
                break
        
        # Compile final results
        final_results = self._compile_final_results(
            simulation_results, performance_history, target_accuracy
        )
        
        return final_results
    
    def _compile_final_results(self,
                             simulation_results: List[Dict],
                             performance_history: List[float],
                             target_accuracy: float) -> Dict[str, Any]:
        """Compile final simulation results with adaptation statistics"""
        
        # Calculate final estimates
        if simulation_results:
            final_estimate = simulation_results[-1].get('estimate', 0)
            estimates = [r.get('estimate', 0) for r in simulation_results]
            estimate_variance = np.var(estimates) if estimates else 0
        else:
            final_estimate = 0
            estimate_variance = 0
        
        # Get learning statistics
        learning_stats = self.parameter_learner.get_learning_statistics()
        convergence_diagnostics = self.convergence_monitor.get_convergence_diagnostics()
        
        # Calculate adaptation effectiveness
        adaptation_effectiveness = self._calculate_adaptation_effectiveness(performance_history)
        
        results = {
            'final_estimate': final_estimate,
            'estimate_variance': estimate_variance,
            'total_iterations': self.current_iteration,
            'total_samples_used': self.total_samples_used,
            'target_accuracy_achieved': performance_history[-1] <= target_accuracy if performance_history else False,
            'final_accuracy': performance_history[-1] if performance_history else None,
            
            # Adaptation statistics
            'learning_statistics': learning_stats,
            'convergence_diagnostics': convergence_diagnostics,
            'adaptation_effectiveness': adaptation_effectiveness,
            'sampling_optimization_stats': self.sampling_optimizer.strategy_stats,
            
            # Performance tracking
            'performance_history': performance_history,
            'simulation_results': simulation_results,
            
            # Configuration
            'configuration': self.config,
            'adaptive_features_used': [
                'adaptive_sampling',
                'parameter_learning',
                'convergence_monitoring'
            ]
        }
        
        self.performance_metrics['final_accuracy'] = results['final_accuracy']
        
        return results
    
    def _calculate_adaptation_effectiveness(self, performance_history: List[float]) -> Dict[str, Any]:
        """Calculate effectiveness of adaptive strategies"""
        if len(performance_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Split performance into early and late periods
        split_point = len(performance_history) // 2
        early_performance = performance_history[:split_point]
        late_performance = performance_history[split_point:]
        
        early_mean = np.mean(early_performance)
        late_mean = np.mean(late_performance)
        
        improvement = (early_mean - late_mean) / early_mean if early_mean > 0 else 0
        
        return {
            'performance_improvement': improvement,
            'early_period_accuracy': early_mean,
            'late_period_accuracy': late_mean,
            'improvement_percentage': improvement * 100,
            'adaptation_success': improvement > 0.05  # 5% improvement threshold
        }
    
    def export_adaptation_report(self, output_path: str) -> str:
        """Export comprehensive adaptation report"""
        try:
            # Compile adaptation data
            adaptation_data = {
                'Engine Configuration': self.config,
                'Performance Metrics': self.performance_metrics,
                'Sampling Optimization': self.sampling_optimizer.strategy_stats,
                'Parameter Learning': self.parameter_learner.get_learning_statistics(),
                'Convergence Monitoring': self.convergence_monitor.get_convergence_diagnostics()
            }
            
            # Create DataFrame for export
            adaptation_summary = []
            for category, data in adaptation_data.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        adaptation_summary.append({
                            'Category': category,
                            'Metric': key,
                            'Value': str(value)
                        })
            
            adaptation_df = pd.DataFrame(adaptation_summary)
            
            # Export to Excel
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                adaptation_df.to_excel(writer, sheet_name='Adaptation_Summary', index=False)
                
                # Parameter learning details
                if self.parameter_learner.update_history:
                    update_data = []
                    for update in self.parameter_learner.update_history:
                        update_data.append({
                            'Parameter': update['parameter'],
                            'Correlation': update['correlation'],
                            'Sample_Size': update['sample_size']
                        })
                    
                    update_df = pd.DataFrame(update_data)
                    update_df.to_excel(writer, sheet_name='Parameter_Updates', index=False)
                
                # Convergence history
                if self.convergence_monitor.convergence_history:
                    convergence_df = pd.DataFrame({
                        'Check_Number': range(len(self.convergence_monitor.convergence_history)),
                        'Convergence_Measure': self.convergence_monitor.convergence_history
                    })
                    convergence_df.to_excel(writer, sheet_name='Convergence_History', index=False)
            
            logger.info(f"Exported adaptation report to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting adaptation report: {str(e)}")
            return ""


# Integration functions
def create_adaptive_monte_carlo_system(base_monte_carlo_engine,
                                     adaptive_config: Dict[str, Any] = None) -> AdaptiveMonteCarloEngine:
    """
    Create adaptive Monte Carlo system from existing engine
    
    Args:
        base_monte_carlo_engine: Existing Monte Carlo engine
        adaptive_config: Configuration for adaptive features
        
    Returns:
        Enhanced adaptive Monte Carlo engine
    """
    # Extract parameter distributions from base engine
    initial_distributions = {}
    
    if hasattr(base_monte_carlo_engine, 'parameter_distributions'):
        for name, dist in base_monte_carlo_engine.parameter_distributions.items():
            if hasattr(dist, 'distribution_type') and hasattr(dist, 'params'):
                initial_distributions[name] = {
                    'type': dist.distribution_type,
                    'params': dist.params
                }
    
    # Create adaptive engine
    adaptive_engine = AdaptiveMonteCarloEngine(
        initial_distributions,
        adaptive_config or {}
    )
    
    logger.info("Created adaptive Monte Carlo system from existing engine")
    return adaptive_engine


# Example usage
if __name__ == "__main__":
    # Define example simulation function
    def example_simulation(n_samples: int, distributions: Dict[str, Any]) -> Dict[str, Any]:
        """Example Monte Carlo simulation function"""
        # Extract parameters
        growth_dist = distributions.get('growth_rate', {'type': 'normal', 'params': {'mean': 0.05, 'std': 0.01}})
        
        # Sample parameters
        if growth_dist['type'] == 'normal':
            growth_samples = np.random.normal(
                growth_dist['params']['mean'],
                growth_dist['params']['std'],
                n_samples
            )
        else:
            growth_samples = np.random.normal(0.05, 0.01, n_samples)
        
        # Simple calculation
        results = np.mean(growth_samples)
        variance = np.var(growth_samples)
        
        # Simulate accuracy (would be real forecast error in practice)
        true_value = 0.045  # "True" parameter value
        accuracy = abs(results - true_value)
        
        return {
            'estimate': results,
            'variance': variance,
            'accuracy': accuracy,
            'parameters_used': {'growth_rate': np.mean(growth_samples)},
            'n_samples': n_samples
        }
    
    # Initialize adaptive engine
    initial_distributions = {
        'growth_rate': {
            'type': 'normal',
            'params': {'mean': 0.05, 'std': 0.02}
        }
    }
    
    config = {
        'initial_sample_size': 500,
        'convergence_threshold': 0.001,
        'parameter_learning_rate': 0.1
    }
    
    adaptive_engine = AdaptiveMonteCarloEngine(initial_distributions, config)
    
    # Run adaptive simulation
    def progress_callback(data):
        if data['iteration'] % 10 == 0:
            print(f"Iteration {data['iteration']}: estimate={data['estimate']:.4f}, "
                  f"accuracy={data['accuracy']:.4f}, samples={data['sample_size']}")
    
    results = adaptive_engine.run_adaptive_simulation(
        example_simulation,
        target_accuracy=0.005,
        max_iterations=100,
        performance_callback=progress_callback
    )
    
    print("\nAdaptive Monte Carlo Results:")
    print(f"Final estimate: {results['final_estimate']:.4f}")
    print(f"Final accuracy: {results['final_accuracy']:.4f}")
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Total samples: {results['total_samples_used']}")
    print(f"Target achieved: {results['target_accuracy_achieved']}")
    print(f"Convergence achieved: {results['convergence_diagnostics']['convergence_achieved']}")
    
    if results['adaptation_effectiveness']['status'] != 'insufficient_data':
        improvement = results['adaptation_effectiveness']['improvement_percentage']
        print(f"Performance improvement: {improvement:.1f}%")