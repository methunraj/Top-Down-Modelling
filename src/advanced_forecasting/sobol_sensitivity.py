"""
Sobol Sensitivity Analysis - Global parameter importance assessment

This module implements Sobol indices for global sensitivity analysis, providing
comprehensive understanding of parameter importance and interactions.

Key capabilities:
1. First-order Sobol indices (main effects)
2. Total-order Sobol indices (total effects including interactions)
3. Second-order interaction indices
4. Confidence intervals for indices
5. Parameter ranking and importance classification
6. Saltelli sampling for efficient computation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from scipy import stats
from scipy.optimize import minimize
import itertools
import warnings

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SobolSensitivityAnalyzer:
    """
    Sobol sensitivity analysis for global parameter importance
    
    Implements efficient computation of Sobol indices using Saltelli sampling
    and provides comprehensive sensitivity analysis results.
    """
    
    def __init__(self, 
                 parameter_ranges: Dict[str, Tuple[float, float]],
                 model_function: Callable,
                 n_bootstrap: int = 100):
        """
        Initialize Sobol sensitivity analyzer
        
        Args:
            parameter_ranges: Dictionary of parameter ranges {name: (min, max)}
            model_function: Model function to analyze
            n_bootstrap: Number of bootstrap samples for confidence intervals
        """
        self.parameter_ranges = parameter_ranges
        self.parameter_names = list(parameter_ranges.keys())
        self.n_parameters = len(self.parameter_names)
        self.model_function = model_function
        self.n_bootstrap = n_bootstrap
        
        # Results storage
        self.sobol_results = {}
        self.sampling_matrices = {}
        self.model_evaluations = {}
        
        logger.info(f"Initialized Sobol analyzer for {self.n_parameters} parameters")
    
    def calculate_sobol_indices(self, 
                              n_samples: int = 1000,
                              confidence_level: float = 0.95,
                              calc_second_order: bool = False) -> Dict[str, Any]:
        """
        Calculate Sobol sensitivity indices
        
        Args:
            n_samples: Base number of samples (total will be larger due to Saltelli method)
            confidence_level: Confidence level for bootstrap intervals
            calc_second_order: Whether to calculate second-order interaction indices
            
        Returns:
            Dictionary with Sobol indices and statistics
        """
        logger.info(f"Calculating Sobol indices with {n_samples} base samples")
        
        # Generate Saltelli sampling matrices
        sampling_matrices = self._generate_saltelli_samples(n_samples)
        self.sampling_matrices = sampling_matrices
        
        # Evaluate model for all sample points
        model_outputs = self._evaluate_model_samples(sampling_matrices)
        self.model_evaluations = model_outputs
        
        # Calculate Sobol indices
        first_order_indices = self._calculate_first_order_indices(model_outputs, n_samples)
        total_order_indices = self._calculate_total_order_indices(model_outputs, n_samples)
        
        # Calculate second-order indices if requested
        second_order_indices = {}
        if calc_second_order:
            second_order_indices = self._calculate_second_order_indices(model_outputs, n_samples)
        
        # Calculate confidence intervals using bootstrap
        confidence_intervals = self._calculate_confidence_intervals(
            model_outputs, n_samples, confidence_level
        )
        
        # Compile results
        results = {
            'first_order': first_order_indices,
            'total_order': total_order_indices,
            'second_order': second_order_indices,
            'confidence_intervals': confidence_intervals,
            'parameter_ranking': self._rank_parameters(first_order_indices, total_order_indices),
            'interaction_strength': self._calculate_interaction_strength(first_order_indices, total_order_indices),
            'variance_explained': self._calculate_variance_explained(first_order_indices),
            'model_statistics': self._calculate_model_statistics(model_outputs)
        }
        
        self.sobol_results = results
        return results
    
    def _generate_saltelli_samples(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Generate Saltelli sampling matrices for efficient Sobol calculation
        
        Args:
            n_samples: Base number of samples
            
        Returns:
            Dictionary of sampling matrices
        """
        # Generate two independent matrices A and B
        A = np.random.uniform(0, 1, (n_samples, self.n_parameters))
        B = np.random.uniform(0, 1, (n_samples, self.n_parameters))
        
        # Scale to parameter ranges
        A_scaled = self._scale_to_parameter_ranges(A)
        B_scaled = self._scale_to_parameter_ranges(B)
        
        # Generate C matrices (A with i-th column from B)
        C_matrices = {}
        for i, param_name in enumerate(self.parameter_names):
            C = A_scaled.copy()
            C[:, i] = B_scaled[:, i]
            C_matrices[f'C_{param_name}'] = C
        
        # Generate D matrices (B with i-th column from A) for total order indices
        D_matrices = {}
        for i, param_name in enumerate(self.parameter_names):
            D = B_scaled.copy()
            D[:, i] = A_scaled[:, i]
            D_matrices[f'D_{param_name}'] = D
        
        sampling_matrices = {
            'A': A_scaled,
            'B': B_scaled,
            **C_matrices,
            **D_matrices
        }
        
        total_samples = len(sampling_matrices) * n_samples
        logger.info(f"Generated Saltelli matrices: {total_samples} total model evaluations")
        
        return sampling_matrices
    
    def _scale_to_parameter_ranges(self, uniform_samples: np.ndarray) -> np.ndarray:
        """Scale uniform [0,1] samples to parameter ranges"""
        scaled_samples = uniform_samples.copy()
        
        for i, param_name in enumerate(self.parameter_names):
            min_val, max_val = self.parameter_ranges[param_name]
            scaled_samples[:, i] = min_val + uniform_samples[:, i] * (max_val - min_val)
        
        return scaled_samples
    
    def _evaluate_model_samples(self, sampling_matrices: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Evaluate model function for all sample matrices"""
        model_outputs = {}
        
        for matrix_name, matrix in sampling_matrices.items():
            logger.debug(f"Evaluating model for {matrix_name}")
            
            outputs = []
            for sample in matrix:
                try:
                    # Convert sample to parameter dictionary
                    param_dict = {name: sample[i] for i, name in enumerate(self.parameter_names)}
                    output = self.model_function(param_dict)
                    outputs.append(output)
                except Exception as e:
                    logger.warning(f"Model evaluation failed for sample: {e}")
                    outputs.append(np.nan)
            
            model_outputs[matrix_name] = np.array(outputs)
        
        return model_outputs
    
    def _calculate_first_order_indices(self, 
                                     model_outputs: Dict[str, np.ndarray],
                                     n_samples: int) -> Dict[str, float]:
        """Calculate first-order Sobol indices"""
        f_A = model_outputs['A']
        f_B = model_outputs['B']
        
        # Estimate total variance
        total_variance = np.var(np.concatenate([f_A, f_B]))
        
        first_order_indices = {}
        
        for i, param_name in enumerate(self.parameter_names):
            f_C = model_outputs[f'C_{param_name}']
            
            # First-order index calculation
            if total_variance > 0:
                # V_i = E[f(A) * f(C_i)] - E[f(A)]^2
                numerator = np.mean(f_A * f_C) - np.mean(f_A) * np.mean(f_C)
                first_order_indices[param_name] = numerator / total_variance
            else:
                first_order_indices[param_name] = 0.0
            
            # Ensure non-negative
            first_order_indices[param_name] = max(0.0, first_order_indices[param_name])
        
        logger.debug(f"Calculated first-order indices: {first_order_indices}")
        return first_order_indices
    
    def _calculate_total_order_indices(self, 
                                     model_outputs: Dict[str, np.ndarray],
                                     n_samples: int) -> Dict[str, float]:
        """Calculate total-order Sobol indices"""
        f_A = model_outputs['A']
        f_B = model_outputs['B']
        
        # Estimate total variance
        total_variance = np.var(np.concatenate([f_A, f_B]))
        
        total_order_indices = {}
        
        for i, param_name in enumerate(self.parameter_names):
            f_D = model_outputs[f'D_{param_name}']
            
            # Total-order index calculation
            if total_variance > 0:
                # T_i = 1 - (E[f(B) * f(D_i)] - E[f(B)]^2) / Var[f]
                numerator = np.mean(f_B * f_D) - np.mean(f_B) * np.mean(f_D)
                total_order_indices[param_name] = 1.0 - numerator / total_variance
            else:
                total_order_indices[param_name] = 0.0
            
            # Ensure in [0, 1]
            total_order_indices[param_name] = max(0.0, min(1.0, total_order_indices[param_name]))
        
        logger.debug(f"Calculated total-order indices: {total_order_indices}")
        return total_order_indices
    
    def _calculate_second_order_indices(self, 
                                      model_outputs: Dict[str, np.ndarray],
                                      n_samples: int) -> Dict[str, float]:
        """Calculate second-order interaction indices"""
        # This requires additional sampling matrices for all parameter pairs
        # For simplicity, we'll estimate from existing data
        
        f_A = model_outputs['A']
        total_variance = np.var(f_A)
        
        second_order_indices = {}
        
        # Calculate for all parameter pairs
        for i, param_i in enumerate(self.parameter_names):
            for j, param_j in enumerate(self.parameter_names[i+1:], i+1):
                # Approximate second-order index
                # This is a simplified calculation - proper implementation would need more matrices
                first_i = self.sobol_results.get('first_order', {}).get(param_i, 0)
                first_j = self.sobol_results.get('first_order', {}).get(param_j, 0)
                total_i = self.sobol_results.get('total_order', {}).get(param_i, 0)
                total_j = self.sobol_results.get('total_order', {}).get(param_j, 0)
                
                # Rough approximation: S_ij ≈ interaction component
                interaction_strength = max(0, min(total_i - first_i, total_j - first_j))
                
                pair_name = f"{param_i}_{param_j}"
                second_order_indices[pair_name] = interaction_strength
        
        return second_order_indices
    
    def _calculate_confidence_intervals(self, 
                                      model_outputs: Dict[str, np.ndarray],
                                      n_samples: int,
                                      confidence_level: float) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate bootstrap confidence intervals for Sobol indices"""
        alpha = 1 - confidence_level
        
        confidence_intervals = {
            'first_order': {},
            'total_order': {}
        }
        
        # Bootstrap resampling
        bootstrap_first_order = {param: [] for param in self.parameter_names}
        bootstrap_total_order = {param: [] for param in self.parameter_names}
        
        for boot_iter in range(self.n_bootstrap):
            # Resample indices
            boot_indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Create bootstrap model outputs
            boot_outputs = {}
            for key, outputs in model_outputs.items():
                boot_outputs[key] = outputs[boot_indices]
            
            # Calculate bootstrap Sobol indices
            boot_first = self._calculate_first_order_indices(boot_outputs, n_samples)
            boot_total = self._calculate_total_order_indices(boot_outputs, n_samples)
            
            # Store bootstrap results
            for param in self.parameter_names:
                bootstrap_first_order[param].append(boot_first.get(param, 0))
                bootstrap_total_order[param].append(boot_total.get(param, 0))
        
        # Calculate confidence intervals
        for param in self.parameter_names:
            # First-order confidence intervals
            first_values = bootstrap_first_order[param]
            first_lower = np.percentile(first_values, 100 * alpha / 2)
            first_upper = np.percentile(first_values, 100 * (1 - alpha / 2))
            confidence_intervals['first_order'][param] = (first_lower, first_upper)
            
            # Total-order confidence intervals
            total_values = bootstrap_total_order[param]
            total_lower = np.percentile(total_values, 100 * alpha / 2)
            total_upper = np.percentile(total_values, 100 * (1 - alpha / 2))
            confidence_intervals['total_order'][param] = (total_lower, total_upper)
        
        return confidence_intervals
    
    def _rank_parameters(self, 
                        first_order: Dict[str, float],
                        total_order: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Rank parameters by importance"""
        # Sort by total-order indices (most comprehensive measure)
        sorted_params_total = sorted(total_order.items(), key=lambda x: x[1], reverse=True)
        sorted_params_first = sorted(first_order.items(), key=lambda x: x[1], reverse=True)
        
        ranking = {}
        for i, (param, total_index) in enumerate(sorted_params_total):
            first_index = first_order.get(param, 0)
            
            # Classification
            if total_index > 0.1:
                importance = 'high'
            elif total_index > 0.05:
                importance = 'medium'
            elif total_index > 0.01:
                importance = 'low'
            else:
                importance = 'negligible'
            
            ranking[param] = {
                'total_order_rank': i + 1,
                'first_order_rank': next((j + 1 for j, (p, _) in enumerate(sorted_params_first) if p == param), None),
                'total_order_index': total_index,
                'first_order_index': first_index,
                'interaction_strength': total_index - first_index,
                'importance_class': importance
            }
        
        return ranking
    
    def _calculate_interaction_strength(self, 
                                      first_order: Dict[str, float],
                                      total_order: Dict[str, float]) -> Dict[str, float]:
        """Calculate interaction strength for each parameter"""
        interaction_strength = {}
        
        for param in self.parameter_names:
            first_idx = first_order.get(param, 0)
            total_idx = total_order.get(param, 0)
            
            # Interaction strength = Total - First order
            interaction = max(0, total_idx - first_idx)
            interaction_strength[param] = interaction
        
        return interaction_strength
    
    def _calculate_variance_explained(self, first_order: Dict[str, float]) -> Dict[str, Any]:
        """Calculate variance explained by different components"""
        total_first_order = sum(first_order.values())
        
        # Interaction effects (approximation)
        interaction_effects = max(0, 1 - total_first_order)
        
        return {
            'main_effects': total_first_order,
            'interaction_effects': interaction_effects,
            'main_effects_percentage': total_first_order * 100,
            'interaction_effects_percentage': interaction_effects * 100,
            'additivity_measure': total_first_order  # Close to 1 means additive model
        }
    
    def _calculate_model_statistics(self, model_outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate model evaluation statistics"""
        all_outputs = np.concatenate(list(model_outputs.values()))
        valid_outputs = all_outputs[~np.isnan(all_outputs)]
        
        return {
            'total_evaluations': len(all_outputs),
            'successful_evaluations': len(valid_outputs),
            'success_rate': len(valid_outputs) / len(all_outputs) if len(all_outputs) > 0 else 0,
            'output_mean': np.mean(valid_outputs) if len(valid_outputs) > 0 else 0,
            'output_std': np.std(valid_outputs) if len(valid_outputs) > 0 else 0,
            'output_range': (np.min(valid_outputs), np.max(valid_outputs)) if len(valid_outputs) > 0 else (0, 0)
        }
    
    def plot_sobol_indices(self, save_path: Optional[str] = None) -> str:
        """
        Create comprehensive Sobol indices visualization
        
        Args:
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        import matplotlib.pyplot as plt
        
        if not self.sobol_results:
            logger.warning("No Sobol results to plot")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Sobol Sensitivity Analysis Results', fontsize=16)
        
        first_order = self.sobol_results['first_order']
        total_order = self.sobol_results['total_order']
        confidence_intervals = self.sobol_results['confidence_intervals']
        
        # Plot 1: First-order vs Total-order indices
        ax1 = axes[0, 0]
        params = list(first_order.keys())
        first_values = [first_order[p] for p in params]
        total_values = [total_order[p] for p in params]
        
        x = np.arange(len(params))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, first_values, width, label='First-order', alpha=0.8)
        bars2 = ax1.bar(x + width/2, total_values, width, label='Total-order', alpha=0.8)
        
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Sobol Index')
        ax1.set_title('Sobol Indices Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Interaction strength
        ax2 = axes[0, 1]
        interaction_strength = self.sobol_results['interaction_strength']
        interaction_values = [interaction_strength[p] for p in params]
        
        bars = ax2.bar(params, interaction_values, alpha=0.7, color='orange')
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Interaction Strength')
        ax2.set_title('Parameter Interaction Effects')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence intervals
        ax3 = axes[1, 0]
        first_ci = confidence_intervals['first_order']
        total_ci = confidence_intervals['total_order']
        
        first_lower = [first_ci[p][0] for p in params]
        first_upper = [first_ci[p][1] for p in params]
        total_lower = [total_ci[p][0] for p in params]
        total_upper = [total_ci[p][1] for p in params]
        
        ax3.errorbar(x - 0.1, first_values, 
                    yerr=[np.array(first_values) - np.array(first_lower),
                          np.array(first_upper) - np.array(first_values)],
                    fmt='o', label='First-order', capsize=5)
        ax3.errorbar(x + 0.1, total_values,
                    yerr=[np.array(total_values) - np.array(total_lower),
                          np.array(total_upper) - np.array(total_values)],
                    fmt='s', label='Total-order', capsize=5)
        
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Sobol Index')
        ax3.set_title('Sobol Indices with Confidence Intervals')
        ax3.set_xticks(x)
        ax3.set_xticklabels(params, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter ranking
        ax4 = axes[1, 1]
        ranking = self.sobol_results['parameter_ranking']
        
        # Create ranking visualization
        total_ranks = [ranking[p]['total_order_rank'] for p in params]
        colors = ['red' if ranking[p]['importance_class'] == 'high' else
                 'orange' if ranking[p]['importance_class'] == 'medium' else
                 'yellow' if ranking[p]['importance_class'] == 'low' else 'gray'
                 for p in params]
        
        bars = ax4.barh(params, total_values, color=colors, alpha=0.7)
        ax4.set_xlabel('Total-order Sobol Index')
        ax4.set_title('Parameter Importance Ranking')
        ax4.grid(True, alpha=0.3)
        
        # Add legend for colors
        import matplotlib.patches as patches
        legend_elements = [
            patches.Patch(color='red', label='High importance'),
            patches.Patch(color='orange', label='Medium importance'),
            patches.Patch(color='yellow', label='Low importance'),
            patches.Patch(color='gray', label='Negligible')
        ]
        ax4.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Save plot
        if save_path:
            output_path = save_path
        else:
            from datetime import datetime
            output_path = f"sobol_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Sobol sensitivity plot to {output_path}")
        return output_path
    
    def export_results(self, output_path: str) -> str:
        """
        Export Sobol results to Excel
        
        Args:
            output_path: Path for the Excel file
            
        Returns:
            Path to saved Excel file
        """
        if not self.sobol_results:
            logger.warning("No Sobol results to export")
            return ""
        
        try:
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_data = []
                for param in self.parameter_names:
                    ranking = self.sobol_results['parameter_ranking'][param]
                    ci_first = self.sobol_results['confidence_intervals']['first_order'][param]
                    ci_total = self.sobol_results['confidence_intervals']['total_order'][param]
                    
                    summary_data.append({
                        'Parameter': param,
                        'First_Order_Index': ranking['first_order_index'],
                        'Total_Order_Index': ranking['total_order_index'],
                        'Interaction_Strength': ranking['interaction_strength'],
                        'Total_Order_Rank': ranking['total_order_rank'],
                        'Importance_Class': ranking['importance_class'],
                        'First_Order_CI_Lower': ci_first[0],
                        'First_Order_CI_Upper': ci_first[1],
                        'Total_Order_CI_Lower': ci_total[0],
                        'Total_Order_CI_Upper': ci_total[1]
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Sobol_Summary', index=False)
                
                # Variance decomposition
                variance_data = [self.sobol_results['variance_explained']]
                variance_df = pd.DataFrame(variance_data)
                variance_df.to_excel(writer, sheet_name='Variance_Decomposition', index=False)
                
                # Model statistics
                stats_data = [self.sobol_results['model_statistics']]
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Model_Statistics', index=False)
            
            logger.info(f"Exported Sobol results to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting Sobol results: {str(e)}")
            return ""


# Integration functions
def integrate_sobol_with_monte_carlo(monte_carlo_engine, n_sensitivity_samples: int = 1000):
    """
    Integrate Sobol sensitivity analysis with Monte Carlo engine
    
    Args:
        monte_carlo_engine: Existing Monte Carlo engine
        n_sensitivity_samples: Number of samples for sensitivity analysis
    """
    # Extract parameter ranges from Monte Carlo distributions
    parameter_ranges = {}
    for name, distribution in monte_carlo_engine.parameter_distributions.items():
        if hasattr(distribution, 'params'):
            if distribution.distribution_type == 'normal':
                mean = distribution.params.get('mean', 0)
                std = distribution.params.get('std', 1)
                # Use 3-sigma range
                parameter_ranges[name] = (mean - 3*std, mean + 3*std)
            elif distribution.distribution_type == 'uniform':
                low = distribution.params.get('low', 0)
                high = distribution.params.get('high', 1)
                parameter_ranges[name] = (low, high)
            else:
                # Default range for other distributions
                parameter_ranges[name] = (0, 1)
    
    # Create sensitivity analyzer
    def model_wrapper(params):
        # Simple model function - would need to be adapted to specific use case
        return sum(params.values())
    
    sobol_analyzer = SobolSensitivityAnalyzer(parameter_ranges, model_wrapper)
    
    # Add to Monte Carlo engine
    monte_carlo_engine.sobol_analyzer = sobol_analyzer
    
    logger.info("Integrated Sobol sensitivity analysis with Monte Carlo engine")
    return monte_carlo_engine


# Example usage
if __name__ == "__main__":
    # Define test model function
    def test_model(params):
        """Test model: Ishigami function"""
        x1 = params.get('x1', 0)
        x2 = params.get('x2', 0)
        x3 = params.get('x3', 0)
        
        a = 7
        b = 0.1
        
        return np.sin(x1) + a * np.sin(x2)**2 + b * x3**4 * np.sin(x1)
    
    # Define parameter ranges
    parameter_ranges = {
        'x1': (-np.pi, np.pi),
        'x2': (-np.pi, np.pi),
        'x3': (-np.pi, np.pi)
    }
    
    # Initialize analyzer
    analyzer = SobolSensitivityAnalyzer(parameter_ranges, test_model)
    
    # Calculate Sobol indices
    results = analyzer.calculate_sobol_indices(n_samples=1000)
    
    print("Sobol Sensitivity Analysis Results:")
    print("=" * 40)
    
    for param, ranking in results['parameter_ranking'].items():
        print(f"{param}:")
        print(f"  First-order index: {ranking['first_order_index']:.4f}")
        print(f"  Total-order index: {ranking['total_order_index']:.4f}")
        print(f"  Interaction strength: {ranking['interaction_strength']:.4f}")
        print(f"  Importance: {ranking['importance_class']}")
        print()
    
    print(f"Main effects explain {results['variance_explained']['main_effects_percentage']:.1f}% of variance")
    print(f"Interactions explain {results['variance_explained']['interaction_effects_percentage']:.1f}% of variance")