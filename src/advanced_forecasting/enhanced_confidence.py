"""
Enhanced Confidence Interval Calculator - Advanced uncertainty quantification

This module provides sophisticated confidence interval calculation methods that go
beyond simple statistical bounds to provide:
1. Bootstrap-based confidence intervals
2. Model uncertainty quantification  
3. Time-varying confidence bounds
4. Scenario-based uncertainty propagation
5. Adaptive confidence estimation based on forecast horizon

Seamlessly integrates with existing ensemble framework while adding enterprise-grade
uncertainty modeling capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from datetime import datetime
from scipy import stats
from scipy.stats import norm, t
from sklearn.utils import resample
import warnings

from src.global_forecasting.base_forecaster import BaseForecaster
from src.utils.math_utils import calculate_confidence_interval

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BootstrapConfidenceCalculator:
    """
    Bootstrap-based confidence interval calculator for robust uncertainty estimation
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_levels: List[float] = None):
        """
        Initialize bootstrap calculator
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_levels: List of confidence levels (e.g., [0.05, 0.95] for 90% CI)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_levels = confidence_levels or [0.025, 0.975]  # 95% CI
        self.bootstrap_results = {}
        
    def calculate_bootstrap_ci(self, 
                             data: pd.DataFrame,
                             forecaster: BaseForecaster,
                             periods: int,
                             frequency: str = 'Y') -> Dict[str, np.ndarray]:
        """
        Calculate bootstrap confidence intervals
        
        Args:
            data: Historical data
            forecaster: Forecaster to bootstrap
            periods: Number of periods to forecast
            frequency: Forecast frequency
            
        Returns:
            Dictionary with confidence intervals
        """
        logger.info(f"Calculating bootstrap confidence intervals with {self.n_bootstrap} samples")
        
        bootstrap_forecasts = []
        
        # Generate bootstrap samples
        for i in range(self.n_bootstrap):
            try:
                # Bootstrap resample the data
                bootstrap_data = resample(data, random_state=i)
                
                # Fit forecaster on bootstrap sample
                forecaster_copy = self._copy_forecaster(forecaster)
                forecaster_copy.fit(bootstrap_data)
                
                # Generate forecast
                forecast = forecaster_copy.forecast(periods, frequency)
                bootstrap_forecasts.append(forecast['value'].values)
                
            except Exception as e:
                logger.warning(f"Error in bootstrap sample {i}: {str(e)}")
                continue
        
        if not bootstrap_forecasts:
            logger.error("No successful bootstrap samples")
            return {}
        
        # Convert to numpy array
        bootstrap_array = np.array(bootstrap_forecasts)
        
        # Calculate percentiles
        percentiles = {}
        for level in self.confidence_levels:
            percentiles[f'p{level}'] = np.percentile(bootstrap_array, level * 100, axis=0)
        
        # Calculate additional statistics
        percentiles['mean'] = np.mean(bootstrap_array, axis=0)
        percentiles['std'] = np.std(bootstrap_array, axis=0)
        percentiles['median'] = np.median(bootstrap_array, axis=0)
        
        self.bootstrap_results = {
            'percentiles': percentiles,
            'raw_forecasts': bootstrap_array,
            'n_successful': len(bootstrap_forecasts)
        }
        
        return percentiles
    
    def _copy_forecaster(self, forecaster: BaseForecaster) -> BaseForecaster:
        """Create a copy of the forecaster"""
        # This is a simplified copy - in practice, you'd want proper deep copying
        forecaster_type = type(forecaster)
        forecaster_copy = forecaster_type(forecaster.config)
        return forecaster_copy


class ModelUncertaintyCalculator:
    """
    Model uncertainty calculator that quantifies uncertainty from model selection
    """
    
    def __init__(self, models: List[BaseForecaster] = None):
        """
        Initialize model uncertainty calculator
        
        Args:
            models: List of different forecasting models
        """
        self.models = models or []
        self.model_weights = {}
        self.model_forecasts = {}
        
    def add_model(self, model: BaseForecaster, weight: float = None):
        """Add a model to the uncertainty calculation"""
        self.models.append(model)
        if weight is not None:
            self.model_weights[model.name] = weight
    
    def calculate_model_uncertainty(self, 
                                  data: pd.DataFrame,
                                  periods: int,
                                  frequency: str = 'Y') -> Dict[str, np.ndarray]:
        """
        Calculate uncertainty due to model selection
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            frequency: Forecast frequency
            
        Returns:
            Dictionary with model uncertainty metrics
        """
        logger.info(f"Calculating model uncertainty across {len(self.models)} models")
        
        model_forecasts = []
        model_names = []
        
        # Generate forecasts from all models
        for model in self.models:
            try:
                model.fit(data)
                forecast = model.forecast(periods, frequency)
                model_forecasts.append(forecast['value'].values)
                model_names.append(model.name)
                
                # Store individual model forecast
                self.model_forecasts[model.name] = forecast
                
            except Exception as e:
                logger.warning(f"Error with model {model.name}: {str(e)}")
                continue
        
        if not model_forecasts:
            logger.error("No successful model forecasts")
            return {}
        
        # Convert to numpy array
        forecasts_array = np.array(model_forecasts)
        
        # Calculate model uncertainty metrics
        model_uncertainty = {
            'mean': np.mean(forecasts_array, axis=0),
            'std': np.std(forecasts_array, axis=0),
            'min': np.min(forecasts_array, axis=0),
            'max': np.max(forecasts_array, axis=0),
            'range': np.max(forecasts_array, axis=0) - np.min(forecasts_array, axis=0),
            'model_agreement': 1.0 - (np.std(forecasts_array, axis=0) / np.mean(forecasts_array, axis=0)),
            'individual_forecasts': {name: forecast for name, forecast in zip(model_names, model_forecasts)}
        }
        
        # Calculate model-weighted forecast if weights are provided
        if self.model_weights:
            weighted_forecast = np.zeros(periods)
            total_weight = 0
            
            for i, name in enumerate(model_names):
                weight = self.model_weights.get(name, 1.0)
                weighted_forecast += weight * model_forecasts[i]
                total_weight += weight
            
            if total_weight > 0:
                model_uncertainty['weighted_mean'] = weighted_forecast / total_weight
        
        return model_uncertainty


class TimeVaryingConfidenceCalculator:
    """
    Time-varying confidence interval calculator that adjusts bounds based on forecast horizon
    """
    
    def __init__(self, 
                 base_confidence: float = 0.95,
                 horizon_adjustment: str = 'sqrt_time',
                 volatility_scaling: bool = True):
        """
        Initialize time-varying confidence calculator
        
        Args:
            base_confidence: Base confidence level
            horizon_adjustment: Method for horizon adjustment ('sqrt_time', 'linear', 'exponential')
            volatility_scaling: Whether to scale by volatility
        """
        self.base_confidence = base_confidence
        self.horizon_adjustment = horizon_adjustment
        self.volatility_scaling = volatility_scaling
        
    def calculate_time_varying_ci(self,
                                 forecast_values: np.ndarray,
                                 base_std: float,
                                 periods: int,
                                 historical_volatility: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Calculate time-varying confidence intervals
        
        Args:
            forecast_values: Point forecasts
            base_std: Base standard deviation
            periods: Number of forecast periods
            historical_volatility: Historical volatility measure
            
        Returns:
            Dictionary with time-varying confidence bounds
        """
        # Calculate alpha for confidence level
        alpha = 1 - self.base_confidence
        z_score = norm.ppf(1 - alpha/2)
        
        # Initialize arrays
        lower_bounds = np.zeros(periods)
        upper_bounds = np.zeros(periods)
        std_adjustments = np.zeros(periods)
        
        # Calculate time-varying adjustments
        for i in range(periods):
            periods_ahead = i + 1
            
            # Base adjustment based on horizon
            if self.horizon_adjustment == 'sqrt_time':
                time_adjustment = np.sqrt(periods_ahead)
            elif self.horizon_adjustment == 'linear':
                time_adjustment = periods_ahead
            elif self.horizon_adjustment == 'exponential':
                time_adjustment = np.exp(periods_ahead * 0.1)
            else:
                time_adjustment = 1.0
            
            # Volatility scaling
            if self.volatility_scaling and historical_volatility is not None:
                volatility_adjustment = 1.0 + (historical_volatility * periods_ahead * 0.1)
            else:
                volatility_adjustment = 1.0
            
            # Combined standard deviation adjustment
            adjusted_std = base_std * time_adjustment * volatility_adjustment
            std_adjustments[i] = adjusted_std
            
            # Calculate bounds
            margin = z_score * adjusted_std
            lower_bounds[i] = forecast_values[i] - margin
            upper_bounds[i] = forecast_values[i] + margin
        
        return {
            'lower': lower_bounds,
            'upper': upper_bounds,
            'std_adjustments': std_adjustments,
            'forecast_values': forecast_values
        }


class EnhancedConfidenceCalculator:
    """
    Comprehensive enhanced confidence interval calculator
    
    Combines multiple uncertainty quantification methods for robust confidence estimation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced confidence calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize sub-calculators
        self.bootstrap_calc = BootstrapConfidenceCalculator(
            n_bootstrap=self.config.get('n_bootstrap', 500),
            confidence_levels=self.config.get('confidence_levels', [0.025, 0.975])
        )
        
        self.model_uncertainty_calc = ModelUncertaintyCalculator()
        
        self.time_varying_calc = TimeVaryingConfidenceCalculator(
            base_confidence=self.config.get('base_confidence', 0.95),
            horizon_adjustment=self.config.get('horizon_adjustment', 'sqrt_time'),
            volatility_scaling=self.config.get('volatility_scaling', True)
        )
        
        # Method weights for combining different uncertainty sources
        self.method_weights = self.config.get('method_weights', {
            'bootstrap': 0.4,
            'model_uncertainty': 0.3,
            'time_varying': 0.3
        })
        
        # Storage
        self.confidence_results = {}
        
    def calculate_enhanced_confidence(self,
                                    forecaster: BaseForecaster,
                                    data: pd.DataFrame,
                                    periods: int,
                                    frequency: str = 'Y',
                                    alternative_models: List[BaseForecaster] = None) -> Dict[str, Any]:
        """
        Calculate enhanced confidence intervals using multiple methods
        
        Args:
            forecaster: Primary forecaster
            data: Historical data
            periods: Number of periods to forecast
            frequency: Forecast frequency
            alternative_models: Alternative models for model uncertainty
            
        Returns:
            Comprehensive confidence results
        """
        logger.info("Calculating enhanced confidence intervals")
        
        # Generate primary forecast
        forecaster.fit(data)
        primary_forecast = forecaster.forecast(periods, frequency)
        forecast_values = primary_forecast['value'].values
        
        results = {
            'primary_forecast': primary_forecast,
            'methods': {},
            'combined': {}
        }
        
        # Method 1: Bootstrap confidence intervals
        try:
            bootstrap_ci = self.bootstrap_calc.calculate_bootstrap_ci(
                data, forecaster, periods, frequency
            )
            results['methods']['bootstrap'] = bootstrap_ci
            logger.info("Completed bootstrap confidence calculation")
        except Exception as e:
            logger.error(f"Error in bootstrap calculation: {str(e)}")
            results['methods']['bootstrap'] = {}
        
        # Method 2: Model uncertainty
        if alternative_models:
            try:
                # Add alternative models
                for model in alternative_models:
                    self.model_uncertainty_calc.add_model(model)
                
                model_uncertainty = self.model_uncertainty_calc.calculate_model_uncertainty(
                    data, periods, frequency
                )
                results['methods']['model_uncertainty'] = model_uncertainty
                logger.info("Completed model uncertainty calculation")
            except Exception as e:
                logger.error(f"Error in model uncertainty calculation: {str(e)}")
                results['methods']['model_uncertainty'] = {}
        
        # Method 3: Time-varying confidence intervals
        try:
            # Calculate historical volatility
            if 'value' in data.columns:
                historical_returns = data['value'].pct_change().dropna()
                historical_volatility = historical_returns.std()
            else:
                historical_volatility = None
            
            # Estimate base standard deviation from residuals or model spread
            if 'bootstrap' in results['methods'] and 'std' in results['methods']['bootstrap']:
                base_std = np.mean(results['methods']['bootstrap']['std'])
            elif 'model_uncertainty' in results['methods'] and 'std' in results['methods']['model_uncertainty']:
                base_std = np.mean(results['methods']['model_uncertainty']['std'])
            else:
                # Fallback: use percentage of forecast values
                base_std = np.mean(forecast_values) * 0.1
            
            time_varying_ci = self.time_varying_calc.calculate_time_varying_ci(
                forecast_values, base_std, periods, historical_volatility
            )
            results['methods']['time_varying'] = time_varying_ci
            logger.info("Completed time-varying confidence calculation")
        except Exception as e:
            logger.error(f"Error in time-varying calculation: {str(e)}")
            results['methods']['time_varying'] = {}
        
        # Combine methods
        combined_ci = self._combine_confidence_methods(results['methods'], forecast_values, periods)
        results['combined'] = combined_ci
        
        # Store results
        self.confidence_results = results
        
        return results
    
    def _combine_confidence_methods(self,
                                  method_results: Dict[str, Dict],
                                  forecast_values: np.ndarray,
                                  periods: int) -> Dict[str, np.ndarray]:
        """
        Combine confidence intervals from different methods
        
        Args:
            method_results: Results from different methods
            forecast_values: Point forecast values
            periods: Number of periods
            
        Returns:
            Combined confidence intervals
        """
        # Initialize arrays for combined bounds
        combined_lower = np.zeros(periods)
        combined_upper = np.zeros(periods)
        combined_std = np.zeros(periods)
        
        # Weights for combination
        total_weight = 0
        method_contributions = {}
        
        # Bootstrap contribution
        if 'bootstrap' in method_results and method_results['bootstrap']:
            bootstrap_res = method_results['bootstrap']
            weight = self.method_weights.get('bootstrap', 0)
            
            if 'p0.025' in bootstrap_res and 'p0.975' in bootstrap_res:
                combined_lower += weight * bootstrap_res['p0.025']
                combined_upper += weight * bootstrap_res['p0.975']
                total_weight += weight
                method_contributions['bootstrap'] = weight
            elif 'std' in bootstrap_res:
                # Fallback: use standard deviation
                margin = 1.96 * bootstrap_res['std']
                combined_lower += weight * (forecast_values - margin)
                combined_upper += weight * (forecast_values + margin)
                total_weight += weight
                method_contributions['bootstrap'] = weight
        
        # Model uncertainty contribution
        if 'model_uncertainty' in method_results and method_results['model_uncertainty']:
            model_res = method_results['model_uncertainty']
            weight = self.method_weights.get('model_uncertainty', 0)
            
            if 'std' in model_res:
                margin = 1.96 * model_res['std']
                combined_lower += weight * (forecast_values - margin)
                combined_upper += weight * (forecast_values + margin)
                total_weight += weight
                method_contributions['model_uncertainty'] = weight
        
        # Time-varying contribution
        if 'time_varying' in method_results and method_results['time_varying']:
            time_res = method_results['time_varying']
            weight = self.method_weights.get('time_varying', 0)
            
            if 'lower' in time_res and 'upper' in time_res:
                combined_lower += weight * time_res['lower']
                combined_upper += weight * time_res['upper']
                total_weight += weight
                method_contributions['time_varying'] = weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_lower /= total_weight
            combined_upper /= total_weight
        else:
            # Fallback: simple percentage bounds
            margin = forecast_values * 0.2  # 20% margin
            combined_lower = forecast_values - margin
            combined_upper = forecast_values + margin
        
        # Calculate combined standard deviation
        combined_std = (combined_upper - combined_lower) / (2 * 1.96)
        
        # Ensure monotonic increasing uncertainty
        for i in range(1, periods):
            if combined_std[i] < combined_std[i-1]:
                combined_std[i] = combined_std[i-1] * 1.05  # Small increase
                
                # Recalculate bounds
                margin = 1.96 * combined_std[i]
                combined_lower[i] = forecast_values[i] - margin
                combined_upper[i] = forecast_values[i] + margin
        
        return {
            'lower': combined_lower,
            'upper': combined_upper,
            'std': combined_std,
            'forecast_values': forecast_values,
            'method_contributions': method_contributions,
            'total_weight': total_weight
        }
    
    def get_confidence_dataframe(self, dates: List[datetime] = None) -> pd.DataFrame:
        """
        Get confidence intervals as a DataFrame
        
        Args:
            dates: Optional list of forecast dates
            
        Returns:
            DataFrame with confidence intervals
        """
        if not self.confidence_results:
            logger.warning("No confidence results available")
            return pd.DataFrame()
        
        combined = self.confidence_results.get('combined', {})
        
        if not combined:
            return pd.DataFrame()
        
        # Create DataFrame
        data = {
            'lower': combined.get('lower', []),
            'value': combined.get('forecast_values', []),
            'upper': combined.get('upper', []),
            'std': combined.get('std', [])
        }
        
        # Add dates if provided
        if dates:
            data['date'] = dates[:len(data['value'])]
        
        return pd.DataFrame(data)
    
    def visualize_confidence_intervals(self, 
                                     historical_data: pd.DataFrame = None,
                                     save_path: Optional[str] = None) -> str:
        """
        Visualize enhanced confidence intervals
        
        Args:
            historical_data: Optional historical data to plot
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not self.confidence_results:
            logger.warning("No confidence results to visualize")
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Confidence Intervals Analysis', fontsize=16)
        
        # Get results
        primary_forecast = self.confidence_results.get('primary_forecast')
        methods = self.confidence_results.get('methods', {})
        combined = self.confidence_results.get('combined', {})
        
        if not primary_forecast or combined == {}:
            return ""
        
        forecast_dates = primary_forecast['date']
        forecast_values = primary_forecast['value']
        
        # Plot 1: Combined confidence intervals
        ax1 = axes[0, 0]
        
        # Historical data
        if historical_data is not None and 'date' in historical_data.columns and 'value' in historical_data.columns:
            ax1.plot(historical_data['date'], historical_data['value'], 'k-', linewidth=2, label='Historical')
        
        # Primary forecast
        ax1.plot(forecast_dates, forecast_values, 'b-', linewidth=3, label='Forecast')
        
        # Combined confidence intervals
        if 'lower' in combined and 'upper' in combined:
            ax1.fill_between(forecast_dates, combined['lower'], combined['upper'],
                           alpha=0.3, color='blue', label='Enhanced CI')
        
        ax1.set_title('Enhanced Confidence Intervals')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Method comparison
        ax2 = axes[0, 1]
        method_names = []
        method_widths = []
        
        for method_name, method_data in methods.items():
            if method_name == 'bootstrap' and 'p0.025' in method_data and 'p0.975' in method_data:
                width = np.mean(method_data['p0.975'] - method_data['p0.025'])
                method_names.append('Bootstrap')
                method_widths.append(width)
            elif method_name == 'model_uncertainty' and 'std' in method_data:
                width = np.mean(method_data['std'] * 2 * 1.96)
                method_names.append('Model Uncertainty')
                method_widths.append(width)
            elif method_name == 'time_varying' and 'lower' in method_data and 'upper' in method_data:
                width = np.mean(method_data['upper'] - method_data['lower'])
                method_names.append('Time Varying')
                method_widths.append(width)
        
        if method_names:
            ax2.bar(method_names, method_widths, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
            ax2.set_title('Confidence Interval Width by Method')
            ax2.set_ylabel('Average CI Width')
            ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Uncertainty evolution
        ax3 = axes[1, 0]
        
        if 'std' in combined:
            periods = range(1, len(combined['std']) + 1)
            ax3.plot(periods, combined['std'], 'r-', linewidth=2, marker='o')
            ax3.set_title('Uncertainty Evolution Over Time')
            ax3.set_xlabel('Periods Ahead')
            ax3.set_ylabel('Standard Deviation')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Method contributions
        ax4 = axes[1, 1]
        
        contributions = combined.get('method_contributions', {})
        if contributions:
            methods = list(contributions.keys())
            weights = list(contributions.values())
            
            ax4.pie(weights, labels=methods, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Method Contributions to Combined CI')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        if save_path:
            output_path = save_path
        else:
            output_path = f"enhanced_confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved enhanced confidence visualization to {output_path}")
        return output_path


class UncertaintyPropagator:
    """
    Uncertainty propagation through the entire forecasting pipeline
    
    This class tracks and propagates uncertainty from data through models to final forecasts
    """
    
    def __init__(self, components: Dict[str, Any] = None):
        """
        Initialize uncertainty propagator
        
        Args:
            components: Dictionary of forecasting pipeline components
        """
        self.components = components or {}
        self.uncertainty_sources = {}
        self.propagated_uncertainty = {}
        
    def add_uncertainty_source(self, 
                             source_name: str,
                             uncertainty_type: str,
                             uncertainty_params: Dict[str, Any]):
        """
        Add an uncertainty source to track
        
        Args:
            source_name: Name of the uncertainty source
            uncertainty_type: Type of uncertainty ('data', 'parameter', 'model', 'structural')
            uncertainty_params: Parameters defining the uncertainty
        """
        self.uncertainty_sources[source_name] = {
            'type': uncertainty_type,
            'params': uncertainty_params
        }
    
    def propagate_uncertainty(self, 
                            forecast_pipeline: Callable,
                            input_data: pd.DataFrame,
                            n_samples: int = 1000) -> Dict[str, Any]:
        """
        Propagate uncertainty through the forecasting pipeline
        
        Args:
            forecast_pipeline: Function representing the forecasting pipeline
            input_data: Input data
            n_samples: Number of uncertainty samples
            
        Returns:
            Propagated uncertainty results
        """
        logger.info(f"Propagating uncertainty through pipeline with {n_samples} samples")
        
        # Generate uncertainty samples
        uncertainty_samples = self._generate_uncertainty_samples(n_samples)
        
        # Run pipeline with each uncertainty sample
        pipeline_outputs = []
        
        for i, sample in enumerate(uncertainty_samples):
            try:
                # Apply uncertainty to input data and components
                perturbed_data, perturbed_components = self._apply_uncertainty_sample(
                    input_data, sample
                )
                
                # Run forecasting pipeline
                output = forecast_pipeline(perturbed_data, perturbed_components)
                pipeline_outputs.append(output)
                
            except Exception as e:
                logger.warning(f"Error in uncertainty sample {i}: {str(e)}")
                continue
        
        # Analyze propagated uncertainty
        propagated_results = self._analyze_propagated_uncertainty(pipeline_outputs)
        
        self.propagated_uncertainty = propagated_results
        return propagated_results
    
    def _generate_uncertainty_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate samples from all uncertainty sources"""
        samples = []
        
        for i in range(n_samples):
            sample = {}
            
            for source_name, source_info in self.uncertainty_sources.items():
                uncertainty_type = source_info['type']
                params = source_info['params']
                
                if uncertainty_type == 'data':
                    # Data uncertainty (e.g., measurement noise)
                    sample[source_name] = np.random.normal(
                        params.get('mean', 0),
                        params.get('std', 0.05)
                    )
                
                elif uncertainty_type == 'parameter':
                    # Parameter uncertainty
                    sample[source_name] = np.random.normal(
                        params.get('mean', 1.0),
                        params.get('std', 0.1)
                    )
                
                elif uncertainty_type == 'model':
                    # Model selection uncertainty
                    models = params.get('models', [])
                    if models:
                        sample[source_name] = np.random.choice(models)
                
                elif uncertainty_type == 'structural':
                    # Structural uncertainty
                    sample[source_name] = np.random.uniform(
                        params.get('min', 0.8),
                        params.get('max', 1.2)
                    )
            
            samples.append(sample)
        
        return samples
    
    def _apply_uncertainty_sample(self, 
                                input_data: pd.DataFrame,
                                sample: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply uncertainty sample to data and components"""
        # Create perturbed data
        perturbed_data = input_data.copy()
        
        # Apply data uncertainties
        for source_name, value in sample.items():
            if source_name.startswith('data_'):
                # Apply to relevant data columns
                if 'value' in perturbed_data.columns:
                    perturbed_data['value'] *= (1 + value)
        
        # Create perturbed components
        perturbed_components = self.components.copy()
        
        # Apply parameter uncertainties
        for source_name, value in sample.items():
            if source_name.startswith('param_'):
                # Apply to relevant parameters
                param_name = source_name.replace('param_', '')
                if param_name in perturbed_components:
                    perturbed_components[param_name] *= value
        
        return perturbed_data, perturbed_components
    
    def _analyze_propagated_uncertainty(self, 
                                      pipeline_outputs: List[Any]) -> Dict[str, Any]:
        """Analyze uncertainty propagated through the pipeline"""
        if not pipeline_outputs:
            return {}
        
        # Convert outputs to numerical arrays if possible
        try:
            if isinstance(pipeline_outputs[0], pd.DataFrame):
                # Extract key columns
                output_arrays = []
                for output in pipeline_outputs:
                    if 'value' in output.columns:
                        output_arrays.append(output['value'].values)
                
                if output_arrays:
                    output_matrix = np.array(output_arrays)
                    
                    results = {
                        'mean': np.mean(output_matrix, axis=0),
                        'std': np.std(output_matrix, axis=0),
                        'percentile_5': np.percentile(output_matrix, 5, axis=0),
                        'percentile_25': np.percentile(output_matrix, 25, axis=0),
                        'percentile_75': np.percentile(output_matrix, 75, axis=0),
                        'percentile_95': np.percentile(output_matrix, 95, axis=0),
                        'n_samples': len(pipeline_outputs)
                    }
                    
                    return results
        
        except Exception as e:
            logger.warning(f"Error analyzing propagated uncertainty: {str(e)}")
        
        return {'n_samples': len(pipeline_outputs)}