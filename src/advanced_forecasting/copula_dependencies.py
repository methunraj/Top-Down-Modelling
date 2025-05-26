"""
Copula-Based Parameter Dependencies - Advanced correlation modeling

This module implements copula-based methods for modeling complex dependencies
between parameters while preserving their marginal distributions.

Key capabilities:
1. Gaussian copulas for linear correlations
2. Student-t copulas for tail dependencies
3. Archimedean copulas (Clayton, Gumbel, Frank)
4. Vine copulas for high-dimensional dependencies
5. Parameter correlation preservation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from scipy import stats
from scipy.stats import multivariate_normal, multivariate_t
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CopulaDistribution:
    """
    Copula-based distribution for modeling parameter dependencies
    
    Separates marginal distributions from dependence structure, allowing
    complex correlations while preserving individual parameter properties.
    """
    
    def __init__(self, 
                 marginal_distributions: Dict[str, Any],
                 copula_type: str = 'gaussian',
                 correlation_matrix: Optional[np.ndarray] = None):
        """
        Initialize copula distribution
        
        Args:
            marginal_distributions: Dictionary of parameter distributions
            copula_type: Type of copula ('gaussian', 'student_t', 'clayton', 'gumbel', 'frank')
            correlation_matrix: Correlation matrix for the copula
        """
        self.marginal_distributions = marginal_distributions
        self.copula_type = copula_type
        self.parameter_names = list(marginal_distributions.keys())
        self.n_parameters = len(self.parameter_names)
        
        # Initialize correlation matrix
        if correlation_matrix is not None:
            self.correlation_matrix = self._validate_correlation_matrix(correlation_matrix)
        else:
            self.correlation_matrix = np.eye(self.n_parameters)
        
        # Copula-specific parameters
        self.copula_params = {}
        self._initialize_copula_parameters()
        
        # Sampling statistics
        self.sampling_stats = {
            'total_samples': 0,
            'correlation_preserved': True,
            'tail_dependence': self._calculate_tail_dependence()
        }
        
        logger.info(f"Initialized {copula_type} copula with {self.n_parameters} parameters")
    
    def _validate_correlation_matrix(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Validate and adjust correlation matrix"""
        # Check dimensions
        if correlation_matrix.shape != (self.n_parameters, self.n_parameters):
            raise ValueError(f"Correlation matrix must be {self.n_parameters}x{self.n_parameters}")
        
        # Check symmetry
        if not np.allclose(correlation_matrix, correlation_matrix.T):
            logger.warning("Correlation matrix not symmetric, making symmetric")
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        
        # Check positive semi-definite
        eigenvals = np.linalg.eigvals(correlation_matrix)
        if np.any(eigenvals < -1e-8):
            logger.warning("Correlation matrix not positive semi-definite, adjusting")
            # Use nearest positive semi-definite matrix
            correlation_matrix = self._nearest_positive_semidefinite(correlation_matrix)
        
        # Ensure diagonal is 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return correlation_matrix
    
    def _nearest_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """Find nearest positive semi-definite matrix"""
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        
        # Set negative eigenvalues to small positive value
        eigenvals = np.maximum(eigenvals, 1e-8)
        
        # Reconstruct matrix
        result = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Ensure diagonal is 1 (correlation matrix property)
        scaling = np.sqrt(np.diag(result))
        result = result / np.outer(scaling, scaling)
        
        return result
    
    def _initialize_copula_parameters(self):
        """Initialize copula-specific parameters"""
        if self.copula_type == 'gaussian':
            # Gaussian copula uses correlation matrix directly
            self.copula_params['correlation'] = self.correlation_matrix
        
        elif self.copula_type == 'student_t':
            # Student-t copula: correlation matrix + degrees of freedom
            self.copula_params['correlation'] = self.correlation_matrix
            self.copula_params['df'] = 5.0  # Default degrees of freedom
        
        elif self.copula_type == 'clayton':
            # Clayton copula parameter (must be positive)
            self.copula_params['theta'] = 1.0
        
        elif self.copula_type == 'gumbel':
            # Gumbel copula parameter (must be >= 1)
            self.copula_params['theta'] = 2.0
        
        elif self.copula_type == 'frank':
            # Frank copula parameter (can be any real number)
            self.copula_params['theta'] = 2.0
        
        else:
            raise ValueError(f"Unsupported copula type: {self.copula_type}")
    
    def _calculate_tail_dependence(self) -> Dict[str, float]:
        """Calculate tail dependence coefficients"""
        if self.copula_type == 'gaussian':
            return {'upper': 0.0, 'lower': 0.0}  # No tail dependence
        elif self.copula_type == 'student_t':
            # Symmetric tail dependence
            df = self.copula_params.get('df', 5)
            rho = self.correlation_matrix[0, 1] if self.n_parameters >= 2 else 0
            tail_dep = 2 * stats.t.cdf(-np.sqrt((df + 1) * (1 - rho) / (1 + rho)), df + 1)
            return {'upper': tail_dep, 'lower': tail_dep}
        elif self.copula_type == 'clayton':
            theta = self.copula_params['theta']
            return {'upper': 0.0, 'lower': 2**(-1/theta)}
        elif self.copula_type == 'gumbel':
            theta = self.copula_params['theta']
            return {'upper': 2 - 2**(1/theta), 'lower': 0.0}
        elif self.copula_type == 'frank':
            return {'upper': 0.0, 'lower': 0.0}  # No tail dependence
        else:
            return {'upper': 0.0, 'lower': 0.0}
    
    def sample_correlated_parameters(self, 
                                   n_samples: int,
                                   method: str = 'inverse_cdf') -> Dict[str, np.ndarray]:
        """
        Generate correlated parameter samples using copula
        
        Args:
            n_samples: Number of samples to generate
            method: Sampling method ('inverse_cdf', 'conditional')
            
        Returns:
            Dictionary of correlated parameter samples
        """
        logger.info(f"Generating {n_samples} correlated samples using {self.copula_type} copula")
        
        # Generate copula samples (uniform margins)
        if self.copula_type in ['gaussian', 'student_t']:
            uniform_samples = self._sample_elliptical_copula(n_samples)
        elif self.copula_type in ['clayton', 'gumbel', 'frank']:
            uniform_samples = self._sample_archimedean_copula(n_samples)
        else:
            raise ValueError(f"Sampling not implemented for {self.copula_type} copula")
        
        # Transform to marginal distributions
        correlated_samples = {}
        for i, param_name in enumerate(self.parameter_names):
            marginal_dist = self.marginal_distributions[param_name]
            
            # Transform uniform samples to marginal distribution
            if hasattr(marginal_dist, 'ppf'):
                # scipy distribution
                correlated_samples[param_name] = marginal_dist.ppf(uniform_samples[:, i])
            elif hasattr(marginal_dist, 'percentile'):
                # Custom distribution with percentile method
                correlated_samples[param_name] = np.array([
                    marginal_dist.percentile(u) for u in uniform_samples[:, i]
                ])
            else:
                logger.warning(f"Unknown distribution type for {param_name}, using uniform samples")
                correlated_samples[param_name] = uniform_samples[:, i]
        
        # Update sampling statistics
        self.sampling_stats['total_samples'] += n_samples
        
        # Verify correlation preservation
        self._verify_correlation_preservation(correlated_samples)
        
        return correlated_samples
    
    def _sample_elliptical_copula(self, n_samples: int) -> np.ndarray:
        """Sample from elliptical copulas (Gaussian, Student-t)"""
        if self.copula_type == 'gaussian':
            # Multivariate normal with given correlation
            mvn_samples = multivariate_normal.rvs(
                mean=np.zeros(self.n_parameters),
                cov=self.correlation_matrix,
                size=n_samples
            )
            
            # Transform to uniform using standard normal CDF
            uniform_samples = stats.norm.cdf(mvn_samples)
        
        elif self.copula_type == 'student_t':
            # Multivariate t-distribution
            df = self.copula_params['df']
            
            # Generate multivariate t samples
            mvt_samples = multivariate_t.rvs(
                loc=np.zeros(self.n_parameters),
                shape=self.correlation_matrix,
                df=df,
                size=n_samples
            )
            
            # Transform to uniform using t CDF
            uniform_samples = stats.t.cdf(mvt_samples, df)
        
        else:
            raise ValueError(f"Unknown elliptical copula: {self.copula_type}")
        
        return uniform_samples
    
    def _sample_archimedean_copula(self, n_samples: int) -> np.ndarray:
        """Sample from Archimedean copulas (Clayton, Gumbel, Frank)"""
        if self.n_parameters != 2:
            raise NotImplementedError("Archimedean copulas currently support only 2 parameters")
        
        theta = self.copula_params['theta']
        
        if self.copula_type == 'clayton':
            # Clayton copula sampling
            u1 = np.random.uniform(0, 1, n_samples)
            u2 = np.random.uniform(0, 1, n_samples)
            
            # Clayton copula inverse conditional distribution
            v2 = (u1**(-theta) * (u2**(-theta/(1+theta)) - 1) + 1)**(-1/theta)
            
            uniform_samples = np.column_stack([u1, v2])
        
        elif self.copula_type == 'gumbel':
            # Gumbel copula sampling using Marshall-Olkin method
            u1 = np.random.uniform(0, 1, n_samples)
            w = np.random.uniform(0, 1, n_samples)
            
            # Stable distribution sampling (approximation)
            stable_samples = self._sample_stable_distribution(n_samples, 1/theta)
            
            u2 = np.exp(-((- np.log(u1))**theta + stable_samples)**(1/theta))
            
            uniform_samples = np.column_stack([u1, u2])
        
        elif self.copula_type == 'frank':
            # Frank copula sampling
            u1 = np.random.uniform(0, 1, n_samples)
            w = np.random.uniform(0, 1, n_samples)
            
            if abs(theta) > 1e-6:
                u2 = -1/theta * np.log(1 + w * (np.exp(-theta) - 1) / 
                                      (w * (np.exp(-theta * u1) - 1) + np.exp(-theta * u1)))
            else:
                # Independence case
                u2 = w
            
            uniform_samples = np.column_stack([u1, u2])
        
        else:
            raise ValueError(f"Unknown Archimedean copula: {self.copula_type}")
        
        return uniform_samples
    
    def _sample_stable_distribution(self, n_samples: int, alpha: float) -> np.ndarray:
        """Sample from stable distribution (approximation for Gumbel copula)"""
        # Approximation using normal distribution for simplicity
        # In practice, use proper stable distribution sampling
        return np.random.normal(0, 1, n_samples)
    
    def _verify_correlation_preservation(self, samples: Dict[str, np.ndarray]):
        """Verify that correlation structure is preserved in samples"""
        if len(samples) < 2:
            return
        
        # Calculate sample correlation matrix
        sample_data = np.column_stack([samples[name] for name in self.parameter_names])
        sample_corr = np.corrcoef(sample_data.T)
        
        # Compare with target correlation (for Gaussian copula)
        if self.copula_type == 'gaussian':
            max_diff = np.max(np.abs(sample_corr - self.correlation_matrix))
            if max_diff > 0.1:  # Tolerance for correlation preservation
                logger.warning(f"Correlation preservation issue: max difference = {max_diff:.3f}")
                self.sampling_stats['correlation_preserved'] = False
            else:
                self.sampling_stats['correlation_preserved'] = True
    
    def fit_copula_to_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit copula parameters to observed data
        
        Args:
            data: DataFrame with parameter observations
            
        Returns:
            Fitted copula parameters
        """
        logger.info(f"Fitting {self.copula_type} copula to data")
        
        # Transform data to uniform margins (pseudo-observations)
        uniform_data = self._transform_to_uniform_margins(data)
        
        # Fit copula parameters
        if self.copula_type == 'gaussian':
            # Estimate correlation matrix
            sample_corr = np.corrcoef(stats.norm.ppf(uniform_data).T)
            self.correlation_matrix = self._validate_correlation_matrix(sample_corr)
            self.copula_params['correlation'] = self.correlation_matrix
        
        elif self.copula_type == 'student_t':
            # Estimate correlation and degrees of freedom
            normal_data = stats.norm.ppf(uniform_data)
            sample_corr = np.corrcoef(normal_data.T)
            
            # Estimate degrees of freedom using maximum likelihood
            def neg_log_likelihood(df):
                try:
                    return -np.sum(multivariate_t.logpdf(normal_data, 
                                                       loc=np.zeros(self.n_parameters),
                                                       shape=sample_corr, 
                                                       df=df))
                except:
                    return np.inf
            
            result = minimize(neg_log_likelihood, x0=[5.0], bounds=[(2.1, 30.0)])
            
            self.correlation_matrix = self._validate_correlation_matrix(sample_corr)
            self.copula_params['correlation'] = self.correlation_matrix
            self.copula_params['df'] = result.x[0]
        
        elif self.copula_type in ['clayton', 'gumbel', 'frank']:
            # Estimate theta parameter for Archimedean copulas
            if uniform_data.shape[1] == 2:
                theta_estimate = self._estimate_archimedean_parameter(uniform_data)
                self.copula_params['theta'] = theta_estimate
            else:
                logger.warning("Archimedean copulas support only 2 parameters, using default theta")
        
        # Update tail dependence
        self.sampling_stats['tail_dependence'] = self._calculate_tail_dependence()
        
        return self.copula_params
    
    def _transform_to_uniform_margins(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data to uniform margins using empirical CDFs"""
        uniform_data = np.zeros_like(data.values)
        
        for i, col in enumerate(data.columns):
            if col in self.parameter_names:
                # Use rank-based transformation (empirical CDF)
                ranks = stats.rankdata(data[col])
                uniform_data[:, i] = ranks / (len(ranks) + 1)
        
        return uniform_data
    
    def _estimate_archimedean_parameter(self, uniform_data: np.ndarray) -> float:
        """Estimate parameter for Archimedean copulas"""
        # Calculate Kendall's tau
        tau = stats.kendalltau(uniform_data[:, 0], uniform_data[:, 1])[0]
        
        # Convert Kendall's tau to copula parameter
        if self.copula_type == 'clayton':
            # tau = theta / (theta + 2)
            if tau > 0:
                theta = 2 * tau / (1 - tau)
                return max(theta, 1e-6)  # Ensure positive
            else:
                return 1e-6
        
        elif self.copula_type == 'gumbel':
            # tau = 1 - 1/theta
            if tau > 0:
                theta = 1 / (1 - tau)
                return max(theta, 1.001)  # Ensure >= 1
            else:
                return 1.001
        
        elif self.copula_type == 'frank':
            # tau = 1 - 4/theta + 4*D_1(theta)/theta, where D_1 is Debye function
            # Approximation for Frank copula
            if abs(tau) < 0.99:
                theta = 4 * tau / (1 - tau)  # Simplified approximation
                return theta
            else:
                return 2.0
        
        return 1.0
    
    def calculate_dependence_measures(self, samples: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Calculate various dependence measures from samples
        
        Args:
            samples: Dictionary of parameter samples
            
        Returns:
            Dictionary of dependence measures
        """
        if len(samples) < 2:
            return {}
        
        # Prepare data
        data_matrix = np.column_stack([samples[name] for name in self.parameter_names])
        
        dependence_measures = {}
        
        # Pearson correlation
        pearson_corr = np.corrcoef(data_matrix.T)
        dependence_measures['pearson_correlation'] = pearson_corr
        
        # Spearman correlation (rank correlation)
        spearman_corr = stats.spearmanr(data_matrix)[0]
        if np.isscalar(spearman_corr):
            spearman_corr = np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])
        dependence_measures['spearman_correlation'] = spearman_corr
        
        # Kendall's tau
        if data_matrix.shape[1] == 2:
            tau = stats.kendalltau(data_matrix[:, 0], data_matrix[:, 1])[0]
            dependence_measures['kendall_tau'] = tau
        
        # Tail dependence (empirical)
        tail_dep = self._estimate_tail_dependence(data_matrix)
        dependence_measures['tail_dependence'] = tail_dep
        
        return dependence_measures
    
    def _estimate_tail_dependence(self, data_matrix: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
        """Estimate empirical tail dependence"""
        if data_matrix.shape[1] != 2:
            return {'upper': 0.0, 'lower': 0.0}
        
        n_samples = len(data_matrix)
        
        # Transform to uniform margins
        u1 = stats.rankdata(data_matrix[:, 0]) / (n_samples + 1)
        u2 = stats.rankdata(data_matrix[:, 1]) / (n_samples + 1)
        
        # Upper tail dependence
        upper_threshold = 1 - threshold
        upper_tail_count = np.sum((u1 > upper_threshold) & (u2 > upper_threshold))
        lower_tail_count = np.sum(u1 > upper_threshold)
        upper_tail_dep = upper_tail_count / lower_tail_count if lower_tail_count > 0 else 0
        
        # Lower tail dependence
        lower_tail_count_both = np.sum((u1 < threshold) & (u2 < threshold))
        lower_tail_count_single = np.sum(u1 < threshold)
        lower_tail_dep = lower_tail_count_both / lower_tail_count_single if lower_tail_count_single > 0 else 0
        
        return {'upper': upper_tail_dep, 'lower': lower_tail_dep}
    
    def get_copula_info(self) -> Dict[str, Any]:
        """Get comprehensive copula information"""
        return {
            'copula_type': self.copula_type,
            'n_parameters': self.n_parameters,
            'parameter_names': self.parameter_names,
            'correlation_matrix': self.correlation_matrix.tolist(),
            'copula_parameters': self.copula_params,
            'tail_dependence': self.sampling_stats['tail_dependence'],
            'sampling_stats': self.sampling_stats
        }


class VineCopula:
    """
    Vine copula for high-dimensional dependencies
    
    Builds complex multivariate dependencies using sequence of bivariate copulas
    arranged in a vine structure (C-vine or D-vine).
    """
    
    def __init__(self, 
                 marginal_distributions: Dict[str, Any],
                 vine_type: str = 'c_vine'):
        """
        Initialize vine copula
        
        Args:
            marginal_distributions: Dictionary of marginal distributions
            vine_type: Type of vine ('c_vine' or 'd_vine')
        """
        self.marginal_distributions = marginal_distributions
        self.vine_type = vine_type
        self.parameter_names = list(marginal_distributions.keys())
        self.n_parameters = len(self.parameter_names)
        
        # Vine structure
        self.vine_structure = self._initialize_vine_structure()
        self.bivariate_copulas = {}
        
        logger.info(f"Initialized {vine_type} vine copula with {self.n_parameters} parameters")
    
    def _initialize_vine_structure(self) -> List[List[Tuple[int, int]]]:
        """Initialize vine structure (tree sequence)"""
        if self.vine_type == 'c_vine':
            # Canonical vine: star structure at each level
            structure = []
            for level in range(self.n_parameters - 1):
                level_pairs = []
                root = level  # Root variable for this level
                for i in range(level + 1, self.n_parameters):
                    level_pairs.append((root, i))
                structure.append(level_pairs)
        
        elif self.vine_type == 'd_vine':
            # D-vine: path structure
            structure = []
            for level in range(self.n_parameters - 1):
                level_pairs = []
                for i in range(self.n_parameters - level - 1):
                    level_pairs.append((i, i + level + 1))
                structure.append(level_pairs)
        
        else:
            raise ValueError(f"Unsupported vine type: {self.vine_type}")
        
        return structure
    
    def fit_vine_copula(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit vine copula to data using sequential estimation
        
        Args:
            data: DataFrame with parameter observations
            
        Returns:
            Fitted vine copula parameters
        """
        logger.info(f"Fitting {self.vine_type} vine copula")
        
        # Transform to uniform margins
        uniform_data = self._transform_to_uniform_margins(data)
        current_data = uniform_data.copy()
        
        # Fit copulas level by level
        for level, pairs in enumerate(self.vine_structure):
            logger.debug(f"Fitting vine level {level + 1}")
            
            for pair in pairs:
                i, j = pair
                
                # Fit bivariate copula for this pair
                pair_data = current_data[:, [i, j]]
                copula = self._fit_bivariate_copula(pair_data)
                
                # Store copula
                copula_key = f"level_{level}_pair_{i}_{j}"
                self.bivariate_copulas[copula_key] = copula
                
                # Transform data for next level (pseudo-observations)
                if level < len(self.vine_structure) - 1:
                    # Calculate conditional distributions for next level
                    pass  # Simplified for this implementation
        
        return {'vine_structure': self.vine_structure, 
                'n_copulas': len(self.bivariate_copulas)}
    
    def _fit_bivariate_copula(self, pair_data: np.ndarray) -> CopulaDistribution:
        """Fit bivariate copula to pair data"""
        # For simplicity, use Gaussian copula
        # In practice, select best copula using AIC/BIC
        
        marginals = {
            'param_0': stats.uniform(0, 1),
            'param_1': stats.uniform(0, 1)
        }
        
        copula = CopulaDistribution(marginals, copula_type='gaussian')
        
        # Convert array to DataFrame for fitting
        df = pd.DataFrame(pair_data, columns=['param_0', 'param_1'])
        copula.fit_copula_to_data(df)
        
        return copula
    
    def _transform_to_uniform_margins(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data to uniform margins"""
        uniform_data = np.zeros_like(data.values)
        
        for i, col in enumerate(data.columns):
            ranks = stats.rankdata(data[col])
            uniform_data[:, i] = ranks / (len(ranks) + 1)
        
        return uniform_data


# Integration functions
def create_copula_enhanced_monte_carlo(base_config: Dict[str, Any],
                                     parameter_correlations: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Create Monte Carlo configuration with copula-based dependencies
    
    Args:
        base_config: Base Monte Carlo configuration
        parameter_correlations: Parameter correlation specifications
        
    Returns:
        Enhanced configuration with copula dependencies
    """
    enhanced_config = base_config.copy()
    
    # Add copula configuration
    enhanced_config['copula_dependencies'] = {
        'enabled': True,
        'copula_type': 'gaussian',  # Default to Gaussian copula
        'correlation_matrix': parameter_correlations,
        'fit_to_data': True,
        'vine_copula': {
            'enabled': False,  # Enable for high-dimensional cases
            'vine_type': 'c_vine'
        }
    }
    
    return enhanced_config


# Example usage
if __name__ == "__main__":
    # Test copula distribution
    marginals = {
        'growth_rate': stats.norm(0.05, 0.02),
        'volatility': stats.lognorm(s=0.5, scale=0.1),
        'indicator_weight': stats.beta(2, 2)
    }
    
    # Create correlation matrix
    correlation_matrix = np.array([
        [1.0, 0.3, -0.2],
        [0.3, 1.0, 0.1],
        [-0.2, 0.1, 1.0]
    ])
    
    # Initialize copula
    copula = CopulaDistribution(marginals, 'gaussian', correlation_matrix)
    
    # Generate correlated samples
    samples = copula.sample_correlated_parameters(1000)
    
    print("Generated correlated samples:")
    for param, values in samples.items():
        print(f"{param}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    # Calculate dependence measures
    dependence = copula.calculate_dependence_measures(samples)
    print(f"\nPearson correlation:\n{dependence['pearson_correlation']}")
    print(f"\nSpearman correlation:\n{dependence['spearman_correlation']}")