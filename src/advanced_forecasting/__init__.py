"""
Advanced Forecasting Module - Next-generation probabilistic forecasting capabilities

This module provides cutting-edge forecasting methods including:
- Monte Carlo simulation-based uncertainty quantification
- Enhanced confidence interval calculation
- Adaptive model selection and ensemble optimization
- Real-time parameter tuning and learning

All components are designed to work seamlessly with existing forecasting infrastructure
while adding enterprise-grade probabilistic modeling capabilities.
"""

from .monte_carlo_engine import MonteCarloDistributor, MonteCarloEnsemble
from .enhanced_confidence import EnhancedConfidenceCalculator, UncertaintyPropagator
from .adaptive_ensemble import AdaptiveEnsembleManager, VolatilityAwareEnsemble

__all__ = [
    'MonteCarloDistributor',
    'MonteCarloEnsemble', 
    'EnhancedConfidenceCalculator',
    'UncertaintyPropagator',
    'AdaptiveEnsembleManager',
    'VolatilityAwareEnsemble'
]

__version__ = "1.0.0"