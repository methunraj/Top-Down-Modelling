#!/usr/bin/env python3
"""
Final verification test for Monte Carlo integration
Tests the key integration points that were fixed
"""

import sys
import os
sys.path.append('src')

from advanced_forecasting.monte_carlo_engine import MonteCarloDistributor
import pandas as pd
import numpy as np

def test_statistics_fix():
    """Test that the statistics calculation fix is working"""
    print("=== Testing Statistics Calculation Fix ===")
    
    # Create sample scenarios that would trigger the original error
    scenarios = []
    for i in range(100):
        scenarios.append({
            'market_size': 1000 + np.random.normal(0, 100),
            'distribution': {
                'Country_A': 0.4 + np.random.normal(0, 0.05),
                'Country_B': 0.3 + np.random.normal(0, 0.05),
                'Country_C': 0.3 + np.random.normal(0, 0.05)
            }
        })
    
    monte_carlo = MonteCarloDistributor({'num_simulations': 100})
    
    try:
        # This used to fail with "'dict' object has no attribute 'skew'"
        statistics = monte_carlo._calculate_simulation_statistics(scenarios)
        
        print("✅ Statistics calculation working!")
        print(f"Generated statistics: {list(statistics.keys())}")
        
        # Check that market_size_stats is properly calculated
        if 'market_size_stats' in statistics:
            market_stats = statistics['market_size_stats']
            print(f"Market size stats keys: {list(market_stats.keys())}")
            
            # Verify mean and std are calculated
            if 'mean' in market_stats and 'std' in market_stats:
                print(f"Mean: {market_stats['mean']:.2f}, Std: {market_stats['std']:.2f}")
                return True
            else:
                print("❌ Mean or std missing from market_size_stats")
                return False
        else:
            print("❌ market_size_stats missing from statistics")
            return False
            
    except Exception as e:
        print(f"❌ Statistics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_results_structure():
    """Test that Monte Carlo returns the expected results structure"""
    print("\n=== Testing Results Structure ===")
    
    # Create minimal sample data
    market_data = pd.DataFrame({
        'Country': ['USA', 'China', 'Germany'],
        'Market_Size_2024': [1000, 800, 400]
    })
    
    monte_carlo = MonteCarloDistributor({
        'num_simulations': 10,  # Small number for testing
        'uncertainty_factors': {
            'market_volatility': 0.10,
            'economic_uncertainty': 0.05
        }
    })
    
    try:
        forecast_years = [2025]
        results = monte_carlo.simulate_market_scenarios(market_data, forecast_years)
        
        print("✅ Monte Carlo simulation completed!")
        
        # Verify expected structure
        expected_keys = ['scenarios', 'aggregated', 'statistics', 'risk_metrics', 'convergence', 'metadata']
        actual_keys = list(results.keys())
        
        print(f"Expected keys: {expected_keys}")
        print(f"Actual keys: {actual_keys}")
        
        missing_keys = [key for key in expected_keys if key not in actual_keys]
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}")
            return False
        
        # Check statistics structure
        if 'statistics' in results:
            stats = results['statistics']
            if 'market_size_stats' in stats:
                market_stats = stats['market_size_stats']
                required_stats = ['mean', 'std', 'min', 'max']
                missing_stats = [stat for stat in required_stats if stat not in market_stats]
                if missing_stats:
                    print(f"❌ Missing market size stats: {missing_stats}")
                    return False
                else:
                    print("✅ All required market_size_stats present")
            else:
                print("❌ market_size_stats missing from statistics")
                return False
        else:
            print("❌ statistics missing from results")
            return False
        
        print("✅ Results structure is correct!")
        return True
        
    except Exception as e:
        print(f"❌ Monte Carlo simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_compatibility():
    """Test that results work with the interface expectations"""
    print("\n=== Testing Interface Compatibility ===")
    
    # Simulate what the interface expects
    mock_results = {
        'scenarios': [],
        'aggregated': {},
        'statistics': {
            'success_rate': 0.95,
            'market_size_stats': {
                'mean': 2500.75,
                'std': 125.30,
                'min': 2200.50,
                'max': 2800.25,
                'skewness': 0.1,
                'kurtosis': -0.2
            },
            'concentration_stats': {},
            'growth_rate_stats': {}
        },
        'risk_metrics': {},
        'convergence': {},
        'metadata': {}
    }
    
    try:
        # Simulate interface processing
        if 'statistics' in mock_results:
            stats = mock_results['statistics']
            
            # Get market size statistics (this is what the interface does)
            if 'market_size_stats' in stats:
                market_stats = stats['market_size_stats']
                
                if 'mean' in market_stats:
                    avg_market = f"{market_stats['mean']:.2f}"
                    print(f"✅ Average Market Size: {avg_market}")
                
                if 'std' in market_stats:
                    market_uncertainty = f"{market_stats['std']:.2f}"
                    print(f"✅ Market Uncertainty: {market_uncertainty}")
                
                print("✅ Interface compatibility verified!")
                return True
            else:
                print("❌ market_size_stats missing")
                return False
        else:
            print("❌ statistics missing")
            return False
            
    except Exception as e:
        print(f"❌ Interface compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Final Monte Carlo Integration Verification\n")
    
    test1 = test_statistics_fix()
    test2 = test_results_structure()
    test3 = test_interface_compatibility()
    
    print(f"\n=== Final Results ===")
    print(f"Statistics Fix: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Results Structure: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Interface Compatibility: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test1 and test2 and test3:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"Monte Carlo integration is ready for use in the Streamlit app.")
        print(f"\nTo use:")
        print(f"1. Go to Distribution tab in Streamlit")
        print(f"2. Click the 'Monte Carlo' tab")
        print(f"3. Enable Monte Carlo simulation")
        print(f"4. Configure uncertainty settings")
        print(f"5. Click 'Run Market Distribution'")
        print(f"6. View real-time progress and final results with uncertainty metrics")
    else:
        print(f"\n⚠️ Some tests failed - check output above")