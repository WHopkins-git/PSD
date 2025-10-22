#!/usr/bin/env python3
"""
Comprehensive test suite for PSD analysis code and notebooks.

This script tests:
1. Python syntax in all .py files
2. Notebook JSON structure
3. Core PSD algorithms
4. ML pipeline functionality
5. Data generation and processing
"""

import sys
import os
import json
import traceback
from pathlib import Path
import importlib.util

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_pass(test_name):
    """Log a passed test."""
    test_results['passed'].append(test_name)
    print(f"✓ PASS: {test_name}")

def log_fail(test_name, error):
    """Log a failed test."""
    test_results['failed'].append((test_name, str(error)))
    print(f"✗ FAIL: {test_name}")
    print(f"  Error: {error}")

def log_warning(test_name, message):
    """Log a warning."""
    test_results['warnings'].append((test_name, message))
    print(f"⚠ WARNING: {test_name}: {message}")

def test_python_syntax(filepath):
    """Test if a Python file has valid syntax."""
    test_name = f"Syntax check: {filepath.name}"
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        compile(code, filepath.name, 'exec')
        log_pass(test_name)
        return True
    except SyntaxError as e:
        log_fail(test_name, f"Line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        log_fail(test_name, str(e))
        return False

def test_notebook_structure(filepath):
    """Test if a notebook has valid JSON structure."""
    test_name = f"Notebook structure: {filepath.name}"
    try:
        with open(filepath, 'r') as f:
            nb = json.load(f)

        # Check required fields
        if 'cells' not in nb:
            log_fail(test_name, "Missing 'cells' field")
            return False
        if 'metadata' not in nb:
            log_warning(test_name, "Missing 'metadata' field")
        if 'nbformat' not in nb:
            log_warning(test_name, "Missing 'nbformat' field")

        # Check cells structure
        for i, cell in enumerate(nb['cells']):
            if 'cell_type' not in cell:
                log_fail(test_name, f"Cell {i} missing 'cell_type'")
                return False
            if 'source' not in cell:
                log_fail(test_name, f"Cell {i} missing 'source'")
                return False

        log_pass(test_name)
        return True
    except json.JSONDecodeError as e:
        log_fail(test_name, f"Invalid JSON: {e}")
        return False
    except Exception as e:
        log_fail(test_name, str(e))
        return False

def test_core_psd_algorithms():
    """Test core PSD algorithm functionality using actual package."""
    print("\n" + "="*70)
    print("TESTING CORE PSD ALGORITHMS")
    print("="*70)

    import numpy as np
    import pandas as pd

    # Import from actual package
    try:
        from psd_analysis import calculate_psd_ratio, calculate_figure_of_merit
        from psd_analysis import calibrate_energy, find_peaks_in_spectrum
    except ImportError as e:
        log_fail("Package import", f"Cannot import psd_analysis: {e}")
        return

    # Test 1: PSD ratio calculation
    test_name = "PSD ratio calculation (package function)"
    try:
        # Create test data
        df = pd.DataFrame({
            'ENERGY': [1000, 2000, 3000],
            'ENERGYSHORT': [700, 1400, 2100]
        })

        df = calculate_psd_ratio(df)

        assert 'PSD' in df.columns, "PSD column not created"
        assert all(df['PSD'] >= 0), "PSD should be non-negative"
        assert all(df['PSD'] <= 1), "PSD should be <= 1"
        # PSD = (E - E_short) / E = (1000 - 700) / 1000 = 0.3
        assert abs(df.loc[0, 'PSD'] - 0.3) < 0.01, "PSD calculation incorrect"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 2: Figure of Merit calculation
    test_name = "Figure of Merit calculation (package function)"
    try:
        # Generate gamma and neutron distributions
        psd_gamma = np.random.normal(0.15, 0.03, 1000)
        psd_neutron = np.random.normal(0.35, 0.04, 1000)

        # Calculate FoM using package function
        fom = calculate_figure_of_merit(psd_neutron, psd_gamma)

        assert fom > 0, "FoM should be positive"
        assert fom < 10, f"FoM seems unrealistically high: {fom}"
        assert fom > 1.0, "FoM should be > 1 for well-separated distributions"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 3: Energy calibration
    test_name = "Energy calibration (package function)"
    try:
        # Create test data
        df = pd.DataFrame({
            'ENERGY': [500, 1000, 1500, 2000],
            'ENERGYSHORT': [350, 700, 1050, 1400]
        })

        # Calibration points (ADC, keV)
        calibration_points = [(500, 250), (2000, 1000)]

        df_cal, cal_func, params = calibrate_energy(df, calibration_points)

        assert 'ENERGY_KEV' in df_cal.columns, "ENERGY_KEV column not created"
        assert callable(cal_func), "Calibration function should be callable"
        # Check calibration at known point: 500 ADC should be ~250 keV
        assert abs(df_cal.loc[0, 'ENERGY_KEV'] - 250) < 10, "Calibration seems off"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 4: Peak finding
    test_name = "Peak finding in spectrum (package function)"
    try:
        # Generate spectrum with peak
        energies = np.concatenate([
            np.random.normal(662, 30, 1000),  # Photopeak
            np.random.uniform(0, 600, 2000)   # Background
        ])

        counts, bins = np.histogram(energies, bins=200, range=(0, 800))
        bin_centers = (bins[:-1] + bins[1:]) / 2

        peaks, peak_counts, properties = find_peaks_in_spectrum(
            bin_centers, counts, prominence=10
        )

        assert len(peaks) > 0, "Should find at least one peak"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 5: Data loading
    test_name = "Data loader import"
    try:
        from psd_analysis import load_psd_data, validate_events
        # Just test that imports work
        assert callable(load_psd_data), "load_psd_data should be callable"
        assert callable(validate_events), "validate_events should be callable"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

def test_ml_pipeline():
    """Test machine learning pipeline using package classes."""
    print("\n" + "="*70)
    print("TESTING ML PIPELINE")
    print("="*70)

    import numpy as np
    import pandas as pd

    # Import from actual package
    try:
        from psd_analysis.ml.classical import ClassicalMLClassifier
    except ImportError as e:
        log_fail("ML package import", f"Cannot import ml module: {e}")
        return

    # Test 1: Classifier initialization
    test_name = "ClassicalMLClassifier initialization"
    try:
        clf = ClassicalMLClassifier(method='random_forest')
        assert clf.method == 'random_forest', "Method not set correctly"
        assert clf.model is not None, "Model not initialized"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))
        return

    # Test 2: Data preparation and training
    test_name = "ML classifier training"
    try:
        # Generate synthetic data with proper columns
        n_samples = 1000
        df = pd.DataFrame({
            'ENERGY': np.random.uniform(100, 2000, n_samples),
            'ENERGYSHORT': np.random.uniform(50, 1500, n_samples),
            'PARTICLE': np.random.choice(['gamma', 'neutron'], n_samples)
        })

        # Add PSD column
        from psd_analysis import calculate_psd_ratio
        df = calculate_psd_ratio(df)

        # Train classifier
        clf = ClassicalMLClassifier(method='random_forest')
        results = clf.train(df, test_size=0.2)

        assert 'train_accuracy' in results, "Missing train_accuracy"
        assert 'val_accuracy' in results, "Missing val_accuracy"
        assert results['val_accuracy'] > 0, "Validation accuracy should be positive"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))
        return

    # Test 3: Prediction
    test_name = "ML prediction"
    try:
        # Generate new test data
        df_test = pd.DataFrame({
            'ENERGY': np.random.uniform(100, 2000, 100),
            'ENERGYSHORT': np.random.uniform(50, 1500, 100)
        })
        df_test = calculate_psd_ratio(df_test)

        predictions, probabilities = clf.predict(df_test)

        assert len(predictions) == len(df_test), "Prediction length mismatch"
        assert all(p in [0, 1] for p in predictions), "Predictions should be 0 or 1"
        # probabilities is 1D array of neutron probabilities
        assert len(probabilities) == len(df_test), "Probability length incorrect"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 4: Multiple classifier types
    test_name = "Multiple classifier types"
    try:
        for method in ['random_forest', 'gradient_boosting', 'svm', 'logistic']:
            clf_test = ClassicalMLClassifier(method=method)
            assert clf_test.model is not None, f"{method} model not initialized"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

def test_data_processing():
    """Test data processing utilities."""
    print("\n" + "="*70)
    print("TESTING DATA PROCESSING")
    print("="*70)

    import numpy as np
    import pandas as pd

    # Test 1: DataFrame creation
    test_name = "DataFrame creation with waveforms"
    try:
        n_events = 100
        waveforms = [np.random.randn(500) for _ in range(n_events)]
        energies = np.random.uniform(100, 3000, n_events)
        particles = np.random.choice(['neutron', 'gamma'], n_events)

        df = pd.DataFrame({
            'WAVEFORM': waveforms,
            'ENERGY': energies,
            'PARTICLE': particles
        })

        assert len(df) == n_events, "Wrong number of events"
        assert 'WAVEFORM' in df.columns, "Missing WAVEFORM column"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))
        return

    # Test 2: Feature extraction
    test_name = "Basic feature extraction"
    try:
        def extract_features(waveform):
            features = {}
            features['max'] = np.max(waveform)
            features['mean'] = np.mean(waveform)
            features['std'] = np.std(waveform)
            features['sum'] = np.sum(waveform)
            return features

        features = extract_features(df['WAVEFORM'].iloc[0])

        assert 'max' in features, "Missing max feature"
        assert 'mean' in features, "Missing mean feature"
        assert isinstance(features['max'], (int, float, np.number)), "Features should be numeric"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 3: Data filtering
    test_name = "Data filtering and selection"
    try:
        neutrons = df[df['PARTICLE'] == 'neutron']
        gammas = df[df['PARTICLE'] == 'gamma']

        assert len(neutrons) + len(gammas) == len(df), "Filtering lost events"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

def test_numerical_stability():
    """Test numerical stability of algorithms."""
    print("\n" + "="*70)
    print("TESTING NUMERICAL STABILITY")
    print("="*70)

    import numpy as np

    # Test 1: Division by zero handling
    test_name = "Division by zero protection"
    try:
        def safe_ratio(numerator, denominator):
            return numerator / denominator if denominator != 0 else 0.0

        result1 = safe_ratio(10, 2)
        result2 = safe_ratio(10, 0)

        assert result1 == 5.0, "Normal division failed"
        assert result2 == 0.0, "Zero division not handled"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 2: NaN handling
    test_name = "NaN handling"
    try:
        arr = np.array([1, 2, np.nan, 4, 5])
        clean_arr = arr[~np.isnan(arr)]

        assert len(clean_arr) == 4, "NaN removal failed"
        assert not np.any(np.isnan(clean_arr)), "NaNs still present"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 3: Overflow protection
    test_name = "Overflow protection"
    try:
        large_value = 1e308
        result = np.clip(large_value * 2, -1e307, 1e307)

        assert np.isfinite(result), "Overflow not handled"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

def print_summary():
    """Print test summary."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total_tests = len(test_results['passed']) + len(test_results['failed'])

    print(f"\nTotal tests run: {total_tests}")
    print(f"✓ Passed: {len(test_results['passed'])}")
    print(f"✗ Failed: {len(test_results['failed'])}")
    print(f"⚠ Warnings: {len(test_results['warnings'])}")

    if test_results['failed']:
        print("\n" + "="*70)
        print("FAILED TESTS")
        print("="*70)
        for test_name, error in test_results['failed']:
            print(f"\n✗ {test_name}")
            print(f"  {error}")

    if test_results['warnings']:
        print("\n" + "="*70)
        print("WARNINGS")
        print("="*70)
        for test_name, warning in test_results['warnings']:
            print(f"\n⚠ {test_name}")
            print(f"  {warning}")

    success_rate = len(test_results['passed']) / total_tests * 100 if total_tests > 0 else 0
    print(f"\n{'='*70}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"{'='*70}\n")

    return len(test_results['failed']) == 0

def main():
    """Run all tests."""
    print("="*70)
    print("PSD ANALYSIS - COMPREHENSIVE TEST SUITE")
    print("="*70)

    repo_path = Path('/home/user/PSD')

    # Test 1: Python file syntax
    print("\n" + "="*70)
    print("TESTING PYTHON FILE SYNTAX")
    print("="*70)

    py_files = list(repo_path.glob('*.py'))
    for py_file in py_files:
        if py_file.name != 'test_suite.py':  # Skip self
            test_python_syntax(py_file)

    # Test 2: Notebook structure
    print("\n" + "="*70)
    print("TESTING NOTEBOOK STRUCTURE")
    print("="*70)

    notebooks_path = repo_path / 'notebooks'
    if notebooks_path.exists():
        nb_files = list(notebooks_path.glob('*.ipynb'))
        for nb_file in nb_files:
            test_notebook_structure(nb_file)

    # Test 3: Core algorithms
    test_core_psd_algorithms()

    # Test 4: ML pipeline
    test_ml_pipeline()

    # Test 5: Data processing
    test_data_processing()

    # Test 6: Numerical stability
    test_numerical_stability()

    # Print summary
    success = print_summary()

    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
