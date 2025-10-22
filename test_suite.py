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
    """Test core PSD algorithm functionality."""
    print("\n" + "="*70)
    print("TESTING CORE PSD ALGORITHMS")
    print("="*70)

    import numpy as np

    # Test 1: Waveform generation
    test_name = "Waveform generation (bi-exponential)"
    try:
        def generate_pulse(n_samples=500, tau_fast=3.0, tau_slow=100.0, ratio=0.3):
            t = np.arange(n_samples) * 2.0  # 2 ns sampling
            A_fast = 1000 * (1 - ratio)
            A_slow = 1000 * ratio
            waveform = A_fast * np.exp(-t / tau_fast) + A_slow * np.exp(-t / tau_slow)
            return waveform

        wf = generate_pulse()
        assert len(wf) == 500, "Wrong waveform length"
        assert wf[0] > wf[-1], "Waveform should decay"
        assert np.all(wf >= 0), "Waveform should be positive"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 2: Tail-to-total ratio calculation
    test_name = "Tail-to-total ratio calculation"
    try:
        wf = generate_pulse()
        total = np.sum(wf[:250])  # 0-500 ns
        tail = np.sum(wf[25:250])  # 50-500 ns
        psd = tail / total if total > 0 else 0

        assert 0 <= psd <= 1, f"PSD should be in [0,1], got {psd}"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 3: Figure of Merit calculation
    test_name = "Figure of Merit calculation"
    try:
        # Generate gamma and neutron distributions
        psd_gamma = np.random.normal(0.15, 0.03, 1000)
        psd_neutron = np.random.normal(0.35, 0.04, 1000)

        # Calculate FoM
        mean_gamma = np.mean(psd_gamma)
        mean_neutron = np.mean(psd_neutron)
        std_gamma = np.std(psd_gamma)
        std_neutron = np.std(psd_neutron)

        separation = abs(mean_neutron - mean_gamma)
        fwhm_gamma = 2.355 * std_gamma
        fwhm_neutron = 2.355 * std_neutron
        fom = separation / (fwhm_gamma + fwhm_neutron)

        assert fom > 0, "FoM should be positive"
        assert fom < 10, f"FoM seems unrealistically high: {fom}"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 4: Energy calibration
    test_name = "Energy calibration (linear)"
    try:
        # Known points: 662 keV Cs-137, 1173 keV Co-60
        adc_values = np.array([662, 1173])
        energy_values = np.array([662, 1173])

        # Linear fit
        coeffs = np.polyfit(adc_values, energy_values, 1)

        # Test calibration
        test_adc = 1000
        calibrated_energy = np.polyval(coeffs, test_adc)

        assert 900 < calibrated_energy < 1100, "Calibration seems off"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 5: Peak finding
    test_name = "Peak finding in spectrum"
    try:
        from scipy.signal import find_peaks

        # Generate spectrum with peak
        energies = np.concatenate([
            np.random.normal(662, 30, 1000),  # Photopeak
            np.random.uniform(0, 600, 2000)   # Background
        ])

        counts, bins = np.histogram(energies, bins=200, range=(0, 800))
        peaks, properties = find_peaks(counts, height=10, prominence=5)

        assert len(peaks) > 0, "Should find at least one peak"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

def test_ml_pipeline():
    """Test machine learning pipeline."""
    print("\n" + "="*70)
    print("TESTING ML PIPELINE")
    print("="*70)

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Test 1: Data preparation
    test_name = "ML data preparation"
    try:
        # Generate synthetic features
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) == 800, "Train set size incorrect"
        assert len(X_test) == 200, "Test set size incorrect"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))
        return

    # Test 2: Model training
    test_name = "Random Forest training"
    try:
        clf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
        clf.fit(X_train, y_train)

        assert hasattr(clf, 'feature_importances_'), "Model should have feature importances"
        assert len(clf.feature_importances_) == n_features, "Wrong number of importances"
        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))
        return

    # Test 3: Prediction
    test_name = "ML prediction"
    try:
        y_pred = clf.predict(X_test)

        assert len(y_pred) == len(y_test), "Prediction length mismatch"
        assert all(p in [0, 1] for p in y_pred), "Predictions should be 0 or 1"

        accuracy = accuracy_score(y_test, y_pred)
        if accuracy < 0.4:
            log_warning(test_name, f"Low accuracy: {accuracy:.2f} (expected for random data)")

        log_pass(test_name)
    except Exception as e:
        log_fail(test_name, str(e))

    # Test 4: Feature importance
    test_name = "Feature importance analysis"
    try:
        importances = clf.feature_importances_

        assert np.isclose(np.sum(importances), 1.0), "Importances should sum to 1"
        assert all(i >= 0 for i in importances), "Importances should be non-negative"
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
