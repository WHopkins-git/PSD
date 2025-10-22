"""
Energy calibration functions
"""

import numpy as np


def find_compton_edge(spectrum, energy_bins, expected_edge_keV,
                      search_width=100, edge_fraction=0.5):
    """
    Locate Compton edge in gamma spectrum

    Parameters:
    -----------
    spectrum : array
        Energy histogram counts
    energy_bins : array
        Bin centers (in ADC or keV)
    expected_edge_keV : float
        Expected Compton edge energy (e.g., 477 keV for Cs-137)
    search_width : float
        Search range around expected value
    edge_fraction : float
        Fraction of max for edge definition (default 0.5)

    Returns:
    --------
    edge_position : float
        Compton edge location in bin units
    """
    # Search region
    if expected_edge_keV:
        # If calibrated, use expected position
        search_mask = np.abs(energy_bins - expected_edge_keV) < search_width
    else:
        # Otherwise search in upper half of spectrum
        search_mask = energy_bins > energy_bins.max() / 2

    search_spectrum = spectrum[search_mask]
    search_bins = energy_bins[search_mask]

    # Find maximum in search region
    max_idx = np.argmax(search_spectrum)
    max_counts = search_spectrum[max_idx]

    # Find edge (point where spectrum drops to edge_fraction * max)
    # Look for transition going right-to-left from max
    threshold = edge_fraction * max_counts
    edge_candidates = search_bins[max_idx:][search_spectrum[max_idx:] > threshold]

    if len(edge_candidates) > 0:
        edge_position = edge_candidates[-1]
    else:
        edge_position = search_bins[max_idx]

    return edge_position


def calibrate_energy(df, calibration_points, method='linear'):
    """
    Energy calibration: convert ADC to keV

    Parameters:
    -----------
    df : DataFrame
        Event data with ENERGY column (ADC units)
    calibration_points : list of tuples
        [(adc1, keV1), (adc2, keV2), ...]
    method : str
        'linear' or 'polynomial' (deg 2)

    Returns:
    --------
    df : DataFrame
        With new column 'ENERGY_KEV'
    cal_func : function
        Calibration function for future use
    cal_params : array
        Fit coefficients
    """
    adc_vals = np.array([p[0] for p in calibration_points])
    kev_vals = np.array([p[1] for p in calibration_points])

    if method == 'linear':
        # E[keV] = a * ADC + b
        cal_params = np.polyfit(adc_vals, kev_vals, 1)
        cal_func = np.poly1d(cal_params)
    elif method == 'polynomial':
        # E[keV] = a * ADC^2 + b * ADC + c
        cal_params = np.polyfit(adc_vals, kev_vals, 2)
        cal_func = np.poly1d(cal_params)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Apply calibration
    df['ENERGY_KEV'] = cal_func(df['ENERGY'])

    # Calculate residuals
    predicted = cal_func(adc_vals)
    residuals = kev_vals - predicted
    rms_error = np.sqrt(np.mean(residuals**2))

    print(f"Calibration ({method}):")
    print(f"  Coefficients: {cal_params}")
    print(f"  RMS error: {rms_error:.2f} keV")

    return df, cal_func, cal_params
