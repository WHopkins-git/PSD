"""
PSD Analysis Toolkit - Core Functions
For neutron/gamma discrimination and NORM source analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.stats import norm
import warnings

# ============================================================================
# 1. DATA LOADING & QUALITY CONTROL
# ============================================================================

def load_psd_data(filename, delimiter=';'):
    """
    Load PSD data from CSV file
    
    Parameters:
    -----------
    filename : str
        Path to CSV file
    delimiter : str
        Column separator (default ';')
    
    Returns:
    --------
    df : pandas DataFrame
        Loaded data with columns: board, channel, timetag, energy, 
        energy_short, flags, probe_code, [samples...]
    """
    df = pd.read_csv(filename, delimiter=delimiter)
    
    # Rename columns for consistency
    df.columns = [col.upper() for col in df.columns]
    
    print(f"Loaded {len(df)} events from {filename}")
    print(f"Columns: {list(df.columns[:7])}")
    
    return df


def validate_events(df, adc_min=0, adc_max=16383, baseline_stability=50):
    """
    Quality control: identify problematic events
    
    Parameters:
    -----------
    df : DataFrame
        Event data
    adc_min, adc_max : int
        ADC range limits (default for 14-bit ADC)
    baseline_stability : float
        Max acceptable baseline RMS
    
    Returns:
    --------
    valid_mask : boolean array
        True for good events
    qc_report : dict
        Quality control statistics
    """
    n_events = len(df)
    
    # Get sample columns
    sample_cols = [col for col in df.columns if col.startswith('SAMPLE')]
    
    if not sample_cols:
        warnings.warn("No sample columns found, skipping waveform QC")
        return np.ones(n_events, dtype=bool), {'total': n_events, 'valid': n_events}
    
    valid = np.ones(n_events, dtype=bool)
    
    # Check for saturation
    samples = df[sample_cols].values
    saturated = (samples <= adc_min + 10) | (samples >= adc_max - 10)
    valid &= ~saturated.any(axis=1)
    
    # Check baseline stability (first 50 samples)
    baseline_samples = samples[:, :min(50, samples.shape[1])]
    baseline_rms = baseline_samples.std(axis=1)
    valid &= baseline_rms < baseline_stability
    
    # Check for pile-up (multiple peaks)
    # Simple method: look for multiple local minima
    # (More sophisticated methods available)
    
    qc_report = {
        'total_events': n_events,
        'valid_events': valid.sum(),
        'saturated': saturated.any(axis=1).sum(),
        'unstable_baseline': (baseline_rms >= baseline_stability).sum(),
        'rejection_rate': 1 - valid.sum() / n_events
    }
    
    print(f"QC Summary: {qc_report['valid_events']}/{n_events} events passed")
    print(f"  Saturation: {qc_report['saturated']} events")
    print(f"  Baseline instability: {qc_report['unstable_baseline']} events")
    
    return valid, qc_report


# ============================================================================
# 2. ENERGY CALIBRATION
# ============================================================================

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


# ============================================================================
# 3. PSD CALCULATION
# ============================================================================

def calculate_psd_ratio(df):
    """
    Calculate PSD parameter using charge integration method
    
    PSD = (Q_total - Q_short) / Q_total = (E - E_short) / E
    
    Parameters:
    -----------
    df : DataFrame
        Must have ENERGY and ENERGYSHORT columns
    
    Returns:
    --------
    df : DataFrame
        With new column 'PSD'
    """
    # Avoid division by zero
    mask = df['ENERGY'] > 0
    
    df['PSD'] = np.nan
    df.loc[mask, 'PSD'] = (df.loc[mask, 'ENERGY'] - df.loc[mask, 'ENERGYSHORT']) / df.loc[mask, 'ENERGY']
    
    return df


def calculate_figure_of_merit(psd_neutron, psd_gamma):
    """
    Calculate Figure of Merit for n/γ separation
    
    FOM = separation / (FWHM_n + FWHM_gamma)
    
    Higher FOM = better discrimination
    FOM > 1.0 is good, FOM > 1.5 is excellent
    
    Parameters:
    -----------
    psd_neutron, psd_gamma : arrays
        PSD distributions for neutrons and gammas
    
    Returns:
    --------
    fom : float
        Figure of merit
    """
    # Calculate means
    mean_n = np.mean(psd_neutron)
    mean_g = np.mean(psd_gamma)
    separation = abs(mean_n - mean_g)
    
    # Calculate FWHM (2.355 * sigma for Gaussian)
    fwhm_n = 2.355 * np.std(psd_neutron)
    fwhm_g = 2.355 * np.std(psd_gamma)
    
    fom = separation / (fwhm_n + fwhm_g)
    
    print(f"Figure of Merit: {fom:.3f}")
    print(f"  Neutron: mean={mean_n:.3f}, FWHM={fwhm_n:.3f}")
    print(f"  Gamma:   mean={mean_g:.3f}, FWHM={fwhm_g:.3f}")
    
    return fom


# ============================================================================
# 4. PSD DISCRIMINATION
# ============================================================================

def define_linear_discrimination(df_calibration, neutron_label='neutron'):
    """
    Define linear PSD discrimination boundary from calibration data
    
    PSD_threshold = a * Energy + b
    
    Parameters:
    -----------
    df_calibration : DataFrame
        Must have columns: ENERGY (or ENERGY_KEV), PSD, PARTICLE_TYPE
    neutron_label : str
        Label for neutron events in PARTICLE_TYPE column
    
    Returns:
    --------
    boundary_func : function
        PSD_threshold(energy)
    params : tuple
        (slope, intercept)
    """
    # Separate particles
    is_neutron = df_calibration['PARTICLE_TYPE'] == neutron_label
    
    neutrons = df_calibration[is_neutron]
    gammas = df_calibration[~is_neutron]
    
    energy_col = 'ENERGY_KEV' if 'ENERGY_KEV' in df_calibration.columns else 'ENERGY'
    
    # Bin by energy and find mean PSD for each particle type
    energy_bins = np.linspace(df_calibration[energy_col].min(), 
                              df_calibration[energy_col].max(), 20)
    
    bin_centers = []
    threshold_vals = []
    
    for i in range(len(energy_bins) - 1):
        e_min, e_max = energy_bins[i], energy_bins[i+1]
        e_center = (e_min + e_max) / 2
        
        n_in_bin = neutrons[(neutrons[energy_col] >= e_min) & 
                             (neutrons[energy_col] < e_max)]
        g_in_bin = gammas[(gammas[energy_col] >= e_min) & 
                          (gammas[energy_col] < e_max)]
        
        if len(n_in_bin) > 10 and len(g_in_bin) > 10:
            # Threshold = midpoint between distributions
            threshold = (n_in_bin['PSD'].mean() + g_in_bin['PSD'].mean()) / 2
            bin_centers.append(e_center)
            threshold_vals.append(threshold)
    
    # Fit linear boundary
    params = np.polyfit(bin_centers, threshold_vals, 1)
    boundary_func = np.poly1d(params)
    
    print(f"Linear discrimination boundary: PSD_cut = {params[0]:.6f} * E + {params[1]:.3f}")
    
    return boundary_func, params


def apply_discrimination(df, boundary_func):
    """
    Classify events as neutron or gamma
    
    Parameters:
    -----------
    df : DataFrame
        Must have ENERGY and PSD columns
    boundary_func : function
        PSD threshold vs energy
    
    Returns:
    --------
    df : DataFrame
        With new column 'PARTICLE' ('neutron' or 'gamma')
    """
    energy_col = 'ENERGY_KEV' if 'ENERGY_KEV' in df.columns else 'ENERGY'
    
    threshold = boundary_func(df[energy_col])
    df['PARTICLE'] = 'gamma'
    df.loc[df['PSD'] > threshold, 'PARTICLE'] = 'neutron'
    
    n_neutrons = (df['PARTICLE'] == 'neutron').sum()
    n_gammas = (df['PARTICLE'] == 'gamma').sum()
    
    print(f"Discrimination results: {n_neutrons} neutrons, {n_gammas} gammas")
    
    return df


# ============================================================================
# 5. SPECTROSCOPY - PEAK FINDING
# ============================================================================

def find_peaks_in_spectrum(energy, counts, prominence=100, distance=20):
    """
    Find peaks in energy spectrum
    
    Parameters:
    -----------
    energy : array
        Energy bin centers
    counts : array
        Spectrum counts
    prominence : float
        Minimum peak prominence above background
    distance : int
        Minimum distance between peaks (in bins)
    
    Returns:
    --------
    peak_energies : array
        Energy values of peaks
    peak_counts : array
        Count values at peaks
    peak_properties : dict
        Additional peak properties
    """
    # Find peaks
    peaks, properties = signal.find_peaks(counts, 
                                          prominence=prominence,
                                          distance=distance)
    
    peak_energies = energy[peaks]
    peak_counts = counts[peaks]
    
    print(f"Found {len(peaks)} peaks:")
    for i, (e, c) in enumerate(zip(peak_energies, peak_counts)):
        print(f"  Peak {i+1}: E = {e:.1f} keV, counts = {c:.0f}")
    
    return peak_energies, peak_counts, properties


def fit_gaussian_peak(energy, counts, peak_energy, fit_width=50):
    """
    Fit Gaussian to peak for accurate centroid and FWHM
    
    Parameters:
    -----------
    energy : array
        Energy bin centers
    counts : array
        Spectrum counts
    peak_energy : float
        Approximate peak location
    fit_width : float
        Width of region to fit (keV)
    
    Returns:
    --------
    fit_params : dict
        Fitted parameters: centroid, amplitude, sigma, fwhm
    """
    # Select fit region
    mask = np.abs(energy - peak_energy) < fit_width
    e_fit = energy[mask]
    c_fit = counts[mask]
    
    # Gaussian + linear background
    def model(e, amp, mu, sigma, bg_slope, bg_offset):
        gaussian = amp * np.exp(-0.5 * ((e - mu) / sigma)**2)
        background = bg_slope * e + bg_offset
        return gaussian + background
    
    # Initial guess
    amp_guess = c_fit.max()
    mu_guess = peak_energy
    sigma_guess = 20  # keV
    
    try:
        popt, pcov = optimize.curve_fit(model, e_fit, c_fit,
                                        p0=[amp_guess, mu_guess, sigma_guess, 0, c_fit.min()])
        
        amp, mu, sigma, bg_slope, bg_offset = popt
        fwhm = 2.355 * abs(sigma)
        resolution = fwhm / mu * 100  # Percent
        
        fit_params = {
            'centroid': mu,
            'amplitude': amp,
            'sigma': abs(sigma),
            'fwhm': fwhm,
            'resolution_percent': resolution,
            'background': (bg_slope, bg_offset)
        }
        
        print(f"Peak fit: E = {mu:.2f} keV, FWHM = {fwhm:.2f} keV, R = {resolution:.1f}%")
        
        return fit_params
    
    except:
        print(f"Warning: Fit failed for peak at {peak_energy:.1f} keV")
        return None


# ============================================================================
# 6. ISOTOPE IDENTIFICATION
# ============================================================================

# Common NORM gamma lines (keV)
ISOTOPE_LIBRARY = {
    'K-40': [1460.8],
    'U-238 series': [
        (63.3, 'Th-234'),
        (92.4, 'Th-234'),
        (1001.0, 'Pa-234m'),
        (186.2, 'Ra-226'),
        (242.0, 'Pb-214'),
        (295.2, 'Pb-214'),
        (351.9, 'Pb-214'),
        (609.3, 'Bi-214'),
        (1120.3, 'Bi-214'),
        (1764.5, 'Bi-214'),
    ],
    'Th-232 series': [
        (238.6, 'Pb-212'),
        (300.1, 'Pb-212'),
        (583.2, 'Tl-208'),
        (860.6, 'Tl-208'),
        (2614.5, 'Tl-208'),
        (911.2, 'Ac-228'),
        (969.0, 'Ac-228'),
    ],
    'Cs-137': [661.7],
    'Co-60': [1173.2, 1332.5],
    'Na-22': [511.0, 1274.5],
}

# Additional functions would go here (isotope matching, peak finding, etc.)