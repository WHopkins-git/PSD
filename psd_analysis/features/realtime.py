"""
Real-time feature extraction for FPGA/DAQ implementation

Minimal feature set optimized for low-latency discrimination
Target: <100 ns latency on FPGA
"""

import numpy as np


def extract_realtime_features(waveform, baseline_samples=50, dt=4.0):
    """
    Extract minimal feature set for FPGA/DAQ real-time discrimination

    Features extracted (5-8 total):
    1. CFD time (Constant Fraction Discriminator)
    2. Charge ratio 0-60 ns
    3. Charge ratio 60-200 ns
    4. Rise time 10-90%
    5. Peak amplitude
    6. Gatti score (if templates available)

    Parameters:
    -----------
    waveform : array
        ADC samples (baseline-subtracted)
    baseline_samples : int
        Number of samples before pulse
    dt : float
        Sampling period (ns)

    Returns:
    --------
    features : dict
        Dictionary of feature values
    """
    features = {}

    # Baseline subtraction
    baseline = np.mean(waveform[:baseline_samples])
    pulse = waveform - baseline

    # Peak amplitude and position
    peak_amp = np.max(pulse)
    peak_idx = np.argmax(pulse)
    features['peak_amplitude'] = peak_amp

    if peak_amp < 10:  # Too small, return defaults
        return {k: 0.0 for k in ['cfd_time', 'charge_ratio_0_60ns',
                                  'charge_ratio_60_200ns', 'rise_time',
                                  'peak_amplitude']}

    # 1. CFD Time (Constant Fraction Discriminator)
    # Zero-crossing of CFD waveform (fraction=0.5, delay=3 samples)
    fraction = 0.5
    delay_samples = 3

    cfd_waveform = pulse - fraction * np.roll(pulse, delay_samples)

    # Find zero-crossing near peak
    search_start = max(0, peak_idx - 10)
    search_end = min(len(pulse), peak_idx + 5)

    cfd_time = 0.0
    for i in range(search_start, search_end - 1):
        if cfd_waveform[i] <= 0 and cfd_waveform[i+1] > 0:
            # Linear interpolation for sub-sample precision
            if cfd_waveform[i+1] != cfd_waveform[i]:
                frac = -cfd_waveform[i] / (cfd_waveform[i+1] - cfd_waveform[i])
                cfd_time = (i + frac) * dt
                break

    features['cfd_time'] = cfd_time

    # 2. Charge Ratios (PSD discrimination)
    # Convert time windows to sample indices
    t_60ns = int(60 / dt)
    t_200ns = int(200 / dt)

    start_idx = peak_idx
    end_60ns = min(len(pulse), peak_idx + t_60ns)
    end_200ns = min(len(pulse), peak_idx + t_200ns)

    charge_0_60 = np.sum(pulse[start_idx:end_60ns])
    charge_0_200 = np.sum(pulse[start_idx:end_200ns])
    charge_60_200 = charge_0_200 - charge_0_60

    # Ratios
    if charge_0_200 > 0:
        features['charge_ratio_0_60ns'] = charge_0_60 / charge_0_200
        features['charge_ratio_60_200ns'] = charge_60_200 / charge_0_200
    else:
        features['charge_ratio_0_60ns'] = 0.0
        features['charge_ratio_60_200ns'] = 0.0

    # 3. Rise Time 10-90%
    threshold_10 = 0.1 * peak_amp
    threshold_90 = 0.9 * peak_amp

    t_10 = 0.0
    t_90 = 0.0

    for i in range(peak_idx):
        if pulse[i] >= threshold_10 and t_10 == 0.0:
            t_10 = i * dt
        if pulse[i] >= threshold_90:
            t_90 = i * dt
            break

    features['rise_time'] = t_90 - t_10

    return features


def extract_realtime_features_with_gatti(waveform, template_neutron, template_gamma,
                                         baseline_samples=50, dt=4.0):
    """
    Real-time features including Gatti optimal filter

    Requires pre-loaded templates (must be stored in FPGA memory)

    Parameters:
    -----------
    waveform : array
        ADC samples
    template_neutron : array
        Normalized neutron template
    template_gamma : array
        Normalized gamma template
    baseline_samples : int
        Samples before pulse
    dt : float
        Sampling period (ns)

    Returns:
    --------
    features : dict
        Feature dictionary including gatti_score
    """
    # Get basic features
    features = extract_realtime_features(waveform, baseline_samples, dt)

    # Baseline subtraction
    baseline = np.mean(waveform[:baseline_samples])
    pulse = waveform - baseline

    # Normalize pulse
    norm = np.sqrt(np.sum(pulse**2))
    if norm > 0:
        pulse_norm = pulse / norm
    else:
        features['gatti_score'] = 0.0
        return features

    # Gatti score: projection onto discriminant vector
    # w = template_neutron - template_gamma (pre-computed)
    discriminant = template_neutron - template_gamma

    # Match lengths
    min_len = min(len(pulse_norm), len(discriminant))
    gatti_score = np.dot(pulse_norm[:min_len], discriminant[:min_len])

    features['gatti_score'] = gatti_score

    return features


def decision_logic_fpga(features, thresholds):
    """
    Simple decision logic for FPGA implementation

    Parameters:
    -----------
    features : dict
        Feature dictionary
    thresholds : dict
        Decision thresholds

    Returns:
    --------
    is_neutron : bool
        True if classified as neutron
    confidence : float
        Decision confidence (0-1)
    """
    # Extract relevant features
    charge_ratio = features.get('charge_ratio_60_200ns', 0.0)
    gatti = features.get('gatti_score', 0.0)

    # Simple linear combination
    score = 0.6 * charge_ratio + 0.4 * gatti

    # Threshold
    threshold = thresholds.get('neutron_threshold', 0.5)
    is_neutron = score > threshold

    # Confidence: distance from threshold
    confidence = abs(score - threshold)

    return is_neutron, confidence


def build_templates_for_fpga(waveforms_neutron, waveforms_gamma, baseline_samples=50):
    """
    Build average templates for FPGA storage

    Parameters:
    -----------
    waveforms_neutron : array (N, samples)
        Collection of neutron waveforms
    waveforms_gamma : array (M, samples)
        Collection of gamma waveforms
    baseline_samples : int
        Samples before pulse

    Returns:
    --------
    template_neutron : array
        Normalized average neutron shape
    template_gamma : array
        Normalized average gamma shape
    """
    # Process neutron waveforms
    neutron_normalized = []
    for wf in waveforms_neutron:
        baseline = np.mean(wf[:baseline_samples])
        pulse = wf - baseline

        # Align at peak
        peak_idx = np.argmax(pulse)
        if peak_idx > baseline_samples and peak_idx < len(pulse) - 100:
            aligned = np.roll(pulse, baseline_samples - peak_idx)

            # Normalize
            norm = np.sqrt(np.sum(aligned**2))
            if norm > 0:
                neutron_normalized.append(aligned / norm)

    # Process gamma waveforms
    gamma_normalized = []
    for wf in waveforms_gamma:
        baseline = np.mean(wf[:baseline_samples])
        pulse = wf - baseline

        # Align at peak
        peak_idx = np.argmax(pulse)
        if peak_idx > baseline_samples and peak_idx < len(pulse) - 100:
            aligned = np.roll(pulse, baseline_samples - peak_idx)

            # Normalize
            norm = np.sqrt(np.sum(aligned**2))
            if norm > 0:
                gamma_normalized.append(aligned / norm)

    # Average
    template_neutron = np.mean(neutron_normalized, axis=0)
    template_gamma = np.mean(gamma_normalized, axis=0)

    # Normalize templates
    template_neutron = template_neutron / np.sqrt(np.sum(template_neutron**2))
    template_gamma = template_gamma / np.sqrt(np.sum(template_gamma**2))

    return template_neutron, template_gamma


# FPGA latency estimates (cycles @ 250 MHz = 4 ns/cycle)
LATENCY_ESTIMATES = {
    'baseline_calc': 50,      # 50 cycles = 200 ns
    'peak_find': 20,          # 20 cycles = 80 ns
    'cfd_calc': 30,           # 30 cycles = 120 ns
    'charge_integration': 15, # 15 cycles = 60 ns
    'gatti_score': 100,       # 100 cycles = 400 ns (dot product)
    'decision': 5,            # 5 cycles = 20 ns
    'total_without_gatti': 120,  # ~480 ns
    'total_with_gatti': 220,     # ~880 ns
}
