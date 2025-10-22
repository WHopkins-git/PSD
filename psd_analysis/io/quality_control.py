"""
Quality control functions for event validation
"""

import numpy as np
import warnings


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
