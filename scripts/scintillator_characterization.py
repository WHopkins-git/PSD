#!/usr/bin/env python
"""
Characterize scintillator properties from measurement data
Usage: python scintillator_characterization.py --neutron ambe.csv --gamma cs137.csv
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from psd_analysis import (
    load_psd_data,
    validate_events,
    calculate_psd_ratio,
    calculate_figure_of_merit
)
from psd_analysis.utils.scintillator import (
    characterize_scintillator_from_data,
    compare_scintillators,
    plot_scintillator_comparison
)


def analyze_decay_times(waveforms, dt=4.0, baseline_samples=50):
    """
    Analyze decay time constants from waveforms

    Parameters:
    -----------
    waveforms : array (N, samples)
        Waveform samples
    dt : float
        Sampling period (ns)
    baseline_samples : int
        Samples before pulse

    Returns:
    --------
    decay_stats : dict
        Decay time statistics
    """
    from scipy.optimize import curve_fit

    decay_constants = {'fast': [], 'slow': []}

    for waveform in waveforms[:1000]:  # Analyze first 1000 events
        # Baseline subtraction
        baseline = np.mean(waveform[:baseline_samples])
        pulse = waveform - baseline

        peak_idx = np.argmax(pulse)
        if peak_idx < baseline_samples or peak_idx > len(pulse) - 100:
            continue

        # Fast component: 0-20 ns after peak
        fast_region = slice(peak_idx, peak_idx + int(20/dt))
        t_fast = np.arange(len(pulse[fast_region])) * dt
        y_fast = pulse[fast_region]

        if len(t_fast) > 3 and y_fast[0] > 0:
            try:
                popt, _ = curve_fit(lambda t, tau: y_fast[0] * np.exp(-t/tau),
                                   t_fast, y_fast, p0=[3.0], bounds=(0.1, 20))
                decay_constants['fast'].append(popt[0])
            except:
                pass

        # Slow component: 20-100 ns after peak
        slow_region = slice(peak_idx + int(20/dt), peak_idx + int(100/dt))
        t_slow = np.arange(len(pulse[slow_region])) * dt
        y_slow = pulse[slow_region]

        if len(t_slow) > 3 and y_slow[0] > 0:
            try:
                popt, _ = curve_fit(lambda t, tau: y_slow[0] * np.exp(-t/tau),
                                   t_slow, y_slow, p0=[30.0], bounds=(10, 200))
                decay_constants['slow'].append(popt[0])
            except:
                pass

    stats = {
        'decay_time_fast': np.median(decay_constants['fast']) if decay_constants['fast'] else 0,
        'decay_time_fast_std': np.std(decay_constants['fast']) if decay_constants['fast'] else 0,
        'decay_time_slow': np.median(decay_constants['slow']) if decay_constants['slow'] else 0,
        'decay_time_slow_std': np.std(decay_constants['slow']) if decay_constants['slow'] else 0,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Scintillator characterization')
    parser.add_argument('--neutron', required=True, help='Neutron source data')
    parser.add_argument('--gamma', required=True, help='Gamma source data')
    parser.add_argument('--name', default='MyDetector', help='Scintillator name')
    parser.add_argument('--compare', nargs='+', default=['EJ-301', 'EJ-309', 'Stilbene'],
                       help='Scintillators to compare against')
    parser.add_argument('--output-dir', default='results/figures',
                       help='Output directory for plots')

    args = parser.parse_args()

    print("="*60)
    print("SCINTILLATOR CHARACTERIZATION")
    print("="*60)

    # Load neutron data
    print(f"\nLoading neutron data from {args.neutron}...")
    df_n = load_psd_data(args.neutron)
    df_n = df_n[validate_events(df_n)[0]]
    df_n = calculate_psd_ratio(df_n)
    df_n['PARTICLE'] = 'neutron'
    print(f"  Loaded {len(df_n)} neutron events")

    # Load gamma data
    print(f"\nLoading gamma data from {args.gamma}...")
    df_g = load_psd_data(args.gamma)
    df_g = df_g[validate_events(df_g)[0]]
    df_g = calculate_psd_ratio(df_g)
    df_g['PARTICLE'] = 'gamma'
    print(f"  Loaded {len(df_g)} gamma events")

    # Combine
    df = pd.concat([df_n, df_g], ignore_index=True)

    # Characterize scintillator
    print(f"\nCharacterizing scintillator...")
    properties = characterize_scintillator_from_data(df, scintillator_name=args.name)

    # Print properties
    print("\n" + "="*60)
    print("MEASURED PROPERTIES")
    print("="*60)
    print(f"\nScintillator: {properties.name}")
    print(f"Type: {properties.type}")

    print(f"\nDecay Times:")
    print(f"  Fast: {properties.decay_time_fast:.2f} ns")
    print(f"  Slow: {properties.decay_time_slow:.2f} ns")
    print(f"  Fast Fraction: {properties.fast_fraction:.3f}")

    print(f"\nTiming:")
    print(f"  Rise Time (10-90%): {properties.rise_time:.2f} ns")

    print(f"\nPSD Performance:")
    print(f"  Figure of Merit: {properties.fom_typical:.3f}")
    print(f"  PSD Quality: {properties.psd_quality}")

    if properties.energy_resolution_662kev > 0:
        print(f"\nEnergy Resolution:")
        print(f"  @ 662 keV: {properties.energy_resolution_662kev:.2f}%")

    # Compare with known scintillators
    print("\n" + "="*60)
    print("COMPARISON WITH STANDARD SCINTILLATORS")
    print("="*60)

    comparison_df = compare_scintillators(args.compare + [args.name], criterion='psd')
    print(comparison_df.to_string())

    # Generate comparison plots
    print(f"\nGenerating comparison plots...")
    plot_scintillator_comparison(args.compare + [args.name])
    plt.savefig(f"{args.output_dir}/scintillator_comparison.png",
               dpi=150, bbox_inches='tight')

    # Plot PSD distributions
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(df_n['PSD'], bins=100, alpha=0.6, label='Neutrons', color='red', density=True)
    ax.hist(df_g['PSD'], bins=100, alpha=0.6, label='Gammas', color='blue', density=True)

    ax.set_xlabel('PSD Parameter', fontsize=12)
    ax.set_ylabel('Normalized Count', fontsize=12)
    ax.set_title(f'PSD Distributions - {args.name} (FoM={properties.fom_typical:.3f})',
                fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{args.output_dir}/psd_distributions.png",
               dpi=150, bbox_inches='tight')

    print("\n" + "="*60)
    print("CHARACTERIZATION COMPLETE")
    print("="*60)
    print(f"Plots saved to: {args.output_dir}/")

    plt.show()


if __name__ == '__main__':
    main()
