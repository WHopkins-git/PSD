#!/usr/bin/env python
"""
Demonstration of timing feature extraction
Usage: python timing_features_demo.py --data data.csv --n-events 100
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from psd_analysis import load_psd_data, validate_events
from psd_analysis.features.timing import TimingFeatureExtractor
from psd_analysis.features.timing_v2 import EnhancedTimingFeatureExtractor


def plot_waveform_with_features(waveform, features, dt=4.0):
    """Plot waveform with extracted features overlaid"""
    fig, ax = plt.subplots(figsize=(14, 6))

    time = np.arange(len(waveform)) * dt
    ax.plot(time, waveform, 'b-', linewidth=1.5, label='Waveform')

    # Mark peak
    peak_idx = np.argmax(waveform)
    ax.plot(time[peak_idx], waveform[peak_idx], 'ro', markersize=10,
           label=f'Peak: {waveform[peak_idx]:.0f} ADC')

    # Mark rise time thresholds
    peak_val = waveform[peak_idx]
    ax.axhline(0.1 * peak_val, color='green', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axhline(0.9 * peak_val, color='red', linestyle='--', alpha=0.5, label='90% threshold')

    # Add feature annotations
    textstr = f"Rise Time (10-90%): {features.get('rise_time_10_90', 0):.2f} ns\n"
    textstr += f"Peak Amplitude: {features.get('peak_amplitude', 0):.0f}\n"
    textstr += f"Decay τ (fast): {features.get('decay_tau_fast', 0):.2f} ns\n"
    textstr += f"PSD Ratio: {features.get('psd_traditional', 0):.3f}"

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('ADC Value', fontsize=12)
    ax.set_title('Waveform with Extracted Features', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def compare_feature_extractors(waveform, dt=4.0, sampling_rate_mhz=250):
    """Compare basic vs enhanced feature extraction"""
    print("\nComparing Feature Extractors:")
    print("="*60)

    # Basic extractor
    basic_extractor = TimingFeatureExtractor(sampling_rate_mhz=sampling_rate_mhz)
    basic_features = basic_extractor.extract_all_features(waveform)

    print(f"\nBasic Extractor: {len(basic_features)} features")
    print("Sample features:")
    for i, (key, value) in enumerate(list(basic_features.items())[:10]):
        print(f"  {key}: {value:.4f}")

    # Enhanced extractor
    enhanced_extractor = EnhancedTimingFeatureExtractor(sampling_rate_mhz=sampling_rate_mhz)
    enhanced_features = enhanced_extractor.extract_all_features(waveform)

    print(f"\nEnhanced Extractor: {len(enhanced_features)} features")
    print("Sample features:")
    for i, (key, value) in enumerate(list(enhanced_features.items())[:10]):
        print(f"  {key}: {value:.4f}")

    print("\n" + "="*60)

    return basic_features, enhanced_features


def main():
    parser = argparse.ArgumentParser(description='Timing feature extraction demo')
    parser.add_argument('--data', required=True, help='Data file')
    parser.add_argument('--n-events', type=int, default=100,
                       help='Number of events to process')
    parser.add_argument('--output-dir', default='results/figures',
                       help='Output directory for plots')

    args = parser.parse_args()

    print("="*60)
    print("TIMING FEATURE EXTRACTION DEMONSTRATION")
    print("="*60)

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = load_psd_data(args.data)

    # Quality control
    valid_mask, qc_report = validate_events(df)
    df_clean = df[valid_mask].copy()

    print(f"Valid events: {qc_report['valid_events']}")

    # Select random events
    n_events = min(args.n_events, len(df_clean))
    sample_df = df_clean.sample(n=n_events, random_state=42)

    # Get waveform columns
    sample_cols = [col for col in df_clean.columns if col.startswith('SAMPLE')]

    if len(sample_cols) == 0:
        print("ERROR: No waveform samples found in data")
        return

    print(f"\nProcessing {n_events} events...")
    print(f"Waveform length: {len(sample_cols)} samples")

    # Extract features from first event for demo
    first_waveform = sample_df[sample_cols].iloc[0].values

    # Compare extractors
    basic_features, enhanced_features = compare_feature_extractors(first_waveform)

    # Plot first waveform with features
    print("\nGenerating visualization...")
    fig, ax = plot_waveform_with_features(first_waveform, enhanced_features)
    plt.savefig(f"{args.output_dir}/waveform_features_demo.png", dpi=150, bbox_inches='tight')

    # Extract features for all events
    print("\nExtracting features for all events...")
    extractor = EnhancedTimingFeatureExtractor(sampling_rate_mhz=250)

    all_features = []
    for idx, row in sample_df.iterrows():
        waveform = row[sample_cols].values
        features = extractor.extract_all_features(waveform)
        all_features.append(features)

    # Create DataFrame
    features_df = pd.DataFrame(all_features)

    # Summary statistics
    print("\nFeature Statistics:")
    print("="*60)
    print(features_df.describe().T[['mean', 'std', 'min', 'max']])

    # Save features
    output_file = f"{args.output_dir}/extracted_features.csv"
    features_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")

    # Plot feature distributions
    print("\nGenerating feature distribution plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    important_features = [
        'psd_traditional', 'rise_time_10_90', 'decay_tau_ratio',
        'charge_ratio_0_60ns', 'peak_amplitude', 'gatti_score'
    ]

    for ax, feature in zip(axes, important_features):
        if feature in features_df.columns:
            ax.hist(features_df[feature].dropna(), bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'Distribution of {feature}', fontsize=12)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/feature_distributions.png", dpi=150, bbox_inches='tight')

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print(f"Plots saved to: {args.output_dir}/")

    plt.show()


if __name__ == '__main__':
    main()
