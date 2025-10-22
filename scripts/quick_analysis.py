#!/usr/bin/env python
"""
Quick PSD analysis script
Usage: python quick_analysis.py data_file.csv
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from psd_analysis import (
    load_psd_data,
    validate_events,
    calculate_psd_ratio
)
from psd_analysis.visualization import plot_psd_scatter, plot_energy_spectra


def main():
    parser = argparse.ArgumentParser(description='Quick PSD analysis')
    parser.add_argument('data_file', help='Path to CSV data file')
    parser.add_argument('--energy-range', nargs=2, type=float, default=None,
                       help='Energy range for analysis (min max)')
    parser.add_argument('--output-dir', default='results/figures',
                       help='Output directory for plots')

    args = parser.parse_args()

    print(f"Loading data from {args.data_file}...")
    df = load_psd_data(args.data_file)

    print("Performing quality control...")
    valid_mask, qc_report = validate_events(df)
    df_clean = df[valid_mask].copy()

    print(f"\nQC Summary:")
    print(f"  Valid events: {qc_report['valid_events']}/{qc_report['total_events']}")
    print(f"  Rejection rate: {qc_report['rejection_rate']*100:.2f}%")

    print("\nCalculating PSD parameter...")
    df_clean = calculate_psd_ratio(df_clean)

    # Plot PSD scatter
    print("\nCreating PSD scatter plot...")
    plot_psd_scatter(df_clean,
                    energy_range=tuple(args.energy_range) if args.energy_range else None,
                    save_path=f"{args.output_dir}/psd_scatter.png")

    # Plot energy spectrum
    print("Creating energy spectrum...")
    plot_energy_spectra(df_clean,
                       energy_range=tuple(args.energy_range) if args.energy_range else None,
                       save_path=f"{args.output_dir}/energy_spectrum.png")

    print(f"\nAnalysis complete! Plots saved to {args.output_dir}/")
    print(f"Total events analyzed: {len(df_clean)}")

    plt.show()


if __name__ == '__main__':
    main()
