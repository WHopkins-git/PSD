#!/usr/bin/env python
"""
Complete energy calibration workflow
Usage: python calibration_workflow.py --source cs137 --data cs137_data.csv
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from psd_analysis import (
    load_psd_data,
    validate_events,
    calculate_psd_ratio,
    calibrate_energy,
    EnergySpectrum
)
from psd_analysis.spectroscopy import find_peaks_in_spectrum, fit_gaussian_peak
from psd_analysis.visualization import plot_calibration_curve, plot_energy_spectra


# Known calibration sources
KNOWN_SOURCES = {
    'cs137': {
        'name': 'Cs-137',
        'compton_edge': 477,  # keV
        'photopeak': 661.7   # keV
    },
    'co60': {
        'name': 'Co-60',
        'photopeaks': [1173.2, 1332.5]  # keV
    },
    'na22': {
        'name': 'Na-22',
        'photopeaks': [511.0, 1274.5]  # keV
    }
}


def calibrate_from_source(df, source_name, method='linear'):
    """
    Automatic calibration from known source

    Parameters:
    -----------
    df : DataFrame
        Event data
    source_name : str
        Name of calibration source ('cs137', 'co60', 'na22')
    method : str
        Calibration method

    Returns:
    --------
    df_calibrated : DataFrame
        Calibrated data
    cal_func : function
        Calibration function
    cal_params : array
        Calibration parameters
    """
    if source_name not in KNOWN_SOURCES:
        raise ValueError(f"Unknown source: {source_name}. Available: {list(KNOWN_SOURCES.keys())}")

    source_info = KNOWN_SOURCES[source_name]
    print(f"\nCalibrating with {source_info['name']}...")

    # Create energy spectrum
    spectrum = EnergySpectrum(df['ENERGY'].values, bins=2000,
                             energy_range=(0, df['ENERGY'].max()))

    # Find peaks
    print("Finding peaks...")
    peak_energies_adc, peak_counts, _ = find_peaks_in_spectrum(
        spectrum.bin_centers,
        spectrum.counts,
        prominence=50,
        distance=20
    )

    # Match peaks to known energies
    calibration_points = []

    if source_name == 'cs137':
        # For Cs-137, identify photopeak (highest peak)
        highest_peak_idx = np.argmax(peak_counts)
        photopeak_adc = peak_energies_adc[highest_peak_idx]
        calibration_points.append((photopeak_adc, source_info['photopeak']))

        print(f"  Photopeak: {photopeak_adc:.1f} ADC → {source_info['photopeak']} keV")

    elif source_name in ['co60', 'na22']:
        # For multi-peak sources, use two highest peaks
        sorted_indices = np.argsort(peak_counts)[::-1]
        top_peaks = peak_energies_adc[sorted_indices[:2]]
        top_peaks = np.sort(top_peaks)  # Ensure ascending order

        expected_energies = sorted(source_info['photopeaks'])

        for adc, kev in zip(top_peaks, expected_energies):
            calibration_points.append((adc, kev))
            print(f"  Peak: {adc:.1f} ADC → {kev} keV")

    if len(calibration_points) < 1:
        raise ValueError("Could not identify enough peaks for calibration")

    # Perform calibration
    df_calibrated, cal_func, cal_params = calibrate_energy(
        df, calibration_points, method=method
    )

    return df_calibrated, cal_func, cal_params, calibration_points


def main():
    parser = argparse.ArgumentParser(description='Energy calibration workflow')
    parser.add_argument('--data', required=True, help='Calibration data file')
    parser.add_argument('--source', required=True,
                       choices=['cs137', 'co60', 'na22'],
                       help='Calibration source')
    parser.add_argument('--method', default='linear',
                       choices=['linear', 'polynomial'],
                       help='Calibration method')
    parser.add_argument('--output-params', default='models/calibration_params.npy',
                       help='Output file for calibration parameters')
    parser.add_argument('--output-dir', default='results/figures',
                       help='Output directory for plots')

    args = parser.parse_args()

    print("="*60)
    print("ENERGY CALIBRATION WORKFLOW")
    print("="*60)

    # Load data
    print(f"\nLoading data from {args.data}...")
    df = load_psd_data(args.data)

    # Quality control
    print("Performing quality control...")
    valid_mask, qc_report = validate_events(df)
    df_clean = df[valid_mask].copy()

    print(f"  Valid events: {qc_report['valid_events']}/{qc_report['total_events']}")

    # Calculate PSD
    df_clean = calculate_psd_ratio(df_clean)

    # Perform calibration
    df_calibrated, cal_func, cal_params, cal_points = calibrate_from_source(
        df_clean, args.source, method=args.method
    )

    # Save calibration parameters
    np.save(args.output_params, cal_params)
    print(f"\nCalibration parameters saved to: {args.output_params}")

    # Plot calibration curve
    print("\nGenerating plots...")
    plot_calibration_curve(cal_points, cal_func,
                          save_path=f"{args.output_dir}/calibration_curve.png")

    # Plot calibrated spectrum
    plot_energy_spectra(df_calibrated,
                       energy_col='ENERGY_KEV',
                       energy_range=(0, 3000),
                       save_path=f"{args.output_dir}/calibrated_spectrum.png")

    # Verify peaks
    print("\nVerifying calibration...")
    hist, bins = np.histogram(df_calibrated['ENERGY_KEV'], bins=1000, range=(0, 3000))
    peak_energies, _, _ = find_peaks_in_spectrum(bins[:-1], hist, prominence=50)

    print("Identified peaks in calibrated spectrum:")
    for peak in peak_energies:
        # Fit Gaussian to get accurate centroid
        fit_result = fit_gaussian_peak(bins[:-1], hist, peak, fit_width=50)
        if fit_result:
            print(f"  {fit_result['centroid']:.2f} keV "
                  f"(FWHM={fit_result['fwhm']:.2f} keV, "
                  f"Resolution={fit_result['resolution_percent']:.1f}%)")

    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Parameters: {cal_params}")
    print(f"Plots saved to: {args.output_dir}/")

    plt.show()


if __name__ == '__main__':
    main()
