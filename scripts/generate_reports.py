#!/usr/bin/env python
"""
Generate comprehensive analysis reports
Usage: python generate_reports.py --data analyzed_data.csv --output report.txt
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from psd_analysis import EnergySpectrum
from psd_analysis.spectroscopy import identify_isotopes


def generate_header(sample_info):
    """Generate report header"""
    lines = []
    lines.append("="*80)
    lines.append("PSD ANALYSIS REPORT")
    lines.append("="*80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if sample_info:
        lines.append("\nSample Information:")
        for key, value in sample_info.items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def generate_qc_section(df, qc_report=None):
    """Generate quality control section"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("QUALITY CONTROL")
    lines.append("="*80)

    if qc_report:
        lines.append(f"\nTotal Events: {qc_report['total_events']}")
        lines.append(f"Valid Events: {qc_report['valid_events']}")
        lines.append(f"Rejection Rate: {qc_report['rejection_rate']*100:.2f}%")
        lines.append(f"\nRejection Reasons:")
        lines.append(f"  Saturated: {qc_report.get('saturated', 0)} events")
        lines.append(f"  Unstable Baseline: {qc_report.get('unstable_baseline', 0)} events")
    else:
        lines.append(f"\nTotal Events: {len(df)}")

    return "\n".join(lines)


def generate_psd_section(df):
    """Generate PSD analysis section"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("PSD ANALYSIS")
    lines.append("="*80)

    if 'PSD' in df.columns:
        lines.append(f"\nPSD Statistics:")
        lines.append(f"  Mean: {df['PSD'].mean():.4f}")
        lines.append(f"  Std Dev: {df['PSD'].std():.4f}")
        lines.append(f"  Min: {df['PSD'].min():.4f}")
        lines.append(f"  Max: {df['PSD'].max():.4f}")

    if 'PARTICLE' in df.columns:
        neutron_count = (df['PARTICLE'] == 'neutron').sum()
        gamma_count = (df['PARTICLE'] == 'gamma').sum()

        lines.append(f"\nParticle Classification:")
        lines.append(f"  Neutrons: {neutron_count} ({neutron_count/len(df)*100:.2f}%)")
        lines.append(f"  Gammas: {gamma_count} ({gamma_count/len(df)*100:.2f}%)")

        if 'PSD' in df.columns:
            neutron_psd = df[df['PARTICLE'] == 'neutron']['PSD']
            gamma_psd = df[df['PARTICLE'] == 'gamma']['PSD']

            if len(neutron_psd) > 0 and len(gamma_psd) > 0:
                separation = abs(neutron_psd.mean() - gamma_psd.mean())
                fwhm_n = 2.355 * neutron_psd.std()
                fwhm_g = 2.355 * gamma_psd.std()
                fom = separation / (fwhm_n + fwhm_g)

                lines.append(f"\nSeparation Quality:")
                lines.append(f"  Figure of Merit: {fom:.3f}")
                lines.append(f"  Neutron PSD: {neutron_psd.mean():.3f} ± {neutron_psd.std():.3f}")
                lines.append(f"  Gamma PSD: {gamma_psd.mean():.3f} ± {gamma_psd.std():.3f}")

    return "\n".join(lines)


def generate_energy_section(df):
    """Generate energy analysis section"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("ENERGY ANALYSIS")
    lines.append("="*80)

    energy_col = 'ENERGY_KEV' if 'ENERGY_KEV' in df.columns else 'ENERGY'

    lines.append(f"\nEnergy Statistics:")
    lines.append(f"  Mean: {df[energy_col].mean():.2f}")
    lines.append(f"  Std Dev: {df[energy_col].std():.2f}")
    lines.append(f"  Min: {df[energy_col].min():.2f}")
    lines.append(f"  Max: {df[energy_col].max():.2f}")

    # Energy bins
    bins = [(0, 300), (300, 800), (800, 2000), (2000, df[energy_col].max())]
    lines.append(f"\nEnergy Distribution:")

    for e_min, e_max in bins:
        if e_max > e_min:
            count = ((df[energy_col] >= e_min) & (df[energy_col] < e_max)).sum()
            lines.append(f"  {e_min}-{e_max:.0f} keV: {count} events ({count/len(df)*100:.1f}%)")

    return "\n".join(lines)


def generate_isotope_section(df):
    """Generate isotope identification section"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("ISOTOPE IDENTIFICATION")
    lines.append("="*80)

    if 'ENERGY_KEV' in df.columns:
        # Select gamma events if available
        if 'PARTICLE' in df.columns:
            df_gamma = df[df['PARTICLE'] == 'gamma']
            lines.append(f"\nAnalyzing {len(df_gamma)} gamma events...")
        else:
            df_gamma = df
            lines.append(f"\nAnalyzing all {len(df)} events...")

        if len(df_gamma) > 100:
            # Perform isotope identification
            results = identify_isotopes(
                df_gamma['ENERGY_KEV'].values,
                prominence=50,
                distance=20,
                tolerance_keV=10,
                min_confidence=0.5
            )

            lines.append(f"\nPeaks Found: {len(results['peaks_found'])}")

            if results['identified_isotopes']:
                lines.append("\nIdentified Isotopes:")
                for isotope, info in results['identified_isotopes'].items():
                    conf_pct = info['confidence_score'] * 100
                    lines.append(f"\n  {isotope}:")
                    lines.append(f"    Confidence: {conf_pct:.1f}%")
                    lines.append(f"    Peaks Matched: {info['peaks_matched']}/{info['peaks_expected']}")
                    energies_str = ", ".join([f"{e:.1f}" for e in info['matched_energies']])
                    lines.append(f"    Energies: {energies_str} keV")
            else:
                lines.append("\nNo isotopes identified above confidence threshold")
        else:
            lines.append("\nInsufficient events for isotope identification (need >100)")
    else:
        lines.append("\nEnergy calibration required for isotope identification")

    return "\n".join(lines)


def generate_ml_section(df):
    """Generate ML classification section"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("MACHINE LEARNING CLASSIFICATION")
    lines.append("="*80)

    if 'PARTICLE_ML' in df.columns:
        neutron_count = (df['PARTICLE_ML'] == 'neutron').sum()
        gamma_count = (df['PARTICLE_ML'] == 'gamma').sum()

        lines.append(f"\nML Classification Results:")
        lines.append(f"  Neutrons: {neutron_count} ({neutron_count/len(df)*100:.2f}%)")
        lines.append(f"  Gammas: {gamma_count} ({gamma_count/len(df)*100:.2f}%)")

        if 'CONFIDENCE' in df.columns:
            lines.append(f"\nConfidence Statistics:")
            lines.append(f"  Mean Confidence: {df['CONFIDENCE'].mean():.3f}")
            lines.append(f"  Low Confidence Events (<0.2): {(df['CONFIDENCE'] < 0.2).sum()}")

        if 'PARTICLE' in df.columns:
            # Compare with traditional PSD
            agreement = (df['PARTICLE'] == df['PARTICLE_ML']).sum()
            lines.append(f"\nComparison with Traditional PSD:")
            lines.append(f"  Agreement: {agreement}/{len(df)} ({agreement/len(df)*100:.2f}%)")
    else:
        lines.append("\nNo ML classification results available")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Generate analysis report')
    parser.add_argument('--data', required=True, help='Analyzed data file (CSV)')
    parser.add_argument('--output', default='report.txt', help='Output report file')
    parser.add_argument('--sample-name', help='Sample name')
    parser.add_argument('--sample-id', help='Sample ID')
    parser.add_argument('--location', help='Measurement location')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    print(f"Loaded {len(df)} events")

    # Sample info
    sample_info = {}
    if args.sample_name:
        sample_info['Sample Name'] = args.sample_name
    if args.sample_id:
        sample_info['Sample ID'] = args.sample_id
    if args.location:
        sample_info['Location'] = args.location
    sample_info['Data File'] = args.data

    # Generate report sections
    report_sections = []

    report_sections.append(generate_header(sample_info))
    report_sections.append(generate_qc_section(df))
    report_sections.append(generate_psd_section(df))
    report_sections.append(generate_energy_section(df))
    report_sections.append(generate_isotope_section(df))
    report_sections.append(generate_ml_section(df))

    # Footer
    report_sections.append("\n" + "="*80)
    report_sections.append("END OF REPORT")
    report_sections.append("="*80)

    # Combine all sections
    full_report = "\n".join(report_sections)

    # Save report
    with open(args.output, 'w') as f:
        f.write(full_report)

    print(f"\nReport saved to: {args.output}")

    # Also print to console
    print("\n" + full_report)


if __name__ == '__main__':
    main()
