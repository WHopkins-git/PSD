#!/usr/bin/env python
"""
Batch process multiple PSD data files
Usage: python batch_process.py --input data/calibration/*.csv --output results/
"""

import argparse
import glob
import os
from pathlib import Path
import pandas as pd

from psd_analysis import (
    load_psd_data,
    validate_events,
    calculate_psd_ratio,
    calibrate_energy
)


def process_file(input_file, output_dir, calibration_params=None):
    """Process a single file"""
    print(f"\nProcessing: {input_file}")

    try:
        # Load data
        df = load_psd_data(input_file)

        # Quality control
        valid_mask, qc_report = validate_events(df)
        df_clean = df[valid_mask].copy()

        print(f"  QC: {qc_report['valid_events']}/{qc_report['total_events']} events passed")

        # Calculate PSD
        df_clean = calculate_psd_ratio(df_clean)

        # Apply calibration if provided
        if calibration_params is not None:
            cal_func = lambda x: calibration_params[0] * x + calibration_params[1]
            df_clean['ENERGY_KEV'] = cal_func(df_clean['ENERGY'])

        # Save processed data
        output_file = Path(output_dir) / (Path(input_file).stem + '_processed.csv')
        df_clean.to_csv(output_file, index=False)

        print(f"  Saved to: {output_file}")

        return {
            'file': input_file,
            'total_events': qc_report['total_events'],
            'valid_events': qc_report['valid_events'],
            'rejection_rate': qc_report['rejection_rate'],
            'output': str(output_file)
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'file': input_file,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Batch process PSD data files')
    parser.add_argument('--input', required=True, help='Input file pattern (e.g., data/*.csv)')
    parser.add_argument('--output', default='results/processed',
                       help='Output directory')
    parser.add_argument('--calibration', nargs=2, type=float, default=None,
                       help='Calibration parameters: slope intercept')
    parser.add_argument('--summary', default='results/batch_summary.csv',
                       help='Summary output file')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get list of files
    input_files = glob.glob(args.input)
    print(f"Found {len(input_files)} files to process")

    if len(input_files) == 0:
        print(f"No files found matching: {args.input}")
        return

    # Process each file
    results = []
    for input_file in input_files:
        result = process_file(input_file, args.output, args.calibration)
        results.append(result)

    # Create summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(args.summary, index=False)

    print(f"\n{'='*60}")
    print("Batch Processing Complete")
    print(f"{'='*60}")
    print(f"Files processed: {len(input_files)}")
    print(f"Successful: {summary_df['error'].isna().sum()}")
    print(f"Failed: {summary_df['error'].notna().sum()}")
    print(f"\nSummary saved to: {args.summary}")


if __name__ == '__main__':
    main()
