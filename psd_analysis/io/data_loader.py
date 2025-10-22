"""
Data loading functions for PSD analysis
"""

import pandas as pd
import numpy as np


def load_psd_data(filename, delimiter=';'):
    """
    Load PSD data from CSV file with waveform samples

    Handles CSV format with structure:
    BOARD;CHANNEL;TIMETAG;ENERGY;ENERGYSHORT;FLAGS;PROBE_CODE;SAMPLES
    where SAMPLES column contains many waveform sample values

    Parameters:
    -----------
    filename : str
        Path to CSV file
    delimiter : str
        Column separator (default ';')

    Returns:
    --------
    df : pandas DataFrame
        Loaded data with columns:
        - BOARD, CHANNEL, TIMETAG, ENERGY, ENERGYSHORT, FLAGS, PROBE_CODE
        - SAMPLE_0, SAMPLE_1, ..., SAMPLE_N (waveform samples)
    """
    # Read file manually to handle variable-length waveforms
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse header
    header = lines[0].strip().split(delimiter)

    # Expected metadata columns
    metadata_cols = ['BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'ENERGYSHORT', 'FLAGS', 'PROBE_CODE']

    # Parse data rows
    data_rows = []
    max_samples = 0

    for line in lines[1:]:
        if line.strip():  # Skip empty lines
            values = line.strip().split(delimiter)

            # First 7 values are metadata
            metadata = values[:7]

            # Remaining values are waveform samples
            samples = values[7:]

            # Convert to appropriate types
            row_data = {
                'BOARD': int(metadata[0]),
                'CHANNEL': int(metadata[1]),
                'TIMETAG': int(metadata[2]),
                'ENERGY': int(metadata[3]),
                'ENERGYSHORT': int(metadata[4]),
                'FLAGS': metadata[5],  # Keep as string (hex)
                'PROBE_CODE': int(metadata[6])
            }

            # Add waveform samples
            for i, sample in enumerate(samples):
                row_data[f'SAMPLE_{i}'] = int(sample)

            max_samples = max(max_samples, len(samples))
            data_rows.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(data_rows)

    # Ensure all waveform columns exist (fill missing with NaN if needed)
    sample_cols = [f'SAMPLE_{i}' for i in range(max_samples)]
    for col in sample_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder columns: metadata first, then samples
    column_order = metadata_cols + sample_cols
    df = df[column_order]

    print(f"Loaded {len(df)} events from {filename}")
    print(f"Metadata columns: {metadata_cols}")
    print(f"Waveform samples per event: {max_samples}")

    return df
