"""
Data loading functions for PSD analysis
"""

import pandas as pd


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
