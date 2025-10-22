"""
PSD parameter calculation functions
"""

import numpy as np


def calculate_psd_ratio(df):
    """
    Calculate PSD parameter using charge integration method

    PSD = (Q_total - Q_short) / Q_total = (E - E_short) / E

    Parameters:
    -----------
    df : DataFrame
        Must have ENERGY and ENERGYSHORT columns

    Returns:
    --------
    df : DataFrame
        With new column 'PSD'
    """
    # Avoid division by zero
    mask = df['ENERGY'] > 0

    df['PSD'] = np.nan
    df.loc[mask, 'PSD'] = (df.loc[mask, 'ENERGY'] - df.loc[mask, 'ENERGYSHORT']) / df.loc[mask, 'ENERGY']

    return df


def calculate_figure_of_merit(psd_neutron, psd_gamma):
    """
    Calculate Figure of Merit for n/γ separation

    FOM = separation / (FWHM_n + FWHM_gamma)

    Higher FOM = better discrimination
    FOM > 1.0 is good, FOM > 1.5 is excellent

    Parameters:
    -----------
    psd_neutron, psd_gamma : arrays
        PSD distributions for neutrons and gammas

    Returns:
    --------
    fom : float
        Figure of merit
    """
    # Calculate means
    mean_n = np.mean(psd_neutron)
    mean_g = np.mean(psd_gamma)
    separation = abs(mean_n - mean_g)

    # Calculate FWHM (2.355 * sigma for Gaussian)
    fwhm_n = 2.355 * np.std(psd_neutron)
    fwhm_g = 2.355 * np.std(psd_gamma)

    fom = separation / (fwhm_n + fwhm_g)

    print(f"Figure of Merit: {fom:.3f}")
    print(f"  Neutron: mean={mean_n:.3f}, FWHM={fwhm_n:.3f}")
    print(f"  Gamma:   mean={mean_g:.3f}, FWHM={fwhm_g:.3f}")

    return fom
