"""
PSD discrimination functions
"""

import numpy as np


def define_linear_discrimination(df_calibration, neutron_label='neutron'):
    """
    Define linear PSD discrimination boundary from calibration data

    PSD_threshold = a * Energy + b

    Parameters:
    -----------
    df_calibration : DataFrame
        Must have columns: ENERGY (or ENERGY_KEV), PSD, PARTICLE_TYPE
    neutron_label : str
        Label for neutron events in PARTICLE_TYPE column

    Returns:
    --------
    boundary_func : function
        PSD_threshold(energy)
    params : tuple
        (slope, intercept)
    """
    # Separate particles
    is_neutron = df_calibration['PARTICLE_TYPE'] == neutron_label

    neutrons = df_calibration[is_neutron]
    gammas = df_calibration[~is_neutron]

    energy_col = 'ENERGY_KEV' if 'ENERGY_KEV' in df_calibration.columns else 'ENERGY'

    # Bin by energy and find mean PSD for each particle type
    energy_bins = np.linspace(df_calibration[energy_col].min(),
                              df_calibration[energy_col].max(), 20)

    bin_centers = []
    threshold_vals = []

    for i in range(len(energy_bins) - 1):
        e_min, e_max = energy_bins[i], energy_bins[i+1]
        e_center = (e_min + e_max) / 2

        n_in_bin = neutrons[(neutrons[energy_col] >= e_min) &
                             (neutrons[energy_col] < e_max)]
        g_in_bin = gammas[(gammas[energy_col] >= e_min) &
                          (gammas[energy_col] < e_max)]

        if len(n_in_bin) > 10 and len(g_in_bin) > 10:
            # Threshold = midpoint between distributions
            threshold = (n_in_bin['PSD'].mean() + g_in_bin['PSD'].mean()) / 2
            bin_centers.append(e_center)
            threshold_vals.append(threshold)

    # Fit linear boundary
    params = np.polyfit(bin_centers, threshold_vals, 1)
    boundary_func = np.poly1d(params)

    print(f"Linear discrimination boundary: PSD_cut = {params[0]:.6f} * E + {params[1]:.3f}")

    return boundary_func, params


def apply_discrimination(df, boundary_func):
    """
    Classify events as neutron or gamma

    Parameters:
    -----------
    df : DataFrame
        Must have ENERGY and PSD columns
    boundary_func : function
        PSD threshold vs energy

    Returns:
    --------
    df : DataFrame
        With new column 'PARTICLE' ('neutron' or 'gamma')
    """
    energy_col = 'ENERGY_KEV' if 'ENERGY_KEV' in df.columns else 'ENERGY'

    threshold = boundary_func(df[energy_col])
    df['PARTICLE'] = 'gamma'
    df.loc[df['PSD'] > threshold, 'PARTICLE'] = 'neutron'

    n_neutrons = (df['PARTICLE'] == 'neutron').sum()
    n_gammas = (df['PARTICLE'] == 'gamma').sum()

    print(f"Discrimination results: {n_neutrons} neutrons, {n_gammas} gammas")

    return df
