"""
Plotting functions for PSD analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_psd_scatter(df, energy_range=None, psd_range=(0, 1), boundary_func=None,
                    bins=(200, 200), figsize=(12, 8), save_path=None):
    """
    Create 2D histogram: Energy vs PSD

    Parameters:
    -----------
    df : DataFrame
        Must have 'ENERGY' (or 'ENERGY_KEV') and 'PSD' columns
    energy_range : tuple, optional
        (min, max) energy range
    psd_range : tuple
        (min, max) PSD range
    boundary_func : function, optional
        PSD discrimination boundary
    bins : tuple
        (x_bins, y_bins)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    energy_col = 'ENERGY_KEV' if 'ENERGY_KEV' in df.columns else 'ENERGY'

    fig, ax = plt.subplots(figsize=figsize)

    # Filter data
    data = df[[energy_col, 'PSD']].dropna()

    if energy_range is not None:
        data = data[(data[energy_col] >= energy_range[0]) &
                   (data[energy_col] <= energy_range[1])]

    # 2D histogram
    h, xedges, yedges, im = ax.hist2d(
        data[energy_col], data['PSD'],
        bins=bins,
        range=[energy_range, psd_range],
        cmap='viridis',
        norm=plt.matplotlib.colors.LogNorm()
    )

    # Overlay boundary
    if boundary_func is not None:
        e_plot = np.linspace(xedges[0], xedges[-1], 500)
        psd_boundary = boundary_func(e_plot)
        ax.plot(e_plot, psd_boundary, 'r--', linewidth=2, label='Discrimination boundary')
        ax.legend(fontsize=12)

    ax.set_xlabel('Energy (keV)' if 'KEV' in energy_col else 'Energy (ADC)', fontsize=14)
    ax.set_ylabel('PSD Parameter', fontsize=14)
    ax.set_title('PSD Scatter Plot', fontsize=16)

    plt.colorbar(im, ax=ax, label='Counts')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_energy_spectra(df, energy_col='ENERGY_KEV', energy_range=None, bins=1000,
                       separate_particles=True, log_scale=True, figsize=(14, 6), save_path=None):
    """
    Plot energy spectrum, optionally separated by particle type

    Parameters:
    -----------
    df : DataFrame
        Event data
    energy_col : str
        Energy column name
    energy_range : tuple, optional
        (min, max) range
    bins : int
        Number of bins
    separate_particles : bool
        Show n/γ separately if 'PARTICLE' column exists
    log_scale : bool
        Use log scale for y-axis
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if energy_range is None:
        energy_range = (df[energy_col].min(), df[energy_col].max())

    if separate_particles and 'PARTICLE' in df.columns:
        # Separate by particle type
        neutrons = df[df['PARTICLE'] == 'neutron'][energy_col]
        gammas = df[df['PARTICLE'] == 'gamma'][energy_col]

        ax.hist(neutrons, bins=bins, range=energy_range, alpha=0.6,
               label=f'Neutrons ({len(neutrons)})', color='red', histtype='step', linewidth=2)
        ax.hist(gammas, bins=bins, range=energy_range, alpha=0.6,
               label=f'Gammas ({len(gammas)})', color='blue', histtype='step', linewidth=2)
        ax.legend(fontsize=12)
    else:
        ax.hist(df[energy_col], bins=bins, range=energy_range, color='blue', alpha=0.7)

    ax.set_xlabel('Energy (keV)' if 'KEV' in energy_col else 'Energy (ADC)', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_title('Energy Spectrum', fontsize=16)
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_calibration_curve(calibration_points, cal_func, figsize=(12, 5), save_path=None):
    """
    Plot energy calibration with residuals

    Parameters:
    -----------
    calibration_points : list of tuples
        [(adc1, keV1), (adc2, keV2), ...]
    cal_func : function
        Calibration function
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    adc_vals = np.array([p[0] for p in calibration_points])
    kev_vals = np.array([p[1] for p in calibration_points])

    # Calibration curve
    adc_plot = np.linspace(adc_vals.min() * 0.9, adc_vals.max() * 1.1, 500)
    kev_plot = cal_func(adc_plot)

    ax1.plot(adc_plot, kev_plot, 'b-', linewidth=2, label='Fitted curve')
    ax1.plot(adc_vals, kev_vals, 'ro', markersize=10, label='Calibration points')
    ax1.set_xlabel('ADC Channel', fontsize=12)
    ax1.set_ylabel('Energy (keV)', fontsize=12)
    ax1.set_title('Energy Calibration', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals
    predicted = cal_func(adc_vals)
    residuals = kev_vals - predicted

    ax2.plot(kev_vals, residuals, 'ro', markersize=10)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Energy (keV)', fontsize=12)
    ax2.set_ylabel('Residual (keV)', fontsize=12)
    ax2.set_title('Calibration Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, (ax1, ax2)
