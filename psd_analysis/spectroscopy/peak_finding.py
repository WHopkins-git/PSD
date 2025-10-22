"""
Peak finding and fitting for spectroscopy
"""

import numpy as np
from scipy import signal, optimize


def find_peaks_in_spectrum(energy, counts, prominence=100, distance=20):
    """
    Find peaks in energy spectrum

    Parameters:
    -----------
    energy : array
        Energy bin centers
    counts : array
        Spectrum counts
    prominence : float
        Minimum peak prominence above background
    distance : int
        Minimum distance between peaks (in bins)

    Returns:
    --------
    peak_energies : array
        Energy values of peaks
    peak_counts : array
        Count values at peaks
    peak_properties : dict
        Additional peak properties
    """
    # Find peaks
    peaks, properties = signal.find_peaks(counts,
                                          prominence=prominence,
                                          distance=distance)

    peak_energies = energy[peaks]
    peak_counts = counts[peaks]

    print(f"Found {len(peaks)} peaks:")
    for i, (e, c) in enumerate(zip(peak_energies, peak_counts)):
        print(f"  Peak {i+1}: E = {e:.1f} keV, counts = {c:.0f}")

    return peak_energies, peak_counts, properties


def fit_gaussian_peak(energy, counts, peak_energy, fit_width=50):
    """
    Fit Gaussian to peak for accurate centroid and FWHM

    Parameters:
    -----------
    energy : array
        Energy bin centers
    counts : array
        Spectrum counts
    peak_energy : float
        Approximate peak location
    fit_width : float
        Width of region to fit (keV)

    Returns:
    --------
    fit_params : dict
        Fitted parameters: centroid, amplitude, sigma, fwhm
    """
    # Select fit region
    mask = np.abs(energy - peak_energy) < fit_width
    e_fit = energy[mask]
    c_fit = counts[mask]

    # Gaussian + linear background
    def model(e, amp, mu, sigma, bg_slope, bg_offset):
        gaussian = amp * np.exp(-0.5 * ((e - mu) / sigma)**2)
        background = bg_slope * e + bg_offset
        return gaussian + background

    # Initial guess
    amp_guess = c_fit.max()
    mu_guess = peak_energy
    sigma_guess = 20  # keV

    try:
        popt, pcov = optimize.curve_fit(model, e_fit, c_fit,
                                        p0=[amp_guess, mu_guess, sigma_guess, 0, c_fit.min()])

        amp, mu, sigma, bg_slope, bg_offset = popt
        fwhm = 2.355 * abs(sigma)
        resolution = fwhm / mu * 100  # Percent

        fit_params = {
            'centroid': mu,
            'amplitude': amp,
            'sigma': abs(sigma),
            'fwhm': fwhm,
            'resolution_percent': resolution,
            'background': (bg_slope, bg_offset)
        }

        print(f"Peak fit: E = {mu:.2f} keV, FWHM = {fwhm:.2f} keV, R = {resolution:.1f}%")

        return fit_params

    except:
        print(f"Warning: Fit failed for peak at {peak_energy:.1f} keV")
        return None
