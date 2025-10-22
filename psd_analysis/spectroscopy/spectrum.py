"""
Energy spectrum analysis and operations
"""

import numpy as np
from scipy import signal, ndimage
import warnings


class EnergySpectrum:
    """
    Class for gamma-ray energy spectrum operations
    """

    def __init__(self, energies=None, counts=None, energy_range=None, bins=1000):
        """
        Create spectrum from event energies or pre-binned counts

        Parameters:
        -----------
        energies : array, optional
            Event energies (will be binned)
        counts : array, optional
            Pre-binned counts
        energy_range : tuple
            (min, max) energy range
        bins : int or array
            Number of bins or bin edges
        """
        if energies is not None:
            # Create histogram from events
            if energy_range is None:
                energy_range = (energies.min(), energies.max())

            self.counts, bin_edges = np.histogram(energies, bins=bins, range=energy_range)
            self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            self.bin_edges = bin_edges
            self.bin_width = bin_edges[1] - bin_edges[0]

        elif counts is not None:
            # Use pre-binned counts
            self.counts = np.array(counts)

            if isinstance(bins, int):
                # Create bin centers from range
                if energy_range is None:
                    raise ValueError("Must provide energy_range if counts are pre-binned")
                self.bin_centers = np.linspace(energy_range[0], energy_range[1], len(counts))
                self.bin_width = (energy_range[1] - energy_range[0]) / len(counts)
            else:
                # Bins are edges
                self.bin_edges = np.array(bins)
                self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
                self.bin_width = self.bin_edges[1] - self.bin_edges[0]

        else:
            raise ValueError("Must provide either energies or counts")

        self.energy_range = (self.bin_centers[0], self.bin_centers[-1])

    def smooth(self, sigma=2, method='gaussian'):
        """
        Smooth spectrum for noise reduction

        Parameters:
        -----------
        sigma : float
            Smoothing parameter (bins)
        method : str
            'gaussian' or 'savgol'

        Returns:
        --------
        smoothed : EnergySpectrum
            New smoothed spectrum
        """
        if method == 'gaussian':
            smoothed_counts = ndimage.gaussian_filter1d(self.counts.astype(float), sigma)
        elif method == 'savgol':
            window = int(sigma * 4) | 1  # Ensure odd
            smoothed_counts = signal.savgol_filter(self.counts, window, polyorder=3)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        # Create new spectrum
        return EnergySpectrum(counts=smoothed_counts,
                            bins=self.bin_centers,
                            energy_range=self.energy_range)

    def rebin(self, new_bins):
        """
        Change binning (coarser or finer)

        Parameters:
        -----------
        new_bins : int
            New number of bins

        Returns:
        --------
        rebinned : EnergySpectrum
            Rebinned spectrum
        """
        # Interpolate to new bins
        new_bin_edges = np.linspace(self.energy_range[0], self.energy_range[1], new_bins + 1)
        new_bin_centers = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2

        # Piecewise constant interpolation (conserve counts)
        new_counts = np.interp(new_bin_centers, self.bin_centers, self.counts)

        return EnergySpectrum(counts=new_counts,
                            bins=new_bin_edges,
                            energy_range=self.energy_range)

    def subtract_background(self, background_spectrum, scale_factor=1.0):
        """
        Background subtraction with proper scaling

        Parameters:
        -----------
        background_spectrum : EnergySpectrum
            Background spectrum to subtract
        scale_factor : float
            Scaling factor for background (e.g., for different measurement times)

        Returns:
        --------
        subtracted : EnergySpectrum
            Background-subtracted spectrum
        """
        if len(self.counts) != len(background_spectrum.counts):
            raise ValueError("Spectra must have same binning")

        subtracted_counts = self.counts - scale_factor * background_spectrum.counts

        # Don't allow negative counts
        subtracted_counts = np.maximum(subtracted_counts, 0)

        return EnergySpectrum(counts=subtracted_counts,
                            bins=self.bin_centers,
                            energy_range=self.energy_range)

    def get_roi_counts(self, energy_min, energy_max):
        """
        Get counts in region of interest

        Parameters:
        -----------
        energy_min, energy_max : float
            Energy range (keV)

        Returns:
        --------
        counts : float
            Total counts in ROI
        uncertainty : float
            Poisson uncertainty (sqrt(N))
        """
        mask = (self.bin_centers >= energy_min) & (self.bin_centers <= energy_max)
        roi_counts = np.sum(self.counts[mask])
        uncertainty = np.sqrt(roi_counts)

        return roi_counts, uncertainty

    def plot(self, ax=None, **kwargs):
        """
        Plot spectrum

        Parameters:
        -----------
        ax : matplotlib axis, optional
            Axis to plot on
        **kwargs : additional arguments for plt.plot

        Returns:
        --------
        ax : matplotlib axis
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        default_kwargs = {'linewidth': 1, 'color': 'blue'}
        default_kwargs.update(kwargs)

        ax.plot(self.bin_centers, self.counts, **default_kwargs)
        ax.set_xlabel('Energy (keV)', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('Energy Spectrum', fontsize=14)
        ax.grid(True, alpha=0.3)

        return ax


def subtract_compton_continuum(spectrum, method='smoothing', sigma=50):
    """
    Estimate and subtract Compton continuum background

    Parameters:
    -----------
    spectrum : EnergySpectrum
        Spectrum to process
    method : str
        'smoothing' (heavily smoothed as background) or 'linear' (interpolate valleys)
    sigma : float
        Smoothing parameter (for smoothing method)

    Returns:
    --------
    subtracted : EnergySpectrum
        Spectrum with continuum removed
    background : array
        Estimated background
    """
    if method == 'smoothing':
        # Heavily smooth to remove peaks, leaving continuum
        background_counts = ndimage.gaussian_filter1d(spectrum.counts.astype(float), sigma)

        # Subtract
        subtracted_counts = spectrum.counts - background_counts
        subtracted_counts = np.maximum(subtracted_counts, 0)

        return EnergySpectrum(counts=subtracted_counts,
                            bins=spectrum.bin_centers,
                            energy_range=spectrum.energy_range), background_counts

    elif method == 'linear':
        # Find local minima (valleys between peaks)
        # These represent the continuum level
        from scipy.signal import find_peaks

        # Invert to find minima
        inverted = -spectrum.counts
        minima_idx, _ = find_peaks(inverted, distance=20)

        # Add endpoints
        minima_idx = np.concatenate([[0], minima_idx, [len(spectrum.counts)-1]])
        minima_energies = spectrum.bin_centers[minima_idx]
        minima_counts = spectrum.counts[minima_idx]

        # Linear interpolation between minima
        background_counts = np.interp(spectrum.bin_centers, minima_energies, minima_counts)

        # Subtract
        subtracted_counts = spectrum.counts - background_counts
        subtracted_counts = np.maximum(subtracted_counts, 0)

        return EnergySpectrum(counts=subtracted_counts,
                            bins=spectrum.bin_centers,
                            energy_range=spectrum.energy_range), background_counts

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_count_rate(spectrum, live_time_sec):
    """
    Calculate count rate (cps) from spectrum

    Parameters:
    -----------
    spectrum : EnergySpectrum
        Energy spectrum
    live_time_sec : float
        Measurement live time

    Returns:
    --------
    total_rate : float
        Total count rate (cps)
    rate_spectrum : array
        Rate per energy bin (cps/keV)
    """
    total_counts = np.sum(spectrum.counts)
    total_rate = total_counts / live_time_sec

    rate_spectrum = spectrum.counts / (live_time_sec * spectrum.bin_width)

    return total_rate, rate_spectrum
