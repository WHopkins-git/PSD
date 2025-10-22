"""Spectroscopy and isotope identification modules"""

from .peak_finding import find_peaks_in_spectrum, fit_gaussian_peak

__all__ = ['find_peaks_in_spectrum', 'fit_gaussian_peak']
