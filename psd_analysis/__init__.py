"""
PSD Analysis Toolkit
Complete toolkit for neutron/gamma discrimination and NORM source analysis
"""

__version__ = '1.0.0'

# Import key functions for easy access
from .io.data_loader import load_psd_data
from .io.quality_control import validate_events
from .calibration.energy_cal import calibrate_energy, find_compton_edge
from .psd.parameters import calculate_psd_ratio, calculate_figure_of_merit
from .psd.discrimination import define_linear_discrimination, apply_discrimination
from .spectroscopy.peak_finding import find_peaks_in_spectrum, fit_gaussian_peak
from .utils.isotope_library import ISOTOPE_LIBRARY

__all__ = [
    'load_psd_data',
    'validate_events',
    'calibrate_energy',
    'find_compton_edge',
    'calculate_psd_ratio',
    'calculate_figure_of_merit',
    'define_linear_discrimination',
    'apply_discrimination',
    'find_peaks_in_spectrum',
    'fit_gaussian_peak',
    'ISOTOPE_LIBRARY',
]
