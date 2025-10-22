"""
PSD Analysis Toolkit
Complete toolkit for neutron/gamma discrimination and NORM source analysis
"""

__version__ = '2.0.0'

# Import key functions for easy access
from .io.data_loader import load_psd_data
from .io.quality_control import validate_events
from .calibration.energy_cal import calibrate_energy, find_compton_edge
from .calibration.efficiency import EfficiencyCurve
from .psd.parameters import calculate_psd_ratio, calculate_figure_of_merit
from .psd.discrimination import define_linear_discrimination, apply_discrimination
from .spectroscopy.peak_finding import find_peaks_in_spectrum, fit_gaussian_peak
from .spectroscopy.spectrum import EnergySpectrum
from .spectroscopy.isotope_id import match_peaks_to_library, identify_decay_chains
from .utils.isotope_library import ISOTOPE_LIBRARY
from .visualization import plot_psd_scatter, plot_energy_spectra

__all__ = [
    'load_psd_data',
    'validate_events',
    'calibrate_energy',
    'find_compton_edge',
    'EfficiencyCurve',
    'calculate_psd_ratio',
    'calculate_figure_of_merit',
    'define_linear_discrimination',
    'apply_discrimination',
    'find_peaks_in_spectrum',
    'fit_gaussian_peak',
    'EnergySpectrum',
    'match_peaks_to_library',
    'identify_decay_chains',
    'ISOTOPE_LIBRARY',
    'plot_psd_scatter',
    'plot_energy_spectra',
]
