"""
Detector efficiency calibration and modeling
"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit
import warnings


class EfficiencyCurve:
    """
    Manages detector absolute efficiency vs energy calibration

    Efficiency = (counts detected) / (gammas emitted toward detector)
    """

    def __init__(self):
        """Initialize empty efficiency curve"""
        self.energies = []
        self.efficiencies = []
        self.uncertainties = []
        self.interpolator = None
        self.fit_method = None

    def add_calibration_point(self, energy_keV, efficiency, uncertainty=None):
        """
        Add measurement from calibrated source

        Parameters:
        -----------
        energy_keV : float
            Gamma-ray energy
        efficiency : float
            Absolute efficiency (0-1)
        uncertainty : float, optional
            Uncertainty in efficiency
        """
        self.energies.append(energy_keV)
        self.efficiencies.append(efficiency)
        self.uncertainties.append(uncertainty if uncertainty is not None else 0.1 * efficiency)

        # Clear interpolator (needs refitting)
        self.interpolator = None

    def fit_efficiency_curve(self, method='log_polynomial', degree=3):
        """
        Fit curve through calibration points

        Parameters:
        -----------
        method : str
            'log_polynomial', 'spline', or 'linear'
        degree : int
            Polynomial degree (for log_polynomial method)

        Returns:
        --------
        fit_params : array or None
            Fit parameters if applicable
        """
        if len(self.energies) < 2:
            raise ValueError("Need at least 2 calibration points")

        energies = np.array(self.energies)
        efficiencies = np.array(self.efficiencies)
        uncertainties = np.array(self.uncertainties)

        # Sort by energy
        sort_idx = np.argsort(energies)
        energies = energies[sort_idx]
        efficiencies = efficiencies[sort_idx]
        uncertainties = uncertainties[sort_idx]

        self.fit_method = method

        if method == 'log_polynomial':
            # Fit log(efficiency) vs log(energy)
            # Physically motivated: eff ~ E^(-n) at high energies
            log_e = np.log(energies)
            log_eff = np.log(efficiencies)
            weights = 1.0 / (uncertainties / efficiencies)  # Propagate to log space

            fit_params = np.polyfit(log_e, log_eff, degree, w=weights)

            def interpolator(e):
                return np.exp(np.polyval(fit_params, np.log(e)))

            self.interpolator = interpolator
            return fit_params

        elif method == 'spline':
            # Cubic spline interpolation
            self.interpolator = UnivariateSpline(energies, efficiencies,
                                                  w=1.0/uncertainties,
                                                  k=min(3, len(energies)-1))
            return None

        elif method == 'linear':
            # Linear interpolation
            self.interpolator = interp1d(energies, efficiencies,
                                          kind='linear',
                                          fill_value='extrapolate')
            return None

        else:
            raise ValueError(f"Unknown fit method: {method}")

    def get_efficiency(self, energy_keV):
        """
        Get interpolated efficiency at any energy

        Parameters:
        -----------
        energy_keV : float or array
            Energy value(s)

        Returns:
        --------
        efficiency : float or array
            Interpolated efficiency
        """
        if self.interpolator is None:
            raise ValueError("Must fit curve first using fit_efficiency_curve()")

        eff = self.interpolator(energy_keV)

        # Ensure physical values (0 to 1)
        eff = np.clip(eff, 0.0, 1.0)

        # Warn if extrapolating
        e_min, e_max = min(self.energies), max(self.energies)
        if np.any(energy_keV < e_min) or np.any(energy_keV > e_max):
            warnings.warn(f"Extrapolating outside calibrated range ({e_min}-{e_max} keV)")

        return eff

    def plot(self, ax=None, energy_range=None, show_points=True):
        """
        Plot efficiency curve

        Parameters:
        -----------
        ax : matplotlib axis, optional
            Axis to plot on
        energy_range : tuple, optional
            (min, max) energy range to plot
        show_points : bool
            Show calibration points
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if self.interpolator is None:
            raise ValueError("Must fit curve first")

        # Energy range for plotting
        if energy_range is None:
            e_min = min(self.energies) * 0.5
            e_max = max(self.energies) * 1.5
        else:
            e_min, e_max = energy_range

        # Plot fitted curve
        e_plot = np.linspace(e_min, e_max, 500)
        eff_plot = self.get_efficiency(e_plot)

        ax.plot(e_plot, eff_plot * 100, 'b-', linewidth=2, label='Fitted curve')

        # Plot calibration points
        if show_points:
            energies = np.array(self.energies)
            efficiencies = np.array(self.efficiencies) * 100
            uncertainties = np.array(self.uncertainties) * 100

            ax.errorbar(energies, efficiencies, yerr=uncertainties,
                       fmt='ro', markersize=8, capsize=5, capthick=2,
                       label='Calibration points')

        ax.set_xlabel('Energy (keV)', fontsize=12)
        ax.set_ylabel('Absolute Efficiency (%)', fontsize=12)
        ax.set_title('Detector Efficiency vs Energy', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')

        return ax


def calculate_efficiency_from_source(counts, activity_bq, branching_ratio,
                                     live_time_sec, distance_cm=None,
                                     solid_angle=None):
    """
    Calculate absolute efficiency from calibrated source measurement

    Parameters:
    -----------
    counts : float
        Net counts in photopeak (background subtracted)
    activity_bq : float
        Source activity in Becquerels
    branching_ratio : float
        Gamma emission probability (0-1)
    live_time_sec : float
        Measurement live time in seconds
    distance_cm : float, optional
        Source-detector distance (cm)
    solid_angle : float, optional
        Solid angle fraction (0-1), if not using distance

    Returns:
    --------
    efficiency : float
        Absolute efficiency
    uncertainty : float
        Poisson uncertainty
    """
    # Total gammas emitted
    gammas_emitted = activity_bq * branching_ratio * live_time_sec

    # Calculate solid angle if distance provided
    if solid_angle is None:
        if distance_cm is None:
            raise ValueError("Must provide either distance_cm or solid_angle")
        # Assume circular detector, 2.54 cm (1 inch) diameter
        detector_radius_cm = 2.54 / 2
        solid_angle = np.pi * detector_radius_cm**2 / (4 * np.pi * distance_cm**2)

    # Gammas directed toward detector
    gammas_toward_detector = gammas_emitted * solid_angle

    # Efficiency
    efficiency = counts / gammas_toward_detector

    # Poisson uncertainty
    uncertainty = np.sqrt(counts) / gammas_toward_detector

    return efficiency, uncertainty


def calculate_activity(peak_counts, efficiency, branching_ratio, live_time_sec):
    """
    Calculate source activity from peak counts

    Parameters:
    -----------
    peak_counts : float
        Net counts in photopeak
    efficiency : float
        Detector efficiency at this energy (0-1)
    branching_ratio : float
        Gamma emission probability (0-1)
    live_time_sec : float
        Measurement live time

    Returns:
    --------
    activity_bq : float
        Estimated activity in Becquerels
    uncertainty_bq : float
        Uncertainty
    """
    activity_bq = peak_counts / (efficiency * branching_ratio * live_time_sec)
    uncertainty_bq = np.sqrt(peak_counts) / (efficiency * branching_ratio * live_time_sec)

    return activity_bq, uncertainty_bq
