"""
Complete implementation of missing modules + ML capabilities
Part 1: Missing core modules
"""

# =============================================================================
# calibration/efficiency.py
# =============================================================================

"""
Detector efficiency calibration and modeling
"""

import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class EfficiencyCurve:
    """
    Detector efficiency vs energy calibration
    """
    
    def __init__(self):
        self.energies = None
        self.efficiencies = None
        self.uncertainties = None
        self.interpolator = None
        self.fit_params = None
        
    def add_calibration_point(self, energy_kev, efficiency, uncertainty=None):
        """
        Add a single calibration point from known source
        
        Parameters:
        -----------
        energy_kev : float
            Gamma energy
        efficiency : float
            Absolute detection efficiency (0-1)
        uncertainty : float
            Uncertainty on efficiency
        """
        if self.energies is None:
            self.energies = [energy_kev]
            self.efficiencies = [efficiency]
            self.uncertainties = [uncertainty if uncertainty else 0.1*efficiency]
        else:
            self.energies.append(energy_kev)
            self.efficiencies.append(efficiency)
            self.uncertainties.append(uncertainty if uncertainty else 0.1*efficiency)
    
    def fit_efficiency_curve(self, method='log_polynomial'):
        """
        Fit efficiency curve through calibration points
        
        Parameters:
        -----------
        method : str
            'log_polynomial' - ln(eff) = poly(ln(E))
            'spline' - cubic spline interpolation
            'semi_empirical' - physics-based functional form
        
        Returns:
        --------
        fit_params : array
            Fitted parameters
        """
        energies = np.array(self.energies)
        efficiencies = np.array(self.efficiencies)
        
        if method == 'log_polynomial':
            # ln(eff) = a + b*ln(E) + c*ln(E)^2 + d*ln(E)^3
            log_e = np.log(energies)
            log_eff = np.log(efficiencies)
            
            self.fit_params = np.polyfit(log_e, log_eff, 3)
            
            # Create interpolator
            self.interpolator = lambda E: np.exp(np.polyval(self.fit_params, np.log(E)))
            
        elif method == 'spline':
            # Cubic spline through points
            self.interpolator = UnivariateSpline(energies, efficiencies, k=3, s=0)
            self.fit_params = None
            
        elif method == 'semi_empirical':
            # Physics-motivated form for scintillators
            # eff(E) = A * exp(-B*E) * (1 - exp(-C*E^D))
            def efficiency_model(E, A, B, C, D):
                return A * np.exp(-B*E) * (1 - np.exp(-C*E**D))
            
            # Fit
            p0 = [0.1, 0.001, 0.01, 0.5]  # Initial guess
            self.fit_params, _ = curve_fit(efficiency_model, energies, efficiencies, 
                                           p0=p0, maxfev=10000)
            
            self.interpolator = lambda E: efficiency_model(E, *self.fit_params)
        
        return self.fit_params
    
    def get_efficiency(self, energy_kev):
        """
        Get efficiency at any energy
        
        Parameters:
        -----------
        energy_kev : float or array
            Energy in keV
        
        Returns:
        --------
        efficiency : float or array
            Interpolated efficiency
        """
        if self.interpolator is None:
            raise ValueError("Must fit curve before interpolating")
        
        return self.interpolator(energy_kev)
    
    def plot_efficiency_curve(self, energy_range=(0, 3000)):
        """
        Plot efficiency curve with calibration points
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot calibration points
        ax.errorbar(self.energies, self.efficiencies, yerr=self.uncertainties,
                   fmt='ro', markersize=10, capsize=5, label='Calibration points')
        
        # Plot fitted curve
        E_plot = np.linspace(energy_range[0], energy_range[1], 500)
        eff_plot = self.get_efficiency(E_plot)
        ax.plot(E_plot, eff_plot, 'b-', linewidth=2, label='Fitted curve')
        
        ax.set_xlabel('Energy (keV)', fontsize=14)
        ax.set_ylabel('Absolute Efficiency', fontsize=14)
        ax.set_title('Detector Efficiency Curve', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig, ax


def calculate_efficiency_from_source(counts, activity_bq, branching_ratio, 
                                     live_time_sec, distance_cm, 
                                     solid_angle_correction=1.0):
    """
    Calculate absolute efficiency from calibrated source measurement
    
    Parameters:
    -----------
    counts : float
        Net counts in photopeak (background subtracted)
    activity_bq : float
        Source activity in Bq
    branching_ratio : float
        Gamma emission probability (0-1)
    live_time_sec : float
        Measurement live time
    distance_cm : float
        Source-detector distance
    solid_angle_correction : float
        Geometric correction factor (default 1.0 for point source)
    
    Returns:
    --------
    efficiency : float
        Absolute efficiency
    uncertainty : float
        Statistical uncertainty
    """
    # Expected gammas emitted toward detector
    gammas_emitted = activity_bq * branching_ratio * live_time_sec * solid_angle_correction
    
    # Efficiency = detected / emitted
    efficiency = counts / gammas_emitted
    
    # Poisson uncertainty
    uncertainty = np.sqrt(counts) / gammas_emitted
    
    return efficiency, uncertainty


# =============================================================================
# psd/optimization.py
# =============================================================================

"""
PSD gate timing optimization and parameter tuning
"""

import numpy as np
from scipy.optimize import minimize_scalar, differential_evolution


def optimize_gate_timing(df_neutrons, df_gammas, 
                         short_gate_range=(5, 100),
                         long_gate_range=(50, 300),
                         sample_cols_prefix='SAMPLES'):
    """
    Optimize PSD gate timing to maximize Figure of Merit
    
    Parameters:
    -----------
    df_neutrons : DataFrame
        Pure neutron events with waveforms
    df_gammas : DataFrame
        Pure gamma events with waveforms
    short_gate_range : tuple
        (min, max) samples for short gate
    long_gate_range : tuple
        (min, max) samples for long gate
    sample_cols_prefix : str
        Prefix for sample columns
    
    Returns:
    --------
    optimal_params : dict
        Best short_gate, long_gate, and resulting FOM
    """
    # Get sample columns
    sample_cols = [col for col in df_neutrons.columns if col.startswith(sample_cols_prefix)]
    n_samples = len(sample_cols)
    
    if n_samples == 0:
        raise ValueError("No waveform samples found")
    
    samples_n = df_neutrons[sample_cols].values
    samples_g = df_gammas[sample_cols].values
    
    # Calculate baseline (first 20 samples)
    baseline_n = samples_n[:, :20].mean(axis=1, keepdims=True)
    baseline_g = samples_g[:, :20].mean(axis=1, keepdims=True)
    
    # Baseline subtract
    samples_n = baseline_n - samples_n  # Negative-going pulses
    samples_g = baseline_g - samples_g
    
    best_fom = 0
    best_short = 0
    best_long = 0
    
    fom_grid = []
    
    # Grid search
    for short_gate in range(short_gate_range[0], min(short_gate_range[1], n_samples), 2):
        for long_gate in range(max(short_gate+10, long_gate_range[0]), 
                              min(long_gate_range[1], n_samples), 5):
            
            # Calculate PSD with these gates
            Q_short_n = samples_n[:, :short_gate].sum(axis=1)
            Q_total_n = samples_n[:, :long_gate].sum(axis=1)
            Q_short_g = samples_g[:, :short_gate].sum(axis=1)
            Q_total_g = samples_g[:, :long_gate].sum(axis=1)
            
            # Avoid division by zero
            mask_n = Q_total_n > 0
            mask_g = Q_total_g > 0
            
            psd_n = (Q_total_n[mask_n] - Q_short_n[mask_n]) / Q_total_n[mask_n]
            psd_g = (Q_total_g[mask_g] - Q_short_g[mask_g]) / Q_total_g[mask_g]
            
            # Calculate FOM
            if len(psd_n) > 10 and len(psd_g) > 10:
                mean_n = psd_n.mean()
                mean_g = psd_g.mean()
                fwhm_n = 2.355 * psd_n.std()
                fwhm_g = 2.355 * psd_g.std()
                
                if fwhm_n + fwhm_g > 0:
                    fom = abs(mean_n - mean_g) / (fwhm_n + fwhm_g)
                    
                    fom_grid.append((short_gate, long_gate, fom))
                    
                    if fom > best_fom:
                        best_fom = fom
                        best_short = short_gate
                        best_long = long_gate
    
    optimal_params = {
        'short_gate_samples': best_short,
        'long_gate_samples': best_long,
        'figure_of_merit': best_fom,
        'fom_grid': fom_grid
    }
    
    print(f"Optimal PSD gates:")
    print(f"  Short gate: {best_short} samples")
    print(f"  Long gate: {best_long} samples")
    print(f"  FOM: {best_fom:.3f}")
    
    return optimal_params


def plot_fom_landscape(optimal_params):
    """
    Visualize FOM as function of gate timings
    """
    fom_grid = optimal_params['fom_grid']
    
    if not fom_grid:
        print("No FOM grid data available")
        return
    
    short_gates = np.array([x[0] for x in fom_grid])
    long_gates = np.array([x[1] for x in fom_grid])
    foms = np.array([x[2] for x in fom_grid])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(short_gates, long_gates, c=foms, s=50, 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Mark optimal point
    ax.plot(optimal_params['short_gate_samples'], 
           optimal_params['long_gate_samples'],
           'r*', markersize=20, label='Optimal')
    
    ax.set_xlabel('Short Gate (samples)', fontsize=14)
    ax.set_ylabel('Long Gate (samples)', fontsize=14)
    ax.set_title('PSD Figure of Merit Landscape', fontsize=16)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Figure of Merit', fontsize=12)
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def optimize_energy_dependent_boundary(df_calibration, energy_bins=10):
    """
    Find optimal PSD threshold as function of energy
    
    Parameters:
    -----------
    df_calibration : DataFrame
        Must have ENERGY_KEV, PSD, and PARTICLE columns
    energy_bins : int
        Number of energy bins for optimization
    
    Returns:
    --------
    boundary_func : function
        PSD_threshold(energy)
    """
    energy_col = 'ENERGY_KEV' if 'ENERGY_KEV' in df_calibration.columns else 'ENERGY'
    
    neutrons = df_calibration[df_calibration['PARTICLE'] == 'neutron']
    gammas = df_calibration[df_calibration['PARTICLE'] == 'gamma']
    
    # Bin by energy
    energy_edges = np.linspace(df_calibration[energy_col].min(),
                              df_calibration[energy_col].max(),
                              energy_bins + 1)
    
    bin_centers = []
    thresholds = []
    
    for i in range(len(energy_edges) - 1):
        e_min, e_max = energy_edges[i], energy_edges[i+1]
        e_center = (e_min + e_max) / 2
        
        n_in_bin = neutrons[(neutrons[energy_col] >= e_min) & 
                            (neutrons[energy_col] < e_max)]
        g_in_bin = gammas[(gammas[energy_col] >= e_min) & 
                         (gammas[energy_col] < e_max)]
        
        if len(n_in_bin) > 20 and len(g_in_bin) > 20:
            # Find threshold that maximizes separation
            psd_n = n_in_bin['PSD'].values
            psd_g = g_in_bin['PSD'].values
            
            # Try different thresholds and pick best
            test_thresholds = np.linspace(psd_g.max(), psd_n.min(), 50)
            best_threshold = test_thresholds[0]
            best_score = 0
            
            for thresh in test_thresholds:
                # Misclassification rates
                n_correct = (psd_n > thresh).sum() / len(psd_n)
                g_correct = (psd_g < thresh).sum() / len(psd_g)
                score = n_correct + g_correct
                
                if score > best_score:
                    best_score = score
                    best_threshold = thresh
            
            bin_centers.append(e_center)
            thresholds.append(best_threshold)
    
    # Fit polynomial through optimal thresholds
    if len(bin_centers) >= 3:
        params = np.polyfit(bin_centers, thresholds, 2)
        boundary_func = np.poly1d(params)
    else:
        # Fallback to constant
        boundary_func = lambda E: np.mean(thresholds)
    
    return boundary_func


# =============================================================================
# spectroscopy/spectrum.py
# =============================================================================

"""
Spectrum generation, manipulation, and analysis
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


class EnergySpectrum:
    """
    Class for gamma-ray energy spectrum operations
    """
    
    def __init__(self, energies=None, counts=None, energy_range=(0, 3000), bins=3000):
        """
        Initialize spectrum
        
        Parameters:
        -----------
        energies : array
            Event energies (if provided, will histogram)
        counts : array  
            Pre-binned counts (if provided, skip histogramming)
        energy_range : tuple
            (min, max) energy in keV
        bins : int
            Number of bins
        """
        self.energy_range = energy_range
        self.n_bins = bins
        
        if counts is not None:
            # Use provided histogram
            self.counts = np.array(counts)
            self.bin_edges = np.linspace(energy_range[0], energy_range[1], len(counts)+1)
        elif energies is not None:
            # Create histogram from events
            self.counts, self.bin_edges = np.histogram(energies, bins=bins, range=energy_range)
        else:
            # Empty spectrum
            self.counts = np.zeros(bins)
            self.bin_edges = np.linspace(energy_range[0], energy_range[1], bins+1)
        
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_width = self.bin_centers[1] - self.bin_centers[0]
    
    def smooth(self, sigma=2):
        """
        Smooth spectrum with Gaussian filter
        
        Parameters:
        -----------
        sigma : float
            Gaussian width in bins
        
        Returns:
        --------
        smoothed_spectrum : EnergySpectrum
        """
        smoothed_counts = gaussian_filter1d(self.counts.astype(float), sigma=sigma)
        
        spec = EnergySpectrum(counts=smoothed_counts, energy_range=self.energy_range)
        return spec
    
    def rebin(self, new_bins):
        """
        Rebin spectrum to different number of bins
        """
        new_counts, new_edges = np.histogram(self.bin_centers, bins=new_bins,
                                            range=self.energy_range, weights=self.counts)
        
        return EnergySpectrum(counts=new_counts, energy_range=self.energy_range)
    
    def subtract_background(self, background_spectrum, scale_factor=1.0):
        """
        Subtract background spectrum
        
        Parameters:
        -----------
        background_spectrum : EnergySpectrum
            Background to subtract
        scale_factor : float
            Scaling for different measurement times
        
        Returns:
        --------
        net_spectrum : EnergySpectrum
        """
        net_counts = self.counts - scale_factor * background_spectrum.counts
        net_counts[net_counts < 0] = 0  # No negative counts
        
        return EnergySpectrum(counts=net_counts, energy_range=self.energy_range)
    
    def get_roi_counts(self, energy_min, energy_max):
        """
        Get total counts in Region of Interest
        
        Returns:
        --------
        counts : float
            Total counts in ROI
        uncertainty : float
            Poisson uncertainty
        """
        mask = (self.bin_centers >= energy_min) & (self.bin_centers <= energy_max)
        counts = self.counts[mask].sum()
        uncertainty = np.sqrt(counts)
        
        return counts, uncertainty
    
    def plot(self, ax=None, **kwargs):
        """
        Plot spectrum
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.bin_centers, self.counts, **kwargs)
        ax.set_xlabel('Energy (keV)', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        return ax


def subtract_compton_continuum(spectrum, method='linear'):
    """
    Estimate and subtract Compton continuum background
    
    Parameters:
    -----------
    spectrum : EnergySpectrum
        Input spectrum
    method : str
        'linear' - linear interpolation between valleys
        'smoothing' - use heavily smoothed spectrum as background
    
    Returns:
    --------
    net_spectrum : EnergySpectrum
        Spectrum with continuum removed
    """
    if method == 'smoothing':
        # Smooth heavily to get continuum
        continuum = spectrum.smooth(sigma=20)
        net = spectrum.subtract_background(continuum, scale_factor=1.0)
        
    elif method == 'linear':
        # Find valleys (local minima) and interpolate
        from scipy.signal import find_peaks
        
        # Invert to find valleys
        valleys_idx, _ = find_peaks(-spectrum.counts, prominence=10, distance=50)
        
        if len(valleys_idx) > 2:
            # Interpolate background through valleys
            background_interp = np.interp(np.arange(len(spectrum.counts)),
                                         valleys_idx,
                                         spectrum.counts[valleys_idx])
            
            net_counts = spectrum.counts - background_interp
            net_counts[net_counts < 0] = 0
            
            net = EnergySpectrum(counts=net_counts, energy_range=spectrum.energy_range)
        else:
            print("Not enough valleys found for interpolation")
            net = spectrum
    
    return net