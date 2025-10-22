"""
Scintillator Characterization and Classification Module

Essential properties for PSD analysis:
1. Light yield (photons/keV, relative to anthracene)
2. Decay time constants (fast/slow components)
3. Light output non-linearity (Birks' law)
4. PSD capability (figure of merit)
5. Energy resolution
6. Timing resolution
7. Temperature dependence
8. Radiation hardness
9. H/C ratio (for organics)
10. Density and stopping power

This module helps:
- Characterize new scintillators
- Compare scintillator performance
- Optimize detector selection
- Predict PSD performance
- Apply material-specific corrections
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json


# =============================================================================
# SCINTILLATOR PROPERTIES DATABASE
# =============================================================================

@dataclass
class ScintillatorProperties:
    """
    Complete scintillator characterization
    """
    name: str
    type: str  # 'organic_liquid', 'organic_plastic', 'inorganic_crystal'
    
    # Light output
    light_yield: float  # photons/keV or % relative to anthracene
    light_yield_unit: str  # 'photons_per_keV' or 'percent_anthracene'
    
    # Timing properties
    decay_time_fast: float  # ns (primary/fast component)
    decay_time_slow: float  # ns (delayed/slow component)
    fast_fraction: float  # Fraction of light in fast component (0-1)
    rise_time: float  # ns (10-90%)
    
    # PSD capability
    psd_quality: str  # 'excellent', 'good', 'moderate', 'poor', 'none'
    fom_typical: float  # Typical FoM at 1 MeVee
    psd_mechanism: str  # 'singlet_triplet', 'exciton', 'self_trapped_exciton'
    
    # Energy resolution
    energy_resolution_662kev: float  # % FWHM at 662 keV (Cs-137)
    
    # Physical properties
    density: float  # g/cm³
    h_c_ratio: Optional[float]  # H/C atomic ratio (organics only)
    refractive_index: float
    
    # Non-linearity (Birks' law)
    birks_kB: Optional[float]  # cm/MeV (None if linear)
    
    # Temperature dependence
    temp_coefficient_light: float  # %/°C (light output)
    temp_coefficient_decay: float  # %/°C (decay time)
    
    # Additional info
    emission_max: float  # nm (peak emission wavelength)
    comments: str
    
    def __repr__(self):
        return f"Scintillator({self.name}, {self.type}, FoM={self.fom_typical:.2f})"


# Comprehensive scintillator database
SCINTILLATOR_DATABASE = {
    
    # =========================================================================
    # ORGANIC LIQUID SCINTILLATORS
    # =========================================================================
    
    'EJ-301': ScintillatorProperties(
        name='EJ-301 (NE-213)',
        type='organic_liquid',
        light_yield=78,
        light_yield_unit='percent_anthracene',
        decay_time_fast=3.2,
        decay_time_slow=32,
        fast_fraction=0.65,
        rise_time=0.9,
        psd_quality='excellent',
        fom_typical=1.8,
        psd_mechanism='singlet_triplet',
        energy_resolution_662kev=8.0,
        density=0.874,
        h_c_ratio=1.213,
        refractive_index=1.505,
        birks_kB=0.01,
        temp_coefficient_light=-0.2,
        temp_coefficient_decay=0.1,
        emission_max=425,
        comments='Gold standard for n/γ PSD, excellent FoM'
    ),
    
    'EJ-309': ScintillatorProperties(
        name='EJ-309',
        type='organic_liquid',
        light_yield=80,
        light_yield_unit='percent_anthracene',
        decay_time_fast=3.5,
        decay_time_slow=33,
        fast_fraction=0.68,
        rise_time=1.0,
        psd_quality='excellent',
        fom_typical=1.7,
        psd_mechanism='singlet_triplet',
        energy_resolution_662kev=7.5,
        density=0.959,
        h_c_ratio=1.25,
        refractive_index=1.57,
        birks_kB=0.009,
        temp_coefficient_light=-0.18,
        temp_coefficient_decay=0.12,
        emission_max=424,
        comments='Non-hazardous alternative to EJ-301'
    ),
    
    # =========================================================================
    # ORGANIC PLASTIC SCINTILLATORS
    # =========================================================================
    
    'EJ-299-33': ScintillatorProperties(
        name='EJ-299-33',
        type='organic_plastic',
        light_yield=55,
        light_yield_unit='percent_anthracene',
        decay_time_fast=3.3,
        decay_time_slow=35,
        fast_fraction=0.58,
        rise_time=1.5,
        psd_quality='good',
        fom_typical=1.3,
        psd_mechanism='singlet_triplet',
        energy_resolution_662kev=12.0,
        density=1.023,
        h_c_ratio=1.104,
        refractive_index=1.58,
        birks_kB=0.012,
        temp_coefficient_light=-0.25,
        temp_coefficient_decay=0.15,
        emission_max=425,
        comments='PSD-capable plastic, rugged'
    ),
    
    'EJ-200': ScintillatorProperties(
        name='EJ-200 (BC-408)',
        type='organic_plastic',
        light_yield=64,
        light_yield_unit='percent_anthracene',
        decay_time_fast=2.1,
        decay_time_slow=0,  # Effectively single component
        fast_fraction=1.0,
        rise_time=0.9,
        psd_quality='none',
        fom_typical=0.0,
        psd_mechanism='none',
        energy_resolution_662kev=15.0,
        density=1.023,
        h_c_ratio=1.104,
        refractive_index=1.58,
        birks_kB=0.0126,
        temp_coefficient_light=-0.3,
        temp_coefficient_decay=0.0,
        emission_max=425,
        comments='Fast timing, no PSD capability'
    ),
    
    # =========================================================================
    # INORGANIC CRYSTALS
    # =========================================================================
    
    'NaI(Tl)': ScintillatorProperties(
        name='NaI(Tl)',
        type='inorganic_crystal',
        light_yield=38000,
        light_yield_unit='photons_per_keV',
        decay_time_fast=230,
        decay_time_slow=230,  # Single exponential
        fast_fraction=1.0,
        rise_time=20,
        psd_quality='poor',
        fom_typical=0.2,
        psd_mechanism='self_trapped_exciton',
        energy_resolution_662kev=6.5,
        density=3.67,
        h_c_ratio=None,
        refractive_index=1.85,
        birks_kB=None,  # Minimal non-linearity
        temp_coefficient_light=-0.2,
        temp_coefficient_decay=0.0,
        emission_max=415,
        comments='Excellent gamma spectroscopy, poor PSD'
    ),
    
    'CLYC': ScintillatorProperties(
        name='CLYC (Cs2LiYCl6:Ce)',
        type='inorganic_crystal',
        light_yield=20000,
        light_yield_unit='photons_per_keV',
        decay_time_fast=50,
        decay_time_slow=1000,
        fast_fraction=0.7,
        rise_time=10,
        psd_quality='excellent',
        fom_typical=2.5,
        psd_mechanism='core_valence',
        energy_resolution_662kev=4.5,
        density=3.31,
        h_c_ratio=None,
        refractive_index=1.81,
        birks_kB=None,
        temp_coefficient_light=-0.1,
        temp_coefficient_decay=0.05,
        emission_max=370,
        comments='Excellent n/γ PSD + gamma spectroscopy'
    ),
    
    'Stilbene': ScintillatorProperties(
        name='Stilbene',
        type='organic_crystal',
        light_yield=60,
        light_yield_unit='percent_anthracene',
        decay_time_fast=4.5,
        decay_time_slow=45,
        fast_fraction=0.60,
        rise_time=1.2,
        psd_quality='excellent',
        fom_typical=2.2,
        psd_mechanism='singlet_triplet',
        energy_resolution_662kev=9.0,
        density=1.16,
        h_c_ratio=0.857,  # C14H12
        refractive_index=1.626,
        birks_kB=0.008,
        temp_coefficient_light=-0.15,
        temp_coefficient_decay=0.08,
        emission_max=410,
        comments='Best PSD crystal, difficult to grow large'
    ),
}


# =============================================================================
# LIGHT OUTPUT NON-LINEARITY (BIRKS' LAW)
# =============================================================================

def birks_law(energy_keV, kB, particle='electron'):
    """
    Calculate light output with Birks' quenching
    
    L/E = S / (1 + kB * dE/dx)
    
    Parameters:
    -----------
    energy_keV : float or array
        Particle energy in keV
    kB : float
        Birks' constant (cm/MeV)
    particle : str
        'electron', 'proton', 'alpha'
    
    Returns:
    --------
    relative_light_output : float or array
        Light output relative to electron at 1 MeV (0-1)
    """
    # Stopping power (MeV/cm) - approximate
    if particle == 'electron':
        dedx = 2.0  # MeV/cm (roughly constant for MeV electrons)
    elif particle == 'proton':
        # Bragg curve approximation
        dedx = 50 * (1.0 / np.sqrt(energy_keV/1000))  # Higher at low energy
    elif particle == 'alpha':
        dedx = 200 * (1.0 / np.sqrt(energy_keV/1000))
    else:
        dedx = 2.0
    
    # Birks' formula
    L_over_E = 1.0 / (1.0 + kB * dedx)
    
    # Normalize to electron at 1 MeV
    L_over_E_ref = 1.0 / (1.0 + kB * 2.0)
    
    return L_over_E / L_over_E_ref


def apply_light_output_correction(energy_dep_keV, particle_type, scintillator_name):
    """
    Convert deposited energy to light output (electron-equivalent energy)
    
    Parameters:
    -----------
    energy_dep_keV : float
        Deposited energy
    particle_type : str
        'gamma', 'neutron', 'proton', 'alpha'
    scintillator_name : str
        Scintillator from database
    
    Returns:
    --------
    energy_ee_keV : float
        Electron-equivalent energy (light output)
    """
    scint = SCINTILLATOR_DATABASE.get(scintillator_name)
    
    if scint is None or scint.birks_kB is None:
        # No correction needed (linear response)
        return energy_dep_keV
    
    kB = scint.birks_kB
    
    # For gamma: minimal quenching (Compton electrons)
    if particle_type == 'gamma':
        particle = 'electron'
        
    # For neutron: depends on recoil proton energy
    elif particle_type == 'neutron':
        particle = 'proton'
    
    else:
        particle = particle_type
    
    # Apply Birks' law
    quenching_factor = birks_law(energy_dep_keV, kB, particle)
    energy_ee_keV = energy_dep_keV * quenching_factor
    
    return energy_ee_keV


# =============================================================================
# SCINTILLATOR CHARACTERIZATION WORKFLOW
# =============================================================================

def characterize_scintillator_from_data(df, scintillator_name='Unknown'):
    """
    Extract scintillator properties from measurement data
    
    Parameters:
    -----------
    df : DataFrame
        Must have calibrated waveforms from known sources
    scintillator_name : str
        Name for this characterization
    
    Returns:
    --------
    properties : ScintillatorProperties
        Measured properties
    """
    print(f"Characterizing scintillator: {scintillator_name}")
    print("="*70)
    
    sample_cols = [col for col in df.columns if col.startswith('SAMPLE')]
    
    if not sample_cols:
        raise ValueError("Need waveform data for characterization")
    
    # =========================================================================
    # 1. DECAY TIME CONSTANTS
    # =========================================================================
    print("\n1. Measuring decay time constants...")
    
    decay_times = []
    fast_fractions = []
    
    for idx, row in df.sample(min(100, len(df))).iterrows():
        waveform = row[sample_cols].values
        baseline = np.mean(waveform[:50])
        pulse = baseline - waveform
        
        if np.max(pulse) < 100:
            continue
        
        peak_idx = np.argmax(pulse)
        
        if peak_idx < len(pulse) - 100:
            tail = pulse[peak_idx:peak_idx+150]
            x = np.arange(len(tail)) * 4  # ns (assuming 250 MHz)
            
            # Bi-exponential fit
            def biexp(t, A_fast, tau_fast, A_slow, tau_slow):
                return A_fast * np.exp(-t/tau_fast) + A_slow * np.exp(-t/tau_slow)
            
            try:
                p0 = [tail[0]*0.7, 5, tail[0]*0.3, 40]
                popt, _ = curve_fit(biexp, x, tail, p0=p0, maxfev=5000,
                                   bounds=([0,1,0,10], [np.inf,50,np.inf,200]))
                
                A_fast, tau_fast, A_slow, tau_slow = popt
                
                decay_times.append((tau_fast, tau_slow))
                fast_frac = A_fast / (A_fast + A_slow)
                fast_fractions.append(fast_frac)
                
            except:
                continue
    
    if decay_times:
        tau_fast_mean = np.mean([d[0] for d in decay_times])
        tau_slow_mean = np.mean([d[1] for d in decay_times])
        fast_frac_mean = np.mean(fast_fractions)
        
        print(f"  Fast decay time: {tau_fast_mean:.2f} ns")
        print(f"  Slow decay time: {tau_slow_mean:.2f} ns")
        print(f"  Fast fraction: {fast_frac_mean:.3f}")
    else:
        tau_fast_mean, tau_slow_mean, fast_frac_mean = 5, 50, 0.5
        print("  Could not fit decay times")
    
    # =========================================================================
    # 2. RISE TIME
    # =========================================================================
    print("\n2. Measuring rise time...")
    
    rise_times = []
    
    for idx, row in df.sample(min(100, len(df))).iterrows():
        waveform = row[sample_cols].values
        baseline = np.mean(waveform[:50])
        pulse = baseline - waveform
        
        if np.max(pulse) < 100:
            continue
        
        pulse_norm = pulse / np.max(pulse)
        peak_idx = np.argmax(pulse_norm)
        
        idx_10 = np.where(pulse_norm[:peak_idx] >= 0.1)[0]
        idx_90 = np.where(pulse_norm[:peak_idx] >= 0.9)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = (idx_90[0] - idx_10[0]) * 4  # ns
            rise_times.append(rise_time)
    
    if rise_times:
        rise_time_mean = np.mean(rise_times)
        print(f"  Rise time (10-90%): {rise_time_mean:.2f} ns")
    else:
        rise_time_mean = 2.0
        print("  Could not measure rise time")
    
    # =========================================================================
    # 3. PSD FIGURE OF MERIT
    # =========================================================================
    print("\n3. Measuring PSD capability...")
    
    if 'PARTICLE' in df.columns and 'PSD' in df.columns:
        neutrons = df[df['PARTICLE'] == 'neutron']['PSD']
        gammas = df[df['PARTICLE'] == 'gamma']['PSD']
        
        if len(neutrons) > 20 and len(gammas) > 20:
            mean_n = neutrons.mean()
            mean_g = gammas.mean()
            fwhm_n = 2.355 * neutrons.std()
            fwhm_g = 2.355 * gammas.std()
            
            fom = abs(mean_n - mean_g) / (fwhm_n + fwhm_g)
            
            if fom > 1.5:
                psd_quality = 'excellent'
            elif fom > 1.0:
                psd_quality = 'good'
            elif fom > 0.5:
                psd_quality = 'moderate'
            else:
                psd_quality = 'poor'
            
            print(f"  Figure of Merit: {fom:.3f}")
            print(f"  PSD quality: {psd_quality}")
        else:
            fom = 0
            psd_quality = 'unknown'
            print("  Insufficient n/γ data for FoM")
    else:
        fom = 0
        psd_quality = 'unknown'
        print("  No PSD data available")
    
    # =========================================================================
    # 4. ENERGY RESOLUTION
    # =========================================================================
    print("\n4. Measuring energy resolution...")
    
    if 'ENERGY_KEV' in df.columns:
        # Find 662 keV peak (Cs-137)
        hist, bins = np.histogram(df['ENERGY_KEV'], bins=1000, range=(0, 3000))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Look for peak near 662 keV
        peak_region = (bin_centers > 600) & (bin_centers < 750)
        if peak_region.sum() > 0:
            peak_idx = peak_region.nonzero()[0][np.argmax(hist[peak_region])]
            peak_energy = bin_centers[peak_idx]
            
            # Fit Gaussian to peak
            fit_region = (bin_centers > peak_energy-50) & (bin_centers < peak_energy+50)
            x_fit = bin_centers[fit_region]
            y_fit = hist[fit_region]
            
            try:
                def gaussian(x, amp, mu, sigma):
                    return amp * np.exp(-0.5*((x-mu)/sigma)**2)
                
                popt, _ = curve_fit(gaussian, x_fit, y_fit, 
                                   p0=[y_fit.max(), peak_energy, 20])
                
                fwhm = 2.355 * abs(popt[2])
                resolution = (fwhm / peak_energy) * 100
                
                print(f"  Energy resolution @ 662 keV: {resolution:.2f}%")
            except:
                resolution = 10.0
                print("  Could not fit resolution")
        else:
            resolution = 10.0
            print("  No 662 keV peak found")
    else:
        resolution = 10.0
        print("  No energy calibration available")
    
    # =========================================================================
    # CREATE PROPERTIES OBJECT
    # =========================================================================
    
    properties = ScintillatorProperties(
        name=scintillator_name,
        type='organic_liquid',  # Assume - update manually
        light_yield=60,  # Estimate - measure with calibrated source
        light_yield_unit='percent_anthracene',
        decay_time_fast=tau_fast_mean,
        decay_time_slow=tau_slow_mean,
        fast_fraction=fast_frac_mean,
        rise_time=rise_time_mean,
        psd_quality=psd_quality,
        fom_typical=fom,
        psd_mechanism='singlet_triplet',  # Assume organic
        energy_resolution_662kev=resolution,
        density=1.0,  # Update with actual value
        h_c_ratio=None,
        refractive_index=1.5,  # Typical
        birks_kB=0.01,  # Typical for organics
        temp_coefficient_light=-0.2,
        temp_coefficient_decay=0.1,
        emission_max=420,
        comments='Characterized from measurement data'
    )
    
    print("\n" + "="*70)
    print("Characterization complete!")
    
    return properties


# =============================================================================
# COMPARISON AND SELECTION
# =============================================================================

def compare_scintillators(scintillators: List[str], criterion='psd'):
    """
    Compare multiple scintillators for selection
    
    Parameters:
    -----------
    scintillators : list
        List of scintillator names
    criterion : str
        'psd', 'light_yield', 'timing', 'energy_resolution'
    
    Returns:
    --------
    comparison : DataFrame
        Comparison table
    """
    import pandas as pd
    
    data = []
    
    for name in scintillators:
        scint = SCINTILLATOR_DATABASE.get(name)
        if scint is None:
            continue
        
        data.append({
            'Name': scint.name,
            'Type': scint.type,
            'Light Yield': f"{scint.light_yield} {scint.light_yield_unit[:10]}",
            'Fast Decay (ns)': scint.decay_time_fast,
            'Slow Decay (ns)': scint.decay_time_slow,
            'Fast Fraction': scint.fast_fraction,
            'PSD Quality': scint.psd_quality,
            'FoM': scint.fom_typical,
            'Energy Res (%)': scint.energy_resolution_662kev,
            'Density (g/cm³)': scint.density
        })
    
    df = pd.DataFrame(data)
    
    # Sort by criterion
    if criterion == 'psd':
        df = df.sort_values('FoM', ascending=False)
    elif criterion == 'light_yield':
        # Convert to numeric for sorting (rough approximation)
        df = df.sort_values('Fast Decay (ns)')  # Proxy
    elif criterion == 'timing':
        df = df.sort_values('Fast Decay (ns)')
    elif criterion == 'energy_resolution':
        df = df.sort_values('Energy Res (%)')
    
    return df


def plot_scintillator_comparison(scintillators: List[str]):
    """
    Visual comparison of scintillator properties
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = []
    foms = []
    light_yields = []
    energy_res = []
    decay_fast = []
    
    for name in scintillators:
        scint = SCINTILLATOR_DATABASE.get(name)
        if scint:
            names.append(scint.name[:15])  # Truncate for plot
            foms.append(scint.fom_typical)
            light_yields.append(scint.light_yield)
            energy_res.append(scint.energy_resolution_662kev)
            decay_fast.append(scint.decay_time_fast)
    
    x = np.arange(len(names))
    
    # FoM comparison
    ax = axes[0, 0]
    bars = ax.bar(x, foms, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('PSD Figure of Merit', fontsize=12, fontweight='bold')
    ax.set_title('PSD Capability', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.axhline(1.0, color='green', linestyle='--', label='Good (FoM=1.0)')
    ax.axhline(1.5, color='orange', linestyle='--', label='Excellent (FoM=1.5)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Energy resolution
    ax = axes[0, 1]
    ax.bar(x, energy_res, color='coral', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Energy Resolution (%) @ 662 keV', fontsize=12, fontweight='bold')
    ax.set_title('Energy Resolution (lower is better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Decay times
    ax = axes[1, 0]
    ax.bar(x, decay_fast, color='mediumseagreen', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Fast Decay Time (ns)', fontsize=12, fontweight='bold')
    ax.set_title('Timing Performance (lower is better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Light yield (rough comparison)
    ax = axes[1, 1]
    ax.bar(x, light_yields, color='gold', edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Relative Light Yield', fontsize=12, fontweight='bold')
    ax.set_title('Light Output (higher is better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    print("Scintillator Characterization Module")
    print("="*70)
    
    # List available scintillators
    print("\nAvailable scintillators in database:")
    for name in SCINTILLATOR_DATABASE.keys():
        scint = SCINTILLATOR_DATABASE[name]
        print(f"  {name}: {scint.psd_quality} PSD (FoM={scint.fom_typical:.2f})")
    
    # Compare common PSD scintillators
    print("\n" + "="*70)
    print("Comparison of PSD-capable scintillators:")
    print("="*70)
    
    comparison = compare_scintillators(
        ['EJ-301', 'EJ-309', 'EJ-299-33', 'CLYC', 'Stilbene'],
        criterion='psd'
    )
    print(comparison.to_string(index=False))
    
    # Plot comparison
    fig = plot_scintillator_comparison(['EJ-301', 'EJ-309', 'Stilbene', 'CLYC', 'EJ-299-33'])
    plt.savefig('scintillator_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved")
    
    print("\n" + "="*70)
    print("Scintillator characterization tools ready!")
    print("\nUse characterize_scintillator_from_data() to measure properties")
    print("Use apply_light_output_correction() for Birks' law corrections")