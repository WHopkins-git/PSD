# %% [markdown]
# # PSD Analysis Notebook
# ## Neutron/Gamma Discrimination and NORM Source Characterization
# 
# This notebook provides an interactive workflow for:
# - Calibration with known sources
# - Unknown NORM source identification
# - Detector performance characterization

# %% Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize
import seaborn as sns

# Import your analysis functions
# from psd_analysis import *

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
# %matplotlib inline  # Uncomment in Jupyter notebook

# %% [markdown]
# ## 1. Load and Inspect Data

# %% Load data
filename = 'your_data_file.csv'  # Update this
df = load_psd_data(filename, delimiter=';')

# Display basic info
print(f"Total events: {len(df)}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
df.head()

# %% Quick statistics
print("Energy statistics:")
print(df['ENERGY'].describe())
print("\nEnergyShort statistics:")
print(df['ENERGYSHORT'].describe())

# %% [markdown]
# ## 2. Quality Control

# %% Validate events
valid_mask, qc_report = validate_events(df, adc_min=0, adc_max=16383)

print(f"\nQuality Control Summary:")
print(f"  Total events: {qc_report['total_events']}")
print(f"  Valid events: {qc_report['valid_events']}")
print(f"  Rejection rate: {qc_report['rejection_rate']*100:.2f}%")

# Apply filter
df_clean = df[valid_mask].copy()
print(f"\nCleaned dataset: {len(df_clean)} events")

# %% [markdown]
# ## 3. Calculate PSD Parameters

# %% Calculate PSD ratio
df_clean = calculate_psd_ratio(df_clean)

# Plot PSD distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_clean['PSD'].dropna(), bins=100, range=(0, 1), 
        alpha=0.7, edgecolor='black')
ax.set_xlabel('PSD Parameter', fontsize=14)
ax.set_ylabel('Counts', fontsize=14)
ax.set_title('PSD Distribution', fontsize=16)
ax.set_yscale('log')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Initial PSD Scatter Plot (Uncalibrated)

# %% PSD scatter plot
fig, ax = plot_psd_scatter(df_clean, psd_range=(0, 1), bins=150)
plt.show()

# Look for two distinct bands:
# - Lower PSD band = gammas
# - Upper PSD band = neutrons (if present)

# %% [markdown]
# ## 5. Energy Spectrum Analysis

# %% Raw energy spectrum
fig, ax = plot_energy_spectra(df_clean, energy_col='ENERGY', bins=1000)
plt.show()

# %% Zoomed view of interesting region
energy_min = 500  # Adjust based on your data
energy_max = 2000

fig, ax = plot_energy_spectra(df_clean, energy_col='ENERGY', 
                              energy_range=(energy_min, energy_max),
                              bins=500)
plt.show()

# %% [markdown]
# ## 6. Energy Calibration (for known sources)
# 
# ### Option A: Compton Edge Method (for gamma sources)

# %% Find Compton edge
# Create histogram
hist, bins = np.histogram(df_clean['ENERGY'], bins=2000, 
                         range=(df_clean['ENERGY'].min(), 
                                df_clean['ENERGY'].max()))
bin_centers = (bins[:-1] + bins[1:]) / 2

# For Cs-137: Compton edge at 477 keV
# For Co-60: Compton edges at ~963 keV and ~1118 keV

# Plot to identify edge visually
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(bin_centers, hist)
ax.set_xlabel('ADC Channel', fontsize=14)
ax.set_ylabel('Counts', fontsize=14)
ax.set_title('Energy Spectrum - Find Compton Edge', fontsize=16)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.show()

# Manually identify the edge position from plot
compton_edge_adc = 1200  # UPDATE THIS based on visual inspection
compton_edge_keV = 477   # For Cs-137

# %% Option B: Photopeak Method (if full-energy peaks visible)
# Use find_peaks_in_spectrum to locate photopeaks

peaks_adc, peaks_counts, _ = find_peaks_in_spectrum(bin_centers, hist,
                                                     prominence=100, 
                                                     distance=50)

print(f"Found {len(peaks_adc)} peaks:")
for i, (adc, counts) in enumerate(zip(peaks_adc, peaks_counts)):
    print(f"  Peak {i+1}: ADC = {adc:.1f}, Counts = {counts:.0f}")

# Assign known energies to peaks
# For Cs-137: 662 keV photopeak
# For Co-60: 1173 and 1332 keV
# For AmBe: neutrons + 4.44 MeV gamma

# Example for Cs-137
calibration_points = [
    (compton_edge_adc, 477),    # Compton edge
    (peaks_adc[0], 662)          # Photopeak (adjust index as needed)
]

# %% Perform calibration
df_clean, cal_func, cal_params = calibrate_energy(df_clean, 
                                                   calibration_points,
                                                   method='linear')

# Plot calibration curve
fig = plot_calibration_curve(calibration_points, cal_func)
plt.show()

# %% [markdown]
# ## 7. Calibrated Spectra

# %% Calibrated energy spectrum
fig, ax = plot_energy_spectra(df_clean, energy_col='ENERGY_KEV',
                              energy_range=(0, 3000), bins=1000)
plt.show()

# %% Calibrated PSD scatter
fig, ax = plot_psd_scatter(df_clean, energy_range=(0, 3000), 
                          psd_range=(0, 1), bins=150)
plt.show()

# %% [markdown]
# ## 8. PSD Discrimination (if neutron/gamma mixture)

# %% Interactive boundary definition
# METHOD 1: Manual graphical cut
print("Observe the PSD scatter plot above.")
print("Estimate the boundary between neutron and gamma bands.")
print("Update the parameters below:")

# Simple linear boundary: PSD_cut = slope * E + intercept
# Adjust these based on your data
slope = 0.0  # Change if boundary is energy-dependent
intercept = 0.3  # Typical separation point

boundary_func = lambda E: slope * E + intercept

# %% Apply discrimination
df_clean = apply_discrimination(df_clean, boundary_func)

# %% Plot discriminated PSD scatter
fig, ax = plot_psd_scatter(df_clean, energy_range=(0, 3000),
                          boundary_func=boundary_func, bins=150)
plt.show()

# %% Separated energy spectra
fig, ax = plot_energy_spectra(df_clean, energy_col='ENERGY_KEV',
                              energy_range=(0, 3000), bins=1000,
                              separate_particles=True)
plt.show()

# %% Calculate discrimination performance
neutrons = df_clean[df_clean['PARTICLE'] == 'neutron']
gammas = df_clean[df_clean['PARTICLE'] == 'gamma']

print(f"\nParticle counts:")
print(f"  Neutrons: {len(neutrons)}")
print(f"  Gammas: {len(gammas)}")
print(f"  n/γ ratio: {len(neutrons)/len(gammas):.3f}")

# If you know the true particle types (from pure source), calculate FOM
# fom = calculate_figure_of_merit(neutrons['PSD'], gammas['PSD'])

# %% [markdown]
# ## 9. NORM Source Identification

# %% Find gamma peaks
# Use gamma-only spectrum
gamma_spectrum = gammas if 'PARTICLE' in df_clean.columns else df_clean
hist_gamma, bins_gamma = np.histogram(gamma_spectrum['ENERGY_KEV'],
                                      bins=3000, range=(0, 3000))
bin_centers_gamma = (bins_gamma[:-1] + bins_gamma[1:]) / 2

peak_energies, peak_counts, _ = find_peaks_in_spectrum(bin_centers_gamma,
                                                        hist_gamma,
                                                        prominence=20,
                                                        distance=10)

print(f"\nFound {len(peak_energies)} peaks in gamma spectrum")

# %% Match peaks to isotope library
matches = match_peaks_to_library(peak_energies, tolerance_keV=10)

# %% Identify decay series
identified_series = identify_decay_chains(matches)

# %% Visualize identified peaks
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(bin_centers_gamma, hist_gamma, 'b-', alpha=0.7, linewidth=1)

# Mark identified peaks
for match in matches:
    e = match['measured_energy']
    ax.axvline(e, color='red', linestyle='--', alpha=0.5)
    ax.text(e, hist_gamma.max() * 0.9, 
            f"{match['nuclide']}\n{e:.0f} keV",
            rotation=90, fontsize=8, va='top')

ax.set_xlabel('Energy (keV)', fontsize=14)
ax.set_ylabel('Counts', fontsize=14)
ax.set_title('Identified Gamma Peaks - NORM Analysis', fontsize=16)
ax.set_yscale('log')
ax.set_xlim(0, 3000)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Activity Calculations (Advanced)
# 
# To calculate activity, you need:
# - Peak net counts (background subtracted)
# - Detection efficiency at that energy
# - Gamma branching ratio
# - Measurement live time
# 
# Activity (Bq) = Net_Counts / (Efficiency × Branching_Ratio × Live_Time)

# %% Example activity calculation
# This is a template - adjust for your specific setup

def calculate_activity(net_counts, efficiency, branching_ratio, live_time_sec):
    """
    Calculate activity in Bq
    
    Parameters:
    -----------
    net_counts : float
        Background-subtracted peak counts
    efficiency : float
        Absolute detection efficiency (0-1)
    branching_ratio : float
        Gamma emission probability (0-1)
    live_time_sec : float
        Measurement live time in seconds
    
    Returns:
    --------
    activity_bq : float
    uncertainty_bq : float (Poisson)
    """
    activity_bq = net_counts / (efficiency * branching_ratio * live_time_sec)
    
    # Poisson uncertainty
    uncertainty_bq = np.sqrt(net_counts) / (efficiency * branching_ratio * live_time_sec)
    
    return activity_bq, uncertainty_bq


# Example for Cs-137 (662 keV, branching ratio = 0.851)
# Adjust these values for your measurement
live_time_sec = 3600  # 1 hour measurement
net_counts_662 = 50000  # Get from peak fitting
efficiency_662 = 0.01  # Depends on your detector geometry
branching_ratio_662 = 0.851

activity, uncertainty = calculate_activity(net_counts_662, efficiency_662,
                                          branching_ratio_662, live_time_sec)

print(f"\nCs-137 Activity: {activity:.1f} ± {uncertainty:.1f} Bq")
print(f"Activity: {activity/1000:.2f} ± {uncertainty/1000:.2f} kBq")

# %% [markdown]
# ## 11. Export Results

# %% Save processed data
df_clean.to_csv('processed_data.csv', index=False)
print("Saved: processed_data.csv")

# %% Generate summary report
report = f"""
PSD ANALYSIS REPORT
{'='*60}

File: {filename}
Total events: {len(df)}
Valid events: {len(df_clean)}

Energy Calibration:
  Method: {'Linear' if len(cal_params) == 2 else 'Polynomial'}
  Parameters: {cal_params}

"""

if 'PARTICLE' in df_clean.columns:
    report += f"""
Particle Discrimination:
  Neutrons: {len(neutrons)} ({len(neutrons)/len(df_clean)*100:.1f}%)
  Gammas: {len(gammas)} ({len(gammas)/len(df_clean)*100:.1f}%)
  n/γ ratio: {len(neutrons)/len(gammas):.3f}

"""

if identified_series:
    report += "Identified Isotopes:\n"
    for series, info in identified_series.items():
        report += f"  {series}: {info['peaks_matched']}/{info['peaks_expected']} peaks\n"

report += f"\n{'='*60}\n"

print(report)

# Save report
with open('analysis_report.txt', 'w') as f:
    f.write(report)

print("Saved: analysis_report.txt")

# %% [markdown]
# ## Summary
# 
# This notebook has guided you through:
# 1. ✅ Loading and validating PSD data
# 2. ✅ Calculating PSD discrimination parameters
# 3. ✅ Energy calibration with known sources
# 4. ✅ Neutron/gamma separation
# 5. ✅ Peak finding and isotope identification
# 6. ✅ NORM source characterization
# 
# ### Next Steps:
# - Fine-tune PSD gate timing for optimal FOM
# - Implement energy-dependent discrimination boundary
# - Build detector efficiency curve
# - Perform time-series analysis for decay verification
# - Compare with baseline/background measurements
# - Calculate dose rates from identified isotopes

# %% [markdown]
# ## APPENDIX A: Advanced PSD Gate Optimization

# %% Gate optimization for calibration sources
# If you have pure neutron and gamma sources, optimize gate timing

def optimize_psd_gates(df_neutrons, df_gammas, short_gate_range=(10, 50)):
    """
    Scan short gate widths to find optimal FOM
    
    Returns best short gate width
    """
    sample_cols = [col for col in df_neutrons.columns if col.startswith('SAMPLE')]
    
    if not sample_cols:
        print("No waveform data available for optimization")
        return None
    
    fom_values = []
    gate_widths = range(short_gate_range[0], short_gate_range[1], 2)
    
    for gate_width in gate_widths:
        # Recalculate ENERGYSHORT with new gate
        # (Simplified - assumes samples are baseline-subtracted)
        samples_n = df_neutrons[sample_cols[:gate_width]].values.sum(axis=1)
        samples_g = df_gammas[sample_cols[:gate_width]].values.sum(axis=1)
        
        # Calculate PSD
        psd_n = (df_neutrons['ENERGY'] - samples_n) / df_neutrons['ENERGY']
        psd_g = (df_gammas['ENERGY'] - samples_g) / df_gammas['ENERGY']
        
        # Calculate FOM
        fom = calculate_figure_of_merit(psd_n, psd_g)
        fom_values.append(fom)
    
    # Plot FOM vs gate width
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gate_widths, fom_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Short Gate Width (samples)', fontsize=14)
    ax.set_ylabel('Figure of Merit', fontsize=14)
    ax.set_title('PSD Gate Optimization', fontsize=16)
    ax.grid(True, alpha=0.3)
    
    best_gate = gate_widths[np.argmax(fom_values)]
    best_fom = max(fom_values)
    ax.axvline(best_gate, color='red', linestyle='--', 
               label=f'Optimal: {best_gate} samples (FOM={best_fom:.2f})')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return best_gate

# %% Run optimization (if you have separated sources)
# best_gate = optimize_psd_gates(df_neutron_source, df_gamma_source)

# %% [markdown]
# ## APPENDIX B: Energy Resolution Analysis

# %% Calculate energy resolution vs energy
def measure_resolution_curve(df, peak_energies_keV, fit_width=50):
    """
    Measure energy resolution at multiple energies
    
    Returns arrays of energies and resolutions (%)
    """
    resolutions = []
    energies = []
    
    hist, bins = np.histogram(df['ENERGY_KEV'], bins=3000, range=(0, 3000))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    for peak_e in peak_energies_keV:
        fit_result = fit_gaussian_peak(bin_centers, hist, peak_e, fit_width)
        
        if fit_result is not None:
            energies.append(fit_result['centroid'])
            resolutions.append(fit_result['resolution_percent'])
    
    return np.array(energies), np.array(resolutions)


# %% Plot resolution curve
# peak_list = [662]  # Add your calibration peak energies
# energies, resolutions = measure_resolution_curve(df_clean, peak_list)

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(energies, resolutions, 'ro-', markersize=10, linewidth=2)
# ax.set_xlabel('Energy (keV)', fontsize=14)
# ax.set_ylabel('Energy Resolution (%)', fontsize=14)
# ax.set_title('Detector Energy Resolution', fontsize=16)
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ## APPENDIX C: Time-Series Analysis

# %% Count rate vs time
def plot_count_rate_vs_time(df, time_bin_seconds=60):
    """
    Plot count rate evolution over measurement
    """
    # Convert timetag to seconds (depends on your digitizer units)
    # Common: timetag in units of 1/sampling_rate
    # Adjust conversion factor as needed
    time_conversion = 1e-6  # Example: microseconds
    
    df['TIME_SEC'] = df['TIMETAG'] * time_conversion
    
    # Bin by time
    time_bins = np.arange(df['TIME_SEC'].min(), 
                         df['TIME_SEC'].max(), 
                         time_bin_seconds)
    
    counts, bin_edges = np.histogram(df['TIME_SEC'], bins=time_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Count rate (Hz)
    count_rate = counts / time_bin_seconds
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_centers / 3600, count_rate, 'b-', linewidth=1)  # Convert to hours
    ax.set_xlabel('Time (hours)', fontsize=14)
    ax.set_ylabel('Count Rate (Hz)', fontsize=14)
    ax.set_title('Count Rate vs Time', fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"Mean count rate: {count_rate.mean():.2f} ± {count_rate.std():.2f} Hz")
    print(f"Total measurement time: {(df['TIME_SEC'].max() - df['TIME_SEC'].min())/3600:.2f} hours")

# %% Run time analysis
# plot_count_rate_vs_time(df_clean, time_bin_seconds=300)  # 5-minute bins

# %% [markdown]
# ## APPENDIX D: Background Subtraction

# %% Background spectrum analysis
def subtract_background(signal_spectrum, background_spectrum, 
                       signal_livetime, background_livetime):
    """
    Subtract background with proper normalization
    
    Parameters:
    -----------
    signal_spectrum : array
        Source + background counts
    background_spectrum : array
        Background-only counts
    signal_livetime : float
        Live time for signal measurement (seconds)
    background_livetime : float
        Live time for background measurement (seconds)
    
    Returns:
    --------
    net_spectrum : array
        Background-subtracted spectrum
    uncertainty : array
        Propagated uncertainty
    """
    # Normalize to same live time
    bg_normalized = background_spectrum * (signal_livetime / background_livetime)
    
    # Subtract
    net_spectrum = signal_spectrum - bg_normalized
    
    # Propagated uncertainty (Poisson)
    uncertainty = np.sqrt(signal_spectrum + bg_normalized * (signal_livetime / background_livetime)**2)
    
    return net_spectrum, uncertainty


# %% Example background subtraction
# Load background measurement
# df_background = load_psd_data('background_measurement.csv')
# df_background = calculate_psd_ratio(df_background)

# Create histograms
# signal_hist, bins = np.histogram(df_clean['ENERGY_KEV'], bins=3000, range=(0, 3000))
# bg_hist, _ = np.histogram(df_background['ENERGY_KEV'], bins=3000, range=(0, 3000))

# Subtract
# net_hist, uncertainty = subtract_background(signal_hist, bg_hist,
#                                            signal_livetime=3600,
#                                            background_livetime=3600)

# Plot comparison
# bin_centers = (bins[:-1] + bins[1:]) / 2
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(bin_centers, signal_hist, label='Signal + Background', alpha=0.7)
# ax.plot(bin_centers, bg_hist, label='Background', alpha=0.7)
# ax.plot(bin_centers, net_hist, label='Net Signal', linewidth=2)
# ax.set_xlabel('Energy (keV)', fontsize=14)
# ax.set_ylabel('Counts', fontsize=14)
# ax.set_title('Background Subtraction', fontsize=16)
# ax.legend(fontsize=12)
# ax.set_yscale('log')
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ## APPENDIX E: Minimum Detectable Activity (MDA)

# %% Calculate MDA
def calculate_mda(background_counts, efficiency, branching_ratio, 
                 live_time, confidence_level=0.95):
    """
    Calculate Minimum Detectable Activity using Currie's formula
    
    MDA = (k_α + k_β) * sqrt(N_bg) / (ε * BR * t)
    
    where k_α = k_β ≈ 1.645 for 95% confidence (one-sided)
    
    Parameters:
    -----------
    background_counts : float
        Counts in ROI from background
    efficiency : float
        Detection efficiency
    branching_ratio : float
        Gamma emission probability
    live_time : float
        Measurement time (seconds)
    confidence_level : float
        Confidence level (0.95 = 95%)
    
    Returns:
    --------
    mda_bq : float
        Minimum detectable activity in Bq
    """
    from scipy.stats import norm
    
    # Critical value for confidence level
    k = norm.ppf(confidence_level)
    
    # Currie formula
    mda_counts = k * (2 + k) * np.sqrt(background_counts)
    
    # Convert to activity
    mda_bq = mda_counts / (efficiency * branching_ratio * live_time)
    
    return mda_bq


# %% Example MDA calculation
# For Cs-137 detection
# bg_counts_662 = 100  # Background counts in ROI
# eff_662 = 0.01
# br_662 = 0.851
# time_sec = 3600

# mda = calculate_mda(bg_counts_662, eff_662, br_662, time_sec)
# print(f"Minimum Detectable Activity (95% CL): {mda:.1f} Bq = {mda/1000:.3f} kBq")

# %% [markdown]
# ## APPENDIX F: Dose Rate Estimation

# %% Estimate dose rate from NORM
def estimate_dose_rate(activities_bq, distance_cm=100):
    """
    Rough estimate of gamma dose rate
    
    Simplified calculation using gamma constant
    Dose rate (μSv/h) ≈ Γ * Activity(MBq) / distance²(m)
    
    Gamma constants (μSv·m²/MBq·h):
    - Cs-137: 0.32
    - Co-60: 1.32
    - Ra-226: 0.86 (with daughters in equilibrium)
    
    Parameters:
    -----------
    activities_bq : dict
        {isotope: activity_in_bq}
    distance_cm : float
        Distance from source
    
    Returns:
    --------
    total_dose_rate : float
        Estimated dose rate in μSv/h
    """
    gamma_constants = {
        'Cs-137': 0.32,
        'Co-60': 1.32,
        'Ra-226': 0.86,
        'K-40': 0.14,
        'U-238 series': 0.5,  # Average for series
        'Th-232 series': 0.6   # Average for series
    }
    
    distance_m = distance_cm / 100
    total_dose = 0
    
    print("\nDose Rate Estimates:")
    print(f"Distance: {distance_cm} cm\n")
    
    for isotope, activity_bq in activities_bq.items():
        activity_mbq = activity_bq / 1e6
        
        if isotope in gamma_constants:
            gamma_const = gamma_constants[isotope]
            dose_rate = gamma_const * activity_mbq / (distance_m ** 2)
            total_dose += dose_rate
            
            print(f"{isotope}:")
            print(f"  Activity: {activity_bq:.0f} Bq = {activity_mbq:.3f} MBq")
            print(f"  Dose rate: {dose_rate:.3f} μSv/h")
        else:
            print(f"{isotope}: No gamma constant available (skipped)")
    
    print(f"\nTotal estimated dose rate: {total_dose:.3f} μSv/h")
    print(f"                           {total_dose*24:.2f} μSv/day")
    print(f"                           {total_dose*24*365/1000:.2f} mSv/year")
    
    return total_dose


# %% Example dose calculation
# activities = {
#     'U-238 series': 5000,  # Bq
#     'Th-232 series': 3000,
#     'K-40': 1000
# }
# dose_rate = estimate_dose_rate(activities, distance_cm=100)

# %% [markdown]
# ## APPENDIX G: Data Export for Further Analysis

# %% Export to ROOT format (if needed)
# Useful for integration with CERN ROOT framework

def export_to_root(df, filename='output.root'):
    """
    Export data to ROOT format
    Requires uproot library: pip install uproot
    """
    try:
        import uproot
        
        # Prepare data
        data_dict = {
            'energy': df['ENERGY_KEV'].values if 'ENERGY_KEV' in df.columns else df['ENERGY'].values,
            'psd': df['PSD'].values,
            'timetag': df['TIMETAG'].values,
        }
        
        if 'PARTICLE' in df.columns:
            # Convert to numeric: neutron=1, gamma=0
            data_dict['particle'] = (df['PARTICLE'] == 'neutron').astype(int).values
        
        # Write to ROOT file
        with uproot.recreate(filename) as f:
            f['tree'] = data_dict
        
        print(f"Exported to ROOT format: {filename}")
        
    except ImportError:
        print("uproot not installed. Install with: pip install uproot")


# %% Export to HDF5
def export_to_hdf5(df, filename='output.h5'):
    """
    Export to HDF5 format (efficient for large datasets)
    """
    df.to_hdf(filename, key='psd_data', mode='w', complevel=9)
    print(f"Exported to HDF5 format: {filename}")


# %% Export summary statistics to JSON
def export_summary_json(results_dict, filename='analysis_summary.json'):
    """
    Export analysis results to machine-readable JSON
    """
    import json
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj
    
    results_clean = convert_types(results_dict)
    
    with open(filename, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"Exported summary to JSON: {filename}")


# %% Example export
# export_to_hdf5(df_clean, 'processed_psd_data.h5')
# export_to_root(df_clean, 'psd_data.root')

# summary_results = {
#     'filename': filename,
#     'total_events': len(df_clean),
#     'calibration_params': cal_params.tolist(),
#     'identified_isotopes': list(identified_series.keys()),
#     'neutron_count': len(neutrons) if 'neutrons' in locals() else 0,
#     'gamma_count': len(gammas) if 'gammas' in locals() else 0
# }
# export_summary_json(summary_results, 'analysis_summary.json')

# %% [markdown]
# ## Analysis Complete!
# 
# ### Checklist:
# - ✅ Data loaded and quality-controlled
# - ✅ PSD parameters calculated
# - ✅ Energy calibration performed
# - ✅ Particle discrimination applied
# - ✅ Isotopes identified
# - ✅ Results exported
# 
# ### For Production Analysis Pipeline:
# 1. Automate this workflow in Python scripts
# 2. Create configuration files for detector-specific parameters
# 3. Implement batch processing for multiple measurements
# 4. Set up database for long-term trending
# 5. Create automated QA/QC checks
# 6. Generate standardized reports (PDF/HTML)
# 
# ### Additional Resources:
# - IAEA Safety Reports for NORM
# - IEEE Nuclear Science papers on PSD techniques
# - Detector manufacturer's documentation for efficiency curves
# - National nuclear data libraries (NNDC, ENSDF)