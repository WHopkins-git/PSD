# Complete PSD Analysis Toolkit - Implementation Guide

## Quick Start Guide

### What You Have Now

1. **Comprehensive Analysis Toolkit Documentation** - Complete overview of all required functions
2. **Python Implementation** - Core functions ready to use
3. **Jupyter Notebook Template** - Interactive analysis workflow
4. **This file** - Implementation roadmap

---

## Understanding Your Data

### What the Numbers Mean

**ADC Values** (e.g., 3253, 1672):
- **NOT** integrated charge
- **Instantaneous voltage measurements** sampled at discrete times
- Think of them as "snapshots" of the detector pulse voltage

**ENERGY** (e.g., 1817):
- Sum of (baseline-subtracted) ADC samples over long gate
- **This IS proportional to charge** collected
- Represents total light/energy deposited

**ENERGYSHORT** (e.g., 118):
- Sum over short gate (fast component only)
- Used for discrimination

**PSD Ratio** (e.g., 0.935):
- (ENERGY - ENERGYSHORT) / ENERGY
- Fraction of energy in tail component
- **High ratio (0.85-0.95) → Neutron**
- **Low ratio (0.15-0.30) → Gamma**

---

## Implementation Roadmap

### Phase 1: Basic Analysis (Week 1)

**Goal:** Get data loaded, visualized, and understood

```python
# Install dependencies
pip install numpy pandas matplotlib scipy seaborn

# Basic workflow
from psd_analysis import *

# 1. Load data
df = load_psd_data('your_file.csv')

# 2. Quality control
valid_mask, qc = validate_events(df)
df_clean = df[valid_mask]

# 3. Calculate PSD
df_clean = calculate_psd_ratio(df_clean)

# 4. Visualize
plot_psd_scatter(df_clean)
plot_energy_spectra(df_clean)
```

**Deliverables:**
- Confirm data loads correctly
- Basic PSD scatter plot
- Raw energy spectrum
- Understanding of data structure

---

### Phase 2: Calibration (Week 2)

**Goal:** Convert ADC to physical energy units

**You'll need:**
- Measurement with known source (Cs-137, Co-60, etc.)
- Knowledge of expected energies

**Steps:**
1. Identify Compton edges or photopeaks in spectrum
2. Match to known energies
3. Perform linear or polynomial fit
4. Validate calibration quality

```python
# Find peaks
hist, bins = np.histogram(df['ENERGY'], bins=2000)
peaks, _, _ = find_peaks_in_spectrum(bins[:-1], hist)

# Calibrate
calibration_points = [
    (peaks[0], 477),   # Cs-137 Compton edge
    (peaks[1], 662)    # Cs-137 photopeak
]
df, cal_func, params = calibrate_energy(df, calibration_points)

# Save calibration
np.save('calibration_params.npy', params)
```

**Deliverables:**
- Calibration curve (ADC → keV)
- Energy resolution measurements
- Calibrated spectra

---

### Phase 3: PSD Discrimination (Week 3)

**Goal:** Separate neutrons and gammas

**For known mixtures:**
```python
# Define boundary (manual or automated)
boundary_func, params = define_linear_discrimination(df_calibration)

# Apply to data
df = apply_discrimination(df, boundary_func)

# Evaluate performance
fom = calculate_figure_of_merit(neutrons['PSD'], gammas['PSD'])
```

**For optimization:**
- Use pure sources (AmBe for neutrons, Cs-137 for gammas)
- Scan gate timing to maximize FOM
- Build energy-dependent boundary if needed

**Deliverables:**
- Discrimination function
- Separated n/γ spectra
- Figure of Merit calculation
- Validation with known sources

---

### Phase 4: NORM Identification (Week 4)

**Goal:** Identify unknown radioactive materials

**Steps:**
1. Apply calibration to unknown sample
2. Discriminate gammas (ignore neutrons for isotope ID)
3. Find peaks in gamma spectrum
4. Match to isotope library
5. Identify decay chains

```python
# Find peaks
peaks, counts, _ = find_peaks_in_spectrum(energy, spectrum)

# Match to library
matches = match_peaks_to_library(peaks)

# Identify series
series = identify_decay_chains(matches)
```

**Common NORM isotopes you'll see:**
- **U-238 series:** 609 keV (Bi-214), 1001 keV (Pa-234m)
- **Th-232 series:** 583 keV (Tl-208), 2614 keV (Tl-208)
- **K-40:** 1461 keV
- **Ra-226:** Often in equilibrium with U-238 daughters

**Deliverables:**
- Identified isotopes with confidence scores
- Peak assignments
- Decay chain analysis
- Estimated activities (if efficiency known)

---

## Detector-Specific Customization

### Things You MUST Adjust for Your Setup

1. **ADC Range:**
   ```python
   # Update in validate_events()
   adc_min = 0
   adc_max = 16383  # For 14-bit ADC (2^14 - 1)
   ```

2. **Sampling Rate:**
   ```python
   # If your timetag units are clock ticks
   sampling_rate_MHz = 250  # Example: 250 MHz
   time_per_sample_ns = 1000 / sampling_rate_MHz  # = 4 ns
   ```

3. **Gate Timings:**
   - **Short gate:** Typically 20-50 ns (5-12 samples at 250 MHz)
   - **Long gate:** Typically 300-500 ns (75-125 samples)
   - Optimize using `optimize_psd_gates()`

4. **Baseline Region:**
   ```python
   # Number of samples before pulse for baseline
   baseline_samples = 50  # Adjust based on your trigger
   ```

5. **Energy Thresholds:**
   ```python
   # Minimum energy for analysis (reject noise)
   energy_threshold_adc = 100  # Adjust based on noise level
   ```

---

## Code Organization

### Recommended Project Structure

```
psd_analysis_project/
│
├── data/
│   ├── calibration/
│   │   ├── cs137_20241021.csv
│   │   ├── co60_20241021.csv
│   │   └── background_20241021.csv
│   ├── norm_samples/
│   │   └── unknown_sample_001.csv
│   └── processed/
│       └── *.h5
│
├── psd_analysis/
│   ├── __init__.py
│   ├── io/
│   │   ├── data_loader.py
│   │   └── quality_control.py
│   ├── calibration/
│   │   ├── energy_cal.py
│   │   └── efficiency.py
│   ├── psd/
│   │   ├── parameters.py
│   │   ├── discrimination.py
│   │   └── optimization.py
│   ├── spectroscopy/
│   │   ├── spectrum.py
│   │   ├── peak_finding.py
│   │   └── isotope_id.py
│   ├── visualization/
│   │   └── plots.py
│   └── utils/
│       ├── isotope_library.py
│       └── physics.py
│
├── notebooks/
│   ├── 01_calibration.ipynb
│   ├── 02_norm_analysis.ipynb
│   └── 03_detector_characterization.ipynb
│
├── scripts/
│   ├── batch_process.py
│   ├── generate_reports.py
│   └── compare_measurements.py
│
├── config/
│   ├── detector_config.yaml
│   └── analysis_params.yaml
│
├── results/
│   ├── figures/
│   ├── reports/
│   └── exports/
│
├── tests/
│   └── test_*.py
│
├── requirements.txt
└── README.md
```

---

## Configuration Files

### detector_config.yaml
```yaml
detector:
  name: "MyScintillator"
  type: "Liquid Scintillator"
  adc_bits: 14
  adc_range: [0, 16383]
  sampling_rate_mhz: 250
  
psd_gates:
  short_gate_samples: 20
  long_gate_samples: 100
  
thresholds:
  energy_min_adc: 100
  saturation_adc: 16000
  
calibration:
  method: "linear"  # or "polynomial"
  reference_source: "Cs-137"
  last_calibration_date: "2024-10-21"
  coefficients: [0.3642, 0.0]  # [slope, intercept]
```

### analysis_params.yaml
```yaml
quality_control:
  max_baseline_rms: 50
  check_pileup: true
  check_saturation: true

spectroscopy:
  peak_finding:
    prominence: 100
    distance: 20  # samples
  
  peak_fitting:
    fit_width_kev: 50
    background_model: "linear"

isotope_identification:
  matching_tolerance_kev: 10
  min_confidence: 0.7
  
discrimination:
  boundary_type: "linear"  # or "energy_dependent"
  psd_range: [0, 1]

output:
  save_figures: true
  figure_format: "png"
  figure_dpi: 150
  export_processed_data: true
  export_format: "hdf5"  # or "root", "csv"
```

---

## Extending to NORM Analysis

### Key Differences: Calibration vs. NORM

| Aspect | Calibration Source | NORM Sample |
|--------|-------------------|-------------|
| **Source** | Known (Cs-137, Co-60) | Unknown mixture |
| **Goal** | Calibrate detector | Identify isotopes |
| **Energy** | Few discrete lines | Many lines, complex |
| **Analysis** | Find peaks → calibrate | Find peaks → match library |
| **Output** | Calibration function | Isotope ID + activities |
| **Neutrons** | May or may not have | Rare (except U/Th with (α,n)) |

### NORM-Specific Considerations

1. **Decay Chain Equilibrium:**
   - U-238 and Th-232 decay through many daughters
   - If in equilibrium: all daughters have same activity
   - Look for multiple peaks from same series

2. **Peak Overlaps:**
   - Many NORM peaks overlap
   - Use peak fitting to resolve
   - Cross-check with other lines from same isotope

3. **Shielding Effects:**
   - Low-energy gammas may be attenuated
   - Self-absorption in sample
   - Affects relative peak intensities

4. **Activity Calculations:**
   - Requires detector efficiency curve
   - Need geometry factor
   - Account for branching ratios

---

## Validation & Quality Assurance

### Daily Checks
- [ ] Background measurement (no source)
- [ ] Check source measurement (Cs-137 or check source)
- [ ] Verify peak position stable (±1-2%)
- [ ] Check count rate within expected range

### Weekly Checks
- [ ] Full energy calibration
- [ ] Resolution check at multiple energies
- [ ] PSD performance (FOM) with known sources
- [ ] Review rejected events statistics

### Monthly Checks
- [ ] Detector efficiency curve validation
- [ ] Long-term stability trending
- [ ] Cross-calibration with reference detector
- [ ] Software version control check

### Validation Dataset
Create a "golden" dataset with:
- Known source (Cs-137)
- Known geometry and activity
- Expected results documented
- Re-analyze periodically to check consistency

---

## Troubleshooting Guide

### Problem: No clear PSD separation

**Possible causes:**
- Gate timing not optimized
- Low light output (low energy events)
- Electronic noise
- Pile-up

**Solutions:**
1. Re-do energy calibration
2. Check peak fitting quality
3. Increase matching tolerance
4. Consider nuclear recoil effects (for neutron interactions)

### Problem: Too many false peaks

**Possible causes:**
- Statistical fluctuations
- Compton continuum structure
- Poor background subtraction

**Solutions:**
1. Increase prominence threshold
2. Smooth spectrum before peak finding
3. Better background subtraction
4. Require minimum peak width

### Problem: Can't identify isotopes

**Possible causes:**
- Weak source (low statistics)
- Complex mixture
- Missing library entries
- Calibration issues

**Solutions:**
1. Longer measurement time
2. Use decay chain analysis
3. Expand isotope library
4. Verify calibration accuracy
5. Check for systematic peak shifts

---

## Performance Benchmarks

### Expected Results (Typical Organic Scintillator)

**Energy Resolution:**
- At 662 keV (Cs-137): 8-12%
- Improves at higher energies (√E dependence)

**PSD Performance:**
- Figure of Merit (FOM): 1.2-2.0 (good to excellent)
- Neutron detection efficiency: 20-60% (energy dependent)
- Gamma rejection: 99%+ with proper discrimination

**Peak Finding:**
- Detection limit: ~10 counts above background (3σ)
- Energy resolution limited peak separation: ~2-3 × FWHM

**Processing Speed:**
- ~10,000-100,000 events/second (Python, single core)
- Can process typical 1 million event file in 10-100 seconds

---

## Advanced Features to Implement Later

### Machine Learning for PSD

Instead of simple linear cuts:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train classifier on labeled data
X = df[['ENERGY_KEV', 'PSD', 'RISE_TIME', 'TAIL_INTEGRAL']].values
y = (df['PARTICLE'] == 'neutron').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Apply to unknown data
df['PARTICLE_ML'] = clf.predict(X)
```

**Advantages:**
- Can use multiple features
- Non-linear boundaries
- Often better performance than simple cuts

### Pulse Shape Analysis Beyond Basic PSD

1. **Rise Time Analysis:**
   - Time from 10% to 90% of peak
   - Different for n vs γ

2. **Tail Slope:**
   - Exponential decay constant
   - Multiple time constants in organic scintillators

3. **Frequency Domain:**
   - FFT of pulse
   - Different frequency content for n vs γ

### Real-Time Analysis

For online monitoring:

```python
import threading
import queue

class RealtimePSDAnalyzer:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.results = []
        
    def process_stream(self):
        while True:
            event = self.data_queue.get()
            if event is None:
                break
            
            # Quick analysis
            psd = self.calculate_psd(event)
            energy = self.calculate_energy(event)
            
            # Classify
            particle = self.classify(energy, psd)
            
            # Update counters
            self.update_statistics(particle, energy)
            
            # Alert if threshold exceeded
            if self.check_alarm_conditions():
                self.trigger_alarm()
```

### Automated Reporting

```python
def generate_report(results, template='norm_report.html'):
    """
    Generate HTML/PDF report with:
    - Measurement metadata
    - QC statistics
    - Identified isotopes
    - Key spectra and plots
    - Activity calculations
    - Dose rate estimates
    """
    from jinja2 import Template
    
    with open(template) as f:
        template = Template(f.read())
    
    html = template.render(
        date=results['date'],
        source=results['source_id'],
        isotopes=results['identified_isotopes'],
        spectra_plots=results['plots'],
        activities=results['activities']
    )
    
    # Convert to PDF
    import pdfkit
    pdfkit.from_string(html, 'report.pdf')
```

---

## Integration with Existing Systems

### Database Integration

For long-term storage and trending:

```python
import sqlite3

def create_results_database():
    conn = sqlite3.connect('psd_results.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE measurements (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            filename TEXT,
            source_type TEXT,
            measurement_time REAL,
            total_events INTEGER,
            neutron_count INTEGER,
            gamma_count INTEGER
        )
    ''')
    
    c.execute('''
        CREATE TABLE isotopes (
            id INTEGER PRIMARY KEY,
            measurement_id INTEGER,
            isotope TEXT,
            confidence REAL,
            activity_bq REAL,
            uncertainty_bq REAL,
            FOREIGN KEY (measurement_id) REFERENCES measurements(id)
        )
    ''')
    
    c.execute('''
        CREATE TABLE calibrations (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            detector_id TEXT,
            cal_params TEXT,
            resolution REAL,
            efficiency_curve TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
```

### LIMS Integration

For lab information management:

```python
def upload_to_lims(results, lims_api_url):
    """
    Upload analysis results to LIMS
    """
    import requests
    
    payload = {
        'sample_id': results['sample_id'],
        'analysis_date': results['timestamp'],
        'isotopes_detected': results['isotopes'],
        'activities': results['activities'],
        'dose_rate': results['dose_rate'],
        'analyst': results['analyst'],
        'instrument': results['detector_id']
    }
    
    response = requests.post(
        f"{lims_api_url}/api/v1/results",
        json=payload,
        headers={'Authorization': f'Bearer {API_TOKEN}'}
    )
    
    return response.status_code == 200
```

---

## Best Practices

### Data Management

1. **Raw Data Preservation:**
   - Never overwrite original data files
   - Keep full waveforms if storage permits
   - Archive with metadata (date, detector, settings)

2. **Processing History:**
   - Track analysis software version
   - Record all processing parameters
   - Enable reproducibility

3. **Backup Strategy:**
   - Regular backups of data and results
   - Off-site storage for critical measurements
   - Version control for code

### Analysis Workflow

1. **Always start with QC:**
   - Check data quality before analysis
   - Document rejection criteria
   - Review rejected events periodically

2. **Validate calibrations:**
   - Regular check source measurements
   - Track calibration stability over time
   - Document calibration procedure

3. **Uncertainty propagation:**
   - Track uncertainties through all calculations
   - Report results with uncertainties
   - Use proper statistical methods

4. **Peer review:**
   - Have results reviewed by another analyst
   - Cross-check against independent methods
   - Document unusual findings

---

## Common Use Cases

### Use Case 1: Daily Monitoring

**Scenario:** Routine NORM screening of environmental samples

**Workflow:**
1. Background measurement (15 min)
2. Sample measurement (1-4 hours)
3. Automated analysis pipeline
4. Flag if activity exceeds threshold
5. Generate summary report

**Key metrics:**
- Total count rate
- Identified isotopes
- Activities vs. limits
- Trends vs. previous measurements

### Use Case 2: Detector Characterization

**Scenario:** Full characterization of new detector

**Measurements needed:**
- Intrinsic background (overnight)
- Energy calibration (multiple sources)
- Efficiency curve (calibrated sources)
- PSD optimization (pure n and γ sources)
- Resolution vs. energy
- Linearity check
- Time stability

**Deliverables:**
- Detector specification sheet
- Calibration curves and parameters
- Operating procedures
- QC limits

### Use Case 3: Unknown Source Investigation

**Scenario:** Identify unknown radioactive material

**Workflow:**
1. Safety assessment (dose rate measurement)
2. Preliminary gamma scan
3. Long measurement for statistics
4. Full spectroscopic analysis
5. Isotope identification
6. Activity determination
7. Compliance check vs. regulations
8. Comprehensive report

**Critical factors:**
- Good energy calibration
- Sufficient statistics
- Proper background subtraction
- Chain of custody documentation

### Use Case 4: Neutron Source Characterization

**Scenario:** Characterize AmBe or other neutron source

**Measurements:**
- PSD scatter plot (verify n/γ separation)
- Neutron energy spectrum
- Gamma spectrum from (α,n) reaction
- Light output calibration
- Absolute neutron flux

**Challenges:**
- Non-linear light output for neutrons
- Energy calibration requires unfolding
- Efficiency varies strongly with energy

---

## Regulatory Compliance

### Documentation Requirements

Many NORM applications require:

1. **Measurement Procedures:**
   - Standard Operating Procedures (SOPs)
   - Calibration procedures
   - QA/QC procedures

2. **Results Documentation:**
   - Analysis reports with uncertainties
   - Calibration certificates
   - QC charts and trending

3. **Quality System:**
   - ISO 17025 (if laboratory accredited)
   - Traceability to standards
   - Proficiency testing participation

### Reporting Formats

Typical report should include:

- **Sample Information:** ID, description, geometry, matrix
- **Measurement Conditions:** Detector, geometry, time, background
- **Energy Calibration:** Method, sources, coefficients, validation
- **Analysis Method:** PSD settings, peak finding, isotope ID
- **Results:** Isotopes, activities, uncertainties, MDAs
- **QC Information:** Check source, control charts, validation
- **Interpretation:** Comparison to limits, recommendations
- **Analyst Information:** Name, qualifications, signature, date

---

## Learning Resources

### Recommended Reading

**Books:**
1. "Radiation Detection and Measurement" - Knoll
2. "Nuclear Electronics" - Nicholson
3. "Handbook of Radioactivity Analysis" - L'Annunziata

**Papers:**
1. Brooks & Klein (1990) - "Neutron/gamma discrimination in organic scintillators"
2. Flaska et al. (2006) - "Digital pulse shape discrimination"
3. Pozzi et al. (2003) - "PSD methods comparison"

**Standards:**
1. IEEE Std 325 - "Test Procedures for Germanium Gamma-Ray Detectors"
2. ISO 18589 - "Gamma-ray spectrometry for NORM"
3. ANSI N42.14 - "Calibration and Use of Gamma-Ray Spectrometers"

### Online Resources

- **Nuclear Data:**
  - NNDC (National Nuclear Data Center)
  - LiveChart of Nuclides (IAEA)
  - ENSDF (Evaluated Nuclear Structure Data File)

- **Software:**
  - ROOT (CERN) - Data analysis framework
  - Geant4 - Detector simulation
  - PyROOT - Python interface to ROOT

---

## Next Steps

### Immediate (This Week)

1. ✅ Load your actual data file
2. ✅ Run basic analysis workflow
3. ✅ Generate your first PSD scatter plot
4. ✅ Create energy histogram
5. ✅ Identify any obvious issues with data

### Short-term (This Month)

1. ⬜ Perform energy calibration with known source
2. ⬜ Optimize PSD gate timing
3. ⬜ Define discrimination boundary
4. ⬜ Analyze first NORM sample
5. ⬜ Set up automated processing

### Medium-term (Next 3 Months)

1. ⬜ Build complete isotope library for your application
2. ⬜ Develop detector efficiency curve
3. ⬜ Implement batch processing
4. ⬜ Create automated reporting
5. ⬜ Set up QA/QC program
6. ⬜ Validate with reference materials

### Long-term (6+ Months)

1. ⬜ Machine learning classification
2. ⬜ Real-time analysis capability
3. ⬜ Database integration
4. ⬜ LIMS connectivity
5. ⬜ Multi-detector systems
6. ⬜ Advanced unfolding methods

---

## Getting Help

### Troubleshooting Checklist

Before asking for help:

1. ✅ Check data file format is correct
2. ✅ Verify column names match expected format
3. ✅ Check for obvious data quality issues
4. ✅ Review error messages carefully
5. ✅ Try with example/test data first
6. ✅ Check that all dependencies installed
7. ✅ Review relevant documentation sections

### Information to Provide

When seeking help, include:

- Detector type and model
- Data acquisition system details
- Sample error messages or plots
- Description of what you've tried
- What you expect vs. what you observe
- Relevant data files (if shareable)

---

## Summary

You now have:

1. ✅ **Understanding** of PSD data structure (ADC values, integration, PSD)
2. ✅ **Complete toolkit** for calibration and NORM analysis
3. ✅ **Python implementation** ready to customize
4. ✅ **Jupyter notebook** for interactive analysis
5. ✅ **Best practices** and workflow guidance
6. ✅ **Troubleshooting** resources
7. ✅ **Path forward** for implementation

**Start simple, validate each step, then build complexity.**

Good luck with your PSD analysis! 🎯☢️:**
1. Optimize gate positions using pure sources
2. Apply energy threshold (reject low-energy)
3. Check for electronic pickup/noise
4. Reduce count rate if pile-up present

### Problem: Energy calibration unstable

**Possible causes:**
- Temperature drift
- PMT voltage drift
- Electronic gain changes

**Solutions:**
1. Temperature stabilization
2. More frequent calibrations
3. Hardware checks
4. Use peak tracking algorithm

### Problem: Peak positions don't match library

**Possible causes:**
- Calibration error
- Doppler broadening
- Unknown source composition

**Solutions