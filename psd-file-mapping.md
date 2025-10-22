# Complete File Organization Guide

## How to Organize All the Code

This guide shows **exactly** how to take the provided code documents and organize them into the proper project structure.

---

## Document Mapping

### Document 1: "PSD Analysis Starter Code (Python)"

This single file needs to be **split** into multiple modules:

```
PSD Analysis Starter Code → Split into these files:
│
├── psd_analysis/io/data_loader.py
│   └── Functions: load_psd_data()
│
├── psd_analysis/io/quality_control.py
│   └── Functions: validate_events()
│
├── psd_analysis/calibration/energy_cal.py
│   └── Functions: find_compton_edge(), calibrate_energy()
│
├── psd_analysis/psd/parameters.py
│   └── Functions: calculate_psd_ratio(), calculate_figure_of_merit()
│
├── psd_analysis/psd/discrimination.py
│   └── Functions: define_linear_discrimination(), apply_discrimination()
│
├── psd_analysis/spectroscopy/peak_finding.py
│   └── Functions: find_peaks_in_spectrum(), fit_gaussian_peak()
│
├── psd_analysis/spectroscopy/isotope_id.py
│   └── Data: ISOTOPE_LIBRARY
│   └── Functions: match_peaks_to_library(), identify_decay_chains()
│
├── psd_analysis/visualization/plots.py
│   └── Functions: plot_psd_scatter(), plot_energy_spectra(), plot_calibration_curve()
│
└── scripts/workflows.py
    └── Functions: analyze_calibration_source(), analyze_norm_source()
```

### Document 2: "PSD Analysis - Missing Modules & ML Implementation"

Contains the NEW modules that were empty:

```
Missing Modules Document → These files:
│
├── psd_analysis/calibration/efficiency.py
│   └── Class: EfficiencyCurve
│   └── Functions: calculate_efficiency_from_source()
│
├── psd_analysis/psd/optimization.py
│   └── Functions: optimize_gate_timing(), plot_fom_landscape()
│           optimize_energy_dependent_boundary()
│
└── psd_analysis/spectroscopy/spectrum.py
    └── Class: EnergySpectrum
    └── Functions: subtract_compton_continuum()
```

### Document 3: "PSD Analysis - Machine Learning Module"

The complete ML implementation:

```
ML Module Document → These files:
│
├── psd_analysis/ml/__init__.py
│
├── psd_analysis/ml/classical.py
│   └── Class: ClassicalMLClassifier
│   └── Functions: plot_ml_performance()
│
└── psd_analysis/ml/deep_learning.py
    └── Classes: WaveformDataset, CNN1DClassifier, TransformerClassifier
                 PhysicsInformedLoss, DeepPSDClassifier
    └── Functions: plot_training_history()
```

### Document 4: "PSD ML Analysis - Complete Example"

Goes into scripts folder:

```
ML Example Document → scripts/ml_analysis_example.py
```

### Document 5: "PSD Analysis Jupyter Notebook Template"

Goes into notebooks folder:

```
Jupyter Template → notebooks/01_psd_analysis.ipynb
```

---

## Step-by-Step Setup

### Step 1: Create Directory Structure

```bash
mkdir -p psd_analysis_project/{psd_analysis/{io,calibration,psd/{ml,},spectroscopy,visualization,utils},notebooks,scripts,data/{calibration,norm_samples,processed},models,config,results/{figures,reports,exports},tests}
```

### Step 2: Create All __init__.py Files

**psd_analysis/__init__.py:**
```python
"""PSD Analysis Toolkit"""

# Import key functions for easy access
from .io.data_loader import load_psd_data
from .io.quality_control import validate_events
from .calibration.energy_cal import calibrate_energy, find_compton_edge
from .calibration.efficiency import EfficiencyCurve
from .psd.parameters import calculate_psd_ratio, calculate_figure_of_merit
from .psd.discrimination import define_linear_discrimination, apply_discrimination
from .psd.optimization import optimize_gate_timing
from .spectroscopy.spectrum import EnergySpectrum
from .spectroscopy.peak_finding import find_peaks_in_spectrum, fit_gaussian_peak
from .spectroscopy.isotope_id import match_peaks_to_library, identify_decay_chains, ISOTOPE_LIBRARY
from .visualization.plots import plot_psd_scatter, plot_energy_spectra, plot_calibration_curve

__version__ = '1.0.0'
```

**psd_analysis/io/__init__.py:**
```python
from .data_loader import load_psd_data
from .quality_control import validate_events
```

**psd_analysis/calibration/__init__.py:**
```python
from .energy_cal import calibrate_energy, find_compton_edge
from .efficiency import EfficiencyCurve, calculate_efficiency_from_source
```

**psd_analysis/psd/__init__.py:**
```python
from .parameters import calculate_psd_ratio, calculate_figure_of_merit
from .discrimination import define_linear_discrimination, apply_discrimination
from .optimization import optimize_gate_timing, plot_fom_landscape
```

**psd_analysis/spectroscopy/__init__.py:**
```python
from .spectrum import EnergySpectrum, subtract_compton_continuum
from .peak_finding import find_peaks_in_spectrum, fit_gaussian_peak
from .isotope_id import match_peaks_to_library, identify_decay_chains, ISOTOPE_LIBRARY
```

**psd_analysis/visualization/__init__.py:**
```python
from .plots import plot_psd_scatter, plot_energy_spectra, plot_calibration_curve
```

**psd_analysis/ml/__init__.py:**
```python
from .classical import ClassicalMLClassifier, plot_ml_performance
try:
    from .deep_learning import DeepPSDClassifier, plot_training_history
except ImportError:
    # PyTorch not available
    pass
```

**psd_analysis/utils/__init__.py:**
```python
# Empty for now, or add utility functions
```

### Step 3: Copy Code to Proper Files

#### From Document 1 (Starter Code):

**psd_analysis/io/data_loader.py:**
```python
"""Data loading functions"""
import pandas as pd

def load_psd_data(filename, delimiter=';'):
    # Copy the load_psd_data function from Document 1, Section 1
    pass
```

**psd_analysis/io/quality_control.py:**
```python
"""Quality control functions"""
import numpy as np
import warnings

def validate_events(df, adc_min=0, adc_max=16383, baseline_stability=50):
    # Copy the validate_events function from Document 1, Section 2
    pass
```

Continue for all sections...

#### From Document 2 (Missing Modules):

**psd_analysis/calibration/efficiency.py:**
```python
# Copy entire content from Document 2, efficiency.py section
```

**psd_analysis/psd/optimization.py:**
```python
# Copy entire content from Document 2, optimization.py section
```

**psd_analysis/spectroscopy/spectrum.py:**
```python
# Copy entire content from Document 2, spectrum.py section
```

#### From Document 3 (ML):

**psd_analysis/ml/classical.py:**
```python
# Copy classical ML section from Document 3
```

**psd_analysis/ml/deep_learning.py:**
```python
# Copy deep learning section from Document 3
```

### Step 4: Create Configuration Files

**config/detector_config.yaml:**
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
```

**config/ml_config.yaml:**
```yaml
classical_ml:
  default_method: "random_forest"
  test_size: 0.2
  random_state: 42
  
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5

deep_learning:
  default_model: "cnn"
  epochs: 30
  batch_size: 64
  learning_rate: 0.001
  use_physics_loss: true
  
  cnn:
    conv_layers: 4
    filters: [32, 64, 128, 256]
  
  transformer:
    d_model: 128
    nhead: 8
    num_layers: 4
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
seaborn>=0.11.0
scikit-learn>=1.0.0
joblib>=1.0.0
pyyaml>=5.4.0

# Optional: for deep learning
# torch>=2.0.0
```

### Step 5: Create Example Scripts

**scripts/quick_analysis.py:**
```python
"""Quick analysis script for single file"""
import sys
sys.path.append('..')

from psd_analysis import (
    load_psd_data, 
    validate_events,
    calculate_psd_ratio,
    plot_psd_scatter,
    plot_energy_spectra
)

if __name__ == "__main__":
    # Load data
    df = load_psd_data('../data/calibration/cs137_source.csv')
    
    # QC
    valid, qc = validate_events(df)
    df = df[valid]
    
    # Calculate PSD
    df = calculate_psd_ratio(df)
    
    # Visualize
    plot_psd_scatter(df)
    plot_energy_spectra(df)
    
    print(f"Analyzed {len(df)} events")
```

**scripts/ml_train.py:**
```python
"""Train ML model"""
import sys
sys.path.append('..')

from psd_analysis import load_psd_data, validate_events, calculate_psd_ratio
from psd_analysis.ml import ClassicalMLClassifier
import pandas as pd

if __name__ == "__main__":
    # Load calibration data
    df_n = load_psd_data('../data/calibration/ambe_source.csv')
    df_g = load_psd_data('../data/calibration/cs137_source.csv')
    
    # Process
    df_n = df_n[validate_events(df_n)[0]]
    df_g = df_g[validate_events(df_g)[0]]
    df_n = calculate_psd_ratio(df_n)
    df_g = calculate_psd_ratio(df_g)
    
    # Label
    df_n['PARTICLE'] = 'neutron'
    df_g['PARTICLE'] = 'gamma'
    
    # Combine
    df_train = pd.concat([df_n, df_g], ignore_index=True)
    
    # Train
    clf = ClassicalMLClassifier(method='random_forest')
    results = clf.train(df_train)
    
    # Save
    clf.save('../models/psd_random_forest.pkl')
    
    print("✓ Model trained and saved")
```

**scripts/ml_predict.py:**
```python
"""Apply trained model"""
import sys
sys.path.append('..')

from psd_analysis import load_psd_data, validate_events, calculate_psd_ratio
from psd_analysis.ml import ClassicalMLClassifier

if __name__ == "__main__":
    # Load model
    clf = ClassicalMLClassifier(method='random_forest')
    clf.load('../models/psd_random_forest.pkl')
    
    # Load data
    df = load_psd_data('../data/norm_samples/unknown_sample.csv')
    df = df[validate_events(df)[0]]
    df = calculate_psd_ratio(df)
    
    # Predict
    predictions, probabilities = clf.predict(df)
    
    df['PARTICLE_ML'] = ['neutron' if p==1 else 'gamma' for p in predictions]
    
    # Save
    df.to_csv('../results/classified_data.csv', index=False)
    
    print(f"✓ Classified {len(df)} events")
    print(f"  Neutrons: {(predictions==1).sum()}")
    print(f"  Gammas: {(predictions==0).sum()}")
```

### Step 6: Create README

Copy Document "README - Complete PSD Analysis Toolkit with ML" to **README.md**

---

## Quick Verification

After setup, verify the structure works:

```python
# Test imports
from psd_analysis import (
    load_psd_data,
    calculate_psd_ratio,
    plot_psd_scatter
)

from psd_analysis.ml import ClassicalMLClassifier

print("✓ All imports successful!")
```

---

## Summary

**You have 5 code documents:**

1. **"PSD Analysis Starter Code"** → Split into psd_analysis/ modules
2. **"Missing Modules & ML"** → efficiency.py, optimization.py, spectrum.py
3. **"Machine Learning Module"** → psd_analysis/ml/
4. **"ML Example"** → scripts/ml_analysis_example.py
5. **"Jupyter Notebook"** → notebooks/01_psd_analysis.ipynb

**Action items:**

1. ✅ Create folder structure
2. ✅ Create all __init__.py files
3. ✅ Copy code sections to proper files
4. ✅ Create config files
5. ✅ Create example scripts
6. ✅ Test imports

**Then you can:**
- Run `python scripts/quick_analysis.py` for basic analysis
- Run `python scripts/ml_train.py` to train ML models
- Run `python scripts/ml_predict.py` to classify unknown data
- Open `notebooks/01_psd_analysis.ipynb` for interactive analysis

---

## File-by-File Checklist

### Core Library Files

- [ ] `psd_analysis/__init__.py`
- [ ] `psd_analysis/io/__init__.py`
- [ ] `psd_analysis/io/data_loader.py` (from Doc 1, Section 1)
- [ ] `psd_analysis/io/quality_control.py` (from Doc 1, Section 2)
- [ ] `psd_analysis/calibration/__init__.py`
- [ ] `psd_analysis/calibration/energy_cal.py` (from Doc 1, Section 3)
- [ ] `psd_analysis/calibration/efficiency.py` (from Doc 2)
- [ ] `psd_analysis/psd/__init__.py`
- [ ] `psd_analysis/psd/parameters.py` (from Doc 1, Section 4)
- [ ] `psd_analysis/psd/discrimination.py` (from Doc 1, Section 5)
- [ ] `psd_analysis/psd/optimization.py` (from Doc 2)
- [ ] `psd_analysis/spectroscopy/__init__.py`
- [ ] `psd_analysis/spectroscopy/spectrum.py` (from Doc 2)
- [ ] `psd_analysis/spectroscopy/peak_finding.py` (from Doc 1, Section 6)
- [ ] `psd_analysis/spectroscopy/isotope_id.py` (from Doc 1, Section 7)
- [ ] `psd_analysis/visualization/__init__.py`
- [ ] `psd_analysis/visualization/plots.py` (from Doc 1, Section 8)
- [ ] `psd_analysis/ml/__init__.py`
- [ ] `psd_analysis/ml/classical.py` (from Doc 3, part 1)
- [ ] `psd_analysis/ml/deep_learning.py` (from Doc 3, part 2)
- [ ] `psd_analysis/utils/__init__.py`
- [ ] `psd_analysis/utils/isotope_library.py` (extract ISOTOPE_LIBRARY dict)

### Scripts

- [ ] `scripts/quick_analysis.py`
- [ ] `scripts/ml_train.py`
- [ ] `scripts/ml_predict.py`
- [ ] `scripts/ml_analysis_example.py` (from Doc 4)
- [ ] `scripts/batch_process.py`

### Notebooks

- [ ] `notebooks/01_psd_analysis.ipynb` (from Doc 5)
- [ ] `notebooks/02_calibration.ipynb` (create from template)
- [ ] `notebooks/03_ml_classification.ipynb` (create from template)
- [ ] `notebooks/04_deep_learning.ipynb` (create from template)

### Configuration

- [ ] `config/detector_config.yaml`
- [ ] `config/ml_config.yaml`
- [ ] `requirements.txt`
- [ ] `README.md` (from README document)

### Data Directories

- [ ] `data/calibration/` (empty, for your data)
- [ ] `data/norm_samples/` (empty, for your data)
- [ ] `data/processed/` (empty, for processed outputs)
- [ ] `models/` (empty, for trained models)
- [ ] `results/figures/` (empty, for plots)
- [ ] `results/reports/` (empty, for reports)
- [ ] `results/exports/` (empty, for exported data)

---

## Detailed Code Extraction Guide

### Example: Extracting from Document 1

**Document 1 structure:**
```
# ============================================================================
# 1. DATA LOADING & QUALITY CONTROL
# ============================================================================

def load_psd_data(filename, delimiter=';'):
    ...

def validate_events(df, adc_min=0, adc_max=16383):
    ...

# ============================================================================
# 2. ENERGY CALIBRATION
# ============================================================================

def find_compton_edge(spectrum, energy_bins, expected_edge_keV):
    ...

def calibrate_energy(df, calibration_points, method='linear'):
    ...

# ... and so on
```

**Extract to files:**

1. **psd_analysis/io/data_loader.py**
   - Copy everything from Section 1 related to `load_psd_data()`
   - Add necessary imports at top
   
2. **psd_analysis/io/quality_control.py**
   - Copy `validate_events()` from Section 1
   - Add necessary imports

3. **psd_analysis/calibration/energy_cal.py**
   - Copy Section 2: `find_compton_edge()` and `calibrate_energy()`
   - Add necessary imports

Continue this pattern for all sections.

---

## Common Pitfalls to Avoid

### ❌ Wrong: Keep everything in one file
```python
# psd_analysis.py (BAD - monolithic file)
def load_psd_data(...):
    pass
def validate_events(...):
    pass
# ... 2000 lines of code ...
```

### ✅ Right: Organized modules
```python
# psd_analysis/io/data_loader.py (GOOD)
def load_psd_data(...):
    pass

# psd_analysis/io/quality_control.py (GOOD)
def validate_events(...):
    pass
```

### ❌ Wrong: Missing __init__.py files
```
psd_analysis/
├── io/
│   ├── data_loader.py
│   └── quality_control.py  # Can't import without __init__.py!
```

### ✅ Right: Proper __init__.py
```
psd_analysis/
├── io/
│   ├── __init__.py  # Makes it a package!
│   ├── data_loader.py
│   └── quality_control.py
```

### ❌ Wrong: Circular imports
```python
# parameters.py
from .discrimination import apply_discrimination  # BAD!

# discrimination.py
from .parameters import calculate_psd_ratio  # Creates circular import!
```

### ✅ Right: Proper import hierarchy
```python
# parameters.py (lower level)
def calculate_psd_ratio(df):
    pass

# discrimination.py (higher level)
from .parameters import calculate_psd_ratio  # GOOD!
```

---

## Testing Your Setup

### Test 1: Basic Imports
```python
# test_imports.py
import sys
sys.path.append('.')

try:
    from psd_analysis import load_psd_data
    print("✓ load_psd_data imported")
except ImportError as e:
    print(f"✗ Failed to import load_psd_data: {e}")

try:
    from psd_analysis.ml import ClassicalMLClassifier
    print("✓ ClassicalMLClassifier imported")
except ImportError as e:
    print(f"✗ Failed to import ClassicalMLClassifier: {e}")

try:
    from psd_analysis.ml import DeepPSDClassifier
    print("✓ DeepPSDClassifier imported (PyTorch available)")
except ImportError:
    print("⚠ DeepPSDClassifier not available (PyTorch not installed)")

print("\nSetup verification complete!")
```

### Test 2: Run Simple Analysis
```python
# test_analysis.py
from psd_analysis import load_psd_data, validate_events, calculate_psd_ratio
import numpy as np

# Create dummy data
dummy_data = {
    'BOARD': [0] * 100,
    'CHANNEL': [0] * 100,
    'TIMETAG': range(100),
    'ENERGY': np.random.randint(1000, 3000, 100),
    'ENERGYSHORT': np.random.randint(100, 500, 100),
    'FLAGS': ['0x0'] * 100,
    'PROBE_CODE': [1] * 100
}

import pandas as pd
df = pd.DataFrame(dummy_data)

# Test functions
df = calculate_psd_ratio(df)
print("✓ PSD calculated")

assert 'PSD' in df.columns
print("✓ PSD column exists")

print("\nFunctional test passed!")
```

### Test 3: ML Test (if PyTorch available)
```python
# test_ml.py
from psd_analysis.ml import ClassicalMLClassifier
import pandas as pd
import numpy as np

# Create synthetic training data
n_samples = 1000
df_train = pd.DataFrame({
    'ENERGY': np.random.randint(1000, 3000, n_samples),
    'ENERGYSHORT': np.random.randint(100, 500, n_samples),
    'PSD': np.random.rand(n_samples)
})

# Create labels (based on PSD for synthetic data)
df_train['PARTICLE'] = ['neutron' if p > 0.5 else 'gamma' for p in df_train['PSD']]

# Train
clf = ClassicalMLClassifier(method='logistic')  # Fast for testing
results = clf.train(df_train, test_size=0.3)

print(f"✓ Model trained")
print(f"  Validation accuracy: {results['val_accuracy']:.3f}")

# Predict
predictions, probs = clf.predict(df_train.head(10))
print(f"✓ Predictions made: {predictions}")

print("\nML test passed!")
```

---

## Alternative: Simplified Single-File Start

If the modular structure is too complex initially, start with a simplified version:

### Minimal Setup (3 files)

```
my_psd_analysis/
├── psd_core.py          # All core functions (Document 1)
├── psd_ml.py            # All ML code (Documents 2-3)
└── analysis.ipynb       # Jupyter notebook (Document 5)
```

**psd_core.py:**
```python
"""All PSD analysis functions in one file"""

# Copy entire Document 1 here
# All functions available as: psd_core.load_psd_data(), etc.
```

**psd_ml.py:**
```python
"""All ML functions in one file"""

# Copy Documents 2 and 3 here
# All ML classes available as: psd_ml.ClassicalMLClassifier(), etc.
```

**analysis.ipynb:**
```python
# First cell
import psd_core
import psd_ml

# Then use directly
df = psd_core.load_psd_data('data.csv')
clf = psd_ml.ClassicalMLClassifier('random_forest')
```

**Pros:** Simple, works immediately
**Cons:** Less maintainable, harder to find functions, can't use selective imports

---

## Transition Path

**Phase 1: Single File (Week 1)**
- Get familiar with functions
- Understand workflows
- Run analyses

**Phase 2: Basic Modules (Week 2-3)**
- Split into 3-4 main modules
- Keep ML separate
- Basic __init__.py files

**Phase 3: Full Structure (Week 4+)**
- Complete modular organization
- Proper packaging
- Unit tests
- Documentation

---

## Final Checklist Before First Run

- [ ] All directories created
- [ ] All __init__.py files in place
- [ ] Core functions extracted from Document 1
- [ ] Missing modules added from Document 2
- [ ] ML modules added from Document 3
- [ ] At least one script/notebook ready
- [ ] requirements.txt created
- [ ] Can run: `python -c "import psd_analysis"`
- [ ] Can run: `python scripts/quick_analysis.py` (or equivalent)
- [ ] Can open Jupyter notebook

**If all checked, you're ready to analyze data!**

---

## Getting Help

**If imports fail:**
1. Check all __init__.py files exist
2. Verify you're running from project root
3. Check for typos in import statements
4. Ensure all dependencies installed: `pip install -r requirements.txt`

**If functions missing:**
1. Verify code copied to correct file
2. Check function is imported in __init__.py
3. Restart Python kernel/session

**If ML fails:**
1. Check scikit-learn installed: `pip install scikit-learn`
2. For deep learning, check PyTorch: `pip install torch`
3. Verify training data has PARTICLE column
4. Check for NaN values: `df.isna().sum()`

---

## You're All Set! 🎉

Your complete PSD analysis toolkit with state-of-the-art ML is ready to use.

**Next steps:**
1. Put your CSV data files in `data/calibration/` or `data/norm_samples/`
2. Run the example scripts or notebooks
3. Train your first ML model
4. Start analyzing!

**Happy analyzing!** 🔬☢️🤖