# PSD Analysis Notebooks

This folder contains example analysis notebooks demonstrating the PSD Analysis Toolkit functionality.

## Important Notes on Package Updates

The `psd_analysis` package has been reorganized and updated. The notebooks in this folder fall into two categories:

### 1. Tutorial Notebooks (Educational)

The numbered tutorial notebooks (`01_basic_psd_workflow.ipynb` through `08_advanced_techniques.ipynb`) contain **inline implementations of algorithms for educational purposes**. These implementations demonstrate the mathematical concepts and physics principles behind PSD analysis.

**Key Points:**
- These notebooks define functions locally to teach the underlying concepts
- The implementations are simplified for clarity and learning
- For production use, please use the functions from the `psd_analysis` package instead
- These notebooks remain valuable for understanding **how** the algorithms work

### 2. Package Usage Example

The `example_psd_analysis.ipynb` notebook demonstrates **how to use the actual psd_analysis package** for real-world analysis workflows.

**Key Points:**
- Imports functions from the installed `psd_analysis` package
- Shows production-ready workflows
- Demonstrates proper package usage patterns
- Use this as a template for your own analysis scripts

## Package Structure

The reorganized package provides the following modules:

```
psd_analysis/
├── io/                 # Data loading and quality control
├── calibration/        # Energy calibration  
├── psd/               # PSD calculation and discrimination
├── features/          # Advanced feature extraction
├── spectroscopy/      # Peak finding and isotope ID
├── ml/                # Machine learning
├── utils/             # Utilities and constants
└── visualization/     # Plotting functions
```

## Quick Start

For using the package in your own work:

```python
from psd_analysis import (
    load_psd_data,
    validate_events,
    calculate_psd_ratio,
    calibrate_energy,
    find_peaks_in_spectrum
)

# Load and process data
df = load_psd_data('your_data.csv')
valid_mask, qc_report = validate_events(df)
df_clean = df[valid_mask]

# Calculate PSD
df_clean = calculate_psd_ratio(df_clean)

# Energy calibration
calibration_points = [(500, 250), (2000, 1000)]  # (ADC, keV)
df_clean, cal_func, params = calibrate_energy(df_clean, calibration_points)
```

## Testing

The package has comprehensive test coverage:

- **test_suite.py**: Tests all core package functionality
- **stress_tests.py**: Tests edge cases, robustness, and performance

Run tests from the repository root:
```bash
python test_suite.py
python stress_tests.py
```

## Getting Help

- See the main README.md for installation and setup
- Check the `docs/` folder for detailed documentation
- Review `example_psd_analysis.ipynb` for package usage patterns
- Tutorial notebooks (01-08) explain the underlying algorithms

## Contributing

If you make improvements to the analysis methods, please:
1. Update the package modules in `psd_analysis/`
2. Add tests to `test_suite.py`
3. Update this documentation as needed

Tutorial notebooks can remain with inline code for educational purposes.
