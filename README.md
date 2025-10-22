# PSD Analysis Toolkit

Complete toolkit for Pulse Shape Discrimination (PSD) analysis, neutron/gamma discrimination, and NORM (Naturally Occurring Radioactive Material) source identification.

## Features

- **Data Loading & Quality Control**: Load and validate PSD data from various formats
- **Energy Calibration**: Calibrate detector energy response using known sources
- **PSD Discrimination**: Separate neutron and gamma events using pulse shape analysis
- **Spectroscopy**: Peak finding and isotope identification
- **Machine Learning**: Advanced classification using classical and deep learning methods
- **Feature Extraction**: Extract advanced timing and shape features from waveforms
- **Scintillator Characterization**: Database and tools for scintillator selection

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/WHopkins-git/PSD.git
cd PSD

# Install the package
pip install -e .
```

### Installation with Optional Dependencies

```bash
# Install with deep learning support
pip install -e ".[deep_learning]"

# Install with Jupyter notebook support
pip install -e ".[notebooks]"

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from psd_analysis import (
    load_psd_data,
    validate_events,
    calculate_psd_ratio,
    calibrate_energy,
    find_peaks_in_spectrum
)

# Load data
df = load_psd_data('your_data.csv')

# Quality control
valid_mask, qc_report = validate_events(df)
df_clean = df[valid_mask]

# Calculate PSD parameter
df_clean = calculate_psd_ratio(df_clean)

# Energy calibration
calibration_points = [(adc1, keV1), (adc2, keV2)]
df_clean, cal_func, params = calibrate_energy(df_clean, calibration_points)

# Find peaks for isotope identification
peaks, counts, properties = find_peaks_in_spectrum(
    df_clean['ENERGY_KEV'],
    df_clean['ENERGY_KEV'].value_counts()
)
```

## Project Structure

```
PSD/
├── psd_analysis/           # Main package
│   ├── io/                # Data loading and quality control
│   ├── calibration/       # Energy calibration
│   ├── psd/              # PSD calculation and discrimination
│   ├── features/         # Feature extraction
│   ├── spectroscopy/     # Peak finding and isotope ID
│   ├── ml/               # Machine learning
│   ├── utils/            # Utilities and constants
│   └── visualization/    # Plotting functions
├── scripts/              # Example scripts
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
├── data/                 # Data directories
├── models/               # Trained models
├── config/               # Configuration files
└── tests/                # Unit tests
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **User Guide**: Getting started and basic usage
- **API Reference**: Detailed function documentation
- **Feature Engineering**: Advanced feature extraction guide
- **ML Guide**: Machine learning classifier guide
- **File Organization**: Project structure and organization

## Examples

### Basic PSD Analysis

```python
from psd_analysis import *

# Load and process data
df = load_psd_data('calibration_source.csv')
df = df[validate_events(df)[0]]
df = calculate_psd_ratio(df)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(df['ENERGY'], df['PSD'], alpha=0.3)
plt.xlabel('Energy (ADC)')
plt.ylabel('PSD')
plt.show()
```

### Machine Learning Classification

```python
from psd_analysis.ml.classical import ClassicalMLClassifier

# Train classifier
clf = ClassicalMLClassifier(method='random_forest')
results = clf.train(df_labeled, test_size=0.2)

# Predict on new data
predictions, probabilities = clf.predict(df_unknown)
```

### Advanced Feature Extraction

```python
from psd_analysis.features.timing_v2 import EnhancedTimingFeatureExtractor

# Extract comprehensive features
extractor = EnhancedTimingFeatureExtractor()
features = extractor.extract_all_features(waveform_samples)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```
PSD Analysis Toolkit (2024)
https://github.com/WHopkins-git/PSD
```

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

This toolkit was developed for radiation detection and nuclear security applications.
