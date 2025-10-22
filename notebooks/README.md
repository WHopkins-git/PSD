# PSD Analysis Tutorial Notebooks

A comprehensive series of Jupyter notebooks teaching Pulse Shape Discrimination (PSD) analysis for neutron/gamma discrimination and isotope identification.

## 📚 Notebook Series Overview

This tutorial series provides a complete, hands-on guide to PSD analysis, from basic concepts to advanced techniques. Each notebook is self-contained with extensive explanations, well-commented code, and practical examples.

### Learning Path

```
Beginner     → 01, 02         (Basics & Calibration)
Intermediate → 03, 04, 05     (Features, ML, Isotopes)
Advanced     → 06, 07, 08     (Deep Learning, Characterization, Production)
```

## 📖 Notebooks

### 1. Basic PSD Workflow (`01_basic_psd_workflow.ipynb`)

**Topics Covered:**
- Physics of organic scintillation
- Singlet vs triplet state emission
- Bi-exponential pulse shapes
- Tail-to-total ratio calculation
- Figure of Merit (FoM)
- Basic neutron/gamma discrimination

**What You'll Learn:**
- Why neutrons and gammas produce different pulse shapes
- How to calculate PSD parameters
- How to visualize and separate particle types
- How to measure discrimination quality

**Key Outputs:**
- PSD scatter plots
- FoM calculations
- Discrimination thresholds

---

### 2. Energy Calibration (`02_energy_calibration.ipynb`)

**Topics Covered:**
- Linear calibration (single-point, two-point)
- Polynomial calibration (higher order)
- Spline calibration (non-linear response)
- Compton edge finding
- Known source calibration
- Calibration validation

**What You'll Learn:**
- How to convert ADC counts to energy (keV)
- Which calibration method to use when
- How to find Compton edges automatically
- How to validate calibration quality

**Key Outputs:**
- Calibration curves
- Energy spectra in keV
- Residual plots

---

### 3. Feature Extraction (`03_feature_extraction.ipynb`)

**Topics Covered:**
- 100+ timing features for ML
- Multiple charge ratios (7 gate configurations)
- Rise/fall time measurements
- Cumulative charge timestamps
- Bi-exponential decay fitting
- Time-over-threshold features
- Template matching
- Gatti optimal filter
- Wavelet and frequency features

**What You'll Learn:**
- How to extract informative features from waveforms
- Which features are most important for PSD
- How to optimize gate configurations
- Advanced signal processing techniques

**Key Outputs:**
- Feature importance rankings
- Feature correlation matrices
- 2D feature space visualizations

---

### 4. Machine Learning Classification (`04_ml_classification.ipynb`)

**Topics Covered:**
- Random Forest classifier
- Support Vector Machines (SVM)
- Gradient Boosting
- Neural Networks (MLP)
- Cross-validation strategies
- Hyperparameter tuning
- Model evaluation (ROC, confusion matrix)
- Feature importance analysis

**What You'll Learn:**
- How ML improves upon traditional PSD (95% → 99% accuracy)
- Which classifier works best for PSD
- How to avoid overfitting
- How to deploy trained models

**Key Outputs:**
- Trained classifiers (>99% accuracy)
- ROC curves (AUC > 0.99)
- Feature importance plots

---

### 5. Isotope Identification (`05_isotope_identification.ipynb`)

**Topics Covered:**
- Gamma spectroscopy basics
- Peak finding algorithms
- Gaussian peak fitting
- Isotope library matching
- Decay chain analysis
- NORM (Naturally Occurring Radioactive Material) detection
- Confidence scoring

**What You'll Learn:**
- How to identify isotopes from energy spectra
- Which isotopes are common in NORM
- How to detect uranium and thorium decay chains
- How to distinguish natural vs artificial sources

**Key Outputs:**
- Identified isotopes with confidence scores
- Annotated energy spectra
- Decay chain reports

**Common Isotopes:**
- **K-40**: 1461 keV (natural background)
- **U-238 chain**: Pb-214 (295, 352 keV), Bi-214 (609, 1764 keV)
- **Th-232 chain**: Ac-228 (911 keV), Tl-208 (2614 keV)
- **Cs-137**: 662 keV (medical, fallout)
- **Co-60**: 1173, 1332 keV (industrial)

---

### 6. Deep Learning for PSD (`06_deep_learning.ipynb`)

**Topics Covered:**
- 1D Convolutional Neural Networks (CNN)
- Transformer architectures
- Physics-informed loss functions
- Training with PyTorch
- GPU acceleration
- Model interpretation
- Comparison with classical ML

**What You'll Learn:**
- How deep learning learns features automatically
- When to use CNN vs Transformer
- How to incorporate physics constraints
- How to achieve state-of-the-art accuracy (99%+)

**Requirements:**
```bash
pip install torch torchvision
```

**Key Outputs:**
- Trained CNN model (~98% accuracy)
- Trained Transformer (~99%+ accuracy)
- Learned filter visualizations

---

### 7. Scintillator Characterization (`07_scintillator_characterization.ipynb`)

**Topics Covered:**
- Light yield measurement
- Decay time constant fitting
- Energy resolution calculation
- PSD Figure of Merit
- Birks' law quenching
- Detector comparison
- Selection guide

**What You'll Learn:**
- How to characterize organic scintillators
- Which properties matter for PSD
- How to compare different scintillators
- Non-linear light output effects

**Scintillators Compared:**
- **EJ-276** (plastic, easy to shape)
- **EJ-309** (liquid, high light yield)
- **Stilbene** (crystal, best PSD)
- **CLYC** (dual neutron/gamma, poor PSD)

**Key Outputs:**
- Decay constant measurements
- FoM calculations
- Comparison tables

---

### 8. Advanced Techniques (`08_advanced_techniques.ipynb`)

**Topics Covered:**
- Real-time processing pipeline
- FPGA-compatible algorithms
- Fixed-point arithmetic
- Physics-informed machine learning
- Uncertainty quantification
- Multi-detector coincidence
- Adaptive thresholds
- Production deployment

**What You'll Learn:**
- How to process waveforms in real-time (<100 ns)
- How to design FPGA/firmware implementations
- How to incorporate physics knowledge into ML
- How to build multi-detector systems

**Key Outputs:**
- Real-time processing engine
- FPGA algorithm designs
- Production-ready classifiers

---

## 🚀 Getting Started

### Prerequisites

```bash
# Required
pip install numpy pandas matplotlib scipy scikit-learn seaborn

# Optional (for notebook 6)
pip install torch torchvision

# Optional (for interactive plots)
pip install plotly ipywidgets
```

### Quick Start

1. **Clone the repository** (if not already done)
2. **Navigate to notebooks directory:**
   ```bash
   cd notebooks
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Start with notebook 01** and work through the series

### Using Your Own Data

Each notebook includes synthetic data generation for learning purposes. To use your own data:

1. **Modify the data loading section** in each notebook
2. **Ensure your data has these columns:**
   - `WAVEFORM`: Array of ADC values (e.g., 500 samples)
   - `ENERGY_ADC`: Integrated charge (ADC counts)
   - `PARTICLE`: Labels ('neutron' or 'gamma') for training
   - `TIMESTAMP`: Event time (optional)

3. **Example data format:**
   ```python
   df = pd.DataFrame({
       'WAVEFORM': [np.array([100, 95, 90, ...]), ...],  # 500 samples each
       'ENERGY_ADC': [1234, 5678, ...],
       'PARTICLE': ['neutron', 'gamma', ...]
   })
   ```

---

## 📊 Expected Results

### Performance Metrics

| Method | Accuracy | Inference Time | Training Data | Hardware |
|--------|----------|----------------|---------------|----------|
| **Traditional PSD** | ~95% | <1 µs | None | Simple threshold |
| **Random Forest** | ~97-98% | ~10 µs | 1,000+ events | CPU |
| **CNN** | ~98-99% | ~1 ms | 10,000+ events | GPU (training) |
| **Transformer** | ~99%+ | ~5 ms | 10,000+ events | GPU (training) |

### When to Use Each Method

**Traditional PSD (tail-to-total):**
- ✅ Real-time FPGA implementation
- ✅ Limited training data
- ✅ Simple setup
- ❌ Lower accuracy (~95%)

**Random Forest:**
- ✅ Good balance of speed and accuracy
- ✅ Works with moderate data (1,000+ events)
- ✅ Interpretable feature importance
- ❌ Requires feature engineering

**Deep Learning (CNN/Transformer):**
- ✅ Highest accuracy (99%+)
- ✅ No feature engineering
- ✅ Learns complex patterns
- ❌ Requires large dataset (10,000+ events)
- ❌ Slower inference
- ❌ Less interpretable

---

## 🎯 Applications

### Neutron Detection
- **Homeland Security**: Border radiation portals
- **Nuclear Safeguards**: Special nuclear material detection
- **Oil & Gas**: Well logging, corrosion detection
- **Research**: Fast neutron spectrometry

### Isotope Identification
- **NORM Detection**: Uranium/thorium in minerals
- **Environmental Monitoring**: Contamination mapping
- **Medical**: Radiopharmaceutical QA
- **Security**: Orphan source detection

### Detector Characterization
- **R&D**: New scintillator development
- **Quality Control**: Production testing
- **Calibration**: Detector commissioning
- **Diagnostics**: Performance degradation monitoring

---

## 📖 Additional Resources

### PSD Physics
- Knoll, G. F. (2010). *Radiation Detection and Measurement*
- Brooks, F. D. (1979). "Development of organic scintillators"
- Zaitseva, N. et al. (2018). "Pulse shape discrimination with organic scintillators"

### Machine Learning for Nuclear
- Kamuda, M. & Sullivan, C. J. (2019). "An automated isotope identification method"
- Peurrung, A. J. (2000). "Recent developments in neutron detection"

### Scintillator Properties
- EJ-276 data sheet (Eljen Technology)
- EJ-309 data sheet (Eljen Technology)
- Saint-Gobain detector catalog

---

## 🛠️ Troubleshooting

### Common Issues

**Problem**: Notebooks won't run
- **Solution**: Install all required packages (`pip install -r requirements.txt`)

**Problem**: Low FoM values (<1.0)
- **Solution**: Check data quality, ensure proper baseline correction, verify gate timings

**Problem**: ML models overfit (train acc >> test acc)
- **Solution**: Use more training data, apply regularization, use GroupKFold validation

**Problem**: Energy calibration fails
- **Solution**: Ensure clean photopeaks, check for saturation, use multiple calibration points

**Problem**: PyTorch not found (notebook 06)
- **Solution**: `pip install torch torchvision` (optional for other notebooks)

---

## 📝 Citation

If you use these notebooks in your research, please cite:

```bibtex
@misc{psd_analysis_notebooks,
  title={PSD Analysis Tutorial Notebooks},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/PSD}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests/examples
5. Submit a pull request

---

## 📧 Contact

For questions, issues, or suggestions:
- **Open an issue** on GitHub
- **Email**: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🎓 Learning Outcomes

After completing this tutorial series, you will be able to:

✅ Understand the physics of pulse shape discrimination
✅ Perform energy calibration for organic scintillators
✅ Extract and engineer features from waveforms
✅ Train and deploy ML classifiers for PSD
✅ Identify radioactive isotopes from gamma spectra
✅ Use deep learning for advanced PSD
✅ Characterize scintillator detector properties
✅ Design real-time processing systems
✅ Deploy production-ready PSD systems

---

**Happy Learning! 🎉**

*Last updated: 2025-10-22*
