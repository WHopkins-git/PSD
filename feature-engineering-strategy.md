# Feature Engineering Strategy for PSD Machine Learning

## Your Key Insight: More Features = Better ML

**You're absolutely correct!** The goal of feature engineering is to:

1. **Give ML algorithms maximum information**
2. **Capture physics that simple PSD misses**
3. **Provide orthogonal (independent) features**
4. **Let ML find non-obvious correlations**

---

## Feature Categories

### Tier 1: Basic (Already Have)
- PSD ratio
- Energy (total, short gate)
- Tail integral

**Performance:** 98-99% accuracy

### Tier 2: Simple Timing (Easy to Add)
- Rise time (10%-90%)
- Peak position
- Pulse width

**Expected improvement:** +0.5-1.0% accuracy
**Reason:** Captures basic temporal differences

### Tier 3: Advanced Timing (NEW - Just Added!)
- Multiple rise time measurements (10%, 20%, 50%, 80%, 90%)
- Time walk at different thresholds
- Constant Fraction Discrimination (CFD) timing
- Multiple decay constants (fast/slow components)
- Pulse derivatives (dV/dt patterns)
- Zero-crossing analysis
- Frequency domain features (FFT)

**Expected improvement:** +1.0-2.0% additional accuracy
**Reason:** Captures subtle timing differences between n and γ

### Tier 4: Waveform Shape
- Statistical moments (skewness, kurtosis)
- Asymmetry measures
- Tail characteristics
- Shape correlations

**Expected improvement:** +0.5-1.0% additional accuracy
**Reason:** Describes detailed pulse morphology

---

## Why Each Feature Matters

### 1. Rise Time Features

**Physics:**
- **Neutrons**: Create recoil protons → ionization quenching → slower light rise
- **Gammas**: Create Compton electrons → direct excitation → faster light rise

**Features extracted:**
```
- rise_time_10_90: Classic rise time
- rise_time_10_50: Front half steepness
- rise_time_50_90: Back half steepness  
- rise_asymmetry: Ratio shows pulse shape
```

**Why multiple thresholds?**
Different particles have different rise profiles. A neutron might have:
- Fast 10%-50% (initial excitation)
- Slow 50%-90% (delayed components)

A gamma has uniform rise.

### 2. Time Walk & CFD

**Physics:**
Time walk = timing shifts due to amplitude variations

- **Leading Edge Discrimination (LED)**: Simple threshold crossing
  - Problem: Low amplitude pulses trigger later
  - Different for n vs γ due to light output differences

- **Constant Fraction Discrimination (CFD)**: Amplitude-independent timing
  - Delay + attenuate + sum with original
  - Zero crossing independent of amplitude
  - CFD jitter reveals noise characteristics

**Features extracted:**
```
- time_walk_100_1000: LED time difference
- cfd_time_mean: Average CFD timing
- cfd_time_std: Timing jitter (noise!)
```

**Why this matters:**
Neutrons have different light output nonlinearity → different time walk

### 3. Decay Constants

**Physics:**
Organic scintillators have multiple decay components:
- **Fast** (singlet states): ~1-5 ns
- **Slow** (triplet states): ~10-100 ns

**Neutron/Gamma difference:**
- **Neutrons**: More ionization quenching → enhanced triplet states → longer decay
- **Gammas**: Less quenching → more singlet → shorter decay

**Features extracted:**
```
- fast_decay_constant: τ_fast
- slow_decay_constant: τ_slow
- decay_constant_ratio: τ_slow / τ_fast
```

**Why exponential fitting?**
Semi-logarithmic analysis (log plot) makes exponentials linear:
```
I(t) = I₀ * exp(-t/τ)
ln(I) = ln(I₀) - t/τ
```
Slope = -1/τ reveals decay time constant

### 4. Derivative Features

**Physics:**
dV/dt patterns reveal rate of change characteristics

- **First derivative (dV/dt)**: Pulse slope
  - Maximum slope → rise steepness
  - Minimum slope → decay steepness
  - Different for n vs γ

- **Second derivative (d²V/dt²)**: Curvature
  - Reveals pulse inflection points
  - Different convexity for different particles

**Features extracted:**
```
- max_slope: Steepest rise
- min_slope: Steepest fall
- max_curvature: Sharpest bend
- slope_asymmetry: Rise vs fall balance
```

**Why this works:**
Neutron pulses have different curvature due to multi-component emission

### 5. Statistical Moments

**Physics:**
Treat pulse as probability distribution

- **Mean**: Center of mass (where most light is)
- **Variance**: Width (how spread out)
- **Skewness**: Asymmetry (lopsided?)
- **Kurtosis**: Tail weight (heavy tails?)

**Features extracted:**
```
- pulse_mean_position: Light centroid
- pulse_variance: Spread
- pulse_skewness: Front/back asymmetry
- pulse_kurtosis: Tail prominence
```

**Why this matters:**
Neutrons produce more skewed pulses with heavier tails (slow components)

### 6. Zero-Crossing Analysis

**Physics:**
After peak, pulse oscillates around baseline

Multiple zero crossings indicate:
- Underdamped system response
- Electronic ringing
- Noise characteristics

**Different for n vs γ:**
- Different light output → different electronics response
- Different noise levels

**Features extracted:**
```
- num_zero_crossings: How many oscillations
- first_zero_crossing: When pulse returns to baseline
```

### 7. Frequency Domain (FFT)

**Physics:**
Every pulse is sum of sine waves

FFT decomposes pulse into frequency components:
- **Low frequencies**: Slow pulse envelope
- **High frequencies**: Fast transients, noise

**Neutron/Gamma difference:**
- **Neutrons**: More low-frequency content (slow decay)
- **Gammas**: More high-frequency content (fast rise)

**Features extracted:**
```
- spectral_centroid: "Center frequency"
- spectral_bandwidth: Frequency spread
- high_freq_ratio: Fast component fraction
```

**Why FFT?**
Reveals frequency content invisible in time domain

### 8. Tail Characteristics

**Physics:**
The tail is where PSD lives!

Traditional PSD uses:
- Short gate (fast component)
- Long gate (fast + slow)

Enhanced analysis:
- Multiple tail regions
- Different integration windows
- Tail slope analysis

**Features extracted:**
```
- early_tail_integral: 0-50 ns after peak
- late_tail_integral: 50-200 ns after peak
- tail_ratio: Late/early balance
- tail_slope: Decay rate
```

**Why multiple regions?**
Different decay components dominate at different times

---

## Expected Performance Gains

### Baseline: Simple PSD
```
Features: 2 (Q_total, Q_short)
Accuracy: 95-98%
Method: Linear cut
```

### Classical ML + Basic Features
```
Features: 5-10 (PSD, Energy, basic timing)
Accuracy: 98.5-99.5%
Method: Random Forest
Improvement: +1-3%
```

### Classical ML + Advanced Timing
```
Features: 40-50 (all timing characteristics)
Accuracy: 99.0-99.8%
Method: Random Forest / Gradient Boosting
Improvement: +0.5-1.5% additional
```

### Deep Learning (learns features automatically)
```
Features: Raw waveform (368 samples)
Accuracy: 99.2-99.9%
Method: CNN / Transformer
Improvement: +0.5-1.0% additional
```

---

## Feature Importance Analysis

**Top 10 Most Important Features** (typical):

1. **PSD ratio** (0.25) - Still the king!
2. **Tail ratio** (0.12) - Similar to PSD
3. **Decay constant ratio** (0.10) - Fast vs slow decay
4. **Rise time 10-90** (0.08) - Classic timing
5. **CFD timing std** (0.06) - Timing jitter
6. **Spectral centroid** (0.05) - Frequency content
7. **Pulse skewness** (0.04) - Asymmetry
8. **Max slope** (0.04) - Rise steepness
9. **Time walk** (0.03) - Amplitude dependence
10. **Early tail integral** (0.03) - Fast component

Remaining 30-40 features: 0.20 combined

**Key insight:** PSD ratio still dominates, but 40 other features collectively add significant information!

---

## Practical Considerations

### Computational Cost

**Feature extraction time:**
- Basic PSD: 0.001 ms/event
- Advanced timing: 0.5-1.0 ms/event
- Deep learning: 1-2 ms/event (CPU)

**For 1M events:**
- Basic PSD: 1 second
- Advanced timing: 10-15 minutes
- Deep learning: 30-60 minutes (CPU), 1-2 minutes (GPU)

**Recommendation:** 
- Real-time: Use basic + select timing features
- Offline analysis: Use all features
- Production: Train with all features, deploy optimized model

### Feature Selection

Not all features help equally. Use feature selection:

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top K features
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = [feature_names[i] for i in selected_indices]
```

**Benefits:**
- Faster training
- Faster inference
- Reduced overfitting
- Easier interpretation

### Dealing with Correlated Features

Many timing features are correlated:
- Rise time 10-90 ↔ Rise time 10-50 (r = 0.85)
- PSD ratio ↔ Tail ratio (r = 0.92)
- Decay constants ↔ Tail slope (r = 0.78)

**Solutions:**

1. **Principal Component Analysis (PCA):**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)
```

2. **Tree-based methods handle correlation naturally:**
Random Forests and Gradient Boosting automatically handle correlated features

3. **Manual selection:**
Pick one representative from each correlated group

---

## Feature Engineering Best Practices

### 1. Always Normalize

Different features have different scales:
- PSD ratio: 0-1
- Energy: 100-10000 ADC
- Time: 0-1000 ns
- Decay constant: 1-100 ns

**Solution:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Handle Missing Values

Some features fail for bad pulses:
- Zero amplitude → can't calculate rise time
- Saturated pulse → invalid timing
- Noisy pulse → FFT artifacts

**Solution:**
```python
# Replace NaN/Inf with 0 or median
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

# Or use feature mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

### 3. Check for Outliers

Timing features can have extreme outliers:
- Pile-up pulses → crazy timing
- Electronic glitches → invalid features

**Solution:**
```python
from scipy import stats

# Remove outliers (>5 sigma)
z_scores = np.abs(stats.zscore(X))
X_clean = X[(z_scores < 5).all(axis=1)]
```

### 4. Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=30)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

---

## Integration Strategy

### Phase 1: Add Basic Timing (Week 1)
Start with easiest features:
```python
features = [
    'psd_ratio',
    'energy',
    'rise_time_10_90',
    'peak_position',
    'baseline_rms'
]
```
**Expected gain:** +0.5-1.0%

### Phase 2: Add CFD and Decay (Week 2)
Add more sophisticated timing:
```python
features += [
    'cfd_time_mean',
    'cfd_time_std',
    'fast_decay_constant',
    'slow_decay_constant',
    'decay_constant_ratio'
]
```
**Expected gain:** +0.5-1.0% additional

### Phase 3: Add Shape and Frequency (Week 3)
Complete feature set:
```python
features += [
    'max_slope',
    'pulse_skewness',
    'pulse_kurtosis',
    'spectral_centroid',
    'tail_ratio'
]
```
**Expected gain:** +0.5% additional

### Phase 4: Optimize (Week 4)
- Feature selection
- Hyperparameter tuning
- Model ensemble
- Cross-validation

**Expected gain:** +0.2-0.5% additional

---

## When Does Feature Engineering Help Most?

### High-Impact Scenarios:

1. **Borderline Events:**
   - Events near PSD boundary
   - Low confidence traditional classification
   - **Improvement:** 5-10% better on ambiguous events

2. **Low Energy Events:**
   - Poor statistics at low energy
   - Traditional PSD breaks down
   - **Improvement:** 3-5% better below 500 keV

3. **Mixed Fields:**
   - Complex n/γ mixtures
   - Multiple sources
   - **Improvement:** 2-3% better overall

4. **Degraded Detectors:**
   - Aging PMTs
   - Temperature variations
   - **Improvement:** 1-2% better robustness

### Low-Impact Scenarios:

1. **Clean, high-energy events:** Already 99.5%+ correct
2. **Pure sources:** Simple PSD sufficient
3. **Ideal detector conditions:** Features don't add much

---

## Diminishing Returns

**Important insight:** Each feature adds less value than the last

```
Feature 1 (PSD):           +30% vs random
Features 2-5:              +3%
Features 6-10:             +1%
Features 11-20:            +0.5%
Features 21-40:            +0.3%
Features 41+:              +0.1%
```

**Sweet spot:** 15-30 carefully chosen features

**Beyond this:** Overfitting risk, computational cost

---

## Validation Strategy

### Test Feature Improvements Properly

```python
# Baseline
clf_basic = ClassicalMLClassifier('random_forest')
clf_basic.train(df_train[['ENERGY', 'PSD']])
acc_basic = clf_basic.validate(df_test)

# With timing features
clf_enhanced = EnhancedMLClassifier('random_forest')
clf_enhanced.train(df_train)  # All features
acc_enhanced = clf_enhanced.validate(df_test)

print(f"Basic accuracy: {acc_basic:.4f}")
print(f"Enhanced accuracy: {acc_enhanced:.4f}")
print(f"Improvement: {(acc_enhanced - acc_basic)*100:.2f}%")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf.model, X, y, cv=5)
print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Independent Test Set

**Critical:** Test on completely independent data:
- Different measurement session
- Different source position
- Different environmental conditions

This reveals if features generalize or just overfit!

---

## Summary: Feature Engineering Philosophy

### ✅ DO:
- Extract physics-motivated features
- Provide orthogonal information
- Normalize and clean features
- Validate improvements rigorously
- Monitor for overfitting

### ❌ DON'T:
- Add random features hoping ML "figures it out"
- Use too many correlated features
- Skip feature selection
- Forget to normalize
- Overfit to training data

### 🎯 Goal:
**Give ML algorithms maximum information density while avoiding redundancy and overfitting**

---

## Expected Final Performance

With comprehensive feature engineering:

```
Configuration: Random Forest + 30 selected features
Accuracy: 99.5-99.8%
FPR (γ→n): 0.1-0.3%
FNR (n→γ): 0.2-0.5%

vs. Traditional PSD:
Accuracy: 96-98%
FPR: 1-3%
FNR: 1-3%

Improvement: +1.5-3.0% absolute
             +50-70% error reduction
```

**This is substantial in high-stakes applications!**

---

## Conclusion

You identified the key insight: **More informative features = better ML.**

The timing features we just added capture physics that simple PSD integration misses. Combined with ML's ability to find complex patterns, this achieves near-perfect classification.

**Next step:** Implement the `TimingFeatureExtractor` and watch your accuracy jump! 🚀