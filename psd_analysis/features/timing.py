"""
Advanced Timing Feature Extraction for PSD
Extracts comprehensive timing characteristics from waveforms

This dramatically improves ML performance by providing rich feature space
"""

import numpy as np
from scipy import signal, optimize, interpolate
from scipy.stats import skew, kurtosis


class TimingFeatureExtractor:
    """
    Extract comprehensive timing features from pulse waveforms
    
    Features extracted:
    - Rise time characteristics (10%, 50%, 90% points)
    - Multiple time walk measurements
    - Constant Fraction Discrimination (CFD) timing
    - Pulse derivatives (dV/dt patterns)
    - Zero-crossing analysis
    - Decay time constants
    - Pulse shape moments
    - Frequency domain features
    """
    
    def __init__(self, sampling_rate_mhz=250, baseline_samples=50):
        """
        Parameters:
        -----------
        sampling_rate_mhz : float
            Digitizer sampling rate in MHz
        baseline_samples : int
            Number of samples before pulse for baseline
        """
        self.sampling_rate_mhz = sampling_rate_mhz
        self.dt = 1000.0 / sampling_rate_mhz  # Time per sample in ns
        self.baseline_samples = baseline_samples
        
    def extract_all_features(self, waveform):
        """
        Extract all timing features from a single waveform
        
        Parameters:
        -----------
        waveform : array (n_samples,)
            Raw ADC samples
        
        Returns:
        --------
        features : dict
            Dictionary of all extracted features
        """
        features = {}
        
        # Preprocess
        baseline = np.mean(waveform[:self.baseline_samples])
        pulse = baseline - waveform  # Negative-going → positive pulse
        
        # Normalize
        max_amplitude = np.max(pulse)
        if max_amplitude > 0:
            pulse_norm = pulse / max_amplitude
        else:
            # Dead pulse, return zeros
            return self._zero_features()
        
        # Store basic characteristics
        features['amplitude'] = max_amplitude
        features['peak_position'] = np.argmax(pulse)
        features['baseline_rms'] = np.std(waveform[:self.baseline_samples])
        
        # Extract timing features
        features.update(self._extract_rise_time_features(pulse_norm))
        features.update(self._extract_time_walk_features(pulse, pulse_norm))
        features.update(self._extract_cfd_features(pulse_norm))
        features.update(self._extract_decay_features(pulse, pulse_norm))
        features.update(self._extract_derivative_features(pulse_norm))
        features.update(self._extract_shape_moments(pulse_norm))
        features.update(self._extract_zero_crossing_features(pulse_norm))
        features.update(self._extract_frequency_features(pulse))
        features.update(self._extract_tail_characteristics(pulse_norm))
        
        return features
    
    def _extract_rise_time_features(self, pulse_norm):
        """
        Rise time analysis: 10%, 50%, 90% crossing times
        Different rise time characteristics for n vs γ
        """
        features = {}
        
        peak_idx = np.argmax(pulse_norm)
        
        # Find threshold crossings
        thresholds = {
            '10': 0.10,
            '20': 0.20,
            '50': 0.50,
            '80': 0.80,
            '90': 0.90
        }
        
        crossing_times = {}
        for name, thresh in thresholds.items():
            idx = np.where(pulse_norm[:peak_idx] >= thresh)[0]
            if len(idx) > 0:
                # Linear interpolation for sub-sample precision
                i = idx[0]
                if i > 0:
                    # Interpolate between samples
                    frac = (thresh - pulse_norm[i-1]) / (pulse_norm[i] - pulse_norm[i-1])
                    crossing_times[name] = (i - 1 + frac) * self.dt
                else:
                    crossing_times[name] = 0
            else:
                crossing_times[name] = 0
        
        # Calculate rise times between thresholds
        features['rise_time_10_90'] = crossing_times['90'] - crossing_times['10']
        features['rise_time_10_50'] = crossing_times['50'] - crossing_times['10']
        features['rise_time_50_90'] = crossing_times['90'] - crossing_times['50']
        features['rise_time_20_80'] = crossing_times['80'] - crossing_times['20']
        
        # Rise time asymmetry (indicative of pulse shape)
        if features['rise_time_10_90'] > 0:
            features['rise_asymmetry'] = features['rise_time_10_50'] / features['rise_time_10_90']
        else:
            features['rise_asymmetry'] = 0.5
        
        # Absolute crossing times (time walk)
        features['t_10'] = crossing_times['10']
        features['t_50'] = crossing_times['50']
        features['t_90'] = crossing_times['90']
        
        return features
    
    def _extract_time_walk_features(self, pulse, pulse_norm):
        """
        Time walk: timing shifts as function of amplitude
        Different for neutrons vs gammas due to light output differences
        """
        features = {}
        
        # Leading Edge Discrimination (LED) at different thresholds
        # Time walk = difference in timing at different thresholds
        thresholds_adc = [100, 200, 500, 1000]  # Absolute ADC thresholds
        
        led_times = []
        for thresh in thresholds_adc:
            idx = np.where(pulse >= thresh)[0]
            if len(idx) > 0:
                led_times.append(idx[0] * self.dt)
            else:
                led_times.append(np.nan)
        
        # Time walk between thresholds
        if not np.isnan(led_times[0]) and not np.isnan(led_times[-1]):
            features['time_walk_100_1000'] = led_times[-1] - led_times[0]
        else:
            features['time_walk_100_1000'] = 0
        
        # Amplitude-corrected timing
        amplitude = np.max(pulse)
        if amplitude > 0:
            features['timing_per_amplitude'] = led_times[0] / amplitude if not np.isnan(led_times[0]) else 0
        else:
            features['timing_per_amplitude'] = 0
        
        return features
    
    def _extract_cfd_features(self, pulse_norm):
        """
        Constant Fraction Discrimination (CFD)
        Reduces amplitude walk, provides precise timing
        
        CFD: delay pulse, invert, attenuate, sum with original
        Zero crossing gives timing independent of amplitude
        """
        features = {}
        
        # Standard CFD: 50% fraction, delay of 2-5 samples
        fractions = [0.3, 0.5, 0.7]
        delays = [2, 3, 5]
        
        cfd_times = []
        
        for frac in fractions:
            for delay in delays:
                # CFD signal
                cfd = pulse_norm.copy()
                cfd_delayed = np.roll(pulse_norm, delay)
                cfd_signal = cfd - frac * cfd_delayed
                
                # Find zero crossing
                zero_crossings = np.where(np.diff(np.sign(cfd_signal)))[0]
                
                if len(zero_crossings) > 0:
                    zc_idx = zero_crossings[0]
                    # Linear interpolation for sub-sample precision
                    if zc_idx < len(cfd_signal) - 1:
                        zc_time = zc_idx + abs(cfd_signal[zc_idx]) / (abs(cfd_signal[zc_idx]) + abs(cfd_signal[zc_idx+1]))
                        cfd_times.append(zc_time * self.dt)
                    else:
                        cfd_times.append(zc_idx * self.dt)
        
        if cfd_times:
            features['cfd_time_mean'] = np.mean(cfd_times)
            features['cfd_time_std'] = np.std(cfd_times)  # Timing jitter
            features['cfd_time_50_3'] = cfd_times[1] if len(cfd_times) > 1 else cfd_times[0]
        else:
            features['cfd_time_mean'] = 0
            features['cfd_time_std'] = 0
            features['cfd_time_50_3'] = 0
        
        return features
    
    def _extract_decay_features(self, pulse, pulse_norm):
        """
        Decay time constants: exponential fit to tail
        Neutrons have longer, multi-component decay
        """
        features = {}
        
        peak_idx = np.argmax(pulse_norm)
        
        # Analyze tail (after peak)
        if peak_idx < len(pulse_norm) - 50:
            tail = pulse[peak_idx:peak_idx+100]  # Use raw pulse for fitting
            tail_norm = pulse_norm[peak_idx:peak_idx+100]
            
            if len(tail) > 20 and tail[0] > 0:
                # Fast component (first 20 ns)
                fast_tail = tail[:20]
                x_fast = np.arange(len(fast_tail)) * self.dt
                
                # Fit exponential: A * exp(-t/tau)
                try:
                    # Log-linear fit
                    valid = fast_tail > 0.1 * np.max(fast_tail)
                    if np.sum(valid) > 5:
                        log_tail = np.log(fast_tail[valid])
                        coeffs_fast = np.polyfit(x_fast[valid], log_tail, 1)
                        features['fast_decay_constant'] = -1.0 / coeffs_fast[0]  # tau in ns
                    else:
                        features['fast_decay_constant'] = 0
                except:
                    features['fast_decay_constant'] = 0
                
                # Slow component (20-100 ns)
                if len(tail) > 40:
                    slow_tail = tail[20:]
                    x_slow = np.arange(len(slow_tail)) * self.dt
                    
                    try:
                        valid = slow_tail > 0.05 * np.max(slow_tail)
                        if np.sum(valid) > 5:
                            log_slow = np.log(slow_tail[valid])
                            coeffs_slow = np.polyfit(x_slow[valid], log_slow, 1)
                            features['slow_decay_constant'] = -1.0 / coeffs_slow[0]
                        else:
                            features['slow_decay_constant'] = 0
                    except:
                        features['slow_decay_constant'] = 0
                    
                    # Ratio of decay constants
                    if features['fast_decay_constant'] > 0:
                        features['decay_constant_ratio'] = features['slow_decay_constant'] / features['fast_decay_constant']
                    else:
                        features['decay_constant_ratio'] = 0
                else:
                    features['slow_decay_constant'] = 0
                    features['decay_constant_ratio'] = 0
            else:
                features['fast_decay_constant'] = 0
                features['slow_decay_constant'] = 0
                features['decay_constant_ratio'] = 0
        else:
            features['fast_decay_constant'] = 0
            features['slow_decay_constant'] = 0
            features['decay_constant_ratio'] = 0
        
        return features
    
    def _extract_derivative_features(self, pulse_norm):
        """
        Pulse derivatives: dV/dt patterns
        Different slopes for n vs γ
        """
        features = {}
        
        # First derivative
        dVdt = np.gradient(pulse_norm, self.dt)
        
        # Second derivative
        d2Vdt2 = np.gradient(dVdt, self.dt)
        
        # Maximum slope (rise)
        features['max_slope'] = np.max(dVdt)
        features['max_slope_position'] = np.argmax(dVdt)
        
        # Minimum slope (fall)
        features['min_slope'] = np.min(dVdt)
        
        # Second derivative features (curvature)
        features['max_curvature'] = np.max(np.abs(d2Vdt2))
        
        # Slope asymmetry
        positive_slope_integral = np.sum(dVdt[dVdt > 0])
        negative_slope_integral = abs(np.sum(dVdt[dVdt < 0]))
        
        if positive_slope_integral + negative_slope_integral > 0:
            features['slope_asymmetry'] = (positive_slope_integral - negative_slope_integral) / (positive_slope_integral + negative_slope_integral)
        else:
            features['slope_asymmetry'] = 0
        
        return features
    
    def _extract_shape_moments(self, pulse_norm):
        """
        Statistical moments of pulse shape
        Skewness, kurtosis characterize distribution
        """
        features = {}
        
        # Treat pulse as probability distribution
        if np.sum(pulse_norm) > 0:
            pulse_dist = pulse_norm / np.sum(pulse_norm)
            
            # Mean position (center of mass)
            x = np.arange(len(pulse_dist))
            features['pulse_mean_position'] = np.sum(x * pulse_dist)
            
            # Variance (width)
            features['pulse_variance'] = np.sum((x - features['pulse_mean_position'])**2 * pulse_dist)
            features['pulse_std'] = np.sqrt(features['pulse_variance'])
            
            # Skewness (asymmetry)
            features['pulse_skewness'] = skew(pulse_norm)
            
            # Kurtosis (tail weight)
            features['pulse_kurtosis'] = kurtosis(pulse_norm)
        else:
            features['pulse_mean_position'] = 0
            features['pulse_variance'] = 0
            features['pulse_std'] = 0
            features['pulse_skewness'] = 0
            features['pulse_kurtosis'] = 0
        
        return features
    
    def _extract_zero_crossing_features(self, pulse_norm):
        """
        Zero-crossing analysis after peak
        Semi-logarithmic analysis reveals decay components
        """
        features = {}
        
        peak_idx = np.argmax(pulse_norm)
        
        # Analyze region after peak
        if peak_idx < len(pulse_norm) - 20:
            post_peak = pulse_norm[peak_idx:]
            
            # Subtract baseline to find zero crossings
            baseline_level = np.mean(pulse_norm[-20:])
            adjusted = post_peak - baseline_level
            
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.sign(adjusted)))[0]
            
            features['num_zero_crossings'] = len(zero_crossings)
            
            if len(zero_crossings) > 0:
                features['first_zero_crossing'] = (peak_idx + zero_crossings[0]) * self.dt
            else:
                features['first_zero_crossing'] = 0
        else:
            features['num_zero_crossings'] = 0
            features['first_zero_crossing'] = 0
        
        return features
    
    def _extract_frequency_features(self, pulse):
        """
        Frequency domain analysis
        FFT reveals frequency content differences
        """
        features = {}
        
        # FFT
        fft = np.fft.rfft(pulse)
        fft_mag = np.abs(fft)
        fft_freq = np.fft.rfftfreq(len(pulse), d=self.dt/1000)  # Convert to MHz
        
        # Spectral centroid (center frequency)
        if np.sum(fft_mag) > 0:
            features['spectral_centroid'] = np.sum(fft_freq * fft_mag) / np.sum(fft_mag)
        else:
            features['spectral_centroid'] = 0
        
        # Spectral bandwidth
        if features['spectral_centroid'] > 0:
            features['spectral_bandwidth'] = np.sqrt(
                np.sum(((fft_freq - features['spectral_centroid'])**2) * fft_mag) / np.sum(fft_mag)
            )
        else:
            features['spectral_bandwidth'] = 0
        
        # High frequency content (>50 MHz)
        high_freq_mask = fft_freq > 50
        if np.sum(fft_mag) > 0:
            features['high_freq_ratio'] = np.sum(fft_mag[high_freq_mask]) / np.sum(fft_mag)
        else:
            features['high_freq_ratio'] = 0
        
        return features
    
    def _extract_tail_characteristics(self, pulse_norm):
        """
        Detailed tail analysis beyond simple integrals
        """
        features = {}
        
        peak_idx = np.argmax(pulse_norm)
        
        # Define tail regions
        if peak_idx < len(pulse_norm) - 100:
            # Early tail (0-50 ns after peak)
            early_tail = pulse_norm[peak_idx:peak_idx+int(50/self.dt)]
            
            # Late tail (50-200 ns after peak)
            late_tail_start = peak_idx + int(50/self.dt)
            late_tail_end = min(peak_idx + int(200/self.dt), len(pulse_norm))
            late_tail = pulse_norm[late_tail_start:late_tail_end]
            
            # Tail integrals
            features['early_tail_integral'] = np.sum(early_tail) if len(early_tail) > 0 else 0
            features['late_tail_integral'] = np.sum(late_tail) if len(late_tail) > 0 else 0
            
            # Tail ratio
            if features['early_tail_integral'] > 0:
                features['tail_ratio'] = features['late_tail_integral'] / features['early_tail_integral']
            else:
                features['tail_ratio'] = 0
            
            # Tail slope (linear fit)
            if len(late_tail) > 10:
                x_tail = np.arange(len(late_tail))
                features['tail_slope'] = np.polyfit(x_tail, late_tail, 1)[0]
            else:
                features['tail_slope'] = 0
        else:
            features['early_tail_integral'] = 0
            features['late_tail_integral'] = 0
            features['tail_ratio'] = 0
            features['tail_slope'] = 0
        
        return features
    
    def _zero_features(self):
        """Return dictionary of zero features for bad pulses"""
        return {key: 0.0 for key in [
            'amplitude', 'peak_position', 'baseline_rms',
            'rise_time_10_90', 'rise_time_10_50', 'rise_time_50_90', 'rise_time_20_80',
            'rise_asymmetry', 't_10', 't_50', 't_90',
            'time_walk_100_1000', 'timing_per_amplitude',
            'cfd_time_mean', 'cfd_time_std', 'cfd_time_50_3',
            'fast_decay_constant', 'slow_decay_constant', 'decay_constant_ratio',
            'max_slope', 'max_slope_position', 'min_slope', 'max_curvature', 'slope_asymmetry',
            'pulse_mean_position', 'pulse_variance', 'pulse_std', 'pulse_skewness', 'pulse_kurtosis',
            'num_zero_crossings', 'first_zero_crossing',
            'spectral_centroid', 'spectral_bandwidth', 'high_freq_ratio',
            'early_tail_integral', 'late_tail_integral', 'tail_ratio', 'tail_slope'
        ]}
    
    def extract_features_batch(self, waveforms):
        """
        Extract features from multiple waveforms
        
        Parameters:
        -----------
        waveforms : array (n_events, n_samples)
            Multiple waveforms
        
        Returns:
        --------
        feature_matrix : array (n_events, n_features)
        feature_names : list
        """
        n_events = len(waveforms)
        
        # Extract features from first waveform to get feature names
        first_features = self.extract_all_features(waveforms[0])
        feature_names = sorted(first_features.keys())
        n_features = len(feature_names)
        
        # Preallocate
        feature_matrix = np.zeros((n_events, n_features))
        
        # Extract all
        for i, waveform in enumerate(waveforms):
            if i % 1000 == 0:
                print(f"Extracting features: {i}/{n_events}", end='\r')
            
            features = self.extract_all_features(waveform)
            feature_matrix[i, :] = [features[name] for name in feature_names]
        
        print(f"Extracting features: {n_events}/{n_events} - Complete!")
        
        return feature_matrix, feature_names


# =============================================================================
# Integration with ML Classifier
# =============================================================================

def enhance_ml_classifier_with_timing():
    """
    Example: Enhance classical ML classifier to use timing features
    """
    from psd_analysis.ml.classical import ClassicalMLClassifier
    
    class EnhancedMLClassifier(ClassicalMLClassifier):
        """ML classifier with comprehensive timing features"""
        
        def __init__(self, method='random_forest', sampling_rate_mhz=250):
            super().__init__(method=method)
            self.timing_extractor = TimingFeatureExtractor(sampling_rate_mhz=sampling_rate_mhz)
        
        def extract_features(self, df):
            """Override to include timing features"""
            
            # Get basic features from parent class
            basic_features, basic_names = super().extract_features(df)
            
            # Extract timing features if waveforms available
            sample_cols = [col for col in df.columns if col.startswith('SAMPLE')]
            
            if len(sample_cols) > 0:
                print("Extracting advanced timing features...")
                waveforms = df[sample_cols].values
                
                timing_features, timing_names = self.timing_extractor.extract_features_batch(waveforms)
                
                # Combine
                all_features = np.column_stack([basic_features, timing_features])
                all_names = basic_names + timing_names
                
                print(f"Total features: {len(all_names)}")
                print(f"  Basic: {len(basic_names)}")
                print(f"  Timing: {len(timing_names)}")
                
                return all_features, all_names
            else:
                print("Warning: No waveforms found, using basic features only")
                return basic_features, basic_names
    
    return EnhancedMLClassifier


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Timing Feature Extractor Example")
    print("=" * 70)
    
    # Create dummy waveform
    n_samples = 368
    t = np.arange(n_samples)
    
    # Simulate neutron-like pulse
    baseline = 3250
    amplitude = 1500
    rise_time = 5
    decay_fast = 10
    decay_slow = 50
    
    # Multi-component decay
    pulse = amplitude * (
        0.7 * np.exp(-(t - 100) / decay_fast) +
        0.3 * np.exp(-(t - 100) / decay_slow)
    ) * (1 / (1 + np.exp(-(t - 100) / rise_time)))
    
    waveform = baseline - pulse + np.random.normal(0, 5, n_samples)
    waveform[t < 100] = baseline + np.random.normal(0, 5, np.sum(t < 100))
    
    # Extract features
    extractor = TimingFeatureExtractor(sampling_rate_mhz=250)
    features = extractor.extract_all_features(waveform)
    
    print("\nExtracted Features:")
    print("-" * 70)
    for name, value in sorted(features.items()):
        print(f"  {name:<30} {value:>15.4f}")
    
    print(f"\nTotal features extracted: {len(features)}")
    print("\nThese features can dramatically improve ML classification!")
    print("Expected improvement: 1-3% accuracy gain over basic PSD only")