"""
Practical demonstration of timing features for PSD
Shows comparison between basic and enhanced features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Import your modules
import sys
sys.path.append('..')
from psd_analysis.features.timing import TimingFeatureExtractor
from psd_analysis import load_psd_data, validate_events, calculate_psd_ratio

# [INSERT ALL THE CODE FROM THE FRAGMENT HERE]
# Including all the functions:
# - compare_basic_vs_enhanced()
# - find_best_features()
# - analyze_low_confidence_events()
# - analyze_feature_correlations()
# - production_workflow_with_timing_features()
# - if __name__ == "__main__": block
```

### Option 2: Module with Examples

Create a new directory for examples:
```
psd_analysis/
├── examples/
│   ├── __init__.py
│   ├── timing_demo.py          # ← The code fragment goes here
│   ├── ml_comparison.py         # Compare different ML methods
│   └── feature_analysis.py      # Feature importance analysis