"""
Proper Evaluation Metrics for PSD Classification
Replaces misleading "accuracy" with physically meaningful metrics

Key metrics:
- Gamma mis-ID rate at fixed neutron acceptance
- PSD Figure of Merit vs energy
- ROC/AUC per energy bin
- Operating point curves
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd


def evaluate_psd_classifier(y_true, y_pred_proba, energies, 
                            energy_bins=[(100, 300), (300, 800), (800, 3000)],
                            neutron_acceptance=0.99):
    """
    Comprehensive PSD evaluation with energy-dependent metrics
    
    Parameters:
    -----------
    y_true : array
        True labels (0=gamma, 1=neutron)
    y_pred_proba : array
        Predicted neutron probability
    energies : array
        Event energies (keVee)
    energy_bins : list of tuples
        Energy ranges for bin analysis
    neutron_acceptance : float
        Target neutron efficiency for gamma mis-ID calculation
    
    Returns:
    --------
    results : dict
        Comprehensive metrics
    """
    results = {
        'overall': {},
        'by_energy': {}
    }
    
    # =========================================================================
    # OVERALL METRICS
    # =========================================================================
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    results['overall']['roc_auc'] = roc_auc
    results['overall']['fpr'] = fpr
    results['overall']['tpr'] = tpr
    results['overall']['thresholds'] = thresholds
    
    # Operating point: gamma mis-ID at target neutron acceptance
    target_idx = np.argmin(np.abs(tpr - neutron_acceptance))
    gamma_misid = fpr[target_idx]
    threshold_at_target = thresholds[target_idx]
    
    results['overall']['neutron_acceptance'] = neutron_acceptance
    results['overall']['gamma_misid_rate'] = gamma_misid
    results['overall']['threshold'] = threshold_at_target
    
    print("="*70)
    print("OVERALL PERFORMANCE")
    print("="*70)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"At {neutron_acceptance*100:.1f}% neutron acceptance:")
    print(f"  Gamma mis-ID rate: {gamma_misid*100:.3f}%")
    print(f"  Operating threshold: {threshold_at_target:.4f}")
    
    # =========================================================================
    # ENERGY-DEPENDENT METRICS
    # =========================================================================
    
    print("\n" + "="*70)
    print("ENERGY-DEPENDENT PERFORMANCE")
    print("="*70)
    
    for e_min, e_max in energy_bins:
        bin_name = f"{e_min}-{e_max}_keVee"
        
        # Select events in this energy bin
        in_bin = (energies >= e_min) & (energies < e_max)
        
        if in_bin.sum() < 10:
            print(f"\n{bin_name}: Insufficient statistics ({in_bin.sum()} events)")
            continue
        
        y_