"""
Complete example: Using ML for PSD analysis
Demonstrates both classical ML and deep learning approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from psd_analysis import load_psd_data, validate_events, calculate_psd_ratio
from psd_analysis.ml.classical import ClassicalMLClassifier, plot_ml_performance
from psd_analysis.ml.deep_learning import DeepPSDClassifier, plot_training_history

# =============================================================================
# PART 1: Classical Machine Learning
# =============================================================================

def example_classical_ml():
    """
    Example workflow for classical ML classification
    """
    print("="*70)
    print("CLASSICAL MACHINE LEARNING FOR PSD")
    print("="*70)
    
    # Load calibration data (known neutron and gamma sources)
    print("\n1. Loading calibration data...")
    df_ambe = load_psd_data('data/calibration/ambe_source.csv')  # Neutron source
    df_cs137 = load_psd_data('data/calibration/cs137_source.csv')  # Gamma source
    
    # QC
    valid_n, _ = validate_events(df_ambe)
    valid_g, _ = validate_events(df_cs137)
    df_ambe = df_ambe[valid_n].copy()
    df_cs137 = df_cs137[valid_g].copy()
    
    # Calculate PSD
    df_ambe = calculate_psd_ratio(df_ambe)
    df_cs137 = calculate_psd_ratio(df_cs137)
    
    # Label data
    df_ambe['PARTICLE'] = 'neutron'
    df_cs137['PARTICLE'] = 'gamma'
    
    # Combine for training
    df_train = pd.concat([df_ambe, df_cs137], ignore_index=True)
    print(f"   Total training samples: {len(df_train)}")
    print(f"   Neutrons: {len(df_ambe)}, Gammas: {len(df_cs137)}")
    
    # Compare different classifiers
    methods = ['random_forest', 'gradient_boosting', 'svm', 'neural_net']
    results_dict = {}
    
    for method in methods:
        print(f"\n2. Training {method} classifier...")
        
        clf = ClassicalMLClassifier(method=method)
        results = clf.train(df_train, test_size=0.2)
        results_dict[method] = results
        
        # Save model
        clf.save(f'models/psd_{method}.pkl')
        
        # Plot performance
        fig = plot_ml_performance(results)
        fig.suptitle(f'{method.replace("_", " ").title()} Performance', 
                    fontsize=16, y=1.00)
        plt.savefig(f'results/figures/ml_{method}_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Compare all methods
    print("\n3. Performance comparison:")
    print(f"{'Method':<20} {'Val Accuracy':<15} {'ROC AUC':<10}")
    print("-" * 45)
    for method, results in results_dict.items():
        print(f"{method:<20} {results['val_accuracy']:<15.4f} {results['roc_auc']:<10.4f}")
    
    # Use best model on unknown data
    print("\n4. Applying to unknown NORM sample...")
    best_method = max(results_dict.items(), key=lambda x: x[1]['val_accuracy'])[0]
    print(f"   Using best model: {best_method}")
    
    # Load best model
    clf_best = ClassicalMLClassifier(method=best_method)
    clf_best.load(f'models/psd_{best_method}.pkl')
    
    # Load unknown sample
    df_unknown = load_psd_data('data/norm_samples/unknown_sample_001.csv')
    valid, _ = validate_events(df_unknown)
    df_unknown = df_unknown[valid].copy()
    df_unknown = calculate_psd_ratio(df_unknown)
    
    # Predict
    predictions, probabilities = clf_best.predict(df_unknown)
    
    df_unknown['PARTICLE_ML'] = ['neutron' if p == 1 else 'gamma' for p in predictions]
    df_unknown['NEUTRON_PROBABILITY'] = probabilities
    
    n_neutrons = (predictions == 1).sum()
    n_gammas = (predictions == 0).sum()
    
    print(f"   Classified {n_neutrons} neutrons and {n_gammas} gammas")
    print(f"   Neutron fraction: {n_neutrons/len(predictions)*100:.2f}%")
    
    # Save results
    df_unknown.to_csv('results/unknown_sample_ml_classified.csv', index=False)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PSD scatter with ML classification
    ax = axes[0]
    neutron_mask = predictions == 1
    gamma_mask = predictions == 0
    
    ax.scatter(df_unknown[gamma_mask]['ENERGY'], df_unknown[gamma_mask]['PSD'],
              c='blue', s=1, alpha=0.3, label='Gamma (ML)')
    ax.scatter(df_unknown[neutron_mask]['ENERGY'], df_unknown[neutron_mask]['PSD'],
              c='red', s=1, alpha=0.3, label='Neutron (ML)')
    ax.set_xlabel('Energy (ADC)', fontsize=12)
    ax.set_ylabel('PSD', fontsize=12)
    ax.set_title('ML Classification Results', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Probability distribution
    ax = axes[1]
    ax.hist(probabilities, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
    ax.set_xlabel('Neutron Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Confidence', fontsize=14)
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('results/figures/ml_classification_unknown_sample.png', dpi=150)
    plt.close()
    
    print("\n✓ Classical ML analysis complete!")
    return df_unknown, clf_best


# =============================================================================
# PART 2: Deep Learning
# =============================================================================

def example_deep_learning():
    """
    Example workflow for deep learning classification
    """
    print("\n" + "="*70)
    print("DEEP LEARNING FOR PSD")
    print("="*70)
    
    # Check if PyTorch is available
    try:
        import torch
    except ImportError:
        print("\nPyTorch not installed. Skipping deep learning example.")
        print("Install with: pip install torch")
        return None
    
    # Load calibration data with waveforms
    print("\n1. Loading calibration data with waveforms...")
    df_ambe = load_psd_data('data/calibration/ambe_source.csv')
    df_cs137 = load_psd_data('data/calibration/cs137_source.csv')
    
    # QC
    valid_n, _ = validate_events(df_ambe)
    valid_g, _ = validate_events(df_cs137)
    df_ambe = df_ambe[valid_n].copy()
    df_cs137 = df_cs137[valid_g].copy()
    
    # Calculate PSD for physics-informed loss
    df_ambe = calculate_psd_ratio(df_ambe)
    df_cs137 = calculate_psd_ratio(df_cs137)
    
    # Label
    df_ambe['PARTICLE'] = 'neutron'
    df_cs137['PARTICLE'] = 'gamma'
    
    # Combine
    df_train = pd.concat([df_ambe, df_cs137], ignore_index=True)
    print(f"   Total training samples: {len(df_train)}")
    
    # Get waveform length
    sample_cols = [col for col in df_train.columns if col.startswith('SAMPLE')]
    input_length = len(sample_cols)
    print(f"   Waveform length: {input_length} samples")
    
    # Train CNN model
    print("\n2. Training 1D CNN model...")
    cnn_model = DeepPSDClassifier(model_type='cnn', input_length=input_length)
    
    history_cnn = cnn_model.train(
        df_train,
        epochs=30,
        batch_size=64,
        learning_rate=0.001,
        use_physics_loss=True,
        val_split=0.2
    )
    
    # Save model
    cnn_model.save('models/psd_cnn.pt')
    
    # Plot training history
    fig = plot_training_history(history_cnn)
    fig.suptitle('CNN Training History', fontsize=16, y=1.02)
    plt.savefig('results/figures/cnn_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Train Transformer model
    print("\n3. Training Transformer model...")
    transformer_model = DeepPSDClassifier(model_type='transformer', input_length=input_length)
    
    history_transformer = transformer_model.train(
        df_train,
        epochs=30,
        batch_size=32,
        learning_rate=0.0005,
        use_physics_loss=True,
        val_split=0.2
    )
    
    # Save model
    transformer_model.save('models/psd_transformer.pt')
    
    # Plot training history
    fig = plot_training_history(history_transformer)
    fig.suptitle('Transformer Training History', fontsize=16, y=1.02)
    plt.savefig('results/figures/transformer_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compare models
    print("\n4. Model comparison:")
    print(f"{'Model':<15} {'Final Val Accuracy':<20}")
    print("-" * 35)
    print(f"{'CNN':<15} {history_cnn['val_acc'][-1]:<20.2f}%")
    print(f"{'Transformer':<15} {history_transformer['val_acc'][-1]:<20.2f}%")
    
    # Apply to unknown data
    print("\n5. Applying best deep learning model to unknown sample...")
    
    # Use best model (assume CNN for this example)
    best_dl_model = cnn_model
    
    # Load unknown sample
    df_unknown = load_psd_data('data/norm_samples/unknown_sample_001.csv')
    valid, _ = validate_events(df_unknown)
    df_unknown = df_unknown[valid].copy()
    df_unknown = calculate_psd_ratio(df_unknown)
    
    # Predict
    predictions, probabilities = best_dl_model.predict(df_unknown)
    
    df_unknown['PARTICLE_DL'] = ['neutron' if p == 1 else 'gamma' for p in predictions]
    df_unknown['NEUTRON_PROBABILITY_DL'] = probabilities
    
    n_neutrons = (predictions == 1).sum()
    n_gammas = (predictions == 0).sum()
    
    print(f"   Classified {n_neutrons} neutrons and {n_gammas} gammas")
    print(f"   Neutron fraction: {n_neutrons/len(predictions)*100:.2f}%")
    
    # Save results
    df_unknown.to_csv('results/unknown_sample_dl_classified.csv', index=False)
    
    print("\n✓ Deep learning analysis complete!")
    return df_unknown, best_dl_model


# =============================================================================
# PART 3: Comparison and Visualization
# =============================================================================

def compare_methods(df_unknown_ml, df_unknown_dl, df_traditional=None):
    """
    Compare traditional PSD, classical ML, and deep learning
    """
    print("\n" + "="*70)
    print("COMPARISON OF METHODS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Traditional PSD (if available)
    if df_traditional is not None and 'PARTICLE' in df_traditional.columns:
        ax = axes[0, 0]
        n_mask = df_traditional['PARTICLE'] == 'neutron'
        g_mask = df_traditional['PARTICLE'] == 'gamma'
        
        ax.scatter(df_traditional[g_mask]['ENERGY'], df_traditional[g_mask]['PSD'],
                  c='blue', s=1, alpha=0.3, label='Gamma')
        ax.scatter(df_traditional[n_mask]['ENERGY'], df_traditional[n_mask]['PSD'],
                  c='red', s=1, alpha=0.3, label='Neutron')
        ax.set_title('Traditional PSD (Linear Cut)', fontsize=14)
        ax.set_xlabel('Energy', fontsize=12)
        ax.set_ylabel('PSD', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Traditional PSD\nNot Available', 
                       ha='center', va='center', fontsize=14, 
                       transform=axes[0, 0].transAxes)
        axes[0, 0].axis('off')
    
    # Classical ML
    ax = axes[0, 1]
    n_mask = df_unknown_ml['PARTICLE_ML'] == 'neutron'
    g_mask = df_unknown_ml['PARTICLE_ML'] == 'gamma'
    
    ax.scatter(df_unknown_ml[g_mask]['ENERGY'], df_unknown_ml[g_mask]['PSD'],
              c='blue', s=1, alpha=0.3, label='Gamma')
    ax.scatter(df_unknown_ml[n_mask]['ENERGY'], df_unknown_ml[n_mask]['PSD'],
              c='red', s=1, alpha=0.3, label='Neutron')
    ax.set_title('Classical ML (Random Forest)', fontsize=14)
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('PSD', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Deep Learning
    ax = axes[1, 0]
    n_mask = df_unknown_dl['PARTICLE_DL'] == 'neutron'
    g_mask = df_unknown_dl['PARTICLE_DL'] == 'gamma'
    
    ax.scatter(df_unknown_dl[g_mask]['ENERGY'], df_unknown_dl[g_mask]['PSD'],
              c='blue', s=1, alpha=0.3, label='Gamma')
    ax.scatter(df_unknown_dl[n_mask]['ENERGY'], df_unknown_dl[n_mask]['PSD'],
              c='red', s=1, alpha=0.3, label='Neutron')
    ax.set_title('Deep Learning (CNN)', fontsize=14)
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('PSD', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Agreement between methods
    ax = axes[1, 1]
    
    # Compare ML and DL classifications
    ml_neutrons = (df_unknown_ml['PARTICLE_ML'] == 'neutron').values
    dl_neutrons = (df_unknown_dl['PARTICLE_DL'] == 'neutron').values
    
    agreement = (ml_neutrons == dl_neutrons).sum() / len(ml_neutrons) * 100
    
    categories = ['Both Gamma', 'ML:G, DL:N', 'ML:N, DL:G', 'Both Neutron']
    counts = [
        ((~ml_neutrons) & (~dl_neutrons)).sum(),
        ((~ml_neutrons) & (dl_neutrons)).sum(),
        ((ml_neutrons) & (~dl_neutrons)).sum(),
        ((ml_neutrons) & (dl_neutrons)).sum()
    ]
    
    colors = ['blue', 'orange', 'orange', 'red']
    ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'ML vs DL Agreement: {agreement:.1f}%', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAgreement between ML and DL: {agreement:.2f}%")
    print(f"Both classified as gamma: {counts[0]}")
    print(f"Disagreement: {counts[1] + counts[2]}")
    print(f"Both classified as neutron: {counts[3]}")


# =============================================================================
# PART 4: Advanced Analysis
# =============================================================================

def analyze_ml_uncertainties(df, ml_classifier):
    """
    Analyze classification uncertainties
    """
    print("\n" + "="*70)
    print("UNCERTAINTY ANALYSIS")
    print("="*70)
    
    # Get probabilities
    predictions, probabilities = ml_classifier.predict(df)
    
    # Uncertainty = distance from decision boundary (0.5)
    uncertainty = np.abs(probabilities - 0.5)
    
    # High confidence: > 0.4 from boundary
    # Medium confidence: 0.2 - 0.4
    # Low confidence: < 0.2
    
    high_conf = (uncertainty > 0.4).sum()
    med_conf = ((uncertainty >= 0.2) & (uncertainty <= 0.4)).sum()
    low_conf = (uncertainty < 0.2).sum()
    
    print(f"\nClassification confidence:")
    print(f"  High confidence: {high_conf} ({high_conf/len(df)*100:.1f}%)")
    print(f"  Medium confidence: {med_conf} ({med_conf/len(df)*100:.1f}%)")
    print(f"  Low confidence: {low_conf} ({low_conf/len(df)*100:.1f}%)")
    
    # Plot uncertainty vs energy
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Uncertainty distribution
    ax = axes[0]
    ax.hist(uncertainty, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0.2, color='orange', linestyle='--', label='Low/Med threshold')
    ax.axvline(0.4, color='red', linestyle='--', label='Med/High threshold')
    ax.set_xlabel('Uncertainty (distance from 0.5)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Uncertainty Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Uncertainty vs Energy
    ax = axes[1]
    scatter = ax.scatter(df['ENERGY'].values, probabilities, 
                        c=uncertainty, s=2, cmap='RdYlGn', vmin=0, vmax=0.5)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    ax.set_xlabel('Energy (ADC)', fontsize=12)
    ax.set_ylabel('Neutron Probability', fontsize=12)
    ax.set_title('Classification Confidence vs Energy', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/classification_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Uncertainty analysis complete!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("PSD ANALYSIS WITH MACHINE LEARNING")
    print("Complete workflow demonstration")
    print("="*70)
    
    # Create output directories
    import os
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # Run classical ML
    df_unknown_ml, clf_ml = example_classical_ml()
    
    # Run deep learning (if PyTorch available)
    try:
        df_unknown_dl, clf_dl = example_deep_learning()
        
        # Compare methods
        compare_methods(df_unknown_ml, df_unknown_dl)
        
        # Analyze uncertainties
        analyze_ml_uncertainties(df_unknown_ml, clf_ml)
        
    except Exception as e:
        print(f"\nDeep learning failed: {e}")
        print("Continuing with classical ML only...")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - models/psd_*.pkl (trained classical ML models)")
    print("  - models/psd_*.pt (trained deep learning models)")
    print("  - results/figures/*.png (performance plots)")
    print("  - results/*_classified.csv (classified data)")
    print("\nNext steps:")
    print("  1. Review performance plots in results/figures/")
    print("  2. Choose best model based on validation metrics")
    print("  3. Apply to your production data")
    print("  4. Monitor classification confidence")
    print("  5. Retrain periodically with new calibration data")
    print("="*70)