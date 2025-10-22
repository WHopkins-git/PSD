#!/usr/bin/env python
"""
Train ML model for PSD classification
Usage: python ml_train.py --neutron ambe.csv --gamma cs137.csv --output models/psd_rf.pkl
"""

import argparse
import pandas as pd

from psd_analysis import load_psd_data, validate_events, calculate_psd_ratio
from psd_analysis.ml.classical import ClassicalMLClassifier


def main():
    parser = argparse.ArgumentParser(description='Train PSD ML classifier')
    parser.add_argument('--neutron', required=True, help='Neutron source data file')
    parser.add_argument('--gamma', required=True, help='Gamma source data file')
    parser.add_argument('--method', default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'svm', 'neural_net', 'logistic'],
                       help='ML method')
    parser.add_argument('--output', default='models/psd_classifier.pkl',
                       help='Output model file')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set fraction')

    args = parser.parse_args()

    # Load neutron data
    print(f"Loading neutron data from {args.neutron}...")
    df_n = load_psd_data(args.neutron)
    df_n = df_n[validate_events(df_n)[0]]
    df_n = calculate_psd_ratio(df_n)
    df_n['PARTICLE'] = 'neutron'

    # Load gamma data
    print(f"Loading gamma data from {args.gamma}...")
    df_g = load_psd_data(args.gamma)
    df_g = df_g[validate_events(df_g)[0]]
    df_g = calculate_psd_ratio(df_g)
    df_g['PARTICLE'] = 'gamma'

    # Combine
    df_train = pd.concat([df_n, df_g], ignore_index=True)
    print(f"\nTraining data: {len(df_n)} neutrons, {len(df_g)} gammas")

    # Train
    print(f"\nTraining {args.method} classifier...")
    clf = ClassicalMLClassifier(method=args.method)
    results = clf.train(df_train, test_size=args.test_size)

    print(f"\nTraining Results:")
    print(f"  Train accuracy: {results['train_accuracy']:.4f}")
    print(f"  Validation accuracy: {results['val_accuracy']:.4f}")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")

    # Save
    clf.save(args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == '__main__':
    main()
