#!/usr/bin/env python
"""
Apply trained ML model to classify unknown data
Usage: python ml_predict.py --model models/psd_rf.pkl --data unknown.csv --output results/classified.csv
"""

import argparse
import numpy as np

from psd_analysis import load_psd_data, validate_events, calculate_psd_ratio
from psd_analysis.ml.classical import ClassicalMLClassifier


def main():
    parser = argparse.ArgumentParser(description='Apply PSD ML classifier')
    parser.add_argument('--model', required=True, help='Trained model file (.pkl)')
    parser.add_argument('--data', required=True, help='Data file to classify')
    parser.add_argument('--output', default='results/classified_data.csv',
                       help='Output file')
    parser.add_argument('--method', default='random_forest', help='ML method used')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    clf = ClassicalMLClassifier(method=args.method)
    clf.load(args.model)

    # Load data
    print(f"Loading data from {args.data}...")
    df = load_psd_data(args.data)
    df = df[validate_events(df)[0]]
    df = calculate_psd_ratio(df)

    print(f"Loaded {len(df)} events")

    # Predict
    print("Classifying events...")
    predictions, probabilities = clf.predict(df)

    df['PARTICLE_ML'] = ['neutron' if p==1 else 'gamma' for p in predictions]
    df['NEUTRON_PROB'] = probabilities
    df['CONFIDENCE'] = np.abs(probabilities - 0.5)

    # Statistics
    n_neutrons = (predictions == 1).sum()
    n_gammas = (predictions == 0).sum()

    print(f"\nClassification Results:")
    print(f"  Neutrons: {n_neutrons} ({n_neutrons/len(df)*100:.2f}%)")
    print(f"  Gammas: {n_gammas} ({n_gammas/len(df)*100:.2f}%)")

    low_conf = (df['CONFIDENCE'] < 0.2).sum()
    print(f"  Low confidence (<0.2): {low_conf} ({low_conf/len(df)*100:.2f}%)")

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
