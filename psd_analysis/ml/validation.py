"""
Validation strategies and data augmentation for PSD ML
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
import warnings


def create_honest_splits(df, group_col='run_id', n_splits=5, test_size=0.2, random_state=42):
    """
    Create train/test splits grouped by run/day/hardware

    Traditional ML splits data randomly → overly optimistic results
    Our approach: entire runs go into either train or test
    Simulates real deployment: train on old data, test on new data

    Parameters:
    -----------
    df : DataFrame
        Data with group_col column
    group_col : str
        Column to group by ('run_id', 'date', 'hardware_id')
    n_splits : int
        Number of CV folds
    test_size : float
        Fraction for final test set
    random_state : int
        Random seed

    Returns:
    --------
    splits : dict
        'train_val': indices for train+validation
        'test': indices for final test
        'cv_folds': list of {train, val} for cross-validation
        'test_groups': which groups went to test
        'train_val_groups': which groups for training
    """
    if group_col not in df.columns:
        warnings.warn(f"Column '{group_col}' not found. Creating random groups.")
        # Create artificial groups
        n_groups = max(10, len(df) // 1000)
        df = df.copy()
        df[group_col] = np.random.randint(0, n_groups, len(df))

    groups = df[group_col].values
    unique_groups = np.unique(groups)

    # Split groups into train_val and test
    n_test_groups = max(1, int(len(unique_groups) * test_size))

    np.random.seed(random_state)
    test_groups = np.random.choice(unique_groups, n_test_groups, replace=False)
    train_val_groups = np.setdiff1d(unique_groups, test_groups)

    # Get indices
    test_mask = np.isin(groups, test_groups)
    train_val_mask = ~test_mask

    test_indices = np.where(test_mask)[0]
    train_val_indices = np.where(train_val_mask)[0]

    # Create CV folds within train_val (also grouped)
    cv_folds = []
    if n_splits > 1:
        gkf = GroupKFold(n_splits=min(n_splits, len(train_val_groups)))
        train_val_data = df.iloc[train_val_indices]
        train_val_groups_arr = train_val_data[group_col].values

        for train_idx, val_idx in gkf.split(train_val_data, groups=train_val_groups_arr):
            # Convert back to original indices
            fold = {
                'train': train_val_indices[train_idx],
                'val': train_val_indices[val_idx]
            }
            cv_folds.append(fold)

    splits = {
        'train_val': train_val_indices,
        'test': test_indices,
        'cv_folds': cv_folds,
        'test_groups': test_groups.tolist(),
        'train_val_groups': train_val_groups.tolist(),
        'group_col': group_col
    }

    print(f"Honest splits created:")
    print(f"  Total groups: {len(unique_groups)}")
    print(f"  Train+Val groups: {len(train_val_groups)} ({len(train_val_indices)} events)")
    print(f"  Test groups: {len(test_groups)} ({len(test_indices)} events)")
    print(f"  CV folds: {n_splits}")

    return splits


def augment_waveform(waveform, baseline_rms=5, amp_jitter=0.05, time_jitter=2,
                     gain_jitter=0.02, noise_injection=True):
    """
    Augment waveform for robustness

    Simulates operational variations:
    - Amplitude jitter (PMT gain fluctuations)
    - Time jitter (trigger variations)
    - Noise injection (baseline RMS matched)
    - Gain scaling (HV variations)

    Parameters:
    -----------
    waveform : array
        Original ADC waveform
    baseline_rms : float
        RMS of baseline noise to inject
    amp_jitter : float
        Amplitude jitter fraction (0-1)
    time_jitter : int
        Time jitter in samples
    gain_jitter : float
        Gain variation fraction (0-1)
    noise_injection : bool
        Add random noise

    Returns:
    --------
    augmented : array
        Augmented waveform
    """
    wf = waveform.copy()

    # 1. Gain jitter (multiplicative)
    if gain_jitter > 0:
        gain_factor = 1.0 + np.random.uniform(-gain_jitter, gain_jitter)
        wf = wf * gain_factor

    # 2. Amplitude jitter (additive)
    if amp_jitter > 0:
        amp_offset = np.random.uniform(-amp_jitter, amp_jitter) * np.max(wf)
        wf = wf + amp_offset

    # 3. Time jitter (shift)
    if time_jitter > 0:
        shift = np.random.randint(-time_jitter, time_jitter + 1)
        wf = np.roll(wf, shift)

    # 4. Noise injection
    if noise_injection and baseline_rms > 0:
        noise = np.random.normal(0, baseline_rms, len(wf))
        wf = wf + noise

    return wf


def augment_batch(waveforms, n_augmented_per_sample=3, **aug_kwargs):
    """
    Augment entire batch of waveforms

    Parameters:
    -----------
    waveforms : array (N, samples)
        Original waveforms
    n_augmented_per_sample : int
        Number of augmented versions per original
    **aug_kwargs : additional arguments for augment_waveform

    Returns:
    --------
    augmented : array ((N * n_aug), samples)
        Original + augmented waveforms
    labels : array
        Labels (if provided, will be replicated)
    """
    augmented_list = [waveforms]  # Include originals

    for _ in range(n_augmented_per_sample):
        augmented = np.array([augment_waveform(wf, **aug_kwargs) for wf in waveforms])
        augmented_list.append(augmented)

    augmented_all = np.concatenate(augmented_list, axis=0)

    return augmented_all


def normalize_features_per_run(X, run_ids):
    """
    Normalize features separately for each run

    Domain adaptation approach to handle detector drift

    Parameters:
    -----------
    X : array (N, features)
        Feature matrix
    run_ids : array (N,)
        Run ID for each event

    Returns:
    --------
    X_normalized : array (N, features)
        Normalized features
    normalization_params : dict
        Per-run normalization parameters
    """
    X_norm = X.copy()
    unique_runs = np.unique(run_ids)

    normalization_params = {}

    for run_id in unique_runs:
        mask = run_ids == run_id
        X_run = X[mask]

        # Per-run mean and std
        mean = np.mean(X_run, axis=0)
        std = np.std(X_run, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero

        # Normalize
        X_norm[mask] = (X_run - mean) / std

        # Store parameters
        normalization_params[run_id] = {'mean': mean, 'std': std}

    return X_norm, normalization_params


def apply_run_normalization(X, run_ids, normalization_params):
    """
    Apply previously computed run-wise normalization

    Parameters:
    -----------
    X : array (N, features)
        Feature matrix
    run_ids : array (N,)
        Run ID for each event
    normalization_params : dict
        Parameters from normalize_features_per_run

    Returns:
    --------
    X_normalized : array (N, features)
        Normalized features
    """
    X_norm = X.copy()

    for run_id in np.unique(run_ids):
        if run_id in normalization_params:
            mask = run_ids == run_id
            params = normalization_params[run_id]

            X_norm[mask] = (X[mask] - params['mean']) / params['std']
        else:
            warnings.warn(f"No normalization params for run_id={run_id}, using identity")

    return X_norm


def create_time_based_splits(df, time_col='timestamp', test_fraction=0.2,
                             val_fraction=0.1):
    """
    Split data by time (chronological split)

    Most realistic for deployment: train on past, test on future

    Parameters:
    -----------
    df : DataFrame
        Data with time_col column
    time_col : str
        Column with timestamps
    test_fraction : float
        Fraction for test set (most recent data)
    val_fraction : float
        Fraction for validation (before test)

    Returns:
    --------
    splits : dict
        'train', 'val', 'test' indices
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found")

    # Sort by time
    sorted_indices = df[time_col].argsort()

    n = len(df)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_test - n_val

    splits = {
        'train': sorted_indices[:n_train],
        'val': sorted_indices[n_train:n_train+n_val],
        'test': sorted_indices[n_train+n_val:]
    }

    print(f"Time-based splits:")
    print(f"  Train: {n_train} events (oldest)")
    print(f"  Val: {n_val} events")
    print(f"  Test: {n_test} events (most recent)")

    return splits


def balance_classes(df, label_col='PARTICLE', method='undersample', random_state=42):
    """
    Balance neutron/gamma classes

    Parameters:
    -----------
    df : DataFrame
        Data with label column
    label_col : str
        Column with class labels
    method : str
        'undersample' (remove majority) or 'oversample' (duplicate minority)
    random_state : int
        Random seed

    Returns:
    --------
    balanced_df : DataFrame
        Balanced dataset
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found")

    # Count classes
    class_counts = df[label_col].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    n_minority = class_counts[minority_class]

    print(f"Original class distribution:")
    print(class_counts)

    if method == 'undersample':
        # Undersample majority to match minority
        df_minority = df[df[label_col] == minority_class]
        df_majority = df[df[label_col] == majority_class]

        df_majority_sampled = df_majority.sample(n=n_minority, random_state=random_state)

        balanced_df = pd.concat([df_minority, df_majority_sampled])

    elif method == 'oversample':
        # Oversample minority to match majority
        n_majority = class_counts[majority_class]

        df_minority = df[df[label_col] == minority_class]
        df_majority = df[df[label_col] == majority_class]

        df_minority_oversampled = df_minority.sample(n=n_majority, replace=True,
                                                     random_state=random_state)

        balanced_df = pd.concat([df_majority, df_minority_oversampled])

    else:
        raise ValueError(f"Unknown method: {method}")

    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\nBalanced class distribution:")
    print(balanced_df[label_col].value_counts())

    return balanced_df
