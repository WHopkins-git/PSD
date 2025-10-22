"""
Machine Learning for PSD Analysis
Includes classical ML and deep learning approaches
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# =============================================================================
# psd/ml/classical.py
# =============================================================================

"""
Classical machine learning classifiers for neutron/gamma discrimination
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib


class ClassicalMLClassifier:
    """
    Wrapper for classical ML methods for PSD
    """
    
    def __init__(self, method='random_forest'):
        """
        Initialize classifier
        
        Parameters:
        -----------
        method : str
            'random_forest', 'gradient_boosting', 'svm', 'neural_net', 'logistic'
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize model
        if method == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif method == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif method == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif method == 'neural_net':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        elif method == 'logistic':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def extract_features(self, df):
        """
        Extract features for classification
        
        Parameters:
        -----------
        df : DataFrame
            Must have ENERGY, PSD columns, optionally waveform samples
        
        Returns:
        --------
        features : array
            Feature matrix
        feature_names : list
            Names of features
        """
        features_list = []
        feature_names = []
        
        # Basic PSD features
        if 'PSD' in df.columns:
            features_list.append(df['PSD'].values)
            feature_names.append('PSD')
        
        if 'ENERGY' in df.columns:
            features_list.append(df['ENERGY'].values)
            feature_names.append('ENERGY')
        elif 'ENERGY_KEV' in df.columns:
            features_list.append(df['ENERGY_KEV'].values)
            feature_names.append('ENERGY_KEV')
        
        if 'ENERGYSHORT' in df.columns:
            features_list.append(df['ENERGYSHORT'].values)
            feature_names.append('ENERGYSHORT')
        
        # Derived features
        if 'ENERGY' in df.columns and 'ENERGYSHORT' in df.columns:
            tail_integral = df['ENERGY'] - df['ENERGYSHORT']
            features_list.append(tail_integral.values)
            feature_names.append('TAIL_INTEGRAL')
            
            tail_fraction = tail_integral / df['ENERGY']
            features_list.append(tail_fraction.values)
            feature_names.append('TAIL_FRACTION')
        
        # Waveform features (if available)
        sample_cols = [col for col in df.columns if col.startswith('SAMPLE')]
        if len(sample_cols) > 0:
            samples = df[sample_cols].values
            
            # Rise time (10% to 90%)
            baseline = samples[:, :20].mean(axis=1)
            pulse_height = baseline - samples.min(axis=1)
            
            threshold_10 = baseline[:, np.newaxis] - 0.1 * pulse_height[:, np.newaxis]
            threshold_90 = baseline[:, np.newaxis] - 0.9 * pulse_height[:, np.newaxis]
            
            rise_times = []
            for i in range(len(samples)):
                idx_10 = np.where(samples[i] < threshold_10[i])[0]
                idx_90 = np.where(samples[i] < threshold_90[i])[0]
                
                if len(idx_10) > 0 and len(idx_90) > 0:
                    rise_times.append(idx_90[0] - idx_10[0])
                else:
                    rise_times.append(0)
            
            features_list.append(np.array(rise_times))
            feature_names.append('RISE_TIME')
            
            # Peak position
            peak_positions = samples.argmin(axis=1)
            features_list.append(peak_positions)
            feature_names.append('PEAK_POSITION')
            
            # Tail slope (exponential decay constant)
            # Fit exponential to tail (samples after peak)
            tail_slopes = []
            for i in range(len(samples)):
                peak_idx = peak_positions[i]
                if peak_idx < len(samples[i]) - 50:
                    tail = samples[i, peak_idx:peak_idx+50]
                    x = np.arange(len(tail))
                    # log-linear fit
                    if (tail > 0).all():
                        slope = np.polyfit(x, np.log(tail + 1), 1)[0]
                        tail_slopes.append(slope)
                    else:
                        tail_slopes.append(0)
                else:
                    tail_slopes.append(0)
            
            features_list.append(np.array(tail_slopes))
            feature_names.append('TAIL_SLOPE')
        
        # Stack features
        features = np.column_stack(features_list)
        self.feature_names = feature_names
        
        return features, feature_names
    
    def train(self, df_train, test_size=0.2):
        """
        Train classifier
        
        Parameters:
        -----------
        df_train : DataFrame
            Must have PARTICLE column ('neutron' or 'gamma') and features
        test_size : float
            Fraction for validation
        
        Returns:
        --------
        results : dict
            Training results and metrics
        """
        # Extract features
        X, feature_names = self.extract_features(df_train)
        y = (df_train['PARTICLE'] == 'neutron').astype(int).values
        
        # Split train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train
        print(f"Training {self.method} classifier...")
        print(f"Features: {feature_names}")
        print(f"Training samples: {len(X_train)} ({y_train.sum()} neutrons, {len(y_train)-y_train.sum()} gammas)")
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        y_pred = self.model.predict(X_val_scaled)
        y_proba = self.model.predict_proba(X_val_scaled)[:, 1]
        
        # Metrics
        print(f"\nTraining accuracy: {train_score:.4f}")
        print(f"Validation accuracy: {val_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Gamma', 'Neutron']))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Gamma  Neutron")
        print(f"Actual Gamma     {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"Actual Neutron   {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        
        results = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'feature_names': feature_names,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return results
    
    def predict(self, df):
        """
        Predict particle type
        
        Parameters:
        -----------
        df : DataFrame
            Events to classify
        
        Returns:
        --------
        predictions : array
            0 = gamma, 1 = neutron
        probabilities : array
            Probability of being neutron
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X, _ = self.extract_features(df)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save(self, filename):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'method': self.method
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load trained model"""
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.method = data['method']
        self.is_fitted = True
        print(f"Model loaded from {filename}")


def plot_ml_performance(results):
    """
    Visualize ML classifier performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Confusion matrix
    ax = axes[0, 0]
    cm = results['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix', fontsize=14)
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=20)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Gamma', 'Neutron'])
    ax.set_yticklabels(['Gamma', 'Neutron'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.colorbar(im, ax=ax)
    
    # ROC curve
    ax = axes[0, 1]
    ax.plot(results['fpr'], results['tpr'], 'b-', linewidth=2, 
            label=f"ROC (AUC = {results['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Probability distribution
    ax = axes[1, 0]
    y_val = results['y_val']
    y_proba = results['y_proba']
    
    ax.hist(y_proba[y_val == 0], bins=50, alpha=0.6, label='Gamma', color='blue')
    ax.hist(y_proba[y_val == 1], bins=50, alpha=0.6, label='Neutron', color='red')
    ax.set_xlabel('Neutron Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Probability Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    
    # Feature importance (if available)
    ax = axes[1, 1]
    if hasattr(results, 'feature_importances'):
        importances = results['feature_importances']
        feature_names = results['feature_names']
        
        indices = np.argsort(importances)[::-1]
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Importance', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14)
    else:
        ax.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


# =============================================================================
# psd/ml/deep_learning.py
# =============================================================================

"""
Deep learning models for PSD with physics-informed loss functions
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


if TORCH_AVAILABLE:
    
    class WaveformDataset(Dataset):
        """
        PyTorch dataset for waveform data
        """
        
        def __init__(self, waveforms, labels):
            """
            Parameters:
            -----------
            waveforms : array (N, num_samples)
                Raw waveform samples
            labels : array (N,)
                0 = gamma, 1 = neutron
            """
            self.waveforms = torch.FloatTensor(waveforms)
            self.labels = torch.LongTensor(labels)
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.waveforms[idx], self.labels[idx]
    
    
    class CNN1DClassifier(nn.Module):
        """
        1D CNN for waveform classification
        Automatically learns discriminative features from raw waveforms
        """
        
        def __init__(self, input_length, num_classes=2):
            super(CNN1DClassifier, self).__init__()
            
            # Convolutional layers
            self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(32)
            self.pool1 = nn.MaxPool1d(2)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool2 = nn.MaxPool1d(2)
            
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool3 = nn.MaxPool1d(2)
            
            self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm1d(256)
            self.pool4 = nn.MaxPool1d(2)
            
            # Calculate flattened size
            self.flat_size = 256 * (input_length // 16)
            
            # Fully connected layers
            self.fc1 = nn.Linear(self.flat_size, 512)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 128)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, num_classes)
            
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # Add channel dimension
            x = x.unsqueeze(1)
            
            # Conv blocks
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.pool4(x)
            
            # Flatten
            x = x.view(-1, self.flat_size)
            
            # Fully connected
            x = self.relu(self.fc1(x))
            x = self.dropout1(x)
            x = self.relu(self.fc2(x))
            x = self.dropout2(x)
            x = self.fc3(x)
            
            return x
    
    
    class TransformerClassifier(nn.Module):
        """
        Transformer-based classifier for waveforms
        Captures long-range temporal dependencies
        """
        
        def __init__(self, input_length, d_model=128, nhead=8, num_layers=4, num_classes=2):
            super(TransformerClassifier, self).__init__()
            
            # Input embedding
            self.input_proj = nn.Linear(1, d_model)
            
            # Positional encoding
            self.pos_encoder = nn.Parameter(torch.randn(1, input_length, d_model))
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Classification head
            self.fc1 = nn.Linear(d_model, 256)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(256, num_classes)
            
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # x shape: (batch, seq_len)
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Project to d_model
            x = self.input_proj(x)  # (batch, seq_len, d_model)
            
            # Add positional encoding
            x = x + self.pos_encoder
            
            # Transformer encoding
            x = self.transformer(x)  # (batch, seq_len, d_model)
            
            # Global average pooling
            x = x.mean(dim=1)  # (batch, d_model)
            
            # Classification
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            
            return x
    
    
    class PhysicsInformedLoss(nn.Module):
        """
        Custom loss function incorporating physics knowledge
        
        Combines:
        1. Standard cross-entropy loss
        2. PSD consistency loss (predictions should respect PSD parameter)
        3. Energy dependence loss (performance should be consistent across energies)
        """
        
        def __init__(self, alpha=0.1, beta=0.05):
            super(PhysicsInformedLoss, self).__init__()
            self.ce_loss = nn.CrossEntropyLoss()
            self.alpha = alpha  # Weight for PSD consistency
            self.beta = beta    # Weight for energy dependence
        
        def forward(self, logits, labels, psd_values=None, energies=None):
            """
            Parameters:
            -----------
            logits : tensor (batch, 2)
                Model predictions
            labels : tensor (batch,)
                True labels
            psd_values : tensor (batch,)
                PSD parameter values
            energies : tensor (batch,)
                Event energies
            """
            # Standard cross-entropy
            ce = self.ce_loss(logits, labels)
            
            # Get predicted probabilities
            probs = torch.softmax(logits, dim=1)[:, 1]  # Neutron probability
            
            loss = ce
            
            # PSD consistency loss
            if psd_values is not None:
                # Predictions should correlate with PSD
                # Neutrons have high PSD, gammas have low PSD
                psd_consistency = torch.abs(probs - psd_values).mean()
                loss = loss + self.alpha * psd_consistency
            
            # Energy-dependent consistency
            if energies is not None:
                # Predictions should be smooth across energy bins
                # Sort by energy
                sorted_idx = torch.argsort(energies)
                sorted_probs = probs[sorted_idx]
                
                # Penalize large jumps in consecutive events
                energy_smoothness = torch.abs(sorted_probs[1:] - sorted_probs[:-1]).mean()
                loss = loss + self.beta * energy_smoothness
            
            return loss
    
    
    class DeepPSDClassifier:
        """
        Wrapper for deep learning PSD classification
        """
        
        def __init__(self, model_type='cnn', input_length=368, device='auto'):
            """
            Parameters:
            -----------
            model_type : str
                'cnn' or 'transformer'
            input_length : int
                Number of waveform samples
            device : str
                'auto', 'cuda', or 'cpu'
            """
            self.model_type = model_type
            self.input_length = input_length
            
            # Set device
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            print(f"Using device: {self.device}")
            
            # Initialize model
            if model_type == 'cnn':
                self.model = CNN1DClassifier(input_length).to(self.device)
            elif model_type == 'transformer':
                self.model = TransformerClassifier(input_length).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            print(f"Initialized {model_type} model")
            print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        def prepare_data(self, df):
            """
            Extract waveforms and labels from dataframe
            """
            # Get sample columns
            sample_cols = [col for col in df.columns if col.startswith('SAMPLE')]
            
            if len(sample_cols) == 0:
                raise ValueError("No waveform samples found in dataframe")
            
            waveforms = df[sample_cols].values.astype(np.float32)
            
            # Normalize waveforms (subtract baseline, normalize amplitude)
            baseline = waveforms[:, :20].mean(axis=1, keepdims=True)
            waveforms = baseline - waveforms  # Negative-going pulses
            
            # Normalize to [0, 1]
            max_vals = waveforms.max(axis=1, keepdims=True)
            max_vals[max_vals == 0] = 1  # Avoid division by zero
            waveforms = waveforms / max_vals
            
            # Get labels
            if 'PARTICLE' in df.columns:
                labels = (df['PARTICLE'] == 'neutron').astype(int).values
            else:
                labels = None
            
            # Get PSD and energy for physics-informed loss
            psd_values = df['PSD'].values if 'PSD' in df.columns else None
            energies = df['ENERGY_KEV'].values if 'ENERGY_KEV' in df.columns else df['ENERGY'].values if 'ENERGY' in df.columns else None
            
            return waveforms, labels, psd_values, energies
        
        def train(self, df_train, epochs=50, batch_size=64, learning_rate=0.001,
                 use_physics_loss=True, val_split=0.2):
            """
            Train deep learning model
            
            Parameters:
            -----------
            df_train : DataFrame
                Training data with waveforms and PARTICLE column
            epochs : int
                Number of training epochs
            batch_size : int
                Batch size
            learning_rate : float
                Learning rate
            use_physics_loss : bool
                Use physics-informed loss function
            val_split : float
                Validation split fraction
            
            Returns:
            --------
            history : dict
                Training history
            """
            # Prepare data
            waveforms, labels, psd_values, energies = self.prepare_data(df_train)
            
            # Split train/validation
            n_val = int(len(waveforms) * val_split)
            indices = np.random.permutation(len(waveforms))
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            
            train_waveforms, train_labels = waveforms[train_idx], labels[train_idx]
            val_waveforms, val_labels = waveforms[val_idx], labels[val_idx]
            
            # Create datasets
            train_dataset = WaveformDataset(train_waveforms, train_labels)
            val_dataset = WaveformDataset(val_waveforms, val_labels)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Loss and optimizer
            if use_physics_loss and psd_values is not None:
                criterion = PhysicsInformedLoss()
                psd_train = torch.FloatTensor(psd_values[train_idx])
                energy_train = torch.FloatTensor(energies[train_idx]) if energies is not None else None
            else:
                criterion = nn.CrossEntropyLoss()
            
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            
            # Training loop
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            
            print(f"\nTraining {self.model_type} model...")
            print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (waveforms_batch, labels_batch) in enumerate(train_loader):
                    waveforms_batch = waveforms_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = self.model(waveforms_batch)
                    
                    if use_physics_loss and psd_values is not None:
                        # Get corresponding PSD and energy values
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + len(waveforms_batch)
                        psd_batch = psd_train[start_idx:end_idx].to(self.device)
                        energy_batch = energy_train[start_idx:end_idx].to(self.device) if energy_train is not None else None
                        
                        loss = criterion(outputs, labels_batch, psd_batch, energy_batch)
                    else:
                        loss = criterion(outputs, labels_batch)
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels_batch.size(0)
                    correct += predicted.eq(labels_batch).sum().item()
                
                train_loss /= len(train_loader)
                train_acc = 100. * correct / total
                
                # Validation
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for waveforms_batch, labels_batch in val_loader:
                        waveforms_batch = waveforms_batch.to(self.device)
                        labels_batch = labels_batch.to(self.device)
                        
                        outputs = self.model(waveforms_batch)
                        loss = nn.CrossEntropyLoss()(outputs, labels_batch)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels_batch.size(0)
                        correct += predicted.eq(labels_batch).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = 100. * correct / total
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_loss)
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            print("\nTraining complete!")
            print(f"Final validation accuracy: {val_acc:.2f}%")
            
            return history
        
        def predict(self, df):
            """
            Predict particle types
            
            Returns:
            --------
            predictions : array
                0 = gamma, 1 = neutron
            probabilities : array
                Neutron probability
            """
            waveforms, _, _, _ = self.prepare_data(df)
            
            self.model.eval()
            predictions = []
            probabilities = []
            
            with torch.no_grad():
                for i in range(0, len(waveforms), 64):
                    batch = torch.FloatTensor(waveforms[i:i+64]).to(self.device)
                    outputs = self.model(batch)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    
                    predictions.extend(preds)
                    probabilities.extend(probs)
            
            return np.array(predictions), np.array(probabilities)
        
        def save(self, filename):
            """Save model"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'input_length': self.input_length
            }, filename)
            print(f"Model saved to {filename}")
        
        def load(self, filename):
            """Load model"""
            checkpoint = torch.load(filename, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model_type = checkpoint['model_type']
            self.input_length = checkpoint['input_length']
            print(f"Model loaded from {filename}")


def plot_training_history(history):
    """
    Plot training curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig