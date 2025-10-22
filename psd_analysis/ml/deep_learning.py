"""
Deep Learning models for PSD classification
Requires PyTorch: pip install torch torchvision
"""

import numpy as np
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Deep learning models will not work. "
                 "Install with: pip install torch torchvision")


if TORCH_AVAILABLE:

    class WaveformDataset(Dataset):
        """PyTorch dataset for waveform data"""

        def __init__(self, waveforms, labels, transform=None):
            """
            Parameters:
            -----------
            waveforms : array (N, samples)
                Raw waveform data
            labels : array (N,)
                Binary labels (0=gamma, 1=neutron)
            transform : callable, optional
                Data augmentation function
            """
            self.waveforms = torch.FloatTensor(waveforms)
            self.labels = torch.LongTensor(labels)
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            waveform = self.waveforms[idx]
            label = self.labels[idx]

            if self.transform:
                waveform = self.transform(waveform)

            return waveform.unsqueeze(0), label  # Add channel dimension


    class CNN1DClassifier(nn.Module):
        """1D CNN for waveform classification"""

        def __init__(self, input_length=368, n_classes=2):
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
            self.fc3 = nn.Linear(128, n_classes)

        def forward(self, x):
            # Conv blocks
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = self.pool4(F.relu(self.bn4(self.conv4(x))))

            # Flatten
            x = x.view(x.size(0), -1)

            # FC layers
            x = self.dropout1(F.relu(self.fc1(x)))
            x = self.dropout2(F.relu(self.fc2(x)))
            x = self.fc3(x)

            return x


    class TransformerClassifier(nn.Module):
        """Transformer-based classifier for waveforms"""

        def __init__(self, input_length=368, d_model=128, nhead=8,
                     num_layers=4, n_classes=2):
            super(TransformerClassifier, self).__init__()

            # Input projection
            self.input_proj = nn.Linear(1, d_model)

            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, max_len=input_length)

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
            self.fc = nn.Linear(d_model, n_classes)

        def forward(self, x):
            # x: (batch, 1, seq_len)
            x = x.transpose(1, 2)  # (batch, seq_len, 1)

            # Project to d_model
            x = self.input_proj(x)  # (batch, seq_len, d_model)

            # Add positional encoding
            x = self.pos_encoder(x)

            # Transformer
            x = self.transformer(x)

            # Global average pooling
            x = x.mean(dim=1)  # (batch, d_model)

            # Classify
            x = self.fc(x)

            return x


    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer"""

        def __init__(self, d_model, max_len=5000):
            super(PositionalEncoding, self).__init__()

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                (-np.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]


    class PhysicsInformedLoss(nn.Module):
        """Custom loss incorporating physics knowledge"""

        def __init__(self, alpha=0.1, beta=0.05):
            super(PhysicsInformedLoss, self).__init__()
            self.alpha = alpha
            self.beta = beta
            self.ce_loss = nn.CrossEntropyLoss()

        def forward(self, outputs, labels, psd_values=None, energies=None):
            # Standard classification loss
            ce = self.ce_loss(outputs, labels)

            total_loss = ce

            # PSD consistency loss (if PSD values provided)
            if psd_values is not None and self.alpha > 0:
                probs = F.softmax(outputs, dim=1)[:, 1]  # P(neutron)
                # Neutrons should have high PSD, gammas low PSD
                psd_consistency = torch.abs(probs - psd_values)
                psd_loss = psd_consistency.mean()
                total_loss = total_loss + self.alpha * psd_loss

            # Energy smoothness loss (predictions should be smooth across energy)
            if energies is not None and self.beta > 0:
                # Sort by energy
                sorted_idx = torch.argsort(energies)
                sorted_probs = F.softmax(outputs, dim=1)[sorted_idx, 1]

                # Penalize large differences between adjacent energies
                energy_smoothness = torch.abs(sorted_probs[1:] - sorted_probs[:-1]).mean()
                total_loss = total_loss + self.beta * energy_smoothness

            return total_loss


    class DeepPSDClassifier:
        """Wrapper for deep learning PSD classification"""

        def __init__(self, model_type='cnn', input_length=368, device='auto'):
            """
            Parameters:
            -----------
            model_type : str
                'cnn' or 'transformer'
            input_length : int
                Waveform length
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

            # Create model
            if model_type == 'cnn':
                self.model = CNN1DClassifier(input_length=input_length)
            elif model_type == 'transformer':
                self.model = TransformerClassifier(input_length=input_length)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.model.to(self.device)
            self.is_fitted = False

        def train(self, df_train, epochs=50, batch_size=64, learning_rate=0.001,
                 use_physics_loss=True, validation_split=0.2):
            """
            Train the model

            Parameters:
            -----------
            df_train : DataFrame
                Training data with waveform samples and 'PARTICLE' column
            epochs : int
                Number of training epochs
            batch_size : int
                Batch size
            learning_rate : float
                Learning rate
            use_physics_loss : bool
                Use physics-informed loss
            validation_split : float
                Validation fraction

            Returns:
            --------
            history : dict
                Training history
            """
            # Extract waveforms
            sample_cols = [col for col in df_train.columns if col.startswith('SAMPLE')]
            waveforms = df_train[sample_cols].values
            labels = (df_train['PARTICLE'] == 'neutron').astype(int).values

            # Get PSD and energy if available
            psd_values = df_train['PSD'].values if 'PSD' in df_train.columns else None
            energies = df_train['ENERGY_KEV'].values if 'ENERGY_KEV' in df_train.columns else None

            # Train/val split
            n_train = int(len(waveforms) * (1 - validation_split))
            train_wf, val_wf = waveforms[:n_train], waveforms[n_train:]
            train_labels, val_labels = labels[:n_train], labels[n_train:]

            # Create datasets
            train_dataset = WaveformDataset(train_wf, train_labels)
            val_dataset = WaveformDataset(val_wf, val_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Loss and optimizer
            if use_physics_loss:
                criterion = PhysicsInformedLoss()
            else:
                criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # Training loop
            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

            for epoch in range(epochs):
                # Train
                self.model.train()
                train_loss, train_correct = 0, 0

                for wf_batch, label_batch in train_loader:
                    wf_batch = wf_batch.to(self.device)
                    label_batch = label_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(wf_batch)
                    loss = criterion(outputs, label_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_correct += (outputs.argmax(1) == label_batch).sum().item()

                train_loss /= len(train_loader)
                train_acc = train_correct / len(train_dataset)

                # Validate
                self.model.eval()
                val_loss, val_correct = 0, 0

                with torch.no_grad():
                    for wf_batch, label_batch in val_loader:
                        wf_batch = wf_batch.to(self.device)
                        label_batch = label_batch.to(self.device)

                        outputs = self.model(wf_batch)
                        loss = criterion(outputs, label_batch)

                        val_loss += loss.item()
                        val_correct += (outputs.argmax(1) == label_batch).sum().item()

                val_loss /= len(val_loader)
                val_acc = val_correct / len(val_dataset)

                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            self.is_fitted = True
            return history

        def predict(self, df):
            """Predict on new data"""
            if not self.is_fitted:
                raise ValueError("Model not trained yet")

            sample_cols = [col for col in df.columns if col.startswith('SAMPLE')]
            waveforms = df[sample_cols].values

            dataset = WaveformDataset(waveforms, np.zeros(len(waveforms)))
            loader = DataLoader(dataset, batch_size=64, shuffle=False)

            self.model.eval()
            all_probs = []

            with torch.no_grad():
                for wf_batch, _ in loader:
                    wf_batch = wf_batch.to(self.device)
                    outputs = self.model(wf_batch)
                    probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs)

            predictions = (np.array(all_probs) > 0.5).astype(int)
            return predictions, np.array(all_probs)

        def save(self, filepath):
            """Save model"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'input_length': self.input_length
            }, filepath)

        def load(self, filepath):
            """Load model"""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_fitted = True


    def plot_training_history(history, save_path=None):
        """Plot training curves"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


else:
    # Dummy classes when PyTorch not available
    class DeepPSDClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed. Install with: pip install torch torchvision")

    def plot_training_history(*args, **kwargs):
        raise ImportError("PyTorch not installed.")
