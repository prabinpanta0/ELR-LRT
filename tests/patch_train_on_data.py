import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from elr_lrt import patch_sequence

def load_dataset(file_path):
    """
    Loads a text dataset from the given file.
    Each non-empty line is treated as one training sample.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Convert each line to a list of byte values.
    data = [list(line.encode('utf-8')) for line in lines]
    return data

class LSTMModel(nn.Module):
    """
    A simple LSTM model for next-byte prediction.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

def pad_patch(patch, pad_length):
    """
    Pads or truncates a patch (list of byte values) to a fixed length.
    """
    patch_tensor = torch.tensor(patch, dtype=torch.float32)
    current_len = patch_tensor.size(0)
    if current_len < pad_length:
        pad_size = pad_length - current_len
        patch_tensor = torch.cat([patch_tensor, torch.zeros(pad_size)])
    elif current_len > pad_length:
        patch_tensor = patch_tensor[:pad_length]
    return patch_tensor

def normalize_tensor(tensor, max_value=255.0):
    """
    Normalizes tensor values to [0, 1] range.
    """
    return tensor / max_value

def prepare_input(sequence, k, theta, theta_r, pad_length):
    """
    Uses the ELR-LRT patch_sequence to segment the input byte sequence,
    pads/truncates each patch to fixed length, and normalizes the values.
    Returns a tensor of shape (1, num_patches, pad_length).
    """
    patches = patch_sequence(sequence, k, theta, theta_r)
    patch_tensors = [pad_patch(p, pad_length) for p in patches]
    patch_tensors = [normalize_tensor(pt) for pt in patch_tensors]
    return torch.stack(patch_tensors).unsqueeze(0)

def train_model(model, data, epochs, lr, k, theta, theta_r, pad_length):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        sample_count = 0
        for seq in data:
            # Skip samples that are too short to form an input-target pair.
            if len(seq) < 2:
                continue
            # Use all but the last byte as input and predict the last byte.
            input_seq = prepare_input(seq[:-1], k, theta, theta_r, pad_length)
            target = torch.tensor([[seq[-1] / 255.0]], dtype=torch.float32)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sample_count += 1
        avg_loss = total_loss / sample_count if sample_count > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

def test_model(model, data, k, theta, theta_r, pad_length):
    model.eval()
    with torch.no_grad():
        for seq in data:
            if len(seq) < 2:
                continue
            input_seq = prepare_input(seq[:-1], k, theta, theta_r, pad_length)
            predicted = model(input_seq).item() * 255.0
            actual = seq[-1]
            print(f"Predicted: {predicted:.2f}, Actual: {actual}")

def main():
    # Path to the real-world dataset file (each line is a sample text)
    dataset_file = os.path.join("tests", "data.txt")
    if not os.path.exists(dataset_file):
        print(f"Dataset file not found: {dataset_file}")
        return

    data = load_dataset(dataset_file)
    print(f"Loaded {len(data)} samples from dataset.")

    # Split dataset into 80% training and 20% testing.
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Patching hyperparameters (adjust as needed)
    k = 3
    theta = 1.5
    theta_r = 0.5

    # Determine pad_length from the first training sample
    sample_patches = patch_sequence(train_data[0][:-1], k, theta, theta_r)
    pad_length = len(sample_patches[0])
    print(f"Detected patch dimension (pad_length): {pad_length}")

    # Model hyperparameters (you might want to increase hidden_dim for a larger dataset)
    input_dim = pad_length
    hidden_dim = 128
    output_dim = 1  # predicting a single byte (normalized)
    num_layers = 4

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    
    # Training hyperparameters
    epochs = 50
    lr = 0.005

    print("Training model on real-world dataset...")
    train_model(model, train_data, epochs, lr, k, theta, theta_r, pad_length)

    print("Testing model on real-world dataset...")
    test_model(model, test_data, k, theta, theta_r, pad_length)

    # print metrics
    print("Metrics:")
    print("Model:", model)
    print("Split Index:", split_idx)
    print("Sample Patches:", sample_patches)
    print("Pad Length:", pad_length)
    print("Input Dimension:", input_dim)
    print("Hidden Dimension:", hidden_dim)
    print("Output Dimension:", output_dim)
    print("Number of Layers:", num_layers)
    print("Epochs:", epochs)
    print("Learning Rate:", lr)



if __name__ == "__main__":
    main()
