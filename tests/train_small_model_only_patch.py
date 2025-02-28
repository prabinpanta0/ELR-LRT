import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from elr_lrt import patch_sequence

# Toy dataset: Short text phrases encoded as byte sequences
def generate_data():
    phrases = [
        "hello world",
        "this is a test",
        "machine learning",
        "artificial intelligence",
        "byte patching",
        "low resource",
        "efficient model",
        "reinforcement learning",
        "dynamic allocation",
        "entropy based"
    ]
    return [list(phrase.encode('utf-8')) for phrase in phrases]

# Improved LSTM Model using an LSTM for sequence modeling
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])  # Use the last layer's hidden state
        return output

# Pad or truncate a patch to a fixed length (pad_length)
def pad_patch(patch, pad_length):
    patch_tensor = torch.tensor(patch, dtype=torch.float32)
    current_len = patch_tensor.size(0)
    if current_len < pad_length:
        pad_size = pad_length - current_len
        patch_tensor = torch.cat([patch_tensor, torch.zeros(pad_size)])
    elif current_len > pad_length:
        patch_tensor = patch_tensor[:pad_length]
    return patch_tensor

# Normalize a tensor to [0, 1] given a maximum value (e.g., 255)
def normalize_tensor(tensor, max_value=255.0):
    return tensor / max_value

# Prepare input: process each patch, pad/truncate to fixed length, and normalize.
def prepare_input(sequence, k, theta, theta_r, pad_length):
    patches = patch_sequence(sequence, k, theta, theta_r)
    patch_tensors = [pad_patch(p, pad_length) for p in patches]
    # Normalize each patch so values are in [0,1]
    patch_tensors = [normalize_tensor(pt) for pt in patch_tensors]
    # Stack patches to get a tensor of shape: (1, num_patches, pad_length)
    return torch.stack(patch_tensors).unsqueeze(0)

# Train the model with normalized targets and extended training epochs.
def train_model(model, data, epochs=100, lr=0.01, k=3, theta=1.5, theta_r=0.5, pad_length=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for seq in data:
            # Use all but the last byte as input; predict the last byte.
            input_seq = prepare_input(seq[:-1], k, theta, theta_r, pad_length)
            # Normalize target to [0,1]
            target = torch.tensor([[seq[-1] / 255.0]], dtype=torch.float32)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(data):.4f}")

# Test the model: scale predictions back to the original range.
def test_model(model, test_data, k=3, theta=1.5, theta_r=0.5, pad_length=10):
    model.eval()
    with torch.no_grad():
        for seq in test_data:
            input_seq = prepare_input(seq[:-1], k, theta, theta_r, pad_length)
            # Multiply by 255 to scale the prediction back
            predicted = model(input_seq).item() * 255.0
            actual = seq[-1]
            print(f"Predicted: {predicted:.2f}, Actual: {actual}")

if __name__ == "__main__":
    # Generate training and test data
    data = generate_data()
    train_data = data[:8]  # First 8 phrases for training
    test_data = data[8:]   # Last 2 phrases for testing

    # Patching hyperparameters
    k = 3
    theta = 1.5
    theta_r = 0.5

    # Determine a fixed patch length using a sample from the training data.
    sample_patches = patch_sequence(train_data[0][:-1], k, theta, theta_r)
    pad_length = len(sample_patches[0])
    print("Detected patch dimension (pad_length):", pad_length)

    # Model hyperparameters
    input_dim = pad_length  # Fixed input dimension per patch
    hidden_dim = 64
    output_dim = 1        # Predicting a single byte value (normalized)
    num_layers = 2

    # Initialize the model
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    print("Training the model:")
    train_model(model, train_data, epochs=100, lr=0.01, k=k, theta=theta, theta_r=theta_r, pad_length=pad_length)

    print("\nTesting the model:")
    test_model(model, test_data, k=k, theta=theta, theta_r=theta_r, pad_length=pad_length)
