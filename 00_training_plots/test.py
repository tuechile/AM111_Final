# curve_fit_snapshots.py
# Fit a 2D curve with a tiny neural net and show 6 snapshots of learning.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# 1. Create curve data 
# ---------------------

# Number of sample points on the curve
N = 20

# Parameter for the circle
theta = np.linspace(0, 2 * np.pi, N)

# True x, y coordinates of the curve (unit circle)
x = theta
y = np.sin(theta)


theta_dense = np.linspace(0, 2*np.pi, 400)
x_dense = theta_dense
y_dense = np.sin(theta_dense)
original_curve = np.stack([x_dense, y_dense], axis=1)

# Stack into shape (N, 2): each row is [x_i, y_i]
curve_points = np.stack([x, y], axis=1)

# Parameter s in [0, 1] (this is what the NN will take as input)
s = np.linspace(0, 1, N)

# Convert to PyTorch tensors
s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(1)   # shape (N, 1)
points_tensor = torch.tensor(curve_points, dtype=torch.float32)  # shape (N, 2)

# -----------------------------
# 2. Define the MLP model
# -----------------------------

class CurveMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),   # input: s (1D)  ->  64 hidden units
            nn.Tanh(),
            nn.Linear(64, 64),  # hidden layer
            nn.Tanh(),
            nn.Linear(64, 2)    # output: (x, y)
        )

    def forward(self, s):
        return self.net(s)

model = CurveMLP()

# Loss function: mean squared error between predicted and true (x, y)
criterion = nn.MSELoss()

# Optimizer: Adam gradient descent
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 3. Training + snapshot saving
# -----------------------------

num_epochs = 5000

# Epochs at which we want to save a snapshot of the predicted curve
save_epochs = [0, 200, 400, 800, 2000, 5000]
snapshots = {}

# List to store MSE values for plotting
mse_history = []

# Save prediction at epoch 0 (before any training)
with torch.no_grad():
    pred0 = model(s_tensor).detach().numpy()
snapshots[0] = pred0
mse_history.append(criterion(model(s_tensor), points_tensor).item())

for epoch in range(num_epochs):
    # Forward pass: compute current prediction
    pred = model(s_tensor)  # shape (N, 2)

    # Compute loss
    loss = criterion(pred, points_tensor)

    # Backward pass + gradient update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store MSE for this epoch
    mse_history.append(loss.item())

    # Print progress every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss = {loss.item():.6f}")

    # Save snapshots at selected epochs (1-based)
    current_epoch = epoch + 1
    if current_epoch in save_epochs:
        with torch.no_grad():
            pred_points = model(s_tensor).detach().numpy()
        snapshots[current_epoch] = pred_points

# -----------------------------
# 4. Plot 6 snapshots in a grid
# -----------------------------

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, epoch in enumerate(save_epochs):
    ax = axes[i]

    # True curve
    ax.plot(curve_points[:, 0], curve_points[:, 1], label='True', color='black')

    ax.plot(original_curve[:, 0], original_curve[:, 1], 
        color='green', linewidth=1, label='Original Function')

    # Predicted curve at this epoch
    pred = snapshots[epoch]
    ax.plot(pred[:, 0], pred[:, 1], '--', label='Predicted', color='red')

    ax.set_title(f"Epoch {epoch}")
    ax.set_aspect('equal')
    ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 5. Plot MSE over epochs
# -----------------------------

def plot_mse_over_epochs(mse_history):
    """Plot MSE (loss) over training epochs."""
    epochs = np.arange(len(mse_history))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mse_history, 'b-', linewidth=2, label='Training MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE Over Training Epochs')
    plt.yscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add vertical lines at snapshot epochs
    for epoch in save_epochs:
        if epoch < len(mse_history):
            plt.axvline(x=epoch, color='r', linestyle='--', alpha=0.7, linewidth=1)
            plt.text(epoch + 50, mse_history[epoch] * 1.1, f'Epoch {epoch}', 
                    rotation=90, verticalalignment='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Plot MSE over epochs
plot_mse_over_epochs(mse_history)
