"""
Neural reconstruction of a handwritten curve with thinning + MLP.

Usage:
    python handwritten_curve_fit.py input.png [optional_out_prefix]

This script will:
1. Load a handwriting image (e.g. a single stroke or character).
2. Binarize and skeletonize (thinning) to get a 1-pixel-wide curve.
3. Extract skeleton pixels, order them into a path, and parameterize by arc length s in [0,1].
4. Train a small MLP to learn f(s) = (x, y).
5. Save visualizations (preprocessing, learning snapshots, loss curve) into a timestamped folder.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque


# -----------------------------
# 1. Load image and skeletonize
# -----------------------------

def load_and_skeletonize(path, out_dir, show_intermediate=True):
    """
    Load an image, convert to grayscale, binarize, and skeletonize.
    Returns a binary skeleton image (True at skeleton pixels).
    Saves intermediate visualization into out_dir.
    """
    # Load as grayscale (float in [0,1])
    img = imread(path, as_gray=True)

    # Otsu thresholding
    thresh = threshold_otsu(img)

    # Assume dark ink on light background:
    #   binary = True where ink is present
    binary = img < thresh

    # Skeletonize (Zhang-Suen-like thinning)
    skeleton = skeletonize(binary)

    if show_intermediate:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        ax = axes.ravel()
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Grayscale")
        ax[1].imshow(binary, cmap="gray")
        ax[1].set_title("Binary")
        ax[2].imshow(skeleton, cmap="gray")
        ax[2].set_title("Skeleton")
        for a in ax:
            a.axis("off")
        plt.tight_layout()

        out_path = out_dir / "preprocessing_grayscale_binary_skeleton.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return skeleton


# -----------------------------
# 2. Extract and order skeleton
# -----------------------------

def extract_ordered_curve_from_skeleton(skeleton):
    """
    Given a skeleton (bool array), build a graph of 8-connected pixels,
    then compute an Eulerian trail (walk that covers all edges, revisiting
    intersection nodes as needed). This gives a single continuous path that
    includes self-intersections instead of skipping branches.

    Returns:
        coords_ordered: array (M, 2) with normalized (x, y) coords in [0,1],
                        with y flipped so orientation matches the original image.
    """
    # Get (row, col) indices where skeleton is True
    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("No skeleton pixels found – check your image / thresholding.")

    N = xs.size
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2), (x, y) in pixel coords

    # Map (row, col) -> node index
    coord_to_idx = {(ys[i], xs[i]): i for i in range(N)}

    # Build adjacency list (undirected, 8-connected)
    adj = [[] for _ in range(N)]
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    for i in range(N):
        r, c = ys[i], xs[i]
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            j = coord_to_idx.get((nr, nc))
            if j is not None:
                adj[i].append(j)

    # Choose a start node for Eulerian trail:
    # - if there are vertices of odd degree, start at one of them
    # - otherwise, just start at 0
    odd_vertices = [i for i in range(N) if len(adj[i]) % 2 == 1]
    if len(odd_vertices) >= 1:
        start = odd_vertices[0]
    else:
        start = 0

    # Hierholzer's algorithm for Eulerian trail in an undirected graph
    adj_copy = [nbrs.copy() for nbrs in adj]
    stack = [start]
    trail = []

    while stack:
        v = stack[-1]
        if adj_copy[v]:
            u = adj_copy[v].pop()
            # remove edge v-u from u's list as well
            try:
                adj_copy[u].remove(v)
            except ValueError:
                pass  # just in case we already removed it
            stack.append(u)
        else:
            trail.append(stack.pop())

    # trail is a list of node indices; reverse to get forward order
    trail = trail[::-1]

    # Convert to pixel coordinates
    coords = coords_pix[trail]  # (M, 2), M ≈ number of edges + 1

    # Normalize to [0,1]
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min + 1e-8)
    coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min + 1e-8)

    # Flip y so plot orientation matches original image
    coords[:, 1] = 1.0 - coords[:, 1]

    return coords

def parameterize_by_arclength(coords):
    """
    Given ordered 2D coordinates, compute arc length parameter s in [0,1].
    Returns:
        s: (N,) array in [0,1]
        coords: unchanged (N,2)
    """
    diffs = coords[1:] - coords[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    s = arc / (arc[-1] + 1e-8)
    return s, coords


# -----------------------------
# 3. Define MLP model
# -----------------------------

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class CurveMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            Sin(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # outputs (x, y)
        )

    def forward(self, s):
        s_scaled = s * 2 * np.pi  # scale input to [0, 2pi]
        return self.net(s_scaled)


# -----------------------------
# 4. Training + snapshots
# -----------------------------

def train_curve_mlp(s, coords, num_epochs=3000, lr=1e-3):
    """
    Train MLP to fit f(s) ~ coords, and record:
      - loss per epoch
      - total gradient norm per epoch
      - per-parameter gradient norms per epoch
    """
    # Prepare tensors
    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(1)       # (N,1)
    points_tensor = torch.tensor(coords, dtype=torch.float32)          # (N,2)

    model = CurveMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Choose snapshot epochs (1-based indexing)
    save_epochs = [0, 50, 200, 800, 1500, num_epochs]
    snapshots = {}

    # For analysis
    losses = []
    total_grad_norms = []
    grad_history = {name: [] for name, _ in model.named_parameters()}

    # Save prediction at epoch 0 (untrained)
    with torch.no_grad():
        pred0 = model(s_tensor).detach().numpy()
    snapshots[0] = pred0

    for epoch in range(num_epochs):
        # Forward
        pred = model(s_tensor)
        loss = criterion(pred, points_tensor)

        optimizer.zero_grad()
        loss.backward()

        # ---- Gradient analysis before the optimizer step ----
        total_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                gnorm = p.grad.detach().norm().item()
                grad_history[name].append(gnorm)
                total_norm += gnorm ** 2
            else:
                grad_history[name].append(0.0)
        total_norm = total_norm ** 0.5
        total_grad_norms.append(total_norm)
        # -----------------------------------------------

        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Loss = {loss.item():.6f}, "
                f"Total grad norm = {total_norm:.6f}"
            )

        current_epoch = epoch + 1
        if current_epoch in save_epochs:
            with torch.no_grad():
                pred_points = model(s_tensor).detach().numpy()
            snapshots[current_epoch] = pred_points

    return model, snapshots, losses, total_grad_norms, grad_history, save_epochs


# -----------------------------
# 5. Visualization
# -----------------------------

def plot_snapshots(snapshots, true_coords, save_epochs, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, epoch in enumerate(save_epochs):
        ax = axes[i]
        ax.plot(true_coords[:, 0], true_coords[:, 1],
                'k.', markersize=2, label="True skeleton")
        pred = snapshots[epoch]
        ax.plot(pred[:, 0], pred[:, 1], 'r-', linewidth=1.0, label="MLP prediction")
        ax.set_title(f"Epoch {epoch}")
        ax.set_aspect('equal')
        ax.legend()

    plt.tight_layout()
    out_path = out_dir / "curve_reconstruction_snapshots.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_loss_and_grad(losses, total_grad_norms, out_dir):
    fig, ax1 = plt.subplots(figsize=(6, 4))

    epochs = np.arange(len(losses))

    # Loss on left axis (log scale)
    color1 = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (log scale)", color=color1)
    ax1.set_yscale("log")
    ax1.plot(epochs, losses, color=color1, label="Loss")
    ax1.tick_params(axis='y', labelcolor=color1)

    # Grad norm on right axis (log scale)
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Total gradient norm (log scale)", color=color2)
    ax2.set_yscale("log")
    ax2.plot(epochs, total_grad_norms, color=color2, alpha=0.7, label="Grad norm")
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    out_path = out_dir / "loss_and_gradnorm_log.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layer_grad_history(grad_history, out_dir, max_layers=4):
    """
    grad_history: dict name -> list of norms per epoch
    Plots up to max_layers parameter tensors (typically weights) to avoid clutter.
    """
    # pick a few interesting parameters (usually the weight matrices)
    weight_keys = [k for k in grad_history.keys() if "weight" in k]
    weight_keys = weight_keys[:max_layers]

    fig = plt.figure(figsize=(6, 4))
    epochs = np.arange(len(next(iter(grad_history.values()))))

    for k in weight_keys:
        plt.plot(epochs, grad_history[k], label=k)

    plt.xlabel("Epoch")
    plt.ylabel("Gradient norm")
    plt.title("Per-layer gradient norms")
    plt.legend()
    plt.grid(True)

    out_path = out_dir / "layer_gradnorms.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_svg_curve(points, path):
    """
    Save a polyline curve as an SVG.

    points: array (M, 2) in normalized [0,1] coordinates (x,y) with y upward.
    SVG has viewBox 0..1 x 0..1. We flip y because SVG's y-axis goes downward.
    """
    path = Path(path)
    with open(path, "w") as f:
        f.write('<svg viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<polyline points="')
        for x, y in points:
            y_svg = 1.0 - y  # flip back for SVG coord system
            f.write(f"{x},{y_svg} ")
        f.write('" fill="none" stroke="black" stroke-width="0.002"/>\n')
        f.write('</svg>\n')


def save_png_curve(points, path, dpi=300):
    """
    Save a curve as a PNG using matplotlib.

    points: array (M, 2) in normalized [0,1] coordinates (x,y) with y upward.
    """
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.plot(points[:, 0], points[:, 1], 'k-')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def export_curve_reconstruction(model, out_dir, num_points=2000):
    """
    Evaluate the trained model on a dense s-grid and export:
      - model weights (.pt)
      - vector curve (.svg)
      - raster curve (.png)
    into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save model weights
    weights_path = out_dir / "curve_model_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to: {weights_path}")

    # 2) Evaluate dense curve
    model.eval()
    s_dense = torch.linspace(0.0, 1.0, num_points).unsqueeze(1)
    with torch.no_grad():
        pts = model(s_dense).cpu().numpy()  # shape (num_points, 2)

    # 3) Save SVG
    svg_path = out_dir / "curve_reconstruction.svg"
    save_svg_curve(pts, svg_path)
    print(f"Saved SVG curve to: {svg_path}")

    # 4) Save PNG
    png_path = out_dir / "curve_reconstruction.png"
    save_png_curve(pts, png_path)
    print(f"Saved PNG curve to: {png_path}")

# -----------------------------
# 6. Main script
# -----------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python handwritten_curve_fit.py input.png epochs learning_rate [optional_out_prefix]")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: input file not found: {img_path}")
        sys.exit(1)

    # Optional second argument: out_prefix
    out_prefix = sys.argv[4] if len(sys.argv) >= 5 else None

    # --- Follow your pattern with Path and datetime ---
    img_path = Path(img_path)
    if out_prefix is None:
        out_prefix = img_path.with_suffix("")
    else:
        out_prefix = Path(out_prefix)

    # --- Run ID (for all outputs) ---
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run ID: {run_id}")
    tagged_prefix = f"{out_prefix}_ID-{run_id}"

    # --- Create output directory ---
    out_dir = Path(tagged_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to directory: {out_dir}")

    # 1) Load + skeletonize
    skeleton = load_and_skeletonize(str(img_path), out_dir, show_intermediate=True)

    # 2) Extract ordered curve coordinates
    coords_ordered = extract_ordered_curve_from_skeleton(skeleton)

    # 3) Parameterize by arc length
    s, coords = parameterize_by_arclength(coords_ordered)

    # 4) Train MLP
    epochs = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    model, snapshots, losses, total_grad_norms, grad_history, save_epochs = train_curve_mlp(
        s, coords, num_epochs=epochs, lr=learning_rate
    )

    # 5) Plot snapshots
    plot_snapshots(snapshots, coords, save_epochs, out_dir)

    # 6) Plot loss curve
    plot_loss_and_grad(losses, total_grad_norms, out_dir)
    plot_layer_grad_history(grad_history, out_dir)

    export_curve_reconstruction(model, out_dir, num_points=2000)

    print("Done.")

if __name__ == "__main__":
    main()