"""
Neural reconstruction of a handwritten curve with thinning + MLP
and learnable point importance weights (Method B).

Usage:
    python handwritten_curve_fit.py input.png epochs learning_rate [optional_out_prefix]

Pipeline:
1. Load a handwriting image (e.g. a single stroke or character).
2. Binarize and skeletonize (thinning) to get a 1-pixel-wide curve.
3. Extract skeleton pixels, order them into a path, and parameterize by arc length s in [0,1].
4. Phase 1 (Method B):
    - Train an MLP f(s) = (x, y) together with learnable per-point weights w_i ∈ (0,1).
    - Loss = weighted error + sparsity penalty on weights.
    - This makes the model focus on "important" points.
5. Use the learned weights to pick the most important points (top fraction).
6. Phase 2:
    - Retrain a standard MLP on only those selected points.
7. Export the final model (weights, SVG, PNG) and some diagnostics into a timestamped folder.
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


# =============================
# Hyperparameters (tune here)
# =============================

# Phase 1: importance learning
IMPORTANCE_EPOCHS_MULTIPLIER = 1.0   # importance_epochs = IMPORTANCE_EPOCHS_MULTIPLIER * epochs
LAMBDA_SPARSITY = 1e-3              # weight sparsity penalty
SIGMOID_EPS = 1e-8                  # numerical epsilon

# How many points to keep after importance learning
KEEP_FRACTION = 0.9   # keep top 90% most important points
MIN_POINTS = 100      # never go below this many points

# Architecture (same for importance and final model)
HIDDEN_DIM = 64
NUM_HIDDEN_LAYERS = 2

# -----------------------------
# 1. Load image and skeletonize
# -----------------------------

def load_and_skeletonize(path, out_dir, show_intermediate=True):
    """
    Load an image, convert to grayscale, binarize, and skeletonize.
    Returns a binary skeleton image (True at skeleton pixels).
    Saves intermediate visualization into out_dir.
    """
    img = imread(path, as_gray=True)  # float in [0,1]

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
    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("No skeleton pixels found – check your image / thresholding.")

    N = xs.size
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2)

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

    # Choose start node: odd-degree vertex if exists, else 0
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
            try:
                adj_copy[u].remove(v)
            except ValueError:
                pass
            stack.append(u)
        else:
            trail.append(stack.pop())

    trail = trail[::-1]  # forward order of node indices

    coords = coords_pix[trail]  # (M, 2)

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
    def __init__(self, hidden_dim=64, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), Sin()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        s_scaled = s * 2 * np.pi  # scale input to [0, 2π]
        return self.net(s_scaled)


# -----------------------------
# 4. Training helpers
# -----------------------------

def train_curve_mlp(s, coords, num_epochs=3000, lr=1e-3,
                    hidden_dim=64, num_hidden_layers=2,
                    verbose=True):
    """
    Standard training: minimize plain MSE on a fixed set of points.
    """
    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
    pts_tensor = torch.tensor(coords, dtype=torch.float32)

    model = CurveMLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(num_epochs):
        pred = model(s_tensor)
        loss = criterion(pred, pts_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch + 1) % 200 == 0:
            print(f"  [Final] Epoch {epoch+1}/{num_epochs}, loss={loss.item():.6e}")

    return model, losses


def compute_mse(model, s, coords):
    """
    Compute MSE between model predictions and given coordinates.
    """
    model.eval()
    with torch.no_grad():
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
        pts_pred = model(s_tensor).cpu().numpy()
    mse = np.mean((pts_pred - coords) ** 2)
    return mse


# -----------------------------
# 5. Method B: learnable weights
# -----------------------------

def train_with_importance_weights(
    s_full,
    coords_full,
    num_epochs,
    lr,
    hidden_dim=64,
    num_hidden_layers=2,
    lambda_sparsity=LAMBDA_SPARSITY
):
    """
    Phase 1: Train model AND learn per-point weights w_i ∈ (0,1) via a
    sparsity-regularized loss.

    We parameterize weights with logits a_i, w_raw = sigmoid(a_i).
    To avoid trivial all-zero collapse:
        - We normalize w_raw to w_norm = w_raw / mean(w_raw)
        - The weighted error uses w_norm, so its scale is independent of mean(w_raw).
        - The sparsity penalty is on mean(w_raw), which prefers small raw weights
          (i.e. fewer effective important points) but cannot shut off the error term.

    Loss:
        L = mean( w_norm * error_i ) + lambda_sparsity * mean(w_raw)
    """
    N = len(s_full)
    s_tensor = torch.tensor(s_full, dtype=torch.float32).unsqueeze(1)  # (N,1)
    pts_tensor = torch.tensor(coords_full, dtype=torch.float32)       # (N,2)

    model = CurveMLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
    # One learnable logit per point
    logits = nn.Parameter(torch.zeros(N, dtype=torch.float32))

    optimizer = optim.Adam(
        list(model.parameters()) + [logits],
        lr=lr
    )

    losses = []
    mean_weights_history = []

    for epoch in range(num_epochs):
        pred = model(s_tensor)
        squared_errors = torch.sum((pred - pts_tensor) ** 2, dim=1)  # (N,)

        w_raw = torch.sigmoid(logits)  # (N,) in (0,1)
        w_mean = w_raw.mean() + SIGMOID_EPS
        w_norm = w_raw / w_mean        # mean(w_norm) ≈ 1

        # Weighted error and sparsity penalty
        weighted_error = (w_norm * squared_errors).mean()
        sparsity_penalty = w_raw.mean()

        loss = weighted_error + lambda_sparsity * sparsity_penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        mean_weights_history.append(w_raw.detach().mean().item())

        if (epoch + 1) % 200 == 0:
            print(
                f"[Imp] Epoch {epoch+1}/{num_epochs}, "
                f"loss={loss.item():.6e}, "
                f"mean(w)={mean_weights_history[-1]:.4f}"
            )

    # Final learned raw weights
    with torch.no_grad():
        w_raw_final = torch.sigmoid(logits).cpu().numpy()

    return model, w_raw_final, losses, mean_weights_history


def select_top_points_by_weight(s_full, coords_full, weights, keep_fraction=KEEP_FRACTION, min_points=MIN_POINTS):
    """
    Select the top fraction of points according to their learned weights.
    Returns reduced (s_sel, coords_sel, idx_sel).
    """
    N = len(s_full)
    assert N == len(weights)

    num_keep = max(min_points, int(N * keep_fraction))
    num_keep = min(num_keep, N)

    # Sort by weight descending
    sorted_idx = np.argsort(-weights)
    keep_idx = np.sort(sorted_idx[:num_keep])

    s_sel = s_full[keep_idx]
    coords_sel = coords_full[keep_idx]

    print(f"Selected top {num_keep}/{N} points (keep_fraction={keep_fraction}).")
    return s_sel, coords_sel, keep_idx


# -----------------------------
# 6. Visualization & export
# -----------------------------

def plot_loss(losses, out_dir, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(len(losses))
    ax.plot(epochs, losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(filename)
    ax.grid(True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weights(weights, out_dir, filename="importance_weights.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(weights)), np.sort(weights)[::-1])
    ax.set_xlabel("Point index (sorted by weight)")
    ax.set_ylabel("Learned weight w_i")
    ax.set_title("Learned point importance (sorted)")
    ax.grid(True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_skeleton_vs_reconstruction(skeleton, model, out_dir, num_points=2000):
    """
    Plot the original skeleton pixels vs. the MLP reconstruction.
    """
    ys, xs = np.nonzero(skeleton)
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)

    # Normalize skeleton pixels to [0,1]
    x_min, y_min = coords_pix.min(axis=0)
    x_max, y_max = coords_pix.max(axis=0)
    coords_pix[:, 0] = (coords_pix[:, 0] - x_min) / (x_max - x_min + 1e-8)
    coords_pix[:, 1] = (coords_pix[:, 1] - y_min) / (y_max - y_min + 1e-8)
    coords_pix[:, 1] = 1.0 - coords_pix[:, 1]  # flip y

    # Evaluate model on dense s-grid
    model.eval()
    s_dense = torch.linspace(0.0, 1.0, num_points).unsqueeze(1)
    with torch.no_grad():
        pts = model(s_dense).cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    plt.plot(coords_pix[:, 0], coords_pix[:, 1], 'k.', markersize=2, label="Skeleton")
    plt.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.0, label="MLP")
    plt.axis('equal')
    plt.axis('off')
    plt.legend()

    out_path = out_dir / "skeleton_vs_reconstruction.png"
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
        pts = model(s_dense).cpu().numpy()

    # 3) Save SVG
    svg_path = out_dir / "curve_reconstruction.svg"
    save_svg_curve(pts, svg_path)
    print(f"Saved SVG curve to: {svg_path}")

    # 4) Save PNG
    png_path = out_dir / "curve_reconstruction.png"
    save_png_curve(pts, png_path)
    print(f"Saved PNG curve to: {png_path}")


# -----------------------------
# 7. Main script
# -----------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python handwritten_curve_fit.py input.png epochs learning_rate [optional_out_prefix]")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: input file not found: {img_path}")
        sys.exit(1)

    epochs = int(sys.argv[2])
    learning_rate = float(sys.argv[3])

    # Optional output prefix
    out_prefix = sys.argv[4] if len(sys.argv) >= 5 else None

    img_path = Path(img_path)
    if out_prefix is None:
        out_prefix = img_path.with_suffix("")
    else:
        out_prefix = Path(out_prefix)

    # Run ID for outputs
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run ID: {run_id}")
    tagged_prefix = f"{out_prefix}_ID-{run_id}"

    out_dir = Path(tagged_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to directory: {out_dir}")

    # 1) Load + skeletonize
    skeleton = load_and_skeletonize(str(img_path), out_dir, show_intermediate=True)

    # 2) Extract ordered curve
    coords_ordered = extract_ordered_curve_from_skeleton(skeleton)

    # 3) Parameterize by arc length (full curve)
    s_full, coords_full = parameterize_by_arclength(coords_ordered)
    N_full = len(s_full)
    print(f"Full curve has {N_full} ordered points.")

    # 4) Phase 1: learn importance weights with Method B
    importance_epochs = max(1, int(IMPORTANCE_EPOCHS_MULTIPLIER * epochs))
    print(f"\n=== Phase 1: Importance learning ({importance_epochs} epochs) ===")

    imp_model, weights, imp_losses, mean_w_history = train_with_importance_weights(
        s_full,
        coords_full,
        num_epochs=importance_epochs,
        lr=learning_rate,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        lambda_sparsity=LAMBDA_SPARSITY
    )

    # Save diagnostics for importance learning
    np.save(out_dir / "importance_weights.npy", weights)
    plot_weights(weights, out_dir, filename="importance_weights.png")
    plot_loss(imp_losses, out_dir, filename="importance_loss_curve.png")

    # 5) Select top points by learned weights
    s_sel, coords_sel, keep_idx = select_top_points_by_weight(
        s_full, coords_full, weights,
        keep_fraction=KEEP_FRACTION,
        min_points=MIN_POINTS
    )
    np.save(out_dir / "selected_indices.npy", keep_idx)

    # 6) Phase 2: train final model on selected points only
    print(f"\n=== Phase 2: Final training on {len(s_sel)} selected points ({epochs} epochs) ===")
    final_model, final_losses = train_curve_mlp(
        s_sel,
        coords_sel,
        num_epochs=epochs,
        lr=learning_rate,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        verbose=True
    )
    plot_loss(final_losses, out_dir, filename="final_loss_curve.png")

    # Evaluate final model on full curve
    final_mse = compute_mse(final_model, s_full, coords_full)
    print(f"\nFinal model MSE on full curve: {final_mse:.6e}")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Num full points: {N_full}\n")
        f.write(f"Num selected points: {len(s_sel)}\n")
        f.write(f"Keep fraction: {KEEP_FRACTION}\n")
        f.write(f"Final MSE (full curve): {final_mse:.6e}\n")
        f.write(f"Lambda sparsity (Method B): {LAMBDA_SPARSITY}\n")

    # 7) Visualization and export
    plot_skeleton_vs_reconstruction(skeleton, final_model, out_dir, num_points=2000)
    export_curve_reconstruction(final_model, out_dir, num_points=2000)

    print("Done.")


if __name__ == "__main__":
    main()