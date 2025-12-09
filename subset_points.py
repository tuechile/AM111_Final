"""
Neural reconstruction of a handwritten curve with thinning + MLP
and iterative learned point pruning (Method A).

Usage:
    python handwritten_curve_fit.py input.png epochs learning_rate [optional_out_prefix]

This script will:
1. Load a handwriting image (e.g. a single stroke or character).
2. Binarize and skeletonize (thinning) to get a 1-pixel-wide curve.
3. Extract skeleton pixels, order them into a path, and parameterize by arc length s in [0,1].
4. Iteratively:
    - Train an MLP f(s) = (x, y) on a subset of points.
    - Compute which training points are "easiest" (lowest error).
    - Remove a block of those points.
   This learns *which points are important* to keep.
5. Select the best model across pruning iterations (small training set, low full-curve error).
6. Save visualizations and the final model (weights, SVG, PNG) into a timestamped folder.
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
# Hyperparameters for pruning
# =============================

# Target MSE on the full curve (in normalized [0,1]^2 space)
TARGET_MSE = 1e-4

# Fraction of training points to remove each pruning iteration
PRUNE_FRACTION = 0.05  # remove some percent of the *easiest* points

# Minimum number of points we'll allow in the training set
MIN_POINTS = 100

# Maximum number of pruning iterations
MAX_PRUNE_ITERS = 10


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
    def __init__(self, hidden_dim=64, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), Sin()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        s_scaled = s * 2 * np.pi  # scale input to [0, 2pi]
        return self.net(s_scaled)


# -----------------------------
# 4. Training (simple)
# -----------------------------

def train_curve_mlp(s, coords, num_epochs=3000, lr=1e-3,
                    hidden_dim=64, num_hidden_layers=2, verbose=True):
    """
    Train MLP to fit f(s) ~ coords.
    Returns:
        model: trained CurveMLP
        losses: list of MSE loss per epoch
    """
    s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(1)  # (N,1)
    points_tensor = torch.tensor(coords, dtype=torch.float32)     # (N,2)

    model = CurveMLP(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(num_epochs):
        pred = model(s_tensor)
        loss = criterion(pred, points_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss = {loss.item():.6e}")

    return model, losses


# -----------------------------
# 5. Learned point pruning
# -----------------------------

def compute_full_mse(model, s_full, coords_full):
    """
    Compute MSE between model predictions and full coordinates.
    """
    model.eval()
    with torch.no_grad():
        s_tensor = torch.tensor(s_full, dtype=torch.float32).unsqueeze(1)
        pred = model(s_tensor).cpu().numpy()
    mse = np.mean((pred - coords_full) ** 2)
    return mse


def prune_points_by_error(s_train, coords_train, model,
                          prune_fraction=0.1, min_points=100):
    """
    Remove a fraction of the *easiest* training points (smallest error).
    Returns the reduced (s_train, coords_train).
    """
    N = len(s_train)
    if N <= min_points:
        return s_train, coords_train

    model.eval()
    with torch.no_grad():
        s_tensor = torch.tensor(s_train, dtype=torch.float32).unsqueeze(1)
        pred = model(s_tensor).cpu().numpy()

    errors = np.sum((pred - coords_train) ** 2, axis=1)  # (N,)

    # sort points by error ascending (easiest first)
    sorted_idx = np.argsort(errors)

    points_remove = int(np.log10(N) * N * prune_fraction)
    num_remove = max(1, points_remove)
    # we remove the easiest 'num_remove' points
    remove_idx = sorted_idx[:num_remove]
    keep_mask = np.ones(N, dtype=bool)
    keep_mask[remove_idx] = False

    s_new = s_train[keep_mask]
    coords_new = coords_train[keep_mask]

    if len(s_new) < min_points:
        # ensure we don't go below MIN_POINTS
        print("  Pruning would go below MIN_POINTS; stopping pruning.")
        return s_train, coords_train

    print(f"  Pruned {num_remove} points (from N={N} to N={len(s_new)})")
    return s_new, coords_new


def iterative_prune_and_train(s_full, coords_full, num_epochs, lr,
                              hidden_dim=64, num_hidden_layers=2,
                              target_mse=TARGET_MSE,
                              prune_fraction=PRUNE_FRACTION,
                              min_points=MIN_POINTS,
                              max_iters=MAX_PRUNE_ITERS):
    """
    Iteratively:
      - Train on current training set (s_train, coords_train).
      - Evaluate MSE on full curve (s_full, coords_full).
      - Remove a block of easiest points from training set.
    Keep the best model (lowest full-curve MSE).
    """
    s_train = s_full.copy()
    coords_train = coords_full.copy()

    best_model = None
    best_mse = float("inf")
    best_losses = None
    best_train_size = len(s_train)

    for it in range(max_iters):
        print(f"\n=== Pruning iteration {it+1}/{max_iters} ===")
        print(f"Training on {len(s_train)} points...")

        model, losses = train_curve_mlp(
            s_train, coords_train,
            num_epochs=num_epochs, lr=lr,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            verbose=True
        )

        mse_full = compute_full_mse(model, s_full, coords_full)
        print(f"  Full-curve MSE = {mse_full:.6e} (N_train = {len(s_train)})")

        # Track best model so far
        improved = mse_full < best_mse
        if improved:
            best_mse = mse_full
            best_model = model
            best_losses = losses
            best_train_size = len(s_train)
            print(f"  -> New best model! MSE={best_mse:.6e}, N_train={best_train_size}")

        # Stopping conditions
        if mse_full <= target_mse and len(s_train) <= min_points * 1.2:
            print("  Target MSE reached with small training set; stopping pruning.")
            break

        if len(s_train) <= min_points:
            print("  Reached minimum number of points; stopping pruning.")
            break

        # Prepare next iteration: prune easiest points
        s_next, coords_next = prune_points_by_error(
            s_train, coords_train, model,
            prune_fraction=prune_fraction,
            min_points=min_points
        )

        # If pruning does not change set size, stop
        if len(s_next) == len(s_train):
            print("  No further pruning possible; stopping.")
            break

        s_train, coords_train = s_next, coords_next

    print(f"\n=== Pruning finished ===")
    print(f"Best full-curve MSE: {best_mse:.6e} with N_train={best_train_size}")
    return best_model, best_losses, best_mse, best_train_size


# -----------------------------
# 6. Visualization & export
# -----------------------------

def plot_loss(losses, out_dir, filename="loss_curve_best_model.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(len(losses))
    ax.plot(epochs, losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_yscale("log")
    ax.set_title("Training loss (best model)")
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
        pts = model(s_dense).cpu().numpy()  # shape (num_points, 2)

    # Plot
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
      - model weights (.pt, fp32)
      - vector curve (.svg)
      - raster curve (.png)
    into out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save model weights (float32)
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

    # 3) Parameterize by arc length (full curve)
    s_full, coords_full = parameterize_by_arclength(coords_ordered)

    # 4) Iterative pruning + training to get best (compressed) model
    hidden_dim = 64
    num_hidden_layers = 2

    best_model, best_losses, best_mse, best_train_size = iterative_prune_and_train(
        s_full, coords_full,
        num_epochs=epochs,
        lr=learning_rate,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        target_mse=TARGET_MSE,
        prune_fraction=PRUNE_FRACTION,
        min_points=MIN_POINTS,
        max_iters=MAX_PRUNE_ITERS
    )

    # 5) Plot training loss for best model
    if best_losses is not None:
        plot_loss(best_losses, out_dir)

    # 6) Plot skeleton vs reconstruction for best model
    plot_skeleton_vs_reconstruction(skeleton, best_model, out_dir, num_points=2000)

    # 7) Export curve reconstruction (weights + SVG + PNG)
    export_curve_reconstruction(best_model, out_dir, num_points=2000)

    print("Done.")


if __name__ == "__main__":
    main()