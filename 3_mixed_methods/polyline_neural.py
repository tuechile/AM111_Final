"""
Neural polyline simplification of a handwritten curve.

Instead of fitting an MLP to all skeleton points, this script learns a small
set of K vertices that define a polyline. The polyline is a piecewise linear
function of arc-length parameter s in [0,1], and the K vertices are optimized
by gradient descent to minimize MSE to the original skeleton curve.

Usage:
    python neural_polyline_simplify.py input.png K_list epochs learning_rate [optional_out_prefix]

Example:
    python neural_polyline_simplify.py stroke.png 16,32,64 2000 1e-2
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


# -----------------------------
# 1. Image loading & skeletonize
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

    # Choose a start node: odd-degree vertex if exists, else 0
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
# 3. Polyline (neural) model
# -----------------------------

def prepare_segmentation(s_full, K):
    """
    Given s_full in [0,1], precompute which polyline segment each point belongs to
    and the interpolation coefficient alpha.

    We place K vertices at parameter locations:
        u_k = k / (K-1), k=0,...,K-1
    and for each s_i, find k such that s_i in [u_k, u_{k+1}].

    Returns:
        seg_idx: (N,) int64 in [0, K-2]
        alpha:   (N,) float32 in [0,1)
    """
    N = len(s_full)
    s_clipped = np.clip(s_full, 0.0, 1.0 - 1e-8)
    t = s_clipped * (K - 1)           # in [0, K-1)
    seg_idx = np.floor(t).astype(int) # in [0, K-2]
    alpha = t - seg_idx               # fractional part in [0,1)
    assert np.all(seg_idx >= 0) and np.all(seg_idx <= K - 2)
    return seg_idx, alpha


class Polyline(nn.Module):
    """
    A learnable polyline with K vertices in 2D.

    Given segment indices seg_idx and interpolation factors alpha, it produces
    piecewise linear interpolation between consecutive vertices.
    """

    def __init__(self, K, init_vertices):
        """
        K: number of vertices
        init_vertices: (K, 2) numpy array for initialization
        """
        super().__init__()
        self.K = K
        verts = torch.tensor(init_vertices, dtype=torch.float32)
        self.vertices = nn.Parameter(verts)  # (K, 2)

    def forward(self, seg_idx, alpha):
        """
        seg_idx: (N,) LongTensor, each in [0, K-2]
        alpha:   (N,) FloatTensor in [0,1)

        Returns:
            (N, 2) predicted coordinates
        """
        v0 = self.vertices[seg_idx]         # (N, 2)
        v1 = self.vertices[seg_idx + 1]     # (N, 2)
        alpha = alpha.unsqueeze(1)          # (N, 1)
        return (1.0 - alpha) * v0 + alpha * v1


def train_polyline(s_full, coords_full, K, num_epochs=2000, lr=1e-2, verbose=True):
    """
    Train a Polyline model with K vertices to fit the full curve.

    s_full:       (N,) arc-length parameters in [0,1]
    coords_full:  (N,2) target coordinates in [0,1]^2
    """
    # Prepare segmentation
    seg_idx_np, alpha_np = prepare_segmentation(s_full, K)
    seg_idx = torch.tensor(seg_idx_np, dtype=torch.long)
    alpha = torch.tensor(alpha_np, dtype=torch.float32)
    pts = torch.tensor(coords_full, dtype=torch.float32)

    # Initialize vertices by sampling along the curve
    N = len(coords_full)
    init_idx = np.linspace(0, N - 1, K, dtype=int)
    init_vertices = coords_full[init_idx]

    model = Polyline(K, init_vertices)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(num_epochs):
        model.train()
        pred = model(seg_idx, alpha)
        loss = criterion(pred, pts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch + 1) % 200 == 0:
            print(f"[Polyline K={K}] Epoch {epoch+1}/{num_epochs}, loss={loss.item():.6e}")

    return model, np.array(losses)


def compute_mse_polyline(model, s_full, coords_full, K):
    """
    Compute MSE between a trained Polyline model and given coordinates.
    """
    seg_idx_np, alpha_np = prepare_segmentation(s_full, K)
    seg_idx = torch.tensor(seg_idx_np, dtype=torch.long)
    alpha = torch.tensor(alpha_np, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(seg_idx, alpha).cpu().numpy()
    mse = np.mean((pred - coords_full) ** 2)
    return mse


# -----------------------------
# 4. Visualization & export
# -----------------------------

def plot_loss(losses, out_dir, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(len(losses))
    ax.plot(epochs, losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title(filename)
    ax.grid(True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_skeleton_vs_polyline(skeleton, model, K, out_dir, filename="skeleton_vs_polyline.png", num_points=1000):
    """
    Plot the original skeleton pixels vs. the learned polyline reconstruction.
    """
    ys, xs = np.nonzero(skeleton)
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)

    # Normalize skeleton pixels to [0,1]
    x_min, y_min = coords_pix.min(axis=0)
    x_max, y_max = coords_pix.max(axis=0)
    coords_pix[:, 0] = (coords_pix[:, 0] - x_min) / (x_max - x_min + 1e-8)
    coords_pix[:, 1] = (coords_pix[:, 1] - y_min) / (y_max - y_min + 1e-8)
    coords_pix[:, 1] = 1.0 - coords_pix[:, 1]  # flip y

    # Sample dense s-grid and evaluate polyline
    s_dense = np.linspace(0.0, 1.0, num_points)
    seg_idx_np, alpha_np = prepare_segmentation(s_dense, K)
    seg_idx = torch.tensor(seg_idx_np, dtype=torch.long)
    alpha = torch.tensor(alpha_np, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pts = model(seg_idx, alpha).cpu().numpy()

    fig = plt.figure(figsize=(4, 4))
    plt.plot(coords_pix[:, 0], coords_pix[:, 1], 'k.', markersize=2, label="Skeleton")
    plt.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.0, label=f"Polyline (K={K})")
    plt.axis('equal')
    plt.axis('off')
    plt.legend()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_svg_polyline(vertices, path):
    """
    Save a polyline as an SVG.

    vertices: array (K, 2) in normalized [0,1] coordinates (x,y) with y upward.
    SVG has viewBox 0..1 x 0..1. We flip y because SVG's y-axis goes downward.
    """
    path = Path(path)
    with open(path, "w") as f:
        f.write('<svg viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<polyline points="')
        for x, y in vertices:
            y_svg = 1.0 - y  # flip back for SVG coord system
            f.write(f"{x},{y_svg} ")
        f.write('" fill="none" stroke="black" stroke-width="0.002"/>\n')
        f.write('</svg>\n')


def save_png_polyline(vertices, path, dpi=300):
    """
    Save a polyline as a PNG using matplotlib.

    vertices: array (K, 2) in normalized [0,1] coordinates (x,y) with y upward.
    """
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.plot(vertices[:, 0], vertices[:, 1], 'k-')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


# -----------------------------
# 5. Main script
# -----------------------------

def main():
    if len(sys.argv) < 5:
        print("Usage: python neural_polyline_simplify.py input.png K_list epochs learning_rate [optional_out_prefix]")
        print("Example: python neural_polyline_simplify.py stroke.png 16,32,64 2000 1e-2")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: input file not found: {img_path}")
        sys.exit(1)

    # K_list: comma-separated, e.g. "16,32,64"
    K_list_str = sys.argv[2]
    K_list = [int(k.strip()) for k in K_list_str.split(",") if k.strip()]

    epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])

    # Optional output prefix
    out_prefix = sys.argv[5] if len(sys.argv) >= 6 else None

    if out_prefix is None:
        out_prefix = img_path.with_suffix("")
    else:
        out_prefix = Path(out_prefix)

    # Run ID for outputs
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Run ID: {run_id}")
    tagged_prefix = f"{out_prefix}_POLY_ID-{run_id}"

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

    # Save original curve
    np.save(out_dir / "coords_full.npy", coords_full)
    np.save(out_dir / "s_full.npy", s_full)

    # 4) For each K, train a polyline model
    summary_lines = []
    for K in K_list:
        print(f"\n=== Training polyline with K={K} vertices ({epochs} epochs) ===")
        poly_model, losses = train_polyline(
            s_full,
            coords_full,
            K=K,
            num_epochs=epochs,
            lr=learning_rate,
            verbose=True
        )

        mse = compute_mse_polyline(poly_model, s_full, coords_full, K)
        print(f"[Result] K={K}, final full-curve MSE={mse:.6e}")

        # Save losses
        np.save(out_dir / f"polyline_K{K}_losses.npy", losses)
        plot_loss(losses, out_dir, filename=f"polyline_K{K}_loss_curve.png")

        # Save vertices
        with torch.no_grad():
            verts = poly_model.vertices.cpu().numpy()
        np.save(out_dir / f"polyline_K{K}_vertices.npy", verts)

        # Plot skeleton vs polyline
        plot_skeleton_vs_polyline(
            skeleton,
            poly_model,
            K,
            out_dir,
            filename=f"skeleton_vs_polyline_K{K}.png",
            num_points=1000
        )

        # Export SVG/PNG
        save_svg_polyline(verts, out_dir / f"polyline_K{K}.svg")
        save_png_polyline(verts, out_dir / f"polyline_K{K}.png")

        summary_lines.append(f"K={K}, MSE={mse:.6e}")

    # 5) Save summary
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Num full points: {N_full}\n")
        f.write(f"Ks tested: {K_list}\n")
        f.write("\nResults:\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\nDone. Summary:")
    for line in summary_lines:
        print("  " + line)


if __name__ == "__main__":
    main()
