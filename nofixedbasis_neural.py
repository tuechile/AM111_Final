"""
Neural smooth simplification of a handwritten curve using
a curve with K control points and a learned neural basis.

Pipeline:
  - Load handwriting image and skeletonize.
  - Extract ordered skeleton curve and parameterize by arc length s in [0,1].
  - For each K in K_list:
      * Fit a K-control-point NeuralBasis curve via gradient descent.
  - Save control points, loss curves, skeleton vs curve plots, and SVG/PNG.

Usage:
    python neural_smooth_curve_neuralbasis_only.py input.png K_list epochs learning_rate [optional_out_prefix]

Example:
    python neural_smooth_curve_neuralbasis_only.py stroke.png 32,64,100 5000 1e-3
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
import torch.nn.functional as F


# -----------------------------
# 1. Image loading & skeletonize
# -----------------------------

def load_and_skeletonize(path, out_dir, show_intermediate=True):
    """
    Load an image, convert to grayscale, binarize, and skeletonize.
    Returns a binary skeleton image (True at skeleton pixels).
    Saves intermediate visualization into out_dir.
    """
    img = imread(path, as_gray=True).astype(np.float32)  # [0,1]

    # Otsu thresholding
    thresh = threshold_otsu(img)

    # Assume dark ink on light background:
    binary = img < thresh

    # Skeletonize (thinning to 1-pixel-wide curve)
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
    intersection nodes as needed). This gives a single continuous path.

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

    # Hierholzer's algorithm for Eulerian trail
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
# 3. Neural basis curve
# -----------------------------

class SinLayer(nn.Module):
    def forward(self, x):
        return torch.sin(np.pi * x)

class NeuralBasisCurve(nn.Module):
    """
    Curve with K control points and a learned basis B_theta(t).

    C(t_i) = B_theta(t_i) @ ctrl
    where:
        - ctrl: (K,2) learnable control points
        - B_theta(t_i): (K,) from an MLP + softmax
    """

    def __init__(self, K, init_ctrl_points):
        super().__init__()
        self.K = K
        self.ctrl = nn.Parameter(
            torch.tensor(init_ctrl_points, dtype=torch.float32)  # (K,2)
        )

        # Tiny MLP: R -> R^K
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            SinLayer(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, K),
        )

    def forward(self, t):
        """
        t: (N,) tensor in [0,1]
        Returns: (N,2) curve samples
        """
        t_in = t.view(-1, 1)               # (N,1)
        logits = self.mlp(t_in)            # (N,K)
        B = torch.softmax(logits, dim=1)   # (N,K), rows sum to 1
        pts = B @ self.ctrl                # (N,2)
        return pts


def basis_smoothness_penalty(model, t):
    """
    Encourage smooth basis functions over t by penalizing finite differences.
    """
    t_sorted, _ = torch.sort(t)
    t_in = t_sorted.view(-1, 1)
    logits = model.mlp(t_in)
    B = torch.softmax(logits, dim=1)          # (N,K)
    dB = B[1:] - B[:-1]                       # (N-1,K)
    return (dB ** 2).mean()


def basis_locality_term(model, t):
    """
    Encourage locality (peaky basis) by *reducing* entropy.
    Returns a term that is larger when basis is more local.
    """
    t_in = t.view(-1, 1)
    logits = model.mlp(t_in)
    B = torch.softmax(logits, dim=1)          # (N,K)
    entropy = - (B * torch.log(B + 1e-8)).sum(dim=1).mean()
    # Low entropy => more local; we want to *maximize* locality,
    # so return negative entropy as a "good" term.
    return -entropy


def ctrl_smoothness_penalty(ctrl):
    """
    Encourage the control polygon to be smooth by penalizing second differences.
    ctrl: (K,2)
    """
    if ctrl.shape[0] < 3:
        return torch.tensor(0.0, dtype=ctrl.dtype, device=ctrl.device)
    second_diff = ctrl[2:] - 2 * ctrl[1:-1] + ctrl[:-2]  # (K-2,2)
    return (second_diff ** 2).mean()


def train_neural_basis(
    s_full,
    coords_full,
    K,
    num_epochs=2000,
    lr=1e-3,
    lam_basis_smooth=1e-3,
    lam_basis_local=1e-3,
    lam_ctrl_smooth=1e-2,
    verbose=True,
):
    """
    Train a K-control-point curve with a learned neural basis.

    s_full: (N,) numpy array in [0,1] (arc-length param)
    coords_full: (N,2) numpy array, target curve
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t_train = torch.tensor(s_full, dtype=torch.float32, device=device)   # (N,)
    pts = torch.tensor(coords_full, dtype=torch.float32, device=device)  # (N,2)

    N = len(coords_full)
    init_idx = np.linspace(0, N - 1, K, dtype=int)
    init_ctrl = coords_full[init_idx]  # (K,2)

    model = NeuralBasisCurve(K, init_ctrl_points=init_ctrl).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(t_train)             # (N,2)
        loss_data = criterion(pred, pts)

        # Regularization terms
        loss = loss_data
        if lam_basis_smooth > 0.0:
            loss = loss + lam_basis_smooth * basis_smoothness_penalty(model, t_train)
        if lam_basis_local > 0.0:
            # locality_term is higher when basis is more peaky / local
            loss = loss - lam_basis_local * basis_locality_term(model, t_train)
        if lam_ctrl_smooth > 0.0:
            loss = loss + lam_ctrl_smooth * ctrl_smoothness_penalty(model.ctrl)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch + 1) % 500 == 0:
            print(
                f"[NeuralBasis K={K}] Epoch {epoch+1}/{num_epochs}, "
                f"loss={loss.item():.6e}, data_loss={loss_data.item():.6e}"
            )

    # Final MSE on training (pure data term)
    model.eval()
    with torch.no_grad():
        pred_final = model(t_train).cpu().numpy()
    mse_final = np.mean((pred_final - coords_full) ** 2)

    return model, np.array(losses), mse_final


def eval_neural_basis_dense(model, num_points=1000):
    """
    Evaluate a trained NeuralBasis model on a dense t-grid in [0,1].
    Returns:
        t_dense: (num_points,)
        pts: (num_points,2)
    """
    device = next(model.parameters()).device
    t_dense = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    t_torch = torch.tensor(t_dense, dtype=torch.float32, device=device)
    with torch.no_grad():
        pts = model(t_torch).cpu().numpy()
    return t_dense, pts


# -----------------------------
# 4. Visualization & export
# -----------------------------

def plot_loss(losses, out_dir, filename, ylabel="Loss"):
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(len(losses))
    ax.plot(epochs, losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(filename)
    ax.grid(True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_skeleton_vs_curve(skeleton, curve_points, label, out_dir, filename):
    """
    Plot original skeleton pixels vs curve_points (M,2).
    curve_points: (M,2) in [0,1]^2, y upwards.
    """
    ys, xs = np.nonzero(skeleton)
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)

    # Normalize skeleton pixels to [0,1]
    x_min, y_min = coords_pix.min(axis=0)
    x_max, y_max = coords_pix.max(axis=0)
    coords_pix[:, 0] = (coords_pix[:, 0] - x_min) / (x_max - x_min + 1e-8)
    coords_pix[:, 1] = (coords_pix[:, 1] - y_min) / (y_max - y_min + 1e-8)
    coords_pix[:, 1] = 1.0 - coords_pix[:, 1]  # flip y

    fig = plt.figure(figsize=(4, 4))
    plt.plot(coords_pix[:, 0], coords_pix[:, 1], 'k.', markersize=2, label="Skeleton")
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'r-', linewidth=1.0, label=label)
    plt.axis('equal')
    plt.axis('off')
    plt.legend()

    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_svg_curve(points, path):
    """
    Save a curve as an SVG polyline.

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


# -----------------------------
# 5. Main script
# -----------------------------

def main():
    if len(sys.argv) < 5:
        print("Usage: python neural_smooth_curve_neuralbasis_only.py input.png K_list epochs learning_rate [optional_out_prefix]")
        print("Example: python neural_smooth_curve_neuralbasis_only.py stroke.png 32,64,100 5000 1e-3")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: input file not found: {img_path}")
        sys.exit(1)

    # K_list: comma-separated, e.g. "32,64,100"
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
    tagged_prefix = f"{out_prefix}_NEURALBASIS_ID-{run_id}"

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

    summary_lines = []

    for K in K_list:
        print(f"\n=== K={K} control points (NeuralBasis) ===")
        nb_model, nb_losses, mse_nb = train_neural_basis(
            s_full,
            coords_full,
            K=K,
            num_epochs=epochs,
            lr=learning_rate,
            lam_basis_smooth=1e-3,
            lam_basis_local=1e-3,
            lam_ctrl_smooth=1e-2,
            verbose=True,
        )
        print(f"[NeuralBasis] K={K}, final MSE={mse_nb:.6e}")

        # Save losses and control points
        np.save(out_dir / f"neuralbasis_K{K}_losses.npy", nb_losses)
        with torch.no_grad():
            nb_ctrl = nb_model.ctrl.detach().cpu().numpy()
        np.save(out_dir / f"neuralbasis_K{K}_ctrl.npy", nb_ctrl)

        plot_loss(nb_losses, out_dir, filename=f"neuralbasis_K{K}_loss_curve.png")

        # Dense evaluation & plots
        _, nb_dense_pts = eval_neural_basis_dense(nb_model, num_points=1000)
        plot_skeleton_vs_curve(
            skeleton,
            nb_dense_pts,
            label=f"NeuralBasis (K={K})",
            out_dir=out_dir,
            filename=f"skeleton_vs_neuralbasis_K{K}.png",
        )
        save_svg_curve(nb_dense_pts, out_dir / f"neuralbasis_K{K}.svg")
        save_png_curve(nb_dense_pts, out_dir / f"neuralbasis_K{K}.png")

        summary_lines.append(f"NeuralBasis K={K}, MSE={mse_nb:.6e}")

    # Summary
    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Num full points: {N_full}\n")
        f.write(f"Ks tested: {K_list}\n\n")
        f.write("Results:\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\nDone. Summary:")
    for line in summary_lines:
        print("  " + line)


if __name__ == "__main__":
    main()