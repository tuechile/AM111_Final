"""
Neural smooth simplification of a handwritten curve using:

  1) A Bézier curve with K control points.
  2) A cubic B-spline with K control points (K >= degree+1).

Pipeline:
  - Load handwriting image and skeletonize.
  - Extract ordered skeleton curve and parameterize by arc length s in [0,1].
  - For each K in K_list:
      * Fit a K-control-point Bézier curve via gradient descent.
      * Fit a K-control-point cubic B-spline via gradient descent.
  - Save control points, loss curves, skeleton vs curve plots, and SVG/PNG.

Usage:
    python neural_smooth_curve_simplify.py input.png K_list epochs learning_rate [optional_out_prefix]

Example:
    python neural_smooth_curve_simplify.py stroke.png 8,16,32 2000 1e-2
"""

import sys
import math
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
# 3. Bézier curve (global smooth)
# -----------------------------

def bernstein_basis_matrix(t, K):
    """
    Compute Bernstein basis matrix for a Bézier curve of degree n=K-1.

    t: (N,) in [0,1]
    Returns:
        B: (N, K) where B[i, j] = C(n, j) t_i^j (1 - t_i)^(n-j)
    """
    t = np.asarray(t, dtype=np.float64)
    N = t.shape[0]
    n = K - 1
    B = np.zeros((N, K), dtype=np.float64)

    for j in range(K):
        binom = math.comb(n, j)
        B[:, j] = binom * (t ** j) * ((1.0 - t) ** (n - j))

    return B.astype(np.float32)


class BezierCurve(nn.Module):
    """
    Bézier curve parameterized by K control points in 2D.
    C(t) = sum_{i=0}^{K-1} B_i,K-1(t) * P_i
    """

    def __init__(self, K, init_ctrl_points, basis_matrix):
        """
        K: number of control points
        init_ctrl_points: (K,2) numpy array
        basis_matrix: (N,K) torch tensor for all training t values
        """
        super().__init__()
        self.K = K
        self.ctrl = nn.Parameter(
            torch.tensor(init_ctrl_points, dtype=torch.float32)
        )  # (K,2)
        # Basis is fixed (no grad)
        self.register_buffer("B", basis_matrix)  # (N,K)

    def forward(self):
        # B: (N,K), ctrl: (K,2) -> (N,2)
        return self.B @ self.ctrl


def train_bezier(s_full, coords_full, K, num_epochs=2000, lr=1e-2, verbose=True):
    """
    Train a K-control-point Bézier curve to fit (s_full, coords_full).
    """
    # Build basis matrix for training points
    B_np = bernstein_basis_matrix(s_full, K)  # (N,K)
    B = torch.tensor(B_np, dtype=torch.float32)

    pts = torch.tensor(coords_full, dtype=torch.float32)  # (N,2)

    # Initialize control points along the curve
    N = len(coords_full)
    init_idx = np.linspace(0, N - 1, K, dtype=int)
    init_ctrl = coords_full[init_idx]  # (K,2)

    model = BezierCurve(K, init_ctrl, B)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(num_epochs):
        model.train()
        pred = model()           # (N,2)
        loss = criterion(pred, pts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch + 1) % 200 == 0:
            print(f"[Bezier K={K}] Epoch {epoch+1}/{num_epochs}, loss={loss.item():.6e}")

    return model, np.array(losses)


def eval_bezier_dense(model, K, num_points=1000):
    """
    Evaluate a trained Bézier model on a dense t-grid in [0,1].
    Returns:
        t_dense: (num_points,)
        pts: (num_points,2)
    """
    t_dense = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    B_dense_np = bernstein_basis_matrix(t_dense, K)
    B_dense = torch.tensor(B_dense_np, dtype=torch.float32)

    with torch.no_grad():
        ctrl = model.ctrl  # (K,2)
        pts = (B_dense @ ctrl).cpu().numpy()
    return t_dense, pts


# -----------------------------
# 4. B-spline curve (smooth + local control)
# -----------------------------

def open_uniform_knots(n_ctrl, degree):
    """
    Open uniform (clamped) knot vector on [0,1].

    n_ctrl: number of control points (basis functions)
    degree: spline degree (e.g. 3 for cubic)
    """
    p = degree
    n_knots = n_ctrl + p + 1
    knots = np.zeros(n_knots, dtype=np.float64)
    knots[-(p+1):] = 1.0  # last p+1 knots = 1

    # internal knots, if any
    n_internal = n_ctrl - p - 1
    if n_internal > 0:
        internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]  # exclude endpoints
        knots[p+1:p+1+n_internal] = internal
    return knots


def bspline_basis_at_t(t, n_ctrl, degree, knots):
    """
    Compute all B-spline basis functions N_{i,degree}(t) for i=0..n_ctrl-1
    using Cox-de Boor recursion.

    n_ctrl: number of control points (basis functions)
    degree: spline degree
    knots: knot vector of length n_ctrl + degree + 1
    """
    p = degree
    # Clamp t to valid range with epsilon to avoid hitting the final knot exactly
    eps = 1e-8
    t = float(np.clip(t, knots[0] + eps, knots[-1] - eps))

    # N_i,0(t)
    N = np.zeros(n_ctrl, dtype=np.float64)
    for i in range(n_ctrl):
        if knots[i] <= t < knots[i+1]:
            N[i] = 1.0
        else:
            N[i] = 0.0

    # Degree elevation up to p
    for k in range(1, p+1):
        N_new = np.zeros(n_ctrl, dtype=np.float64)
        for i in range(n_ctrl):
            # left term
            denom1 = knots[i + k] - knots[i]
            if denom1 != 0.0:
                term1 = (t - knots[i]) / denom1 * N[i]
            else:
                term1 = 0.0

            # right term
            denom2 = knots[i + k + 1] - knots[i + 1]
            if denom2 != 0.0 and i + 1 < n_ctrl:
                term2 = (knots[i + k + 1] - t) / denom2 * N[i + 1]
            else:
                term2 = 0.0

            N_new[i] = term1 + term2
        N = N_new

    return N


def bspline_basis_matrix(t, n_ctrl, degree):
    """
    Compute B-spline basis matrix for all t.

    t: (N,)
    Returns:
        B: (N, n_ctrl) where B[i,j] = N_{j,degree}(t_i)
    """
    t = np.asarray(t, dtype=np.float64)
    knots = open_uniform_knots(n_ctrl, degree)
    N_pts = t.shape[0]
    B = np.zeros((N_pts, n_ctrl), dtype=np.float64)
    for idx in range(N_pts):
        B[idx, :] = bspline_basis_at_t(t[idx], n_ctrl, degree, knots)
    return B.astype(np.float32)


class BSplineCurve(nn.Module):
    """
    B-spline curve parameterized by n_ctrl control points in 2D.
    C(t) = sum_{i=0}^{n_ctrl-1} N_{i,p}(t) * P_i
    """

    def __init__(self, n_ctrl, init_ctrl_points, basis_matrix, degree=3):
        """
        n_ctrl: number of control points
        init_ctrl_points: (n_ctrl,2) numpy array
        basis_matrix: (N, n_ctrl) torch tensor for training t values
        degree: spline degree (default: 3)
        """
        super().__init__()
        self.n_ctrl = n_ctrl
        self.degree = degree
        self.ctrl = nn.Parameter(
            torch.tensor(init_ctrl_points, dtype=torch.float32)
        )
        self.register_buffer("B", basis_matrix)  # (N, n_ctrl)

    def forward(self):
        return self.B @ self.ctrl   # (N,2)


def train_bspline(s_full, coords_full, n_ctrl, degree=3, num_epochs=2000, lr=1e-2, verbose=True):
    """
    Train a cubic (degree=3 by default) B-spline with n_ctrl control points
    to fit (s_full, coords_full).
    """
    if n_ctrl <= degree:
        raise ValueError(f"n_ctrl={n_ctrl} must be > degree={degree} for B-spline.")

    B_np = bspline_basis_matrix(s_full, n_ctrl, degree)  # (N, n_ctrl)
    B = torch.tensor(B_np, dtype=torch.float32)
    pts = torch.tensor(coords_full, dtype=torch.float32)  # (N,2)

    N = len(coords_full)
    init_idx = np.linspace(0, N - 1, n_ctrl, dtype=int)
    init_ctrl = coords_full[init_idx]  # (n_ctrl,2)

    model = BSplineCurve(n_ctrl, init_ctrl, B, degree=degree)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(num_epochs):
        model.train()
        pred = model()
        loss = criterion(pred, pts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if verbose and (epoch + 1) % 200 == 0:
            print(f"[B-spline K={n_ctrl}] Epoch {epoch+1}/{num_epochs}, loss={loss.item():.6e}")

    return model, np.array(losses)


def eval_bspline_dense(model, n_ctrl, degree=3, num_points=1000):
    """
    Evaluate a trained B-spline model on a dense t-grid in [0,1].
    Returns:
        t_dense: (num_points,)
        pts: (num_points,2)
    """
    t_dense = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    B_dense_np = bspline_basis_matrix(t_dense, n_ctrl, degree)
    B_dense = torch.tensor(B_dense_np, dtype=torch.float32)

    with torch.no_grad():
        ctrl = model.ctrl   # (n_ctrl,2)
        pts = (B_dense @ ctrl).cpu().numpy()
    return t_dense, pts


# -----------------------------
# 5. Visualization & export
# -----------------------------

def plot_loss(losses, out_dir, filename, ylabel="MSE loss"):
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
    Plot original skeleton pixels vs curve_points (N,2).
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
# 6. Main script
# -----------------------------

def main():
    if len(sys.argv) < 5:
        print("Usage: python neural_smooth_curve_simplify.py input.png K_list epochs learning_rate [optional_out_prefix]")
        print("Example: python neural_smooth_curve_simplify.py stroke.png 8,16,32 2000 1e-2")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: input file not found: {img_path}")
        sys.exit(1)

    # K_list: comma-separated, e.g. "8,16,32"
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
    tagged_prefix = f"{out_prefix}_SMOOTH_ID-{run_id}"

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
    degree_bspline = 3  # cubic

    for K in K_list:
        print(f"\n=== K={K} control points ===")

        # ------------- Bézier -------------
        print(f"Training Bézier with K={K} control points ({epochs} epochs)")
        bez_model, bez_losses = train_bezier(
            s_full,
            coords_full,
            K=K,
            num_epochs=epochs,
            lr=learning_rate,
            verbose=True
        )

        # Evaluate on training set
        with torch.no_grad():
            B_train = torch.tensor(bernstein_basis_matrix(s_full, K), dtype=torch.float32)
            bez_train_pred = (B_train @ bez_model.ctrl).cpu().numpy()
        mse_bez = np.mean((bez_train_pred - coords_full) ** 2)
        print(f"[Bezier] K={K}, final MSE={mse_bez:.6e}")

        # Save losses and control points
        np.save(out_dir / f"bezier_K{K}_losses.npy", bez_losses)
        with torch.no_grad():
            bez_ctrl = bez_model.ctrl.cpu().numpy()
        np.save(out_dir / f"bezier_K{K}_ctrl.npy", bez_ctrl)

        plot_loss(bez_losses, out_dir, filename=f"bezier_K{K}_loss_curve.png")

        # Dense evaluation & plots
        _, bez_dense_pts = eval_bezier_dense(bez_model, K, num_points=1000)
        plot_skeleton_vs_curve(
            skeleton,
            bez_dense_pts,
            label=f"Bezier (K={K})",
            out_dir=out_dir,
            filename=f"skeleton_vs_bezier_K{K}.png"
        )
        save_svg_curve(bez_dense_pts, out_dir / f"bezier_K{K}.svg")
        save_png_curve(bez_dense_pts, out_dir / f"bezier_K{K}.png")

        summary_lines.append(f"Bezier K={K}, MSE={mse_bez:.6e}")

        # ------------- B-spline -------------
        if K <= degree_bspline:
            print(f"Skipping B-spline for K={K} (need K > degree={degree_bspline}).")
            summary_lines.append(f"BSpline K={K}: skipped (K <= degree)")
            continue

        print(f"Training B-spline (degree={degree_bspline}) with K={K} control points ({epochs} epochs)")
        bs_model, bs_losses = train_bspline(
            s_full,
            coords_full,
            n_ctrl=K,
            degree=degree_bspline,
            num_epochs=epochs,
            lr=learning_rate,
            verbose=True
        )

        with torch.no_grad():
            B_train_bs = torch.tensor(bspline_basis_matrix(s_full, K, degree_bspline), dtype=torch.float32)
            bs_train_pred = (B_train_bs @ bs_model.ctrl).cpu().numpy()
        mse_bs = np.mean((bs_train_pred - coords_full) ** 2)
        print(f"[B-spline] K={K}, final MSE={mse_bs:.6e}")

        np.save(out_dir / f"bspline_K{K}_losses.npy", bs_losses)
        with torch.no_grad():
            bs_ctrl = bs_model.ctrl.cpu().numpy()
        np.save(out_dir / f"bspline_K{K}_ctrl.npy", bs_ctrl)

        plot_loss(bs_losses, out_dir, filename=f"bspline_K{K}_loss_curve.png")

        _, bs_dense_pts = eval_bspline_dense(bs_model, K, degree=degree_bspline, num_points=1000)
        plot_skeleton_vs_curve(
            skeleton,
            bs_dense_pts,
            label=f"B-spline (K={K})",
            out_dir=out_dir,
            filename=f"skeleton_vs_bspline_K{K}.png"
        )
        save_svg_curve(bs_dense_pts, out_dir / f"bspline_K{K}.svg")
        save_png_curve(bs_dense_pts, out_dir / f"bspline_K{K}.png")

        summary_lines.append(f"BSpline K={K}, MSE={mse_bs:.6e}")

    # 7) Save summary
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
