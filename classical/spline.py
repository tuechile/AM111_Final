"""
Classical spline reconstruction of a handwritten curve.

Now also saves:
 - spline_tck.npz : mathematical representation of the spline (knots, control points, degree)
 - spline_geometry.csv : sampled s, x(s), y(s), curvature(s)
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from scipy.interpolate import splprep, splev


# -----------------------------
# 1. Load image and skeletonize
# -----------------------------

def load_and_skeletonize(path, out_dir, show_intermediate=True):
    img = imread(path, as_gray=True)

    thresh = threshold_otsu(img)
    binary = img < thresh
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
    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("No skeleton pixels found.")

    N = xs.size
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)

    coord_to_idx = {(ys[i], xs[i]): i for i in range(N)}

    adj = [[] for _ in range(N)]
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]

    for i in range(N):
        r, c = ys[i], xs[i]
        for dr, dc in neighbors:
            j = coord_to_idx.get((r + dr, c + dc))
            if j is not None:
                adj[i].append(j)

    odd_vertices = [i for i in range(N) if len(adj[i]) % 2 == 1]
    start = odd_vertices[0] if odd_vertices else 0

    # Hierholzer's algorithm
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

    trail = trail[::-1]
    coords = coords_pix[trail]

    # Normalize into [0,1]
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    coords = (coords - min_vals) / (max_vals - min_vals + 1e-8)

    # Flip y upward
    coords[:, 1] = 1.0 - coords[:, 1]

    return coords


def parameterize_by_arclength(coords):
    diffs = coords[1:] - coords[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    s = arc / (arc[-1] + 1e-8)
    return s, coords


# -----------------------------
# 3. Spline fitting utilities
# -----------------------------

def fit_parametric_spline(s, coords, smooth_factor=1e-4, degree=3):
    x = coords[:, 0]
    y = coords[:, 1]
    tck, _ = splprep([x, y], u=s, s=smooth_factor, k=degree)
    return tck


def compute_curvature_along_spline(tck, num_points=2000):
    s_dense = np.linspace(0.0, 1.0, num_points)

    # Position
    x, y = splev(s_dense, tck)

    # First derivative
    dx, dy = splev(s_dense, tck, der=1)

    # Second derivative
    ddx, ddy = splev(s_dense, tck, der=2)

    # Curvature κ(s)
    num = dx * ddy - dy * ddx
    denom = (dx**2 + dy**2)**1.5 + 1e-12
    kappa = num / denom

    pts = np.stack([x, y], axis=1)
    return s_dense, pts, kappa


def save_tck_npz(tck, path):
    t, c, k = tck
    cx, cy = c
    np.savez(path, t=t, cx=cx, cy=cy, k=k)


# -----------------------------
# 4. Saving utilities
# -----------------------------

def save_svg_curve(points, path):
    path = Path(path)
    with open(path, "w") as f:
        f.write('<svg viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<polyline points="')
        for x, y in points:
            y_svg = 1.0 - y
            f.write(f"{x},{y_svg} ")
        f.write('" fill="none" stroke="black" stroke-width="0.002"/>\n')
        f.write('</svg>\n')


def save_png_curve(points, path, dpi=300):
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.plot(points[:, 0], points[:, 1], 'k-')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def plot_skeleton_vs_spline(coords, spline_points, out_dir):
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=2, label="Skeleton")
    ax.plot(spline_points[:, 0], spline_points[:, 1], 'r-', linewidth=1.0, label="Spline")
    ax.set_aspect('equal')
    ax.legend()
    ax.axis('off')
    fig.savefig(out_dir / "spline_fit_vs_skeleton.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# 5. Main script
# -----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python spline.py input.png [num_points] [smooth_factor] [optional_out_prefix]")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    num_points = int(sys.argv[2]) if len(sys.argv) >= 3 else 2000
    smooth_factor = float(sys.argv[3]) if len(sys.argv) >= 4 else 1e-4
    out_prefix = sys.argv[4] if len(sys.argv) >= 5 else None

    if out_prefix is None:
        out_prefix = img_path.with_suffix("")
    else:
        out_prefix = Path(out_prefix)

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(f"{out_prefix}_SplineID-{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Saving to: {out_dir}")
    print(f"num_points={num_points}, smooth_factor={smooth_factor}")

    skeleton = load_and_skeletonize(str(img_path), out_dir, show_intermediate=True)
    coords_ordered = extract_ordered_curve_from_skeleton(skeleton)
    s, coords = parameterize_by_arclength(coords_ordered)
    tck = fit_parametric_spline(s, coords, smooth_factor=smooth_factor, degree=3)

    # Sample spline + curvature
    s_dense, spline_pts, kappa = compute_curvature_along_spline(tck, num_points)

    # --- SAVE NEW FILES ---
    # 1. Mathematical spline representation
    npz_path = out_dir / "spline_tck.npz"
    save_tck_npz(tck, npz_path)
    print(f"Saved spline representation (tck) to: {npz_path}")

    # 2. CSV containing s, x(s), y(s), curvature
    csv_path = out_dir / "spline_geometry.csv"
    data = np.column_stack([s_dense, spline_pts[:, 0], spline_pts[:, 1], kappa])
    np.savetxt(csv_path, data, delimiter=",",
               header="s,x,y,curvature", comments="")
    print(f"Saved spline geometry CSV to: {csv_path}")

    # Existing visual outputs
    plot_skeleton_vs_spline(coords, spline_pts, out_dir)
    save_png_curve(spline_pts, out_dir / "spline_curve.png")
    save_svg_curve(spline_pts, out_dir / "spline_curve.svg")

    print("Done.")


if __name__ == "__main__":
    main()