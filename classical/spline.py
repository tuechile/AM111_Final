"""
Classical spline reconstruction of a handwritten curve.

Usage:
    python handwritten_curve_spline_fit.py input.png [num_points] [smooth_factor] [optional_out_prefix]

Arguments:
    input.png        : path to input image (single stroke or character)
    num_points       : # of points to sample on the reconstructed spline (default: 2000)
    smooth_factor    : smoothing parameter 's' for splprep (default: 1e-4; 0 gives interpolating spline)
    optional_out_prefix : base name for the output directory (default: derived from input name)

This script will:
1. Load a handwriting image.
2. Binarize and skeletonize to get a 1-pixel-wide curve.
3. Extract skeleton pixels, order them into a path, and parameterize by arc length s in [0,1].
4. Fit a parametric cubic B-spline to (x(s), y(s)).
5. Save:
   - a PNG comparing skeleton vs spline,
   - a dense spline rendering (PNG),
   - a vector spline path (SVG).
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
    """
    Load an image, convert to grayscale, binarize, and skeletonize.
    Returns a binary skeleton image (True at skeleton pixels).
    Saves intermediate visualization into out_dir.
    """
    img = imread(path, as_gray=True)

    # Otsu thresholding
    thresh = threshold_otsu(img)

    # Assume dark ink on light background:
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
        coords: array (M, 2) with normalized (x, y) coords in [0,1],
                with y flipped so orientation matches the original image.
    """
    ys, xs = np.nonzero(skeleton)
    if xs.size == 0:
        raise ValueError("No skeleton pixels found – check your image / thresholding.")

    N = xs.size
    coords_pix = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2) (x,y)

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

    # Choose a start node for Eulerian trail
    odd_vertices = [i for i in range(N) if len(adj[i]) % 2 == 1]
    if len(odd_vertices) >= 1:
        start = odd_vertices[0]
    else:
        start = 0

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

    trail = trail[::-1]  # forward order
    coords = coords_pix[trail]   # (M,2)

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
# 3. Spline fitting
# -----------------------------

def fit_parametric_spline(s, coords, smooth_factor=1e-4, degree=3):
    """
    Fit a 2D parametric B-spline (x(s), y(s)) to the given data.

    Args:
        s            : (N,) parameter values in [0,1] (arc length normalized).
        coords       : (N,2) array with x,y coordinates.
        smooth_factor: 's' parameter for splprep (0 for interpolation).
        degree       : spline degree k (1..5, typical 3 for cubic).

    Returns:
        tck : tuple returned by splprep (spline representation).
    """
    x = coords[:, 0]
    y = coords[:, 1]

    # splprep wants a sequence of 1D arrays; we pass u=s to enforce our parametrization
    tck, _ = splprep([x, y], u=s, s=smooth_factor, k=degree)
    return tck


def sample_spline(tck, num_points=2000):
    """
    Sample a fitted spline uniformly in s in [0,1].

    Returns:
        points: (num_points, 2) array of sampled (x,y) coordinates.
    """
    s_dense = np.linspace(0.0, 1.0, num_points)
    x_dense, y_dense = splev(s_dense, tck)
    pts = np.stack([x_dense, y_dense], axis=1)
    return pts


# -----------------------------
# 4. Saving utilities
# -----------------------------

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


def plot_skeleton_vs_spline(coords, spline_points, out_dir):
    """
    Plot the original skeleton points vs the fitted spline.
    """
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=2, label="Skeleton")
    ax.plot(spline_points[:, 0], spline_points[:, 1], 'r-', linewidth=1.0, label="Spline")
    ax.set_aspect('equal')
    ax.legend()
    ax.axis('off')
    out_path = out_dir / "spline_fit_vs_skeleton.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


# -----------------------------
# 5. Main script
# -----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python spline.py input.png [num_points] [smooth_factor] [optional_out_prefix]")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: input file not found: {img_path}")
        sys.exit(1)

    # Optional arguments
    num_points = int(sys.argv[2]) if len(sys.argv) >= 3 else 2000
    smooth_factor = float(sys.argv[3]) if len(sys.argv) >= 4 else 1e-4
    out_prefix = sys.argv[4] if len(sys.argv) >= 5 else None

    # Output directory naming (similar pattern to your NN script)
    img_path = Path(img_path)
    if out_prefix is None:
        out_prefix = img_path.with_suffix("")
    else:
        out_prefix = Path(out_prefix)

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    tagged_prefix = f"{out_prefix}_SplineID-{run_id}"
    out_dir = Path(tagged_prefix)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Saving outputs to directory: {out_dir}")
    print(f"num_points={num_points}, smooth_factor={smooth_factor}")

    # 1) Load + skeletonize
    skeleton = load_and_skeletonize(str(img_path), out_dir, show_intermediate=True)

    # 2) Extract ordered curve coordinates
    coords_ordered = extract_ordered_curve_from_skeleton(skeleton)

    # 3) Parameterize by arc length
    s, coords = parameterize_by_arclength(coords_ordered)

    # 4) Fit spline
    tck = fit_parametric_spline(s, coords, smooth_factor=smooth_factor, degree=3)

    # 5) Sample dense spline curve
    spline_pts = sample_spline(tck, num_points=num_points)

    # 6) Visualizations & exports
    plot_skeleton_vs_spline(coords, spline_pts, out_dir)

    spline_png_path = out_dir / "spline_curve.png"
    save_png_curve(spline_pts, spline_png_path)
    print(f"Saved spline PNG to: {spline_png_path}")

    spline_svg_path = out_dir / "spline_curve.svg"
    save_svg_curve(spline_pts, spline_svg_path)
    print(f"Saved spline SVG to: {spline_svg_path}")

    print("Done.")


if __name__ == "__main__":
    main()
