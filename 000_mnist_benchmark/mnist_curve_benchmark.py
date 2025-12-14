"""
Benchmark the handwritten_curve_fit pipeline on many MNIST digits.

Usage:
    python mnist_curve_benchmark.py epochs learning_rate num_samples [optional_out_prefix]

Example:
    python mnist_curve_benchmark.py 2000 1e-3 200 mnist_bench

This script:
  - Loads MNIST training set (28x28 grayscale digits).
  - For each chosen image:
      * skeletonizes it,
      * runs the 3-phase curve fit pipeline from handwritten_curve_fit.py,
      * records best K, compression ratio, and final MSE.
  - Saves per-sample stats and a few summary plots.

Requires:
  - torch
  - torchvision
  - numpy
  - matplotlib
  - scikit-image
  - handwritten_curve_fit.py in the same directory
"""

import sys
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

import torch
from torchvision import datasets, transforms

# Import your existing pipeline pieces
from auto_prune import (
    extract_ordered_curve_from_skeleton,
    parameterize_by_arclength,
    train_with_importance_weights,
    search_best_K,
    get_top_k_indices_by_weight,
    train_curve_mlp,
    compute_mse,
    plot_loss,
    plot_skeleton_vs_reconstruction,
    # hyperparameters / constants
    TARGET_MSE,
    MIN_POINTS,
    IMPORTANCE_EPOCHS_MULTIPLIER,
    SEARCH_EPOCHS_MULTIPLIER,
    MAX_SEARCH_STEPS,
    K_TOL,
    LAMBDA_SPARSITY,
    HIDDEN_DIM,
    NUM_HIDDEN_LAYERS,
)


# -------------------------------------
# 1. Skeletonization for MNIST images
# -------------------------------------

def skeletonize_mnist_image(img_tensor, out_dir=None, show_intermediate=False, idx=None):
    """
    Take a MNIST image (torch.Tensor, shape [1, H, W] in [0,1]) and return a
    skeletonized boolean array suitable for extract_ordered_curve_from_skeleton.

    Optionally saves a grayscale/binary/skeleton visualization for debugging.
    """
    img = img_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # (H, W) in [0,1]

    # Otsu thresholding
    thresh = threshold_otsu(img)

    # MNIST digits are dark ink on light background
    binary = img < thresh

    # Skeletonize
    skeleton = skeletonize(binary)

    if show_intermediate and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        ax = axes.ravel()
        ax[0].imshow(img, cmap="gray")
        ax[0].set_title(f"MNIST grayscale (idx={idx})")
        ax[1].imshow(binary, cmap="gray")
        ax[1].set_title("Binary")
        ax[2].imshow(skeleton, cmap="gray")
        ax[2].set_title("Skeleton")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "mnist_preprocessing.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    return skeleton


# -------------------------------------
# 2. Benchmark loop for many samples
# -------------------------------------

def run_single_mnist_sample(img_tensor,
                            label,
                            epochs,
                            lr,
                            importance_epochs,
                            search_epochs,
                            root_out_dir,
                            sample_index,
                            dataset_index,
                            save_visuals=True):
    """
    Run the full 3-phase pipeline on a single MNIST digit.

    Returns a dict with summary statistics.
    """
    # Decide per-sample output directory (optional)
    sample_dir = None
    if save_visuals:
        sample_dir = root_out_dir / f"sample_{sample_index:05d}_idx-{dataset_index}_label-{label}"
        sample_dir.mkdir(parents=True, exist_ok=True)

    # 1) Skeletonize
    skeleton = skeletonize_mnist_image(
        img_tensor,
        out_dir=sample_dir,
        show_intermediate=save_visuals,
        idx=dataset_index,
    )

    # Guard: if nothing in skeleton, skip
    if np.count_nonzero(skeleton) == 0:
        raise ValueError("No skeleton pixels found for this sample.")

    # 2) Extract ordered curve
    coords_ordered = extract_ordered_curve_from_skeleton(skeleton)

    # 3) Parameterize by arc length
    s_full, coords_full = parameterize_by_arclength(coords_ordered)
    N_full = len(s_full)

    # 4) Phase 1: Importance learning
    imp_model, weights, imp_losses, mean_w_history = train_with_importance_weights(
        s_full,
        coords_full,
        num_epochs=importance_epochs,
        lr=lr,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        lambda_sparsity=LAMBDA_SPARSITY,
    )

    if save_visuals:
        np.save(sample_dir / "importance_weights.npy", weights)
        # Reuse plot_loss (it uses out_dir / filename)
        plot_loss(imp_losses, sample_dir, filename="importance_loss_curve.png")
        # Simple weights plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(np.sort(weights)[::-1])
        ax.set_title("Sorted importance weights")
        ax.set_xlabel("Index (sorted)")
        ax.set_ylabel("w_i")
        ax.grid(True)
        fig.savefig(sample_dir / "importance_weights_sorted.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 5) Phase 2: K-search
    best_K, best_K_mse = search_best_K(
        s_full,
        coords_full,
        weights,
        target_mse=TARGET_MSE,
        epochs_for_search=search_epochs,
        lr=lr,
        min_points=MIN_POINTS,
        max_steps=MAX_SEARCH_STEPS,
        k_tol=K_TOL,
    )

    # 6) Phase 3: Final training on best_K points
    final_keep_idx = get_top_k_indices_by_weight(weights, best_K)
    s_sel_final = s_full[final_keep_idx]
    coords_sel_final = coords_full[final_keep_idx]

    final_model, final_losses = train_curve_mlp(
        s_sel_final,
        coords_sel_final,
        num_epochs=epochs,
        lr=lr,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        verbose=False,
    )

    if save_visuals:
        plot_loss(final_losses, sample_dir, filename="final_loss_curve.png")
        plot_skeleton_vs_reconstruction(skeleton, final_model, sample_dir, num_points=2000)

    # Evaluate final model on full curve
    final_mse = compute_mse(final_model, s_full, coords_full)

    # Save per-sample summary if we have a sample_dir
    if save_visuals and sample_dir is not None:
        with open(sample_dir / "summary.txt", "w") as f:
            f.write(f"MNIST idx: {dataset_index}\n")
            f.write(f"Label: {label}\n")
            f.write(f"Num full points: {N_full}\n")
            f.write(f"Best K points: {best_K}\n")
            f.write(f"Compression ratio: {best_K / N_full:.4f}\n")
            f.write(f"Target MSE: {TARGET_MSE:.6e}\n")
            f.write(f"Best-K search MSE (search phase): {best_K_mse:.6e}\n")
            f.write(f"Final MSE (full curve): {final_mse:.6e}\n")
            f.write(f"Lambda sparsity (Method B): {LAMBDA_SPARSITY}\n")
            f.write(f"Importance epochs: {importance_epochs}\n")
            f.write(f"Search epochs per K: {search_epochs}\n")
            f.write(f"Final epochs: {epochs}\n")

    return {
        "dataset_index": dataset_index,
        "label": int(label),
        "N_full": int(N_full),
        "best_K": int(best_K),
        "compression_ratio": float(best_K / N_full),
        "search_mse": float(best_K_mse),
        "final_mse": float(final_mse),
    }


# -------------------------------------
# 3. Main benchmark entry point
# -------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python mnist_curve_benchmark.py epochs learning_rate num_samples [optional_out_prefix]")
        sys.exit(1)

    epochs = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    num_samples = int(sys.argv[3])
    out_prefix = sys.argv[4] if len(sys.argv) >= 5 else "mnist_curve_benchmark"

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    root_out_dir = Path(f"{out_prefix}_ID-{run_id}")
    root_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Saving outputs to directory: {root_out_dir}")

    # Derive epochs for importance and search phases from your multipliers
    importance_epochs = max(1, int(IMPORTANCE_EPOCHS_MULTIPLIER * epochs))
    search_epochs = max(1, int(SEARCH_EPOCHS_MULTIPLIER * epochs))

    print(f"Importance learning epochs: {importance_epochs}")
    print(f"Search epochs per K: {search_epochs}")

    # Load MNIST (training set)
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    print(f"MNIST train set size: {len(mnist_train)}")

    # Randomly choose samples (without replacement)
    all_indices = list(range(len(mnist_train)))
    random.shuffle(all_indices)
    chosen_indices = all_indices[:num_samples]

    print(f"Benchmarking on {num_samples} random MNIST digits")

    # Only save detailed visualizations for the first few samples
    NUM_VISUAL_SAMPLES = min(10, num_samples)

    stats = []
    num_failed = 0

    for i, dataset_idx in enumerate(chosen_indices):
        img_tensor, label = mnist_train[dataset_idx]
        print(f"\n===== Sample {i+1}/{num_samples} (dataset idx = {dataset_idx}, label = {label}) =====")

        save_visuals = (i < NUM_VISUAL_SAMPLES)

        try:
            sample_stats = run_single_mnist_sample(
                img_tensor=img_tensor,
                label=label,
                epochs=epochs,
                lr=learning_rate,
                importance_epochs=importance_epochs,
                search_epochs=search_epochs,
                root_out_dir=root_out_dir,
                sample_index=i,
                dataset_index=dataset_idx,
                save_visuals=save_visuals,
            )
            stats.append(sample_stats)

            print(
                f"  -> N_full={sample_stats['N_full']}, "
                f"K={sample_stats['best_K']} "
                f"(ratio={sample_stats['compression_ratio']:.3f}), "
                f"final MSE={sample_stats['final_mse']:.3e}"
            )
        except Exception as e:
            num_failed += 1
            print(f"  !! Failed on sample (idx={dataset_idx}, label={label}) with error: {e}")

    # Convert stats to arrays / save
    if not stats:
        print("No successful samples. Exiting.")
        sys.exit(0)

    # Save as CSV
    csv_path = root_out_dir / "mnist_curve_stats.csv"
    with open(csv_path, "w") as f:
        f.write("dataset_index,label,N_full,best_K,compression_ratio,search_mse,final_mse\n")
        for s in stats:
            f.write(
                f"{s['dataset_index']},{s['label']},"
                f"{s['N_full']},{s['best_K']},"
                f"{s['compression_ratio']:.6f},"
                f"{s['search_mse']:.8e},{s['final_mse']:.8e}\n"
            )
    print(f"\nSaved per-sample stats to: {csv_path}")

    # Summary statistics
    N_full_arr = np.array([s["N_full"] for s in stats])
    K_arr = np.array([s["best_K"] for s in stats])
    ratio_arr = np.array([s["compression_ratio"] for s in stats])
    final_mse_arr = np.array([s["final_mse"] for s in stats])

    print("\n===== Summary over successful samples =====")
    print(f"Successful samples: {len(stats)} / {num_samples} (failed: {num_failed})")
    print(f"Avg N_full: {N_full_arr.mean():.1f}")
    print(f"Avg best_K: {K_arr.mean():.1f}")
    print(f"Avg compression ratio K/N_full: {ratio_arr.mean():.4f}")
    print(f"Median compression ratio: {np.median(ratio_arr):.4f}")
    print(f"Avg final MSE: {final_mse_arr.mean():.4e}")
    print(f"Median final MSE: {np.median(final_mse_arr):.4e}")

    # Save a small summary text
    with open(root_out_dir / "summary_overview.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Num samples requested: {num_samples}\n")
        f.write(f"Num successful samples: {len(stats)}\n")
        f.write(f"Num failed samples: {num_failed}\n\n")
        f.write(f"Avg N_full: {N_full_arr.mean():.1f}\n")
        f.write(f"Avg best_K: {K_arr.mean():.1f}\n")
        f.write(f"Avg compression ratio K/N_full: {ratio_arr.mean():.4f}\n")
        f.write(f"Median compression ratio: {np.median(ratio_arr):.4f}\n")
        f.write(f"Avg final MSE: {final_mse_arr.mean():.4e}\n")
        f.write(f"Median final MSE: {np.median(final_mse_arr):.4e}\n")

    # Summary plots
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(K_arr, bins=20)
    ax.set_title("Distribution of best_K over MNIST samples")
    ax.set_xlabel("best_K")
    ax.set_ylabel("count")
    fig.savefig(root_out_dir / "hist_best_K.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(ratio_arr, bins=20)
    ax.set_title("Distribution of compression ratio K/N_full")
    ax.set_xlabel("K / N_full")
    ax.set_ylabel("count")
    fig.savefig(root_out_dir / "hist_compression_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(final_mse_arr, bins=20)
    ax.set_title("Distribution of final MSE")
    ax.set_xlabel("final MSE")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    fig.savefig(root_out_dir / "hist_final_mse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nDone. Summary plots and stats saved.")


if __name__ == "__main__":
    main()
