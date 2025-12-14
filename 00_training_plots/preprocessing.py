import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import os

import torch
import torch.nn as nn

# =========================
# 1. DATA STRUCTURES
# =========================

@dataclass
class Stroke:
    # points: (N, 2) array of [x, y]
    points: np.ndarray

@dataclass
class LetterExample:
    label: str            # e.g. "ă" or "vi_cursive_sample_1"
    strokes: list         # list[Stroke]


# =========================
# 2. STROKE CAPTURE
# =========================

def plot_epoch_grid(X_true, epoch_preds):
    """
    X_true: (N,2) array of true (x,y) points
    epoch_preds: dict {epoch: (N,2) array of predicted (x,y)}

    Shows a 2x3 grid of subplots, one per epoch.
    """
    # Ensure epochs are sorted and we have 6 of them
    epochs = sorted(epoch_preds.keys())
    assert len(epochs) == 6, f"Expected 6 checkpoints, got {len(epochs)}"

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.ravel()

    # Precompute common axis limits from true curve
    x_min, x_max = X_true[:,0].min(), X_true[:,0].max()
    y_min, y_max = X_true[:,1].min(), X_true[:,1].max()
    pad_x = 0.05 * (x_max - x_min)
    pad_y = 0.05 * (y_max - y_min)

    for ax, epoch in zip(axes, epochs):
        pred = epoch_preds[epoch]

        # True curve
        ax.plot(X_true[:,0], X_true[:,1], color="black", label="True", linewidth=2)
        # Predicted curve
        ax.plot(pred[:,0], pred[:,1], "--", color="red", label="Predicted", linewidth=2)

        ax.set_title(f"Epoch {epoch}")
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

        # Only put legend on the first subplot to avoid clutter
        if epoch == epochs[0]:
            ax.legend()

    plt.tight_layout()
    plt.show()


def plot_stroke(points, title="Stroke", color="red"):
    """Plot a single stroke (N,2)."""
    plt.plot(points[:,0], points[:,1], color=color)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.grid(True)

def show_before_after(stroke_raw, stroke_norm):
    plt.figure(figsize=(10,4))

    # --- Left: Raw ---
    plt.subplot(1,2,1)
    plt.plot(stroke_raw[:,0], stroke_raw[:,1], color="blue")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()   # 👈 match the drawing canvas
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Before Normalize (Raw)")
    plt.grid(True)

    # --- Right: Normalized ---
    plt.subplot(1,2,2)
    plt.plot(stroke_norm[:,0], stroke_norm[:,1], color="green")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("After Normalize")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def visualize_training_progress(X_true, X_pred, epoch, save_dir="training_plots"):
    """
    Plot true curve vs predicted (NN output) every checkpoint.
    X_true, X_pred: (N,2) or (N,F) arrays. Uses only the first 2 dimensions (x,y).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(6,3))
    plt.plot(X_true[:,0], X_true[:,1], label="True", linewidth=2, color="black")
    plt.plot(X_pred[:,0], X_pred[:,1], "--", label="Predicted", linewidth=2, color="red")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"NN Approximation at Epoch {epoch}")
    plt.legend()
    plt.grid(True)

class StrokeCollector:
    """
    Use left mouse button to draw.
    Release to end a stroke.
    Press 'enter' when you're done with the whole letter.
    Press 'escape' to discard and quit.
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Draw your cursive letter.\nLeft-click+drag = stroke, Enter = finish, Esc = quit")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.invert_yaxis()  # optional, feels more like screen coords
        self.strokes = []
        self._current_xs = []
        self._current_ys = []
        self._drawing = False

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.finished = False
        self.cancelled = False

    def on_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self._drawing = True
            self._current_xs = [event.xdata]
            self._current_ys = [event.ydata]

    def on_motion(self, event):
        if self._drawing and event.inaxes == self.ax:
            self._current_xs.append(event.xdata)
            self._current_ys.append(event.ydata)
            self.ax.plot(self._current_xs[-2:], self._current_ys[-2:], linewidth=2)
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 1 and self._drawing:
            self._drawing = False
            if len(self._current_xs) > 1:
                pts = np.stack([self._current_xs, self._current_ys], axis=1)
                self.strokes.append(Stroke(points=pts))
            self._current_xs = []
            self._current_ys = []

    def on_key(self, event):
        if event.key == 'enter':
            self.finished = True
            plt.close(self.fig)
        elif event.key == 'escape':
            self.cancelled = True
            plt.close(self.fig)

    def collect(self) -> list:
        plt.show()
        if self.cancelled:
            return []
        return self.strokes


# =========================
# 3. PREPROCESSING
# =========================

def normalize_stroke(points: np.ndarray) -> np.ndarray:
    """
    Center and scale stroke to fit roughly in [-1, 1] x [-1, 1].
    points: (N, 2)
    """
    # Translate so mean is at origin
    mean = points.mean(axis=0, keepdims=True)
    pts = points - mean

    # Scale by max abs coordinate
    max_abs = np.abs(pts).max()
    if max_abs > 0:
        pts = pts / max_abs
    return pts


def resample_stroke(points: np.ndarray, num_points: int = 100) -> np.ndarray:
    """
    Resample a stroke to a fixed number of points using arc-length parameterization.
    points: (N, 2)
    returns: (num_points, 2)
    """
    if points.shape[0] < 2:
        # Degenerate stroke; just repeat the point
        return np.repeat(points[:1], num_points, axis=0)

    # Calculate cumulative arc length
    deltas = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = arc[-1]
    if total_length == 0:
        return np.repeat(points[:1], num_points, axis=0)

    arc_norm = arc / total_length

    # Target positions in [0, 1]
    target = np.linspace(0.0, 1.0, num_points)

    # Interpolate x and y separately along arc-length
    xs = np.interp(target, arc_norm, points[:, 0])
    ys = np.interp(target, arc_norm, points[:, 1])

    return np.stack([xs, ys], axis=1)


def compute_features(points: np.ndarray,
                     add_derivatives: bool = True,
                     add_curvature: bool = True) -> np.ndarray:
    """
    Turn a (N, 2) position sequence into feature vectors.
    Base features: x, y
    Optional: dx, dy, curvature
    returns: (N, F)
    """
    x = points[:, 0]
    y = points[:, 1]
    feats = [x[:, None], y[:, None]]

    if add_derivatives or add_curvature:
        # finite differences
        dx = np.gradient(x)
        dy = np.gradient(y)
        if add_derivatives:
            feats.extend([dx[:, None], dy[:, None]])

        if add_curvature:
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            # curvature formula for param curve (x(t), y(t))
            eps = 1e-6
            num = np.abs(dx * ddy - dy * ddx)
            den = (dx * dx + dy * dy) ** 1.5 + eps
            kappa = num / den
            feats.append(kappa[:, None])

    return np.concatenate(feats, axis=1)


def preprocess_letter(letter_example: LetterExample,
                      num_points_per_stroke=100,
                      add_derivatives=True,
                      add_curvature=True):

    processed = []
    for stroke in letter_example.strokes:

        raw_pts = stroke.points

        # ------ (A) Normalize --------
        norm_pts = normalize_stroke(raw_pts)

        # Display comparison
        show_before_after(raw_pts, norm_pts)

        # ------ (B) Resample ---------
        pts = resample_stroke(norm_pts, num_points=num_points_per_stroke)

        # ------ (C) Feature Calculation ----
        feats = compute_features(
            pts,
            add_derivatives=add_derivatives,
            add_curvature=add_curvature
        )
        processed.append(feats)

    return processed



# =========================
# 4. SAVING / LOADING DATASETS
# =========================

def save_letter_example(letter_example: LetterExample,
                        processed_strokes: list,
                        out_dir: str):
    """
    Save:
      - raw strokes as JSON
      - processed data as .npz
    """
    os.makedirs(out_dir, exist_ok=True)
    base_name = letter_example.label.replace(" ", "_")

    # Save raw strokes
    raw_path = os.path.join(out_dir, base_name + "_raw.json")
    raw_dict = {
        "label": letter_example.label,
        "strokes": [stroke.points.tolist() for stroke in letter_example.strokes]
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_dict, f, ensure_ascii=False, indent=2)

    # Save processed numpy arrays (one per stroke)
    proc_path = os.path.join(out_dir, base_name + "_processed.npz")
    # store under names s0, s1, ...
    np.savez(proc_path, **{f"s{i}": s for i, s in enumerate(processed_strokes)})

    print(f"Saved raw strokes to {raw_path}")
    print(f"Saved processed data to {proc_path}")


def load_processed_letter(npz_path: str):
    """
    Load processed strokes from .npz file.
    Returns list of arrays.
    """
    data = np.load(npz_path)
    strokes = [data[k] for k in sorted(data.files)]
    return strokes


# =========================
# 5. SIMPLE NEURAL NET DEMO
# =========================

class SimpleStrokeAutoencoder(nn.Module):
    """
    Very small autoencoder:
      input: (N, F)
      output: (N, F)
    You would train it to reconstruct the stroke.
    """
    def __init__(self, feature_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def plot_multi_epoch_progress(X_true, epoch_preds):
    """
    Show a single matplotlib figure with:
      - True stroke
      - Predictions at epochs 500,1000,...,3000
    """
    plt.figure(figsize=(8,4))

    # Plot true curve
    plt.plot(X_true[:,0], X_true[:,1], label="True", linewidth=3, color="black")

    # Plot each predicted curve
    for epoch in sorted(epoch_preds.keys()):
        pred = epoch_preds[epoch]
        plt.plot(pred[:,0], pred[:,1], "--", label=f"Epoch {epoch}")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.legend()
    plt.title("NN approximations at epochs 500–3000")
    plt.tight_layout()
    plt.show()


def demo_train_autoencoder(strokes: list, epochs: int = 3000):
    """
    Train an autoencoder and show predictions at
    epochs 500, 1000, 1500, 2000, 2500, 3000
    in a 2x3 grid of subplots.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Combine strokes into one big dataset
    X = np.concatenate(strokes, axis=0)  # (total_points, F)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    feature_dim = X.shape[1]
    model = SimpleStrokeAutoencoder(feature_dim=feature_dim, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Epochs we want to visualize
    checkpoints = [1, 100, 500, 1000, 2000, 3000]
    epoch_preds = {}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, X_t)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs}, loss={loss.item():.6f}")

        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                pred = model(X_t).cpu().numpy()
            # store only x,y columns for plotting
            epoch_preds[epoch] = pred[:, :2]

    # After training, show 2x3 subplot grid
    X_xy = X[:, :2]
    plot_epoch_grid(X_true=X_xy, epoch_preds=epoch_preds)

    return model



# =========================
# 6. MAIN: INTERACTIVE CAPTURE + PIPELINE
# =========================

if __name__ == "__main__":
    label = input("Enter label/name for this letter (e.g. 'a_cursive_1'): ").strip() or "letter"

    collector = StrokeCollector()
    strokes = collector.collect()

    if not strokes:
        print("No strokes collected or cancelled.")
        exit(0)

    letter_example = LetterExample(label=label, strokes=strokes)

    # Preprocess (you can tweak these hyperparameters)
    processed_strokes = preprocess_letter(
        letter_example,
        num_points_per_stroke=100,
        add_derivatives=True,
        add_curvature=True,
    )

    # Save to disk
    save_letter_example(letter_example, processed_strokes, out_dir="cursive_dataset")

    # Optional: quick NN demo
    do_demo = input("Run simple autoencoder demo on this letter? [y/N]: ").strip().lower()
    if do_demo == "y":
        model, X, recon = demo_train_autoencoder(processed_strokes, epochs=3000)

        # Plot original vs reconstructed for sanity
        # We’ll only plot x,y, not derivatives/curvature.
        plt.figure(figsize=(6, 3))
        plt.plot(X[:, 0], X[:, 1], label="original", linewidth=2)
        plt.plot(recon[:, 0], recon[:, 1], "--", label="reconstructed")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        plt.title("Original vs reconstructed stroke (autoencoder)")
        plt.show()
        