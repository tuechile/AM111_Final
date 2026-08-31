Experimenting weights to compare a classical spline
baseline against several neural parameterizations, and benchmarks the neural
approach across the MNIST dataset

## Pipeline

Each stage below takes a stroke image, binarizes and skeletonizes it into an
ordered 1-pixel-wide curve parameterized by arc length `s ∈ [0, 1]`, then fits
a model `f(s) = (x, y)` to that curve.

| Stage | Script(s) | Approach |
|---|---|---|
| [0_classical](0_classical/) | `spline.py` | Classical B-spline fit (baseline). |
| [1_train_using_all_points](1_train_using_all_points/) | `experiment.py`, `skeletonize.py` | MLP trained on every skeleton point (no compression). |
| [2_auto_prune](2_auto_prune/) | `auto_prune.py`, `subset_points.py`, `prune_reward.py` | Learns the smallest point subset `K` that still meets a target MSE, via iterative pruning (Method A) or learnable per-point importance weights (Method B). |
| [3_mixed_methods](3_mixed_methods/) | `curve_neural.py`, `polyline_neural.py`, `aggressive.py` | Fits a fixed number of control points `K` using a Bézier curve, a cubic B-spline, or a learned polyline. |
| [nofixedbasis_neural.py](nofixedbasis_neural.py) | — | Same idea as stage 3, but with a learned (non-fixed) neural basis instead of Bézier/B-spline/polyline. |
| [000_mnist_benchmark](000_mnist_benchmark/) | `mnist_curve_benchmark.py` | Runs the stage-2 pruning pipeline across many MNIST digits and aggregates compression ratio / MSE statistics. |
| [00_training_plots](00_training_plots/) | `preprocessing.py`, `test.py` | Exploratory scripts used to generate the illustrative training-snapshot plots (`epoch_*.png`). |

[data/](data/) holds the MNIST dataset, downloaded automatically by `torchvision` on first use.

## Output folders

Every run writes its results into a timestamped directory named
`<config>_ID-<timestamp>/` (e.g. `c_redo_ID-20251215_0229/`), containing
reconstruction plots, loss curves, saved model weights, and — where
applicable — a `summary.txt` with the run's metrics. Only the most recent run
per configuration is kept in the repository; earlier, superseded re-runs of
the same configuration have been removed to keep the repository size down.
