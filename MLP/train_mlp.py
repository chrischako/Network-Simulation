"""Train and use an MLP regressor to predict network data rates.

The network simulator writes one result row for each transmitter/receiver
pair.  Each row contains the pair identifiers, the distance between the two
nodes, the free-space path loss (FSPL), the signal-to-interference-plus-noise
ratio (SINR), and the resulting Shannon capacity.  This script treats the
capacity as the value to predict and the other numeric values as input
features.

The model has three outputs: FSPL, SINR, and ``log10(capacity)``.  FSPL and
SINR are predicted rather than supplied as inputs, which avoids target
leakage and makes it possible to calculate an error for each output.  The
capacity target is logarithmic because rates range from thousands to tens of
millions of bits per second.  Predictions are converted back to ordinary
capacity values before they are displayed or written to disk.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# joblib serializes the trained preprocessing and neural-network pipeline so
# it can be loaded later without retraining.
import joblib
# Matplotlib creates the comparison figure after the test set is evaluated.
import matplotlib.pyplot as plt
# NumPy stores the parsed feature matrix and performs numerical conversions.
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# These names and this order define the public input format of the model. FSPL
# and SINR are deliberately absent: they are prediction targets now. The same
# order must be used by --predict when predicting one row.
FEATURE_NAMES = (
    "t_count",
    "r_count",
    "tx_idx",
    "rx_idx",
    "distance",
)

# The MLP produces these three values for every input row. The final target is
# stored internally as log10(capacity), then converted back to bps for output.
TARGET_NAMES = ("fspl_db", "sinr_db", "capacity_bps")

# A result row in results.txt looks like:
# T=1, R=16 | distance=62.5073 m | fspl=75.9646 dB |
# sinr=-17.8288 dB | capacity=4.718138e+05 bps
# The named groups let the parser extract only the numeric values it needs.
RESULT_PATTERN = re.compile(
    r"T=(?P<tx>\d+),\s*R=(?P<rx>\d+)\s*\|\s*"
    r"distance=(?P<distance>[\d.eE+-]+)\s*m\s*\|\s*"
    r"fspl=(?P<fspl>[\d.eE+-]+)\s*dB\s*\|\s*"
    r"sinr=(?P<sinr>[\d.eE+-]+)\s*dB\s*\|\s*"
    r"capacity=(?P<capacity>[\d.eE+-]+)\s*bps"
)
TRIAL_PATTERN = re.compile(
    r"Trial\s+(?P<trial>\d+):\s*T=(?P<t>\d+),\s*R=(?P<r>\d+)"
)


def load_results(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse simulator text into input features and three target values.

    ``results.txt`` stores the transmitter and receiver counts on a separate
    ``Trial`` line, followed by several pair-result lines.  The current role
    counts are therefore remembered while the file is scanned.  For every
    pair-result line, one feature row and one capacity target are appended.

    Args:
        path: Text file produced by ``metrics.save_all_results``.

    Returns:
        A tuple ``(features, targets)``. ``features`` has one row per network
        link and columns in ``FEATURE_NAMES`` order. ``targets`` has columns
        for FSPL in dB, SINR in dB, and capacity in bps.

    Raises:
        ValueError: If a result row appears before its trial role counts, or
            if the file contains no recognizable result rows.
    """
    features = []
    targets = []
    t_count = r_count = None

    # Read all lines as text because the source file also contains separators,
    # headers, and trial labels that are not numeric data rows.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        trial_match = TRIAL_PATTERN.search(line)
        if trial_match:
            # These values apply to every link until the next Trial line.
            t_count = int(trial_match.group("t"))
            r_count = int(trial_match.group("r"))
            continue

        result_match = RESULT_PATTERN.search(line)
        if not result_match:
            continue
        if t_count is None or r_count is None:
            raise ValueError(f"Missing trial role counts before line {line_number}")

        # Convert only model inputs to the feature matrix. FSPL and SINR are
        # kept for the target matrix below, not fed into the model as inputs.
        features.append(
            [
                t_count,
                r_count,
                int(result_match.group("tx")),
                int(result_match.group("rx")),
                float(result_match.group("distance")),
            ]
        )
        # These are the three supervised-learning targets the MLP must learn.
        targets.append(
            [
                float(result_match.group("fspl")),
                float(result_match.group("sinr")),
                float(result_match.group("capacity")),
            ]
        )

    if not features:
        raise ValueError(f"No result rows found in {path}")
    # float64 preserves the decimal values and is the standard numeric type
    # expected by scikit-learn estimators.
    return np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def resolve_data_path(requested: Path) -> Path:
    """Find the requested input file and handle the project's filename typo.

    The command defaults to ``relut.txt`` because that is the filename
    requested for the training data.  This repository currently contains
    ``results.txt`` instead, so that name is used as a fallback when
    ``relut.txt`` does not exist.
    """
    if requested.exists():
        return requested
    if requested.name == "relut.txt":
        fallback = requested.with_name("results.txt")
        if fallback.exists():
            print(f"{requested} was not found; using {fallback} instead.")
            return fallback
    raise FileNotFoundError(f"Input file not found: {requested}")


def build_model() -> Pipeline:
    """Build the preprocessing and multi-output MLP regression pipeline.

    StandardScaler puts every input feature on a comparable scale.  Without
    scaling, values such as node indexes and distances would have a very
    different influence from SINR and FSPL values.  The MLP then uses two
    fully connected hidden layers with ReLU activation to learn nonlinear
    relationships between the network measurements and the three targets.
    MLPRegressor accepts a two-dimensional target matrix, so its output layer
    contains one output for FSPL, one for SINR, and one for log-capacity.
    """
    return Pipeline(
        [
            # Fit the scaling parameters only on the training data through the
            # pipeline, preventing test-set information from leaking into fit.
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    # The first layer learns broad interactions; the second
                    # layer compresses those interactions before the output.
                    hidden_layer_sizes=(64, 32),
                    # ReLU is a standard nonlinear activation for regression
                    # networks and avoids restricting the output to a range.
                    activation="relu",
                    # Adam adapts the update size for each learned weight.
                    solver="adam",
                    # This is the initial size of each Adam optimization step.
                    learning_rate_init=0.001,
                    # Stop after convergence or this many passes through data.
                    max_iter=2000,
                    # Reserve part of the training data to stop before
                    # overfitting when validation performance stops improving.
                    early_stopping=True,
                    validation_fraction=0.15,
                    # Repeated runs produce the same split, initialization,
                    # and optimization behavior for easier comparison.
                    random_state=42,
                ),
            ),
        ]
    )


def plot_results(actual: np.ndarray, predicted: np.ndarray, plot_path: Path) -> None:
    """Save a visual comparison of measured and MLP-predicted data rates.

    The left panel uses logarithmic axes because the rates cover several
    orders of magnitude.  Points near the dashed diagonal are accurate.  The
    right panel sorts both arrays by actual rate, making it easy to see where
    the model follows the overall low-to-high rate pattern and where it misses
    individual values.
    """
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert bps to Mbps only for readable plot labels; the model and saved
    # CSV continue to use bps as their precise unit.
    actual_mbps = actual / 1e6
    predicted_mbps = predicted / 1e6
    limits = [min(actual_mbps.min(), predicted_mbps.min()), max(actual_mbps.max(), predicted_mbps.max())]

    # Create both views in one figure so the scatter plot and ordered-rate
    # comparison can be inspected together.
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(actual_mbps, predicted_mbps, alpha=0.7, edgecolors="none")
    axes[0].plot(limits, limits, "k--", label="Perfect prediction")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Actual data rate (Mbps)")
    axes[0].set_ylabel("Predicted data rate (Mbps)")
    axes[0].set_title("Predicted vs actual")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)

    # Sorting changes only the display order.  It does not change the test
    # samples or the evaluation metrics calculated in train().
    order = np.argsort(actual_mbps)
    sample_numbers = np.arange(1, len(actual_mbps) + 1)
    axes[1].plot(sample_numbers, actual_mbps[order], label="Actual", linewidth=2)
    axes[1].plot(sample_numbers, predicted_mbps[order], label="Predicted", linewidth=2)
    axes[1].set_xlabel("Test samples sorted by actual rate")
    axes[1].set_ylabel("Data rate (Mbps)")
    axes[1].set_title("Test-set rates")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)


def train(data_path: Path, model_path: Path, predictions_path: Path, plot_path: Path) -> None:
    """Train, evaluate, save, and plot the multi-output predictor."""
    features, raw_targets = load_results(data_path)
    # Capacity spans several orders of magnitude, so train on log10(capacity).
    # FSPL and SINR remain in dB and are learned as ordinary regression values.
    targets = raw_targets.copy()
    targets[:, 2] = np.log10(targets[:, 2])

    # Keep one fifth of the rows completely unseen during training.
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    # Put all three output columns on comparable scales, fitting only on the
    # training targets to avoid leaking test-set information.
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    model = build_model()
    model.fit(x_train, y_train_scaled)

    # Reverse target scaling and then reverse the log transform for capacity.
    predicted_targets = target_scaler.inverse_transform(model.predict(x_test))
    actual_targets = y_test.copy()
    predicted_targets[:, 2] = np.power(10.0, predicted_targets[:, 2])
    actual_targets[:, 2] = np.power(10.0, actual_targets[:, 2])
    predicted_fspl, predicted_sinr, predicted_capacity = predicted_targets.T
    actual_fspl, actual_sinr, actual_capacity = actual_targets.T

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target_scaler": target_scaler,
            "feature_names": FEATURE_NAMES,
            "target_names": TARGET_NAMES,
        },
        model_path,
    )

    connection_labels = np.array(
        [f"T{int(tx)} -> R{int(rx)}" for tx, rx in x_test[:, 2:4]]
    )
    with predictions_path.open("w", encoding="utf-8", newline="") as predictions_file:
        predictions_file.write(
            "connection,distance_m,actual_fspl_db,predicted_fspl_db,"
            "fspl_error_db,actual_sinr_db,predicted_sinr_db,sinr_error_db,"
            "actual_capacity_bps,predicted_capacity_bps,capacity_error_bps\n"
        )
        for connection, row, actual_fspl_value, predicted_fspl_value, actual_sinr_value, predicted_sinr_value, actual_rate, predicted_rate in zip(
            connection_labels,
            x_test,
            actual_fspl,
            predicted_fspl,
            actual_sinr,
            predicted_sinr,
            actual_capacity,
            predicted_capacity,
        ):
            predictions_file.write(
                f"{connection},{row[4]:.6f},{actual_fspl_value:.6f},"
                f"{predicted_fspl_value:.6f},{predicted_fspl_value - actual_fspl_value:.6f},"
                f"{actual_sinr_value:.6f},{predicted_sinr_value:.6f},"
                f"{predicted_sinr_value - actual_sinr_value:.6f},"
                f"{actual_rate:.6f},{predicted_rate:.6f},"
                f"{predicted_rate - actual_rate:.6f}\n"
            )

    plot_results(actual_capacity, predicted_capacity, plot_path)

    # Report separate metrics because the outputs use different units.
    capacity_rmse = np.sqrt(mean_squared_error(actual_capacity, predicted_capacity))
    print(f"Rows used: {len(features)} (train={len(x_train)}, test={len(x_test)})")
    print(
        f"FSPL  MAE: {mean_absolute_error(actual_fspl, predicted_fspl):.4f} dB | "
        f"R^2: {r2_score(actual_fspl, predicted_fspl):.4f}"
    )
    print(
        f"SINR  MAE: {mean_absolute_error(actual_sinr, predicted_sinr):.4f} dB | "
        f"R^2: {r2_score(actual_sinr, predicted_sinr):.4f}"
    )
    print(f"Capacity MAE: {mean_absolute_error(actual_capacity, predicted_capacity):,.2f} bps")
    print(f"Capacity RMSE: {capacity_rmse:,.2f} bps")
    print(f"Capacity R^2: {r2_score(actual_capacity, predicted_capacity):.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved test predictions: {predictions_path}")
    print(f"Saved plot: {plot_path}")


def predict(model_path: Path, values: list[float]) -> None:
    """Load a saved model and predict FSPL, SINR, and capacity for one link."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    target_scaler = bundle["target_scaler"]
    if len(values) != len(FEATURE_NAMES):
        names = ", ".join(FEATURE_NAMES)
        raise ValueError(f"Expected {len(FEATURE_NAMES)} values in this order: {names}")
    # The saved pipeline scales inputs. Undo target scaling, then convert the
    # third output from log10(capacity) to bits per second.
    prediction = target_scaler.inverse_transform(model.predict([values]))[0]
    fspl_db, sinr_db, log_capacity = prediction
    rate_bps = float(np.power(10.0, log_capacity))
    print(f"Predicted FSPL: {fspl_db:.4f} dB")
    print(f"Predicted SINR: {sinr_db:.4f} dB")
    print(f"Predicted data rate: {rate_bps:,.2f} bps ({rate_bps / 1e6:,.4f} Mbps)")


def main() -> None:
    """Parse command-line arguments and choose training or prediction mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    # Paths are configurable so a different simulator output or output folder
    # can be used without editing the source code.
    parser.add_argument("--data", type=Path, default=Path("relut.txt"))
    parser.add_argument("--model", type=Path, default=Path("mlp_data_rate.joblib"))
    parser.add_argument("--predictions", type=Path, default=Path("mlp_predictions.csv"))
    parser.add_argument("--plot", type=Path, default=Path("mlp_results.png"))
    parser.add_argument(
        "--predict",
        nargs=len(FEATURE_NAMES),
        type=float,
        metavar=("T", "R", "TX", "RX", "DIST"),
        # Five values are required because they correspond one-to-one with
        # FEATURE_NAMES and are passed to the saved model in that exact order.
        help="Predict FSPL, SINR, and rate using t_count r_count tx_idx rx_idx distance",
    )
    args = parser.parse_args()

    # --predict skips training and requires an existing saved model.
    if args.predict is not None:
        predict(args.model, args.predict)
        return

    # Normal execution trains a new model, evaluates it, and writes all three
    # outputs: the serialized model, CSV predictions, and PNG plot.
    data_path = resolve_data_path(args.data)
    train(data_path, args.model, args.predictions, args.plot)


if __name__ == "__main__":
    main()