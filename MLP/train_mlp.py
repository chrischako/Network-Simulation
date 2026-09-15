"""Train and use an MLP regressor to predict network data rates.

The network simulator writes one result row for each transmitter/receiver
pair.  Each row contains the pair identifiers, the distance between the two
nodes, the free-space path loss (FSPL), the signal-to-interference-plus-noise
ratio (SINR), and the resulting Shannon capacity.  This script treats the
capacity as the value to predict and the other numeric values as input
features.

The model predicts ``log10(capacity)`` instead of capacity directly.  The
capacities range from thousands to tens of millions of bits per second, so
the logarithm gives the neural network a better-scaled regression target.
Predictions are converted back to bits per second before they are displayed
or written to disk.
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


# These names and this order define the public input format of the model.  The
# same order must be used by the --predict command when predicting one row.
FEATURE_NAMES = (
    "t_count",
    "r_count",
    "tx_idx",
    "rx_idx",
    "distance",
    "fspl",
    "sinr_db",
)

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
    """Parse simulator text into a feature matrix and a target vector.

    ``results.txt`` stores the transmitter and receiver counts on a separate
    ``Trial`` line, followed by several pair-result lines.  The current role
    counts are therefore remembered while the file is scanned.  For every
    pair-result line, one feature row and one capacity target are appended.

    Args:
        path: Text file produced by ``metrics.save_all_results``.

    Returns:
        A tuple ``(features, capacities)``. ``features`` has one row per
        network link and columns in ``FEATURE_NAMES`` order. ``capacities``
        contains the corresponding data rates in bits per second.

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

        # Convert the captured strings to numbers immediately.  This gives
        # scikit-learn a numeric matrix instead of a list of text values.
        features.append(
            [
                t_count,
                r_count,
                int(result_match.group("tx")),
                int(result_match.group("rx")),
                float(result_match.group("distance")),
                float(result_match.group("fspl")),
                float(result_match.group("sinr")),
            ]
        )
        # Capacity is the supervised-learning target: the value the MLP must
        # learn to estimate from the features above.
        targets.append(float(result_match.group("capacity")))

    if not features:
        raise ValueError(f"No result rows found in {path}")
    # float64 is used for all columns because the distance, FSPL, SINR, and
    # capacity values contain decimals and can span a wide numerical range.
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
    """Build the preprocessing and MLP regression pipeline.

    StandardScaler puts every input feature on a comparable scale.  Without
    scaling, values such as node indexes and distances would have a very
    different influence from SINR and FSPL values.  The MLP then uses two
    fully connected hidden layers with ReLU activation to learn nonlinear
    relationships between the network measurements and data rate.
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
    """Train, evaluate, save, and plot the data-rate predictor.

    The data is split into 80% training rows and 20% held-out test rows.  The
    test rows are never used to fit the model; they provide an estimate of how
    well the trained MLP predicts unseen examples from the same data source.
    """
    features, capacities = load_results(data_path)
    # Log targets make the wide range of rates easier for the MLP to learn.
    log_capacities = np.log10(capacities)
    # random_state makes the train/test membership repeatable.  test_size=0.2
    # reserves one fifth of the rows for the final evaluation.
    x_train, x_test, y_train, y_test = train_test_split(
        features, log_capacities, test_size=0.2, random_state=42
    )

    model = build_model()
    # Pipeline.fit first learns feature scaling from x_train, then trains the
    # MLP on the scaled features and log-capacity targets.
    model.fit(x_train, y_train)
    # The network predicts log10(capacity), so 10** reverses that transform.
    predicted = np.power(10.0, model.predict(x_test))
    actual = np.power(10.0, y_test)

    # Store the complete pipeline, not just the neural network.  Loading this
    # object later automatically applies the same StandardScaler parameters.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, model_path)

    # Keep the link identity beside each prediction.  x_test contains the
    # original, unscaled feature values, so columns 2 and 3 identify the exact
    # transmitter and receiver pair used for that test prediction.
    prediction_rows = np.column_stack(
        (x_test[:, 2], x_test[:, 3], actual, predicted, predicted - actual)
    )
    np.savetxt(
        predictions_path,
        prediction_rows,
        delimiter=",",
        header=(
            "transmitter_idx,receiver_idx,actual_capacity_bps,"
            "predicted_capacity_bps,error_bps"
        ),
        comments="",
    )
    # Produce a reusable image rather than requiring an interactive display.
    plot_results(actual, predicted, plot_path)

    # MAE is the average absolute mistake, RMSE emphasizes large mistakes, and
    # R^2 measures how much variation is explained relative to a mean baseline.
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"Rows used: {len(features)} (train={len(x_train)}, test={len(x_test)})")
    print(f"MAE:  {mean_absolute_error(actual, predicted):,.2f} bps")
    print(f"RMSE: {rmse:,.2f} bps")
    print(f"R^2:  {r2_score(actual, predicted):.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved test predictions: {predictions_path}")
    print(f"Saved plot: {plot_path}")


def predict(model_path: Path, values: list[float]) -> None:
    """Load a saved model and predict one data rate from seven feature values."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    if len(values) != len(FEATURE_NAMES):
        names = ", ".join(FEATURE_NAMES)
        raise ValueError(f"Expected {len(FEATURE_NAMES)} values in this order: {names}")
    # The saved pipeline performs scaling, and its MLP returns log10(capacity).
    # Convert that one prediction back to the user-facing bps unit.
    rate_bps = float(np.power(10.0, model.predict([values])[0]))
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
        metavar=("T", "R", "TX", "RX", "DIST", "FSPL", "SINR"),
        # Seven values are required because they correspond one-to-one with
        # FEATURE_NAMES and are passed to the saved model in that exact order.
        help="Predict one rate using t_count r_count tx_idx rx_idx distance fspl sinr_db",
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