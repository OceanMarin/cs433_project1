"""Run six fixed baselines and copy the highest-validation-F1 CSV to submission.csv."""

import json
import shutil
from pathlib import Path

import numpy as np
import implementations as impl

ROOT = Path(__file__).resolve().parent
METHODS = [
    "mean_squared_error_gd", "mean_squared_error_sgd", "least_squares",
    "ridge_regression", "logistic_regression", "reg_logistic_regression",
]


def fit(method, x, y):
    initial_w = np.zeros(x.shape[1])
    np.random.seed(433)
    if method == "least_squares":
        return impl.least_squares(y, x)[0]
    if method == "ridge_regression":
        return impl.ridge_regression(y, x, 0.001)[0]
    if method == "mean_squared_error_sgd":
        return impl.mean_squared_error_sgd(y, x, initial_w, 2 * len(y), 0.00005)[0]
    if method == "mean_squared_error_gd":
        return impl.mean_squared_error_gd(y, x, initial_w, 500, 0.01)[0]
    labels = (y == 1).astype(float)
    if method == "logistic_regression":
        return impl.logistic_regression(labels, x, initial_w, 500, 0.1)[0]
    return impl.reg_logistic_regression(labels, x, 0.001, initial_w, 500, 0.1)[0]


def main():
    with np.load(ROOT / "data/dataset/preprocessed.npz", allow_pickle=False) as data:
        x, x_test = data["x_train"], data["x_test"]
        y, test_ids = data["y_train"], data["test_ids"]
    x = np.column_stack((np.ones(len(x)), x))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))

    indices = np.random.default_rng(433).permutation(len(y))
    split = int(0.8 * len(y))
    train, validation = indices[:split], indices[split:]
    train_x, val_x = x[train], x[validation]
    result_dir = ROOT / "result"
    result_dir.mkdir(exist_ok=True)
    scores = {}

    for method in METHODS:
        print(f"Running {method}...", flush=True)
        weights = fit(method, train_x, y[train])
        # A zero logit is also the default 0.5 probability threshold for Logistic.
        pred = np.where(val_x @ weights >= 0, 1, -1)
        tp = np.sum((pred == 1) & (y[validation] == 1))
        f1 = float(2 * tp / max(np.sum(pred == 1) + np.sum(y[validation] == 1), 1))
        scores[method] = {"validation_f1": f1, "validation_accuracy": float(np.mean(pred == y[validation]))}
        print(f"  validation F1={f1:.6f}; refitting on all rows", flush=True)

        weights = fit(method, x, y)
        if not np.isfinite(weights).all():
            raise ValueError(f"Nonfinite weights from {method}")
        predictions = np.where(x_test @ weights >= 0, 1, -1)
        np.savetxt(
            result_dir / f"predictions_{method}.csv",
            np.column_stack((test_ids, predictions)),
            fmt="%d", delimiter=",", header="Id,Prediction", comments="",
        )

    best = max(scores, key=lambda method: scores[method]["validation_f1"])
    shutil.copyfile(result_dir / f"predictions_{best}.csv", ROOT / "submission.csv")
    (result_dir / "scores.json").write_text(
        json.dumps({"selected": best, "models": scores}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Selected {best}: F1={scores[best]['validation_f1']:.6f} -> submission.csv")


if __name__ == "__main__":
    main()
