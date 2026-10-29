"""Run six fixed baselines and copy the highest-validation-F1 CSV to submission.csv."""

import json
import shutil
from pathlib import Path

import numpy as np
import implementations as impl

ROOT = Path(__file__).resolve().parent
K_FOLD, SEED = 5, 1         
LOGISTIC_THRESHOLD = 0.2
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
        x, x_test = data["x_train"].astype(float), data["x_test"].astype(float)
        y, test_ids = data["y_train"], data["test_ids"]
    x = np.column_stack((np.ones(len(x)), x))
    x_test = np.column_stack((np.ones(len(x_test)), x_test))

    
    result_dir = ROOT / "result"
    result_dir.mkdir(exist_ok=True)
    scores = {}

    for method in METHODS:
        threshold = np.log(LOGISTIC_THRESHOLD / (1 - LOGISTIC_THRESHOLD)) if "logistic" in method else 0.0
        fit_fn = lambda yy, xx, m=method: fit(m, xx, yy)
        tr_f1, te_f1 = impl.cross_validation_score(y, x, fit_fn, k_fold=K_FOLD, seed=SEED, threshold=threshold)
        scores[method] = {"cv_train_f1": tr_f1, "cv_test_f1": te_f1, "score_threshold": float(threshold)}
        print(f"{method:24s} CV train F1 {tr_f1:.4f}  test F1 {te_f1:.4f}", flush=True)

        weights = fit(method, x, y)
        if not np.isfinite(weights).all():
            raise ValueError(f"Nonfinite weights from {method}")
        predictions = np.where(x_test @ weights >= threshold, 1, -1)
        np.savetxt(
            result_dir / f"predictions_{method}.csv",
            np.column_stack((test_ids, predictions)),
            fmt="%d", delimiter=",", header="Id,Prediction", comments="",
        )

    best = max(scores, key=lambda m: scores[m]["cv_test_f1"])
    shutil.copyfile(result_dir / f"predictions_{best}.csv", ROOT / "submission.csv")
    (result_dir / "scores.json").write_text(
        json.dumps({"selected": best, "models": scores}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Selected {best}: CV test F1={scores[best]['cv_test_f1']:.4f} -> submission.csv")


if __name__ == "__main__":
    main()
