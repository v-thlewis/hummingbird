"""Benchmark Hummingbird's PyTorch SVC predict() for one checkout of the repository.

Imports `hummingbird` from whatever is first on PYTHONPATH, so `remeasure.py` can point it at
different git refs. Prints one JSON object per configuration.
"""

import argparse
import json
import resource
import sys
import time

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.svm import SVC

import hummingbird
import hummingbird.ml

CONFIGS = {
    # Matches the configuration described when the optimization was proposed upstream.
    "4cls-rbf-32f-b256": dict(n_classes=4, n_features=32, n_train=1000, kernel="rbf", batch=256, iters=320),
    # Matches the agent's original measurement (8 classes, 2000 x 50, rbf).
    "8cls-rbf-50f-b2000": dict(n_classes=8, n_features=50, n_train=2000, kernel="rbf", batch=2000, iters=20),
    "8cls-linear-50f-b2000": dict(n_classes=8, n_features=50, n_train=2000, kernel="linear", batch=2000, iters=20),
}


def run(name, cfg, seed):
    X, y = make_classification(
        n_samples=cfg["n_train"] + cfg["batch"],
        n_features=cfg["n_features"],
        n_informative=min(cfg["n_features"], 10),
        n_classes=cfg["n_classes"],
        n_clusters_per_class=1,
        flip_y=0.1,
        random_state=seed,
    )
    X = X.astype(np.float32)
    X_train, y_train, X_test = X[: cfg["n_train"]], y[: cfg["n_train"]], X[cfg["n_train"] :]
    model = SVC(kernel=cfg["kernel"], random_state=seed).fit(X_train, y_train)
    hb_model = hummingbird.ml.convert(model, "torch")

    expected = model.predict(X_test)
    actual = hb_model.predict(X_test)
    hb_model.predict(X_test)  # warm-up

    wall, cpu = [], []
    for _ in range(cfg["iters"]):
        w0, c0 = time.perf_counter(), time.process_time()
        hb_model.predict(X_test)
        wall.append(time.perf_counter() - w0)
        cpu.append(time.process_time() - c0)

    return {
        "config": name,
        "hummingbird_path": hummingbird.__file__,
        "n_support_vectors": int(model.support_vectors_.shape[0]),
        "predictions_match_sklearn": bool(np.array_equal(expected, actual)),
        "wall_ms_median": float(np.median(wall) * 1000),
        "wall_ms_total": float(np.sum(wall) * 1000),
        "cpu_ms_median": float(np.median(cpu) * 1000),
        "cpu_ms_total": float(np.sum(cpu) * 1000),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="torch intra-op threads (1 reduces noise)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for name in args.config or sorted(CONFIGS):
        print(json.dumps(run(name, CONFIGS[name], args.seed)), flush=True)


if __name__ == "__main__":
    main()
