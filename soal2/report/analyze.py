"""Reproduce the SETAR calculations and figures for the report.

Only condition-number diagnostics use NumPy's linear algebra routines. Both
least-squares solutions are computed by the algorithms implemented below.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def design(returns: np.ndarray, start: int) -> tuple[np.ndarray, np.ndarray]:
    """One row per return from start onward; previous two returns are known."""
    a = np.zeros((len(returns) - start, 6), dtype=float)
    b = returns[start:].copy()
    for row, t in enumerate(range(start, len(returns))):
        lag1, lag2 = returns[t - 1], returns[t - 2]
        if lag1 >= 0:
            a[row, :3] = (1, lag1, lag2)
        else:
            a[row, 3:] = (1, lag1, lag2)
    return a, b


def gaussian_pivot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a square linear system by partial-pivot Gaussian elimination."""
    u = a.astype(float).copy()
    y = b.astype(float).copy()
    n = len(y)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(u[k:, k])))
        if u[pivot, k] == 0:
            raise ValueError("Singular system")
        if pivot != k:
            u[[k, pivot]] = u[[pivot, k]]
            y[[k, pivot]] = y[[pivot, k]]
        for i in range(k + 1, n):
            factor = u[i, k] / u[k, k]
            u[i, k:] -= factor * u[k, k:]
            y[i] -= factor * y[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(u[i, i + 1 :], x[i + 1 :])) / u[i, i]
    return x


def givens_least_squares(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply each Givens rotation to two rows of [A | b], then back-solve."""
    r = a.astype(float).copy()
    d = b.astype(float).copy()
    m, n = r.shape
    for i in range(n):
        for j in range(i + 1, m):
            if r[j, i] == 0:
                continue
            alpha = np.hypot(r[i, i], r[j, i])
            c, s = r[i, i] / alpha, r[j, i] / alpha
            row_i = r[i, i:].copy()
            row_j = r[j, i:].copy()
            r[i, i:] = c * row_i + s * row_j
            r[j, i:] = -s * row_i + c * row_j
            di, dj = d[i], d[j]
            d[i], d[j] = c * di + s * dj, -s * di + c * dj
            r[j, i] = 0.0
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (d[i] - np.dot(r[i, i + 1 :n], x[i + 1 :])) / r[i, i]
    return x


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error**2)))


def main() -> None:
    train = pd.read_csv(DATA / "stock_train.csv")
    test = pd.read_csv(DATA / "stock_test.csv")
    all_close = np.r_[train.Close.to_numpy(), test.Close.to_numpy()]
    all_returns = np.r_[np.nan, np.diff(all_close) / all_close[:-1]]
    split = len(train)
    train_a, train_b = design(all_returns[:split], 3)
    test_a, test_b = design(all_returns, split)

    normal_x = gaussian_pivot(train_a.T @ train_a, train_a.T @ train_b)
    givens_x = givens_least_squares(train_a, train_b)
    train_pred = train_a @ givens_x
    test_pred = test_a @ givens_x
    train_error = train_pred - train_b
    test_error = test_pred - test_b
    metrics = {
        "prices_train": len(train),
        "prices_test": len(test),
        "returns_train": len(train) - 1,
        "observations_train": len(train_b),
        "observations_test": len(test_b),
        "bull_train": int(np.count_nonzero(train_a[:, 0])),
        "bear_train": int(np.count_nonzero(train_a[:, 3])),
        "bull_test": int(np.count_nonzero(test_a[:, 0])),
        "bear_test": int(np.count_nonzero(test_a[:, 3])),
        "condition_a": float(np.linalg.cond(train_a)),
        "condition_ata": float(np.linalg.cond(train_a.T @ train_a)),
        "normal_coefficients": normal_x.tolist(),
        "givens_coefficients": givens_x.tolist(),
        "coefficient_difference_norm": float(np.linalg.norm(normal_x - givens_x)),
        "normal_residual": float(np.linalg.norm(train_a @ normal_x - train_b)),
        "givens_residual": float(np.linalg.norm(train_error)),
        "normal_train_rmse": rmse(train_a @ normal_x - train_b),
        "givens_train_rmse": rmse(train_error),
        "normal_test_rmse": rmse(test_a @ normal_x - test_b),
        "givens_test_rmse": rmse(test_error),
        "max_abs_train_return": float(np.max(np.abs(train_b))),
        "max_abs_train_error": float(np.max(np.abs(train_error))),
        "max_abs_test_error": float(np.max(np.abs(test_error))),
        "max_train_error_index": int(np.argmax(np.abs(train_error))),
        "max_train_error_date": str(train.Date.iloc[3 + np.argmax(np.abs(train_error))]),
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.5, 4))
    x_train = np.arange(len(train_b))
    x_test = np.arange(len(train_b), len(train_b) + len(test_b))
    ax.plot(x_train, train_b, color="#374151", linewidth=0.9, label="Aktual")
    ax.plot(x_test, test_b, color="#374151", linewidth=0.9)
    ax.plot(x_train, train_pred, color="#2563eb", linewidth=1.0, label="Prediksi SETAR")
    ax.plot(x_test, test_pred, color="#2563eb", linewidth=1.0)
    ax.axvline(len(train_b) - 0.5, color="#dc2626", linestyle="--", linewidth=1.2, label="Batas latih/uji")
    ax.set(xlabel="Indeks observasi", ylabel="Return harian", title="Return aktual dan prediksi SETAR")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "overlay.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.plot(x_train, train_error, linewidth=0.8, color="#2563eb", label="Latih")
    ax.plot(x_test, test_error, linewidth=0.8, color="#d97706", label="Uji")
    ax.axvline(len(train_b) - 0.5, color="#dc2626", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set(xlabel="Indeks observasi", ylabel="Prediksi - aktual", title="Residual per observasi")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "residuals.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
