"""
check_forest_label.py
---------------------
Standalone sanity check: verify that sklearn IsolationForest.predict()
agrees with (decision_function < 0) on 1000 random 2D points.

This is NOT wired into the server — it only needs numpy + scikit-learn.
It prints the mismatch count and exits 0 when they agree, 1 otherwise.

Run from ai_engine/ (inside the project venv):

    python tests/check_forest_label.py
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def main() -> int:
    rng = np.random.RandomState(42)

    # Fit on random 2D data: mostly inliers plus a few far outliers so the
    # forest has a real boundary to learn.
    X_fit = rng.randn(500, 2)
    X_fit = np.vstack([X_fit, rng.uniform(-20.0, 20.0, size=(50, 2))])

    forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
    )
    forest.fit(X_fit)

    # 1000 random test points.
    X_test = rng.randn(1000, 2) * 3.0

    mismatches = 0
    for i in range(X_test.shape[0]):
        x = X_test[i:i + 1]
        pred_label = int(forest.predict(x)[0])          # +1 inlier / -1 outlier
        score = float(forest.decision_function(x)[0])
        score_label = -1 if score < 0 else 1            # (decision_function < 0)
        if pred_label != score_label:
            mismatches += 1
            if mismatches <= 5:
                print(
                    f"  mismatch: predict={pred_label}  "
                    f"score={score:.6f}  score_label={score_label}"
                )

    print(f"points={X_test.shape[0]}  mismatch_count={mismatches}")
    if mismatches == 0:
        print("OK: decision_function < 0 matches predict() == -1 on all points")
        return 0
    print("MISMATCH: keep the original predict() path")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
