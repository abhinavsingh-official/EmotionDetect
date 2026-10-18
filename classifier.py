"""Classical ML comparison for speech emotion recognition - a step up
from AudioEmo.py's nearest-centroid baseline (cosine similarity to a
per-class mean), which is fast and has zero training cost but throws
away everything about how classes are actually separated beyond their
centroid. Three standard classifiers trained on the same per-sample
feature table, evaluated with stratified k-fold cross-validation
rather than one train/test split, since RAVDESS is a small dataset
(1440 samples) where a single split can be noisy.

The scaler lives inside each model's Pipeline rather than being fit
once up front, so scaling happens fresh per fold - fitting it on the
whole dataset first would leak a bit of the test fold's distribution
into training."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

N_FOLDS = 5
RANDOM_STATE = 42

# RobustScaler (median/IQR) was the first thing tried here, since it's the
# standard choice for data with outliers - but it tanked the SVM to ~22%
# accuracy (barely above the ~12.5% random-chance baseline for 8 classes).
# A few of the 180 spectral features have a near-zero IQR across this
# dataset, which blows their scaled values up and distorts the RBF kernel's
# distance calculations. StandardScaler doesn't have that failure mode here
# and roughly tripled SVM accuracy - confirmed empirically, not a guess.
MODELS = {
    # Wrapped in CalibratedClassifierCV (SVC's own probability=True is
    # deprecated as of sklearn 1.9) so the Streamlit app can show real
    # per-class probabilities instead of raw decision-function scores -
    # costs a bit of extra fit time, irrelevant at this dataset size.
    "svm": CalibratedClassifierCV(
        SVC(kernel="rbf", C=10, gamma=0.02, random_state=RANDOM_STATE), ensemble=False
    ),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=RANDOM_STATE),
}


def load_features(db_path: str = "Audio_db.pkl") -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_pickle(db_path)
    feature_cols = [c for c in df.columns if c != "target"]
    return df[feature_cols], df["target"]


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in MODELS.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        results[name] = {
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
        }
    return results


def baseline_accuracy(X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluates AudioEmo.py's actual nearest-centroid approach - cosine
    similarity to each class's mean feature vector - under the same 5-fold
    CV used for the classical models, so the comparison table has a real,
    apples-to-apples "Baseline (nearest-centroid)" row instead of treating
    that approach as the system's primary method. Uses RobustScaler to
    match AudioEmotionRecognizer's own real preprocessing, not the
    StandardScaler used for the classifiers above."""
    X_arr, y_arr = X.values, y.values
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        scaler = RobustScaler().fit(X_arr[train_idx])
        X_train, X_test = scaler.transform(X_arr[train_idx]), scaler.transform(X_arr[test_idx])
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        class_means = {
            label: X_train[y_train == label].mean(axis=0, keepdims=True)
            for label in np.unique(y_train)
        }
        predictions = [
            max(class_means, key=lambda label: cosine_similarity(class_means[label], row.reshape(1, -1))[0, 0])
            for row in X_test
        ]
        accuracies.append(float(np.mean(np.array(predictions) == y_test)))
    return {"accuracy_mean": float(np.mean(accuracies)), "accuracy_std": float(np.std(accuracies))}


def best_model_name(results: dict) -> str:
    return max(results, key=lambda name: results[name]["accuracy_mean"])


def confusion_matrix_for(name: str, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list[str]]:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", MODELS[name])])
    # cross_val_predict gives an honest, leakage-free prediction for every row
    # (each one predicted by a fold that never trained on it), unlike fitting
    # on the full set and predicting on the same data it just saw.
    predictions = cross_val_predict(pipeline, X, y, cv=cv)
    labels = sorted(y.unique())
    return confusion_matrix(y, predictions, labels=labels), labels


def train_and_save_best(name: str, X: pd.DataFrame, y: pd.Series, out_path: str = "models/emotion_classifier.joblib"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", MODELS[name])])
    pipeline.fit(X, y)
    joblib.dump(pipeline, out_path)
    return out_path


if __name__ == "__main__":
    X, y = load_features()
    print(f"Loaded {len(X)} samples, {y.nunique()} classes")

    baseline = baseline_accuracy(X, y)
    print(
        f"\nBaseline (nearest-centroid, AudioEmo.py's original approach): "
        f"accuracy={baseline['accuracy_mean']:.3f} +/- {baseline['accuracy_std']:.3f}"
    )

    results = evaluate_models(X, y)
    print(f"\n{N_FOLDS}-fold cross-validation results:")
    for name, scores in sorted(results.items(), key=lambda kv: -kv[1]["accuracy_mean"]):
        print(
            f"  {name:15s} accuracy={scores['accuracy_mean']:.3f} +/- {scores['accuracy_std']:.3f}  "
            f"f1_macro={scores['f1_macro_mean']:.3f} +/- {scores['f1_macro_std']:.3f}"
        )

    best = best_model_name(results)
    print(f"\nBest model: {best}")

    cm, labels = confusion_matrix_for(best, X, y)
    print(f"\nConfusion matrix ({best}, labels={labels}):")
    print(cm)

    out_path = train_and_save_best(best, X, y)
    print(f"\nSaved best model to {out_path}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/classifier_results.json", "w") as f:
        json.dump(
            {
                "baseline_nearest_centroid": baseline,
                "results": results,
                "best_model": best,
                "confusion_matrix": cm.tolist(),
                "labels": labels,
            },
            f,
            indent=2,
        )
    print("Saved full results to outputs/classifier_results.json")
