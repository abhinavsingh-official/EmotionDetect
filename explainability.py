"""What does the best classical model actually look at?

The original plan assumed the best model would turn out to be a linear SVM,
where "largest weight magnitude" is a one-line answer. In practice the best
model here (see classifier.py) is an RBF-kernel SVM - a kernel trick means
there's no single weight vector to read coefficients off, so raw weight
magnitude doesn't apply. Permutation importance works regardless of the
model's internals: shuffle one feature across all samples, see how much
accuracy drops, repeat per feature. A feature the model actually relies on
will hurt a lot when scrambled; one it ignores won't move the needle. Not a
full SHAP/LIME pass - a cheap, honest, real signal, same spirit as the
original plan.

Each of the 180 features is one of three audio feature families
(AudioEmo.py concatenates them in this order): 128 Mel-spectrogram bands,
40 MFCCs, then 12 chroma bins. Reporting importance by family, not just by
raw column index, is what actually means something to a reader."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classifier import MODELS, RANDOM_STATE, load_features

N_MEL, N_MFCC, N_CHROMA = 128, 40, 12


def feature_family(index: int) -> str:
    if index < N_MEL:
        return f"mel[{index}]"
    if index < N_MEL + N_MFCC:
        return f"mfcc[{index - N_MEL}]"
    return f"chroma[{index - N_MEL - N_MFCC}]"


def top_features(model_name: str = "svm", top_n: int = 15, n_repeats: int = 10) -> list[dict]:
    X, y = load_features()
    # A held-out split (not CV) is enough here - this is a diagnostic pass
    # over the trained model's behavior, not a claim about its accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", MODELS[model_name])])
    pipeline.fit(X_train, y_train)

    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=n_repeats, random_state=RANDOM_STATE, scoring="accuracy"
    )

    ranked_idx = np.argsort(result.importances_mean)[::-1][:top_n]
    return [
        {
            "feature": feature_family(i),
            "importance_mean": float(result.importances_mean[i]),
            "importance_std": float(result.importances_std[i]),
        }
        for i in ranked_idx
    ]


def family_summary(ranked: list[dict]) -> dict[str, int]:
    summary = {"mel": 0, "mfcc": 0, "chroma": 0}
    for row in ranked:
        family = row["feature"].split("[")[0]
        summary[family] += 1
    return summary


if __name__ == "__main__":
    # Reads back whichever model classifier.py most recently found to be
    # best, rather than assuming it's always the SVM - the ranking between
    # SVM/MLP/RF is close enough that it can flip depending on exact model
    # settings, and this should explain whatever's actually been deployed.
    results_path = Path("outputs/classifier_results.json")
    best_model = json.loads(results_path.read_text())["best_model"] if results_path.exists() else "svm"

    ranked = top_features(model_name=best_model)

    print(f"Top 15 most important features for the {best_model} model (by permutation importance):")
    for row in ranked:
        print(f"  {row['feature']:12s} importance={row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")

    summary = family_summary(ranked)
    print(f"\nOf the top 15, {summary['mfcc']} are MFCCs, {summary['mel']} are Mel bands, "
          f"{summary['chroma']} are chroma bins.")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/explainability_results.json", "w") as f:
        json.dump({"model": best_model, "top_features": ranked, "family_summary": summary}, f, indent=2)
    print("\nSaved to outputs/explainability_results.json")
