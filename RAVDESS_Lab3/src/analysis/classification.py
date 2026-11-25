"""Actor-grouped cross-validation for neutral vs acted-sad speech."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Predeclared, domain-motivated columns. Never pick features using all labels before CV.
DEFAULT_COLUMNS = (("RMS", "Mean"), ("ZCR", "Mean"),
                   ("Spectral Centroid", "Median"), ("F0", "Median"))


def grouped_classification(neutral: pd.DataFrame, sad: pd.DataFrame,
                           selected_columns=DEFAULT_COLUMNS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate logistic regression, holding out both conditions of one actor.

    Returns the aggregate metrics and one prediction per actor/condition.
    """
    if len(neutral) < 3 or not neutral.index.equals(sad.index):
        raise ValueError("At least three matched actors in identical order are required")
    columns = list(selected_columns)
    X = pd.concat([neutral.loc[:, columns], sad.loc[:, columns]], ignore_index=True)
    y = np.array([0] * len(neutral) + [1] * len(sad))
    groups = np.tile(neutral.index.to_numpy(), 2)
    models = {
        "Logistic regression": make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
        "Standardized logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)),
    }
    metric_rows, prediction_rows = [], []
    folds = LeaveOneGroupOut()
    for name, model in models.items():
        probabilities = np.empty(len(y), dtype=float)
        predicted = np.empty(len(y), dtype=int)
        for train_idx, test_idx in folds.split(X, y, groups):
            fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
            probabilities[test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
            predicted[test_idx] = fitted.predict(X.iloc[test_idx])
        metric_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y, predicted),
            "Precision": precision_score(y, predicted, zero_division=0),
            "Recall": recall_score(y, predicted, zero_division=0),
            "F1": f1_score(y, predicted, zero_division=0),
            "ROC-AUC": roc_auc_score(y, probabilities),
        })
        prediction_rows.extend({"Model": name, "Actor": actor, "True": int(target), "Predicted": int(pred), "P(Sad)": prob}
                               for actor, target, pred, prob in zip(groups, y, predicted, probabilities))
    return pd.DataFrame(metric_rows).set_index("Model"), pd.DataFrame(prediction_rows)
