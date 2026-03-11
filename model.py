from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path: str = "model.joblib") -> None:
        artifact_path = Path(model_path)
        if not artifact_path.is_absolute():
            artifact_path = Path(__file__).resolve().parent / artifact_path

        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.categorical_columns = list(artifact["categorical_columns"])
        self.numerical_columns = list(artifact["numerical_columns"])
        self.feature_columns = list(artifact["feature_columns"])
        self.categorical_fill_values = dict(artifact["categorical_fill_values"])
        self.numerical_fill_values = dict(artifact["numerical_fill_values"])
        self.integer_columns = list(artifact.get("integer_columns", []))
        self.raw_feature_columns = self.numerical_columns + self.categorical_columns

    def predict(self, data: pd.DataFrame) -> Any:
        features = self._prepare_features(data)
        return self.model.predict(features)

    def predict_proba(self, data: pd.DataFrame) -> Any:
        features = self._prepare_features(data)
        return self.model.predict_proba(features)

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        frame = data.copy()
        missing_columns = [
            column for column in self.raw_feature_columns if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(
                "Missing required columns: " + ", ".join(sorted(missing_columns))
            )

        frame = frame[self.raw_feature_columns]

        for column in self.integer_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        for column, fill_value in self.categorical_fill_values.items():
            frame[column] = frame[column].fillna(fill_value)

        for column, fill_value in self.numerical_fill_values.items():
            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = numeric_series.fillna(fill_value)

        for column in self.integer_columns:
            frame[column] = frame[column].astype(int)

        encoded = pd.concat(
            [
                frame[self.numerical_columns],
                pd.get_dummies(frame[self.categorical_columns]),
            ],
            axis=1,
        )

        aligned = encoded.reindex(columns=self.feature_columns, fill_value=0)
        return aligned
