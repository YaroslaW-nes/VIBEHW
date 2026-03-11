from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap


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
        self._explainer = shap.TreeExplainer(self.model)

    def predict(self, data: pd.DataFrame) -> Any:
        features = self._prepare_features(data)
        return self.model.predict(features)

    def predict_proba(self, data: pd.DataFrame) -> Any:
        features = self._prepare_features(data)
        return self.model.predict_proba(features)

    def explain(self, data: pd.DataFrame) -> dict[str, Any]:
        raw_frame = self._prepare_raw_frame(data)
        features = self._encode_frame(raw_frame)
        explanation = self._explainer(features, check_additivity=False)

        shap_values = explanation.values
        base_values = explanation.base_values

        if getattr(shap_values, "ndim", 0) == 3:
            shap_values = shap_values[:, :, 1]
        if getattr(base_values, "ndim", 0) > 1:
            base_values = base_values[:, 1]

        aggregated = []
        sample_shap_values = shap_values[0]
        for raw_column in self.raw_feature_columns:
            feature_names = self._get_feature_names_for_raw_column(raw_column)
            shap_sum = sum(
                float(sample_shap_values[self.feature_columns.index(feature_name)])
                for feature_name in feature_names
            )
            aggregated.append(
                {
                    "feature": raw_column,
                    "value": raw_frame.iloc[0][raw_column],
                    "shap_value": shap_sum,
                    "abs_shap_value": abs(shap_sum),
                }
            )

        aggregated_frame = pd.DataFrame(aggregated).sort_values(
            by="abs_shap_value", ascending=False
        )
        return {
            "base_value": float(base_values[0]),
            "feature_contributions": aggregated_frame.reset_index(drop=True),
        }

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_raw_frame(data)
        return self._encode_frame(frame)

    def _prepare_raw_frame(self, data: pd.DataFrame) -> pd.DataFrame:
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

        return frame

    def _encode_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        encoded = pd.concat(
            [
                frame[self.numerical_columns],
                pd.get_dummies(frame[self.categorical_columns]),
            ],
            axis=1,
        )

        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    def _get_feature_names_for_raw_column(self, raw_column: str) -> list[str]:
        if raw_column in self.numerical_columns:
            return [raw_column]

        prefix = f"{raw_column}_"
        return [
            feature_name
            for feature_name in self.feature_columns
            if feature_name.startswith(prefix)
        ]
