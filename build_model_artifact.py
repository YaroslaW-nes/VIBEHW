from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


TRAIN_URL = (
    "https://raw.githubusercontent.com/hse-ds/ml-hse-nes/refs/heads/main/"
    "2024/homeworks/homework_4/adult_train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/hse-ds/ml-hse-nes/refs/heads/main/"
    "2024/homeworks/homework_4/adult_test.csv"
)
INTEGER_COLUMNS = [
    "Age",
    "fnlwgt",
    "Education_Num",
    "Capital_Gain",
    "Capital_Loss",
    "Hours_per_week",
]


def _load_csv(path: Path, url: str) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    frame = pd.read_csv(url)
    frame.to_csv(path, index=False)
    return frame


def _prepare_raw_frames(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _load_csv(base_dir / "adult_train.csv", TRAIN_URL)
    test = _load_csv(base_dir / "adult_test.csv", TEST_URL)

    test = test[
        (test["Target"] == " >50K.") | (test["Target"] == " <=50K.")
    ].copy()

    train.loc[train["Target"] == " <=50K", "Target"] = 0
    train.loc[train["Target"] == " >50K", "Target"] = 1
    test.loc[test["Target"] == " <=50K.", "Target"] = 0
    test.loc[test["Target"] == " >50K.", "Target"] = 1
    train["Target"] = train["Target"].astype(int)
    test["Target"] = test["Target"].astype(int)

    for column in INTEGER_COLUMNS:
        test[column] = test[column].astype(int)

    return train, test


def _build_artifact(base_dir: Path) -> dict:
    train_raw, test_raw = _prepare_raw_frames(base_dir)

    categorical_columns = [
        column for column in train_raw.columns if train_raw[column].dtype.name == "object"
    ]
    numerical_columns = [
        column for column in train_raw.columns if train_raw[column].dtype.name != "object"
    ]

    categorical_fill_values = {
        column: train_raw[column].mode()[0] for column in categorical_columns
    }
    numerical_fill_values = {
        column: train_raw[column].median() for column in numerical_columns
    }

    for column, fill_value in categorical_fill_values.items():
        train_raw[column] = train_raw[column].fillna(fill_value)
        test_raw[column] = test_raw[column].fillna(fill_value)

    for column, fill_value in numerical_fill_values.items():
        train_raw[column] = train_raw[column].fillna(fill_value)
        test_raw[column] = test_raw[column].fillna(fill_value)

    train_encoded = pd.concat(
        [
            train_raw[numerical_columns],
            pd.get_dummies(train_raw[categorical_columns]),
        ],
        axis=1,
    )
    test_encoded = pd.concat(
        [
            test_raw[numerical_columns],
            pd.get_dummies(test_raw[categorical_columns]),
        ],
        axis=1,
    )

    missing_test_columns = [
        column for column in train_encoded.columns if column not in test_encoded.columns
    ]
    for column in missing_test_columns:
        test_encoded[column] = 0

    extra_test_columns = [
        column for column in test_encoded.columns if column not in train_encoded.columns
    ]
    if extra_test_columns:
        test_encoded = test_encoded.drop(columns=extra_test_columns)

    train_encoded = train_encoded[train_encoded.columns]
    test_encoded = test_encoded[train_encoded.columns]

    model = DecisionTreeClassifier(max_depth=3, random_state=17)
    x_train = train_encoded.drop(columns=["Target"])
    y_train = train_encoded["Target"].astype(int)
    model.fit(x_train, y_train)

    artifact = {
        "model": model,
        "categorical_columns": categorical_columns,
        "numerical_columns": [column for column in numerical_columns if column != "Target"],
        "feature_columns": list(x_train.columns),
        "categorical_fill_values": {
            column: value
            for column, value in categorical_fill_values.items()
            if column != "Target"
        },
        "numerical_fill_values": {
            column: value
            for column, value in numerical_fill_values.items()
            if column != "Target"
        },
        "integer_columns": INTEGER_COLUMNS,
    }

    notebook_predictions = model.predict(test_encoded.drop(columns=["Target"]))
    artifact["reference_test_predictions"] = notebook_predictions
    return artifact


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    artifact = _build_artifact(base_dir)
    artifact_path = base_dir / "model.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"Saved artifact to {artifact_path}")


if __name__ == "__main__":
    main()
