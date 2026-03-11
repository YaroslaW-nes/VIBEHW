from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
import streamlit as st

from model import Predictor


st.set_page_config(page_title="Income Scoring", page_icon="📊", layout="centered")


LABELS = {
    "Age": "Age",
    "Workclass": "Workclass",
    "fnlwgt": "Final Weight",
    "Education": "Education",
    "Education_Num": "Education Num",
    "Martial_Status": "Marital Status",
    "Occupation": "Occupation",
    "Relationship": "Relationship",
    "Race": "Race",
    "Sex": "Sex",
    "Capital_Gain": "Capital Gain",
    "Capital_Loss": "Capital Loss",
    "Hours_per_week": "Hours per Week",
    "Country": "Country",
}

FIELD_ORDER = [
    "Age",
    "Workclass",
    "fnlwgt",
    "Education",
    "Education_Num",
    "Martial_Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital_Gain",
    "Capital_Loss",
    "Hours_per_week",
    "Country",
]

NUMERIC_RANGES = {
    "Age": (17, 90),
    "fnlwgt": (12285, 1484705),
    "Education_Num": (1, 16),
    "Capital_Gain": (0, 99999),
    "Capital_Loss": (0, 4356),
    "Hours_per_week": (1, 99),
}

TIPS = [
    "Возраст часто повышает вероятность высокого дохода: у более старших людей обычно больше опыта, выше квалификация и уже построенная карьера.",
    "Большее количество лет обучения и более высокий уровень образования обычно связаны с доступом к более высокооплачиваемым позициям.",
    "Высокие `Capital_Gain` и ненулевые инвестиционные доходы часто сигнализируют о более обеспеченном профиле.",
    "Большее число рабочих часов в неделю может повышать шанс высокого дохода, особенно для full-time и managerial ролей.",
    "Профессия, тип занятости и семейный статус тоже влияют на прогноз, потому что модель ищет статистические паттерны в обучающих данных, а не причинные связи.",
]


@st.cache_resource
def load_predictor() -> Predictor:
    model_path = Path(__file__).resolve().parent / "model.joblib"
    return Predictor(str(model_path))


@st.cache_data
def load_random_examples() -> pd.DataFrame | None:
    data_path = Path(__file__).resolve().parent / "adult_train.csv"
    if not data_path.exists():
        return None
    frame = pd.read_csv(data_path)
    return frame[FIELD_ORDER].copy()


def get_category_options(predictor: Predictor, column: str) -> list[str]:
    prefix = f"{column}_"
    values = [
        feature[len(prefix) :]
        for feature in predictor.feature_columns
        if feature.startswith(prefix)
    ]
    if values:
        return sorted(values)
    fill_value = predictor.categorical_fill_values[column]
    return [str(fill_value)]


def get_default_values(
    predictor: Predictor, category_options: dict[str, list[str]]
) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in predictor.numerical_columns:
        values[column] = int(predictor.numerical_fill_values[column])
    for column in predictor.categorical_columns:
        fill_value = predictor.categorical_fill_values[column]
        values[column] = fill_value if fill_value in category_options[column] else category_options[column][0]
    return values


def get_random_values(
    predictor: Predictor, category_options: dict[str, list[str]]
) -> dict[str, object]:
    examples = load_random_examples()
    if examples is not None and not examples.empty:
        row = examples.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
        values = {}
        for column in FIELD_ORDER:
            if column in predictor.integer_columns:
                values[column] = int(row[column])
            else:
                values[column] = row[column]
        return values

    values = get_default_values(predictor, category_options)
    for column, (low, high) in NUMERIC_RANGES.items():
        values[column] = random.randint(low, high)
    for column, options in category_options.items():
        values[column] = random.choice(options)
    return values


def sync_form_state(values: dict[str, object]) -> None:
    for column, value in values.items():
        st.session_state[column] = value


def ensure_form_state(
    predictor: Predictor, category_options: dict[str, list[str]]
) -> None:
    defaults = get_default_values(predictor, category_options)
    for column, value in defaults.items():
        st.session_state.setdefault(column, value)


def build_input_frame() -> pd.DataFrame:
    row = {
        "Age": int(st.session_state["Age"]),
        "Workclass": st.session_state["Workclass"],
        "fnlwgt": int(st.session_state["fnlwgt"]),
        "Education": st.session_state["Education"],
        "Education_Num": int(st.session_state["Education_Num"]),
        "Martial_Status": st.session_state["Martial_Status"],
        "Occupation": st.session_state["Occupation"],
        "Relationship": st.session_state["Relationship"],
        "Race": st.session_state["Race"],
        "Sex": st.session_state["Sex"],
        "Capital_Gain": int(st.session_state["Capital_Gain"]),
        "Capital_Loss": int(st.session_state["Capital_Loss"]),
        "Hours_per_week": int(st.session_state["Hours_per_week"]),
        "Country": st.session_state["Country"],
    }
    return pd.DataFrame([row])


def render_result(prediction: int, probability: float | None) -> None:
    income_label = ">50K" if prediction == 1 else "<=50K"
    st.success(f"Результат скоринга: {income_label}")
    if probability is not None:
        st.metric("Вероятность класса >50K", f"{probability:.2%}")


def render_tips() -> None:
    with st.expander("Подсказки по признакам", expanded=True):
        st.write(
            "Эти подсказки объясняют общую логику модели на обучающих данных. "
            "Они не гарантируют результат для конкретного человека."
        )
        for tip in TIPS:
            st.write(f"- {tip}")


def main() -> None:
    st.title("Скоринг дохода")
    st.caption("Форма использует модель из `model.joblib` через класс `Predictor`.")

    try:
        predictor = load_predictor()
    except Exception as exc:
        st.error(f"Не удалось загрузить модель: {exc}")
        st.stop()

    category_options = {
        column: get_category_options(predictor, column)
        for column in predictor.categorical_columns
    }
    ensure_form_state(predictor, category_options)

    actions_col, _ = st.columns([1, 2])
    with actions_col:
        if st.button("Сгенерировать случайные данные", use_container_width=True):
            sync_form_state(get_random_values(predictor, category_options))

    render_tips()

    with st.form("scoring_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.number_input(LABELS["Age"], min_value=0, key="Age")
            st.selectbox(LABELS["Workclass"], category_options["Workclass"], key="Workclass")
            st.number_input(LABELS["fnlwgt"], min_value=0, key="fnlwgt")
            st.selectbox(LABELS["Education"], category_options["Education"], key="Education")
            st.number_input(LABELS["Education_Num"], min_value=0, key="Education_Num")
            st.selectbox(
                LABELS["Martial_Status"],
                category_options["Martial_Status"],
                key="Martial_Status",
            )
            st.selectbox(LABELS["Occupation"], category_options["Occupation"], key="Occupation")

        with col_right:
            st.selectbox(
                LABELS["Relationship"],
                category_options["Relationship"],
                key="Relationship",
            )
            st.selectbox(LABELS["Race"], category_options["Race"], key="Race")
            st.selectbox(LABELS["Sex"], category_options["Sex"], key="Sex")
            st.number_input(LABELS["Capital_Gain"], min_value=0, key="Capital_Gain")
            st.number_input(LABELS["Capital_Loss"], min_value=0, key="Capital_Loss")
            st.number_input(
                LABELS["Hours_per_week"],
                min_value=0,
                key="Hours_per_week",
            )
            st.selectbox(LABELS["Country"], category_options["Country"], key="Country")

        submitted = st.form_submit_button("Рассчитать")

    if not submitted:
        return

    try:
        frame = build_input_frame()
        prediction = int(predictor.predict(frame)[0])

        probability = None
        if hasattr(predictor.model, "predict_proba"):
            probability = float(predictor.predict_proba(frame)[0][1])

        render_result(prediction, probability)
        with st.expander("Входные данные"):
            st.dataframe(frame, use_container_width=True)
    except Exception as exc:
        st.error("Не удалось выполнить расчёт.")
        st.exception(exc)


if __name__ == "__main__":
    main()
