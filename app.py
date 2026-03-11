from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
import streamlit as st

from model import Predictor


st.set_page_config(page_title="Income Scoring", page_icon="📊", layout="wide")


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

FIELD_HELP = {
    "Age": "Older candidates often have more work experience and more stable career trajectories.",
    "Education_Num": "More years of formal education often correlate with higher-paying jobs in the training data.",
    "Capital_Gain": "Investment income is often associated with higher-income profiles.",
    "Hours_per_week": "A full-time or overtime workload can increase the probability of higher income.",
    "Workclass": "Type of employer matters because income distributions differ across employment sectors.",
    "Occupation": "Occupation is one of the strongest drivers of income differences in the Adult dataset.",
}

TIPS = [
    "Возраст часто повышает вероятность высокого дохода: у более старших людей обычно больше опыта, выше квалификация и уже построенная карьера.",
    "Большее количество лет обучения и более высокий уровень образования обычно связаны с доступом к более высокооплачиваемым позициям.",
    "Высокие Capital Gain и ненулевые инвестиционные доходы часто сигнализируют о более обеспеченном профиле.",
    "Большее число рабочих часов в неделю может повышать шанс высокого дохода, особенно для full-time и managerial ролей.",
    "Профессия, тип занятости и семейный статус тоже влияют на прогноз, потому что модель ищет статистические паттерны в обучающих данных, а не причинные связи.",
]

PRESETS = {
    "Typical low income": {
        "Age": 24,
        "Workclass": " Private",
        "fnlwgt": 180000,
        "Education": " HS-grad",
        "Education_Num": 9,
        "Martial_Status": " Never-married",
        "Occupation": " Other-service",
        "Relationship": " Not-in-family",
        "Race": " White",
        "Sex": " Female",
        "Capital_Gain": 0,
        "Capital_Loss": 0,
        "Hours_per_week": 35,
        "Country": " United-States",
    },
    "Typical high income": {
        "Age": 47,
        "Workclass": " Private",
        "fnlwgt": 210000,
        "Education": " Bachelors",
        "Education_Num": 13,
        "Martial_Status": " Married-civ-spouse",
        "Occupation": " Exec-managerial",
        "Relationship": " Husband",
        "Race": " White",
        "Sex": " Male",
        "Capital_Gain": 7688,
        "Capital_Loss": 0,
        "Hours_per_week": 50,
        "Country": " United-States",
    },
}


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


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(215, 232, 255, 0.9), transparent 28%),
                linear-gradient(180deg, #f4f7fb 0%, #eef3f8 100%);
        }
        .hero-card, .result-card, .section-card, .sidebar-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(31, 41, 55, 0.08);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 1rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #132238;
            margin-bottom: 0.6rem;
        }
        .kpi-label {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.15rem;
        }
        .kpi-value {
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 700;
        }
        .positive-note {
            color: #0f766e;
            font-weight: 600;
        }
        .negative-note {
            color: #b91c1c;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        values[column] = (
            fill_value if fill_value in category_options[column] else category_options[column][0]
        )
    return values


def sanitize_values(
    values: dict[str, object], predictor: Predictor, category_options: dict[str, list[str]]
) -> dict[str, object]:
    sanitized = {}
    defaults = get_default_values(predictor, category_options)
    for column in FIELD_ORDER:
        value = values.get(column, defaults[column])
        if column in predictor.integer_columns:
            low, high = NUMERIC_RANGES[column]
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = defaults[column]
            value = max(low, min(high, value))
        else:
            if value not in category_options[column]:
                value = defaults[column]
        sanitized[column] = value
    return sanitized


def get_random_values(
    predictor: Predictor, category_options: dict[str, list[str]]
) -> dict[str, object]:
    examples = load_random_examples()
    if examples is not None and not examples.empty:
        row = examples.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
        return sanitize_values(row.to_dict(), predictor, category_options)

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
    row = {column: st.session_state[column] for column in FIELD_ORDER}
    for column in NUMERIC_RANGES:
        row[column] = int(row[column])
    return pd.DataFrame([row])


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Income Scoring Dashboard</div>
            <div class="hero-subtitle">
                Введите профиль человека, получите прогноз дохода и локальную интерпретацию через SHAP values.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_toolbar(
    predictor: Predictor, category_options: dict[str, list[str]]
) -> None:
    action_col, preset_low_col, preset_high_col, reset_col = st.columns([1, 1, 1, 1])

    with action_col:
        if st.button("Случайный профиль", use_container_width=True):
            sync_form_state(get_random_values(predictor, category_options))
            st.rerun()

    with preset_low_col:
        if st.button("Typical low income", use_container_width=True):
            sync_form_state(
                sanitize_values(PRESETS["Typical low income"], predictor, category_options)
            )
            st.rerun()

    with preset_high_col:
        if st.button("Typical high income", use_container_width=True):
            sync_form_state(
                sanitize_values(PRESETS["Typical high income"], predictor, category_options)
            )
            st.rerun()

    with reset_col:
        if st.button("Сбросить форму", use_container_width=True):
            sync_form_state(get_default_values(predictor, category_options))
            st.rerun()


def render_context_panels() -> None:
    preview_col, tips_col = st.columns([1, 1.4])

    with preview_col:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-title">Live Preview</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        preview = {
            "Age": st.session_state["Age"],
            "Education": st.session_state["Education"],
            "Occupation": st.session_state["Occupation"],
            "Workclass": st.session_state["Workclass"],
            "Hours/week": st.session_state["Hours_per_week"],
        }
        st.json(preview)

    with tips_col:
        with st.expander("Подсказки по признакам", expanded=True):
            for tip in TIPS:
                st.write(f"- {tip}")


def render_section_header(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_result(prediction: int, probability: float | None) -> None:
    income_label = ">50K" if prediction == 1 else "<=50K"
    tone = "positive-note" if prediction == 1 else "negative-note"
    probability_text = "n/a" if probability is None else f"{probability:.2%}"
    st.markdown(
        f"""
        <div class="result-card">
            <div class="kpi-label">Model Verdict</div>
            <div class="kpi-value">{income_label}</div>
            <div class="{tone}">
                Вероятность класса &gt;50K: {probability_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_summary(contributions: pd.DataFrame) -> None:
    positive = contributions.sort_values("shap_value", ascending=False).head(3)
    negative = contributions.sort_values("shap_value", ascending=True).head(3)

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("#### Что повысило шанс >50K")
        for _, row in positive.iterrows():
            if row["shap_value"] <= 0:
                continue
            st.write(
                f"- **{row['feature']}** = `{row['value']}` ({row['shap_value']:+.4f})"
            )

    with neg_col:
        st.markdown("#### Что понизило шанс >50K")
        for _, row in negative.iterrows():
            if row["shap_value"] >= 0:
                continue
            st.write(
                f"- **{row['feature']}** = `{row['value']}` ({row['shap_value']:+.4f})"
            )


def render_shap(explanation: dict[str, object]) -> None:
    contributions = explanation["feature_contributions"].copy()
    chart_data = contributions[["feature", "shap_value"]].set_index("feature")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">SHAP Explanation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_col, _ = st.columns([1, 2])
    with metric_col:
        st.metric("Базовое значение модели", f"{float(explanation['base_value']):.4f}")
    st.caption(
        "Положительное значение повышает вероятность класса >50K, отрицательное снижает её."
    )
    render_feature_summary(contributions)
    st.bar_chart(chart_data, use_container_width=True)

    with st.expander("Полная таблица вкладов"):
        st.dataframe(contributions, use_container_width=True)


def render_form(predictor: Predictor, category_options: dict[str, list[str]]) -> bool:
    with st.form("scoring_form"):
        profile_col, job_col, finance_col = st.columns(3)

        with profile_col:
            render_section_header("Профиль")
            st.number_input(
                LABELS["Age"],
                min_value=NUMERIC_RANGES["Age"][0],
                max_value=NUMERIC_RANGES["Age"][1],
                key="Age",
                help=FIELD_HELP["Age"],
            )
            st.selectbox(
                LABELS["Sex"],
                category_options["Sex"],
                key="Sex",
            )
            st.selectbox(
                LABELS["Race"],
                category_options["Race"],
                key="Race",
            )
            st.selectbox(
                LABELS["Relationship"],
                category_options["Relationship"],
                key="Relationship",
            )
            st.selectbox(
                LABELS["Country"],
                category_options["Country"],
                key="Country",
            )

        with job_col:
            render_section_header("Работа и образование")
            st.selectbox(
                LABELS["Workclass"],
                category_options["Workclass"],
                key="Workclass",
                help=FIELD_HELP["Workclass"],
            )
            st.selectbox(
                LABELS["Occupation"],
                category_options["Occupation"],
                key="Occupation",
                help=FIELD_HELP["Occupation"],
            )
            st.selectbox(
                LABELS["Education"],
                category_options["Education"],
                key="Education",
            )
            st.number_input(
                LABELS["Education_Num"],
                min_value=NUMERIC_RANGES["Education_Num"][0],
                max_value=NUMERIC_RANGES["Education_Num"][1],
                key="Education_Num",
                help=FIELD_HELP["Education_Num"],
            )
            st.selectbox(
                LABELS["Martial_Status"],
                category_options["Martial_Status"],
                key="Martial_Status",
            )
            st.number_input(
                LABELS["Hours_per_week"],
                min_value=NUMERIC_RANGES["Hours_per_week"][0],
                max_value=NUMERIC_RANGES["Hours_per_week"][1],
                key="Hours_per_week",
                help=FIELD_HELP["Hours_per_week"],
            )

        with finance_col:
            render_section_header("Финансы")
            st.number_input(
                LABELS["fnlwgt"],
                min_value=NUMERIC_RANGES["fnlwgt"][0],
                max_value=NUMERIC_RANGES["fnlwgt"][1],
                key="fnlwgt",
            )
            st.number_input(
                LABELS["Capital_Gain"],
                min_value=NUMERIC_RANGES["Capital_Gain"][0],
                max_value=NUMERIC_RANGES["Capital_Gain"][1],
                key="Capital_Gain",
                help=FIELD_HELP["Capital_Gain"],
            )
            st.number_input(
                LABELS["Capital_Loss"],
                min_value=NUMERIC_RANGES["Capital_Loss"][0],
                max_value=NUMERIC_RANGES["Capital_Loss"][1],
                key="Capital_Loss",
            )

            st.markdown(
                """
                <div class="section-card" style="margin-top: 1rem;">
                    <div class="section-title">Как читать результат</div>
                    <div>
                        Модель оценивает вероятность класса <strong>&gt;50K</strong>.
                        После расчёта ниже появятся verdict, вероятность и SHAP-вклады признаков.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        submit_col, _ = st.columns([1, 3])
        with submit_col:
            submitted = st.form_submit_button("Рассчитать", use_container_width=True)
    return submitted


def main() -> None:
    apply_styles()
    render_hero()

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
    render_toolbar(predictor, category_options)
    render_context_panels()

    submitted = render_form(predictor, category_options)
    if not submitted:
        return

    try:
        frame = build_input_frame()
        prediction = int(predictor.predict(frame)[0])
        explanation = predictor.explain(frame)

        probability = None
        if hasattr(predictor.model, "predict_proba"):
            probability = float(predictor.predict_proba(frame)[0][1])

        result_col, detail_col = st.columns([1.1, 1.9])
        with result_col:
            render_result(prediction, probability)
            with st.expander("Входные данные"):
                st.dataframe(frame, use_container_width=True)
        with detail_col:
            render_shap(explanation)
    except Exception as exc:
        st.error("Не удалось выполнить расчёт.")
        st.exception(exc)


if __name__ == "__main__":
    main()
