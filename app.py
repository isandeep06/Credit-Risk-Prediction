from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Credit Risk Studio", page_icon="💳", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "random_forest_credit_model.pkl"
ENCODER_DIR = BASE_DIR / "encoders"
DATA_PATH = BASE_DIR / "german_credit_data.csv"
ACCENT = "#4F46E5"
SURFACE = "#0B1021"
CARD = "#11162A"
MUTED = "#9CA3AF"
PRESET_PROFILES = {
    "Confident borrower": {
        "Age": 35,
        "Sex": "male",
        "Job": 2,
        "Housing": "own",
        "Saving accounts": "rich",
        "Checking account": "moderate",
        "Credit amount": 2500,
        "Duration": 18,
    },
    "Budget conscious": {
        "Age": 29,
        "Sex": "female",
        "Job": 1,
        "Housing": "rent",
        "Saving accounts": "little",
        "Checking account": "little",
        "Credit amount": 1200,
        "Duration": 12,
    },
    "Higher exposure": {
        "Age": 46,
        "Sex": "male",
        "Job": 3,
        "Housing": "own",
        "Saving accounts": "moderate",
        "Checking account": "little",
        "Credit amount": 8000,
        "Duration": 36,
    },
}

FEATURE_COLS = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
]
CAT_COLS = ["Sex", "Housing", "Saving accounts", "Checking account"]
FILL_VALUES = {"Saving accounts": "little", "Checking account": "little"}

encoder_files = {
    "Sex": "Sex_label_encoder.pkl",
    "Housing": "Housing_label_encoder.pkl",
    "Saving accounts": "Saving accounts_label_encoder.pkl",
    "Checking account": "Checking account_label_encoder.pkl",
}


def normalize_categories(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for col, fill_val in FILL_VALUES.items():
        if col in df:
            df[col] = df[col].replace("NA", pd.NA).fillna(fill_val)
    return df


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = {col: joblib.load(ENCODER_DIR / fname) for col, fname in encoder_files.items()}
    explainer = shap.TreeExplainer(model)
    return model, encoders, explainer


def encode_frame(df_raw: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = normalize_categories(df_raw)
    for col in CAT_COLS:
        df[col] = encoders[col].transform(df[col])
    return df


def load_background_df(encoders: dict) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[FEATURE_COLS]
    return encode_frame(df, encoders)


model, encoders, explainer = load_artifacts()
background_df = load_background_df(encoders)

st.markdown(
    f"""
    <style>
    .stApp {{
        background: radial-gradient(circle at 20% 20%, #1e1b4b 0%, #0b1021 45%, #060815 100%);
        color: #e5e7eb;
    }}
    .app-card {{
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        background: linear-gradient(135deg, {CARD}, #0c1224);
        box-shadow: 0 25px 70px rgba(0,0,0,0.35);
    }}
    .section-title {{
        font-size: 1.05rem;
        letter-spacing: 0.02em;
        font-weight: 600;
        margin-bottom: 0.35rem;
        color: #e5e7eb;
    }}
    .muted {{ color: {MUTED}; font-size: 0.92rem; }}
    div[data-testid="stMetricValue"] {{ color: #f8fafc; }}
    div[data-testid="stMetricDelta"] {{ color: {ACCENT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Credit Risk Predictor")
st.markdown(
    "<span class='muted'>Tailored what-if analysis with SHAP-based local explanations.</span>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("**How to use**")
    st.write("1) Fill inputs or pick a preset. 2) Run prediction. 3) Review SHAP tabs for drivers.")
    st.markdown("**Model**")
    st.write("Random Forest (credit risk) fitted on german_credit_data.csv. Encoders stored under /encoders. SHAP uses TreeExplainer.")
    st.markdown("**Reading SHAP**")
    st.write("Positive SHAP pushes toward Good; negative toward Bad. Bar chart = global impact; waterfall = this scenario.")

with st.container():
    cols = st.columns([1.1, 0.9, 1])
    with cols[0]:
        st.markdown("<div class='section-title'>Applicant profile</div>", unsafe_allow_html=True)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
        job = st.slider("Job seniority (0-3)", min_value=0, max_value=3, value=1)

    with cols[1]:
        st.markdown("<div class='section-title'>Accounts</div>", unsafe_allow_html=True)
        saving_account = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich", "NA"], index=0)
        checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "NA"], index=0)
        housing = st.selectbox("Housing", ["own", "rent", "free"], index=0)

    with cols[2]:
        st.markdown("<div class='section-title'>Credit terms</div>", unsafe_allow_html=True)
        credit_amount = st.number_input("Credit Amount", min_value=100, max_value=100000, value=5000, step=500)
        duration = st.slider("Duration (months)", min_value=1, max_value=60, value=24)

user_input = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Job": [job],
    "Housing": [housing],
    "Saving accounts": [saving_account],
    "Checking account": [checking_account],
    "Credit amount": [credit_amount],
    "Duration": [duration],
})

preset_name = st.selectbox("Preset scenario", ["Custom"] + list(PRESET_PROFILES.keys()))
active_input = user_input.copy()
if preset_name != "Custom":
    preset = PRESET_PROFILES[preset_name]
    for col, val in preset.items():
        active_input[col] = [val]
    st.info(f"Using preset: {preset_name}.")

encoded_input = encode_frame(active_input.copy(), encoders)

cta_col, _ = st.columns([0.6, 0.4])
with cta_col:
    predict_clicked = st.button("Run prediction", type="primary", use_container_width=True)

if predict_clicked:
    proba = model.predict_proba(encoded_input)[0]
    prediction = int(proba.argmax())
    label = "Good" if prediction == 1 else "Bad"
    st.metric(label="Predicted risk", value=label, delta=f"{proba[1] * 100:.1f}% good")

    # Quick probability readout for transparency (no extra box)
    st.write("Class probabilities")
    st.dataframe(pd.DataFrame({"Bad": [proba[0]], "Good": [proba[1]]}), use_container_width=True)

    shap_values = explainer.shap_values(encoded_input)
    if isinstance(shap_values, list):
        sample_values = shap_values[prediction][0]
        base_value = explainer.expected_value[prediction]
    else:
        values = shap_values
        if values.ndim == 3:
            # shape: (samples, features, outputs)
            sample_values = values[0, :, prediction]
            base_val = explainer.expected_value
            base_value = base_val[prediction] if isinstance(base_val, (list, np.ndarray)) else base_val
        elif values.ndim == 2:
            sample_values = values[0]
            base_value = explainer.expected_value
        else:
            sample_values = values
            base_value = explainer.expected_value

    explanation = shap.Explanation(
        values=sample_values,
        base_values=base_value,
        data=encoded_input.iloc[0],
        feature_names=FEATURE_COLS,
    )

    tab_local, tab_global = st.tabs(["Why this prediction", "Model-wide drivers"])
    with tab_local:
        fig, _ = plt.subplots(figsize=(9, 6))
        shap.plots.waterfall(explanation, max_display=len(FEATURE_COLS), show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with tab_global:
        global_shap_values = explainer.shap_values(background_df)
        if isinstance(global_shap_values, list):
            global_values = global_shap_values[1]
        else:
            global_values = global_shap_values

        # Top feature impact metrics for quick scan
        mean_abs = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": np.abs(global_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).head(5)
        st.markdown("**Top drivers (mean |SHAP|, top 5)**")
        st.dataframe(mean_abs.reset_index(drop=True))

        plt.figure(figsize=(9, 6))
        shap.summary_plot(global_values, background_df, plot_type="bar", show=False)
        plt.tight_layout()
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=True)