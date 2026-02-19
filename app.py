import os
import streamlit as st
import pandas as pd
import joblib
from src.shap_utils import shap_single
import streamlit as st
import pandas as pd
import joblib
from src.shap_utils import shap_single

st.set_page_config(page_title="GDSC Drug Sensitivity Predictor", layout="wide")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_best_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return model, features


model, feature_cols = load_model()

st.title("🧬 OncoResponseAI")
st.caption("Explainable AI Platform for Cancer Drug Sensitivity Prediction (GDSC)")


# Sidebar
st.sidebar.header("Input Parameters")

AUC = st.sidebar.slider("AUC", 0.5, 1.5, 0.9)
Z_SCORE = st.sidebar.slider("Z_SCORE", -3.0, 3.0, 0.0)
DRUG_ID = st.sidebar.number_input("DRUG_ID", value=1003)

TARGET = st.sidebar.text_input("TARGET", "TOP1")
TARGET_PATHWAY = st.sidebar.text_input("TARGET_PATHWAY", "DNA replication")

plant_only = st.sidebar.checkbox("🌱 Show Plant-based Drugs Only")

# Input dataframe
input_df = pd.DataFrame([{
    'AUC': AUC,
    'Z_SCORE': Z_SCORE,
    'DRUG_ID': DRUG_ID,
    'TARGET': TARGET,
    'TARGET_PATHWAY': TARGET_PATHWAY
}])

input_df = pd.get_dummies(input_df)

for col in feature_cols:
    if col not in input_df:
        input_df[col] = 0

input_df = input_df[feature_cols]

# Prediction
pred = model.predict(input_df)[0]

st.metric("Predicted LN_IC50", f"{pred:.3f}")

# SHAP
st.subheader("🔍 SHAP Explanation")
fig = shap_single(model, input_df)
st.pyplot(fig)

st.info("Lower LN_IC50 = Higher drug sensitivity")
