import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Breast Cancer Detector", page_icon="🩺", layout="centered")
st.title("🩺 Breast Cancer Detector")

@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")

pipe = load_model()
features = pipe.feature_names_in_

st.write("Upload a CSV with the following columns:")
st.code(", ".join(features), language="text")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        pred_class = pipe.predict(df[features])
        prob_benign = pipe.predict_proba(df[features])[:, 1]
        df_out = df.copy()
        df_out["pred_class"] = pred_class
        df_out["prob_benign"] = prob_benign
        
        st.success("Predictions complete!")
        st.dataframe(df_out.head(20))
        
        csv = df_out.to_csv(index=False)
        st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")

st.caption("Note: In this dataset, 1 = benign, 0 = malignant")
