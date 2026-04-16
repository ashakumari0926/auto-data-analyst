import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_analyst_automate import clean_data, run_ml


# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Auto Data Analyst", layout="wide")

st.title("🤖 Auto Data Analyst")
st.write("Upload CSV → Auto EDA → Visualization → ML Model")


# -----------------------
# UPLOAD FILE
# -----------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # SAFE DISPLAY (FIX ARROW ERROR)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head().astype(str))

    st.subheader("📌 Shape")
    st.write(df.shape)

    st.subheader("📌 Column Types")
    st.write(df.dtypes)

    st.subheader("📌 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📌 Describe Data")
    st.write(df.describe(include="all"))


    # -----------------------
    # CLEAN DATA
    # -----------------------
    df = clean_data(df)
    st.success("✅ Data Cleaned Successfully")


    # -----------------------
    # VISUALIZATION
    # -----------------------
    st.subheader("📊 Histograms")

    num_cols = df.select_dtypes(include=np.number).columns

    if len(num_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        df[num_cols].hist(ax=ax)
        st.pyplot(fig)
        plt.clf()

    st.subheader("🔥 Correlation Heatmap")

    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


    # -----------------------
    # ML MODEL
    # -----------------------
    st.subheader("🤖 Model Training")

    model, X_test, y_test, pred, score, metric = run_ml(df)

    st.write(f"### {metric}: {score:.4f}")


    # -----------------------
    # PREDICTIONS
    # -----------------------
    st.subheader("📌 Sample Predictions")

    result_df = pd.DataFrame({
        "Actual": y_test.iloc[:10].values,
        "Predicted": pred[:10]
    })

    st.dataframe(result_df.astype(str))