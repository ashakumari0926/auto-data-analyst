import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score


# -----------------------
# CLEAN DATA
# -----------------------
def clean_data(df):
    df = df.copy()

    # remove duplicates
    df = df.drop_duplicates()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())

        else:
            df[col] = df[col].fillna(
                df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            )

    return df


# -----------------------
# ML MODEL
# -----------------------
def run_ml(df):
    df = df.copy()

    target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    # convert categorical
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if y.dtype == "object" or y.nunique() < 10:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        metric = "Accuracy"

    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = r2_score(y_test, pred)
        metric = "R2 Score"

    return model, X_test, y_test, pred, score, metric