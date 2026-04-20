import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Page settings
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Title
st.title("🏠 Boston House Price Prediction")
st.markdown("Enter house details below to predict price")

# Sidebar inputs
st.sidebar.header("Input Features")

feature_names = [
    'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
    'DIS','RAD','TAX','PTRATIO','B','LSTAT'
]

inputs = []
for feature in feature_names:
    val = st.sidebar.number_input(feature, min_value=0.0, value=1.0)
    inputs.append(val)

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict([inputs])

    st.success(f"💰 Estimated Price: ${prediction[0]*1000:.2f}")

    st.info("This prediction is based on a trained machine learning model using Boston housing dataset.")

    # Feature importance (only works for Random Forest)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        features = pd.Series(importance, index=feature_names)

        st.subheader("Feature Importance")
        st.bar_chart(features.sort_values(ascending=False))