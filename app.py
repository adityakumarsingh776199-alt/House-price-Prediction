"""Price estimator:  streamlit run app.py   (run train.py first)"""
import json, joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="PriceLens")
model = joblib.load("outputs/model.joblib"); metrics = json.load(open("outputs/metrics.json"))
st.title("PriceLens — House Price Estimator")
area = st.slider("Area (sqft)", 400, 4000, 1200, 50); bed = st.slider("Bedrooms", 1, 6, 3)
bath = st.slider("Bathrooms", 1, 5, 2); age = st.slider("Age (years)", 0, 40, 8)
loc = st.selectbox("Locality", ["Sector 5", "Model Town", "Civil Lines", "Green Park", "City Center", "Old Town", "Industrial Area", "Outskirts"])
p = model.predict(pd.DataFrame([{"area_sqft": area, "bedrooms": bed, "bathrooms": bath, "age_years": age, "locality": loc}]))[0]
st.metric("Predicted price", f"₹{p / 1e5:,.1f} lakh")
st.caption(f"Model: {metrics['best_model']} · test R² {metrics['test_r2']} · RMSE ₹{metrics['test_rmse'] / 1e5:.1f} lakh")
st.image("outputs/feature_importance.png")
