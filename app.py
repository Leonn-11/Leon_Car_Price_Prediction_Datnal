import streamlit as st
import pandas as pd
import pickle

# ============================
# LOAD MODEL & FITUR
# ============================
model = pickle.load(open("xgb_model.pkl", "rb"))
feature_names = pickle.load(open("xgb_features.pkl", "rb"))

# Load dataset untuk pilihan dropdown
df = pd.read_csv("car_price_prediction_.csv")

st.title("🚗 Car Price Prediction (XGBoost)")
st.write("Masukkan detail mobil untuk memprediksi harga jual.")

# ============================
# FORM INPUT USER
# ============================

brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
year = st.number_input("Tahun Mobil", min_value=1980, max_value=2025, value=2015)
engine = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)

fuel = st.selectbox("Fuel Type", sorted(df["Fuel Type"].unique()))
trans = st.selectbox("Transmission", sorted(df["Transmission"].unique()))
mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000, step=1000)
condition = st.selectbox("Condition", sorted(df["Condition"].unique()))

# ============================
# BUAT DATAFRAME USER INPUT
# ============================
user_df = pd.DataFrame([{
    "Brand": brand,
    "Year": year,
    "Engine Size": engine,
    "Fuel Type": fuel,
    "Transmission": trans,
    "Mileage": mileage,
    "Condition": condition
}])

# One-hot encoding
user_encoded = pd.get_dummies(user_df)

# Reindex agar sama dengan fitur training
user_encoded = user_encoded.reindex(columns=feature_names, fill_value=0)

# ============================
# PREDIKSI
# ============================
if st.button("Prediksi Harga"):
    pred = model.predict(user_encoded)[0]
    st.success(f"💰 Perkiraan harga mobil: **$USD {pred:,.0f}**")
