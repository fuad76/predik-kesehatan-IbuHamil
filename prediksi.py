import streamlit as st
import pandas as pd
import joblib

st.title("🩺 Analisis Prediksi Kesehatan Ibu Hamil")
st.write("Masukkan data ibu hamil untuk memprediksi tingkat kesehatannya (Low/Mid/High Level).")
st.write("📌 KETERANGAN:")
st.write("Low level: Kesehatan ibu hamil rendah")
st.write("Mid level: Kesehatan ibu hamil sedang")
st.write("High level: Kesehatan ibu hamil Tinggi")

# load model
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# input data
age = st.number_input("Age")
sbp = st.number_input("SystolicBP")
dbp = st.number_input("DiastolicBP")
bs = st.number_input("Blood Sugar")
temp = st.number_input("Body Temperature")
hr = st.number_input("Heart Rate")

# prediksi
if st.button("Prediksi"):
    df = pd.DataFrame([[age, sbp, dbp, bs, temp, hr]],
                      columns=["Age","SystolicBP","DiastolicBP","BS","BodyTemp","HeartRate"])
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)[0]
    st.success(f"Hasil Prediksi: {pred}")
