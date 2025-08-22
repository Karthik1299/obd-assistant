import streamlit as st
import requests
import json

BACKEND_URL = "http://127.0.0.1:8000"  # Update to Render URL later

st.title("OBD-II Technician Assistant")

code = st.text_input("Enter OBD-II Code (e.g., P0301)")
if st.button("Get Diagnosis"):
    if code:
        try:
            response = requests.post(f"{BACKEND_URL}/query", json={"code": code})
            data = response.json()
            st.markdown(data["response"])
        except Exception as e:
            st.error(f"Error: {e}")

if st.button("Show Recent Queries"):
    try:
        history = requests.get(f"{BACKEND_URL}/history").json()
        for item in history:
            st.write(f"**{item['timestamp']} - Code: {item['code']}**")
            st.markdown(item["response"])
    except Exception as e:
        st.error(f"Error: {e}")