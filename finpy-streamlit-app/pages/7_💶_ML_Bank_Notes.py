import json
import requests
import streamlit as st
import os 

if os.environ.get("BACKEND_URL") is not None:
    BACKEND_URL = os.environ.get("BACKEND_URL")
else:
    BACKEND_URL = "http://localhost:8080"

st.title("Bank notes")

st.write('Select bank notes details from below 👇')
variance = st.slider('Variance', -10, 10, 0)
skewness = st.slider('Skewness', -10, 10, 0)
kurtosis = st.slider('Kurtosis', -10, 10, 0)
entropy = st.slider('Entropy', -10, 10, 0)

input = {'variance': variance, 'skewness': skewness, 'kurtosis': kurtosis, 'entropy': entropy,}

if st.button('Calculate genuine or forged'):
    # res = requests.post(url = 'http://finpy-api:8080/ml_models/bank_notes/predict', data=json.dumps(input))
    res = requests.post(url = f'{BACKEND_URL}/ml_models/bank_notes/predict', data=json.dumps(input))
    # st.subheader(f'Response from FastAPI: {res.text}')
    response_json = res.json()
    response_json_subset = response_json['prediction']
    st.subheader(f'Response from FastAPI: {response_json_subset}')


if st.button('Calculate genuine or forged Production'):
    res = requests.post(url = 'https://api.finpy.tech/ml_models/bank_notes/predict', data=json.dumps(input))
    # st.subheader(f'Response from FastAPI: {res.text}')
    response_json = res.json()
    response_json_subset = response_json['prediction']
    st.subheader(f'Response from FastAPI: {response_json_subset}')
