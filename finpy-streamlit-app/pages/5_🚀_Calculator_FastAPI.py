import streamlit as st
import json
import requests

st.title('Basic Calculator App 🧮')

# Taking user inputs
option = st.selectbox('What operation would you like to perform?', 
                      ('Addition', 'Substraction', 'Multiplication', 'Division'))

st.write('Select the numbers from the sliders below 👇')
x = st.slider('X', 0, 100, 20)
y = st.slider('Y', 0, 130, 10)

# Converting the inputs into a json format
input = {'operation': option, 'x': x, 'y': y}

if st.button('Calculate'):
    res = requests.post(url = 'http://finpy-api:8080/calculate', data=json.dumps(input))
    st.subheader(f'Response from FastAPI: {res.text}')

if st.button('Calculate Production'):
    res = requests.post(url = 'https://api.finpy.tech/calculate', data=json.dumps(input))
    st.subheader(f'Response from FastAPI: {res.text}')