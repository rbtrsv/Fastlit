import streamlit as st
import requests
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import plotly.subplots as sp

option = st.selectbox(
    'For what company would you like to update data?',
    ('EasySales', 'Confidas', 'Gapminder', 'FlowX'))

# st.write('You selected:', option)

passwords_dictionary = {'EasySales': '12345', 'Confidas': 'parola123', 'Gapminder': 'password'}

password = st.text_input("Enter the password for the selected company", type="password")

if not password:
    st.stop()

# Allow only .csv and .xlsx files to be uploaded
if passwords_dictionary[f'{option}'] == password:
    uploaded_file = st.file_uploader("Upload spreadsheet", type=["csv", "xlsx"])

if not uploaded_file:
    st.stop()

if uploaded_file.type == "text/csv":
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.dataframe(df.head())

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate('/Users/robert.radoslav/Developer/hf-code-streamlit/hf-db-fb-firebase-adminsdk.json')

# app = firebase_admin.initialize_app(cred)

@st.cache(allow_output_mutation=True)
def create_app():
    app = firebase_admin.initialize_app(cred)
    return app
    
create_app()

db = firestore.client()

if uploaded_file:
    for indices, row in df.iterrows():
            doc_ref = db.collection("Portfolio companies").document(str(option)).collection("Years").document(str(row['Year']))

            doc_ref.set({
                "Year": int(row['Year']),
                "Revenue": int(row['Revenue']),
                "EBITDA": int(row['EBITDA']),
                "Net Profit": int(row['Net Profit']),
                u'Timestamp': firestore.SERVER_TIMESTAMP
            })

users = list(db.collection(u'Portfolio companies').document(str(option)).collection("Years").stream())

users_dict = list(map(lambda x: x.to_dict(), users))
df_uploaded = pd.DataFrame(users_dict)
df_uploaded