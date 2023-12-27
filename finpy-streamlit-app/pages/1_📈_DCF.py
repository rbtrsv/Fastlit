import streamlit as st
import requests
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
import plotly.subplots as sp

st.set_page_config(page_title="Discounted Cash Flow", page_icon="📈")

st.markdown("# Discounted Cash Flow 💸")
st.sidebar.header("Discounted Cash Flow 💸")
# st.sidebar.markdown("# Discounted Cash Flow 💸")
st.write(
    """Discounted cash flow (DCF) refers to a valuation method that estimates the value of an investment using its expected future cash flows."""
)