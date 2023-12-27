import streamlit as st
import pandas as pd
import pandas as pd
import requests
import string
import time
import numpy as np
import datetime as dt
import plotly.graph_objects as go
import plotly.subplots as sp

pd.options.display.float_format = '{:,.3f}'.format
pd.options.display.max_columns = None
pd.options.display.max_rows = None

from eod import EodHistoricalData
api_key = '62cc0bbb785900.00386114'
api_key = 'demo'
client = EodHistoricalData(api_key)

st.set_page_config(page_title="DataFrame Demo", page_icon="📊")

st.markdown("# Percent of Stocks Above 50-Day Average 🎉")
st.sidebar.header("Percent of Stocks Above 50-Day Average 🎉")
st.write(
    """The calculation is straightforward: simply divide the number of stocks above their XX-day moving average by the total number of stocks in the underlying index."""
)

# snp500 = pd.read_csv("/Users/robert.radoslav/Developer/streamlit_tut/datasets/constituents_csv.csv")
# symbols = snp500['Symbol'].sort_values().tolist()

# ticker = st.sidebar.selectbox(
#     'Choose a S&P 500 Stock',
#      symbols)

# stocks_list = []
# stocks_list.append(str(ticker) + ".US")

stocks_list = ["AAPL.US"]

for stock in stocks_list:
    globals()[f"data_prices_{stock}"] = client.get_prices_eod(f'{stock}', period='d', order='d', from_='2005-01-05')
    globals()[f"data_prices_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_prices_{stock}"], orient='columns')
    globals()[f"data_prices_{stock}"]['Symbol'] = stock
    globals()[f"data_prices_{stock}"]['date'] = pd.to_datetime(globals()[f"data_prices_{stock}"]['date'], format='%Y-%m-%d', exact=True)

    globals()[f"data_sma_{stock}"] = client.get_instrument_ta(f'{stock}', function='sma', from_='2005-01-05', to='', period=50, filter_='')
    globals()[f"data_sma_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_sma_{stock}"], orient='columns')
    globals()[f"data_sma_{stock}"]['Symbol'] = stock
    globals()[f"data_sma_{stock}"]['date'] = pd.to_datetime(globals()[f"data_sma_{stock}"]['date'], format='%Y-%m-%d', exact=True)
    globals()[f"data_sma_{stock}"].sort_values(by='date', ascending=False, inplace=True)


for stock in stocks_list:
    globals()[f"data_a50sma_{stock}"] = pd.merge(globals()[f"data_sma_{stock}"], globals()[f"data_prices_{stock}"], how="left", on=["date"])
    globals()[f"data_a50sma_{stock}"]['Above'] = np.where(globals()[f"data_a50sma_{stock}"]['adjusted_close'] > globals()[f"data_a50sma_{stock}"]['sma'], 1, 0)


a50sma_all = pd.DataFrame()

for stock in stocks_list:
    a50sma_all = a50sma_all.append(globals()[f"data_a50sma_{stock}"])

a50sma_all.reset_index(inplace=True)
a50sma_all.drop(columns=['index'], inplace=True)
a50sma_all.head(25)
st.table(a50sma_all.head(25))

value_counts_a50sma_all = a50sma_all.groupby(['date']).agg({'Above': [np.mean]})
value_counts_a50sma_all.sort_values(by='date', ascending=False, inplace=True)
value_counts_a50sma_all.reset_index(inplace=True)
value_counts_a50sma_all.head(25)
st.table(value_counts_a50sma_all.head(25))


fig = go.Figure([go.Scatter(x=value_counts_a50sma_all['date'], y=value_counts_a50sma_all['Above']['mean'].to_list())])
fig.show()

st.plotly_chart(fig, use_container_width=True)