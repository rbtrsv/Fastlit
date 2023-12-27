import streamlit as st
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

st.set_page_config(page_title="Mapping Demo", page_icon="🌍")

st.markdown("# Greenblatt's Magic Formula 💰")
st.sidebar.header("Greenblatt's Magic Formula 💰")
st.write(
    """Magic formula investing tells you how to approach value investing from a methodical and unemotional perspective."""
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
    globals()[f"data_bs_qtrly_{stock}"] = client.get_fundamental_equity(f"{stock}", filter_='Financials::Balance_Sheet::quarterly')
    globals()[f"data_bs_qtrly_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_bs_qtrly_{stock}"], orient='index')
    globals()[f"data_bs_qtrly_{stock}"]['Symbol'] = stock
    globals()[f"data_bs_qtrly_{stock}"].reset_index(inplace=True)
    globals()[f"data_bs_qtrly_{stock}"].drop(columns=['index'], inplace=True)
    globals()[f"data_bs_qtrly_{stock}"]['date'] = pd.to_datetime(globals()[f"data_bs_qtrly_{stock}"]['date'], format='%Y-%m-%d', exact=True)

    globals()[f"data_is_qtrly_{stock}"] = client.get_fundamental_equity(f"{stock}", filter_='Financials::Income_Statement::quarterly')
    globals()[f"data_is_qtrly_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_is_qtrly_{stock}"], orient='index')
    globals()[f"data_is_qtrly_{stock}"]['Symbol'] = stock
    globals()[f"data_is_qtrly_{stock}"].reset_index(inplace=True)
    globals()[f"data_is_qtrly_{stock}"].drop(columns=['index'], inplace=True)
    globals()[f"data_is_qtrly_{stock}"]['date'] = pd.to_datetime(globals()[f"data_is_qtrly_{stock}"]['date'], format='%Y-%m-%d', exact=True)

    globals()[f"data_cf_qtrly_{stock}"] = client.get_fundamental_equity(f"{stock}", filter_='Financials::Cash_Flow::quarterly')
    globals()[f"data_cf_qtrly_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_cf_qtrly_{stock}"], orient='index')
    globals()[f"data_cf_qtrly_{stock}"]['Symbol'] = stock
    globals()[f"data_cf_qtrly_{stock}"].reset_index(inplace=True)
    globals()[f"data_cf_qtrly_{stock}"].drop(columns=['index'], inplace=True)
    globals()[f"data_cf_qtrly_{stock}"]['date'] = pd.to_datetime(globals()[f"data_cf_qtrly_{stock}"]['date'], format='%Y-%m-%d', exact=True)

    globals()[f"data_highlights_{stock}"] = client.get_fundamental_equity(f"{stock}", filter_='Highlights')
    globals()[f"data_highlights_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_highlights_{stock}"], orient='index')
    globals()[f"data_highlights_{stock}"]['Symbol'] = stock
    globals()[f"data_highlights_{stock}"].reset_index(inplace=True)
    mapping = {globals()[f"data_highlights_{stock}"].columns[0]: 'Metric', globals()[f"data_highlights_{stock}"].columns[1]: 'Value'}
    globals()[f"data_highlights_{stock}"] = globals()[f"data_highlights_{stock}"].rename(columns=mapping)

    globals()[f"data_valuation_{stock}"] = client.get_fundamental_equity(f"{stock}", filter_='Valuation')
    globals()[f"data_valuation_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_valuation_{stock}"], orient='index')
    globals()[f"data_valuation_{stock}"]['Symbol'] = stock
    globals()[f"data_valuation_{stock}"].reset_index(inplace=True)
    mapping = {globals()[f"data_valuation_{stock}"].columns[0]: 'Metric', globals()[f"data_valuation_{stock}"].columns[1]: 'Value'}
    globals()[f"data_valuation_{stock}"] = globals()[f"data_valuation_{stock}"].rename(columns=mapping)

    globals()[f"data_share_stats_{stock}"] = client.get_fundamental_equity(f"{stock}", filter_='SharesStats')
    globals()[f"data_share_stats_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_share_stats_{stock}"], orient='index')
    globals()[f"data_share_stats_{stock}"]['Symbol'] = stock
    globals()[f"data_share_stats_{stock}"].reset_index(inplace=True)
    mapping = {globals()[f"data_share_stats_{stock}"].columns[0]: 'Metric', globals()[f"data_share_stats_{stock}"].columns[1]: 'Value'}
    globals()[f"data_share_stats_{stock}"] = globals()[f"data_share_stats_{stock}"].rename(columns=mapping)

    globals()[f"data_prices_{stock}"] = client.get_prices_eod(f"{stock}", period='d', order='d', from_='2010-01-05')
    globals()[f"data_prices_{stock}"] = pd.DataFrame.from_dict(globals()[f"data_prices_{stock}"], orient='columns')
    globals()[f"data_prices_{stock}"]['Symbol'] = stock
    globals()[f"data_prices_{stock}"]['date'] = pd.to_datetime(globals()[f"data_prices_{stock}"]['date'], format='%Y-%m-%d', exact=True)


balance_sheet_stocks = pd.DataFrame()
income_statement_stocks = pd.DataFrame()
cash_flow_stocks = pd.DataFrame()
highlights_stocks = pd.DataFrame()
valuation_stocks = pd.DataFrame()

for stock in stocks_list:
    balance_sheet_stocks = balance_sheet_stocks.append(globals()[f"data_bs_qtrly_{stock}"])
    income_statement_stocks = income_statement_stocks.append(globals()[f"data_is_qtrly_{stock}"])
    cash_flow_stocks = cash_flow_stocks.append(globals()[f"data_cf_qtrly_{stock}"])
    highlights_stocks = highlights_stocks.append(globals()[f"data_highlights_{stock}"])
    valuation_stocks = valuation_stocks.append(globals()[f"data_valuation_{stock}"])

clist = list(balance_sheet_stocks.columns)
clist_new = clist[-1:]+clist[:-1]
balance_sheet_stocks = balance_sheet_stocks[clist_new]

clist = list(income_statement_stocks.columns)
clist_new = clist[-1:]+clist[:-1]
income_statement_stocks = income_statement_stocks[clist_new]

clist = list(cash_flow_stocks.columns)
clist_new = clist[-1:]+clist[:-1]
cash_flow_stocks = cash_flow_stocks[clist_new]

is_bl_stocks = pd.merge(income_statement_stocks, balance_sheet_stocks, how="outer", on=["Symbol", "date"])
is_bl_cf_stocks = pd.merge(is_bl_stocks, cash_flow_stocks, how="outer", on=["Symbol", "date"])
# is_bl_cf_stocks

year = 2022

for stock in stocks_list:
    globals()[f"data_shares_outstanding_{stock}"] =  globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['commonStockSharesOutstanding'].head(1).astype(float).sum()
    
    # globals()[f"data_adjusted_close_{stock}"] = float(globals()[f"data_prices_{stock}"]['adjusted_close'].head(1))
    globals()[f"data_adjusted_close_{stock}"] = globals()[f"data_prices_{stock}"][(globals()[f"data_prices_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_prices_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['adjusted_close'].head(1).astype(float).sum()
    
    globals()[f"data_market_cap_{stock}"] = globals()[f"data_highlights_{stock}"][globals()[f"data_highlights_{stock}"]['Metric'] == 'MarketCapitalization']['Value'].astype(float)
    globals()[f"data_market_cap_calculated_{stock}"] = globals()[f"data_shares_outstanding_{stock}"] * globals()[f"data_adjusted_close_{stock}"]
    ### Use the one below to replace globals()[f"data_market_cap_{stock}"] with he value of globals()[f"data_market_cap_calculated_{stock}"]
    globals()[f"data_market_cap_{stock}"] = globals()[f"data_shares_outstanding_{stock}"] * globals()[f"data_adjusted_close_{stock}"]

    # globals()[f"data_total_debt_{stock}"] = globals()[f"data_bs_qtrly_{stock}"].head(1).loc[:, 'shortLongTermDebtTotal'].astype(float).sum()
    globals()[f"data_total_debt_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['shortLongTermDebtTotal'].head(1).astype(float).sum()
    
    # globals()[f"data_cash_equiv_{stock}"] = globals()[f"data_bs_qtrly_{stock}"].head(1).loc[:, 'cash'].astype(float).sum()
    globals()[f"data_cash_equiv_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['cash'].head(1).astype(float).sum()

    globals()[f"data_ev_{stock}"] = globals()[f"data_valuation_{stock}"][globals()[f"data_valuation_{stock}"]['Metric'] == 'EnterpriseValue']['Value'].astype(float)
    globals()[f"data_ev_calculated_{stock}"] = globals()[f"data_market_cap_calculated_{stock}"] + globals()[f"data_total_debt_{stock}"] - globals()[f"data_cash_equiv_{stock}"]
    ### Use the one below to replace globals()[f"data_ev_{stock}"] with he value of globals()[f"data_ev_calculated_{stock}"]
    globals()[f"data_ev_{stock}"] = globals()[f"data_market_cap_calculated_{stock}"] + globals()[f"data_total_debt_{stock}"] - globals()[f"data_cash_equiv_{stock}"]

    ##### Note that I use globals()[f"data_market_cap_{stock}"] and globals()[f"data_ev_{stock}"]. If I want to backtest I need to use globals()[f"data_market_cap_calculated_{stock}"] and globals()[f"data_ev_calculated_{stock}"].

    # globals()[f"data_ebit_{stock}"] = globals()[f"data_is_qtrly_{stock}"].head(4).loc[:, 'ebit'].astype(float).sum()
    globals()[f"data_ebit_{stock}"] = globals()[f"data_is_qtrly_{stock}"][(globals()[f"data_is_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_is_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['ebit'].astype(float).sum()
    globals()[f"data_earnings_yield_{stock}"] = globals()[f"data_ebit_{stock}"] / globals()[f"data_ev_{stock}"]

    # globals()[f"data_nfa_ppe_{stock}"] = globals()[f"data_bs_qtrly_{stock}"].head(1).loc[:, 'propertyPlantEquipment'].astype(float).sum()
    globals()[f"data_nfa_ppe_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['propertyPlantEquipment'].head(1).astype(float).sum()
    
    # globals()[f"data_current_assets_{stock}"] = globals()[f"data_bs_qtrly_{stock}"].head(1).loc[:, 'totalCurrentAssets'].astype(float).sum()
    globals()[f"data_current_assets_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['totalCurrentAssets'].head(1).astype(float).sum()

    # globals()[f"data_current_liabilities_{stock}"] = globals()[f"data_bs_qtrly_{stock}"].head(1).loc[:, 'totalCurrentLiabilities'].astype(float).sum()
    globals()[f"data_current_liabilities_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['totalCurrentLiabilities'].head(1).astype(float).sum()
    
    # globals()[f"data_long_term_debt_{stock}"] = globals()[f"data_bs_qtrly_{stock}"].head(1).loc[:, 'longTermDebt'].astype(float).sum()
    globals()[f"data_long_term_debt_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['longTermDebt'].head(1).astype(float).sum()

    # globals()[f"data_total_debt_{stock}"] = globals()[f"data_bs_qtrly_{stock}"].head(1).loc[:, 'shortLongTermDebtTotal'].astype(float).sum()
    globals()[f"data_total_debt_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['shortLongTermDebtTotal'].head(1).astype(float).sum()
    
    # Net Working Capital according to Greenblatt (https://www.valuesignals.com/Glossary/Details/Net_Working_Capital?securityId=13381)
    # globals()[f"data_nwc_{stock}"] = globals()[f"data_current_assets_{stock}"] - globals()[f"data_current_liabilities_{stock}"]
    # globals()[f"data_nwc_{stock}"] = globals()[f"data_bs_qtrly_{stock}"][(globals()[f"data_bs_qtrly_{stock}"]['date'] >= pd.to_datetime(f'{year}-1-31')) & (globals()[f"data_bs_qtrly_{stock}"]['date'] <= pd.to_datetime(f'{year}-12-31'))]['netWorkingCapital'].head(1).astype(float).sum()
    globals()[f"data_nwc_{stock}"] = max((globals()[f"data_current_assets_{stock}"] - globals()[f"data_cash_equiv_{stock}"]) - (globals()[f"data_current_liabilities_{stock}"] - (globals()[f"data_total_debt_{stock}"] - globals()[f"data_long_term_debt_{stock}"])), 0)
   
    globals()[f"data_roc_{stock}"] = globals()[f"data_ebit_{stock}"] / (globals()[f"data_nfa_ppe_{stock}"] + globals()[f"data_nwc_{stock}"])

for stock in stocks_list:
    globals()[f"data_greenblatt_{stock}"] = pd.DataFrame(columns = ['Symbol','Market Capitalization', 'Total Debt', 'Cash Equivalents', 'Enterprise Value', 'EBIT', 'Net Fixed Assets', 'Current Assets', 'Current Liabilities', 'Long Term Debt', 'Net Working Capital', 'Earnings Yield', 'Return on Capital'])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Symbol'] = stock
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Market Capitalization'] = float(globals()[f"data_market_cap_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Total Debt'] = float(globals()[f"data_total_debt_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Cash Equivalents'] = float(globals()[f"data_cash_equiv_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Enterprise Value'] = float(globals()[f"data_ev_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'EBIT'] = float(globals()[f"data_ebit_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Net Fixed Assets'] = float(globals()[f"data_nfa_ppe_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Current Assets'] = float(globals()[f"data_current_assets_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Current Liabilities'] = float(globals()[f"data_current_liabilities_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Long Term Debt'] = float(globals()[f"data_long_term_debt_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Net Working Capital'] = float(globals()[f"data_nwc_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Earnings Yield'] = float(globals()[f"data_earnings_yield_{stock}"])
    globals()[f"data_greenblatt_{stock}"].loc[0, 'Return on Capital'] = float(globals()[f"data_roc_{stock}"])

greenblatt_stocks = pd.DataFrame()

for stock in stocks_list:
    greenblatt_stocks = greenblatt_stocks.append(globals()[f"data_greenblatt_{stock}"])

greenblatt_stocks.reset_index(inplace=True)
greenblatt_stocks.drop(columns=['index'], inplace=True)
# greenblatt_stocks

greenblatt_stocks['Earnings Yield Rank'] = greenblatt_stocks['Earnings Yield'].rank(method='max')
greenblatt_stocks['Return on Capital Rank'] = greenblatt_stocks['Return on Capital'].rank(method='max')
greenblatt_stocks['Combined Score'] = greenblatt_stocks['Earnings Yield Rank'] / greenblatt_stocks['Earnings Yield Rank'].max() + greenblatt_stocks['Return on Capital Rank'] / greenblatt_stocks['Return on Capital Rank'].max()
# greenblatt_stocks

st.table(greenblatt_stocks)