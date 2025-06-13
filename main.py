# API:
#   - Risk free rate from FRED (10 year Treasury yield)
#   - Market return S&P 500 
#   - Stock 

import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_risk_free_rate():
    """Fetch the latest 10-yr US Treasury yield (%) from FRED."""
    FRED_API_KEY = "10fc1d0cde03a5fb96a6f5515d429518"

    params = {
        "series_id": "DGS10",
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    }
    r = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params)
    data = r.json()["observations"]
    
    for obs in reversed(data):
        if obs["value"] != ".":
            return float(obs["value"]) / 100.0
    raise RuntimeError("No valid treasury yield found")

def fetch_market_return(period_days=365):
    end = datetime.today()
    start = end - timedelta(days=period_days + 10)
    sp = yf.download("^GSPC", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    sp = sp["Adj Close"].dropna()
    past_price = sp.iloc[0]
    latest_price = sp.iloc[-1]
    return (latest_price / past_price) - 1.0

def fetch_prices_from_capitaliq(
    ticker: str,
    start: str,
    end: str
) -> pd.DataFrame:
    CIQ_KEY = "CAPITALIQ_API_KEY"

    url = f"https://api.capitaliq.com/v1/companies/{ticker}/pricing/historical"
    params = {
        "startDate": start,
        "endDate":   end,
        "frequency": "Daily",
        "fields":    "OpenPrice,HighPrice,LowPrice,ClosePrice"
    }
    headers = {"Authorization": f"Bearer {CIQ_KEY}"}
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()

    records = resp.json()["data"]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df.rename(columns={
        "OpenPrice":  "Open",
        "HighPrice":  "High",
        "LowPrice":   "Low",
        "ClosePrice": "Close"
    })

def fetch_beta_from_capitaliq_historical(
    ticker: str,
    market_ticker: str = "^GSPC",
    period_days: int = 365,
) -> dict:
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=period_days + 10)
    start, end = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

    df_stock  = fetch_prices_from_capitaliq(ticker, start, end)

    df_mkt_raw = yf.download(market_ticker, start=start, end=end)
    df_market  = df_mkt_raw[["Open","High","Low","Close"]]

    df = pd.concat(
        [df_stock.add_prefix("stock_"), df_market.add_prefix("mkt_")],
        axis=1
    ).dropna()
    ret = df.pct_change().dropna()

    result = {}
    for price in ("open","high","low","close"):
        s_col = f"stock_{price}"
        m_col = f"mkt_{price}"

        cov = ret[s_col].cov(ret[m_col])
        var = ret[m_col].var()
        # ini per hari masih bingung mau gimana
        result[f"beta_{price}"] = cov / var

    return result


