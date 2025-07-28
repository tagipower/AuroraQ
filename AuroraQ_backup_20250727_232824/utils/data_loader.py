# utils/data_loader.py

import pandas as pd

def load_historical_data():
    price_path = "data/price/BTCUSDT_5m.csv"
    sentiment_path = "data/sentiment/sentiment_scores.csv"
    regime_path = "data/regime/long_term_regime.csv"

    price_df = pd.read_csv(price_path, parse_dates=["timestamp"])
    sentiment_df = pd.read_csv(sentiment_path, parse_dates=["timestamp"])
    regime_df = pd.read_csv(regime_path, parse_dates=["timestamp"])

    df = price_df.merge(sentiment_df, on="timestamp", how="left")
    df = df.merge(regime_df, on="timestamp", how="left")
    df = df.fillna(method="ffill")

    return df
