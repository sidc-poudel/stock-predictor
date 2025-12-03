from flask import Flask, render_template
import yfinance as yf
import requests
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from transformers import pipeline

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------
NEWS_API_KEY = "dbabbc90e1314ac2916e62ce29a2d76e"
DEFAULT_SYMBOL = "BROS"

# HuggingFace sentiment pipeline
sentiment_pipe = pipeline("sentiment-analysis")


# -----------------------------
# Fetch stock price
# -----------------------------
def get_price(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="1d")
    if df.empty:
        return None
    price = df["Close"].iloc[-1]
    return round(price, 2)


# -----------------------------
# Predict next day price using last 5 closes
# -----------------------------
def predict_next_day_price(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="10d")
    if df.empty or len(df) < 6:
        return None
    closes = df["Close"].values
    # Build linear regression on last 5-day windows
    X = np.array([closes[i:i + 5] for i in range(len(closes) - 5)])
    y = closes[5:]
    model = LinearRegression()
    model.fit(X, y)
    last_5 = closes[-5:].reshape(1, -1)
    pred = model.predict(last_5)
    return round(float(pred[0]), 2)


# -----------------------------
# Fetch relevant news
# -----------------------------
def get_relevance_score(article, company_name=""):
    score = 0
    company_name = company_name.upper()
    title = article.get("title", "").upper()
    description = article.get("description", "").upper()

    if company_name not in title and company_name not in description:
        return 0
    if company_name in title:
        score += 5
    if company_name in description:
        score += 3

    keywords = ["STOCK", "SHARES", "MARKET", "EARNINGS", "PROFIT", "LOSS", "FORECAST"]
    for word in keywords:
        if word in title or word in description:
            score += 1

    return score


def get_stock_news(ticker):
    stock = yf.Ticker(ticker)
    company_name = stock.info.get("shortName", "")
    if not company_name:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "pageSize": 20
    }
    try:
        response = requests.get(url, params=params).json()
    except:
        return []

    if "articles" not in response:
        return []

    articles = response["articles"]
    filtered_articles = []
    for article in articles:
        relevance = get_relevance_score(article, company_name)
        if relevance > 0:
            article["relevance"] = relevance
            # Run sentiment analysis
            try:
                sentiment = sentiment_pipe(article.get("description", "")[:512])[0]
                article["sentiment_label"] = sentiment["label"]
                article["sentiment_score"] = round(sentiment["score"], 2)
            except:
                article["sentiment_label"] = "N/A"
                article["sentiment_score"] = 0
            filtered_articles.append(article)

    filtered_articles.sort(key=lambda x: (x["relevance"], x["publishedAt"]), reverse=True)
    return filtered_articles


# -----------------------------
# Candlestick chart
# -----------------------------
def get_chart(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="6mo", interval="1d")
    if df.empty:
        return []
    df = df.reset_index()
    candles = []
    for i, row in df.iterrows():
        date_str = str(row["Date"]).split(" ")[0]
        candles.append({
            "x": date_str,
            "y": [
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"])
            ]
        })
    return candles


# -----------------------------
# FLASK ROUTE
# -----------------------------
@app.route("/")
def index():
    symbol = DEFAULT_SYMBOL
    price = get_price(symbol)
    predicted_price = predict_next_day_price(symbol)
    news = get_stock_news(symbol)
    candles = get_chart(symbol)

    return render_template(
        "index.html",
        symbol=symbol,
        price=price,
        predicted_price=predicted_price,
        news=news,
        candles=json.dumps(candles)
    )


# -----------------------------
# RUN FLASK
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
