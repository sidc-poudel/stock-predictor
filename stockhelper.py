from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.io as pio
import requests

app = Flask(__name__)

# --------------------------
# Load FinBERT model
# --------------------------
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def finbert_score(text):
    if not text:
        return 0
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        result = model(**inputs)
    scores = torch.softmax(result.logits, dim=1).numpy()[0]
    sentiment = ["Negative", "Neutral", "Positive"][np.argmax(scores)]
    return sentiment, float(scores[np.argmax(scores)])

# --------------------------
# News API (Bing)
# --------------------------
BING_KEY = "demo"
BING_URL = "https://api.bing.microsoft.com/v7.0/news/search"

def get_news(query):
    headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
    params = {"q": query, "count": 5, "sortBy": "Date"}

    try:
        r = requests.get(BING_URL, headers=headers, params=params)
        data = r.json()
        articles = []

        if "value" not in data:
            return []

        for item in data["value"]:
            title = item["name"]
            url = item["url"]
            desc = item.get("description", "")

            # Relevancy score = simple keyword match
            score = title.lower().count(query.lower()) + desc.lower().count(query.lower())
            score = round(score + 1.0, 2)

            sent, sent_score = finbert_score(desc)

            articles.append({
                "title": title,
                "url": url,
                "desc": desc,
                "relevancy": score,
                "sentiment": sent,
                "sentiment_score": round(sent_score, 3)
            })

        # Sort MOST relevant at top
        articles = sorted(articles, key=lambda x: -x["relevancy"])
        return articles

    except:
        return []

# --------------------------
# Prediction (very simple)
# --------------------------
def simple_predict(prices):
    if len(prices) < 5:
        return "Not enough data"

    recent = prices[-5:]
    avg = np.mean(recent[:-1])
    if recent[-1] > avg:
        return "Possible slight upward trend"
    return "Possible slight downward trend"


# --------------------------
# Flask Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    ticker = None
    fig_json = None
    prediction = None
    news = []

    if request.method == "POST":
        ticker = request.form["ticker"].upper()

        stock = yf.Ticker(ticker)

        try:
            df = stock.history(period="1mo")
            if df.empty:
                return render_template("index.html", error="Invalid ticker")

            # ---------------- Chart ----------------
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            fig.update_layout(height=500)
            fig_json = pio.to_json(fig)

            prediction = simple_predict(df["Close"].values)

            news = get_news(ticker)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html",
                           ticker=ticker,
                           fig_json=fig_json,
                           prediction=prediction,
                           news=news)

if __name__ == "__main__":
    app.run(debug=True)
