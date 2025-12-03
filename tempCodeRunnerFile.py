# -----------------------------
# Fetch stock news
# -----------------------------
from newspaper import Article

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
            # Run sentiment analysis on full article
            try:
                url = article.get("url", "")
                text = ""
                if url:
                    try:
                        news_article = Article(url)
                        news_article.download()
                        news_article.parse()
                        text = news_article.text
                    except:
                        text = article.get("description", "")
                if not text.strip():
                    article["sentiment_label"] = "N/A"
                    article["sentiment_score"] = 0
                else:
                    sentiment = analyze_full_text(text)
                    article["sentiment_label"] = sentiment["label"]
                    article["sentiment_score"] = round(sentiment["score"], 2)
            except:
                article["sentiment_label"] = "N/A"
                article["sentiment_score"] = 0
            filtered_articles.append(article)

    filtered_articles.sort(key=lambda x: (x["relevance"], x["publishedAt"]), reverse=True)
    return filtered_articles
