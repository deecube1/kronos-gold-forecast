
import requests
import torch
from transformers import pipeline

FINNHUB_API_KEY_ENV = None  # will be set from environment

def get_gold_news_sentiment(api_key):
    """Fetch Gold news and run FinBERT sentiment analysis."""
    try:
        # Fetch news
        resp = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": "general", "token": api_key}
        )
        data = resp.json()

        gold_keywords = ['gold', 'xau', 'bullion', 'precious metal',
                         'fed', 'inflation', 'rate cut', 'dollar', 'yields']

        gold_news = []
        for item in data:
            headline = item.get('headline', '')
            summary  = item.get('summary', '')
            if any(kw in headline.lower() or kw in summary.lower() for kw in gold_keywords):
                gold_news.append(headline)

        if not gold_news:
            return None

        # Load FinBERT
        finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )

        # Run sentiment
        results = finbert(gold_news[:10])  # max 10 headlines

        # Gold-adjusted sentiment
        def gold_sentiment(headline, result):
            label = result['label']
            score = result['score']
            h = headline.lower()
            dollar_bearish = any(w in h for w in ['dollar falls', 'dollar weakens', 'dollar drops', 'dollar dips'])
            yields_down    = any(w in h for w in ['yields fall', 'yields drop', 'rate cut', 'yields ease'])
            gold_bullish   = any(w in h for w in ['gold rises', 'gold surges', 'gold hits high', 'gold gains'])
            if dollar_bearish or yields_down or gold_bullish:
                if label == 'negative': label = 'positive'
                elif label == 'positive': label = 'negative'
            return label, score

        scores = []
        headlines_sentiment = []
        for headline, result in zip(gold_news[:10], results):
            gold_label, score = gold_sentiment(headline, result)
            scores.append(1 if gold_label == 'positive' else -1 if gold_label == 'negative' else 0)
            emoji = "📈" if gold_label == "positive" else "📉" if gold_label == "negative" else "➡️"
            headlines_sentiment.append(f"{emoji} {headline[:80]}")

        avg = sum(scores) / len(scores) if scores else 0
        if avg > 0.2:
            overall = "📈 BULLISH"
        elif avg < -0.2:
            overall = "📉 BEARISH"
        else:
            overall = "➡️ NEUTRAL"

        return {
            "overall": overall,
            "score": avg,
            "count": len(gold_news),
            "headlines": headlines_sentiment[:5],
        }

    except Exception as e:
        return None
