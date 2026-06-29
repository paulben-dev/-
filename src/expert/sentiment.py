"""Sentiment analysis for social media posts using FinBERT."""
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# FinBERT label mapping
LABEL_MAP = {0: "Neutral", 1: "Bullish", 2: "Bearish"}


class FinBERTSentiment:
    """Financial sentiment analysis using ProsusAI/finbert."""

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert"
            ).to(self._device)
            self._model.eval()
        return self._model

    def analyze(self, text: str) -> str:
        """Classify text as Bullish, Bearish, or Neutral."""
        if not text or not text.strip():
            return "Neutral"
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            return LABEL_MAP[prediction]
        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return "Neutral"


class SentimentRouter:
    """Routes posts to the correct sentiment analyzer based on source."""

    def __init__(self, finbert: FinBERTSentiment | None = None):
        self.finbert = finbert or FinBERTSentiment()

    def label_post(self, post) -> "Post":
        """Label a post with sentiment, using FinBERT only for unlabeled sources."""
        if post.source == "stocktwits":
            return post  # Already self-labeled
        sentiment = self.finbert.analyze(post.content)
        post.sentiment = sentiment
        return post

    def label_posts(self, posts: list) -> list:
        """Label a batch of posts."""
        return [self.label_post(p) for p in posts]
