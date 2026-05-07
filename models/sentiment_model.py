"""
Sentiment Analysis Model
Uses a pre-trained DistilBERT model fine-tuned on SST-2 (Stanford Sentiment Treebank)
for fast, accurate sentiment classification.
"""

from transformers import pipeline
import torch


class SentimentAnalyser:
    """
    Wraps HuggingFace DistilBERT sentiment pipeline.
    Labels: POSITIVE / NEGATIVE with confidence score.
    """

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print("Loading sentiment model...")
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
            truncation=True,
            max_length=512
        )
        print("Model loaded successfully.")

    def analyse(self, text: str) -> dict:
        """
        Analyse sentiment of a single text string.

        Returns:
            {
                "label": "POSITIVE" | "NEGATIVE",
                "score": float (0-1),
                "emoji": "😊" | "😠"
            }
        """
        result = self.pipeline(text)[0]
        return {
            "label": result["label"],
            "score": round(result["score"] * 100, 2),
            "emoji": "😊" if result["label"] == "POSITIVE" else "😠"
        }

    def analyse_batch(self, texts: list) -> list:
        """
        Analyse a list of texts and return aggregated results.

        Returns:
            List of individual results + summary stats
        """
        results = []
        for text in texts:
            if text.strip():
                r = self.analyse(text)
                r["text"] = text
                results.append(r)
        return results

    def get_summary(self, results: list) -> dict:
        """Compute summary statistics from a batch of results."""
        if not results:
            return {}

        positive = [r for r in results if r["label"] == "POSITIVE"]
        negative = [r for r in results if r["label"] == "NEGATIVE"]
        avg_score = sum(r["score"] for r in results) / len(results)

        return {
            "total":         len(results),
            "positive":      len(positive),
            "negative":      len(negative),
            "positive_pct":  round(len(positive) / len(results) * 100, 1),
            "negative_pct":  round(len(negative) / len(results) * 100, 1),
            "avg_confidence": round(avg_score, 2),
            "overall_sentiment": "POSITIVE" if len(positive) >= len(negative) else "NEGATIVE"
        }
