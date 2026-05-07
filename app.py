"""
Sentiment Analysis Dashboard — Flask App
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from models.sentiment_model import SentimentAnalyser
from utils.text_utils import parse_reviews, clean_text

app = Flask(__name__)
analyser = SentimentAnalyser()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyse", methods=["POST"])
def analyse():
    """Single text analysis endpoint."""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text   = clean_text(text)
    result = analyser.analyse(text)
    return jsonify(result)


@app.route("/analyse_batch", methods=["POST"])
def analyse_batch():
    """Batch analysis endpoint — accepts newline-separated reviews."""
    data  = request.get_json()
    raw   = data.get("text", "").strip()

    if not raw:
        return jsonify({"error": "No text provided"}), 400

    reviews = parse_reviews(raw)
    if not reviews:
        return jsonify({"error": "No valid reviews found"}), 400

    results = analyser.analyse_batch(reviews)
    summary = analyser.get_summary(results)

    return jsonify({
        "results": results,
        "summary": summary
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
