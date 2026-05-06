# Sentiment Analysis Dashboard

A real-time **Sentiment Analysis Dashboard** built with DistilBERT, Flask, and Chart.js. Supports both single text analysis and batch review analysis with visualised results.

---

## Demo

![Dashboard Preview](static/preview.png)

---

## Features

- **Single Text Analysis** — Paste any text and get instant POSITIVE/NEGATIVE sentiment with confidence score
- **Batch Review Analysis** — Paste multiple reviews (one per line), get individual results + summary statistics
- **Visual Dashboard** — Doughnut chart breakdown, stats grid, colour-coded review list
- **DistilBERT Backbone** — Fine-tuned on SST-2, fast and accurate on CPU and GPU

---

## Architecture

```
User Input (text)
      │
      ▼
Text Preprocessing (clean, split)
      │
      ▼
DistilBERT Sentiment Pipeline (HuggingFace Transformers)
      │
      ▼
Flask REST API (/analyse, /analyse_batch)
      │
      ▼
Interactive Dashboard (Chart.js + Vanilla JS)
```

---

## Installation & Run

```bash
git clone https://github.com/Adityacodes06/sentiment-dashboard.git
cd sentiment-dashboard
pip install -r requirements.txt
python app.py
```

Then open: [http://localhost:5000](http://localhost:5000)

---

## Project Structure

```
sentiment-dashboard/
├── models/
│   └── sentiment_model.py    # DistilBERT sentiment pipeline wrapper
├── utils/
│   └── text_utils.py         # Text cleaning and preprocessing
├── templates/
│   └── index.html            # Full dashboard UI
├── app.py                    # Flask app and API routes
└── requirements.txt
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve dashboard UI |
| `/analyse` | POST | Single text sentiment |
| `/analyse_batch` | POST | Batch review analysis |

**Example request:**
```json
POST /analyse
{ "text": "This product is absolutely amazing!" }
```

**Example response:**
```json
{ "label": "POSITIVE", "score": 99.8, "emoji": "😊" }
```

---

## Tech Stack

- [HuggingFace Transformers](https://huggingface.co/) — DistilBERT model
- [Flask](https://flask.palletsprojects.com/) — REST API backend
- [Chart.js](https://www.chartjs.org/) — Data visualisation
- [PyTorch](https://pytorch.org/) — Model inference

---

## Author

**Aditya Srivastava**
B.E. Chemical Engineering | BITS Pilani, Goa | 2026
[LinkedIn](https://www.linkedin.com/in/aditya-srivastava-a0857a254) · adityasrivastava784@gmail.com


*Automated maintenance update: 2026-05-06 18:20:14*
