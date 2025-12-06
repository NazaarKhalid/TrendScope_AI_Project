# TrendScope 📈

TrendScope is an AI-powered trend analysis and forecasting tool. It scrapes real-time social media data from Reddit, analyzes sentiment using deep learning (NLP), forecasts trend momentum with machine learning (XGBoost), and provides explainable insights through SHAP values.

## 🚀 Features

### Backend (Python / FastAPI)
- Advanced Scraping: Uses PRAW with a time-sliced, buffered scraping algorithm for balanced 30-day data collection.
- Strict Filtering: Regex-based whole-word filtering removes irrelevant noise.
- Deep Learning NLP: Uses cardiffnlp/twitter-roberta-base-sentiment-latest for high-accuracy sentiment detection.
- Predictive Modeling: Trains an XGBoost classifier on-the-fly to predict whether a trend is increasing, decreasing, or stable.
- Explainable AI: Generates SHAP plots to show why a prediction was made.
- Visualization: Creates dynamic trend graphs via Matplotlib.

### Frontend (React)
- Interactive Dashboard: Displays Volume, Sentiment Distribution, Sentiment Trends, and Top Posts.
- Search Scopes: Choose between “All of Reddit” or selected subreddits.
- Transparent Loading: Indicates every backend stage (Scraping → NLP → ML).
- Smart Alerts: Warns when model accuracy drops below 75% or when data is insufficient.

## 🛠️ Tech Stack

### Backend
- Python 3.10+
- FastAPI, Uvicorn
- PRAW
- Pandas, Scikit-Learn
- XGBoost
- Transformers (Hugging Face)
- SHAP
- Matplotlib

### Frontend
- React.js
- Axios
- CSS3

## 📋 Prerequisites
- Python 3.9+
- Node.js & npm
- Reddit Developer Account

## ⚙️ Installation & Setup

### 1. Clone the Repository
```
git clone https://github.com/your-username/TrendScope.git
cd TrendScope
```

### 2. Backend Setup
```
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
REDDIT_CLIENT_ID="your_client_id"
REDDIT_CLIENT_SECRET="your_client_secret"
REDDIT_USER_AGENT="python:TrendScope:v1.0 (by /u/yourusername)"
```

### 3. Frontend Setup
```
cd frontend
npm install
npm start
```

## 🏃‍♂️ Running the Application
Backend:
```
cd backend
uvicorn app.main:app --reload
```

Frontend:
```
cd frontend
npm start
```

## 📂 Project Structure
```
TrendScope/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── trendscope.py
│   ├── requirements.txt
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── App.css
│   └── package.json
└── README.md
```

