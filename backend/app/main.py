from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import base64
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trendscope import scrape_topic, run_trendscope_analysis, load_nlp_model

# Initialize FastAPI with CORS middleware
app = FastAPI(title="TrendScope API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation and response structure
class AnalysisRequest(BaseModel):
    keyword: str = Body(..., example="Tesla")
    subreddits: str = Body(..., example="technology+electricvehicles+elonmusk")
    limit: Optional[int] = Body(500, example=300)

class SentimentData(BaseModel):
    positive_posts: int
    negative_posts: int
    post_volume: int

class TopPost(BaseModel):
    title: str
    upvotes: int
    comments_count: int
    sentiment_category: str
    timestamp: str

class AnalysisResult(BaseModel):
    topic_name: str
    posts_loaded: int
    ml_status: str
    test_accuracy: str
    forecast: str
    strategy: str
    recent_sentiment: SentimentData
    
    shap_summary_b64: str
    shap_force_b64: str
    vol_chart_b64: str
    sent_dist_chart_b64: str
    sent_trend_chart_b64: str
    
    top_posts: List[TopPost]

@app.on_event("startup")
async def startup_event():
    # Pre-loads NLP model on server startup to reduce latency
    try:
        load_nlp_model()
    except Exception as e:
        print(f"FATAL: Failed to load NLP model on startup: {e}")

@app.get("/")
def read_root():
    return {"message": "TrendScope API is running. Use the /analyze endpoint."}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_trend(request: AnalysisRequest):
    # Main endpoint: Scrapes data, analyzes trends, and returns results
    try:
        # 1. Scrape Data
        df_scraped = scrape_topic(
            keyword=request.keyword,
            subreddit=request.subreddits,
            limit=request.limit
        )

        # 2. Run Analysis
        analysis_data = run_trendscope_analysis(
            df=df_scraped,
            topic_name=request.keyword
        )
        
        # Returns 422 for data sufficiency errors
        if "error" in analysis_data:
            raise HTTPException(status_code=422, detail=analysis_data['error'])

        # 3. Encode Images for Transport
        def enc(img_bytes): return base64.b64encode(img_bytes).decode('utf-8')

        response = {
            "topic_name": analysis_data['topic_name'],
            "posts_loaded": analysis_data['posts_loaded'],
            "ml_status": analysis_data['ml_status'],
            "test_accuracy": analysis_data['test_accuracy'],
            "forecast": analysis_data['forecast'],
            "strategy": analysis_data['strategy'],
            "recent_sentiment": analysis_data['recent_sentiment'],
            
            "shap_summary_b64": enc(analysis_data['plots']['shap_summary']),
            "shap_force_b64": enc(analysis_data['plots']['shap_force']),
            "vol_chart_b64": enc(analysis_data['plots']['vol_chart']),
            "sent_dist_chart_b64": enc(analysis_data['plots']['sent_dist_chart']),
            "sent_trend_chart_b64": enc(analysis_data['plots']['sent_trend_chart']),
            
            "top_posts": analysis_data['top_posts']
        }

        return response

    except HTTPException as e:
        # Passes specific HTTP exceptions (like 422) through to frontend
        raise e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Data Error: {e}")
    except Exception as e:
        # Catches unexpected server crashes
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")