import pandas as pd
import numpy as np
import xgboost as xgb
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import praw
from datetime import datetime
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import io
import re
from dotenv import load_dotenv

# Use non-GUI backend for server stability
matplotlib.use('Agg')

load_dotenv()

CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "placeholder")
CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "placeholder")
USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "placeholder")

nlp_pipeline = None

def load_nlp_model():
    # Deep learning NLP model (roberta-base-sentiment) to calculate sentiment of posts
    global nlp_pipeline
    if nlp_pipeline is None:
        print("Loading TrendScope NLP model...")
        nlp_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=-1
        )
        print("TrendScope NLP engine is ready.")
    return nlp_pipeline

def scrape_topic(keyword: str, subreddit: str, limit: int = 500) -> pd.DataFrame:
    # Scrapes Reddit using Time-Slicing (30 days / 6 chunks) to ensure balanced historical data
    print(f"Starting Buffered Time-Slice Scraping for '{keyword}' in r/{subreddit}...")
    
    reddit = praw.Reddit(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT
    )
    
    subreddit_obj = reddit.subreddit(subreddit)
    unique_posts = {}
    
    # Regex pattern for strict whole-word matching
    target_keyword = keyword.lower()
    pattern = r'\b{}\b'.format(re.escape(target_keyword))

    days_to_look_back = 30
    number_of_slices = 6
    slice_duration = days_to_look_back // number_of_slices
    target_per_slice = (limit // number_of_slices) + 10
    SEARCH_BUFFER_MULTIPLIER = 10 
    
    current_time = int(datetime.now().timestamp())
    
    # Iterates through time chunks to fetch evenly distributed data
    for i in range(number_of_slices):
        end_epoch = current_time - (i * slice_duration * 86400)
        start_epoch = end_epoch - (slice_duration * 86400)
        
        query = f'"{keyword}" timestamp:{start_epoch}..{end_epoch}'
        
        print(f"   ... Slice {i+1}/{number_of_slices}...")
        
        try:
            slice_gen = subreddit_obj.search(
                query, 
                syntax='cloudsearch', 
                limit=target_per_slice * SEARCH_BUFFER_MULTIPLIER
            )
            
            count_for_this_slice = 0
            
            for post in slice_gen:
                if count_for_this_slice >= target_per_slice:
                    break

                if post.selftext in ['[deleted]', '[removed]']: continue
                
                # Filters irrelevant posts using regex
                content_combined = (post.title + " " + post.selftext).lower()
                if not re.search(pattern, content_combined):
                    continue

                if post.id not in unique_posts:
                    unique_posts[post.id] = {
                        'timestamp': datetime.fromtimestamp(post.created_utc),
                        'title': post.title,
                        'text_for_nlp': f"{post.title} {post.selftext[:500]}",
                        'upvotes': post.score,
                        'comments_count': post.num_comments
                    }
                    count_for_this_slice += 1
                    
        except Exception as e:
            print(f"      Warning: Slice {i+1} failed ({e}). Continuing...")
            continue

    # Fallback mechanism if time-slicing yields insufficient data
    if len(unique_posts) < 50:
        print("   (!) Low data from slices. Engaging Deep Fallback...")
        fallback_limit = limit * 5
        for post in subreddit_obj.search(keyword, sort='relevance', time_filter='year', limit=fallback_limit):
            if len(unique_posts) >= limit: break
            if post.selftext in ['[deleted]', '[removed]']: continue
            content_combined = (post.title + " " + post.selftext).lower()
            if not re.search(pattern, content_combined): continue
            
            unique_posts[post.id] = {
                'timestamp': datetime.fromtimestamp(post.created_utc),
                'title': post.title,
                'text_for_nlp': f"{post.title} {post.selftext[:500]}",
                'upvotes': post.score,
                'comments_count': post.num_comments
            }
    
    df = pd.DataFrame(list(unique_posts.values()))
    
    if df.empty:
        print("Warning: No posts found after strict filtering.")
        return pd.DataFrame(columns=['timestamp', 'title', 'text_for_nlp', 'upvotes', 'comments_count'])
        
    df = df.sort_values('timestamp')
    print(f"SUCCESS: Scraped {len(df)} high-quality posts.")
    return df

def map_sentiment_label(label_str):
    # Standardizes diverse model outputs (LABEL_0, negative) into unified categories
    l = label_str.lower()
    if 'negative' in l or 'label_0' in l: return 'Negative'
    if 'neutral' in l or 'label_1' in l: return 'Neutral'
    if 'positive' in l or 'label_2' in l: return 'Positive'
    return 'Neutral'

def run_trendscope_analysis(df: pd.DataFrame, topic_name: str) -> dict:
    if len(df) < 50:
        return {"error": "Not enough data (less than 50 posts).", "posts_loaded": len(df)}

    nlp_model = load_nlp_model()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calculates sentiment for each post using RoBERTa
    print("Running NLP sentiment scan...")
    sentiments = []
    texts = df['text_for_nlp'].tolist()
    
    for text in texts:
        try:
            res = nlp_model(str(text)[:1500], truncation=True, max_length=512)[0]
            cat = map_sentiment_label(res['label'])
        except Exception as e:
            cat = 'Neutral'
        sentiments.append(cat)

    df['sentiment_category'] = sentiments
    df['sent_Positive'] = (df['sentiment_category'] == 'Positive').astype(int)
    df['sent_Neutral'] = (df['sentiment_category'] == 'Neutral').astype(int)
    df['sent_Negative'] = (df['sentiment_category'] == 'Negative').astype(int)

    # Aggregates data into 3-Day buckets to smooth out daily volatility
    df.set_index('timestamp', inplace=True)
    df_3d = df.resample('3D').agg({
        'upvotes': 'sum', 
        'comments_count': 'sum',
        'sent_Positive': 'sum', 
        'sent_Negative': 'sum',
        'sent_Neutral': 'sum',
        'text_for_nlp': 'count'
    }).rename(columns={'text_for_nlp': 'post_volume'})

    df_3d = df_3d.fillna(0)
    
    # Feature Engineering for ML inputs
    df_3d['upvotes_lag1'] = df_3d['upvotes'].shift(1).fillna(0)
    df_3d['upvotes_trend'] = (df_3d['upvotes'] - df_3d['upvotes_lag1']) / (df_3d['upvotes_lag1'] + 1)
    df_3d['net_sentiment'] = (df_3d['sent_Positive'] - df_3d['sent_Negative']) / (df_3d['post_volume'] + 1)
    df_3d['next_upvotes'] = df_3d['upvotes'].shift(-1)
    
    # Defines classification targets based on 10% threshold
    threshold = 0.10
    def get_class(row):
        if pd.isna(row['next_upvotes']): return np.nan
        curr = row['upvotes'] + 1
        nxt = row['next_upvotes']
        change = (nxt - curr) / curr
        if change > threshold: return 'Increase'
        elif change < -threshold: return 'Decrease'
        else: return 'Same'
    df_3d['target'] = df_3d.apply(get_class, axis=1)

    # Prepares training data for XGBoost
    data_ml = df_3d.dropna().copy()
    features = ['upvotes', 'comments_count', 'sent_Positive', 'sent_Negative', 'post_volume', 'upvotes_trend', 'net_sentiment']
    
    if len(data_ml) < 5: 
        return {"error": "Not enough history intervals. Try increasing the scraping limit.", "posts_loaded": len(df)}

    X = data_ml[features]
    y = data_ml['target']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    split = int(len(X) * 0.8)
    if split >= len(X): split = len(X) - 1
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_enc[:split], y_enc[split:]
    
    # Trains XGBoost classifier to predict future momentum
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), eval_metric='mlogloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(X_test)) if len(X_test) > 0 else None
    last_row = df_3d.iloc[[-1]][features]
    future_pred_code = model.predict(last_row)[0]
    future_pred = le.inverse_transform([future_pred_code])[0]

    # Generates SHAP plots for model interpretability
    print("Generating SHAP plots...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    pred_class = int(future_pred_code)
    
    # SHAP Summary Plot
    plt.close('all') 
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values[:, :, pred_class], X_train, show=False)
    buf_summary = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf_summary, format='png', bbox_inches='tight')
    buf_summary.seek(0)
    summary_plot_bytes = buf_summary.read()
    plt.close(fig)

    # SHAP Force Plot (Local Prediction)
    plt.close('all')
    row_pos = -1
    instance = X.iloc[[row_pos]]
    local_sv = explainer(instance)
    force_plot = shap.force_plot(
        local_sv.base_values[0][pred_class],
        local_sv.values[0][:, pred_class],
        instance.iloc[0],
        feature_names=X.columns.tolist(),
        matplotlib=True, 
        show=False
    )
    buf_force = io.BytesIO()
    force_plot.savefig(buf_force, format='png', bbox_inches='tight')
    buf_force.seek(0)
    force_plot_bytes = buf_force.read()
    plt.close('all')

    # Generates Dashboard Visualizations (Volume, Sentiment Distribution, Trends)
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'pos': '#10B981', 'neu': '#9CA3AF', 'neg': '#EF4444', 'vol': '#4F46E5'}

    # 1. Volume Line Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_3d.index, df_3d['post_volume'], color=colors['vol'], linewidth=2.5, marker='o')
    ax.set_title("Mentions Trend Over Time", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Mentions")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')) # Month Day
    plt.xticks(rotation=45)
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    vol_plot_bytes = buf.read()
    plt.close(fig)

    # 2. Sentiment Donut Chart
    total_pos = df['sent_Positive'].sum()
    total_neu = df['sent_Neutral'].sum()
    total_neg = df['sent_Negative'].sum()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [total_pos, total_neu, total_neg]
    labels = ['Positive', 'Neutral', 'Negative']
    pie_colors = [colors['pos'], colors['neu'], colors['neg']]
    
    if sum(sizes) > 0:
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          startangle=90, colors=pie_colors, 
                                          pctdistance=0.85, wedgeprops=dict(width=0.3))
        plt.setp(autotexts, size=10, weight="bold", color="white")
    else:
        ax.text(0.5, 0.5, "No Sentiment Data", ha='center', va='center')
        
    ax.set_title("Overall Sentiment Breakdown", fontsize=12, fontweight='bold')
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    dist_plot_bytes = buf.read()
    plt.close(fig)

    # 3. Sentiment Trend Line Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_3d.index, df_3d['sent_Positive'], label='Positive', color=colors['pos'], linewidth=2)
    ax.plot(df_3d.index, df_3d['sent_Neutral'], label='Neutral', color=colors['neu'], linewidth=2, linestyle='--')
    ax.plot(df_3d.index, df_3d['sent_Negative'], label='Negative', color=colors['neg'], linewidth=2)
    ax.set_title("Sentiment Over Time", fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    trend_plot_bytes = buf.read()
    plt.close(fig)

    # 4 Extracts Top 5 Posts by Engagement
    df_reset = df.reset_index()
    top_posts_df = df_reset.nlargest(5, 'upvotes')[['title', 'upvotes', 'comments_count', 'sentiment_category', 'timestamp']]
    top_posts_df['timestamp'] = top_posts_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    top_posts_list = top_posts_df.to_dict(orient='records')

    # Compiles results dictionary
    latest_data = df_3d.iloc[-1]
    
    return {
        "topic_name": topic_name,
        "posts_loaded": len(df),
        "ml_status": "SUCCESS",
        "test_accuracy": f"{test_accuracy:.2%}" if test_accuracy else "N/A",
        "forecast": future_pred.upper(),
        "strategy": "COMMIT. Engagement is expected to grow." if future_pred == 'Increase'
                    else ("PIVOT. Interest is in decline." if future_pred == 'Decrease' else "MAINTAIN."),
        "recent_sentiment": {
            "positive_posts": int(latest_data['sent_Positive']),
            "negative_posts": int(latest_data['sent_Negative']),
            "post_volume": int(latest_data['post_volume'])
        },
        "plots": {
            "shap_summary": summary_plot_bytes,
            "shap_force": force_plot_bytes,
            "vol_chart": vol_plot_bytes,
            "sent_dist_chart": dist_plot_bytes,
            "sent_trend_chart": trend_plot_bytes
        },
        "top_posts": top_posts_list
    }