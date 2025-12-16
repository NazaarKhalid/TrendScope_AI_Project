import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // Manages application state (inputs, results, loading status, errors)
  const [keyword, setKeyword] = useState('');
  
  // Search Scope State (Global vs Custom)
  const [searchMode, setSearchMode] = useState('global');
  const [customSubs, setCustomSubs] = useState(''); 

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  
  const [error, setError] = useState(null); 
  const [popupError, setPopupError] = useState(null); 

  const [activeTab, setActiveTab] = useState('forecast');

  // Parses accuracy string to determine if warning is needed
  const getAccuracyVal = () => {
    if (!results || !results.test_accuracy) return 0;
    const val = parseFloat(results.test_accuracy.replace('%', ''));
    return isNaN(val) ? 0 : val;
  };

  // Simulates backend progress stages (Scraping -> NLP -> ML -> Viz)
  useEffect(() => {
    let timers = [];
    if (loading) {
      setLoadingMsg("Fetching data from Reddit...");
      timers.push(setTimeout(() => setLoadingMsg("Analyzing sentiment with NLP..."), 4000));
      timers.push(setTimeout(() => setLoadingMsg("Training predictive models (XGBoost)..."), 10000));
      timers.push(setTimeout(() => setLoadingMsg("Generating visualizations..."), 15000));
    }
    return () => timers.forEach(t => clearTimeout(t));
  }, [loading]);

  // Validates inputs and sends POST request to backend
  const handleAnalyze = async () => {
    if (!keyword) {
      setError("Please enter a keyword.");
      return;
    }

    if (searchMode === 'custom' && !customSubs.trim()) {
      setError("Please enter at least one subreddit for custom search.");
      return;
    }

    const subredditsToSend = searchMode === 'global' ? 'all' : customSubs;

    setLoading(true);
    setError(null);
    setPopupError(null); 
    setResults(null);
    setActiveTab('forecast');

    try {
      const response = await axios.post('http://127.0.0.1:8000/analyze', {
        keyword: keyword,
        subreddits: subredditsToSend,
        limit: 500
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
      
      // Catches specific 422 errors for insufficient data popups
      if (err.response && err.response.status === 422) {
        setPopupError(err.response.data.detail);
      } else {
        setError("An error occurred. Ensure backend is running and keyword is valid.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      
      {/* Navigation bar */}
      <nav className="navbar">
        <div className="logo">TrendScope</div>
        <div className="nav-actions">
          <button className="nav-btn">Subscription</button>
          <div className="account-icon">👤</div>
        </div>
      </nav>

      {/* Error popup modal for insufficient data */}
      {popupError && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-icon">⚠️</div>
            <h3>Insufficient Data</h3>
            <p>{popupError}</p>
            <div className="modal-actions">
              <button className="modal-btn" onClick={() => setPopupError(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Search Area with Scope Toggle */}
      {!results && !loading && (
        <div className="search-container">
          <h1 className="hero-title">Discover the Next Trend</h1>
          
          <div className="search-bar-wrapper">
            <input
              type="text"
              placeholder="Enter keyword (e.g., 'Tesla', 'Nike')"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              className="search-input"
              onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
            />
            <button onClick={handleAnalyze} className="analyze-btn">Analyze</button>
          </div>

          <div className="scope-container">
            <div className="scope-toggle">
              <button 
                className={`scope-btn ${searchMode === 'global' ? 'active' : ''}`}
                onClick={() => setSearchMode('global')}
              >
                🌍 All of Reddit
              </button>
              <button 
                className={`scope-btn ${searchMode === 'custom' ? 'active' : ''}`}
                onClick={() => setSearchMode('custom')}
              >
                🎯 Specific Subreddits
              </button>
            </div>
            {searchMode === 'custom' && (
              <div className="custom-subs-wrapper fade-in">
                <input 
                  type="text" 
                  className="custom-subs-input"
                  placeholder="e.g. technology+marketing+stocks"
                  value={customSubs}
                  onChange={(e) => setCustomSubs(e.target.value)}
                />
                <small>Separate multiple subreddits with a plus (+)</small>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading Spinner */}
      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <p className="loading-text fade-in">{loadingMsg}</p>
        </div>
      )}

      {/* Results Dashboard */}
      {results && !loading && (
        <div className="dashboard">
          <button className="back-btn" onClick={() => setResults(null)}>← Back to Search</button>
          
          <div className="header-section">
            <h2>Analysis for: <span className="highlight">{results.topic_name}</span></h2>
            <div className="meta-badge">Posts Scraped: {results.posts_loaded}</div>
            <div className="meta-badge">Model Accuracy: {results.test_accuracy}</div>
          </div>

          {/* Accuracy Warning */}
          {getAccuracyVal() < 75 && getAccuracyVal() > 0 && (
            <div className="accuracy-warning">
              <span>⚠️</span>
              <div>
                <strong>Low Confidence Model:</strong> The model accuracy is below 75%. 
                This often happens with erratic data or new trends. Predictions may be less reliable.
              </div>
            </div>
          )}

          {/* Tab Navigation */}
          <div className="tabs-container">
            <button 
              className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`} 
              onClick={() => setActiveTab('forecast')}
            >
              Forecast
            </button>
            <button 
              className={`tab-btn ${activeTab === 'graphs' ? 'active' : ''}`} 
              onClick={() => setActiveTab('graphs')}
            >
              Graphs
            </button>
            <button 
              className={`tab-btn ${activeTab === 'shap' ? 'active' : ''}`} 
              onClick={() => setActiveTab('shap')}
            >
              SHAP Explanation
            </button>
          </div>

          <div className="tab-content">
            
            {/* Forecast Tab */}
            {activeTab === 'forecast' && (
              <div className="metrics-grid">
                <div className={`card forecast-card ${results.forecast.toLowerCase()}`}>
                  <h3>Momentum Forecast</h3>
                  <div className="big-stat">{results.forecast}</div>
                  <p className="strategy-text">{results.strategy}</p>
                </div>

                <div className="card sentiment-card">
                  <h3>Recent Sentiment (Last 3 Days)</h3>
                  <div className="sentiment-bars">
                    <div className="stat-row">
                      <span>Positive</span>
                      <div className="progress-bar">
                        <div className="fill pos" style={{width: `${(results.recent_sentiment.positive_posts / (results.recent_sentiment.post_volume || 1)) * 100}%`}}></div>
                      </div>
                      <span>{results.recent_sentiment.positive_posts}</span>
                    </div>
                    <div className="stat-row">
                      <span>Negative</span>
                      <div className="progress-bar">
                        <div className="fill neg" style={{width: `${(results.recent_sentiment.negative_posts / (results.recent_sentiment.post_volume || 1)) * 100}%`}}></div>
                      </div>
                      <span>{results.recent_sentiment.negative_posts}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Graphs Tab */}
            {activeTab === 'graphs' && (
              <div className="graphs-dashboard">
                <div className="viz-card">
                  <img src={`data:image/png;base64,${results.vol_chart_b64}`} alt="Volume Trend" />
                </div>
                <div className="viz-card">
                  <img src={`data:image/png;base64,${results.sent_dist_chart_b64}`} alt="Sentiment Dist" />
                </div>
                <div className="viz-card">
                  <img src={`data:image/png;base64,${results.sent_trend_chart_b64}`} alt="Sentiment Trend" />
                </div>
                <div className="viz-card table-card">
                  <h3>🔥 Top Posts by Engagement</h3>
                  <div className="table-responsive">
                    <table className="posts-table">
                      <thead>
                        <tr>
                          <th>Date</th>
                          <th>Title</th>
                          <th>👍</th>
                          <th>💬</th>
                          <th>Sentiment</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.top_posts.map((post, idx) => (
                          <tr key={idx}>
                            <td className="date-cell">{post.timestamp}</td>
                            <td className="title-cell" title={post.title}>
                              {post.title.length > 50 ? post.title.substring(0, 50) + "..." : post.title}
                            </td>
                            <td>{post.upvotes}</td>
                            <td>{post.comments_count}</td>
                            <td>
                              <span className={`badge ${post.sentiment_category.toLowerCase()}`}>
                                {post.sentiment_category}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* SHAP Tab */}
            {activeTab === 'shap' && (
              <div className="viz-section">
                <div className="viz-card">
                  <h3>Global Feature Importance</h3>
                  <img 
                    src={`data:image/png;base64,${results.shap_summary_b64}`} 
                    alt="SHAP Summary Plot" 
                    className="shap-img"
                  />
                </div>
                <div className="viz-card">
                  <h3>Latest Prediction Factors</h3>
                  <img 
                    src={`data:image/png;base64,${results.shap_force_b64}`} 
                    alt="SHAP Force Plot" 
                    className="shap-img"
                  />
                </div>
              </div>
            )}

          </div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}
    </div>
  );
}

export default App;