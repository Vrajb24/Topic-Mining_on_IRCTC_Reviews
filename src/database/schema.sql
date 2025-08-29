-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT UNIQUE,
    content TEXT NOT NULL,
    rating INTEGER,
    language TEXT,
    date_posted TIMESTAMP,
    date_scraped TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    thumbs_up INTEGER DEFAULT 0,
    user_name TEXT,
    reply_content TEXT
);

-- Topics table
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_name TEXT NOT NULL,
    topic_words TEXT,
    topic_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Review-Topic mapping
CREATE TABLE IF NOT EXISTS review_topics (
    review_id INTEGER,
    topic_id INTEGER,
    confidence REAL,
    FOREIGN KEY (review_id) REFERENCES reviews (id),
    FOREIGN KEY (topic_id) REFERENCES topics (id)
);

-- Sentiment analysis results
CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER,
    sentiment_score REAL,
    sentiment_label TEXT,
    confidence REAL,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews (id)
);

-- Analytics cache
CREATE TABLE IF NOT EXISTS analytics_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT,
    metric_value TEXT,
    date_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expiry TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_reviews_date ON reviews(date_posted);
CREATE INDEX idx_reviews_processed ON reviews(processed);
CREATE INDEX idx_sentiments_review ON sentiments(review_id);
CREATE INDEX idx_review_topics_review ON review_topics(review_id);
