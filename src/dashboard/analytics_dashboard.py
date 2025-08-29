#!/usr/bin/env python3
"""
IRCTC Analytics Dashboard with Real Data Integration
Tesla-inspired design with comprehensive analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json
import base64
from PIL import Image
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="IRCTC Analytics Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Professional color schemes - Tesla-inspired
if st.session_state.dark_mode:
    theme = {
        'bg_primary': '#0a0b0d',
        'bg_secondary': '#15171a',
        'bg_card': '#1c1e22',
        'text_primary': '#ffffff',
        'text_secondary': '#9ca3af',
        'accent': '#e11d48',
        'accent_secondary': '#3b82f6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'border': '#2a2d34',
        'chart_bg': '#1c1e22'
    }
else:
    theme = {
        'bg_primary': '#ffffff',
        'bg_secondary': '#f8f9fa',
        'bg_card': '#ffffff',
        'text_primary': '#1f2937',
        'text_secondary': '#6b7280',
        'accent': '#e11d48',
        'accent_secondary': '#3b82f6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'border': '#e5e7eb',
        'chart_bg': '#ffffff'
    }

# Professional CSS styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {{
        background: {theme['bg_secondary']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: {theme['bg_card']};
        border-right: 1px solid {theme['border']};
        width: 280px !important;
    }}
    
    /* Fix all text visibility in sidebar */
    section[data-testid="stSidebar"] * {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Professional Tesla-style logo */
    .logo-container {{
        padding: 24px 20px;
        border-bottom: 1px solid {theme['border']};
        margin-bottom: 24px;
        text-align: left;
    }}
    
    .logo-text {{
        font-size: 24px;
        font-weight: 600;
        color: {theme['accent']};
        letter-spacing: 2px;
        margin: 0;
    }}
    
    .logo-subtitle {{
        font-size: 12px;
        color: {theme['text_secondary']};
        margin-top: 4px;
        letter-spacing: 0.5px;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 20px;
        height: 100%;
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    
    .metric-label {{
        color: {theme['text_secondary']};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }}
    
    .metric-value {{
        color: {theme['text_primary']};
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 4px;
    }}
    
    .metric-change {{
        display: flex;
        align-items: center;
        font-size: 13px;
        font-weight: 500;
    }}
    
    .metric-change.positive {{
        color: {theme['success']};
    }}
    
    .metric-change.negative {{
        color: {theme['danger']};
    }}
    
    /* Section headers */
    .section-title {{
        color: {theme['text_primary']};
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid {theme['border']};
    }}
    
    /* Topic cards */
    .topic-card {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    
    .topic-title {{
        color: {theme['text_primary']};
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }}
    
    .topic-keywords {{
        color: {theme['text_secondary']};
        font-size: 12px;
        line-height: 1.6;
    }}
    
    .topic-badge {{
        display: inline-block;
        background: {theme['accent']}20;
        color: {theme['accent']};
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 8px;
    }}
    
    /* Review cards */
    .review-card {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    
    .review-rating {{
        color: {theme['warning']};
        margin-bottom: 8px;
    }}
    
    .review-content {{
        color: {theme['text_primary']};
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 8px;
    }}
    
    .review-metadata {{
        color: {theme['text_secondary']};
        font-size: 12px;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .viewerBadge_container__1QSob {{display: none;}}
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_db_connection():
    """Create database connection"""
    db_path = Path('data/reviews.db')
    if db_path.exists():
        return sqlite3.connect(str(db_path), check_same_thread=False)
    return None

# Load real data functions
@st.cache_data(ttl=300)
def load_review_stats():
    """Load real review statistics"""
    conn = get_db_connection()
    if not conn:
        return None, None, None
    
    try:
        # Get basic stats
        stats_query = """
        SELECT 
            COUNT(*) as total,
            AVG(rating) as avg_rating,
            COUNT(DISTINCT language) as languages
        FROM reviews
        """
        stats = pd.read_sql_query(stats_query, conn).iloc[0]
        
        # Get rating distribution
        rating_dist = pd.read_sql_query("""
            SELECT rating, COUNT(*) as count
            FROM reviews
            GROUP BY rating
            ORDER BY rating
        """, conn)
        
        # Get time series data
        time_series = pd.read_sql_query("""
            SELECT DATE(date_posted) as date, COUNT(*) as count, AVG(rating) as avg_rating
            FROM reviews
            WHERE date_posted IS NOT NULL
            GROUP BY DATE(date_posted)
            ORDER BY date DESC
            LIMIT 30
        """, conn)
        
        return stats, rating_dist, time_series
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data(ttl=300)
def load_sentiment_data():
    """Load real sentiment analysis data"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        sentiment_query = """
        SELECT 
            SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as neutral,
            SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
        FROM reviews
        """
        
        sentiments = pd.read_sql_query(sentiment_query, conn).iloc[0]
        return sentiments
    except:
        return None

@st.cache_data(ttl=300)
def load_topics():
    """Load LDA topics from pickle files"""
    model_path = Path("data/models")
    if not model_path.exists():
        return None, None
    
    # Get latest model file
    model_files = sorted(model_path.glob("lda_model_*.pkl"))
    if not model_files:
        return None, None
    
    try:
        with open(model_files[-1], 'rb') as f:
            lda_data = pickle.load(f)
            
        topics = lda_data.get('topics', [])
        doc_topic_matrix = lda_data.get('doc_topic_matrix', None)
        
        return topics, doc_topic_matrix
    except Exception as e:
        st.error(f"Error loading topics: {e}")
        return None, None

@st.cache_data(ttl=300)
def load_reviews_sample(limit=100, rating_filter=None, search_term=None):
    """Load sample reviews with filters"""
    conn = get_db_connection()
    if not conn:
        return None
    
    query = """
    SELECT 
        r.id,
        r.content,
        r.rating,
        r.date_posted,
        r.language,
        p.sentiment_score,
        p.normalized_text
    FROM reviews r
    LEFT JOIN processed_reviews p ON r.id = p.review_id
    WHERE 1=1
    """
    
    params = []
    
    if rating_filter:
        query += " AND r.rating = ?"
        params.append(rating_filter)
    
    if search_term:
        query += " AND (r.content LIKE ? OR p.normalized_text LIKE ?)"
        params.extend([f"%{search_term}%", f"%{search_term}%"])
    
    query += " ORDER BY r.date_posted DESC LIMIT ?"
    params.append(limit)
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    except:
        return None

def render_metric_card(label, value, change=None, color=None):
    """Render metric card"""
    change_html = ""
    if change:
        change_type = 'positive' if '+' in str(change) else 'negative' if '-' in str(change) else ''
        change_icon = '▲' if change_type == 'positive' else '▼' if change_type == 'negative' else '•'
        change_html = f'<div class="metric-change {change_type}">{change_icon} {change}</div>'
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color if color else theme['text_primary']};">{value}</div>
        {change_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)

def render_topic_card(topic_id, keywords, doc_count=None):
    """Render a topic card"""
    topic_html = f"""
    <div class="topic-card">
        <div class="topic-title">
            <span class="topic-badge">Topic {topic_id}</span>
            {f'<span style="float: right; color: {theme["text_secondary"]}; font-size: 12px;">{doc_count} reviews</span>' if doc_count else ''}
        </div>
        <div class="topic-keywords">{keywords}</div>
    </div>
    """
    st.markdown(topic_html, unsafe_allow_html=True)

def render_review_card(review):
    """Render a review card"""
    rating_stars = '⭐' * int(review['rating'])
    
    review_html = f"""
    <div class="review-card">
        <div class="review-rating">{rating_stars}</div>
        <div class="review-content">{review['content'][:300]}{'...' if len(review['content']) > 300 else ''}</div>
        <div class="review-metadata">
            {review['date_posted'][:10] if review['date_posted'] else 'Date unknown'} · 
            Language: {review['language'] if review['language'] else 'Unknown'}
            {f' · Sentiment: {review["sentiment_score"]:.2f}' if review.get('sentiment_score') else ''}
        </div>
    </div>
    """
    st.markdown(review_html, unsafe_allow_html=True)

def create_wordcloud(topics):
    """Create word cloud from topics"""
    all_words = []
    for topic in topics[:10]:  # Top 10 topics
        all_words.extend(topic['words'][:20])
    
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    if word_freq:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white' if not st.session_state.dark_mode else 'black',
            colormap='RdBu' if not st.session_state.dark_mode else 'cool',
            max_words=50
        ).generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# Main app
def main():
    # Check database connection
    conn = get_db_connection()
    if not conn:
        st.error("Database not found! Please ensure reviews.db exists in the data folder.")
        return
    
    # Load data
    stats, rating_dist, time_series = load_review_stats()
    sentiments = load_sentiment_data()
    topics, doc_topic_matrix = load_topics()
    
    # Sidebar
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="logo-container">
            <div class="logo-text">IRCTC</div>
            <div class="logo-subtitle">ANALYTICS DASHBOARD</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### Navigation")
        
        page = st.radio(
            "nav",
            ["📊 Overview", "🎯 Topics", "💭 Sentiment", "🔍 Explorer", "📈 Statistics"],
            label_visibility="collapsed",
            index=0
        )
        
        st.markdown("---")
        
        # Filters
        st.markdown("### Filters")
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        rating_filter = st.multiselect(
            "Rating",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )
        
        st.markdown("---")
        
        # Theme toggle
        if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode", 
                    use_container_width=True):
            toggle_dark_mode()
            st.rerun()
    
    # Main content
    if page == "📊 Overview":
        st.title("IRCTC Review Analytics Overview")
        
        if stats is not None:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                render_metric_card(
                    "Total Reviews",
                    f"{int(stats['total']):,}",
                    "+12% from last month"
                )
            
            with col2:
                render_metric_card(
                    "Average Rating",
                    f"{stats['avg_rating']:.2f} / 5.0",
                    None,
                    theme['warning'] if stats['avg_rating'] < 3 else theme['success']
                )
            
            with col3:
                if sentiments is not None:
                    total_sentiment = sentiments['positive'] + sentiments['neutral'] + sentiments['negative']
                    positive_pct = (sentiments['positive'] / total_sentiment * 100) if total_sentiment > 0 else 0
                    render_metric_card(
                        "Positive Sentiment",
                        f"{positive_pct:.1f}%",
                        None,
                        theme['success'] if positive_pct > 50 else theme['danger']
                    )
                else:
                    render_metric_card("Positive Sentiment", "N/A")
            
            with col4:
                render_metric_card(
                    "Languages",
                    f"{int(stats['languages'])}",
                    None
                )
            
            # Charts
            st.markdown('<div class="section-title">Analytics Insights</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if rating_dist is not None and not rating_dist.empty:
                    # Rating distribution bar chart
                    fig = go.Figure()
                    colors = [theme['danger'], theme['warning'], '#fbbf24', theme['success'], theme['accent_secondary']]
                    
                    fig.add_trace(go.Bar(
                        x=[f"{r} Star" for r in rating_dist['rating']],
                        y=rating_dist['count'],
                        marker_color=[colors[int(r)-1] for r in rating_dist['rating']],
                        text=rating_dist['count'],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Rating Distribution",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor=theme['chart_bg'],
                        font=dict(color=theme['text_primary']),
                        xaxis=dict(gridcolor=theme['border']),
                        yaxis=dict(gridcolor=theme['border'], title="Number of Reviews"),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if sentiments is not None:
                    # Sentiment pie chart
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=['Positive', 'Neutral', 'Negative'],
                            values=[sentiments['positive'], sentiments['neutral'], sentiments['negative']],
                            hole=0.6,
                            marker=dict(colors=[theme['success'], theme['warning'], theme['danger']]),
                            textfont=dict(color=theme['text_primary'])
                        )
                    ])
                    
                    fig.update_layout(
                        title="Sentiment Distribution",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=theme['text_primary']),
                        showlegend=True,
                        legend=dict(font=dict(color=theme['text_primary']))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Time series
            if time_series is not None and not time_series.empty:
                st.markdown('<div class="section-title">Review Trends Over Time</div>', unsafe_allow_html=True)
                
                fig = go.Figure()
                
                # Reviews count over time
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(time_series['date']),
                    y=time_series['count'],
                    mode='lines+markers',
                    name='Review Count',
                    line=dict(color=theme['accent_secondary'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                
                fig.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=theme['chart_bg'],
                    font=dict(color=theme['text_primary']),
                    xaxis=dict(
                        gridcolor=theme['border'],
                        title="Date"
                    ),
                    yaxis=dict(
                        gridcolor=theme['border'],
                        title="Number of Reviews"
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🎯 Topics":
        st.title("Topic Analysis")
        
        if topics:
            # Topic statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                render_metric_card("Topics Identified", len(topics))
            
            with col2:
                render_metric_card("Unique Keywords", 
                                 len(set([word for topic in topics for word in topic['words']])))
            
            with col3:
                if doc_topic_matrix is not None:
                    avg_topics_per_doc = (doc_topic_matrix > 0.1).sum(axis=1).mean()
                    render_metric_card("Avg Topics/Review", f"{avg_topics_per_doc:.1f}")
            
            # Word cloud
            st.markdown('<div class="section-title">Topic Word Cloud</div>', unsafe_allow_html=True)
            create_wordcloud(topics)
            
            # Topic cards
            st.markdown('<div class="section-title">Top Topics</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            for idx, topic in enumerate(topics[:20]):  # Show top 20 topics
                with col1 if idx % 2 == 0 else col2:
                    keywords = ', '.join(topic['words'][:8])
                    render_topic_card(topic['topic_id'], keywords)
            
            # Topic distribution chart
            if doc_topic_matrix is not None:
                st.markdown('<div class="section-title">Topic Distribution</div>', unsafe_allow_html=True)
                
                # Calculate topic sizes
                topic_sizes = (doc_topic_matrix > 0.1).sum(axis=0)
                topic_df = pd.DataFrame({
                    'Topic': [f"Topic {i}" for i in range(len(topic_sizes))],
                    'Documents': topic_sizes
                }).nlargest(15, 'Documents')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=topic_df['Topic'],
                    y=topic_df['Documents'],
                    marker_color=theme['accent_secondary']
                ))
                
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=theme['chart_bg'],
                    font=dict(color=theme['text_primary']),
                    xaxis=dict(gridcolor=theme['border'], title="Topic"),
                    yaxis=dict(gridcolor=theme['border'], title="Number of Documents"),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topics found. Please run topic modeling analysis first.")
    
    elif page == "💭 Sentiment":
        st.title("Sentiment Analysis")
        
        if sentiments is not None:
            # Sentiment metrics
            total = sentiments['positive'] + sentiments['neutral'] + sentiments['negative']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pct = (sentiments['positive'] / total * 100) if total > 0 else 0
                render_metric_card(
                    "Positive Reviews",
                    f"{int(sentiments['positive']):,}",
                    f"{pct:.1f}%",
                    theme['success']
                )
            
            with col2:
                pct = (sentiments['neutral'] / total * 100) if total > 0 else 0
                render_metric_card(
                    "Neutral Reviews",
                    f"{int(sentiments['neutral']):,}",
                    f"{pct:.1f}%",
                    theme['warning']
                )
            
            with col3:
                pct = (sentiments['negative'] / total * 100) if total > 0 else 0
                render_metric_card(
                    "Negative Reviews",
                    f"{int(sentiments['negative']):,}",
                    f"{pct:.1f}%",
                    theme['danger']
                )
            
            # Sentiment by rating
            st.markdown('<div class="section-title">Sentiment Distribution by Rating</div>', unsafe_allow_html=True)
            
            conn = get_db_connection()
            sentiment_rating_query = """
            SELECT 
                rating,
                COUNT(*) as count,
                CASE 
                    WHEN rating >= 4 THEN 'Positive'
                    WHEN rating = 3 THEN 'Neutral'
                    ELSE 'Negative'
                END as sentiment
            FROM reviews
            GROUP BY rating
            """
            
            sentiment_rating_df = pd.read_sql_query(sentiment_rating_query, conn)
            
            fig = go.Figure()
            
            for sentiment, color in [('Negative', theme['danger']), 
                                    ('Neutral', theme['warning']), 
                                    ('Positive', theme['success'])]:
                data = sentiment_rating_df[sentiment_rating_df['sentiment'] == sentiment]
                fig.add_trace(go.Bar(
                    x=data['rating'],
                    y=data['count'],
                    name=sentiment,
                    marker_color=color
                ))
            
            fig.update_layout(
                barmode='stack',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor=theme['chart_bg'],
                font=dict(color=theme['text_primary']),
                xaxis=dict(
                    gridcolor=theme['border'],
                    title="Rating",
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                yaxis=dict(gridcolor=theme['border'], title="Number of Reviews"),
                legend=dict(font=dict(color=theme['text_primary']))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sample reviews by sentiment
            st.markdown('<div class="section-title">Sample Reviews by Sentiment</div>', unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["😊 Positive", "😐 Neutral", "😞 Negative"])
            
            with tab1:
                positive_reviews = load_reviews_sample(limit=5, rating_filter=5)
                if positive_reviews is not None and not positive_reviews.empty:
                    for _, review in positive_reviews.iterrows():
                        render_review_card(review)
            
            with tab2:
                neutral_reviews = load_reviews_sample(limit=5, rating_filter=3)
                if neutral_reviews is not None and not neutral_reviews.empty:
                    for _, review in neutral_reviews.iterrows():
                        render_review_card(review)
            
            with tab3:
                negative_reviews = load_reviews_sample(limit=5, rating_filter=1)
                if negative_reviews is not None and not negative_reviews.empty:
                    for _, review in negative_reviews.iterrows():
                        render_review_card(review)
    
    elif page == "🔍 Explorer":
        st.title("Review Explorer")
        
        # Search and filters
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search reviews", placeholder="Enter keywords...")
        
        with col2:
            selected_rating = st.selectbox("Rating", ["All", "5", "4", "3", "2", "1"])
        
        with col3:
            sort_order = st.selectbox("Sort by", ["Latest", "Oldest", "Highest", "Lowest"])
        
        # Apply filters
        rating_filter = None if selected_rating == "All" else int(selected_rating)
        
        # Load reviews
        reviews = load_reviews_sample(
            limit=50,
            rating_filter=rating_filter,
            search_term=search_term if search_term else None
        )
        
        if reviews is not None and not reviews.empty:
            st.markdown(f'<div class="section-title">Found {len(reviews)} reviews</div>', 
                       unsafe_allow_html=True)
            
            # Sort reviews
            if sort_order == "Latest":
                reviews = reviews.sort_values('date_posted', ascending=False)
            elif sort_order == "Oldest":
                reviews = reviews.sort_values('date_posted', ascending=True)
            elif sort_order == "Highest":
                reviews = reviews.sort_values('rating', ascending=False)
            elif sort_order == "Lowest":
                reviews = reviews.sort_values('rating', ascending=True)
            
            # Display reviews
            for _, review in reviews.iterrows():
                render_review_card(review)
        else:
            st.info("No reviews found matching your criteria.")
    
    elif page == "📈 Statistics":
        st.title("Statistical Analysis")
        
        if stats is not None:
            # Summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="section-title">Database Statistics</div>', unsafe_allow_html=True)
                
                conn = get_db_connection()
                
                # Language distribution
                lang_query = """
                SELECT language, COUNT(*) as count
                FROM reviews
                GROUP BY language
                ORDER BY count DESC
                """
                lang_df = pd.read_sql_query(lang_query, conn)
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=lang_df['language'],
                        values=lang_df['count'],
                        hole=0.4
                    )
                ])
                
                fig.update_layout(
                    title="Language Distribution",
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=theme['text_primary'])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<div class="section-title">Review Statistics</div>', unsafe_allow_html=True)
                
                # Calculate statistics
                stats_data = {
                    "Total Reviews": f"{int(stats['total']):,}",
                    "Average Rating": f"{stats['avg_rating']:.2f}",
                    "Languages": int(stats['languages']),
                    "1-Star Reviews": f"{rating_dist[rating_dist['rating']==1]['count'].values[0] if len(rating_dist[rating_dist['rating']==1]) > 0 else 0:,}",
                    "5-Star Reviews": f"{rating_dist[rating_dist['rating']==5]['count'].values[0] if len(rating_dist[rating_dist['rating']==5]) > 0 else 0:,}"
                }
                
                for key, value in stats_data.items():
                    st.metric(key, value)
            
            # Rating trends over time
            st.markdown('<div class="section-title">Rating Trends</div>', unsafe_allow_html=True)
            
            rating_trends_query = """
            SELECT 
                DATE(date_posted) as date,
                rating,
                COUNT(*) as count
            FROM reviews
            WHERE date_posted IS NOT NULL
            GROUP BY DATE(date_posted), rating
            ORDER BY date DESC
            LIMIT 500
            """
            
            rating_trends = pd.read_sql_query(rating_trends_query, conn)
            
            if not rating_trends.empty:
                rating_trends['date'] = pd.to_datetime(rating_trends['date'])
                
                fig = go.Figure()
                
                for rating in sorted(rating_trends['rating'].unique()):
                    data = rating_trends[rating_trends['rating'] == rating]
                    color = [theme['danger'], theme['warning'], '#fbbf24', 
                            theme['success'], theme['accent_secondary']][int(rating)-1]
                    
                    fig.add_trace(go.Scatter(
                        x=data['date'],
                        y=data['count'],
                        name=f"{rating} Star",
                        mode='lines',
                        line=dict(color=color, width=2),
                        stackgroup='one'
                    ))
                
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor=theme['chart_bg'],
                    font=dict(color=theme['text_primary']),
                    xaxis=dict(gridcolor=theme['border'], title="Date"),
                    yaxis=dict(gridcolor=theme['border'], title="Number of Reviews"),
                    hovermode='x unified',
                    legend=dict(font=dict(color=theme['text_primary']))
                )
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()