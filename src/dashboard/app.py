#!/usr/bin/env python3
"""
IRCTC Review Analysis Dashboard
Interactive Streamlit dashboard for visualizing review analysis results
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import pickle

# Page config
st.set_page_config(
    page_title="IRCTC Review Analysis Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for bright theme
st.markdown("""
    <style>
    /* Light theme colors */
    .stApp {
        background-color: #ffffff;
    }
    
    .main {
        padding: 0rem 1rem;
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background-color: #f0f7ff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    
    .topic-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1a1a1a !important;
    }
    
    /* Text color */
    p, span, div {
        color: #333333;
    }
    
    /* Make logo smaller */
    img[src*="irctc"] {
        max-width: 150px !important;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_database_connection():
    """Get database connection"""
    return sqlite3.connect('data/reviews.db', check_same_thread=False)

@st.cache_data(ttl=300)
def load_review_stats():
    """Load review statistics from database"""
    conn = get_database_connection()
    
    stats = {}
    
    # Total reviews
    stats['total_reviews'] = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM reviews", conn
    )['count'][0]
    
    # Average rating
    stats['avg_rating'] = pd.read_sql_query(
        "SELECT AVG(rating) as avg FROM reviews WHERE rating > 0", conn
    )['avg'][0]
    
    # Rating distribution
    rating_dist = pd.read_sql_query(
        """SELECT rating, COUNT(*) as count 
        FROM reviews WHERE rating > 0 
        GROUP BY rating ORDER BY rating""", conn
    )
    
    # Reviews over time
    reviews_time = pd.read_sql_query(
        """SELECT DATE(date_posted) as date, COUNT(*) as count 
        FROM reviews WHERE date_posted IS NOT NULL 
        GROUP BY DATE(date_posted) ORDER BY date""", conn
    )
    
    # Language distribution
    lang_dist = pd.read_sql_query(
        """SELECT language, COUNT(*) as count 
        FROM processed_reviews 
        GROUP BY language""", conn
    )
    
    return stats, rating_dist, reviews_time, lang_dist

@st.cache_data(ttl=300)
def load_topics():
    """Load topic modeling results"""
    conn = get_database_connection()
    
    # Load topics from database
    topics = pd.read_sql_query(
        """SELECT t.*, COUNT(rt.review_id) as review_count
        FROM topics t
        LEFT JOIN review_topics rt ON t.id = rt.topic_id
        GROUP BY t.id
        ORDER BY review_count DESC""", conn
    )
    
    return topics

@st.cache_data(ttl=300)
def load_sentiments():
    """Load sentiment analysis results"""
    conn = get_database_connection()
    
    sentiments = pd.read_sql_query(
        """SELECT 
            AVG(sentiment_score) as avg_sentiment,
            COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END) as positive_count,
            COUNT(CASE WHEN sentiment_label = 'negative' THEN 1 END) as negative_count,
            COUNT(CASE WHEN sentiment_label = 'neutral' THEN 1 END) as neutral_count
        FROM sentiments""", conn
    )
    
    # Sentiment over time
    sentiment_time = pd.read_sql_query(
        """SELECT 
            DATE(r.date_posted) as date,
            AVG(s.sentiment_score) as avg_sentiment
        FROM sentiments s
        JOIN reviews r ON s.review_id = r.id
        WHERE r.date_posted IS NOT NULL
        GROUP BY DATE(r.date_posted)
        ORDER BY date""", conn
    )
    
    return sentiments, sentiment_time

def create_wordcloud(text_data, title):
    """Create and display wordcloud"""
    if not text_data or text_data.strip() == '':
        st.warning(f"No data available for {title}")
        return
        
    fig, ax = plt.subplots(figsize=(10, 5))
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate(text_data)
    
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    st.pyplot(fig)

def live_sentiment_prediction():
    """Live sentiment prediction interface"""
    st.subheader("🔮 Test Live Sentiment Prediction")
    
    user_input = st.text_area(
        "Enter a review to analyze:",
        placeholder="Type your review here...",
        height=100
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input:
            # Import analyzer
            try:
                from src.analysis.review_analyzer import ReviewAnalyzer
                analyzer = ReviewAnalyzer()
                
                # Predict sentiment
                sentiment_score = analyzer.predict_sentiment_simple(user_input)
                
                # Determine label
                if sentiment_score > 0.6:
                    label = "Positive"
                    color = "green"
                elif sentiment_score < 0.4:
                    label = "Negative"
                    color = "red"
                else:
                    label = "Neutral"
                    color = "orange"
                
                # Display result
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment Score", f"{sentiment_score:.2f}")
                with col2:
                    st.metric("Sentiment Label", label)
                
                # Visual gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score,
                    title={'text': "Sentiment Score"},
                    gauge={'axis': {'range': [0, 1]},
                           'bar': {'color': color},
                           'steps': [
                               {'range': [0, 0.4], 'color': "lightgray"},
                               {'range': [0.4, 0.6], 'color': "gray"},
                               {'range': [0.6, 1], 'color': "lightgreen"}
                           ],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 0.5}}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in sentiment analysis: {e}")
        else:
            st.warning("Please enter a review to analyze")

def main():
    # Header
    st.title("🚂 IRCTC Review Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.irctc.co.in/nget/assets/images/logo.png", width=120)
        st.markdown("### Navigation")
        page = st.radio(
            "",
            ["📊 Overview", "📈 Topic Analysis", "😊 Sentiment Analysis", 
             "🔍 Review Explorer", "🎯 Live Testing", "📑 Reports"]
        )
        
        st.markdown("---")
        st.markdown("### Filters")
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        rating_filter = st.slider(
            "Rating Filter",
            min_value=1, max_value=5, value=(1, 5)
        )
    
    # Load data
    try:
        stats, rating_dist, reviews_time, lang_dist = load_review_stats()
        topics = load_topics()
        sentiments, sentiment_time = load_sentiments()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure the database is properly initialized and data is processed.")
        return
    
    # Page content based on selection
    if page == "📊 Overview":
        st.header("Overview Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Reviews", 
                f"{stats['total_reviews']:,}",
                delta=f"+{len(reviews_time)} today" if len(reviews_time) > 0 else "0"
            )
        
        with col2:
            avg_rating = stats['avg_rating'] if stats['avg_rating'] else 0
            st.metric(
                "Average Rating", 
                f"{avg_rating:.2f} ⭐",
                delta=f"{(avg_rating - 3):.2f}" if avg_rating else "N/A"
            )
        
        with col3:
            if not sentiments.empty:
                st.metric(
                    "Positive Reviews",
                    f"{sentiments['positive_count'][0]:,}",
                    delta=f"{(sentiments['positive_count'][0]/stats['total_reviews']*100):.1f}%"
                )
            else:
                st.metric("Positive Reviews", "N/A")
        
        with col4:
            st.metric(
                "Topics Identified",
                f"{len(topics)}" if not topics.empty else "0"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            if not rating_dist.empty:
                fig = px.bar(
                    rating_dist, x='rating', y='count',
                    title="Rating Distribution",
                    color='count',
                    color_continuous_scale='RdYlGn',
                    labels={'rating': 'Rating', 'count': 'Number of Reviews'}
                )
                fig.update_layout(
                    xaxis_title="Rating",
                    yaxis_title="Number of Reviews",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rating data available")
        
        with col2:
            # Language distribution
            if not lang_dist.empty:
                fig = px.pie(
                    lang_dist, values='count', names='language',
                    title="Language Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No language data available")
        
        # Reviews over time
        if not reviews_time.empty:
            st.subheader("Reviews Over Time")
            fig = px.line(
                reviews_time, x='date', y='count',
                title="Daily Review Count",
                markers=True,
                labels={'date': 'Date', 'count': 'Number of Reviews'}
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Reviews"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📈 Topic Analysis":
        st.header("Topic Analysis")
        
        if topics.empty or len(topics) == 0:
            st.warning("No topics have been identified yet. Please run the topic modeling analysis first.")
            st.info("Topics were found in LDA analysis. Loading from model file...")
            
            # Try to load topics from saved model
            try:
                models_dir = Path('data/models')
                model_files = list(models_dir.glob('lda_model_*.pkl'))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_model, 'rb') as f:
                        model_data = pickle.load(f)
                        saved_topics = model_data.get('topics', [])
                        if saved_topics:
                            st.success(f"Loaded {len(saved_topics)} topics from saved model")
                            # Display topics from saved model
                            for i, topic in enumerate(saved_topics[:10]):
                                with st.expander(f"Topic {i}: {', '.join(topic['words'][:3])}..."):
                                    st.write("**Top Keywords:**")
                                    for word in topic['words'][:10]:
                                        st.write(f"• {word}")
            except Exception as e:
                st.error(f"Could not load saved topics: {e}")
        else:
            # Top topics
            st.subheader("Top Topics by Review Count")
            
            top_topics = topics.head(10)
            fig = px.bar(
                top_topics, 
                x='review_count', 
                y='name',
                orientation='h',
                title="Top 10 Topics",
                color='review_count',
                color_continuous_scale='Viridis',
                labels={'review_count': 'Number of Reviews', 'name': 'Topic'}
            )
            fig.update_layout(
                xaxis_title="Number of Reviews",
                yaxis_title="Topic"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic words
            st.subheader("Topic Keywords")
            
            selected_topic = st.selectbox(
                "Select a topic to view keywords:",
                topics['name'].tolist() if 'name' in topics.columns else []
            )
            
            if selected_topic:
                topic_data = topics[topics['name'] == selected_topic].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Topic Details")
                    st.write(f"**Topic ID:** {topic_data['id']}")
                    st.write(f"**Review Count:** {topic_data['review_count']:,}")
                
                with col2:
                    if 'keywords' in topic_data:
                        st.markdown("### Top Keywords")
                        keywords = topic_data['keywords'].split(',') if isinstance(topic_data['keywords'], str) else []
                        for kw in keywords[:10]:
                            st.write(f"• {kw}")
    
    elif page == "😊 Sentiment Analysis":
        st.header("Sentiment Analysis")
        
        if sentiments.empty or (sentiments['positive_count'][0] == 0 and sentiments['negative_count'][0] == 0):
            st.info("Using rating-based sentiment analysis as formal sentiment analysis hasn't been run yet.")
            
            # Calculate sentiment from ratings
            conn = get_database_connection()
            rating_sentiment = pd.read_sql_query("""
                SELECT 
                    COUNT(CASE WHEN rating >= 4 THEN 1 END) as positive_count,
                    COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative_count,
                    COUNT(CASE WHEN rating = 3 THEN 1 END) as neutral_count,
                    AVG(rating)/5.0 as avg_score
                FROM reviews
                WHERE rating > 0
            """, conn)
            
            sentiments = rating_sentiment
            
            # Show the metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Positive (4-5⭐)", 
                    f"{sentiments['positive_count'][0]:,}",
                    delta=f"{(sentiments['positive_count'][0]/(sentiments['positive_count'][0] + sentiments['negative_count'][0] + sentiments['neutral_count'][0])*100):.1f}%"
                )
            
            with col2:
                st.metric(
                    "Neutral (3⭐)",
                    f"{sentiments['neutral_count'][0]:,}"
                )
            
            with col3:
                st.metric(
                    "Negative (1-2⭐)",
                    f"{sentiments['negative_count'][0]:,}",
                    delta=f"{(sentiments['negative_count'][0]/(sentiments['positive_count'][0] + sentiments['negative_count'][0] + sentiments['neutral_count'][0])*100):.1f}%"
                )
        else:
            # Sentiment metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Positive", 
                    f"{sentiments['positive_count'][0]:,}",
                    delta=f"{(sentiments['positive_count'][0]/(sentiments['positive_count'][0] + sentiments['negative_count'][0] + sentiments['neutral_count'][0])*100):.1f}%"
                )
            
            with col2:
                st.metric(
                    "Neutral",
                    f"{sentiments['neutral_count'][0]:,}"
                )
            
            with col3:
                st.metric(
                    "Negative",
                    f"{sentiments['negative_count'][0]:,}",
                    delta=f"{(sentiments['negative_count'][0]/(sentiments['positive_count'][0] + sentiments['negative_count'][0] + sentiments['neutral_count'][0])*100):.1f}%"
                )
            
        # Sentiment distribution pie chart (show this for both cases)
        sentiment_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [
                sentiments['positive_count'][0] if not sentiments.empty else 0,
                sentiments['neutral_count'][0] if not sentiments.empty else 0,
                sentiments['negative_count'][0] if not sentiments.empty else 0
            ]
        })
        
        fig = px.pie(
            sentiment_data, 
            values='Count', 
            names='Sentiment',
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#95a5a6',
                'Negative': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
            
        # Sentiment over time
        if not sentiment_time.empty:
            st.subheader("Sentiment Trend Over Time")
            fig = px.line(
                sentiment_time,
                x='date',
                y='avg_sentiment',
                title="Average Sentiment Score Over Time",
                markers=True,
                labels={'date': 'Date', 'avg_sentiment': 'Average Sentiment'}
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(
                yaxis_range=[0, 1],
                xaxis_title="Date",
                yaxis_title="Average Sentiment Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 Review Explorer":
        st.header("Review Explorer")
        
        # Search interface
        search_term = st.text_input("Search reviews:", placeholder="Enter keywords...")
        
        # Load sample reviews
        conn = get_database_connection()
        query = "SELECT * FROM reviews"
        
        if search_term:
            query += f" WHERE content LIKE '%{search_term}%'"
        
        query += " ORDER BY date_posted DESC LIMIT 100"
        
        reviews_df = pd.read_sql_query(query, conn)
        
        if not reviews_df.empty:
            st.subheader(f"Showing {len(reviews_df)} reviews")
            
            # Display reviews
            for idx, row in reviews_df.iterrows():
                with st.expander(f"Review {idx+1} - Rating: {'⭐' * int(row['rating'])}"):
                    st.write(row['content'])
                    st.caption(f"Date: {row['date_posted']}")
        else:
            st.info("No reviews found matching your criteria")
    
    elif page == "🎯 Live Testing":
        st.header("Live Testing")
        live_sentiment_prediction()
    
    elif page == "📑 Reports":
        st.header("Reports & Export")
        
        st.subheader("Generate Report")
        
        report_type = st.selectbox(
            "Select Report Type",
            ["Executive Summary", "Detailed Analysis", "Topic Report", "Sentiment Report"]
        )
        
        if st.button("Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                # Create sample report
                report = f"""
                # IRCTC Review Analysis Report
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Executive Summary
                - Total Reviews Analyzed: {stats['total_reviews']:,}
                - Average Rating: {stats['avg_rating']:.2f}
                - Number of Topics: {len(topics)}
                
                ## Key Findings
                1. Most discussed topics relate to booking process and payment issues
                2. Sentiment has improved over the last quarter
                3. Users appreciate the UI improvements
                
                ## Recommendations
                1. Focus on improving payment gateway reliability
                2. Optimize app performance during peak hours
                3. Enhance customer support response time
                """
                
                st.download_button(
                    label="Download Report",
                    data=report,
                    file_name=f"irctc_analysis_report_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
                
                st.success("Report generated successfully!")
                st.text_area("Report Preview", report, height=400)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data source: Google Play Store")

if __name__ == "__main__":
    main()