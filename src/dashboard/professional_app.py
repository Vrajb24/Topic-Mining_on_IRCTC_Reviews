#!/usr/bin/env python3
"""
Professional IRCTC Review Analysis Dashboard
Clean, modern UI inspired by Tesla analytics dashboard
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

# Professional color schemes - Force dark mode for better visibility
theme = {
    'bg_primary': '#0e1117',
    'bg_secondary': '#1a1d24',
    'bg_card': '#1e2329',
    'text_primary': '#ffffff',
    'text_secondary': '#b8bcc8',
    'accent': '#3e7bfa',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'border': '#2d3139',
    'chart_bg': '#1e2329'
}

# CSS for professional styling
st.markdown(f"""
<style>
    /* Main app styling */
    .stApp {{
        background: {theme['bg_secondary']} !important;
    }}
    
    /* Main container */
    .main > div {{
        background: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: {theme['bg_card']};
        border-right: 1px solid {theme['border']};
    }}
    
    /* Fix text visibility in sidebar */
    section[data-testid="stSidebar"] .stRadio > label {{
        color: {theme['text_primary']} !important;
        font-weight: 500;
    }}
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Professional metric cards */
    .metric-card {{
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        transition: transform 0.2s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .metric-label {{
        color: {theme['text_secondary']};
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }}
    
    .metric-value {{
        color: {theme['text_primary']};
        font-size: 32px;
        font-weight: 700;
        line-height: 1.2;
    }}
    
    .metric-change {{
        display: flex;
        align-items: center;
        margin-top: 8px;
        font-size: 14px;
        font-weight: 500;
    }}
    
    .metric-change.positive {{
        color: {theme['success']};
    }}
    
    .metric-change.negative {{
        color: {theme['danger']};
    }}
    
    /* Navigation items */
    .nav-link {{
        display: flex;
        align-items: center;
        padding: 10px 16px;
        margin: 4px 0;
        border-radius: 8px;
        color: {theme['text_primary']};
        text-decoration: none;
        transition: all 0.2s;
        cursor: pointer;
    }}
    
    .nav-link:hover {{
        background: {theme['bg_secondary']};
        transform: translateX(2px);
    }}
    
    .nav-link.active {{
        background: rgba(62, 123, 250, 0.2);
        border-left: 3px solid {theme['accent']};
    }}
    
    /* Section headers */
    .section-title {{
        color: {theme['text_primary']};
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid {theme['border']};
    }}
    
    /* Progress bars */
    .progress-container {{
        background: {theme['bg_card']} !important;
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }}
    
    .progress-header {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }}
    
    .progress-label {{
        color: {theme['text_primary']};
        font-weight: 500;
        font-size: 14px;
    }}
    
    .progress-value {{
        color: {theme['text_secondary']};
        font-size: 14px;
    }}
    
    .progress-bar {{
        background: {theme['bg_secondary']};
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }}
    
    .progress-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }}
    
    /* Logo section */
    .logo-container {{
        padding: 20px;
        text-align: center;
        border-bottom: 1px solid {theme['border']};
        margin-bottom: 20px;
    }}
    
    .logo-img {{
        max-width: 150px;
        margin-bottom: 10px;
    }}
    
    /* Icon styling */
    .nav-icon {{
        width: 20px;
        height: 20px;
        margin-right: 12px;
        fill: {theme['text_primary']};
    }}
    
    /* Dark mode toggle */
    .theme-toggle {{
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 999;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Override Streamlit default backgrounds */
    .stTabs {{
        background-color: transparent !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
        background: {theme['bg_card']} !important;
        padding: 0.75rem;
        border-radius: 0.75rem;
        border: 1px solid {theme['border']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {theme['text_secondary']} !important;
        border: none !important;
        padding: 0.5rem 1.5rem;
        border-radius: 0.5rem;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {theme['accent']} !important;
        color: white !important;
    }}
    
    /* Streamlit containers */
    div[data-testid="stHorizontalBlock"] > div {{
        background: {theme['bg_card']} !important;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid {theme['border']};
    }}
    
    /* All text elements */
    .stMarkdown, .stText {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Metrics override */
    [data-testid="metric-container"] {{
        background: {theme['bg_card']} !important;
        border: 1px solid {theme['border']};
        border-radius: 8px;
        padding: 1rem;
    }}
    
    [data-testid="metric-container"] label {{
        color: {theme['text_secondary']} !important;
    }}
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Charts background */
    .js-plotly-plot .plotly {{
        background: {theme['bg_card']} !important;
    }}
    
    /* Columns background */
    .element-container {{
        background: transparent !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: {theme['bg_card']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']};
    }}
    
    .streamlit-expanderContent {{
        background: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border']};
    }}
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {{
        background: {theme['bg_card']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border']};
    }}
    
    /* Data tables */
    .dataframe {{
        background: {theme['bg_card']} !important;
    }}
    
    .dataframe th {{
        background: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
    }}
    
    .dataframe td {{
        background: {theme['bg_card']} !important;
        color: {theme['text_primary']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_db_connection():
    """Create database connection"""
    return sqlite3.connect('data/reviews.db', check_same_thread=False)

# Load data functions
@st.cache_data(ttl=300)
def load_review_stats():
    """Load review statistics"""
    conn = get_db_connection()
    
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
        WHERE date_posted >= date('now', '-30 days')
        GROUP BY DATE(date_posted)
        ORDER BY date
    """, conn)
    
    return stats, rating_dist, time_series

@st.cache_data(ttl=300)
def load_sentiment_data():
    """Load sentiment analysis data"""
    conn = get_db_connection()
    
    sentiment_query = """
    SELECT 
        SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
        SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as neutral,
        SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
    FROM reviews
    """
    
    sentiments = pd.read_sql_query(sentiment_query, conn).iloc[0]
    return sentiments

@st.cache_data(ttl=300)
def load_department_data():
    """Load department segregation data"""
    conn = get_db_connection()
    
    dept_query = """
    SELECT 
        rc.department,
        COUNT(*) as count,
        AVG(r.rating) as avg_rating
    FROM reviews r
    LEFT JOIN review_classifications rc ON r.id = rc.review_id
    WHERE rc.department IN ('app', 'railway')
    GROUP BY rc.department
    """
    
    return pd.read_sql_query(dept_query, conn)

@st.cache_data(ttl=300)
def load_topics():
    """Load topic modeling results"""
    try:
        # Load the improved topics data
        with open('data/models/improved_topics.pkl', 'rb') as f:
            data = pickle.load(f)
        
        # Create topics dataframe from app and railway topics
        topics_data = []
        
        # Add app topics
        for topic in data.get('topics', {}).get('app', [])[:5]:
            keywords = [w[0] for w in topic.get('words', [])[:5]]
            topics_data.append({
                'topic_id': f"APP-{topic.get('topic_id', 0)}",
                'department': 'App',
                'keywords': ', '.join(keywords),
                'category': topic.get('category', 'uncategorized'),
                'relevance': topic.get('relevance_score', 0),
                'doc_count': topic.get('doc_count', 0)
            })
        
        # Add railway topics
        for topic in data.get('topics', {}).get('railway', [])[:5]:
            keywords = [w[0] for w in topic.get('words', [])[:5]]
            topics_data.append({
                'topic_id': f"RAIL-{topic.get('topic_id', 0)}",
                'department': 'Railway',
                'keywords': ', '.join(keywords),
                'category': topic.get('category', 'uncategorized'),
                'relevance': topic.get('relevance_score', 0),
                'doc_count': topic.get('doc_count', 0)
            })
        
        return pd.DataFrame(topics_data)
    except Exception as e:
        st.error(f"Error loading topics: {e}")
        return pd.DataFrame()

def hex_to_rgba(hex_color, alpha=0.2):
    """Convert hex color to rgba format"""
    if hex_color.endswith('20'):  # Handle legacy format
        hex_color = hex_color[:-2]
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

def create_mini_chart(data, color, chart_type='line'):
    """Create mini chart for metric cards"""
    fig = go.Figure()
    
    if chart_type == 'line':
        fig.add_trace(go.Scatter(
            y=data,
            mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=hex_to_rgba(color, 0.2),
            showlegend=False,
            hoverinfo='skip'
        ))
    elif chart_type == 'bar':
        fig.add_trace(go.Bar(
            y=data,
            marker_color=color,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#1e2329',
        plot_bgcolor='#1e2329',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    
    return fig

def render_metric_card(label, value, change=None, change_type='neutral', chart_data=None, chart_color='#3e7bfa'):
    """Render a professional metric card"""
    change_icon = '↑' if change_type == 'positive' else '↓' if change_type == 'negative' else '→'
    change_class = 'positive' if change_type == 'positive' else 'negative' if change_type == 'negative' else ''
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-change {change_class}">{change_icon} {change}</div>' if change else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    if chart_data is not None:
        fig = create_mini_chart(chart_data, chart_color)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_progress_bar(label, value, max_value=100, color='#3e7bfa'):
    """Render a progress bar"""
    percentage = (value / max_value) * 100
    
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-header">
            <span class="progress-label">{label}</span>
            <span class="progress-value">{value}/{max_value} ({percentage:.1f}%)</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {percentage}%; background: {color};"></div>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

# Main app
def main():
    # Load data
    try:
        stats, rating_dist, time_series = load_review_stats()
        sentiments = load_sentiment_data()
        topics = load_topics()
        dept_data = load_department_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the database is initialized with review data.")
        return
    
    # Sidebar
    with st.sidebar:
        # Logo placeholder
        st.markdown("""
        <div class="logo-container">
            <div style="width: 150px; height: 60px; background: rgba(62, 123, 250, 0.2); border: 2px solid #3e7bfa; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin: 0 auto;">
                <span style="color: #3e7bfa; font-weight: bold;">IRCTC LOGO</span>
            </div>
            <h3 style="margin-top: 10px; color: var(--text-primary);">Analytics Dashboard</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Navigation")
        
        # Professional navigation without emojis
        page = st.radio(
            "nav",
            ["Reports", "Topics", "Sentiment", "Explorer", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("### Support")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Started", use_container_width=True):
                st.info("Opening documentation...")
        
        with col2:
            if st.button("Settings", use_container_width=True):
                st.info("Opening settings...")
    
    # Dark mode toggle
    col_header = st.columns([6, 1])
    with col_header[1]:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", 
                     key="theme_toggle",
                     help="Toggle dark/light mode"):
            toggle_dark_mode()
            st.rerun()
    
    # Main content area
    if page == "Reports":
        # Header
        st.markdown('<h1 style="color: var(--text-primary); margin-bottom: 30px;">Reports Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Filters row
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            timeframe = st.selectbox("Timeframe", ["All-time", "Last 30 days", "Last 7 days", "Today"])
        with col2:
            people = st.selectbox("People", ["All", "Active Users", "New Users"])
        with col3:
            topic_filter = st.selectbox("Topic", ["All", "Top Topics", "Trending"])
        with col4:
            if st.button("↓ Download", use_container_width=True):
                st.info("Generating report...")
        
        st.markdown("---")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reviews = int(stats['total'])
            render_metric_card(
                "Total Reviews",
                f"{total_reviews:,}",
                f"{int(stats['languages'])} languages",
                "positive",
                np.random.randint(80000, 90000, 10),
                theme['accent']
            )
        
        with col2:
            app_count = dept_data[dept_data['department'] == 'app']['count'].values[0] if len(dept_data[dept_data['department'] == 'app']) > 0 else 0
            render_metric_card(
                "App Issues",
                f"{app_count:,}",
                f"{(app_count/total_reviews*100):.1f}% of total",
                "negative",
                np.random.randint(30000, 35000, 10),
                '#8b5cf6'
            )
        
        with col3:
            railway_count = dept_data[dept_data['department'] == 'railway']['count'].values[0] if len(dept_data[dept_data['department'] == 'railway']) > 0 else 0
            render_metric_card(
                "Railway Issues",
                f"{railway_count:,}",
                f"{(railway_count/total_reviews*100):.1f}% of total",
                "negative",
                np.random.randint(7000, 9000, 10),
                '#10b981'
            )
        
        with col4:
            avg_rating = float(stats['avg_rating']) if stats['avg_rating'] else 0
            render_metric_card(
                "Avg Rating",
                f"{avg_rating:.2f} ⭐",
                "Critical" if avg_rating < 3 else "Good",
                "negative" if avg_rating < 3 else "positive",
                np.random.uniform(1, 3, 10),
                theme['danger'] if avg_rating < 3 else theme['success']
            )
        
        # Charts section
        st.markdown('<div class="section-title">Analytics Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Activity chart
            if not time_series.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=time_series['date'],
                    y=time_series['count'],
                    marker_color=theme['accent'],
                    name='Daily Reviews'
                ))
                
                fig.update_layout(
                    title="Activity Over Time",
                    height=300,
                    paper_bgcolor=theme['bg_card'],
                    plot_bgcolor=theme['bg_card'],
                    font=dict(color=theme['text_primary']),
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis=dict(
                        showgrid=False,
                        gridcolor=theme['border']
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor=theme['border']
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating distribution
            if not rating_dist.empty:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=[f"{r} Star" for r in rating_dist['rating']],
                        values=rating_dist['count'],
                        hole=0.4,
                        marker=dict(colors=[theme['danger'], theme['warning'], '#fbbf24', theme['success'], theme['accent']])
                    )
                ])
                
                fig.update_layout(
                    title="Rating Distribution",
                    height=300,
                    paper_bgcolor=theme['bg_card'],
                    plot_bgcolor=theme['bg_card'],
                    font=dict(color=theme['text_primary']),
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Progress section - Department Comparison
        st.markdown('<div class="section-title">Department Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Issue Categories")
            if not dept_data.empty:
                for _, dept in dept_data.iterrows():
                    color = '#8b5cf6' if dept['department'] == 'app' else '#10b981'
                    render_progress_bar(
                        f"{dept['department'].title()} Issues",
                        int(dept['count']),
                        total_reviews,
                        color
                    )
        
        with col2:
            st.markdown("#### Sentiment Distribution")
            render_progress_bar("Negative Reviews", int(sentiments['negative']), total_reviews, theme['danger'])
            render_progress_bar("Neutral Reviews", int(sentiments['neutral']), total_reviews, theme['warning'])
            render_progress_bar("Positive Reviews", int(sentiments['positive']), total_reviews, theme['success'])
    
    elif page == "Topics":
        st.markdown('<h1 style="color: var(--text-primary);">Topic Analysis</h1>', unsafe_allow_html=True)
        
        if not topics.empty:
            # Department split
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### App Issues Topics")
                app_topics = topics[topics['department'] == 'App']
                if not app_topics.empty:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=app_topics['category'],
                            y=app_topics['doc_count'],
                            marker_color='#8b5cf6',
                            text=app_topics['doc_count'],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="App Issue Distribution",
                        height=300,
                        paper_bgcolor=theme['bg_card'],
                        plot_bgcolor=theme['bg_card'],
                        font=dict(color=theme['text_primary']),
                        xaxis_title="Category",
                        yaxis_title="Document Count",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Railway Issues Topics")
                railway_topics = topics[topics['department'] == 'Railway']
                if not railway_topics.empty:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=railway_topics['category'],
                            y=railway_topics['doc_count'],
                            marker_color='#10b981',
                            text=railway_topics['doc_count'],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Railway Issue Distribution",
                        height=300,
                        paper_bgcolor=theme['bg_card'],
                        plot_bgcolor=theme['bg_card'],
                        font=dict(color=theme['text_primary']),
                        xaxis_title="Category",
                        yaxis_title="Document Count",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Topic keywords
            st.markdown('<div class="section-title">Topic Details</div>', unsafe_allow_html=True)
            
            for _, topic in topics.iterrows():
                icon = "📱" if topic['department'] == 'App' else "🚂"
                with st.expander(f"{icon} {topic['category'].replace('_', ' ').title()}", expanded=False):
                    st.write(f"**Department:** {topic['department']}")
                    st.write(f"**Keywords:** {topic['keywords']}")
                    st.write(f"**Relevance Score:** {topic['relevance']:.2f}")
                    st.write(f"**Document Count:** {topic['doc_count']:,}")
        else:
            st.info("No topic analysis available. Please run topic modeling first.")
    
    elif page == "Sentiment":
        st.markdown('<h1 style="color: var(--text-primary);">Sentiment Analysis</h1>', unsafe_allow_html=True)
        
        # Sentiment metrics
        col1, col2, col3 = st.columns(3)
        
        total = sentiments['positive'] + sentiments['neutral'] + sentiments['negative']
        
        with col1:
            render_metric_card(
                "Positive Reviews",
                f"{int(sentiments['positive']):,}",
                f"{(sentiments['positive']/total*100):.1f}%",
                "positive"
            )
        
        with col2:
            render_metric_card(
                "Neutral Reviews",
                f"{int(sentiments['neutral']):,}",
                f"{(sentiments['neutral']/total*100):.1f}%",
                "neutral"
            )
        
        with col3:
            render_metric_card(
                "Negative Reviews",
                f"{int(sentiments['negative']):,}",
                f"{(sentiments['negative']/total*100):.1f}%",
                "negative"
            )
        
        # Sentiment pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[sentiments['positive'], sentiments['neutral'], sentiments['negative']],
                hole=0.4,
                marker=dict(colors=[theme['success'], theme['warning'], theme['danger']])
            )
        ])
        
        fig.update_layout(
            title="Sentiment Distribution",
            height=400,
            paper_bgcolor=theme['bg_card'],
            plot_bgcolor=theme['bg_card'],
            font=dict(color=theme['text_primary']),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Explorer":
        st.markdown('<h1 style="color: var(--text-primary);">Review Explorer</h1>', unsafe_allow_html=True)
        
        # Search and filters
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            search = st.text_input("Search reviews", placeholder="Enter keywords...")
        with col2:
            rating_filter = st.selectbox("Rating", ["All", "5", "4", "3", "2", "1"])
        with col3:
            sort_by = st.selectbox("Sort by", ["Latest", "Oldest", "Highest", "Lowest"])
        
        # Display sample reviews
        conn = get_db_connection()
        query = "SELECT * FROM reviews LIMIT 10"
        reviews_df = pd.read_sql_query(query, conn)
        
        if not reviews_df.empty:
            for _, review in reviews_df.iterrows():
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(f"**{'⭐' * int(review['rating'])}**")
                        st.write(review['content'][:200] + "..." if len(str(review['content'])) > 200 else review['content'])
                    with col2:
                        st.caption(f"Rating: {review['rating']}")
                        st.caption(str(review['date_posted'])[:10] if review['date_posted'] else "N/A")
                    st.markdown("---")
        else:
            st.info("No reviews found.")
    
    elif page == "Settings":
        st.markdown('<h1 style="color: var(--text-primary);">Settings</h1>', unsafe_allow_html=True)
        
        st.markdown("### Display Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Enable notifications", value=True)
            st.checkbox("Show tooltips", value=True)
            st.checkbox("Auto-refresh data", value=False)
        
        with col2:
            refresh_rate = st.select_slider(
                "Refresh rate (seconds)",
                options=[30, 60, 120, 300, 600],
                value=300
            )
            
            chart_style = st.selectbox(
                "Chart style",
                ["Default", "Minimal", "Detailed"]
            )
        
        st.markdown("### Data Settings")
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")
        
        if st.button("Reload Database"):
            st.rerun()

if __name__ == "__main__":
    main()