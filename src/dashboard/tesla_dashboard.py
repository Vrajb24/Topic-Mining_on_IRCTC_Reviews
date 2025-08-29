#!/usr/bin/env python3
"""
Tesla-Inspired Analytics Dashboard
Professional UI with modern design elements
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

# Page config
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
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
    
    section[data-testid="stSidebar"] .stRadio > label {{
        color: {theme['text_primary']} !important;
        font-weight: 500;
        font-size: 14px;
    }}
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: {theme['text_primary']} !important;
    }}
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {{
        color: {theme['text_primary']} !important;
    }}
    
    /* Radio button styling */
    section[data-testid="stSidebar"] .stRadio > div {{
        background: transparent !important;
    }}
    
    section[data-testid="stSidebar"] .stRadio > div > label > div {{
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
    
    /* Navigation items */
    .nav-section {{
        padding: 0 16px;
        margin-bottom: 24px;
    }}
    
    .nav-title {{
        color: {theme['text_secondary']};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        padding-left: 4px;
    }}
    
    .nav-item {{
        display: flex;
        align-items: center;
        padding: 10px 12px;
        margin: 4px 0;
        border-radius: 6px;
        color: {theme['text_primary']};
        text-decoration: none;
        transition: all 0.2s;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
    }}
    
    .nav-item:hover {{
        background: {theme['bg_secondary']};
    }}
    
    .nav-item.active {{
        background: {theme['accent']}15;
        color: {theme['accent']};
    }}
    
    .nav-icon {{
        width: 18px;
        height: 18px;
        margin-right: 12px;
        opacity: 0.8;
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
    
    .metric-change.neutral {{
        color: {theme['text_secondary']};
    }}
    
    /* Section headers */
    .section-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid {theme['border']};
    }}
    
    .section-title {{
        color: {theme['text_primary']};
        font-size: 18px;
        font-weight: 600;
    }}
    
    /* Progress bars */
    .progress-item {{
        display: flex;
        align-items: center;
        padding: 12px 16px;
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        margin-bottom: 8px;
    }}
    
    .progress-icon {{
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 16px;
        font-size: 20px;
    }}
    
    .progress-content {{
        flex: 1;
    }}
    
    .progress-title {{
        color: {theme['text_primary']};
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 4px;
    }}
    
    .progress-bar-container {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .progress-bar {{
        flex: 1;
        height: 6px;
        background: {theme['bg_secondary']};
        border-radius: 3px;
        overflow: hidden;
    }}
    
    .progress-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease;
    }}
    
    .progress-percent {{
        color: {theme['text_secondary']};
        font-size: 13px;
        font-weight: 500;
        min-width: 45px;
        text-align: right;
    }}
    
    /* Leaderboard items */
    .leaderboard-item {{
        display: flex;
        align-items: center;
        padding: 12px;
        background: {theme['bg_card']};
        border: 1px solid {theme['border']};
        border-radius: 8px;
        margin-bottom: 8px;
    }}
    
    .leaderboard-rank {{
        width: 24px;
        font-size: 14px;
        font-weight: 700;
        color: {theme['text_primary']};
        margin-right: 12px;
    }}
    
    .leaderboard-avatar {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: {theme['accent']}20;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-weight: 600;
        color: {theme['accent']};
    }}
    
    .leaderboard-info {{
        flex: 1;
    }}
    
    .leaderboard-name {{
        color: {theme['text_primary']};
        font-size: 14px;
        font-weight: 600;
    }}
    
    .leaderboard-stats {{
        color: {theme['text_secondary']};
        font-size: 12px;
        margin-top: 2px;
    }}
    
    .leaderboard-trend {{
        font-size: 18px;
    }}
    
    /* Buttons */
    .custom-button {{
        background: {theme['accent']};
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}
    
    .custom-button:hover {{
        background: {theme['accent']}dd;
        transform: translateY(-1px);
    }}
    
    .secondary-button {{
        background: {theme['bg_secondary']};
        color: {theme['text_primary']};
        border: 1px solid {theme['border']};
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }}
    
    .secondary-button:hover {{
        background: {theme['bg_card']};
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .viewerBadge_container__1QSob {{display: none;}}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme['bg_secondary']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme['border']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme['text_secondary']};
    }}
</style>
""", unsafe_allow_html=True)

# SVG Icons
icons = {
    'chart': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 13h2v8H3zm4-8h2v16H7zm4-2h2v18h-2zm4 4h2v14h-2zm4-4h2v18h-2z"/></svg>',
    'users': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>',
    'activity': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>',
    'trending': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>',
    'settings': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/></svg>',
    'download': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>',
    'library': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-1 9H9V9h10v2zm-4 4H9v-2h6v2zm4-8H9V5h10v2z"/></svg>',
    'help': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/></svg>',
    'search': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>',
    'arrow_up': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 14l5-5 5 5z"/></svg>',
    'arrow_down': '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 10l5 5 5-5z"/></svg>',
}

# Database connection
@st.cache_resource
def get_db_connection():
    """Create database connection"""
    db_path = Path('data/reviews.db')
    if not db_path.exists():
        # Create a dummy database for demonstration
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE reviews (
                id INTEGER PRIMARY KEY,
                content TEXT,
                rating INTEGER,
                created_at TIMESTAMP,
                language TEXT
            )
        ''')
        # Insert sample data
        sample_data = []
        for i in range(90000):
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.15, 0.2, 0.3, 0.25])
            created_at = datetime.now() - timedelta(days=np.random.randint(0, 365))
            sample_data.append((
                f"Review content {i}",
                rating,
                created_at.strftime('%Y-%m-%d %H:%M:%S'),
                np.random.choice(['en', 'hi', 'ta', 'te'])
            ))
        cursor.executemany(
            "INSERT INTO reviews (content, rating, created_at, language) VALUES (?, ?, ?, ?)",
            sample_data
        )
        conn.commit()
        return conn
    return sqlite3.connect(str(db_path), check_same_thread=False)

# Load data functions
@st.cache_data(ttl=300)
def load_review_stats():
    """Load review statistics"""
    try:
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
            SELECT DATE(created_at) as date, COUNT(*) as count, AVG(rating) as avg_rating
            FROM reviews
            WHERE created_at >= date('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        """, conn)
        
        return stats, rating_dist, time_series
    except Exception as e:
        # Return dummy data if database fails
        stats = pd.Series({'total': 90000, 'avg_rating': 3.5, 'languages': 4})
        rating_dist = pd.DataFrame({
            'rating': [1, 2, 3, 4, 5],
            'count': [9000, 13500, 18000, 27000, 22500]
        })
        time_series = pd.DataFrame({
            'date': pd.date_range(end=datetime.now(), periods=30),
            'count': np.random.randint(2000, 4000, 30),
            'avg_rating': np.random.uniform(3, 4, 30)
        })
        return stats, rating_dist, time_series

@st.cache_data(ttl=300)
def load_sentiment_data():
    """Load sentiment analysis data"""
    try:
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
    except:
        return pd.Series({'positive': 31679, 'neutral': 4062, 'negative': 54259})

def render_metric_card_with_chart(label, value, change=None, chart_data=None, color='#3b82f6'):
    """Render metric card with mini chart"""
    
    # Determine change direction
    change_type = 'neutral'
    change_icon = ''
    if change:
        if '+' in change or (change and not '-' in change):
            change_type = 'positive'
            change_icon = '▲'
        elif '-' in change:
            change_type = 'negative'
            change_icon = '▼'
        else:
            change_icon = '•'
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-change {change_type}">{change_icon} {change}</div>' if change else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Add mini chart if data provided
    if chart_data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=chart_data,
            mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)',  # Light blue with transparency
            showlegend=False,
            hovertemplate='%{y}<extra></extra>'
        ))
        
        fig.update_layout(
            height=60,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_progress_item(icon, title, percentage, color):
    """Render a progress item with icon"""
    progress_html = f"""
    <div class="progress-item">
        <div class="progress-icon" style="background: {color}20;">
            {icon}
        </div>
        <div class="progress-content">
            <div class="progress-title">{title}</div>
            <div class="progress-bar-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage}%; background: {color};"></div>
                </div>
                <div class="progress-percent">{percentage}% Correct</div>
            </div>
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

def render_leaderboard_item(rank, name, score, stats, trend='up'):
    """Render a leaderboard item"""
    trend_icon = '▲' if trend == 'up' else '▼'
    trend_color = theme['success'] if trend == 'up' else theme['danger']
    
    initials = ''.join([n[0].upper() for n in name.split()[:2]])
    
    leaderboard_html = f"""
    <div class="leaderboard-item">
        <div class="leaderboard-rank">{rank}</div>
        <div class="leaderboard-avatar">{initials}</div>
        <div class="leaderboard-info">
            <div class="leaderboard-name">{name}</div>
            <div class="leaderboard-stats">{score} Points · {stats}</div>
        </div>
        <div class="leaderboard-trend" style="color: {trend_color};">{trend_icon}</div>
    </div>
    """
    st.markdown(leaderboard_html, unsafe_allow_html=True)

# Main app
def main():
    # Load data
    stats, rating_dist, time_series = load_review_stats()
    sentiments = load_sentiment_data()
    
    # Sidebar
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="logo-container">
            <div class="logo-text">TESLA</div>
            <div class="logo-subtitle">ANALYTICS PLATFORM</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown('<div class="nav-title">Navigation</div>', unsafe_allow_html=True)
        
        page_options = {
            "📊 Reports": "Reports",
            "📚 Library": "Library",
            "👥 People": "People",
            "📈 Activities": "Activities",
            "⚙️ Settings": "Settings"
        }
        
        page = st.radio(
            "nav",
            list(page_options.values()),
            label_visibility="collapsed",
            index=0
        )
        
        st.markdown('<div style="margin: 24px 0; border-bottom: 1px solid ' + theme['border'] + ';"></div>', 
                   unsafe_allow_html=True)
        
        # Support section
        st.markdown('<div class="nav-title">Support</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Get Started", use_container_width=True):
                st.info("Opening documentation...")
        
        with col2:
            if st.button("⚙️ Settings", use_container_width=True):
                st.info("Opening settings...")
    
    # Main content area
    if page == "Reports":
        # Header with theme toggle
        col1, col2 = st.columns([10, 1])
        with col1:
            st.markdown('<h1 style="color: ' + theme['text_primary'] + '; margin-bottom: 8px;">Reports</h1>', 
                       unsafe_allow_html=True)
        with col2:
            if st.button("🌙" if not st.session_state.dark_mode else "☀️", 
                        help="Toggle dark/light mode"):
                toggle_dark_mode()
                st.rerun()
        
        # Download button
        col1, col2 = st.columns([10, 2])
        with col2:
            st.markdown(f"""
            <button class="custom-button">
                {icons['download']} Download
            </button>
            """, unsafe_allow_html=True)
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            timeframe = st.selectbox("Timeframe:", ["All-time", "Last 30 days", "Last 7 days", "Today"])
        with col2:
            people = st.selectbox("People:", ["All", "Active Users", "New Users"])
        with col3:
            topic_filter = st.selectbox("Topic:", ["All", "Top Topics", "Trending"])
        
        st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_metric_card_with_chart(
                "Active Users",
                f"{27:,}/80",
                "+12% from last month",
                np.random.randint(20, 40, 12),
                theme['accent_secondary']
            )
        
        with col2:
            render_metric_card_with_chart(
                "Questions Answered",
                "3,298",
                None,
                np.random.randint(2800, 3500, 12),
                theme['success']
            )
        
        with col3:
            render_metric_card_with_chart(
                "Av. Session Length",
                "2m 34s",
                None,
                np.random.uniform(2, 3, 12),
                theme['warning']
            )
        
        with col4:
            # Activity chart in fourth column
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Activity</div>
                <div style="height: 80px;">
            """, unsafe_allow_html=True)
            
            # Monthly activity bar chart
            months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            values = [100, 150, 180, 200, 280, 300, 260, 300, 350, 100, 400, 420]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=months,
                y=values,
                marker_color=theme['accent_secondary'],
                hovertemplate='%{x}: %{y}<extra></extra>'
            ))
            
            fig.update_layout(
                height=100,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    visible=True,
                    showgrid=False,
                    tickfont=dict(size=8),
                    fixedrange=True
                ),
                yaxis=dict(visible=False, fixedrange=True),
                showlegend=False,
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Knowledge Metrics Row
        st.markdown('<div style="margin: 30px 0;"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_metric_card_with_chart(
                "Starting Knowledge 🔗",
                "64%",
                None,
                np.random.uniform(60, 70, 12),
                theme['accent_secondary']
            )
        
        with col2:
            render_metric_card_with_chart(
                "Current Knowledge",
                "86%",
                None,
                np.random.uniform(80, 90, 12),
                theme['success']
            )
        
        with col3:
            render_metric_card_with_chart(
                "Knowledge Gain",
                "+34%",
                None,
                np.random.uniform(30, 40, 12),
                theme['success']
            )
        
        # Topics Section
        st.markdown('<div style="margin: 40px 0;"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="section-title">Weakest Topics</div>', unsafe_allow_html=True)
            render_progress_item("🍽️", "Food Safety", 74, theme['warning'])
            render_progress_item("📋", "Compliance Basics Procedures", 52, theme['danger'])
            render_progress_item("🌐", "Company Networking", 36, theme['danger'])
        
        with col2:
            st.markdown(f'<div class="section-title">Strongest Topics</div>', unsafe_allow_html=True)
            render_progress_item("😷", "Covid Protocols", 95, theme['success'])
            render_progress_item("🔒", "Cyber Security Basics", 92, theme['success'])
            render_progress_item("📱", "Social Media Policies", 89, theme['success'])
        
        # Leaderboards
        st.markdown('<div style="margin: 40px 0;"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="section-title">User Leaderboard</div>', unsafe_allow_html=True)
            render_leaderboard_item(1, "Jesse Thomas", 637, "98% Correct", "up")
            render_leaderboard_item(2, "Thisal Mathiyazhagan", 637, "89% Correct", "down")
        
        with col2:
            st.markdown(f'<div class="section-title">Groups Leaderboard</div>', unsafe_allow_html=True)
            render_leaderboard_item(1, "Houston Facility", 52, "User · 97% Correct", "up")
            render_leaderboard_item(2, "Test Group", 52, "User · 95% Correct", "down")
    
    elif page == "People":
        st.markdown('<h1 style="color: ' + theme['text_primary'] + ';">People Analytics</h1>', 
                   unsafe_allow_html=True)
        
        # User metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            render_metric_card_with_chart(
                "Total Users",
                "1,234",
                "+8% from last month",
                np.random.randint(1000, 1300, 12),
                theme['accent_secondary']
            )
        
        with col2:
            render_metric_card_with_chart(
                "Active Today",
                "342",
                "+15% from yesterday",
                np.random.randint(250, 400, 7),
                theme['success']
            )
        
        with col3:
            render_metric_card_with_chart(
                "New Users",
                "56",
                "This week",
                np.random.randint(5, 15, 7),
                theme['warning']
            )
        
        with col4:
            render_metric_card_with_chart(
                "Engagement Rate",
                "78%",
                "+3% from last week",
                np.random.uniform(70, 85, 12),
                theme['success']
            )
    
    elif page == "Activities":
        st.markdown('<h1 style="color: ' + theme['text_primary'] + ';">Activity Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sentiment Analysis
        col1, col2, col3 = st.columns(3)
        
        total = sentiments['positive'] + sentiments['neutral'] + sentiments['negative']
        
        with col1:
            render_metric_card_with_chart(
                "Positive Reviews",
                f"{int(sentiments['positive']):,}",
                f"{(sentiments['positive']/total*100):.1f}%",
                None,
                theme['success']
            )
        
        with col2:
            render_metric_card_with_chart(
                "Neutral Reviews",
                f"{int(sentiments['neutral']):,}",
                f"{(sentiments['neutral']/total*100):.1f}%",
                None,
                theme['warning']
            )
        
        with col3:
            render_metric_card_with_chart(
                "Negative Reviews",
                f"{int(sentiments['negative']):,}",
                f"{(sentiments['negative']/total*100):.1f}%",
                None,
                theme['danger']
            )
        
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
            legend=dict(
                font=dict(color=theme['text_primary'])
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Library":
        st.markdown('<h1 style="color: ' + theme['text_primary'] + ';">Resource Library</h1>', 
                   unsafe_allow_html=True)
        
        st.info("Resource library coming soon...")
    
    elif page == "Settings":
        st.markdown('<h1 style="color: ' + theme['text_primary'] + ';">Settings</h1>', 
                   unsafe_allow_html=True)
        
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
        
        if st.button("Reload Dashboard"):
            st.rerun()

if __name__ == "__main__":
    main()