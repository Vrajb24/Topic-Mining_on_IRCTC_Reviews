#!/usr/bin/env python3
"""
Modern IRCTC Review Analysis Dashboard
Professional analytics UI with dark mode support
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

# Initialize session state for dark mode
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Toggle dark mode function
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# SVG Icons for professional look
icons = {
    'overview': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/></svg>''',
    'topics': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>''',
    'sentiment': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/></svg>''',
    'explorer': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l-5.5 9h11z M12 22l5.5-9h-11z"/></svg>''',
    'settings': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/></svg>''',
    'support': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/></svg>''',
    'moon': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M9.5 2c-1.82 0-3.53.5-5 1.35 2.99 1.73 5 4.95 5 8.65s-2.01 6.92-5 8.65c1.47.85 3.18 1.35 5 1.35 5.52 0 10-4.48 10-10S15.02 2 9.5 2z"/></svg>''',
    'sun': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zm0-5v3.5L12 2zm0 14.5V22l.01-2.5zM3.5 12H1v.01L3.5 12zm18.5 0h-2.5l2.49.01zM5.99 6.31L3.87 4.19l2.12 2.12zm12.02 0l2.12-2.12-2.12 2.12zm-12.02 11.38l-2.12 2.12 2.12-2.12zm12.02 0l2.12 2.12-2.12-2.12z"/></svg>'''
}

# Apply theme based on dark mode state
if st.session_state.dark_mode:
    # Dark theme
    theme_colors = {
        'bg': '#0e1117',
        'bg_secondary': '#1a1d24',
        'card': '#1e2329',
        'text': '#ffffff',
        'text_secondary': '#b8bcc8',
        'accent': '#3e7bfa',  # Blue accent like in UI_idea
        'success': '#10b981',
        'danger': '#ef4444',
        'warning': '#f59e0b',
        'border': '#2d3139'
    }
else:
    # Light theme with better contrast
    theme_colors = {
        'bg': '#ffffff',
        'bg_secondary': '#f5f7fa',
        'card': '#ffffff',
        'text': '#1a1a1a',  # Much darker text for better visibility
        'text_secondary': '#4a5568',  # Darker secondary text
        'accent': '#3e7bfa',
        'success': '#10b981',
        'danger': '#ef4444',
        'warning': '#f59e0b',
        'border': '#e2e8f0'
    }

# Custom CSS
st.markdown(f"""
    <style>
    /* Main theme */
    .stApp {{
        background-color: {theme_colors['bg_secondary']};
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {theme_colors['card']} !important;
        border-right: 1px solid {theme_colors['border']};
    }}
    
    section[data-testid="stSidebar"] .element-container {{
        color: {theme_colors['text']};
    }}
    
    /* Logo styling */
    .logo-container {{
        display: flex;
        align-items: center;
        padding: 20px 0;
        border-bottom: 1px solid {theme_colors['border']};
        margin-bottom: 20px;
    }}
    
    .logo-text {{
        font-size: 24px;
        font-weight: bold;
        color: {theme_colors['accent']};
        margin-left: 10px;
    }}
    
    /* Navigation items - Fixed text visibility */
    .nav-item {{
        display: flex;
        align-items: center;
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
        color: {theme_colors['text']};
    }}
    
    .nav-item:hover {{
        background-color: {theme_colors['bg_secondary']};
    }}
    
    .nav-item.active {{
        background-color: {theme_colors['accent']}22;
        border-left: 3px solid {theme_colors['accent']};
    }}
    
    /* Cards */
    .metric-card {{
        background: {theme_colors['card']};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid {theme_colors['border']};
        margin-bottom: 20px;
    }}
    
    .metric-value {{
        font-size: 36px;
        font-weight: 700;
        color: {theme_colors['text']};
        margin: 8px 0;
    }}
    
    .metric-label {{
        font-size: 14px;
        color: {theme_colors['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-change {{
        font-size: 14px;
        margin-top: 8px;
    }}
    
    .metric-change.positive {{
        color: {theme_colors['success']};
    }}
    
    .metric-change.negative {{
        color: {theme_colors['danger']};
    }}
    
    /* Progress bars */
    .progress-container {{
        margin: 20px 0;
    }}
    
    .progress-header {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
    }}
    
    .progress-label {{
        font-size: 14px;
        font-weight: 500;
        color: {theme_colors['text']};
    }}
    
    .progress-value {{
        font-size: 14px;
        color: {theme_colors['text_secondary']};
    }}
    
    .progress-bar {{
        height: 8px;
        background-color: {theme_colors['bg_secondary']};
        border-radius: 4px;
        overflow: hidden;
    }}
    
    .progress-fill {{
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }}
    
    .progress-fill.good {{
        background: linear-gradient(90deg, {theme_colors['success']}aa, {theme_colors['success']});
    }}
    
    .progress-fill.warning {{
        background: linear-gradient(90deg, {theme_colors['warning']}aa, {theme_colors['warning']});
    }}
    
    .progress-fill.danger {{
        background: linear-gradient(90deg, {theme_colors['danger']}aa, {theme_colors['danger']});
    }}
    
    /* Section headers */
    .section-header {{
        font-size: 18px;
        font-weight: 600;
        color: {theme_colors['text']};
        margin: 24px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid {theme_colors['border']};
    }}
    
    /* Topic items */
    .topic-item {{
        display: flex;
        align-items: center;
        padding: 12px;
        margin: 8px 0;
        background: {theme_colors['bg_secondary']};
        border-radius: 8px;
        border: 1px solid {theme_colors['border']};
    }}
    
    .topic-icon {{
        width: 40px;
        height: 40px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        font-size: 20px;
    }}
    
    .topic-details {{
        flex: 1;
    }}
    
    .topic-name {{
        font-weight: 500;
        color: {theme_colors['text']};
        margin-bottom: 4px;
    }}
    
    /* Hide streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom header */
    .main-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        border-bottom: 1px solid {theme_colors['border']};
        margin-bottom: 24px;
    }}
    
    .header-title {{
        font-size: 28px;
        font-weight: 700;
        color: {theme_colors['text']};
    }}
    
    /* Dropdown filters */
    .filter-container {{
        display: flex;
        gap: 12px;
        align-items: center;
    }}
    
    /* Dark mode toggle */
    .dark-mode-toggle {{
        position: fixed;
        top: 14px;
        right: 60px;
        z-index: 999;
    }}
    </style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_database_connection():
    return sqlite3.connect('data/reviews.db', check_same_thread=False)

# Load data functions
@st.cache_data(ttl=300)
def load_review_stats():
    conn = get_database_connection()
    
    # Total reviews
    total = pd.read_sql_query("SELECT COUNT(*) as count FROM reviews", conn)['count'][0]
    
    # Average rating
    avg_rating = pd.read_sql_query(
        "SELECT AVG(rating) as avg FROM reviews WHERE rating > 0", conn
    )['avg'][0]
    
    # Rating distribution
    rating_dist = pd.read_sql_query(
        "SELECT rating, COUNT(*) as count FROM reviews WHERE rating > 0 GROUP BY rating", conn
    )
    
    # Daily reviews for activity chart
    daily_reviews = pd.read_sql_query(
        """SELECT DATE(date_posted) as date, COUNT(*) as count 
        FROM reviews WHERE date_posted IS NOT NULL 
        GROUP BY DATE(date_posted) 
        ORDER BY date DESC LIMIT 30""", conn
    )
    
    # Sentiment counts based on ratings
    sentiments = pd.read_sql_query(
        """SELECT 
            COUNT(CASE WHEN rating >= 4 THEN 1 END) as positive,
            COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative,
            COUNT(CASE WHEN rating = 3 THEN 1 END) as neutral
        FROM reviews WHERE rating > 0""", conn
    )
    
    return {
        'total': total,
        'avg_rating': avg_rating,
        'rating_dist': rating_dist,
        'daily_reviews': daily_reviews,
        'sentiments': sentiments
    }

@st.cache_data(ttl=300)
def load_topics_from_model():
    """Load topics from saved model"""
    try:
        models_dir = Path('data/models')
        model_files = list(models_dir.glob('lda_model_*.pkl'))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            with open(latest_model, 'rb') as f:
                model_data = pickle.load(f)
                return model_data.get('topics', [])
    except:
        pass
    return []

def create_mini_sparkline(data, color):
    """Create a mini sparkline chart"""
    fig = go.Figure()
    # Convert hex color to rgba for transparency
    if color.startswith('#'):
        # Extract RGB values from hex
        hex_color = color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:6], 16)
        fillcolor = f'rgba({r},{g},{b},0.2)'
    else:
        fillcolor = color
    
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=fillcolor,
        showlegend=False,
        hoverinfo='none'
    ))
    
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig

def main():
    # Sidebar
    with st.sidebar:
        # Logo section
        st.markdown("""
            <div class="logo-container">
                <span style="font-size: 32px;">🚂</span>
                <span class="logo-text">IRCTC Analytics</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Overview", "📈 Topics", "💭 Sentiment", "🔍 Explorer", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Support section
        st.markdown("### Support")
        if st.button("📚 Documentation"):
            st.info("View documentation at docs.irctc-analytics.com")
        
        if st.button("💬 Get Help"):
            st.info("Contact support@irctc-analytics.com")
    
    # Dark mode toggle in top right
    st.markdown("""
        <div class="dark-mode-toggle">
    """, unsafe_allow_html=True)
    
    col_toggle = st.columns([0.8, 0.2])
    with col_toggle[1]:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", 
                    help="Toggle dark/light mode",
                    on_click=toggle_dark_mode):
            pass
    
    # Load data
    stats = load_review_stats()
    topics = load_topics_from_model()
    
    # Main content based on page selection
    if page == "📊 Overview":
        # Header with filters
        col_header = st.columns([2, 1, 1, 1])
        with col_header[0]:
            st.markdown(f"<h1 class='header-title'>Dashboard</h1>", unsafe_allow_html=True)
        
        with col_header[1]:
            timeframe = st.selectbox("Timeframe", ["All-time", "Last 30 days", "Last 7 days", "Today"])
        
        with col_header[2]:
            category = st.selectbox("Category", ["All", "Positive", "Negative", "Neutral"])
        
        with col_header[3]:
            if st.button("📥 Download Report"):
                st.info("Generating report...")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Reviews</div>
                    <div class="metric-value">{stats['total']:,}</div>
                    <div class="metric-change positive">↑ 12% from last month</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Mini chart
            sample_data = np.random.randint(50, 100, 20)
            fig = create_mini_sparkline(sample_data, theme_colors['accent'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            avg_rating = stats['avg_rating'] if stats['avg_rating'] else 0
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Rating</div>
                    <div class="metric-value">{avg_rating:.2f}</div>
                    <div class="metric-change negative">↓ 0.3 from last month</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Mini chart
            sample_data = np.random.uniform(2, 3, 20)
            fig = create_mini_sparkline(sample_data, theme_colors['warning'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col3:
            satisfaction = (avg_rating / 5 * 100) if avg_rating else 0
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Satisfaction Rate</div>
                    <div class="metric-value">{satisfaction:.0f}%</div>
                    <div class="metric-change negative">↓ 5% from baseline</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Mini chart
            sample_data = np.random.uniform(40, 60, 20)
            fig = create_mini_sparkline(sample_data, theme_colors['danger'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        with col4:
            response_rate = 85  # Placeholder
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Response Rate</div>
                    <div class="metric-value">{response_rate}%</div>
                    <div class="metric-change positive">↑ 8% improvement</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Mini chart
            sample_data = np.random.uniform(75, 90, 20)
            fig = create_mini_sparkline(sample_data, theme_colors['success'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Activity chart
        st.markdown(f"<div class='section-header'>Daily Activity</div>", unsafe_allow_html=True)
        
        if not stats['daily_reviews'].empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=stats['daily_reviews']['date'],
                y=stats['daily_reviews']['count'],
                marker_color=theme_colors['accent'],
                hovertemplate='%{y} reviews<br>%{x}<extra></extra>'
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    color=theme_colors['text_secondary']
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=theme_colors['border'],
                    color=theme_colors['text_secondary']
                ),
                hoverlabel=dict(
                    bgcolor=theme_colors['card'],
                    font_color=theme_colors['text']
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Two column section for problem areas and top performers
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown(f"<div class='section-header'>Problem Areas</div>", unsafe_allow_html=True)
            
            problem_topics = [
                ("🔐", "Login Issues", 35, "danger"),
                ("💳", "Payment Failures", 48, "warning"),
                ("🐌", "App Performance", 62, "warning"),
                ("📱", "UI/UX Problems", 71, "good")
            ]
            
            for icon, name, score, status in problem_topics:
                color = theme_colors[status if status != 'good' else 'success']
                st.markdown(f"""
                    <div class="topic-item">
                        <div class="topic-icon" style="background: {color}20;">
                            {icon}
                        </div>
                        <div class="topic-details">
                            <div class="topic-name">{name}</div>
                            <div class="progress-bar">
                                <div class="progress-fill {status}" style="width: {score}%"></div>
                            </div>
                        </div>
                        <div style="color: {theme_colors['text_secondary']}; font-size: 14px;">
                            {score}% affected
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown(f"<div class='section-header'>Top Positive Topics</div>", unsafe_allow_html=True)
            
            positive_topics = [
                ("✅", "Booking Success", 92, "good"),
                ("🎫", "Ticket Availability", 87, "good"),
                ("📱", "App Updates", 78, "good"),
                ("💰", "Refund Process", 65, "warning")
            ]
            
            for icon, name, score, status in positive_topics:
                color = theme_colors['success' if status == 'good' else 'warning']
                st.markdown(f"""
                    <div class="topic-item">
                        <div class="topic-icon" style="background: {color}20;">
                            {icon}
                        </div>
                        <div class="topic-details">
                            <div class="topic-name">{name}</div>
                            <div class="progress-bar">
                                <div class="progress-fill {status}" style="width: {score}%"></div>
                            </div>
                        </div>
                        <div style="color: {theme_colors['text_secondary']}; font-size: 14px;">
                            {score}% satisfaction
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    elif page == "📈 Topics":
        st.markdown(f"<h1 class='header-title'>Topic Analysis</h1>", unsafe_allow_html=True)
        
        if topics:
            # Topic distribution
            topic_counts = {}
            for i, topic in enumerate(topics[:15]):
                topic_counts[f"Topic {i+1}"] = np.random.randint(100, 1000)  # Placeholder
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(topic_counts.keys()),
                y=list(topic_counts.values()),
                marker_color=theme_colors['accent'],
                text=list(topic_counts.values()),
                textposition='outside'
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    color=theme_colors['text_secondary']
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor=theme_colors['border'],
                    color=theme_colors['text_secondary']
                ),
                font=dict(color=theme_colors['text'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Topic details
            st.markdown(f"<div class='section-header'>Topic Keywords</div>", unsafe_allow_html=True)
            
            for i, topic in enumerate(topics[:10]):
                with st.expander(f"Topic {i+1}: {', '.join(topic['words'][:3])}..."):
                    keywords = topic['words'][:10]
                    st.write("**Top Keywords:**")
                    cols = st.columns(2)
                    for j, keyword in enumerate(keywords):
                        with cols[j % 2]:
                            st.write(f"• {keyword}")
        else:
            st.info("No topics identified yet. Run topic modeling analysis first.")
    
    elif page == "💭 Sentiment":
        st.markdown(f"<h1 class='header-title'>Sentiment Analysis</h1>", unsafe_allow_html=True)
        
        sentiments = stats['sentiments']
        
        # Sentiment metrics
        col1, col2, col3 = st.columns(3)
        
        total_reviews = sentiments['positive'][0] + sentiments['negative'][0] + sentiments['neutral'][0]
        
        with col1:
            pos_pct = (sentiments['positive'][0] / total_reviews * 100) if total_reviews > 0 else 0
            st.markdown(f"""
                <div class="metric-card" style="border-left: 3px solid {theme_colors['success']};">
                    <div class="metric-label">Positive Reviews</div>
                    <div class="metric-value" style="color: {theme_colors['success']};">
                        {sentiments['positive'][0]:,}
                    </div>
                    <div class="metric-change">{pos_pct:.1f}% of total</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            neu_pct = (sentiments['neutral'][0] / total_reviews * 100) if total_reviews > 0 else 0
            st.markdown(f"""
                <div class="metric-card" style="border-left: 3px solid {theme_colors['warning']};">
                    <div class="metric-label">Neutral Reviews</div>
                    <div class="metric-value" style="color: {theme_colors['warning']};">
                        {sentiments['neutral'][0]:,}
                    </div>
                    <div class="metric-change">{neu_pct:.1f}% of total</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            neg_pct = (sentiments['negative'][0] / total_reviews * 100) if total_reviews > 0 else 0
            st.markdown(f"""
                <div class="metric-card" style="border-left: 3px solid {theme_colors['danger']};">
                    <div class="metric-label">Negative Reviews</div>
                    <div class="metric-value" style="color: {theme_colors['danger']};">
                        {sentiments['negative'][0]:,}
                    </div>
                    <div class="metric-change">{neg_pct:.1f}% of total</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Sentiment pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[sentiments['positive'][0], sentiments['neutral'][0], sentiments['negative'][0]],
            hole=.4,
            marker=dict(colors=[theme_colors['success'], theme_colors['warning'], theme_colors['danger']])
        )])
        
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=theme_colors['text']),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🔍 Explorer":
        st.markdown(f"<h1 class='header-title'>Review Explorer</h1>", unsafe_allow_html=True)
        
        # Search interface
        search_term = st.text_input("Search reviews", placeholder="Enter keywords...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rating_filter = st.selectbox("Rating", ["All", "5⭐", "4⭐", "3⭐", "2⭐", "1⭐"])
        with col2:
            sentiment_filter = st.selectbox("Sentiment", ["All", "Positive", "Neutral", "Negative"])
        with col3:
            sort_by = st.selectbox("Sort by", ["Most Recent", "Highest Rated", "Lowest Rated"])
        
        # Load and display reviews
        conn = get_database_connection()
        query = "SELECT * FROM reviews"
        conditions = []
        
        if search_term:
            conditions.append(f"content LIKE '%{search_term}%'")
        
        if rating_filter != "All":
            rating_num = int(rating_filter[0])
            conditions.append(f"rating = {rating_num}")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if sort_by == "Most Recent":
            query += " ORDER BY date_posted DESC"
        elif sort_by == "Highest Rated":
            query += " ORDER BY rating DESC"
        else:
            query += " ORDER BY rating ASC"
        
        query += " LIMIT 50"
        
        reviews_df = pd.read_sql_query(query, conn)
        
        if not reviews_df.empty:
            st.markdown(f"<div class='section-header'>Found {len(reviews_df)} reviews</div>", 
                       unsafe_allow_html=True)
            
            for idx, row in reviews_df.iterrows():
                rating_color = theme_colors['success'] if row['rating'] >= 4 else (
                    theme_colors['warning'] if row['rating'] == 3 else theme_colors['danger']
                )
                
                st.markdown(f"""
                    <div class="topic-item">
                        <div style="flex: 1;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: {rating_color}; font-weight: 500;">
                                    {'⭐' * int(row['rating'])}
                                </span>
                                <span style="color: {theme_colors['text_secondary']}; font-size: 12px;">
                                    {row['date_posted'][:10] if row['date_posted'] else 'N/A'}
                                </span>
                            </div>
                            <div style="color: {theme_colors['text']};">
                                {row['content'][:200]}...
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No reviews found matching your criteria")
    
    elif page == "⚙️ Settings":
        st.markdown(f"<h1 class='header-title'>Settings</h1>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='section-header'>Display Preferences</div>", unsafe_allow_html=True)
        
        # Theme settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Theme Mode</div>
            """, unsafe_allow_html=True)
            
            theme_choice = st.radio(
                "Choose theme",
                ["Light Mode", "Dark Mode"],
                index=1 if st.session_state.dark_mode else 0
            )
            
            if (theme_choice == "Dark Mode" and not st.session_state.dark_mode) or \
               (theme_choice == "Light Mode" and st.session_state.dark_mode):
                toggle_dark_mode()
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Data Refresh</div>
            """, unsafe_allow_html=True)
            
            refresh_interval = st.selectbox(
                "Auto-refresh interval",
                ["Never", "5 minutes", "15 minutes", "30 minutes", "1 hour"]
            )
            
            if st.button("🔄 Refresh Now"):
                st.cache_data.clear()
                st.success("Data refreshed successfully!")
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='section-header'>Export Options</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Dashboard as PDF", use_container_width=True):
                st.info("Generating PDF report...")
        
        with col2:
            if st.button("📁 Export Raw Data as CSV", use_container_width=True):
                st.info("Preparing CSV export...")
        
        st.markdown(f"<div class='section-header'>About</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="metric-card">
                <div style="color: {theme_colors['text']};">
                    <h4>IRCTC Review Analytics Dashboard</h4>
                    <p style="color: {theme_colors['text_secondary']};">
                        Version: 2.0.0<br>
                        Last Updated: {datetime.now().strftime('%B %d, %Y')}<br>
                        Total Reviews Analyzed: {stats['total']:,}<br>
                        <br>
                        Built with Streamlit, Plotly, and Python<br>
                        © 2025 IRCTC Analytics Team
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()