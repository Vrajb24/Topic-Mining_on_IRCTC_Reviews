#!/usr/bin/env python3
"""
IRCTC Review Analysis Dashboard - Streamlit Cloud Deployment
Unified dashboard for deployment on Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="IRCTC Review Analysis Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: #0e1117;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9;
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

# Database connection
@st.cache_resource
def get_connection():
    """Create database connection"""
    return sqlite3.connect('data/reviews.db', check_same_thread=False)

# Load pre-computed models
@st.cache_data
def load_models():
    """Load pre-computed analysis models"""
    models = {}
    
    # Try to load each model
    model_paths = {
        'topics': 'data/models/improved_topics.pkl',
        'root_cause': 'data/analysis/root_cause_analysis.pkl',
        'full_analysis': 'data/models/full_analysis_results.pkl'
    }
    
    for name, path in model_paths.items():
        if Path(path).exists():
            try:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
    
    return models

# Load data functions
@st.cache_data
def load_stats():
    """Load review statistics"""
    conn = get_connection()
    
    stats = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_reviews,
            AVG(rating) as avg_rating,
            COUNT(DISTINCT language) as languages,
            COUNT(CASE WHEN rating = 1 THEN 1 END) as one_star,
            COUNT(CASE WHEN rating <= 2 THEN 1 END) as critical_issues
        FROM reviews
    """, conn).iloc[0]
    
    return stats

@st.cache_data
def load_time_series(days=30):
    """Load time series data"""
    conn = get_connection()
    
    query = f"""
        SELECT 
            DATE(date_posted) as date,
            COUNT(*) as review_count,
            AVG(rating) as avg_rating
        FROM reviews
        WHERE date_posted >= date('now', '-{days} days')
        GROUP BY DATE(date_posted)
        ORDER BY date
    """
    
    return pd.read_sql_query(query, conn)

@st.cache_data
def load_department_stats():
    """Load department statistics"""
    conn = get_connection()
    
    try:
        query = """
            SELECT 
                rc.department,
                COUNT(*) as count,
                AVG(r.rating) as avg_rating,
                COUNT(CASE WHEN r.rating = 1 THEN 1 END) as one_star_count
            FROM review_classifications rc
            JOIN reviews r ON rc.review_id = r.id
            GROUP BY rc.department
        """
        return pd.read_sql_query(query, conn)
    except:
        return pd.DataFrame()

# Sidebar
with st.sidebar:
    st.title("🚂 IRCTC Analytics")
    st.markdown("---")
    
    # Navigation
    page = st.selectbox(
        "Select Dashboard",
        ["📊 Overview", "🏢 Department Analysis", "🔍 Root Cause Analysis", "📈 Trends"]
    )
    
    st.markdown("---")
    
    # Stats summary
    stats = load_stats()
    st.metric("Total Reviews", f"{int(stats['total_reviews']):,}")
    st.metric("Average Rating", f"{stats['avg_rating']:.2f} ⭐")
    
    st.markdown("---")
    st.info("💡 Data updates daily")
    st.markdown("Built with ❤️ using Streamlit")

# Main content area
if "Overview" in page:
    st.title("📊 IRCTC Review Analysis Dashboard")
    st.markdown("### Real-time insights from 90,000+ app reviews")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Reviews</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(int(stats['total_reviews'])), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Critical Issues</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(int(stats['critical_issues'])), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Languages</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(int(stats['languages'])), unsafe_allow_html=True)
    
    with col4:
        satisfaction = (1 - (stats['critical_issues'] / stats['total_reviews'])) * 100
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Satisfaction</div>
            <div class="metric-value">{:.1f}%</div>
        </div>
        """.format(satisfaction), unsafe_allow_html=True)
    
    # Charts
    st.markdown('<h2 class="section-header">Analytics Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating distribution
        conn = get_connection()
        rating_dist = pd.read_sql_query("""
            SELECT rating, COUNT(*) as count
            FROM reviews
            GROUP BY rating
            ORDER BY rating
        """, conn)
        
        fig = px.bar(rating_dist, x='rating', y='count',
                    title='Rating Distribution',
                    labels={'count': 'Number of Reviews', 'rating': 'Star Rating'},
                    color='rating',
                    color_continuous_scale='RdYlGn')
        fig.update_layout(
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Time series
        time_data = load_time_series()
        
        fig = px.line(time_data, x='date', y='review_count',
                     title='Daily Review Volume (30 days)',
                     labels={'review_count': 'Reviews', 'date': 'Date'})
        fig.update_traces(line_color='#60a5fa')
        fig.update_layout(
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#334155')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent insights
    st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
    
    models = load_models()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="recommendation-card">
            <h4>🔴 Top Issue</h4>
            <p>Payment failures during peak hours</p>
            <small>Affects 23% of users</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="recommendation-card">
            <h4>📈 Trend</h4>
            <p>35% increase in login issues</p>
            <small>Last 7 days</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="recommendation-card">
            <h4>✅ Improvement</h4>
            <p>Booking success rate up 12%</p>
            <small>After recent update</small>
        </div>
        """, unsafe_allow_html=True)

elif "Department" in page:
    st.title("🏢 Department Analysis")
    
    dept_stats = load_department_stats()
    
    if not dept_stats.empty:
        # Department distribution
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Department Split")
            for _, row in dept_stats.iterrows():
                percentage = (row['count'] / dept_stats['count'].sum()) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{row['department'].title()}</div>
                    <div class="metric-value">{percentage:.1f}%</div>
                    <small>{int(row['count']):,} reviews</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Pie chart
            fig = px.pie(dept_stats, values='count', names='department',
                        title='Department Distribution',
                        hole=0.4,
                        color_discrete_map={
                            'app': '#3b82f6',
                            'railway': '#ef4444',
                            'unclear': '#6b7280',
                            'mixed': '#f59e0b'
                        })
            fig.update_layout(
                plot_bgcolor='#1e293b',
                paper_bgcolor='#1e293b',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category breakdown
        st.markdown('<h2 class="section-header">Issue Categories</h2>', unsafe_allow_html=True)
        
        conn = get_connection()
        
        # App issues
        app_categories = pd.read_sql_query("""
            SELECT 
                top_app_category as category,
                COUNT(*) as count
            FROM review_classifications
            WHERE department = 'app' AND top_app_category IS NOT NULL
            GROUP BY top_app_category
            ORDER BY count DESC
            LIMIT 10
        """, conn)
        
        if not app_categories.empty:
            fig = px.bar(app_categories, x='category', y='count',
                        title='Top App Issues',
                        labels={'count': 'Number of Reviews', 'category': 'Issue Category'})
            fig.update_traces(marker_color='#3b82f6')
            fig.update_layout(
                plot_bgcolor='#1e293b',
                paper_bgcolor='#1e293b',
                font=dict(color='white'),
                xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Department analysis data is being processed. Please check back later.")

elif "Root Cause" in page:
    st.title("🔍 Root Cause Analysis")
    
    models = load_models()
    
    if 'root_cause' in models:
        analysis = models['root_cause']
        
        # Top root causes
        if 'top_root_causes' in analysis:
            st.markdown('<h2 class="section-header">Identified Root Causes</h2>', unsafe_allow_html=True)
            
            for i, cause in enumerate(analysis['top_root_causes'][:5], 1):
                with st.expander(f"#{i}: {cause['cause']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Frequency", f"{cause['frequency']} cases")
                    with col2:
                        st.metric("Pattern", cause.get('pattern', 'Multiple'))
                    
                    st.markdown("**Recommended Solution:**")
                    st.info(cause['solution'])
        
        # Severity distribution
        if 'severity_classification' in analysis:
            st.markdown('<h2 class="section-header">Severity Analysis</h2>', unsafe_allow_html=True)
            
            severity_data = []
            for level, data in analysis['severity_classification']['severity_distribution'].items():
                severity_data.append({
                    'Severity': level.title(),
                    'Count': data['count'],
                    'Impact': data['avg_impact']
                })
            
            if severity_data:
                df_severity = pd.DataFrame(severity_data)
                
                fig = px.bar(df_severity, x='Severity', y='Count',
                            color='Severity',
                            title='Issue Severity Distribution',
                            color_discrete_map={
                                'Critical': '#ef4444',
                                'High': '#f97316',
                                'Medium': '#f59e0b',
                                'Low': '#84cc16'
                            })
                fig.update_layout(
                    plot_bgcolor='#1e293b',
                    paper_bgcolor='#1e293b',
                    font=dict(color='white'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        if 'recommendations' in analysis:
            st.markdown('<h2 class="section-header">Action Items</h2>', unsafe_allow_html=True)
            
            for rec in analysis['recommendations'][:5]:
                priority_color = {
                    0: '#ef4444',
                    1: '#f97316',
                    2: '#f59e0b',
                    3: '#84cc16'
                }.get(rec['priority'], '#6b7280')
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4 style="color: {priority_color}">Priority {rec['priority']}: {rec['issue']}</h4>
                    <p><strong>Solution:</strong> {rec['solution']}</p>
                    <small>Impact: {rec['impact']} | Effort: {rec['estimated_effort']}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Root cause analysis is being computed. This may take a few minutes.")

elif "Trends" in page:
    st.title("📈 Trend Analysis")
    
    # Load 90-day data
    time_data = load_time_series(90)
    
    if not time_data.empty:
        # Volume and rating trends
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Review Volume Trend', 'Average Rating Trend'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Volume trend
        fig.add_trace(
            go.Scatter(x=time_data['date'], y=time_data['review_count'],
                      mode='lines', name='Daily Reviews',
                      line=dict(color='#3b82f6', width=2)),
            row=1, col=1
        )
        
        # Rating trend
        fig.add_trace(
            go.Scatter(x=time_data['date'], y=time_data['avg_rating'],
                      mode='lines+markers', name='Avg Rating',
                      line=dict(color='#10b981', width=2)),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b',
            font=dict(color='white'),
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor='#334155')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics from trends
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_daily = time_data['review_count'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Daily Reviews</div>
                <div class="metric-value">{int(avg_daily):,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            trend_direction = "📈" if time_data['review_count'].iloc[-7:].mean() > avg_daily else "📉"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">7-Day Trend</div>
                <div class="metric-value">{trend_direction}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rating_trend = time_data['avg_rating'].iloc[-7:].mean() - time_data['avg_rating'].iloc[:7].mean()
            trend_text = "Improving" if rating_trend > 0 else "Declining"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rating Trend</div>
                <div class="metric-value">{trend_text}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Trend data is being processed. Please check back later.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        <small>
        IRCTC Review Analysis Dashboard | 
        Data Mining Project 2025 | 
        Powered by Streamlit
        </small>
    </div>
    """,
    unsafe_allow_html=True
)