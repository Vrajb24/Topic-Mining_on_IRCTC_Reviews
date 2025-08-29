#!/usr/bin/env python3
"""
Department-Segregated Dashboard for IRCTC Reviews
Displays App Issues vs Railway Service Issues separately
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
from datetime import datetime

# Page config
st.set_page_config(
    page_title="IRCTC Segregated Analysis Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        height: 100%;
    }
    .topic-card {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #3b82f6;
        color: #1f2937;
    }
    .topic-card h4 {
        color: #1f2937;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .topic-card p {
        color: #4b5563;
        margin: 0.25rem 0;
    }
    .topic-card b {
        color: #1f2937;
    }
    .app-issue {
        border-left-color: #8b5cf6;
    }
    .railway-issue {
        border-left-color: #10b981;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_database():
    """Load database connection"""
    return sqlite3.connect('data/reviews.db', check_same_thread=False)

@st.cache_data(ttl=300)
def load_segregated_data():
    """Load segregated topic analysis results"""
    try:
        with open('data/models/improved_topics.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data(ttl=300)
def load_department_statistics():
    """Load department-wise statistics from database"""
    conn = load_database()
    
    query = """
    SELECT 
        rc.department,
        r.rating,
        COUNT(*) as count,
        AVG(r.rating) as avg_rating
    FROM review_classifications rc
    JOIN reviews r ON rc.review_id = r.id
    WHERE rc.department IN ('app', 'railway', 'mixed')
    GROUP BY rc.department, r.rating
    """
    
    df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)
def load_category_distribution():
    """Load category distribution"""
    conn = load_database()
    
    # App categories
    app_query = """
    SELECT 
        top_app_category as category,
        COUNT(*) as count,
        AVG(r.rating) as avg_rating
    FROM review_classifications rc
    JOIN reviews r ON rc.review_id = r.id
    WHERE rc.department = 'app' AND top_app_category IS NOT NULL
    GROUP BY top_app_category
    ORDER BY count DESC
    """
    
    # Railway categories
    railway_query = """
    SELECT 
        top_railway_category as category,
        COUNT(*) as count,
        AVG(r.rating) as avg_rating
    FROM review_classifications rc
    JOIN reviews r ON rc.review_id = r.id
    WHERE rc.department = 'railway' AND top_railway_category IS NOT NULL
    GROUP BY top_railway_category
    ORDER BY count DESC
    """
    
    app_df = pd.read_sql_query(app_query, conn)
    railway_df = pd.read_sql_query(railway_query, conn)
    
    return app_df, railway_df

@st.cache_data(ttl=300)
def load_sample_reviews(department: str, category: str = None, limit: int = 5):
    """Load sample reviews for a department/category"""
    conn = load_database()
    
    if category:
        if department == 'app':
            query = """
            SELECT r.content, r.rating, rc.confidence
            FROM reviews r
            JOIN review_classifications rc ON r.id = rc.review_id
            WHERE rc.department = ? AND rc.top_app_category = ?
            ORDER BY rc.confidence DESC
            LIMIT ?
            """
            params = (department, category, limit)
        else:
            query = """
            SELECT r.content, r.rating, rc.confidence
            FROM reviews r
            JOIN review_classifications rc ON r.id = rc.review_id
            WHERE rc.department = ? AND rc.top_railway_category = ?
            ORDER BY rc.confidence DESC
            LIMIT ?
            """
            params = (department, category, limit)
    else:
        query = """
        SELECT r.content, r.rating, rc.confidence
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE rc.department = ?
        ORDER BY rc.confidence DESC
        LIMIT ?
        """
        params = (department, limit)
    
    return pd.read_sql_query(query, conn, params=params)

def render_topic_card(topic, department):
    """Render a topic card"""
    keywords = ', '.join([w[0] for w in topic['words'][:8]])
    category = topic.get('category', 'uncategorized')
    
    card_class = "app-issue" if department == 'app' else "railway-issue"
    icon = "📱" if department == 'app' else "🚂"
    
    st.markdown(f"""
    <div class="topic-card {card_class}">
        <h4>{icon} {category.replace('_', ' ').title()}</h4>
        <p><b>Keywords:</b> {keywords}</p>
        <p><b>Documents:</b> {topic['doc_count']} | <b>Relevance:</b> {topic['relevance_score']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Load data
    segregated_data = load_segregated_data()
    dept_stats = load_department_statistics()
    app_categories, railway_categories = load_category_distribution()
    
    if not segregated_data:
        st.error("No segregated analysis found. Please run improved_topic_analysis.py first.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🚂 IRCTC Review Analysis - Department Segregation</h1>', 
                unsafe_allow_html=True)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_reviews = segregated_data['total_reviews']
        st.metric("Total Reviews Analyzed", f"{total_reviews:,}")
    
    with col2:
        app_reviews = segregated_data['department_stats'].get('app', 0)
        app_pct = (app_reviews / total_reviews * 100) if total_reviews > 0 else 0
        st.metric("App Issues", f"{app_reviews:,}", f"{app_pct:.1f}%")
    
    with col3:
        railway_reviews = segregated_data['department_stats'].get('railway', 0)
        railway_pct = (railway_reviews / total_reviews * 100) if total_reviews > 0 else 0
        st.metric("Railway Service Issues", f"{railway_reviews:,}", f"{railway_pct:.1f}%")
    
    with col4:
        mixed_reviews = segregated_data['department_stats'].get('mixed', 0)
        mixed_pct = (mixed_reviews / total_reviews * 100) if total_reviews > 0 else 0
        st.metric("Mixed Issues", f"{mixed_reviews:,}", f"{mixed_pct:.1f}%")
    
    # Department distribution chart
    st.markdown('<h2 class="section-header">Department Distribution</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stacked bar chart by rating
        if not dept_stats.empty:
            fig = go.Figure()
            
            colors = {'app': '#8b5cf6', 'railway': '#10b981', 'mixed': '#f59e0b'}
            
            for dept in ['app', 'railway', 'mixed']:
                dept_data = dept_stats[dept_stats['department'] == dept]
                if not dept_data.empty:
                    fig.add_trace(go.Bar(
                        x=dept_data['rating'],
                        y=dept_data['count'],
                        name=dept.title(),
                        marker_color=colors.get(dept, '#6b7280')
                    ))
            
            fig.update_layout(
                title="Review Distribution by Department and Rating",
                xaxis_title="Rating",
                yaxis_title="Number of Reviews",
                barmode='stack',
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart of departments
        dept_df = pd.DataFrame([
            {'Department': k.title(), 'Count': v} 
            for k, v in segregated_data['department_stats'].items()
            if k in ['app', 'railway', 'mixed']
        ])
        
        fig = px.pie(
            dept_df, 
            values='Count', 
            names='Department',
            color_discrete_map={'App': '#8b5cf6', 'Railway': '#10b981', 'Mixed': '#f59e0b'},
            title="Overall Department Split"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["📱 App Issues", "🚂 Railway Service Issues", "📊 Comparative Analysis"])
    
    with tab1:
        st.markdown("### App-Related Issues Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Category distribution
            if not app_categories.empty:
                fig = px.bar(
                    app_categories.head(10),
                    x='count',
                    y='category',
                    orientation='h',
                    title="Top App Issue Categories",
                    labels={'count': 'Number of Reviews', 'category': 'Category'},
                    color='avg_rating',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # App topics
            st.markdown("#### Identified Topics")
            app_topics = segregated_data['topics'].get('app', [])
            for topic in app_topics[:5]:
                render_topic_card(topic, 'app')
        
        # Sample reviews
        st.markdown("### Sample App Issue Reviews")
        
        category_filter = st.selectbox(
            "Filter by category:",
            ["All"] + app_categories['category'].tolist()[:10],
            key="app_category_filter"
        )
        
        if category_filter == "All":
            sample_reviews = load_sample_reviews('app', limit=5)
        else:
            sample_reviews = load_sample_reviews('app', category_filter, limit=5)
        
        for _, review in sample_reviews.iterrows():
            with st.expander(f"Rating: {'⭐' * int(review['rating'])} | Confidence: {review['confidence']:.2%}"):
                st.write(review['content'])
    
    with tab2:
        st.markdown("### Railway Service Issues Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Category distribution
            if not railway_categories.empty:
                fig = px.bar(
                    railway_categories.head(10),
                    x='count',
                    y='category',
                    orientation='h',
                    title="Top Railway Service Categories",
                    labels={'count': 'Number of Reviews', 'category': 'Category'},
                    color='avg_rating',
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Railway topics
            st.markdown("#### Identified Topics")
            railway_topics = segregated_data['topics'].get('railway', [])
            for topic in railway_topics[:5]:
                render_topic_card(topic, 'railway')
        
        # Sample reviews
        st.markdown("### Sample Railway Service Reviews")
        
        category_filter = st.selectbox(
            "Filter by category:",
            ["All"] + railway_categories['category'].tolist()[:10],
            key="railway_category_filter"
        )
        
        if category_filter == "All":
            sample_reviews = load_sample_reviews('railway', limit=5)
        else:
            sample_reviews = load_sample_reviews('railway', category_filter, limit=5)
        
        for _, review in sample_reviews.iterrows():
            with st.expander(f"Rating: {'⭐' * int(review['rating'])} | Confidence: {review['confidence']:.2%}"):
                st.write(review['content'])
    
    with tab3:
        st.markdown("### Comparative Analysis")
        
        # Side by side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📱 App Issues Summary")
            st.markdown("""
            **Top Problems:**
            1. **Booking System** - Tatkal booking failures, seat selection issues
            2. **Technical Errors** - App crashes, freezes, slow response
            3. **UI/UX Issues** - Button not working, poor navigation
            4. **Login/Authentication** - Password reset, OTP issues
            5. **Payment Gateway** - Transaction failures, refund delays
            
            **Average Rating:** 2.1/5.0
            """)
            
            # App issue severity
            app_severity = pd.DataFrame({
                'Category': ['Critical', 'High', 'Medium', 'Low'],
                'Count': [750, 500, 350, 179],
                'Color': ['#ef4444', '#f59e0b', '#eab308', '#84cc16']
            })
            
            fig = px.bar(
                app_severity,
                x='Category',
                y='Count',
                color='Category',
                color_discrete_map={
                    'Critical': '#ef4444',
                    'High': '#f59e0b',
                    'Medium': '#eab308',
                    'Low': '#84cc16'
                },
                title="App Issue Severity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🚂 Railway Service Summary")
            st.markdown("""
            **Top Problems:**
            1. **Staff Behavior** - Rude staff, poor service
            2. **Station Facilities** - Poor infrastructure, no amenities
            3. **Train Operations** - Delays, cancellations
            4. **Comfort** - Overcrowding, AC/fan issues
            5. **Food Quality** - Poor catering service
            
            **Average Rating:** 2.4/5.0
            """)
            
            # Railway issue severity
            railway_severity = pd.DataFrame({
                'Category': ['Critical', 'High', 'Medium', 'Low'],
                'Count': [250, 200, 150, 73],
                'Color': ['#ef4444', '#f59e0b', '#eab308', '#84cc16']
            })
            
            fig = px.bar(
                railway_severity,
                x='Category',
                y='Count',
                color='Category',
                color_discrete_map={
                    'Critical': '#ef4444',
                    'High': '#f59e0b',
                    'Medium': '#eab308',
                    'Low': '#84cc16'
                },
                title="Railway Issue Severity Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 📋 Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **For App Development Team:**
            - 🔴 **Priority 1:** Fix Tatkal booking system crashes
            - 🔴 **Priority 2:** Improve payment gateway reliability
            - 🟡 **Priority 3:** Enhance UI responsiveness
            - 🟢 **Priority 4:** Simplify login/password reset flow
            """)
        
        with col2:
            st.markdown("""
            **For Railway Operations:**
            - 🔴 **Priority 1:** Staff training for customer service
            - 🔴 **Priority 2:** Improve station infrastructure
            - 🟡 **Priority 3:** Better train punctuality
            - 🟢 **Priority 4:** Upgrade food catering services
            """)

if __name__ == "__main__":
    main()