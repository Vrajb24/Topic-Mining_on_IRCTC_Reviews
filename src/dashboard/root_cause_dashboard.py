#!/usr/bin/env python3
"""
Root Cause Analysis Dashboard for IRCTC Reviews
Comprehensive visualization of root cause analysis results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from pathlib import Path
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="IRCTC Root Cause Analysis Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3b82f6;
    }
    .metric-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #334155;
        height: 100%;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .root-cause-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .recommendation-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: white;
    }
    .recommendation-card h4 {
        color: #60a5fa;
        margin-bottom: 0.5rem;
    }
    .recommendation-card p {
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .severity-critical {
        background: #fee2e2;
        border-left-color: #ef4444;
    }
    .severity-high {
        background: #fed7aa;
        border-left-color: #f97316;
    }
    .severity-medium {
        background: #fef3c7;
        border-left-color: #f59e0b;
    }
    .anomaly-card {
        background: #7f1d1d;
        border: 1px solid #991b1b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        color: #fca5a5;
    }
    .cluster-card {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        transition: transform 0.2s;
        color: #e2e8f0;
    }
    .cluster-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        background: #334155;
    }
    .stTabs {
        margin-top: 3rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: #1e293b;
        padding: 0.75rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600;
        background: #0f172a;
        color: #94a3b8;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #334155;
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important;
        color: white !important;
        border-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_analysis_results():
    """Load root cause analysis results"""
    try:
        with open('data/analysis/root_cause_analysis.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def render_overview_metrics(results):
    """Render overview metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_scenarios = len(results['contextual_patterns']['identified_scenarios'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Identified Scenarios</div>
            <div class="metric-value">{total_scenarios}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_anomalies = results['anomaly_detection']['summary']['total_anomalies']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Detected Anomalies</div>
            <div class="metric-value">{total_anomalies}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_clusters = results['clustering_results']['total_clusters']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Issue Clusters</div>
            <div class="metric-value">{total_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        top_root_causes = len(results['top_root_causes'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Root Causes</div>
            <div class="metric-value">{top_root_causes}</div>
        </div>
        """, unsafe_allow_html=True)

def render_temporal_analysis(temporal_data):
    """Render temporal analysis charts"""
    st.markdown('<h2 class="section-header">📊 Temporal Pattern Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Daily trends chart
        if temporal_data['daily_trends']:
            df_trends = pd.DataFrame(temporal_data['daily_trends'])
            
            # Group by date and department
            fig = go.Figure()
            
            for dept in df_trends['department'].unique():
                if pd.notna(dept):
                    dept_data = df_trends[df_trends['department'] == dept].groupby('date')['count'].sum()
                    fig.add_trace(go.Scatter(
                        x=dept_data.index,
                        y=dept_data.values,
                        mode='lines+markers',
                        name=dept.title(),
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Issue Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Number of Issues",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tatkal peak hours
        if temporal_data['tatkal_peak_hours']:
            df_hours = pd.DataFrame(temporal_data['tatkal_peak_hours'])
            
            fig = go.Figure(go.Bar(
                x=df_hours['hour'],
                y=df_hours['count'],
                marker_color='#8b5cf6'
            ))
            
            fig.update_layout(
                title="Tatkal Booking Issues by Hour",
                xaxis_title="Hour of Day",
                yaxis_title="Number of Issues",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Spike detection
    if temporal_data['spikes']:
        st.subheader("🚨 Detected Spikes")
        for spike in temporal_data['spikes'][:5]:
            severity_color = "#ef4444" if spike['severity'] == 'high' else "#f59e0b"
            st.markdown(f"""
            <div class="anomaly-card">
                <strong style="color: {severity_color};">{spike['date']}</strong> - 
                {spike['category']}: {spike['count']} issues 
                (Z-score: {spike['z_score']})
            </div>
            """, unsafe_allow_html=True)

def render_root_causes(results):
    """Render root cause analysis"""
    st.markdown('<h2 class="section-header">🎯 Top Root Causes</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Root causes chart
        if results['top_root_causes']:
            df_causes = pd.DataFrame(results['top_root_causes'][:10])
            
            fig = go.Figure(go.Bar(
                y=df_causes['cause'],
                x=df_causes['frequency'],
                orientation='h',
                marker=dict(
                    color=df_causes['frequency'],
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="Root Cause Frequency Analysis",
                xaxis_title="Frequency",
                yaxis_title="Root Cause",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top root causes with solutions
        st.subheader("Solutions")
        for i, cause in enumerate(results['top_root_causes'][:5], 1):
            st.markdown(f"""
            <div class="root-cause-card">
                <h4 style="color: white; margin: 0;">#{i} {cause['cause']}</h4>
                <p style="margin: 0.5rem 0; opacity: 0.9;">Frequency: {cause['frequency']}</p>
                <p style="margin: 0; font-size: 0.9rem;">
                    <strong>Solution:</strong> {cause['solution']}
                </p>
            </div>
            """, unsafe_allow_html=True)

def render_severity_analysis(severity_data):
    """Render severity analysis"""
    st.markdown('<h2 class="section-header">⚠️ Severity Classification</h2>', unsafe_allow_html=True)
    
    if severity_data and 'severity_distribution' in severity_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Severity distribution chart
            severity_counts = []
            severity_labels = []
            colors = []
            
            severity_colors = {
                'critical': '#ef4444',
                'high': '#f97316',
                'medium': '#f59e0b',
                'low': '#84cc16'
            }
            
            for level in ['critical', 'high', 'medium', 'low']:
                if level in severity_data['severity_distribution']:
                    severity_counts.append(severity_data['severity_distribution'][level]['count'])
                    severity_labels.append(level.title())
                    colors.append(severity_colors[level])
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=severity_labels,
                    values=severity_counts,
                    hole=0.4,
                    marker=dict(colors=colors)
                )
            ])
            
            fig.update_layout(
                title="Issue Severity Distribution",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity metrics
            st.subheader("Impact Analysis")
            
            for level in ['critical', 'high', 'medium', 'low']:
                if level in severity_data['severity_distribution']:
                    data = severity_data['severity_distribution'][level]
                    avg_impact = data['avg_impact']
                    count = data['count']
                    
                    severity_class = f"severity-{level}"
                    st.markdown(f"""
                    <div class="recommendation-card {severity_class}">
                        <strong>{level.title()}</strong><br>
                        Count: {count}<br>
                        Avg Impact: {avg_impact:.1f}/10
                    </div>
                    """, unsafe_allow_html=True)

def render_clustering_analysis(clustering_data):
    """Render clustering analysis"""
    st.markdown('<h2 class="section-header">🔬 Issue Clustering Analysis</h2>', unsafe_allow_html=True)
    
    if clustering_data and 'clusters' in clustering_data:
        # Create columns for cluster cards
        clusters = clustering_data['clusters']
        
        # Show top 6 clusters in 2 rows of 3
        for row in range(0, min(6, len(clusters)), 3):
            cols = st.columns(3)
            
            for i, col in enumerate(cols):
                if row + i < len(clusters):
                    cluster = clusters[row + i]
                    
                    with col:
                        st.markdown(f"""
                        <div class="cluster-card">
                            <h4>Cluster {cluster['cluster_id'] + 1}</h4>
                            <p><strong>Size:</strong> {cluster['size']} issues</p>
                            <p><strong>Key Terms:</strong> {', '.join(cluster['top_terms'][:3])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if cluster['sample_reviews']:
                            with st.expander("Sample Reviews"):
                                for review in cluster['sample_reviews'][:2]:
                                    st.write(f"- {review[:150]}...")

def render_five_why_analysis(five_why_data):
    """Render 5-Why analysis"""
    st.markdown('<h2 class="section-header">🔄 5-Why Analysis</h2>', unsafe_allow_html=True)
    
    if five_why_data and 'why_trees' in five_why_data:
        # Display why trees
        why_trees = five_why_data['why_trees']
        
        if why_trees:
            # Create tabs for different categories
            category_names = list(why_trees.keys())[:5]
            if category_names:
                tabs = st.tabs(category_names)
                
                for tab, category in zip(tabs, category_names):
                    with tab:
                        tree = why_trees[category]
                        
                        if 'root_problems' in tree:
                            for problem_data in tree['root_problems'][:3]:
                                st.markdown(f"**Problem:** {problem_data['problem']}")
                                
                                if problem_data['causes']:
                                    st.markdown("**Root Causes:**")
                                    for cause, freq in problem_data['causes']:
                                        st.markdown(f"- {cause} (Frequency: {freq})")
                                
                                st.markdown("---")

def render_anomaly_detection(anomaly_data):
    """Render anomaly detection results"""
    st.markdown('<h2 class="section-header">🚨 Anomaly Detection</h2>', unsafe_allow_html=True)
    
    if anomaly_data and 'anomalies' in anomaly_data:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Anomaly timeline
            anomalies = anomaly_data['anomalies']
            
            if anomalies:
                df_anomalies = pd.DataFrame(anomalies)
                
                # Create scatter plot
                fig = go.Figure()
                
                # Group by anomaly type
                for anomaly_type in df_anomalies['type'].unique():
                    type_data = df_anomalies[df_anomalies['type'] == anomaly_type]
                    
                    color = '#ef4444' if 'spike' in anomaly_type else '#3b82f6'
                    
                    fig.add_trace(go.Scatter(
                        x=type_data['date'],
                        y=type_data['z_score'],
                        mode='markers',
                        name=anomaly_type.replace('_', ' ').title(),
                        marker=dict(
                            size=12,
                            color=color,
                            symbol='diamond' if 'rating' in anomaly_type else 'circle'
                        )
                    ))
                
                fig.update_layout(
                    title="Anomaly Detection Timeline",
                    xaxis_title="Date",
                    yaxis_title="Z-Score",
                    height=400,
                    hovermode='closest'
                )
                
                # Add threshold lines
                fig.add_hline(y=2, line_dash="dash", line_color="gray", annotation_text="Threshold")
                fig.add_hline(y=-2, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anomaly summary
            st.subheader("Summary")
            
            summary = anomaly_data.get('summary', {})
            st.metric("Total Anomalies", summary.get('total_anomalies', 0))
            st.metric("Volume Anomalies", summary.get('volume_anomalies', 0))
            st.metric("Rating Anomalies", summary.get('rating_anomalies', 0))
            
            # Recent anomalies
            st.subheader("Recent Anomalies")
            for anomaly in anomalies[:3]:
                st.markdown(f"""
                <div class="anomaly-card">
                    <strong>{anomaly['date']}</strong><br>
                    Type: {anomaly['type'].replace('_', ' ').title()}<br>
                    Z-Score: {anomaly['z_score']}
                </div>
                """, unsafe_allow_html=True)

def render_recommendations(recommendations):
    """Render recommendations"""
    st.markdown('<h2 class="section-header">💡 Recommendations</h2>', unsafe_allow_html=True)
    
    if recommendations:
        col1, col2 = st.columns(2)
        
        # Split recommendations
        mid_point = len(recommendations) // 2
        
        with col1:
            for rec in recommendations[:mid_point]:
                impact_color = {
                    'critical': '#ef4444',
                    'high': '#f97316',
                    'medium': '#f59e0b',
                    'low': '#84cc16'
                }.get(rec['impact'], '#6b7280')
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>Priority {rec['priority']}</h4>
                    <p><strong>Issue:</strong> {rec['issue']}</p>
                    <p><strong>Solution:</strong> {rec['solution']}</p>
                    <p>
                        <span style="color: {impact_color};">Impact: {rec['impact'].upper()}</span> | 
                        Effort: {rec['estimated_effort'].upper()}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            for rec in recommendations[mid_point:]:
                impact_color = {
                    'critical': '#ef4444',
                    'high': '#f97316',
                    'medium': '#f59e0b',
                    'low': '#84cc16'
                }.get(rec['impact'], '#6b7280')
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>Priority {rec['priority']}</h4>
                    <p><strong>Issue:</strong> {rec['issue']}</p>
                    <p><strong>Solution:</strong> {rec['solution']}</p>
                    <p>
                        <span style="color: {impact_color};">Impact: {rec['impact'].upper()}</span> | 
                        Effort: {rec['estimated_effort'].upper()}
                    </p>
                </div>
                """, unsafe_allow_html=True)

def render_statistical_insights(statistical_data):
    """Render statistical insights"""
    st.markdown('<h2 class="section-header">📈 Statistical Insights</h2>', unsafe_allow_html=True)
    
    if statistical_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Correlation Analysis")
            
            correlations = statistical_data.get('correlations', {})
            for key, value in correlations.items():
                if value is not None:
                    label = key.replace('_', ' ').title()
                    color = '#ef4444' if value < -0.3 else '#22c55e' if value > 0.3 else '#6b7280'
                    st.markdown(f"""
                    <div style="margin-bottom: 0.5rem;">
                        <strong>{label}:</strong> 
                        <span style="color: {color};">{value:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Department Comparison")
            
            test_data = statistical_data.get('statistical_test', {})
            if test_data:
                t_stat = test_data.get('t_statistic')
                p_val = test_data.get('p_value')
                t_stat_str = f"{t_stat:.3f}" if t_stat is not None else "N/A"
                p_val_str = f"{p_val:.4f}" if p_val is not None else "N/A"
                
                st.markdown(f"""
                <div class="metric-card">
                    <p style="color: #e2e8f0;"><strong>T-Statistic:</strong> {t_stat_str}</p>
                    <p style="color: #e2e8f0;"><strong>P-Value:</strong> {p_val_str}</p>
                    <p style="color: #e2e8f0;"><strong>Result:</strong> {test_data.get('significance', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.subheader("Summary Statistics")
            
            summary = statistical_data.get('summary_stats', {})
            
            if 'app' in summary and 'railway' in summary:
                # Create comparison
                metrics_df = pd.DataFrame({
                    'Department': ['App', 'Railway'],
                    'Mean Rating': [
                        summary['app'].get('mean_rating', 0),
                        summary['railway'].get('mean_rating', 0)
                    ],
                    'Count': [
                        summary['app'].get('count', 0),
                        summary['railway'].get('count', 0)
                    ]
                })
                
                fig = go.Figure(data=[
                    go.Bar(name='Mean Rating', x=metrics_df['Department'], y=metrics_df['Mean Rating']),
                ])
                
                fig.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

def main():
    # Load analysis results
    results = load_analysis_results()
    
    if not results:
        st.error("No root cause analysis results found. Please run src/analysis/root_cause_analyzer.py first.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🔍 IRCTC Root Cause Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Last updated
    if 'analysis_timestamp' in results:
        timestamp = datetime.fromisoformat(results['analysis_timestamp'])
        st.caption(f"Last updated: {timestamp.strftime('%B %d, %Y at %I:%M %p')}")
    
    # Overview metrics
    render_overview_metrics(results)
    
    # Add spacing between metrics and tabs
    st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Temporal Analysis",
        "🎯 Root Causes",
        "⚠️ Severity & Impact",
        "🔬 Advanced Analytics",
        "💡 Recommendations"
    ])
    
    with tab1:
        if 'temporal_patterns' in results:
            render_temporal_analysis(results['temporal_patterns'])
        
        if 'anomaly_detection' in results:
            render_anomaly_detection(results['anomaly_detection'])
    
    with tab2:
        render_root_causes(results)
        
        if 'five_why_analysis' in results:
            render_five_why_analysis(results['five_why_analysis'])
    
    with tab3:
        if 'severity_classification' in results:
            render_severity_analysis(results['severity_classification'])
        
        # Contextual patterns
        if 'contextual_patterns' in results:
            st.markdown('<h2 class="section-header">🎯 Contextual Patterns</h2>', unsafe_allow_html=True)
            
            scenarios = results['contextual_patterns'].get('identified_scenarios', {})
            if scenarios:
                # Sort by count
                sorted_scenarios = sorted(scenarios.items(), key=lambda x: x[1]['count'], reverse=True)
                
                for scenario_name, scenario_data in sorted_scenarios[:5]:
                    if scenario_data['count'] > 0:
                        with st.expander(f"{scenario_name.replace('_', ' ').title()} ({scenario_data['count']} occurrences)"):
                            st.markdown(f"**Root Cause:** {scenario_data['root_cause']}")
                            st.markdown(f"**Solution:** {scenario_data['solution']}")
    
    with tab4:
        if 'clustering_results' in results:
            render_clustering_analysis(results['clustering_results'])
        
        if 'statistical_analysis' in results:
            render_statistical_insights(results['statistical_analysis'])
    
    with tab5:
        if 'recommendations' in results:
            render_recommendations(results['recommendations'])
        
        # Export functionality
        st.markdown("### 📥 Export Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download Full Report"):
                # Convert results to JSON for download
                json_str = json.dumps(results, default=str, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"root_cause_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Generate Executive Summary"):
                st.info("Executive summary generation will be available in the next update.")
        
        with col3:
            if st.button("Schedule Regular Analysis"):
                st.info("Automated scheduling will be available in the next update.")

if __name__ == "__main__":
    main()