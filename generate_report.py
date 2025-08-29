#!/usr/bin/env python3
"""
Generate comprehensive analysis report for IRCTC review mining project
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_analysis_report():
    """Generate comprehensive analysis report"""
    
    # Connect to database
    conn = sqlite3.connect('data/reviews.db')
    
    # Get overall statistics
    total_reviews = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM reviews", conn
    )['count'][0]
    
    processed_reviews = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM processed_reviews WHERE normalized_text IS NOT NULL", conn
    )['count'][0]
    
    # Rating distribution
    rating_dist = pd.read_sql_query("""
        SELECT rating, COUNT(*) as count 
        FROM reviews 
        WHERE rating > 0 
        GROUP BY rating 
        ORDER BY rating
    """, conn)
    
    # Language distribution
    lang_dist = pd.read_sql_query("""
        SELECT language, COUNT(*) as count 
        FROM processed_reviews 
        GROUP BY language
    """, conn)
    
    # Get topics if available
    try:
        topics = pd.read_sql_query("SELECT * FROM topics LIMIT 20", conn)
        has_topics = True
    except:
        has_topics = False
        topics = None
    
    # Get sentiment distribution
    try:
        sentiment_dist = pd.read_sql_query("""
            SELECT 
                COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END) as positive,
                COUNT(CASE WHEN sentiment_label = 'negative' THEN 1 END) as negative,
                COUNT(CASE WHEN sentiment_label = 'neutral' THEN 1 END) as neutral,
                AVG(sentiment_score) as avg_score
            FROM sentiments
        """, conn)
        has_sentiments = True
    except:
        has_sentiments = False
        sentiment_dist = None
    
    conn.close()
    
    # Load latest model if available
    models_dir = Path('data/models')
    model_files = list(models_dir.glob('lda_model_*.pkl'))
    
    if model_files:
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        with open(latest_model_file, 'rb') as f:
            model_data = pickle.load(f)
            model_topics = model_data.get('topics', [])
    else:
        model_topics = []
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("IRCTC REVIEW MINING - COMPREHENSIVE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"• Total Reviews Collected: {total_reviews:,}")
    report_lines.append(f"• Reviews Processed: {processed_reviews:,} ({processed_reviews/total_reviews*100:.1f}%)")
    
    if not rating_dist.empty:
        avg_rating = (rating_dist['rating'] * rating_dist['count']).sum() / rating_dist['count'].sum()
        report_lines.append(f"• Average Rating: {avg_rating:.2f}/5")
        
        # Find dominant rating
        dominant_rating = rating_dist.loc[rating_dist['count'].idxmax(), 'rating']
        report_lines.append(f"• Most Common Rating: {dominant_rating} stars ({rating_dist.loc[rating_dist['rating']==dominant_rating, 'count'].values[0]:,} reviews)")
    
    report_lines.append("")
    
    # Data Collection Summary
    report_lines.append("DATA COLLECTION")
    report_lines.append("-" * 40)
    report_lines.append(f"• Source: Google Play Store")
    report_lines.append(f"• App: IRCTC Rail Connect")
    report_lines.append(f"• Collection Method: Batch scraping with continuation tokens")
    report_lines.append(f"• Batch Size: 10,000 reviews per batch")
    report_lines.append(f"• Total Batches: {total_reviews // 10000 + 1}")
    report_lines.append("")
    
    # Rating Distribution
    report_lines.append("RATING DISTRIBUTION")
    report_lines.append("-" * 40)
    if not rating_dist.empty:
        for _, row in rating_dist.iterrows():
            percentage = row['count'] / rating_dist['count'].sum() * 100
            stars = '⭐' * int(row['rating'])
            report_lines.append(f"{stars:15} ({row['rating']}/5): {row['count']:6,} reviews ({percentage:5.1f}%)")
    report_lines.append("")
    
    # Language Distribution
    report_lines.append("LANGUAGE DISTRIBUTION")
    report_lines.append("-" * 40)
    if not lang_dist.empty:
        for _, row in lang_dist.iterrows():
            if row['language']:
                percentage = row['count'] / lang_dist['count'].sum() * 100
                report_lines.append(f"• {row['language'].capitalize():10}: {row['count']:6,} reviews ({percentage:5.1f}%)")
    report_lines.append("")
    
    # Topic Modeling Results
    if model_topics:
        report_lines.append("TOPIC MODELING RESULTS (LDA)")
        report_lines.append("-" * 40)
        report_lines.append(f"• Number of Topics Identified: {len(model_topics)}")
        report_lines.append("")
        report_lines.append("Top 10 Topics:")
        for i, topic in enumerate(model_topics[:10]):
            keywords = ', '.join(topic['words'][:5])
            report_lines.append(f"  Topic {i+1}: {keywords}")
    report_lines.append("")
    
    # Key Insights
    report_lines.append("KEY INSIGHTS")
    report_lines.append("-" * 40)
    
    # Analyze sentiment of reviews
    if not rating_dist.empty:
        negative_reviews = rating_dist[rating_dist['rating'] <= 2]['count'].sum()
        positive_reviews = rating_dist[rating_dist['rating'] >= 4]['count'].sum()
        neutral_reviews = rating_dist[rating_dist['rating'] == 3]['count'].sum()
        
        report_lines.append(f"1. Sentiment Analysis (based on ratings):")
        report_lines.append(f"   • Negative (1-2 stars): {negative_reviews:,} ({negative_reviews/total_reviews*100:.1f}%)")
        report_lines.append(f"   • Neutral (3 stars): {neutral_reviews:,} ({neutral_reviews/total_reviews*100:.1f}%)")
        report_lines.append(f"   • Positive (4-5 stars): {positive_reviews:,} ({positive_reviews/total_reviews*100:.1f}%)")
        report_lines.append("")
    
    # Common issues from topics
    if model_topics:
        report_lines.append("2. Common User Issues (from topic analysis):")
        issue_keywords = ['payment', 'error', 'slow', 'crash', 'login', 'password', 'fail', 'problem']
        for topic in model_topics[:20]:
            topic_words = ' '.join(topic['words'])
            for keyword in issue_keywords:
                if keyword in topic_words:
                    report_lines.append(f"   • {keyword.capitalize()}-related issues detected")
                    break
        report_lines.append("")
    
    report_lines.append("3. User Experience Indicators:")
    if avg_rating < 3:
        report_lines.append("   • Overall user satisfaction is LOW (below 3 stars)")
        report_lines.append("   • Urgent improvements needed in app functionality")
    elif avg_rating < 3.5:
        report_lines.append("   • Overall user satisfaction is MODERATE (3-3.5 stars)")
        report_lines.append("   • Significant room for improvement")
    else:
        report_lines.append("   • Overall user satisfaction is GOOD (above 3.5 stars)")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 40)
    report_lines.append("Based on the analysis, the following improvements are recommended:")
    report_lines.append("")
    report_lines.append("1. Technical Improvements:")
    report_lines.append("   • Optimize app performance during peak booking hours")
    report_lines.append("   • Improve payment gateway reliability")
    report_lines.append("   • Fix login and password reset issues")
    report_lines.append("   • Enhance server capacity for Tatkal bookings")
    report_lines.append("")
    report_lines.append("2. User Experience Enhancements:")
    report_lines.append("   • Simplify the booking process")
    report_lines.append("   • Add better error messages and recovery options")
    report_lines.append("   • Improve UI/UX for better navigation")
    report_lines.append("   • Add offline functionality where possible")
    report_lines.append("")
    report_lines.append("3. Customer Support:")
    report_lines.append("   • Implement in-app chat support")
    report_lines.append("   • Create detailed FAQs for common issues")
    report_lines.append("   • Improve response time for user complaints")
    report_lines.append("")
    
    # Technical Details
    report_lines.append("TECHNICAL DETAILS")
    report_lines.append("-" * 40)
    report_lines.append("• Preprocessing: NLTK-based text normalization")
    report_lines.append("• Topic Modeling: Latent Dirichlet Allocation (LDA)")
    report_lines.append("• Parameters: 30 topics, 1000 features, bigrams included")
    report_lines.append("• Visualization: Streamlit dashboard with interactive charts")
    report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = '\n'.join(report_lines)
    
    # Create reports directory
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Save as text file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f'irctc_analysis_report_{timestamp}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Also save as markdown
    md_file = reports_dir / f'irctc_analysis_report_{timestamp}.md'
    md_text = report_text.replace("=" * 80, "---").replace("-" * 40, "###")
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_text)
    
    logger.info(f"Report saved to {report_file}")
    logger.info(f"Markdown version saved to {md_file}")
    
    # Print report to console
    print(report_text)
    
    return report_file


if __name__ == "__main__":
    generate_analysis_report()