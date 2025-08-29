#!/usr/bin/env python3
"""
Run full analysis on all 90,000 reviews
"""

import sys
sys.path.append('src/modeling')

from improved_topic_analysis import ImprovedTopicAnalyzer, print_topic_summary
import pandas as pd
import sqlite3
import logging
from pathlib import Path
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_all_reviews(db_path: str = 'data/reviews.db'):
    """Analyze ALL reviews in the database"""
    
    analyzer = ImprovedTopicAnalyzer()
    
    # Load ALL reviews (use content directly if normalized_text is not available)
    logger.info("Loading ALL reviews from database...")
    conn = sqlite3.connect(db_path)
    
    # Get all reviews - using content directly
    query = """
    SELECT 
        r.id,
        r.content,
        r.rating,
        COALESCE(p.normalized_text, r.content) as text_to_analyze
    FROM reviews r
    LEFT JOIN processed_reviews p ON r.id = p.review_id
    WHERE r.content IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    logger.info(f"Loaded {len(df)} reviews for analysis")
    
    # Process in batches to avoid memory issues
    batch_size = 10000
    all_classifications = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        batch_classifications = []
        for idx, row in batch.iterrows():
            if idx % 1000 == 0:
                logger.info(f"  Processed {idx}/{len(df)} reviews")
            
            text = row['text_to_analyze']
            classification = analyzer.classify_review(text)
            classification['review_id'] = row['id']
            batch_classifications.append(classification)
        
        all_classifications.extend(batch_classifications)
    
    # Add classifications to dataframe
    classification_df = pd.DataFrame(all_classifications)
    df = df.merge(classification_df, left_on='id', right_on='review_id', how='left')
    
    # Department statistics
    dept_stats = df['department'].value_counts()
    logger.info("\nDepartment Distribution:")
    for dept, count in dept_stats.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {dept}: {count} ({percentage:.1f}%)")
    
    # Category statistics
    app_categories = df[df['department'] == 'app']['top_app_category'].value_counts()
    railway_categories = df[df['department'] == 'railway']['top_railway_category'].value_counts()
    
    logger.info(f"\nTop App Issue Categories ({len(df[df['department'] == 'app'])} reviews):")
    for cat, count in app_categories.head(10).items():
        if cat:
            logger.info(f"  {cat}: {count}")
    
    logger.info(f"\nTop Railway Issue Categories ({len(df[df['department'] == 'railway'])} reviews):")
    for cat, count in railway_categories.head(10).items():
        if cat:
            logger.info(f"  {cat}: {count}")
    
    # Extract topics for each department with larger sample
    results = {
        'total_reviews': len(df),
        'department_stats': dept_stats.to_dict(),
        'app_categories': app_categories.to_dict(),
        'railway_categories': railway_categories.to_dict(),
        'topics': {'app': [], 'railway': [], 'mixed': []},
        'timestamp': datetime.now().isoformat()
    }
    
    # Process app reviews (use more reviews for better topics)
    app_reviews = df[df['department'] == 'app']
    if len(app_reviews) > 50:
        logger.info(f"\nExtracting topics from {len(app_reviews)} app reviews...")
        # Sample up to 10000 for topic modeling to balance quality and speed
        sample_size = min(10000, len(app_reviews))
        app_texts = app_reviews.sample(n=sample_size, random_state=42)['content'].tolist()
        app_topics = analyzer.extract_relevant_topics(app_texts, 'app', n_topics=20)
        
        for topic in app_topics['topics']:
            topic['category'] = analyzer.categorize_topic(topic['words'], 'app')
        
        results['topics']['app'] = app_topics['topics']
    
    # Process railway reviews
    railway_reviews = df[df['department'] == 'railway']
    if len(railway_reviews) > 50:
        logger.info(f"\nExtracting topics from {len(railway_reviews)} railway reviews...")
        sample_size = min(10000, len(railway_reviews))
        railway_texts = railway_reviews.sample(n=sample_size, random_state=42)['content'].tolist()
        railway_topics = analyzer.extract_relevant_topics(railway_texts, 'railway', n_topics=20)
        
        for topic in railway_topics['topics']:
            topic['category'] = analyzer.categorize_topic(topic['words'], 'railway')
        
        results['topics']['railway'] = railway_topics['topics']
    
    # Save ALL classifications to database
    logger.info("\nSaving ALL classifications to database...")
    cursor = conn.cursor()
    
    # Clear existing classifications
    cursor.execute("DELETE FROM review_classifications")
    
    # Insert all new classifications
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            logger.info(f"  Saved {idx}/{len(df)} classifications")
        
        cursor.execute("""
        INSERT OR REPLACE INTO review_classifications 
        (review_id, department, confidence, app_score, railway_score, top_app_category, top_railway_category)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row['id'], 
            row['department'], 
            row['confidence'],
            row['app_score'],
            row['railway_score'],
            row['top_app_category'],
            row['top_railway_category']
        ))
    
    conn.commit()
    conn.close()
    
    # Save complete results
    output_path = Path('data/models/full_analysis_results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\nFull analysis results saved to {output_path}")
    
    # Also update the improved_topics.pkl for backward compatibility
    with open('data/models/improved_topics.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("RUNNING FULL ANALYSIS ON ALL 90,000 REVIEWS")
    logger.info("="*80)
    
    results = analyze_all_reviews()
    
    print_topic_summary(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Total reviews analyzed: {results['total_reviews']:,}")
    print("="*80)