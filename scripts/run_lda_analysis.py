#!/usr/bin/env python3
"""
Run LDA Topic Modeling on the full dataset
Simplified version focusing on basic LDA analysis
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from datetime import datetime

# ML libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_reviews(db_path='data/reviews.db'):
    """Load processed reviews from database"""
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            r.id,
            r.content as original,
            p.normalized_text as processed,
            r.rating,
            r.date_posted
        FROM reviews r
        LEFT JOIN processed_reviews p ON r.id = p.review_id
        WHERE p.normalized_text IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df)} processed reviews")
    return df


def run_lda_analysis(texts, n_topics=30):
    """Run LDA topic modeling"""
    logger.info("="*60)
    logger.info("RUNNING LDA TOPIC MODELING")
    logger.info("="*60)
    
    # Vectorization
    logger.info("Creating document-term matrix...")
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    logger.info(f"Document-term matrix shape: {doc_term_matrix.shape}")
    logger.info(f"Sparsity: {(doc_term_matrix.nnz / (doc_term_matrix.shape[0] * doc_term_matrix.shape[1]) * 100):.2f}%")
    
    # LDA model
    logger.info(f"Training LDA with {n_topics} topics...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method='batch',
        learning_offset=50,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    doc_topic_matrix = lda.fit_transform(doc_term_matrix)
    
    # Calculate metrics
    perplexity = lda.perplexity(doc_term_matrix)
    logger.info(f"Perplexity: {perplexity:.2f}")
    
    # Get topic words
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    logger.info("\nTop Topics:")
    logger.info("="*60)
    
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-15:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = topic[top_indices]
        
        topics.append({
            'topic_id': topic_idx,
            'words': top_words[:10],
            'weights': top_weights[:10].tolist()
        })
        
        # Print top 5 words for each topic
        logger.info(f"Topic {topic_idx}: {', '.join(top_words[:5])}")
    
    # Save model
    models_dir = Path('data/models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = models_dir / f'lda_model_{timestamp}.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': lda,
            'vectorizer': vectorizer,
            'topics': topics,
            'doc_topic_matrix': doc_topic_matrix
        }, f)
    
    logger.info(f"\nModel saved to {model_path}")
    
    # Save topics to database
    conn = sqlite3.connect('data/reviews.db')
    cursor = conn.cursor()
    
    # Create topics table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY,
            name TEXT,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert topics
    for topic in topics[:20]:  # Save top 20 topics
        topic_name = f"Topic_{topic['topic_id']}"
        keywords = ', '.join(topic['words'][:10])
        cursor.execute(
            "INSERT INTO topics (id, topic_name, topic_words, topic_size) VALUES (?, ?, ?, ?)",
            (topic['topic_id'], topic_name, keywords, 100)  # Default size 100
        )
    
    conn.commit()
    conn.close()
    
    logger.info(f"Saved {min(20, len(topics))} topics to database")
    
    return lda, vectorizer, topics, doc_topic_matrix


def analyze_topic_distribution(doc_topic_matrix, df):
    """Analyze topic distribution across ratings"""
    logger.info("\n" + "="*60)
    logger.info("TOPIC DISTRIBUTION ANALYSIS")
    logger.info("="*60)
    
    # Get dominant topic for each document
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    
    # Add to dataframe
    df['dominant_topic'] = dominant_topics
    
    # Analyze by rating
    rating_topic_dist = df.groupby(['rating', 'dominant_topic']).size().unstack(fill_value=0)
    
    logger.info("\nTop topics by rating:")
    for rating in sorted(df['rating'].unique()):
        if rating > 0:
            top_topics = rating_topic_dist.loc[rating].nlargest(3)
            logger.info(f"Rating {rating}: Topics {top_topics.index.tolist()}")
    
    return dominant_topics


def main():
    """Main function"""
    logger.info("="*60)
    logger.info("LDA TOPIC MODELING ON FULL DATASET")
    logger.info("="*60)
    
    # Load processed reviews
    df = load_processed_reviews()
    
    if len(df) == 0:
        logger.error("No processed reviews found. Run preprocessing first.")
        return
    
    # Use processed text
    texts = df['processed'].dropna().tolist()
    logger.info(f"Using {len(texts)} processed reviews for modeling")
    
    # Run LDA
    lda, vectorizer, topics, doc_topic_matrix = run_lda_analysis(texts, n_topics=30)
    
    # Analyze distribution
    dominant_topics = analyze_topic_distribution(doc_topic_matrix, df)
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)
    logger.info(f"Total reviews analyzed: {len(texts)}")
    logger.info(f"Number of topics: {len(topics)}")
    logger.info(f"Average topic probability: {doc_topic_matrix.mean():.4f}")
    logger.info(f"Max topic probability: {doc_topic_matrix.max():.4f}")
    
    # Topic sizes
    topic_sizes = pd.Series(dominant_topics).value_counts().head(10)
    logger.info("\nTop 10 topics by document count:")
    for topic_id, count in topic_sizes.items():
        percentage = (count / len(texts)) * 100
        logger.info(f"Topic {topic_id}: {count} documents ({percentage:.1f}%)")
    
    logger.info("\n" + "="*60)
    logger.info("LDA ANALYSIS COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    main()