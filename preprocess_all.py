#!/usr/bin/env python3
"""
Preprocess all 90k reviews for topic modeling
Simplified version without spacy dependency
"""

import sqlite3
import re
import string
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime
import nltk
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')


class SimplePreprocessor:
    """Simple text preprocessor for reviews"""
    
    def __init__(self):
        """Initialize with stop words"""
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stop words for IRCTC reviews
        custom_stops = {
            'irctc', 'app', 'application', 'railway', 'train', 
            'ticket', 'booking', 'book', 'please', 'need', 
            'want', 'can', 'will', 'would', 'could', 'should'
        }
        self.stop_words.update(custom_stops)
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove stop words and short tokens
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        # Join back
        processed = ' '.join(tokens)
        
        return processed if processed else None
    
    def detect_language(self, text):
        """Simple language detection based on character patterns"""
        if not text:
            return 'unknown'
        
        # Check for Hindi/Devanagari characters
        devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        if devanagari_pattern.search(text):
            return 'hindi'
        
        # Check for common Hindi romanized words
        hindi_words = ['hai', 'nahi', 'karo', 'aap', 'mein', 'kar', 'ho', 'ka', 'ki', 'ke']
        text_lower = text.lower()
        hindi_count = sum(1 for word in hindi_words if word in text_lower.split())
        
        if hindi_count >= 2:
            return 'hinglish'
        
        return 'english'


def process_batch(reviews_batch, preprocessor):
    """Process a batch of reviews"""
    results = []
    
    for idx, row in reviews_batch.iterrows():
        content = row['content']
        
        # Detect language
        language = preprocessor.detect_language(content)
        
        # Preprocess
        processed = preprocessor.preprocess(content)
        
        results.append({
            'review_id': row['id'],
            'original_text': content,
            'normalized_text': processed,
            'language': language,
            'processed_at': datetime.now()
        })
    
    return results


def main():
    """Main preprocessing function"""
    logger.info("="*60)
    logger.info("PREPROCESSING ALL REVIEWS")
    logger.info("="*60)
    
    # Connect to database
    conn = sqlite3.connect('data/reviews.db')
    
    # Get total review count
    total_count = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM reviews", conn
    )['count'][0]
    
    logger.info(f"Total reviews to process: {total_count}")
    
    # Create processed reviews table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processed_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT UNIQUE,
            original_text TEXT,
            normalized_text TEXT,
            language TEXT,
            processed_at TIMESTAMP
        )
    """)
    
    # Initialize preprocessor
    preprocessor = SimplePreprocessor()
    
    # Process in batches
    batch_size = 1000
    processed_count = 0
    
    # Check how many are already processed
    existing_count = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM processed_reviews", conn
    )['count'][0]
    
    logger.info(f"Already processed: {existing_count}")
    
    # Get unprocessed reviews
    query = """
        SELECT r.* FROM reviews r
        LEFT JOIN processed_reviews p ON r.id = p.review_id
        WHERE p.review_id IS NULL
        LIMIT ?
    """
    
    while True:
        # Fetch batch
        batch_df = pd.read_sql_query(query, conn, params=(batch_size,))
        
        if batch_df.empty:
            break
        
        logger.info(f"Processing batch of {len(batch_df)} reviews...")
        
        # Process batch
        results = process_batch(batch_df, preprocessor)
        
        # Save to database
        for result in results:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO processed_reviews 
                    (review_id, original_text, normalized_text, language, processed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    result['review_id'],
                    result['original_text'],
                    result['normalized_text'],
                    result['language'],
                    result['processed_at']
                ))
            except Exception as e:
                logger.error(f"Error inserting review {result['review_id']}: {e}")
        
        conn.commit()
        processed_count += len(batch_df)
        logger.info(f"Total processed: {existing_count + processed_count}/{total_count}")
    
    # Get final statistics
    stats = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN normalized_text IS NOT NULL THEN 1 END) as valid,
            COUNT(CASE WHEN language = 'english' THEN 1 END) as english,
            COUNT(CASE WHEN language = 'hindi' THEN 1 END) as hindi,
            COUNT(CASE WHEN language = 'hinglish' THEN 1 END) as hinglish
        FROM processed_reviews
    """, conn)
    
    conn.close()
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total processed: {stats['total'][0]}")
    logger.info(f"Valid (non-empty): {stats['valid'][0]}")
    logger.info(f"English: {stats['english'][0]}")
    logger.info(f"Hindi: {stats['hindi'][0]}")
    logger.info(f"Hinglish: {stats['hinglish'][0]}")


if __name__ == "__main__":
    main()