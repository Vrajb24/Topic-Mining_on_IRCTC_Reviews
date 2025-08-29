#!/usr/bin/env python3
"""
Text Preprocessing Pipeline for IRCTC Reviews
Handles multilingual text (English, Hindi, Hinglish)
"""

import re
import string
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

# NLP libraries
import nltk
import spacy
from langdetect import detect, LangDetectException
from googletrans import Translator
from deep_translator import GoogleTranslator

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocessor for multilingual review text"""
    
    def __init__(self, db_path: str = 'data/reviews.db'):
        """Initialize preprocessor with database connection"""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        
        # Initialize language tools
        self.translator = GoogleTranslator(source='auto', target='en')
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("spaCy model not found. Some features may be limited.")
            self.nlp = None
        
        # Hindi/Hinglish patterns
        self.hindi_pattern = re.compile(r'[\u0900-\u097F]+')  # Devanagari script
        
        # Common IRCTC/Railway specific terms to preserve
        self.domain_terms = {
            'irctc', 'pnr', 'tatkal', 'rac', 'wl', 'cnf', 'booking', 
            'ticket', 'train', 'railway', 'coach', 'berth', 'platform',
            'otp', 'upi', 'payment', 'refund', 'cancellation', 'app'
        }
        
        # Expand contractions
        self.contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'ll": " will", "'ve": " have", "'re": " are",
            "'d": " would", "'m": " am", "it's": "it is", "let's": "let us"
        }
        
        # Get stopwords for multiple languages
        self.stop_words_en = set(stopwords.words('english'))
        try:
            self.stop_words_hi = set(stopwords.words('hindi'))
        except:
            self.stop_words_hi = set()
        
        # Custom stopwords to keep for domain relevance
        self.keep_words = {'no', 'not', 'very', 'too', 'worst', 'best', 'good', 'bad'}
        self.stop_words_en = self.stop_words_en - self.keep_words
        
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            # Check for Hindi characters
            if self.hindi_pattern.search(text):
                # Check if mixed with English (Hinglish)
                english_words = len(re.findall(r'[a-zA-Z]+', text))
                hindi_chars = len(self.hindi_pattern.findall(text))
                
                if english_words > 0 and hindi_chars > 0:
                    return 'hinglish'
                elif hindi_chars > english_words:
                    return 'hi'
            
            # Use langdetect for pure language detection
            lang = detect(text)
            return lang
        except:
            return 'en'  # Default to English
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (Indian format)
        text = re.sub(r'\b\d{10}\b', '', text)
        text = re.sub(r'\+91[-\s]?\d{10}', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def normalize_text(self, text: str, remove_stopwords: bool = True) -> str:
        """Normalize text with optional stopword removal"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [token for token in tokens 
                     if token not in self.stop_words_en 
                     or token in self.domain_terms]
        
        # Remove single characters except 'i'
        tokens = [token for token in tokens if len(token) > 1 or token == 'i']
        
        # Preserve domain-specific terms
        normalized_tokens = []
        for token in tokens:
            if token in self.domain_terms:
                normalized_tokens.append(token.upper())  # Keep domain terms in uppercase
            else:
                normalized_tokens.append(token)
        
        return ' '.join(normalized_tokens)
    
    def translate_to_english(self, text: str, source_lang: str = 'auto') -> str:
        """Translate non-English text to English"""
        try:
            if source_lang in ['hi', 'hinglish'] or self.hindi_pattern.search(text):
                # Use deep-translator for better Hindi support
                translated = self.translator.translate(text)
                return translated if translated else text
            return text
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
    
    def extract_features(self, text: str) -> Dict:
        """Extract additional features from text"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'has_numbers': bool(re.search(r'\d', text)),
            'sentiment_words': {
                'positive': 0,
                'negative': 0
            }
        }
        
        # Count sentiment words
        positive_words = {'good', 'great', 'excellent', 'best', 'awesome', 'nice', 'perfect', 'love'}
        negative_words = {'bad', 'worst', 'terrible', 'awful', 'horrible', 'pathetic', 'useless', 'waste'}
        
        text_lower = text.lower()
        for word in positive_words:
            features['sentiment_words']['positive'] += text_lower.count(word)
        for word in negative_words:
            features['sentiment_words']['negative'] += text_lower.count(word)
        
        return features
    
    def preprocess_review(self, review: str, translate: bool = False) -> Dict:
        """Complete preprocessing pipeline for a single review"""
        # Detect language
        lang = self.detect_language(review)
        
        # Clean text
        cleaned = self.clean_text(review)
        
        # Translate if needed
        translated = cleaned
        if translate and lang != 'en':
            translated = self.translate_to_english(cleaned, lang)
        
        # Normalize
        normalized = self.normalize_text(translated)
        
        # Extract features
        features = self.extract_features(cleaned)
        
        return {
            'original': review,
            'cleaned': cleaned,
            'translated': translated,
            'normalized': normalized,
            'language': lang,
            'features': features
        }
    
    def process_batch(self, reviews: List[str], translate: bool = False) -> pd.DataFrame:
        """Process a batch of reviews"""
        results = []
        
        for review in tqdm(reviews, desc="Processing reviews"):
            try:
                processed = self.preprocess_review(review, translate)
                results.append(processed)
            except Exception as e:
                logger.error(f"Error processing review: {e}")
                results.append({
                    'original': review,
                    'cleaned': '',
                    'translated': '',
                    'normalized': '',
                    'language': 'unknown',
                    'features': {}
                })
        
        return pd.DataFrame(results)
    
    def process_database_reviews(self, limit: Optional[int] = None, translate: bool = False) -> pd.DataFrame:
        """Process reviews from database"""
        # Load reviews
        query = "SELECT id, content, rating FROM reviews"
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded {len(df)} reviews from database")
        
        # Process reviews
        processed_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
            try:
                processed = self.preprocess_review(row['content'], translate)
                processed['review_id'] = row['id']
                processed['rating'] = row['rating']
                processed_data.append(processed)
            except Exception as e:
                logger.error(f"Error processing review {row['id']}: {e}")
        
        result_df = pd.DataFrame(processed_data)
        
        # Save processed data
        self.save_processed_data(result_df)
        
        return result_df
    
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to database"""
        # Create processed reviews table if not exists
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                cleaned_text TEXT,
                translated_text TEXT,
                normalized_text TEXT,
                language TEXT,
                text_length INTEGER,
                word_count INTEGER,
                positive_words INTEGER,
                negative_words INTEGER,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (review_id) REFERENCES reviews (id)
            )
        """)
        
        # Insert processed data
        for _, row in df.iterrows():
            features = row.get('features', {})
            sentiment = features.get('sentiment_words', {})
            
            cursor.execute("""
                INSERT OR REPLACE INTO processed_reviews (
                    review_id, cleaned_text, translated_text, normalized_text,
                    language, text_length, word_count, positive_words, negative_words
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('review_id'),
                row.get('cleaned'),
                row.get('translated'),
                row.get('normalized'),
                row.get('language'),
                features.get('text_length'),
                features.get('word_count'),
                sentiment.get('positive', 0),
                sentiment.get('negative', 0)
            ))
        
        self.conn.commit()
        logger.info(f"Saved {len(df)} processed reviews to database")
    
    def get_language_distribution(self) -> Dict:
        """Get distribution of languages in reviews"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT language, COUNT(*) as count
            FROM processed_reviews
            GROUP BY language
            ORDER BY count DESC
        """)
        
        return dict(cursor.fetchall())
    
    def get_preprocessing_stats(self) -> Dict:
        """Get statistics about preprocessing"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total processed
        cursor.execute("SELECT COUNT(*) FROM processed_reviews")
        stats['total_processed'] = cursor.fetchone()[0]
        
        # Language distribution
        stats['language_distribution'] = self.get_language_distribution()
        
        # Average text lengths
        cursor.execute("""
            SELECT 
                AVG(text_length) as avg_length,
                AVG(word_count) as avg_words,
                AVG(positive_words) as avg_positive,
                AVG(negative_words) as avg_negative
            FROM processed_reviews
        """)
        
        result = cursor.fetchone()
        stats['avg_text_length'] = round(result[0], 2) if result[0] else 0
        stats['avg_word_count'] = round(result[1], 2) if result[1] else 0
        stats['avg_positive_words'] = round(result[2], 2) if result[2] else 0
        stats['avg_negative_words'] = round(result[3], 2) if result[3] else 0
        
        return stats

def main():
    """Main function to run preprocessing"""
    logger.info("Starting preprocessing pipeline...")
    
    preprocessor = TextPreprocessor()
    
    # Process reviews (limit to 1000 for testing, remove limit for full processing)
    df = preprocessor.process_database_reviews(limit=1000, translate=False)
    
    # Get statistics
    stats = preprocessor.get_preprocessing_stats()
    
    logger.info("Preprocessing Statistics:")
    logger.info(f"Total Processed: {stats['total_processed']}")
    logger.info(f"Language Distribution: {stats['language_distribution']}")
    logger.info(f"Average Text Length: {stats['avg_text_length']}")
    logger.info(f"Average Word Count: {stats['avg_word_count']}")
    logger.info(f"Average Positive Words: {stats['avg_positive_words']}")
    logger.info(f"Average Negative Words: {stats['avg_negative_words']}")
    
    # Sample processed reviews
    print("\n📝 Sample Processed Reviews:")
    print("="*50)
    
    for i in range(min(3, len(df))):
        print(f"\nReview {i+1}:")
        print(f"Original: {df.iloc[i]['original'][:100]}...")
        print(f"Cleaned: {df.iloc[i]['cleaned'][:100]}...")
        print(f"Normalized: {df.iloc[i]['normalized'][:100]}...")
        print(f"Language: {df.iloc[i]['language']}")
        print(f"Rating: {df.iloc[i]['rating']}")
    
    return df

if __name__ == "__main__":
    processed_df = main()