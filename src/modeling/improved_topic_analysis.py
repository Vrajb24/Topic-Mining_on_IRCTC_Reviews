#!/usr/bin/env python3
"""
Improved Topic Analysis with Department Segregation
Separates App issues from Railway Service issues
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging
import pickle
from datetime import datetime
import re
from typing import List, Dict, Tuple

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTopicAnalyzer:
    """Improved topic analyzer with department segregation"""
    
    def __init__(self):
        """Initialize the analyzer"""
        
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
        
        # App-specific keywords and patterns
        self.app_patterns = {
            'ui_issues': [
                'button', 'screen', 'display', 'interface', 'ui', 'ux', 'design', 'layout', 
                'menu', 'page', 'scroll', 'tap', 'click', 'navigate', 'back button'
            ],
            'technical_errors': [
                'crash', 'hang', 'freeze', 'bug', 'error', 'glitch', 'lag', 'slow', 
                'loading', 'response', 'stuck', 'not working', 'blank screen', 'force close'
            ],
            'login_auth': [
                'login', 'password', 'otp', 'authentication', 'signin', 'signup', 
                'register', 'account', 'profile', 'forgot password', 'reset', 'captcha',
                'user id', 'username', 'verification'
            ],
            'payment_issues': [
                'payment', 'transaction', 'refund', 'debit', 'credit', 'upi', 'wallet', 
                'netbanking', 'card', 'money', 'deducted', 'failed', 'pending', 'gateway'
            ],
            'booking_system': [
                'booking', 'tatkal', 'confirm', 'waitlist', 'availability', 'seat', 
                'berth', 'quota', 'reservation', 'pnr', 'ticket', 'cancellation'
            ],
            'server_connectivity': [
                'server', 'connectivity', 'network', 'internet', 'offline', 'timeout', 
                'connection', '502', '503', 'down', 'maintenance', 'unable to connect'
            ],
            'app_features': [
                'feature', 'option', 'function', 'setting', 'notification', 'update', 
                'version', 'download', 'install', 'permission', 'cache'
            ]
        }
        
        # Railway service keywords
        self.railway_patterns = {
            'train_operations': [
                'train', 'delay', 'late', 'cancelled', 'platform', 'schedule', 'timing', 
                'punctual', 'departure', 'arrival', 'running status', 'route', 'halt'
            ],
            'cleanliness_hygiene': [
                'clean', 'dirty', 'hygiene', 'toilet', 'washroom', 'garbage', 'maintenance', 
                'dust', 'smell', 'sanitation', 'sweep', 'wash', 'bathroom', 'filthy'
            ],
            'food_catering': [
                'food', 'meal', 'catering', 'pantry', 'water', 'quality', 'fresh', 
                'taste', 'breakfast', 'lunch', 'dinner', 'snacks', 'tea', 'coffee'
            ],
            'staff_service': [
                'staff', 'conductor', 'tc', 'tte', 'behaviour', 'rude', 'helpful', 
                'service', 'attendant', 'porter', 'coolie', 'railway police', 'rpf'
            ],
            'comfort_facilities': [
                'seat', 'ac', 'fan', 'bedding', 'comfort', 'crowded', 'space', 'coach', 
                'window', 'door', 'light', 'charging', 'luggage', 'sleeper', 'general'
            ],
            'safety_security': [
                'safety', 'security', 'theft', 'police', 'emergency', 'accident', 
                'secure', 'chain', 'lock', 'robbery', 'harassment', 'women safety'
            ],
            'station_facilities': [
                'station', 'platform', 'waiting', 'announcement', 'facility', 'parking', 
                'escalator', 'lift', 'bridge', 'enquiry', 'cloak room', 'retiring room'
            ]
        }
        
        # Create comprehensive stopwords list
        self.stop_words = set(stopwords.words('english'))
        # Add generic Hindi words
        self.stop_words.update([
            'hai', 'nahi', 'ka', 'ki', 'ke', 'se', 'ho', 'ye', 'aur', 'ko', 'hi', 
            'nhi', 'bhi', 'thi', 'tha', 'hain', 'kya', 'kar', 'me', 'ne', 'to', 
            'aap', 'app', 'irctc', 'railway', 'indian', 'rail'
        ])
        # Add generic sentiment words
        self.stop_words.update([
            'good', 'bad', 'nice', 'poor', 'worst', 'best', 'excellent', 'terrible',
            'horrible', 'awesome', 'great', 'pathetic', 'useless', 'useful'
        ])
    
    def classify_review(self, text: str) -> Dict[str, any]:
        """Classify review into app or railway category with confidence scores"""
        
        text_lower = text.lower()
        
        # Score for app issues
        app_scores = {}
        for category, keywords in self.app_patterns.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                app_scores[category] = score
        
        # Score for railway issues  
        railway_scores = {}
        for category, keywords in self.railway_patterns.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                railway_scores[category] = score
        
        total_app = sum(app_scores.values())
        total_railway = sum(railway_scores.values())
        total = total_app + total_railway
        
        if total == 0:
            return {
                'department': 'unclear',
                'confidence': 0,
                'app_score': 0,
                'railway_score': 0,
                'top_app_category': None,
                'top_railway_category': None
            }
        
        app_confidence = total_app / total
        railway_confidence = total_railway / total
        
        # Determine primary department
        if app_confidence > 0.65:
            department = 'app'
            confidence = app_confidence
        elif railway_confidence > 0.65:
            department = 'railway'
            confidence = railway_confidence
        else:
            department = 'mixed'
            confidence = max(app_confidence, railway_confidence)
        
        # Get top categories
        top_app = max(app_scores.items(), key=lambda x: x[1])[0] if app_scores else None
        top_railway = max(railway_scores.items(), key=lambda x: x[1])[0] if railway_scores else None
        
        return {
            'department': department,
            'confidence': confidence,
            'app_score': app_confidence,
            'railway_score': railway_confidence,
            'top_app_category': top_app,
            'top_railway_category': top_railway,
            'app_categories': app_scores,
            'railway_categories': railway_scores
        }
    
    def extract_relevant_topics(self, texts: List[str], department: str, n_topics: int = 15) -> Dict:
        """Extract relevant topics for a specific department"""
        
        # Select appropriate keywords for vectorizer
        if department == 'app':
            domain_words = [word for category in self.app_patterns.values() for word in category]
        elif department == 'railway':
            domain_words = [word for category in self.railway_patterns.values() for word in category]
        else:
            domain_words = []
        
        # Create custom vocabulary focusing on domain-specific terms
        vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=3,
            max_df=0.7,
            ngram_range=(1, 3),
            stop_words=list(self.stop_words),
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic words
        )
        
        # Create document-term matrix
        try:
            doc_term_matrix = vectorizer.fit_transform(texts)
        except ValueError as e:
            logger.error(f"Vectorization failed: {e}")
            return {'topics': [], 'error': str(e)}
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=30,
            learning_method='batch',
            random_state=42,
            n_jobs=-1
        )
        
        doc_topic_matrix = lda.fit_transform(doc_term_matrix)
        
        # Extract topics with filtering
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            # Get top words for this topic
            top_indices = topic.argsort()[-30:][::-1]
            top_words = []
            
            for idx in top_indices:
                word = feature_names[idx]
                score = topic[idx]
                
                # Filter out generic words and single characters
                if (len(word) > 2 and 
                    word not in self.stop_words and
                    not word.isdigit()):
                    top_words.append((word, score))
                
                if len(top_words) >= 10:
                    break
            
            if len(top_words) >= 5:  # Only keep topics with enough relevant words
                # Calculate topic relevance score
                relevance = sum(score for _, score in top_words[:5])
                
                topics.append({
                    'topic_id': topic_idx,
                    'department': department,
                    'words': top_words,
                    'relevance_score': relevance,
                    'doc_count': np.sum(doc_topic_matrix[:, topic_idx] > 0.1)
                })
        
        # Sort by relevance
        topics = sorted(topics, key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            'topics': topics[:10],  # Return top 10 most relevant topics
            'vectorizer': vectorizer,
            'lda_model': lda,
            'doc_topic_matrix': doc_topic_matrix
        }
    
    def categorize_topic(self, topic_words: List[Tuple[str, float]], department: str) -> str:
        """Categorize a topic based on its words"""
        
        words_str = ' '.join([w[0] for w in topic_words[:5]])
        
        if department == 'app':
            for category, keywords in self.app_patterns.items():
                if any(kw in words_str for kw in keywords[:5]):
                    return category
            return 'app_other'
        
        elif department == 'railway':
            for category, keywords in self.railway_patterns.items():
                if any(kw in words_str for kw in keywords[:5]):
                    return category
            return 'railway_other'
        
        return 'uncategorized'

def analyze_reviews_with_segregation(db_path: str = 'data/reviews.db', sample_size: int = 10000):
    """Main function to analyze reviews with department segregation"""
    
    analyzer = ImprovedTopicAnalyzer()
    
    # Load reviews
    logger.info("Loading reviews from database...")
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        r.id,
        r.content,
        r.rating,
        p.normalized_text
    FROM reviews r
    LEFT JOIN processed_reviews p ON r.id = p.review_id
    WHERE r.content IS NOT NULL
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(sample_size,))
    logger.info(f"Loaded {len(df)} reviews")
    
    # Classify reviews
    logger.info("Classifying reviews by department...")
    classifications = []
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            logger.info(f"Processed {idx}/{len(df)} reviews")
        
        text = row['normalized_text'] if row['normalized_text'] else row['content']
        classification = analyzer.classify_review(text)
        classifications.append(classification)
    
    # Add classification results to dataframe
    df['department'] = [c['department'] for c in classifications]
    df['confidence'] = [c['confidence'] for c in classifications]
    df['app_score'] = [c['app_score'] for c in classifications]
    df['railway_score'] = [c['railway_score'] for c in classifications]
    df['top_app_category'] = [c['top_app_category'] for c in classifications]
    df['top_railway_category'] = [c['top_railway_category'] for c in classifications]
    
    # Department statistics
    dept_stats = df['department'].value_counts()
    logger.info("\nDepartment Distribution:")
    for dept, count in dept_stats.items():
        percentage = (count / len(df)) * 100
        logger.info(f"  {dept}: {count} ({percentage:.1f}%)")
    
    # Category statistics for app issues
    app_categories = df[df['department'] == 'app']['top_app_category'].value_counts()
    logger.info("\nTop App Issue Categories:")
    for cat, count in app_categories.head(10).items():
        if cat:
            logger.info(f"  {cat}: {count}")
    
    # Category statistics for railway issues
    railway_categories = df[df['department'] == 'railway']['top_railway_category'].value_counts()
    logger.info("\nTop Railway Issue Categories:")
    for cat, count in railway_categories.head(10).items():
        if cat:
            logger.info(f"  {cat}: {count}")
    
    # Extract topics for each department
    results = {
        'total_reviews': len(df),
        'department_stats': dept_stats.to_dict(),
        'app_categories': app_categories.to_dict(),
        'railway_categories': railway_categories.to_dict(),
        'topics': {'app': [], 'railway': [], 'mixed': []}
    }
    
    # Process app reviews
    app_reviews = df[df['department'] == 'app']
    if len(app_reviews) > 50:
        logger.info(f"\nExtracting topics from {len(app_reviews)} app reviews...")
        app_texts = app_reviews['content'].tolist()
        app_topics = analyzer.extract_relevant_topics(app_texts, 'app', n_topics=15)
        
        # Categorize each topic
        for topic in app_topics['topics']:
            topic['category'] = analyzer.categorize_topic(topic['words'], 'app')
        
        results['topics']['app'] = app_topics['topics']
    
    # Process railway reviews
    railway_reviews = df[df['department'] == 'railway']
    if len(railway_reviews) > 50:
        logger.info(f"\nExtracting topics from {len(railway_reviews)} railway reviews...")
        railway_texts = railway_reviews['content'].tolist()
        railway_topics = analyzer.extract_relevant_topics(railway_texts, 'railway', n_topics=15)
        
        # Categorize each topic
        for topic in railway_topics['topics']:
            topic['category'] = analyzer.categorize_topic(topic['words'], 'railway')
        
        results['topics']['railway'] = railway_topics['topics']
    
    # Save classification to database
    logger.info("\nSaving classifications to database...")
    cursor = conn.cursor()
    
    # Create table for department classifications
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS review_classifications (
        review_id INTEGER PRIMARY KEY,
        department TEXT,
        confidence REAL,
        app_score REAL,
        railway_score REAL,
        top_app_category TEXT,
        top_railway_category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (review_id) REFERENCES reviews (id)
    )
    """)
    
    # Insert classifications
    for _, row in df.iterrows():
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
    
    # Save results to pickle
    output_path = Path('data/models/improved_topics.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\nResults saved to {output_path}")
    
    return results

def print_topic_summary(results: Dict):
    """Print a formatted summary of topics"""
    
    print("\n" + "="*80)
    print("IMPROVED TOPIC ANALYSIS - DEPARTMENT SEGREGATION")
    print("="*80)
    
    print(f"\nTotal Reviews Analyzed: {results['total_reviews']}")
    print(f"Department Distribution: {results['department_stats']}")
    
    print("\n" + "-"*40)
    print("APP-SPECIFIC TOPICS:")
    print("-"*40)
    
    for topic in results['topics'].get('app', [])[:5]:
        words = ', '.join([w[0] for w in topic['words'][:8]])
        print(f"\nTopic {topic['topic_id']} ({topic['category']}):")
        print(f"  Keywords: {words}")
        print(f"  Documents: {topic['doc_count']}")
        print(f"  Relevance: {topic['relevance_score']:.2f}")
    
    print("\n" + "-"*40)
    print("RAILWAY SERVICE TOPICS:")
    print("-"*40)
    
    for topic in results['topics'].get('railway', [])[:5]:
        words = ', '.join([w[0] for w in topic['words'][:8]])
        print(f"\nTopic {topic['topic_id']} ({topic['category']}):")
        print(f"  Keywords: {words}")
        print(f"  Documents: {topic['doc_count']}")
        print(f"  Relevance: {topic['relevance_score']:.2f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Run analysis
    results = analyze_reviews_with_segregation(sample_size=5000)
    
    # Print summary
    print_topic_summary(results)