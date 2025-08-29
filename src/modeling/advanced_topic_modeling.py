#!/usr/bin/env python3
"""
Advanced Topic Modeling with App vs Railway Service Segregation
Uses BERT embeddings and improved topic extraction
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import pickle
from datetime import datetime

# NLP and ML libraries
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from umap import UMAP
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTopicAnalyzer:
    """Advanced topic modeling with department segregation"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the analyzer with BERT model"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load sentence transformer for embeddings
        self.sentence_model = SentenceTransformer(model_name, device=self.device)
        
        # Define app-related keywords
        self.app_keywords = {
            'ui': ['button', 'screen', 'display', 'interface', 'ui', 'ux', 'design', 'layout', 'menu', 'page'],
            'technical': ['crash', 'hang', 'freeze', 'bug', 'error', 'glitch', 'lag', 'slow', 'loading', 'response'],
            'login': ['login', 'password', 'otp', 'authentication', 'signin', 'signup', 'register', 'account', 'profile'],
            'payment': ['payment', 'transaction', 'refund', 'debit', 'credit', 'upi', 'wallet', 'netbanking', 'card'],
            'booking_system': ['booking', 'tatkal', 'confirm', 'waitlist', 'availability', 'seat', 'berth', 'quota'],
            'server': ['server', 'connectivity', 'network', 'internet', 'offline', 'timeout', 'connection'],
            'features': ['feature', 'option', 'function', 'setting', 'notification', 'update', 'version'],
        }
        
        # Define railway service keywords
        self.railway_keywords = {
            'train_service': ['train', 'delay', 'late', 'cancelled', 'platform', 'schedule', 'timing', 'punctual'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'toilet', 'washroom', 'garbage', 'maintenance', 'dust'],
            'food': ['food', 'meal', 'catering', 'pantry', 'water', 'quality', 'fresh', 'taste'],
            'staff': ['staff', 'conductor', 'tc', 'tte', 'behaviour', 'rude', 'helpful', 'service'],
            'comfort': ['seat', 'ac', 'fan', 'bedding', 'comfort', 'crowded', 'space', 'coach'],
            'safety': ['safety', 'security', 'theft', 'police', 'emergency', 'accident', 'secure'],
            'station': ['station', 'platform', 'waiting', 'announcement', 'facility', 'parking'],
        }
        
        # Combined stopwords (English + Hindi common words)
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['app', 'irctc', 'railway', 'train', 'hai', 'nahi', 'ka', 'ki', 'ke', 
                               'se', 'ho', 'ye', 'aur', 'ko', 'hi', 'nhi', 'bhi', 'thi', 'tha',
                               'good', 'bad', 'nice', 'poor', 'worst', 'best', 'excellent'])
    
    def classify_review_department(self, text: str) -> Dict[str, float]:
        """Classify if review is about app or railway service"""
        text_lower = text.lower()
        
        # Count keyword matches
        app_score = 0
        railway_score = 0
        
        # Check app keywords
        for category, keywords in self.app_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    app_score += 1
        
        # Check railway keywords
        for category, keywords in self.railway_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    railway_score += 1
        
        # Normalize scores
        total = app_score + railway_score
        if total == 0:
            return {'app': 0.5, 'railway': 0.5, 'category': 'mixed'}
        
        app_prob = app_score / total
        railway_prob = railway_score / total
        
        # Determine primary category
        if app_prob > 0.7:
            category = 'app'
        elif railway_prob > 0.7:
            category = 'railway'
        else:
            category = 'mixed'
        
        return {
            'app': app_prob,
            'railway': railway_prob,
            'category': category
        }
    
    def extract_specific_topics(self, texts: List[str], department: str = 'all') -> Dict:
        """Extract specific topics for a department using BERTopic"""
        
        if department == 'app':
            keywords = self.app_keywords
        elif department == 'railway':
            keywords = self.railway_keywords
        else:
            keywords = {**self.app_keywords, **self.railway_keywords}
        
        # Custom vectorizer with relevant keywords
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=list(self.stop_words),
            min_df=5,
            max_df=0.8
        )
        
        # UMAP for dimensionality reduction
        umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=10,
            verbose=True,
            calculate_probabilities=True
        )
        
        # Fit the model
        logger.info(f"Fitting BERTopic for {department} department...")
        topics, probs = topic_model.fit_transform(texts)
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        
        # Extract detailed topics
        detailed_topics = []
        for topic_id in range(len(topic_info) - 1):  # Exclude outlier topic (-1)
            if topic_id == -1:
                continue
            
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                # Filter out generic words
                filtered_words = []
                for word, score in topic_words[:20]:
                    if word not in self.stop_words and len(word) > 2:
                        filtered_words.append((word, score))
                
                if len(filtered_words) >= 3:
                    detailed_topics.append({
                        'topic_id': topic_id,
                        'words': filtered_words[:10],
                        'count': len([t for t in topics if t == topic_id]),
                        'department': department
                    })
        
        return {
            'model': topic_model,
            'topics': detailed_topics,
            'topic_assignments': topics,
            'probabilities': probs
        }
    
    def categorize_topics(self, topics: List[Dict]) -> Dict:
        """Categorize topics into specific issue types"""
        categorized = {
            'app_issues': {
                'technical': [],
                'payment': [],
                'booking': [],
                'ui_ux': [],
                'login': [],
                'server': []
            },
            'railway_issues': {
                'service_quality': [],
                'cleanliness': [],
                'food_catering': [],
                'staff_behavior': [],
                'safety_security': [],
                'timing_delays': []
            }
        }
        
        for topic in topics:
            words_str = ' '.join([w[0] for w in topic['words'][:5]])
            
            # Categorize app issues
            if topic['department'] == 'app':
                if any(kw in words_str for kw in ['crash', 'hang', 'error', 'bug', 'freeze']):
                    categorized['app_issues']['technical'].append(topic)
                elif any(kw in words_str for kw in ['payment', 'transaction', 'refund', 'money']):
                    categorized['app_issues']['payment'].append(topic)
                elif any(kw in words_str for kw in ['booking', 'tatkal', 'ticket', 'seat']):
                    categorized['app_issues']['booking'].append(topic)
                elif any(kw in words_str for kw in ['button', 'screen', 'interface', 'design']):
                    categorized['app_issues']['ui_ux'].append(topic)
                elif any(kw in words_str for kw in ['login', 'password', 'otp', 'account']):
                    categorized['app_issues']['login'].append(topic)
                elif any(kw in words_str for kw in ['server', 'connection', 'network', 'timeout']):
                    categorized['app_issues']['server'].append(topic)
            
            # Categorize railway issues
            elif topic['department'] == 'railway':
                if any(kw in words_str for kw in ['delay', 'late', 'cancel', 'time']):
                    categorized['railway_issues']['timing_delays'].append(topic)
                elif any(kw in words_str for kw in ['clean', 'dirty', 'hygiene', 'toilet']):
                    categorized['railway_issues']['cleanliness'].append(topic)
                elif any(kw in words_str for kw in ['food', 'meal', 'catering', 'water']):
                    categorized['railway_issues']['food_catering'].append(topic)
                elif any(kw in words_str for kw in ['staff', 'conductor', 'behaviour', 'rude']):
                    categorized['railway_issues']['staff_behavior'].append(topic)
                elif any(kw in words_str for kw in ['safety', 'security', 'theft', 'police']):
                    categorized['railway_issues']['safety_security'].append(topic)
                else:
                    categorized['railway_issues']['service_quality'].append(topic)
        
        return categorized

def process_reviews_with_segregation(db_path: str = 'data/reviews.db'):
    """Process reviews and segregate by department"""
    
    # Initialize analyzer
    analyzer = AdvancedTopicAnalyzer()
    
    # Load reviews from database
    conn = sqlite3.connect(db_path)
    
    # Get processed reviews
    query = """
    SELECT 
        r.id,
        r.content,
        r.rating,
        p.normalized_text
    FROM reviews r
    LEFT JOIN processed_reviews p ON r.id = p.review_id
    WHERE p.normalized_text IS NOT NULL
    LIMIT 5000
    """
    
    df = pd.read_sql_query(query, conn)
    logger.info(f"Loaded {len(df)} reviews for analysis")
    
    # Classify reviews by department
    logger.info("Classifying reviews by department...")
    classifications = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(df)} reviews")
        
        text = row['normalized_text'] or row['content']
        classification = analyzer.classify_review_department(text)
        classifications.append(classification)
    
    df['department'] = [c['category'] for c in classifications]
    df['app_score'] = [c['app'] for c in classifications]
    df['railway_score'] = [c['railway'] for c in classifications]
    
    # Split reviews by department
    app_reviews = df[df['department'] == 'app']['normalized_text'].tolist()
    railway_reviews = df[df['department'] == 'railway']['normalized_text'].tolist()
    mixed_reviews = df[df['department'] == 'mixed']['normalized_text'].tolist()
    
    logger.info(f"App reviews: {len(app_reviews)}")
    logger.info(f"Railway reviews: {len(railway_reviews)}")
    logger.info(f"Mixed reviews: {len(mixed_reviews)}")
    
    # Extract topics for each department
    all_topics = []
    
    if len(app_reviews) > 50:
        logger.info("Extracting app-specific topics...")
        app_results = analyzer.extract_specific_topics(app_reviews, 'app')
        all_topics.extend(app_results['topics'])
    
    if len(railway_reviews) > 50:
        logger.info("Extracting railway-specific topics...")
        railway_results = analyzer.extract_specific_topics(railway_reviews, 'railway')
        all_topics.extend(railway_results['topics'])
    
    # Categorize topics
    categorized_topics = analyzer.categorize_topics(all_topics)
    
    # Save results
    results = {
        'total_reviews': len(df),
        'department_distribution': df['department'].value_counts().to_dict(),
        'all_topics': all_topics,
        'categorized_topics': categorized_topics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to pickle
    output_path = Path('data/models/advanced_topics.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Results saved to {output_path}")
    
    # Update database with department classifications
    cursor = conn.cursor()
    
    # Add department column if not exists
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS review_departments (
        review_id INTEGER PRIMARY KEY,
        department TEXT,
        app_score REAL,
        railway_score REAL,
        FOREIGN KEY (review_id) REFERENCES reviews (id)
    )
    """)
    
    # Insert classifications
    for idx, row in df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO review_departments (review_id, department, app_score, railway_score)
        VALUES (?, ?, ?, ?)
        """, (row['id'], row['department'], row['app_score'], row['railway_score']))
    
    conn.commit()
    conn.close()
    
    return results

def print_analysis_summary(results: Dict):
    """Print a summary of the analysis"""
    print("\n" + "="*80)
    print("ADVANCED TOPIC ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal Reviews Analyzed: {results['total_reviews']}")
    print(f"Department Distribution:")
    for dept, count in results['department_distribution'].items():
        percentage = (count / results['total_reviews']) * 100
        print(f"  - {dept.capitalize()}: {count} ({percentage:.1f}%)")
    
    print(f"\nTotal Topics Identified: {len(results['all_topics'])}")
    
    print("\n" + "-"*40)
    print("APP-RELATED ISSUES:")
    print("-"*40)
    
    for category, topics in results['categorized_topics']['app_issues'].items():
        if topics:
            print(f"\n{category.replace('_', ' ').title()}:")
            for topic in topics[:2]:  # Show top 2 topics per category
                words = ', '.join([w[0] for w in topic['words'][:5]])
                print(f"  - Topic {topic['topic_id']}: {words} ({topic['count']} reviews)")
    
    print("\n" + "-"*40)
    print("RAILWAY SERVICE ISSUES:")
    print("-"*40)
    
    for category, topics in results['categorized_topics']['railway_issues'].items():
        if topics:
            print(f"\n{category.replace('_', ' ').title()}:")
            for topic in topics[:2]:  # Show top 2 topics per category
                words = ', '.join([w[0] for w in topic['words'][:5]])
                print(f"  - Topic {topic['topic_id']}: {words} ({topic['count']} reviews)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    
    # Run the analysis
    results = process_reviews_with_segregation()
    
    # Print summary
    print_analysis_summary(results)