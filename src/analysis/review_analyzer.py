#!/usr/bin/env python3
"""
Review Analysis Module with Topic Modeling and Sentiment Analysis
For IRCTC Review Analysis System
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import pickle
import json
from datetime import datetime
from tqdm import tqdm

# ML/NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Advanced sentiment analysis will be limited.")

# Topic modeling
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logging.warning("BERTopic not available. Using LDA for topic modeling.")

# Word cloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logging.warning("WordCloud not available. Skipping word cloud generation.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReviewAnalyzer:
    """Comprehensive review analysis with topic modeling and sentiment analysis"""
    
    def __init__(self, db_path: str = 'data/reviews.db'):
        """Initialize analyzer with database connection"""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        
        # Models
        self.topic_model = None
        self.sentiment_model = None
        self.embedding_model = None
        
        # Results cache
        self.topics = None
        self.sentiments = None
        
    def load_processed_reviews(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load preprocessed reviews from database"""
        query = """
            SELECT 
                pr.review_id,
                pr.normalized_text,
                pr.cleaned_text,
                pr.language,
                r.rating,
                r.date_posted
            FROM processed_reviews pr
            JOIN reviews r ON pr.review_id = r.id
            WHERE pr.normalized_text IS NOT NULL 
            AND pr.normalized_text != ''
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded {len(df)} processed reviews")
        
        return df
    
    def perform_topic_modeling(
        self, 
        texts: List[str],
        method: str = 'bertopic',
        n_topics: Optional[int] = None,
        min_topic_size: int = 10
    ) -> Dict:
        """Perform topic modeling on texts"""
        
        logger.info(f"Starting topic modeling with method: {method}")
        
        if method == 'bertopic' and BERTOPIC_AVAILABLE:
            return self._bertopic_modeling(texts, n_topics, min_topic_size)
        else:
            return self._lda_modeling(texts, n_topics or 20)
    
    def _bertopic_modeling(
        self, 
        texts: List[str],
        n_topics: Optional[int],
        min_topic_size: int
    ) -> Dict:
        """BERTopic modeling for advanced topic extraction"""
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        if not self.embedding_model:
            # Use multilingual model for Hindi/English mix
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Configure UMAP for dimension reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            prediction_data=True
        )
        
        # Create BERTopic model
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=self.embedding_model,
            nr_topics=n_topics,
            top_n_words=10,
            verbose=True
        )
        
        # Fit model
        logger.info("Fitting BERTopic model...")
        topics, probs = self.topic_model.fit_transform(texts, embeddings)
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        # Get representative docs for each topic
        topic_docs = {}
        for topic_id in topic_info['Topic'].unique():
            if topic_id != -1:  # Skip outlier topic
                docs = self.topic_model.get_representative_docs(topic_id)
                topic_docs[topic_id] = docs[:3] if docs else []
        
        results = {
            'topics': topics,
            'probabilities': probs,
            'topic_info': topic_info,
            'topic_docs': topic_docs,
            'embeddings': embeddings,
            'num_topics': len(set(topics)) - 1  # Exclude outlier topic -1
        }
        
        logger.info(f"Found {results['num_topics']} topics")
        
        return results
    
    def _lda_modeling(self, texts: List[str], n_topics: int) -> Dict:
        """Fallback LDA topic modeling"""
        
        logger.info("Using LDA for topic modeling...")
        
        # Create document-term matrix
        vectorizer = CountVectorizer(
            max_features=1000,
            max_df=0.8,
            min_df=5,
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            random_state=42,
            verbose=1
        )
        
        lda_output = lda.fit_transform(doc_term_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'id': topic_idx,
                'words': top_words,
                'weights': topic[top_words_idx].tolist()
            })
        
        # Assign topics to documents
        doc_topics = lda_output.argmax(axis=1)
        
        results = {
            'topics': doc_topics.tolist(),
            'probabilities': lda_output,
            'topic_words': topics,
            'num_topics': n_topics
        }
        
        return results
    
    def perform_sentiment_analysis(
        self, 
        texts: List[str],
        method: str = 'transformer'
    ) -> Dict:
        """Perform sentiment analysis on texts"""
        
        logger.info(f"Starting sentiment analysis with method: {method}")
        
        if method == 'transformer' and TRANSFORMERS_AVAILABLE:
            return self._transformer_sentiment(texts)
        else:
            return self._rule_based_sentiment(texts)
    
    def _transformer_sentiment(self, texts: List[str]) -> Dict:
        """Transformer-based sentiment analysis"""
        
        logger.info("Loading sentiment analysis model...")
        
        # Use multilingual sentiment model
        if not self.sentiment_model:
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1  # Use CPU
            )
        
        sentiments = []
        scores = []
        
        # Process in batches
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i+batch_size]
            
            try:
                results = self.sentiment_model(batch, truncation=True, max_length=512)
                
                for result in results:
                    # Convert 5-star rating to sentiment
                    label = result['label']
                    score = result['score']
                    
                    if '1' in label or '2' in label:
                        sentiment = 'negative'
                        sentiment_score = -score
                    elif '3' in label:
                        sentiment = 'neutral'
                        sentiment_score = 0
                    else:
                        sentiment = 'positive'
                        sentiment_score = score
                    
                    sentiments.append(sentiment)
                    scores.append(sentiment_score)
                    
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                # Fallback to neutral
                for _ in batch:
                    sentiments.append('neutral')
                    scores.append(0)
        
        return {
            'sentiments': sentiments,
            'scores': scores,
            'distribution': pd.Series(sentiments).value_counts().to_dict()
        }
    
    def _rule_based_sentiment(self, texts: List[str]) -> Dict:
        """Simple rule-based sentiment analysis"""
        
        positive_words = {
            'good', 'great', 'excellent', 'best', 'awesome', 
            'nice', 'perfect', 'love', 'wonderful', 'fantastic'
        }
        
        negative_words = {
            'bad', 'worst', 'terrible', 'awful', 'horrible',
            'pathetic', 'useless', 'waste', 'poor', 'disgusting'
        }
        
        sentiments = []
        scores = []
        
        for text in texts:
            text_lower = text.lower()
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if neg_count > pos_count:
                sentiments.append('negative')
                scores.append(-neg_count / max(len(text.split()), 1))
            elif pos_count > neg_count:
                sentiments.append('positive')
                scores.append(pos_count / max(len(text.split()), 1))
            else:
                sentiments.append('neutral')
                scores.append(0)
        
        return {
            'sentiments': sentiments,
            'scores': scores,
            'distribution': pd.Series(sentiments).value_counts().to_dict()
        }
    
    def analyze_reviews(
        self,
        limit: Optional[int] = None,
        topic_method: str = 'lda',
        sentiment_method: str = 'rule'
    ) -> Dict:
        """Complete analysis pipeline"""
        
        # Load data
        df = self.load_processed_reviews(limit)
        
        if df.empty:
            logger.error("No processed reviews found")
            return {}
        
        texts = df['normalized_text'].tolist()
        
        # Topic modeling
        topic_results = self.perform_topic_modeling(
            texts,
            method=topic_method,
            n_topics=20
        )
        
        # Sentiment analysis
        sentiment_results = self.perform_sentiment_analysis(
            texts,
            method=sentiment_method
        )
        
        # Combine results
        df['topic'] = topic_results['topics']
        df['sentiment'] = sentiment_results['sentiments']
        df['sentiment_score'] = sentiment_results['scores']
        
        # Analysis by topic
        topic_sentiment = df.groupby('topic')['sentiment'].value_counts().unstack(fill_value=0)
        topic_avg_rating = df.groupby('topic')['rating'].mean()
        
        # Save results
        self.save_analysis_results(df, topic_results, sentiment_results)
        
        return {
            'dataframe': df,
            'topic_results': topic_results,
            'sentiment_results': sentiment_results,
            'topic_sentiment': topic_sentiment,
            'topic_avg_rating': topic_avg_rating,
            'summary': self.generate_summary(df, topic_results, sentiment_results)
        }
    
    def save_analysis_results(self, df: pd.DataFrame, topics: Dict, sentiments: Dict):
        """Save analysis results to database"""
        
        # Create analysis results table
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                topic_id INTEGER,
                sentiment TEXT,
                sentiment_score REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (review_id) REFERENCES reviews (id)
            )
        """)
        
        # Insert results
        for idx, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO analysis_results (
                    review_id, topic_id, sentiment, sentiment_score
                ) VALUES (?, ?, ?, ?)
            """, (
                row['review_id'],
                row['topic'],
                row['sentiment'],
                row['sentiment_score']
            ))
        
        self.conn.commit()
        logger.info(f"Saved {len(df)} analysis results to database")
    
    def generate_summary(self, df: pd.DataFrame, topics: Dict, sentiments: Dict) -> Dict:
        """Generate analysis summary"""
        
        summary = {
            'total_reviews': len(df),
            'num_topics': topics.get('num_topics', 0),
            'sentiment_distribution': sentiments['distribution'],
            'avg_sentiment_score': np.mean(sentiments['scores']),
            'avg_rating': df['rating'].mean(),
            'most_negative_topics': [],
            'most_positive_topics': []
        }
        
        # Find most positive/negative topics
        topic_sentiments = df.groupby('topic')['sentiment_score'].mean().sort_values()
        
        if len(topic_sentiments) > 0:
            summary['most_negative_topics'] = topic_sentiments.head(3).index.tolist()
            summary['most_positive_topics'] = topic_sentiments.tail(3).index.tolist()
        
        return summary
    
    def visualize_results(self, results: Dict):
        """Create visualizations of analysis results"""
        
        if not results:
            logger.warning("No results to visualize")
            return
        
        df = results['dataframe']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values)
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].set_xlabel('Sentiment')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Topics distribution
        topic_counts = df['topic'].value_counts().head(10)
        axes[0, 1].barh(topic_counts.index.astype(str), topic_counts.values)
        axes[0, 1].set_title('Top 10 Topics')
        axes[0, 1].set_xlabel('Count')
        axes[0, 1].set_ylabel('Topic ID')
        
        # 3. Rating vs Sentiment
        sentiment_rating = df.groupby('sentiment')['rating'].mean()
        axes[1, 0].bar(sentiment_rating.index, sentiment_rating.values)
        axes[1, 0].set_title('Average Rating by Sentiment')
        axes[1, 0].set_xlabel('Sentiment')
        axes[1, 0].set_ylabel('Average Rating')
        
        # 4. Sentiment over time (if date available)
        if 'date_posted' in df.columns:
            df['date'] = pd.to_datetime(df['date_posted'])
            daily_sentiment = df.groupby([df['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
            daily_sentiment.plot(ax=axes[1, 1])
            axes[1, 1].set_title('Sentiment Trend Over Time')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend(title='Sentiment')
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('data/analysis_results.png')
        plt.savefig(output_path)
        logger.info(f"Saved visualization to {output_path}")
        
        plt.show()


def main():
    """Main function to run analysis"""
    logger.info("Starting review analysis...")
    
    analyzer = ReviewAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_reviews(
        limit=1000,  # Analyze 1000 reviews for testing
        topic_method='lda',  # Use LDA for now (faster)
        sentiment_method='rule'  # Use rule-based for now (no model download)
    )
    
    if results:
        summary = results['summary']
        
        print("\n📊 Analysis Summary")
        print("=" * 50)
        print(f"Total Reviews Analyzed: {summary['total_reviews']}")
        print(f"Number of Topics Found: {summary['num_topics']}")
        print(f"Average Rating: {summary['avg_rating']:.2f}")
        print(f"Average Sentiment Score: {summary['avg_sentiment_score']:.3f}")
        print(f"\nSentiment Distribution:")
        for sentiment, count in summary['sentiment_distribution'].items():
            print(f"  {sentiment}: {count}")
        
        print(f"\nMost Negative Topics: {summary['most_negative_topics']}")
        print(f"Most Positive Topics: {summary['most_positive_topics']}")
        
        # Generate visualizations
        analyzer.visualize_results(results)
        
        # Show sample reviews by topic
        df = results['dataframe']
        print("\n📝 Sample Reviews by Topic")
        print("=" * 50)
        
        for topic in df['topic'].value_counts().head(3).index:
            print(f"\n Topic {topic}:")
            samples = df[df['topic'] == topic].head(2)
            for _, row in samples.iterrows():
                print(f"  [{row['sentiment']}] {row['cleaned_text'][:100]}...")
    
    return results


if __name__ == "__main__":
    analysis_results = main()