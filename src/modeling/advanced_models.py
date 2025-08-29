#!/usr/bin/env python3
"""
Advanced Topic Modeling with BERTopic and Comparison Framework
Implements multiple approaches from the PDF:
1. LDA (baseline)
2. Word2Vec embeddings + clustering
3. BERT embeddings + HDBSCAN (BERTopic)
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import pickle
from datetime import datetime

# NLP and ML libraries
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import nltk
from gensim.models import Word2Vec

# Advanced models
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import hdbscan

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedTopicModeler:
    """Advanced topic modeling with multiple approaches"""
    
    def __init__(self, db_path: str = 'data/reviews.db'):
        """Initialize with database connection"""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.models_dir = Path('data/models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.results = {}
        
    def load_processed_reviews(self, limit: int = None) -> pd.DataFrame:
        """Load processed reviews from database"""
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
        
        if limit:
            query += f" LIMIT {limit}"
            
        df = pd.read_sql_query(query, self.conn)
        logger.info(f"Loaded {len(df)} processed reviews")
        return df
    
    def approach1_lda(self, texts: List[str], n_topics: int = 20) -> Dict:
        """
        Approach 1: Latent Dirichlet Allocation (LDA)
        As implemented in the PDF - baseline approach
        """
        logger.info("="*60)
        logger.info("APPROACH 1: LDA Topic Modeling")
        logger.info("="*60)
        
        # Vectorization
        vectorizer = CountVectorizer(
            max_features=800,
            min_df=4,
            ngram_range=(1, 1),
            stop_words='english'
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        logger.info(f"Document-term matrix shape: {doc_term_matrix.shape}")
        logger.info(f"Sparsity: {(doc_term_matrix.nnz / (doc_term_matrix.shape[0] * doc_term_matrix.shape[1]) * 100):.2f}%")
        
        # LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=10,
            learning_offset=20,
            random_state=42,
            n_jobs=-1
        )
        
        lda.fit(doc_term_matrix)
        
        # Calculate perplexity
        perplexity = lda.perplexity(doc_term_matrix)
        logger.info(f"Perplexity: {perplexity:.2f}")
        
        # Get topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic[top_indices].tolist()
            })
        
        # Store results
        self.models['lda'] = {
            'model': lda,
            'vectorizer': vectorizer
        }
        
        results = {
            'method': 'LDA',
            'n_topics': n_topics,
            'perplexity': perplexity,
            'topics': topics,
            'doc_topic_matrix': lda.transform(doc_term_matrix)
        }
        
        self.results['lda'] = results
        logger.info(f"LDA completed with {n_topics} topics")
        
        return results
    
    def approach2_word2vec(self, texts: List[str], n_clusters: int = 20) -> Dict:
        """
        Approach 2: Word2Vec Embeddings with Clustering
        Custom embeddings trained on the dataset
        """
        logger.info("="*60)
        logger.info("APPROACH 2: Word2Vec Embeddings + Clustering")
        logger.info("="*60)
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec
        w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            seed=42
        )
        
        logger.info(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
        
        # Get document embeddings (average of word embeddings)
        doc_embeddings = []
        for tokens in tokenized_texts:
            word_vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
            if word_vecs:
                doc_vec = np.mean(word_vecs, axis=0)
            else:
                doc_vec = np.zeros(100)
            doc_embeddings.append(doc_vec)
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(doc_embeddings)
        
        # Calculate metrics
        silhouette = silhouette_score(doc_embeddings, clusters)
        davies_bouldin = davies_bouldin_score(doc_embeddings, clusters)
        
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f}")
        
        # Extract topics from clusters
        topics = []
        for cluster_id in range(n_clusters):
            cluster_docs = [texts[i] for i, c in enumerate(clusters) if c == cluster_id]
            if cluster_docs:
                # Get most frequent words in cluster
                cluster_text = ' '.join(cluster_docs)
                words = cluster_text.split()
                word_freq = pd.Series(words).value_counts().head(10)
                topics.append({
                    'topic_id': cluster_id,
                    'words': word_freq.index.tolist(),
                    'weights': (word_freq.values / word_freq.values.sum()).tolist(),
                    'size': len(cluster_docs)
                })
        
        # Store results
        self.models['word2vec'] = {
            'model': w2v_model,
            'kmeans': kmeans
        }
        
        results = {
            'method': 'Word2Vec + KMeans',
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'topics': topics,
            'clusters': clusters,
            'embeddings': doc_embeddings
        }
        
        self.results['word2vec'] = results
        logger.info(f"Word2Vec completed with {n_clusters} clusters")
        
        return results
    
    def approach3_bertopic(self, texts: List[str], min_topic_size: int = 10) -> Dict:
        """
        Approach 3: BERT Embeddings with BERTopic (HDBSCAN + UMAP)
        State-of-the-art approach as described in the PDF
        """
        logger.info("="*60)
        logger.info("APPROACH 3: BERTopic (BERT + HDBSCAN + UMAP)")
        logger.info("="*60)
        
        # Use multilingual BERT for Indian languages
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Configure UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN for clustering
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=5,
            metric='euclidean',
            prediction_data=True
        )
        
        # Create BERTopic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            nr_topics='auto',
            verbose=True
        )
        
        # Fit the model
        topics, probabilities = topic_model.fit_transform(texts)
        
        # Get topic information
        topic_info = topic_model.get_topic_info()
        n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
        
        logger.info(f"Found {n_topics} topics (excluding outliers)")
        
        # Calculate metrics
        embeddings = embedding_model.encode(texts)
        
        # Filter out outliers for metrics
        valid_indices = [i for i, t in enumerate(topics) if t != -1]
        if len(valid_indices) > 1:
            valid_embeddings = embeddings[valid_indices]
            valid_topics = [topics[i] for i in valid_indices]
            silhouette = silhouette_score(valid_embeddings, valid_topics)
        else:
            silhouette = -1
            
        logger.info(f"Silhouette Score: {silhouette:.4f}")
        
        # Extract topics
        topics_list = []
        for topic_id in range(-1, n_topics):
            if topic_id == -1 and not topic_model.get_topic(topic_id):
                continue
            topic_words = topic_model.get_topic(topic_id)
            if topic_words:
                topics_list.append({
                    'topic_id': topic_id,
                    'words': [word for word, _ in topic_words[:10]],
                    'weights': [weight for _, weight in topic_words[:10]],
                    'size': len([t for t in topics if t == topic_id])
                })
        
        # Store results
        self.models['bertopic'] = {
            'model': topic_model,
            'embedding_model': embedding_model
        }
        
        results = {
            'method': 'BERTopic',
            'n_topics': n_topics,
            'silhouette_score': silhouette,
            'topics': topics_list,
            'topic_assignments': topics,
            'probabilities': probabilities,
            'embeddings': embeddings,
            'topic_info': topic_info
        }
        
        self.results['bertopic'] = results
        logger.info(f"BERTopic completed with {n_topics} topics")
        
        return results
    
    def compare_approaches(self) -> pd.DataFrame:
        """Compare all approaches with metrics"""
        comparison = []
        
        for method_name, result in self.results.items():
            metrics = {
                'Method': result['method'],
                'Number of Topics': result.get('n_topics', result.get('n_clusters', 'N/A')),
                'Silhouette Score': result.get('silhouette_score', 'N/A'),
                'Perplexity': result.get('perplexity', 'N/A'),
                'Davies-Bouldin Index': result.get('davies_bouldin_index', 'N/A')
            }
            comparison.append(metrics)
        
        df_comparison = pd.DataFrame(comparison)
        return df_comparison
    
    def save_models(self):
        """Save all trained models"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for method_name, model_data in self.models.items():
            model_path = self.models_dir / f'{method_name}_model_{timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Saved {method_name} model to {model_path}")
    
    def visualize_topics(self, method: str = 'bertopic'):
        """Create visualizations for topic models"""
        if method not in self.results:
            logger.error(f"Method {method} not found in results")
            return
        
        result = self.results[method]
        
        # Create word clouds for top topics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, topic in enumerate(result['topics'][:6]):
            if i >= 6:
                break
            
            # Create word cloud from topic words
            word_freq = dict(zip(topic['words'], topic['weights']))
            wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(word_freq)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f"Topic {topic['topic_id']}")
            axes[i].axis('off')
        
        plt.suptitle(f'Top Topics - {method.upper()}')
        plt.tight_layout()
        plt.savefig(f'data/models/topics_wordcloud_{method}.png')
        plt.show()
        
        logger.info(f"Saved topic visualizations for {method}")


def main():
    """Main function to run all topic modeling approaches"""
    logger.info("="*60)
    logger.info("ADVANCED TOPIC MODELING COMPARISON")
    logger.info("="*60)
    
    # Initialize modeler
    modeler = AdvancedTopicModeler()
    
    # Load processed reviews
    df = modeler.load_processed_reviews(limit=None)  # Use all reviews
    
    if len(df) == 0:
        logger.error("No processed reviews found. Run preprocessing first.")
        return
    
    texts = df['processed'].dropna().tolist()
    logger.info(f"Using {len(texts)} processed reviews for modeling")
    
    # Run all approaches
    logger.info("\nRunning topic modeling approaches...")
    
    # Approach 1: LDA
    lda_results = modeler.approach1_lda(texts, n_topics=20)
    
    # Approach 2: Word2Vec
    w2v_results = modeler.approach2_word2vec(texts, n_clusters=20)
    
    # Approach 3: BERTopic
    bertopic_results = modeler.approach3_bertopic(texts, min_topic_size=10)
    
    # Compare approaches
    comparison_df = modeler.compare_approaches()
    
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("="*60)
    print(comparison_df.to_string())
    
    # Save comparison to CSV
    comparison_df.to_csv('data/models/model_comparison.csv', index=False)
    logger.info("\nSaved comparison to data/models/model_comparison.csv")
    
    # Generate visualizations
    for method in ['lda', 'word2vec', 'bertopic']:
        modeler.visualize_topics(method)
    
    # Save models
    modeler.save_models()
    
    # Display top topics from BERTopic (best performing)
    logger.info("\n" + "="*60)
    logger.info("TOP TOPICS FROM BERTOPIC (BEST MODEL)")
    logger.info("="*60)
    
    for topic in bertopic_results['topics'][:10]:
        logger.info(f"\nTopic {topic['topic_id']} (Size: {topic['size']})")
        logger.info(f"Words: {', '.join(topic['words'][:5])}")
    
    return modeler


if __name__ == "__main__":
    modeler = main()