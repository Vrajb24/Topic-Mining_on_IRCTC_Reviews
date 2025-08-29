#!/usr/bin/env python3
"""
Comprehensive Root Cause Analysis System for IRCTC Reviews
Implements multiple analytical approaches to identify root causes of issues
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple, Any
import pickle
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RootCauseAnalyzer:
    """Comprehensive root cause analysis for review data"""
    
    def __init__(self, db_path: str = 'data/reviews.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Root cause patterns
        self.root_cause_patterns = {
            'infrastructure': {
                'keywords': ['server', 'slow', 'timeout', 'down', 'crash', 'lag', 'hang'],
                'description': 'System infrastructure and performance issues'
            },
            'integration': {
                'keywords': ['payment gateway', 'otp', 'bank', 'third party', 'api', 'integration'],
                'description': 'Third-party service integration failures'
            },
            'data_integrity': {
                'keywords': ['wrong', 'incorrect', 'mismatch', 'error', 'invalid', 'corrupted'],
                'description': 'Data accuracy and consistency issues'
            },
            'user_experience': {
                'keywords': ['confusing', 'difficult', 'cannot find', 'unclear', 'complicated'],
                'description': 'User interface and experience problems'
            },
            'business_logic': {
                'keywords': ['not allowed', 'restricted', 'quota', 'rules', 'policy', 'limit'],
                'description': 'Business rules and policy constraints'
            },
            'authentication': {
                'keywords': ['login', 'password', 'authentication', 'session', 'expired', 'logout'],
                'description': 'Authentication and session management issues'
            },
            'capacity': {
                'keywords': ['tatkal', 'peak', 'rush', 'heavy load', 'busy', 'traffic'],
                'description': 'System capacity and scaling issues'
            }
        }
        
        # Failure chains
        self.failure_chains = {
            'booking_failure': [
                'login_attempt', 'session_creation', 'seat_selection', 
                'payment_initiation', 'payment_completion', 'ticket_generation'
            ],
            'payment_failure': [
                'payment_gateway_connection', 'bank_authentication', 
                'transaction_processing', 'confirmation_receipt'
            ],
            'auth_failure': [
                'credential_validation', 'otp_generation', 'otp_delivery', 
                'otp_validation', 'session_establishment'
            ]
        }
        
        # Severity levels
        self.severity_levels = {
            'critical': {
                'keywords': ['money lost', 'payment deducted no ticket', 'cannot book', 
                            'complete failure', 'not working at all'],
                'weight': 5
            },
            'high': {
                'keywords': ['major issue', 'frequently fails', 'always problem', 
                            'never works', 'serious problem'],
                'weight': 4
            },
            'medium': {
                'keywords': ['sometimes', 'occasionally', 'often', 'usually'],
                'weight': 3
            },
            'low': {
                'keywords': ['minor', 'small issue', 'suggestion', 'could be better'],
                'weight': 2
            }
        }
    
    def analyze_temporal_patterns(self, days: int = 30) -> Dict:
        """Analyze temporal patterns of issues"""
        
        query = """
        SELECT 
            DATE(r.date_posted) as date,
            rc.department,
            rc.top_app_category,
            rc.top_railway_category,
            r.rating,
            COUNT(*) as count
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE r.date_posted > datetime('now', '-{} days')
        GROUP BY DATE(r.date_posted), rc.department, rc.top_app_category, rc.top_railway_category
        """.format(days)
        
        df = pd.read_sql_query(query, self.conn)
        
        # Identify spike patterns
        spikes = self._detect_spikes(df)
        
        # Hour-wise analysis for tatkal issues
        tatkal_query = """
        SELECT 
            strftime('%H', r.date_posted) as hour,
            COUNT(*) as count
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE rc.top_app_category = 'booking_system'
        AND r.content LIKE '%tatkal%'
        GROUP BY hour
        """
        
        tatkal_hourly = pd.read_sql_query(tatkal_query, self.conn)
        
        return {
            'daily_trends': df.to_dict('records'),
            'spikes': spikes,
            'tatkal_peak_hours': tatkal_hourly.to_dict('records'),
            'analysis_period': f'{days} days'
        }
    
    def _detect_spikes(self, df: pd.DataFrame, threshold: float = 2.0) -> List[Dict]:
        """Detect anomalous spikes in issue frequency"""
        spikes = []
        
        for category in df['top_app_category'].unique():
            if pd.isna(category):
                continue
                
            cat_data = df[df['top_app_category'] == category].groupby('date')['count'].sum()
            
            if len(cat_data) > 3:
                mean = cat_data.mean()
                std = cat_data.std()
                
                for date, count in cat_data.items():
                    z_score = (count - mean) / std if std > 0 else 0
                    if z_score > threshold:
                        spikes.append({
                            'date': date,
                            'category': category,
                            'count': int(count),
                            'z_score': round(z_score, 2),
                            'severity': 'high' if z_score > 3 else 'medium'
                        })
        
        return spikes
    
    def extract_failure_chains(self, limit: int = 100) -> Dict:
        """Extract and analyze failure chains from reviews"""
        
        query = """
        SELECT r.content, rc.department, rc.top_app_category
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE r.rating <= 2
        AND rc.department IN ('app', 'railway')
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=(limit,))
        
        chains = defaultdict(list)
        
        for _, row in df.iterrows():
            content = row['content'].lower()
            
            # Extract sequence indicators
            sequence_patterns = [
                r'first[,\s]+(.+?)[,\s]+then[,\s]+(.+)',
                r'after[,\s]+(.+?)[,\s]+(.+)',
                r'when[,\s]+(.+?)[,\s]+(.+)',
                r'tried[,\s]+(.+?)[,\s]+but[,\s]+(.+)'
            ]
            
            for pattern in sequence_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        chains[row['department']].append({
                            'step1': match[0][:100],
                            'step2': match[1][:100],
                            'category': row['top_app_category']
                        })
        
        return {
            'failure_chains': dict(chains),
            'chain_count': {k: len(v) for k, v in chains.items()}
        }
    
    def perform_statistical_analysis(self) -> Dict:
        """Perform statistical correlation and regression analysis"""
        
        # Get data for analysis
        query = """
        SELECT 
            rc.department,
            rc.confidence,
            rc.app_score,
            rc.railway_score,
            r.rating,
            LENGTH(r.content) as review_length,
            CASE WHEN r.content LIKE '%!%' THEN 1 ELSE 0 END as has_exclamation,
            CASE WHEN r.content LIKE '%?%' THEN 1 ELSE 0 END as has_question,
            CASE WHEN UPPER(r.content) = r.content THEN 1 ELSE 0 END as all_caps
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE rc.department IN ('app', 'railway')
        LIMIT 10000
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        # Select only numeric columns for correlation
        numeric_cols = ['confidence', 'app_score', 'railway_score', 'rating', 
                       'review_length', 'has_exclamation', 'has_question', 'all_caps']
        df_numeric = df[numeric_cols]
        
        # Correlation analysis
        correlations = df_numeric.corr()
        
        # Key correlations
        key_correlations = {
            'rating_vs_confidence': correlations.loc['rating', 'confidence'] if 'confidence' in correlations.index else None,
            'rating_vs_review_length': correlations.loc['rating', 'review_length'] if 'review_length' in correlations.index else None,
            'rating_vs_exclamation': correlations.loc['rating', 'has_exclamation'] if 'has_exclamation' in correlations.index else None,
            'app_score_vs_rating': correlations.loc['app_score', 'rating'] if 'app_score' in correlations.index and 'rating' in correlations.columns else None
        }
        
        # Statistical tests
        app_ratings = df[df['department'] == 'app']['rating']
        railway_ratings = df[df['department'] == 'railway']['rating']
        
        if len(app_ratings) > 0 and len(railway_ratings) > 0:
            t_stat, p_value = stats.ttest_ind(app_ratings, railway_ratings)
            significance = 'significant' if p_value < 0.05 else 'not significant'
        else:
            t_stat, p_value, significance = None, None, None
        
        return {
            'correlations': key_correlations,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significance': significance,
                'interpretation': 'App and Railway issues have significantly different ratings' 
                                if significance == 'significant' else 
                                'No significant difference in ratings between departments'
            },
            'summary_stats': {
                'app': {
                    'mean_rating': float(app_ratings.mean()) if len(app_ratings) > 0 else None,
                    'std_rating': float(app_ratings.std()) if len(app_ratings) > 0 else None,
                    'count': len(app_ratings)
                },
                'railway': {
                    'mean_rating': float(railway_ratings.mean()) if len(railway_ratings) > 0 else None,
                    'std_rating': float(railway_ratings.std()) if len(railway_ratings) > 0 else None,
                    'count': len(railway_ratings)
                }
            }
        }
    
    def extract_five_whys(self, sample_size: int = 50) -> Dict:
        """Automated 5-Why analysis using causal patterns"""
        
        query = """
        SELECT r.content, rc.department, rc.top_app_category, rc.top_railway_category
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE r.rating <= 2
        AND (r.content LIKE '%because%' 
             OR r.content LIKE '%due to%' 
             OR r.content LIKE '%when%'
             OR r.content LIKE '%after%'
             OR r.content LIKE '%since%')
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, self.conn, params=(sample_size,))
        
        why_chains = defaultdict(list)
        
        # Causal indicators
        causal_patterns = [
            (r'(.+?)\s+because\s+(.+)', 'because'),
            (r'(.+?)\s+due to\s+(.+)', 'due to'),
            (r'when\s+(.+?)[,\s]+(.+)', 'when'),
            (r'after\s+(.+?)[,\s]+(.+)', 'after'),
            (r'since\s+(.+?)[,\s]+(.+)', 'since')
        ]
        
        for _, row in df.iterrows():
            content = row['content'].lower()
            category = row['top_app_category'] or row['top_railway_category'] or 'unknown'
            
            for pattern, indicator in causal_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    for match in matches:
                        why_chains[category].append({
                            'problem': match[0][:100],
                            'cause': match[1][:100],
                            'indicator': indicator,
                            'department': row['department']
                        })
        
        # Build why trees
        why_trees = self._build_why_trees(why_chains)
        
        return {
            'causal_chains': dict(why_chains),
            'why_trees': why_trees,
            'total_causal_patterns': sum(len(v) for v in why_chains.values())
        }
    
    def _build_why_trees(self, why_chains: Dict) -> Dict:
        """Build hierarchical why trees from causal chains"""
        trees = {}
        
        for category, chains in why_chains.items():
            if not chains:
                continue
                
            # Group similar problems
            problems = defaultdict(list)
            for chain in chains:
                problems[chain['problem'][:30]].append(chain['cause'])
            
            # Build tree structure
            tree = {
                'category': category,
                'root_problems': []
            }
            
            for problem, causes in list(problems.items())[:5]:  # Top 5 problems
                tree['root_problems'].append({
                    'problem': problem,
                    'causes': Counter(causes).most_common(3)
                })
            
            trees[category] = tree
        
        return trees
    
    def classify_severity(self) -> Dict:
        """Classify issues by severity and business impact"""
        
        severity_results = defaultdict(list)
        
        query = """
        SELECT 
            r.id,
            r.content,
            r.rating,
            rc.department,
            rc.top_app_category,
            rc.top_railway_category
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE rc.department IN ('app', 'railway')
        LIMIT 5000
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        for _, row in df.iterrows():
            content = row['content'].lower()
            
            # Determine severity
            severity = 'low'
            for level, config in self.severity_levels.items():
                if any(keyword in content for keyword in config['keywords']):
                    severity = level
                    break
            
            # Business impact scoring
            impact_score = self._calculate_business_impact(content, row['department'])
            
            severity_results[severity].append({
                'department': row['department'],
                'category': row['top_app_category'] or row['top_railway_category'],
                'impact_score': impact_score,
                'rating': row['rating']
            })
        
        # Aggregate results
        summary = {}
        for severity, items in severity_results.items():
            df_severity = pd.DataFrame(items)
            if not df_severity.empty:
                summary[severity] = {
                    'count': len(items),
                    'avg_impact': df_severity['impact_score'].mean(),
                    'avg_rating': df_severity['rating'].mean(),
                    'top_categories': df_severity['category'].value_counts().head(3).to_dict()
                }
        
        return {
            'severity_distribution': summary,
            'total_analyzed': len(df)
        }
    
    def _calculate_business_impact(self, content: str, department: str) -> float:
        """Calculate business impact score for an issue"""
        score = 0
        
        # Revenue impact keywords
        revenue_keywords = ['payment', 'refund', 'money', 'booking', 'ticket', 'transaction']
        score += sum(2 for keyword in revenue_keywords if keyword in content)
        
        # User retention impact
        retention_keywords = ['uninstall', 'never use', 'worst', 'terrible', 'useless']
        score += sum(3 for keyword in retention_keywords if keyword in content)
        
        # Operational impact
        operational_keywords = ['staff', 'delay', 'late', 'cancel', 'dirty']
        score += sum(1.5 for keyword in operational_keywords if keyword in content)
        
        # Critical functionality
        critical_keywords = ['cannot book', 'not working', 'failed', 'error', 'crash']
        score += sum(2.5 for keyword in critical_keywords if keyword in content)
        
        return min(score, 10)  # Cap at 10
    
    def extract_contextual_patterns(self) -> Dict:
        """Extract contextual patterns and scenarios"""
        
        scenarios = {
            'peak_hour_failure': {
                'keywords': ['10am', 'tatkal', 'server', 'down', 'timeout'],
                'root_cause': 'Infrastructure scaling during peak hours',
                'solution': 'Auto-scaling, CDN implementation, load balancing',
                'count': 0
            },
            'payment_loop': {
                'keywords': ['payment', 'deducted', 'no ticket', 'multiple', 'retry'],
                'root_cause': 'Transaction state management failure',
                'solution': 'Idempotent payment API, better state handling',
                'count': 0
            },
            'session_timeout': {
                'keywords': ['session', 'expired', 'logout', 'login again', 'timeout'],
                'root_cause': 'Short session timeout configuration',
                'solution': 'Increase session timeout, implement session refresh',
                'count': 0
            },
            'otp_failure': {
                'keywords': ['otp', 'not received', 'expired', 'invalid', 'resend'],
                'root_cause': 'SMS gateway issues or delays',
                'solution': 'Multiple OTP channels, faster SMS provider',
                'count': 0
            },
            'search_failure': {
                'keywords': ['search', 'not found', 'no results', 'wrong station'],
                'root_cause': 'Search algorithm or data issues',
                'solution': 'Fuzzy search, better indexing, data validation',
                'count': 0
            }
        }
        
        # Count scenario occurrences
        query = """
        SELECT r.content
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE rc.department = 'app'
        LIMIT 10000
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        for _, row in df.iterrows():
            content = row['content'].lower()
            
            for scenario_name, scenario in scenarios.items():
                if sum(1 for keyword in scenario['keywords'] if keyword in content) >= 2:
                    scenarios[scenario_name]['count'] += 1
        
        # Sort by frequency
        sorted_scenarios = sorted(scenarios.items(), key=lambda x: x[1]['count'], reverse=True)
        
        return {
            'identified_scenarios': dict(sorted_scenarios),
            'total_reviews_analyzed': len(df)
        }
    
    def perform_clustering_analysis(self, n_clusters: int = 10) -> Dict:
        """Cluster similar issues using ML"""
        
        query = """
        SELECT r.content, rc.department, rc.top_app_category
        FROM reviews r
        JOIN review_classifications rc ON r.id = rc.review_id
        WHERE r.rating <= 2
        AND rc.department = 'app'
        LIMIT 5000
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        if len(df) < n_clusters:
            return {'error': 'Insufficient data for clustering'}
        
        # Vectorize reviews
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(df['content'])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Get top terms for each cluster
        cluster_info = []
        feature_names = vectorizer.get_feature_names_out()
        
        for i in range(n_clusters):
            # Get cluster center
            center = kmeans.cluster_centers_[i]
            
            # Get top terms
            top_indices = center.argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            # Get cluster size
            cluster_size = (clusters == i).sum()
            
            # Sample reviews from cluster
            cluster_reviews = df[clusters == i]['content'].head(3).tolist()
            
            cluster_info.append({
                'cluster_id': i,
                'size': int(cluster_size),
                'top_terms': top_terms[:5],
                'sample_reviews': cluster_reviews[:2]
            })
        
        # Sort by size
        cluster_info.sort(key=lambda x: x['size'], reverse=True)
        
        return {
            'clusters': cluster_info,
            'total_clusters': n_clusters,
            'total_reviews': len(df)
        }
    
    def detect_anomalies(self) -> Dict:
        """Detect anomalous patterns in reviews"""
        
        query = """
        SELECT 
            DATE(r.date_posted) as date,
            COUNT(*) as daily_count,
            AVG(r.rating) as avg_rating,
            COUNT(CASE WHEN r.rating = 1 THEN 1 END) as one_star_count
        FROM reviews r
        GROUP BY DATE(r.date_posted)
        HAVING daily_count > 10
        ORDER BY date DESC
        LIMIT 90
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        anomalies = []
        
        # Detect volume anomalies
        mean_count = df['daily_count'].mean()
        std_count = df['daily_count'].std()
        
        for _, row in df.iterrows():
            z_score = (row['daily_count'] - mean_count) / std_count if std_count > 0 else 0
            
            if abs(z_score) > 2:
                anomalies.append({
                    'date': row['date'],
                    'type': 'volume_spike' if z_score > 0 else 'volume_drop',
                    'daily_count': int(row['daily_count']),
                    'z_score': round(z_score, 2),
                    'avg_rating': round(row['avg_rating'], 2)
                })
        
        # Detect rating anomalies
        mean_rating = df['avg_rating'].mean()
        std_rating = df['avg_rating'].std()
        
        for _, row in df.iterrows():
            z_score = (row['avg_rating'] - mean_rating) / std_rating if std_rating > 0 else 0
            
            if abs(z_score) > 2:
                existing = next((a for a in anomalies if a['date'] == row['date']), None)
                if not existing:
                    anomalies.append({
                        'date': row['date'],
                        'type': 'rating_anomaly',
                        'avg_rating': round(row['avg_rating'], 2),
                        'z_score': round(z_score, 2)
                    })
        
        return {
            'anomalies': sorted(anomalies, key=lambda x: x['date'], reverse=True),
            'summary': {
                'total_anomalies': len(anomalies),
                'volume_anomalies': sum(1 for a in anomalies if 'volume' in a['type']),
                'rating_anomalies': sum(1 for a in anomalies if 'rating' in a['type'])
            }
        }
    
    def generate_root_cause_summary(self) -> Dict:
        """Generate comprehensive root cause analysis summary"""
        
        logger.info("Starting comprehensive root cause analysis...")
        
        # Run all analyses
        temporal = self.analyze_temporal_patterns()
        failure_chains = self.extract_failure_chains()
        statistical = self.perform_statistical_analysis()
        five_whys = self.extract_five_whys()
        severity = self.classify_severity()
        contextual = self.extract_contextual_patterns()
        clustering = self.perform_clustering_analysis()
        anomalies = self.detect_anomalies()
        
        # Identify top root causes
        top_root_causes = self._identify_top_root_causes(contextual, five_whys)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(top_root_causes, severity, temporal)
        
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'temporal_patterns': temporal,
            'failure_chains': failure_chains,
            'statistical_analysis': statistical,
            'five_why_analysis': five_whys,
            'severity_classification': severity,
            'contextual_patterns': contextual,
            'clustering_results': clustering,
            'anomaly_detection': anomalies,
            'top_root_causes': top_root_causes,
            'recommendations': recommendations
        }
        
        # Save results
        output_path = Path('data/analysis/root_cause_analysis.pkl')
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(summary, f)
        
        logger.info(f"Root cause analysis saved to {output_path}")
        
        return summary
    
    def _identify_top_root_causes(self, contextual: Dict, five_whys: Dict) -> List[Dict]:
        """Identify and rank top root causes"""
        
        root_causes = []
        
        # From contextual patterns
        for scenario_name, scenario in contextual['identified_scenarios'].items():
            if scenario['count'] > 0:
                root_causes.append({
                    'cause': scenario['root_cause'],
                    'frequency': scenario['count'],
                    'solution': scenario['solution'],
                    'source': 'contextual_analysis'
                })
        
        # Sort by frequency
        root_causes.sort(key=lambda x: x['frequency'], reverse=True)
        
        return root_causes[:10]
    
    def _generate_recommendations(self, root_causes: List, severity: Dict, temporal: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Based on top root causes
        for i, cause in enumerate(root_causes[:5], 1):
            recommendations.append({
                'priority': i,
                'issue': cause['cause'],
                'solution': cause['solution'],
                'impact': 'high' if cause['frequency'] > 100 else 'medium',
                'estimated_effort': 'medium'
            })
        
        # Based on severity
        if 'critical' in severity['severity_distribution']:
            critical_count = severity['severity_distribution']['critical']['count']
            if critical_count > 100:
                recommendations.insert(0, {
                    'priority': 0,
                    'issue': 'High number of critical issues',
                    'solution': 'Immediate task force to address critical failures',
                    'impact': 'critical',
                    'estimated_effort': 'high'
                })
        
        return recommendations


def main():
    """Main execution function"""
    analyzer = RootCauseAnalyzer()
    
    # Generate comprehensive analysis
    results = analyzer.generate_root_cause_summary()
    
    # Print summary
    print("\n" + "="*80)
    print("ROOT CAUSE ANALYSIS SUMMARY")
    print("="*80)
    
    # Top root causes
    print("\nTOP ROOT CAUSES:")
    for i, cause in enumerate(results['top_root_causes'][:5], 1):
        print(f"{i}. {cause['cause']}")
        print(f"   Frequency: {cause['frequency']} | Solution: {cause['solution']}")
    
    # Key recommendations
    print("\nKEY RECOMMENDATIONS:")
    for rec in results['recommendations'][:5]:
        print(f"Priority {rec['priority']}: {rec['solution']}")
        print(f"   Impact: {rec['impact']} | Effort: {rec['estimated_effort']}")
    
    # Anomalies
    if results['anomaly_detection']['anomalies']:
        print(f"\nDETECTED ANOMALIES: {len(results['anomaly_detection']['anomalies'])}")
        for anomaly in results['anomaly_detection']['anomalies'][:3]:
            print(f"  - {anomaly['date']}: {anomaly['type']} (z-score: {anomaly['z_score']})")
    
    print("\n" + "="*80)
    print("Analysis complete. Full results saved to data/analysis/root_cause_analysis.pkl")
    

if __name__ == "__main__":
    main()