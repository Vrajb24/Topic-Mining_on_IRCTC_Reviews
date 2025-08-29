#!/usr/bin/env python3
"""
Automated Report Generator for IRCTC Root Cause Analysis
Generates comprehensive reports in multiple formats
"""

import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, analysis_path: str = 'data/analysis/root_cause_analysis.pkl'):
        self.analysis_path = analysis_path
        self.load_analysis()
        self.report_date = datetime.now()
    
    def load_analysis(self):
        """Load analysis results"""
        try:
            with open(self.analysis_path, 'rb') as f:
                self.results = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"Analysis file not found: {self.analysis_path}")
            self.results = None
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary"""
        if not self.results:
            return "No analysis results available."
        
        summary = f"""
# IRCTC Review Analysis - Executive Summary
Generated: {self.report_date.strftime('%B %d, %Y')}

## Key Findings

### 1. Critical Issues Identified
"""
        
        # Top root causes
        if 'top_root_causes' in self.results:
            for i, cause in enumerate(self.results['top_root_causes'][:3], 1):
                summary += f"- **{cause['cause']}** (Frequency: {cause['frequency']})\n"
                summary += f"  - Solution: {cause['solution']}\n"
        
        summary += "\n### 2. Impact Analysis\n"
        
        # Severity distribution
        if 'severity_classification' in self.results:
            severity = self.results['severity_classification'].get('severity_distribution', {})
            if 'critical' in severity:
                summary += f"- **Critical Issues:** {severity['critical']['count']} cases\n"
            if 'high' in severity:
                summary += f"- **High Priority Issues:** {severity['high']['count']} cases\n"
        
        summary += "\n### 3. Department-wise Breakdown\n"
        
        # Statistical insights
        if 'statistical_analysis' in self.results:
            stats = self.results['statistical_analysis'].get('summary_stats', {})
            if 'app' in stats:
                summary += f"- **App Issues:** {stats['app']['count']} reviews, Avg Rating: {stats['app']['mean_rating']:.2f}\n"
            if 'railway' in stats:
                summary += f"- **Railway Issues:** {stats['railway']['count']} reviews, Avg Rating: {stats['railway']['mean_rating']:.2f}\n"
        
        summary += "\n### 4. Anomalies Detected\n"
        
        if 'anomaly_detection' in self.results:
            anomalies = self.results['anomaly_detection'].get('summary', {})
            summary += f"- **Total Anomalies:** {anomalies.get('total_anomalies', 0)}\n"
            summary += f"- **Volume Spikes:** {anomalies.get('volume_anomalies', 0)}\n"
            summary += f"- **Rating Anomalies:** {anomalies.get('rating_anomalies', 0)}\n"
        
        summary += "\n## Priority Recommendations\n\n"
        
        # Recommendations
        if 'recommendations' in self.results:
            for rec in self.results['recommendations'][:5]:
                summary += f"### Priority {rec['priority']}: {rec['issue']}\n"
                summary += f"**Solution:** {rec['solution']}\n"
                summary += f"**Impact:** {rec['impact'].upper()} | **Effort:** {rec['estimated_effort'].upper()}\n\n"
        
        summary += """
## Next Steps

1. **Immediate Actions**
   - Form a task force to address critical infrastructure issues
   - Implement auto-scaling for peak hour traffic
   - Fix payment gateway timeout issues

2. **Short-term (1-3 months)**
   - Improve session management
   - Enhance error handling and recovery
   - Implement better monitoring and alerting

3. **Long-term (3-6 months)**
   - Complete infrastructure overhaul
   - Implement AI-based predictive maintenance
   - Establish continuous improvement process

---
*This report is automatically generated from comprehensive root cause analysis of 90,000+ IRCTC reviews*
"""
        
        return summary
    
    def generate_technical_report(self) -> str:
        """Generate detailed technical report"""
        if not self.results:
            return "No analysis results available."
        
        report = f"""
# IRCTC Technical Analysis Report
Generated: {self.report_date.strftime('%B %d, %Y at %I:%M %p')}

## 1. Temporal Analysis
"""
        
        # Temporal patterns
        if 'temporal_patterns' in self.results:
            temporal = self.results['temporal_patterns']
            report += f"- Analysis Period: {temporal.get('analysis_period', 'N/A')}\n"
            
            if temporal.get('spikes'):
                report += f"\n### Detected Spikes ({len(temporal['spikes'])} total)\n"
                for spike in temporal['spikes'][:10]:
                    report += f"- {spike['date']}: {spike['category']} ({spike['count']} issues, Z-score: {spike['z_score']})\n"
            
            if temporal.get('tatkal_peak_hours'):
                report += "\n### Tatkal Booking Peak Hours\n"
                peak_hours = temporal['tatkal_peak_hours']
                if peak_hours:
                    sorted_hours = sorted(peak_hours, key=lambda x: x['count'], reverse=True)[:5]
                    for hour_data in sorted_hours:
                        report += f"- Hour {hour_data['hour']}: {hour_data['count']} issues\n"
        
        report += "\n## 2. Failure Chain Analysis\n"
        
        # Failure chains
        if 'failure_chains' in self.results:
            chains = self.results['failure_chains']
            if 'chain_count' in chains:
                for dept, count in chains['chain_count'].items():
                    report += f"- {dept.title()} Department: {count} failure chains identified\n"
        
        report += "\n## 3. Statistical Analysis\n"
        
        # Statistical insights
        if 'statistical_analysis' in self.results:
            stats = self.results['statistical_analysis']
            
            if 'correlations' in stats:
                report += "\n### Key Correlations\n"
                for key, value in stats['correlations'].items():
                    if value is not None:
                        report += f"- {key.replace('_', ' ').title()}: {value:.3f}\n"
            
            if 'statistical_test' in stats:
                test = stats['statistical_test']
                report += f"\n### Department Comparison (T-Test)\n"
                report += f"- T-Statistic: {test.get('t_statistic', 'N/A')}\n"
                report += f"- P-Value: {test.get('p_value', 'N/A')}\n"
                report += f"- Result: {test.get('interpretation', 'N/A')}\n"
        
        report += "\n## 4. Clustering Analysis\n"
        
        # Clustering results
        if 'clustering_results' in self.results:
            clustering = self.results['clustering_results']
            report += f"- Total Clusters: {clustering.get('total_clusters', 0)}\n"
            report += f"- Reviews Analyzed: {clustering.get('total_reviews', 0)}\n"
            
            if 'clusters' in clustering:
                report += "\n### Top Issue Clusters\n"
                for cluster in clustering['clusters'][:5]:
                    report += f"\n**Cluster {cluster['cluster_id'] + 1}** (Size: {cluster['size']})\n"
                    report += f"- Key Terms: {', '.join(cluster['top_terms'][:5])}\n"
        
        report += "\n## 5. Contextual Pattern Analysis\n"
        
        # Contextual patterns
        if 'contextual_patterns' in self.results:
            patterns = self.results['contextual_patterns']
            if 'identified_scenarios' in patterns:
                report += "\n### Identified Scenarios\n"
                scenarios = patterns['identified_scenarios']
                
                # Sort by count
                sorted_scenarios = sorted(scenarios.items(), key=lambda x: x[1]['count'], reverse=True)
                
                for scenario_name, scenario_data in sorted_scenarios[:10]:
                    if scenario_data['count'] > 0:
                        report += f"\n**{scenario_name.replace('_', ' ').title()}**\n"
                        report += f"- Occurrences: {scenario_data['count']}\n"
                        report += f"- Root Cause: {scenario_data['root_cause']}\n"
                        report += f"- Solution: {scenario_data['solution']}\n"
        
        report += "\n## 6. Five-Why Analysis\n"
        
        # Five-why analysis
        if 'five_why_analysis' in self.results:
            five_why = self.results['five_why_analysis']
            report += f"- Total Causal Patterns: {five_why.get('total_causal_patterns', 0)}\n"
            
            if 'why_trees' in five_why:
                report += "\n### Sample Causal Chains\n"
                trees = five_why['why_trees']
                for category in list(trees.keys())[:3]:
                    tree = trees[category]
                    if 'root_problems' in tree:
                        report += f"\n**{category.replace('_', ' ').title()}**\n"
                        for problem in tree['root_problems'][:2]:
                            report += f"- Problem: {problem['problem']}\n"
                            if problem['causes']:
                                for cause, freq in problem['causes'][:2]:
                                    report += f"  → Cause: {cause} (Freq: {freq})\n"
        
        report += """

## 7. Technical Recommendations

### Infrastructure
1. Implement auto-scaling with Kubernetes
2. Deploy CDN for static content
3. Implement circuit breakers for third-party services
4. Add Redis caching layer

### Application
1. Implement idempotent APIs
2. Add retry logic with exponential backoff
3. Improve error handling and recovery
4. Implement proper logging and monitoring

### Database
1. Optimize slow queries
2. Implement read replicas
3. Add proper indexing
4. Implement connection pooling

### Monitoring
1. Set up real-time alerts
2. Implement APM (Application Performance Monitoring)
3. Add business metrics dashboards
4. Implement SLA tracking

---
*Generated by Root Cause Analysis System*
"""
        
        return report
    
    def generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON format report"""
        if not self.results:
            return {"error": "No analysis results available"}
        
        # Clean up the results for JSON serialization
        clean_results = {}
        
        for key, value in self.results.items():
            if key in ['temporal_patterns', 'failure_chains', 'statistical_analysis', 
                      'five_why_analysis', 'severity_classification', 'contextual_patterns',
                      'clustering_results', 'anomaly_detection', 'top_root_causes', 
                      'recommendations']:
                clean_results[key] = self._clean_for_json(value)
        
        clean_results['report_metadata'] = {
            'generated_at': self.report_date.isoformat(),
            'analysis_timestamp': self.results.get('analysis_timestamp', ''),
            'report_type': 'comprehensive_root_cause_analysis'
        }
        
        return clean_results
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def save_reports(self, output_dir: str = 'data/reports'):
        """Save all report formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = self.report_date.strftime('%Y%m%d_%H%M%S')
        
        # Save executive summary
        exec_summary = self.generate_executive_summary()
        exec_path = output_path / f'executive_summary_{timestamp}.md'
        with open(exec_path, 'w') as f:
            f.write(exec_summary)
        logger.info(f"Executive summary saved to {exec_path}")
        
        # Save technical report
        tech_report = self.generate_technical_report()
        tech_path = output_path / f'technical_report_{timestamp}.md'
        with open(tech_path, 'w') as f:
            f.write(tech_report)
        logger.info(f"Technical report saved to {tech_path}")
        
        # Save JSON report
        json_report = self.generate_json_report()
        json_path = output_path / f'analysis_report_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        logger.info(f"JSON report saved to {json_path}")
        
        return {
            'executive_summary': exec_path,
            'technical_report': tech_path,
            'json_report': json_path
        }


def main():
    """Main execution function"""
    generator = ReportGenerator()
    
    if not generator.results:
        print("Please run root_cause_analyzer.py first to generate analysis results.")
        return
    
    print("\n" + "="*80)
    print("GENERATING ANALYSIS REPORTS")
    print("="*80)
    
    # Generate and save reports
    saved_files = generator.save_reports()
    
    print("\nReports generated successfully:")
    for report_type, path in saved_files.items():
        print(f"- {report_type.replace('_', ' ').title()}: {path}")
    
    # Print executive summary preview
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY PREVIEW")
    print("="*80)
    
    summary = generator.generate_executive_summary()
    preview = summary.split('\n')[:30]
    print('\n'.join(preview))
    print("\n... (continued in file)")
    
    print("\n" + "="*80)
    print("Report generation complete!")


if __name__ == "__main__":
    main()