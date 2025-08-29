#!/usr/bin/env python3
"""
Complete Automated Pipeline for IRCTC Review Analysis
Scrapes new reviews, updates models, runs analysis, and generates reports
"""

import sys
import os
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd
import pickle
import json

# Add src to path
sys.path.append('src')
sys.path.append('src/scraping')
sys.path.append('src/modeling')
sys.path.append('src/analysis')

from src.scraping.batch_scraper import BatchReviewScraper
from src.analysis.root_cause_analyzer import RootCauseAnalyzer
from src.analysis.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IRCTCPipeline:
    """Complete automated pipeline for IRCTC review analysis"""
    
    def __init__(self):
        self.db_path = 'data/reviews.db'
        self.models_path = Path('data/models')
        self.reports_path = Path('data/reports')
        self.scraping_state_path = 'data/scraping_state.json'
        
        # Ensure directories exist
        self.models_path.mkdir(exist_ok=True)
        self.reports_path.mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
    
    def get_current_review_count(self):
        """Get current number of reviews in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reviews")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Error getting review count: {e}")
            return 0
    
    def scrape_new_reviews(self, max_new_reviews=1000):
        """Scrape new reviews (default: 1000 new reviews per run)"""
        logger.info(f"=== STEP 1: SCRAPING NEW REVIEWS (Target: {max_new_reviews}) ===")
        
        initial_count = self.get_current_review_count()
        logger.info(f"Current reviews in database: {initial_count:,}")
        
        try:
            scraper = BatchReviewScraper()
            
            # Calculate batches needed (assuming 1000 reviews per batch)
            reviews_per_batch = 1000
            batches_needed = max(1, max_new_reviews // reviews_per_batch)
            
            logger.info(f"Running {batches_needed} batch(es) to collect ~{max_new_reviews} reviews")
            
            # Run scraping
            scraper.run_batch_scraping(
                total_batches=batches_needed
            )
            
            scraper.close()
            
            final_count = self.get_current_review_count()
            new_reviews = final_count - initial_count
            logger.info(f"✅ Scraping complete. Added {new_reviews:,} new reviews")
            
            return new_reviews > 0
            
        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}")
            return False
    
    def update_topic_models(self):
        """Update topic models with latest data"""
        logger.info("=== STEP 2: UPDATING TOPIC MODELS ===")
        
        try:
            # Run the full analysis script
            result = subprocess.run([
                sys.executable, 'run_full_analysis.py'
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("✅ Topic models updated successfully")
                logger.info("Model outputs:")
                for line in result.stdout.split('\n')[-10:]:  # Last 10 lines
                    if line.strip():
                        logger.info(f"  {line}")
                return True
            else:
                logger.error(f"❌ Topic modeling failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Topic modeling timed out (>1 hour)")
            return False
        except Exception as e:
            logger.error(f"❌ Topic modeling error: {e}")
            return False
    
    def run_root_cause_analysis(self):
        """Run comprehensive root cause analysis"""
        logger.info("=== STEP 3: RUNNING ROOT CAUSE ANALYSIS ===")
        
        try:
            analyzer = RootCauseAnalyzer()
            results = analyzer.generate_root_cause_summary()  # Use correct method
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.models_path / f"root_cause_analysis_{timestamp}.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Also update the main file for dashboards
            with open('data/analysis/root_cause_analysis.pkl', 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"✅ Root cause analysis complete. Results saved to {output_file}")
            
            # Print summary
            if 'detected_patterns' in results:
                patterns = results['detected_patterns']
                logger.info(f"  - Detected {len(patterns)} patterns")
            
            if 'root_causes' in results:
                root_causes = results['root_causes']
                logger.info(f"  - Identified {len(root_causes)} root causes")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Root cause analysis failed: {e}")
            return False
    
    def generate_reports(self):
        """Generate automated reports"""
        logger.info("=== STEP 4: GENERATING REPORTS ===")
        
        try:
            generator = ReportGenerator()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate executive summary
            exec_report = generator.generate_executive_summary()
            exec_file = self.reports_path / f"executive_summary_{timestamp}.md"
            with open(exec_file, 'w') as f:
                f.write(exec_report)
            
            # Generate technical report
            tech_report = generator.generate_technical_report()
            tech_file = self.reports_path / f"technical_report_{timestamp}.md"
            with open(tech_file, 'w') as f:
                f.write(tech_report)
            
            # Generate JSON export
            json_report = generator.generate_json_report()
            json_file = self.reports_path / f"analysis_report_{timestamp}.json"
            with open(json_file, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            logger.info(f"✅ Reports generated:")
            logger.info(f"  - Executive: {exec_file}")
            logger.info(f"  - Technical: {tech_file}")
            logger.info(f"  - JSON: {json_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return False
    
    def restart_dashboards(self):
        """Restart dashboards to show latest data"""
        logger.info("=== STEP 5: RESTARTING DASHBOARDS ===")
        
        try:
            # Kill existing dashboards
            subprocess.run(['pkill', '-f', 'streamlit'], check=False)
            time.sleep(3)
            
            # Start dashboards in background
            dashboards = [
                ('src/dashboard/professional_app.py', 8502),
                ('src/dashboard/segregated_dashboard.py', 8505),
                ('src/dashboard/root_cause_dashboard.py', 8506)
            ]
            
            for script, port in dashboards:
                subprocess.Popen([
                    'streamlit', 'run', script,
                    '--server.port', str(port),
                    '--server.headless', 'true'
                ])
                time.sleep(2)
            
            logger.info("✅ Dashboards restarted:")
            for script, port in dashboards:
                dashboard_name = script.split('/')[-1].replace('_', ' ').replace('.py', '').title()
                logger.info(f"  - {dashboard_name}: http://localhost:{port}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dashboard restart failed: {e}")
            return False
    
    def run_complete_pipeline(self, max_new_reviews=1000, skip_scraping=False):
        """Run the complete pipeline"""
        start_time = datetime.now()
        logger.info("🚀 STARTING COMPLETE IRCTC ANALYSIS PIPELINE")
        logger.info(f"Timestamp: {start_time}")
        logger.info("="*80)
        
        success_steps = 0
        total_steps = 5
        
        # Step 1: Scrape new reviews
        if not skip_scraping:
            if self.scrape_new_reviews(max_new_reviews):
                success_steps += 1
        else:
            logger.info("=== STEP 1: SKIPPING SCRAPING (skip_scraping=True) ===")
            success_steps += 1
        
        # Step 2: Update topic models
        if self.update_topic_models():
            success_steps += 1
        
        # Step 3: Run root cause analysis
        if self.run_root_cause_analysis():
            success_steps += 1
        
        # Step 4: Generate reports
        if self.generate_reports():
            success_steps += 1
        
        # Step 5: Restart dashboards
        if self.restart_dashboards():
            success_steps += 1
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*80)
        logger.info("🎯 PIPELINE EXECUTION SUMMARY")
        logger.info(f"Started: {start_time}")
        logger.info(f"Completed: {end_time}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Success Rate: {success_steps}/{total_steps} ({success_steps/total_steps*100:.1f}%)")
        
        if success_steps == total_steps:
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            logger.warning(f"⚠️ PIPELINE COMPLETED WITH {total_steps - success_steps} FAILURES")
        
        logger.info("="*80)
        
        return success_steps == total_steps

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run IRCTC Complete Analysis Pipeline')
    parser.add_argument('--max-reviews', type=int, default=1000, 
                       help='Maximum new reviews to scrape (default: 1000)')
    parser.add_argument('--skip-scraping', action='store_true',
                       help='Skip scraping step (use existing data)')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run only analysis and reports (skip scraping and modeling)')
    
    args = parser.parse_args()
    
    pipeline = IRCTCPipeline()
    
    if args.analysis_only:
        # Run only steps 3-5
        logger.info("Running analysis-only pipeline...")
        success = True
        success &= pipeline.run_root_cause_analysis()
        success &= pipeline.generate_reports()
        success &= pipeline.restart_dashboards()
    else:
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            max_new_reviews=args.max_reviews,
            skip_scraping=args.skip_scraping
        )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())