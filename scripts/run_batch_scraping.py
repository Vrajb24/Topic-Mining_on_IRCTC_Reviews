#!/usr/bin/env python3
"""
Script to run batch scraping with proper configuration
"""

from src.scraping.batch_scraper import BatchReviewScraper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting batch scraping for IRCTC reviews")
    logger.info("Target: 90,000 more reviews (9 batches of 10k each)")
    
    scraper = BatchReviewScraper()
    
    # We already have batch 0 (first 10k), so start from batch 1
    # Since we already have 10k reviews, we'll update the state
    scraper.state['current_batch'] = 1
    scraper.state['total_scraped'] = 10000
    scraper._save_state()
    
    try:
        # Run 9 more batches (batches 1-9) to get total 100k
        scraper.run_batch_scraping(total_batches=10, start_batch=1)
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()