#!/usr/bin/env python3
"""
Script to run full batch scraping to collect 100k reviews total
"""

from src.scraping.batch_scraper import BatchReviewScraper
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("STARTING FULL BATCH SCRAPING FOR 100K REVIEWS")
    logger.info("="*60)
    logger.info("Target: Collect remaining ~90,000 reviews")
    logger.info("This will take several hours. You can interrupt and resume anytime.")
    logger.info("="*60)
    
    scraper = BatchReviewScraper()
    
    # Reset batch size to 10k
    scraper.BATCH_SIZE = 10000
    
    # Check current state
    logger.info(f"Current state - Batch: {scraper.state['current_batch']}, Total scraped: {scraper.state['total_scraped']}")
    
    # If we haven't started or are at batch 0/1, start from batch 1
    if scraper.state['current_batch'] <= 1:
        scraper.state['current_batch'] = 1
        if scraper.state['total_scraped'] < 10000:
            scraper.state['total_scraped'] = 10005  # We have ~10k from initial scraping
        scraper._save_state()
    
    try:
        # Run batches until we get 100k total (10 batches of 10k each)
        # The scraper will continue from where it left off
        start_time = time.time()
        
        scraper.run_batch_scraping(total_batches=10, start_batch=scraper.state['current_batch'])
        
        elapsed = time.time() - start_time
        logger.info(f"Total time: {elapsed/3600:.2f} hours")
        
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("SCRAPING INTERRUPTED BY USER")
        logger.info("Progress has been saved. Run this script again to continue.")
        logger.info("="*60)
        scraper._save_state()
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        import traceback
        logger.error(traceback.format_exc())
        scraper._save_state()
    finally:
        scraper.display_final_stats()
        scraper.close()
        logger.info("Scraping session ended.")

if __name__ == "__main__":
    main()