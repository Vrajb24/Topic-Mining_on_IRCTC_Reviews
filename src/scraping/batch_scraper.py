#!/usr/bin/env python3
"""
Batch Scraper for IRCTC Reviews
Scrapes reviews in batches of 10k with proper continuation token management
"""

import sqlite3
import json
import time
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from google_play_scraper import app, reviews, Sort
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchReviewScraper:
    """Batch scraper with continuation token management"""
    
    IRCTC_APP_ID = 'cris.org.in.prs.ima'
    BATCH_SIZE = 10000  # 10k reviews per batch
    STATE_FILE = 'data/scraping_state.json'
    
    def __init__(self, db_path: str = 'data/reviews.db'):
        """Initialize the batch scraper"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._ensure_directories()
        self.state = self._load_state()
        
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        Path('logs').mkdir(exist_ok=True)
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        Path('data/tokens').mkdir(parents=True, exist_ok=True)
        
    def _load_state(self) -> Dict:
        """Load scraping state from file"""
        state_path = Path(self.STATE_FILE)
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state: Batch {state.get('current_batch', 0)}, "
                          f"Total scraped: {state.get('total_scraped', 0)}")
                return state
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        # Initialize new state
        return {
            'current_batch': 0,
            'total_scraped': 0,
            'continuation_token': None,
            'batch_history': [],
            'last_review_id': None
        }
    
    def _save_state(self):
        """Save current scraping state"""
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
        logger.info(f"State saved: Batch {self.state['current_batch']}, "
                   f"Total: {self.state['total_scraped']}")
    
    def _save_continuation_token(self, token, batch_num: int):
        """Save continuation token for resuming later"""
        if token:
            token_file = Path(f'data/tokens/token_batch_{batch_num}.pkl')
            with open(token_file, 'wb') as f:
                pickle.dump(token, f)
            logger.info(f"Saved continuation token for batch {batch_num}")
    
    def _load_continuation_token(self, batch_num: int):
        """Load continuation token for a batch"""
        token_file = Path(f'data/tokens/token_batch_{batch_num}.pkl')
        if token_file.exists():
            try:
                with open(token_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading token: {e}")
        return None
    
    def get_existing_review_ids(self) -> set:
        """Get all existing review IDs to avoid duplicates"""
        self.cursor.execute("SELECT review_id FROM reviews")
        existing_ids = {row[0] for row in self.cursor.fetchall()}
        logger.info(f"Found {len(existing_ids)} existing reviews in database")
        return existing_ids
    
    def scrape_batch(self, batch_num: int, target_count: int = BATCH_SIZE) -> Tuple[List[Dict], any]:
        """
        Scrape a single batch of reviews
        
        Returns:
            Tuple of (reviews list, continuation_token)
        """
        logger.info(f"="*60)
        logger.info(f"Starting Batch {batch_num} - Target: {target_count} reviews")
        logger.info(f"="*60)
        
        # Get existing review IDs to check for duplicates
        existing_ids = self.get_existing_review_ids()
        
        # Load continuation token from previous batch
        continuation_token = None
        if batch_num > 0:
            continuation_token = self._load_continuation_token(batch_num - 1)
            if continuation_token:
                logger.info(f"Using continuation token from batch {batch_num - 1}")
            else:
                logger.warning(f"No continuation token found for batch {batch_num - 1}")
        
        batch_reviews = []
        reviews_fetched = 0
        fetch_size = 200  # Fetch in smaller chunks
        
        pbar = tqdm(total=target_count, desc=f"Batch {batch_num}")
        
        while reviews_fetched < target_count:
            try:
                # Calculate how many reviews to fetch
                remaining = target_count - reviews_fetched
                count = min(fetch_size, remaining)
                
                # Fetch reviews
                result, new_token = reviews(
                    self.IRCTC_APP_ID,
                    lang='en',
                    country='in',
                    sort=Sort.NEWEST,
                    count=count,
                    continuation_token=continuation_token
                )
                
                if not result:
                    logger.warning("No more reviews available")
                    break
                
                # Filter out duplicates
                new_reviews = []
                for review in result:
                    review_id = review.get('reviewId')
                    if review_id not in existing_ids:
                        new_reviews.append(review)
                        existing_ids.add(review_id)
                    
                batch_reviews.extend(new_reviews)
                reviews_fetched += len(result)
                pbar.update(len(result))
                
                # Update continuation token
                continuation_token = new_token
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
                # Break if no more reviews
                if not new_token:
                    logger.info("No more reviews available from Play Store")
                    break
                    
            except Exception as e:
                logger.error(f"Error in batch scraping: {e}")
                time.sleep(5)
                continue
        
        pbar.close()
        
        logger.info(f"Batch {batch_num} complete: {len(batch_reviews)} new reviews")
        
        # Save continuation token for next batch
        if continuation_token:
            self._save_continuation_token(continuation_token, batch_num)
        
        return batch_reviews, continuation_token
    
    def save_batch_to_database(self, reviews_data: List[Dict], batch_num: int) -> int:
        """Save a batch of reviews to database"""
        saved_count = 0
        
        for review in tqdm(reviews_data, desc=f"Saving batch {batch_num}"):
            try:
                # Prepare data
                review_id = review.get('reviewId', '')
                content = review.get('content', '')
                rating = review.get('score', 0)
                
                # Parse date
                review_date = review.get('at')
                if review_date:
                    date_posted = review_date.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_posted = None
                
                # Additional metadata
                thumbs_up = review.get('thumbsUpCount', 0)
                user_name = review.get('userName', '')
                reply_content = review.get('replyContent', '')
                
                # Insert into database
                self.cursor.execute("""
                    INSERT OR IGNORE INTO reviews (
                        review_id, content, rating, language,
                        date_posted, date_scraped, processed,
                        thumbs_up, user_name, reply_content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    review_id, content, rating, 'auto',
                    date_posted, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    False, thumbs_up, user_name, reply_content
                ))
                
                if self.cursor.rowcount > 0:
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving review: {e}")
                continue
        
        self.conn.commit()
        logger.info(f"Saved {saved_count} reviews from batch {batch_num}")
        return saved_count
    
    def save_batch_to_json(self, reviews_data: List[Dict], batch_num: int):
        """Save batch to JSON for backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/raw/batch_{batch_num}_{timestamp}.json'
        
        filepath = Path(filename)
        
        # Convert datetime objects to strings
        json_data = []
        for review in reviews_data:
            review_copy = review.copy()
            for key, value in review_copy.items():
                if hasattr(value, 'strftime'):
                    review_copy[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            json_data.append(review_copy)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved batch {batch_num} to {filepath}")
    
    def run_batch_scraping(self, total_batches: int = 10, start_batch: Optional[int] = None):
        """
        Run batch scraping for specified number of batches
        
        Args:
            total_batches: Total number of batches to scrape (default 10 = 100k reviews)
            start_batch: Starting batch number (uses state if not specified)
        """
        if start_batch is None:
            start_batch = self.state['current_batch']
        
        logger.info(f"Starting batch scraping from batch {start_batch}")
        logger.info(f"Target: {total_batches} batches × {self.BATCH_SIZE} = {total_batches * self.BATCH_SIZE:,} reviews")
        
        for batch_num in range(start_batch, total_batches):
            try:
                # Update state
                self.state['current_batch'] = batch_num
                
                # Scrape batch
                batch_reviews, continuation_token = self.scrape_batch(batch_num)
                
                if not batch_reviews:
                    logger.warning(f"No reviews scraped in batch {batch_num}")
                    if not continuation_token:
                        logger.info("No more reviews available. Stopping.")
                        break
                    continue
                
                # Save to database
                saved_count = self.save_batch_to_database(batch_reviews, batch_num)
                
                # Save JSON backup
                self.save_batch_to_json(batch_reviews, batch_num)
                
                # Update state
                self.state['total_scraped'] += saved_count
                self.state['batch_history'].append({
                    'batch_num': batch_num,
                    'scraped': len(batch_reviews),
                    'saved': saved_count,
                    'timestamp': datetime.now().isoformat()
                })
                
                if batch_reviews:
                    self.state['last_review_id'] = batch_reviews[-1].get('reviewId')
                
                # Save state after each batch
                self._save_state()
                
                # Display progress
                logger.info(f"Progress: {batch_num + 1}/{total_batches} batches")
                logger.info(f"Total scraped so far: {self.state['total_scraped']:,}")
                
                # Delay between batches
                if batch_num < total_batches - 1:
                    logger.info("Waiting 10 seconds before next batch...")
                    time.sleep(10)
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                self._save_state()
                raise
        
        # Final statistics
        self.display_final_stats()
    
    def display_final_stats(self):
        """Display final scraping statistics"""
        logger.info("="*60)
        logger.info("BATCH SCRAPING COMPLETE")
        logger.info("="*60)
        
        # Get database stats
        self.cursor.execute("SELECT COUNT(*) FROM reviews")
        total_in_db = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT AVG(rating) FROM reviews WHERE rating > 0")
        avg_rating = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT rating, COUNT(*) as count
            FROM reviews
            WHERE rating > 0
            GROUP BY rating
            ORDER BY rating
        """)
        rating_dist = dict(self.cursor.fetchall())
        
        logger.info(f"Total reviews in database: {total_in_db:,}")
        logger.info(f"Reviews scraped in this session: {self.state['total_scraped']:,}")
        logger.info(f"Average rating: {avg_rating:.2f}")
        logger.info(f"Rating distribution: {rating_dist}")
        logger.info(f"Total batches processed: {len(self.state['batch_history'])}")
    
    def reset_state(self):
        """Reset scraping state (use with caution)"""
        self.state = {
            'current_batch': 0,
            'total_scraped': 0,
            'continuation_token': None,
            'batch_history': [],
            'last_review_id': None
        }
        self._save_state()
        logger.info("State reset to initial values")
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")


def main():
    """Main function to run batch scraping"""
    logger.info("="*60)
    logger.info("IRCTC BATCH REVIEW SCRAPER")
    logger.info("="*60)
    
    scraper = BatchReviewScraper()
    
    try:
        # Check current state
        logger.info(f"Current state: Batch {scraper.state['current_batch']}, "
                   f"Total scraped: {scraper.state['total_scraped']}")
        
        # Run batch scraping for 10 batches (100k reviews total)
        # Will continue from last saved state
        scraper.run_batch_scraping(total_batches=10)
        
    except KeyboardInterrupt:
        logger.info("\nScraping interrupted by user")
        scraper._save_state()
    except Exception as e:
        logger.error(f"Error during batch scraping: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        scraper.close()


if __name__ == "__main__":
    main()