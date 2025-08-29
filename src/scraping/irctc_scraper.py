#!/usr/bin/env python3
"""
Google Play Store Review Scraper for IRCTC App
This module scrapes reviews from the Google Play Store for the IRCTC app.
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging
from google_play_scraper import app, reviews, Sort
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IRCTCReviewScraper:
    """Scraper for IRCTC app reviews from Google Play Store"""
    
    IRCTC_APP_ID = 'cris.org.in.prs.ima'  # IRCTC Rail Connect app ID
    
    def __init__(self, db_path: str = 'data/reviews.db'):
        """Initialize the scraper with database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._ensure_logs_dir()
        
    def _ensure_logs_dir(self):
        """Ensure logs directory exists"""
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
    
    def get_app_info(self) -> Dict:
        """Get IRCTC app information from Play Store"""
        try:
            app_info = app(self.IRCTC_APP_ID)
            logger.info(f"App: {app_info['title']}")
            logger.info(f"Developer: {app_info['developer']}")
            logger.info(f"Current rating: {app_info['score']}")
            logger.info(f"Total reviews: {app_info['reviews']}")
            logger.info(f"Installs: {app_info['installs']}")
            return app_info
        except Exception as e:
            logger.error(f"Error fetching app info: {e}")
            return {}
    
    def scrape_reviews(self, count: int = 100000, batch_size: int = 200) -> List[Dict]:
        """
        Scrape reviews from Google Play Store
        
        Args:
            count: Total number of reviews to scrape
            batch_size: Number of reviews to fetch in each batch
            
        Returns:
            List of review dictionaries
        """
        all_reviews = []
        continuation_token = None
        
        logger.info(f"Starting to scrape {count} reviews for IRCTC app...")
        logger.info(f"App ID: {self.IRCTC_APP_ID}")
        
        # Progress bar
        pbar = tqdm(total=count, desc="Scraping reviews")
        
        while len(all_reviews) < count:
            try:
                # Fetch batch of reviews
                batch_count = min(batch_size, count - len(all_reviews))
                
                result, continuation_token = reviews(
                    self.IRCTC_APP_ID,
                    lang='en',  # Will get reviews in all languages
                    country='in',  # India
                    sort=Sort.NEWEST,  # Get newest reviews first
                    count=batch_count,
                    continuation_token=continuation_token
                )
                
                if not result:
                    logger.warning("No more reviews available")
                    break
                
                all_reviews.extend(result)
                pbar.update(len(result))
                
                # Log progress
                if len(all_reviews) % 1000 == 0:
                    logger.info(f"Scraped {len(all_reviews)} reviews so far...")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
                # Break if no continuation token (no more reviews)
                if not continuation_token:
                    logger.info("No more reviews available to scrape")
                    break
                    
            except Exception as e:
                logger.error(f"Error scraping reviews: {e}")
                logger.info(f"Retrying after delay...")
                time.sleep(5)  # Wait before retrying
                continue
        
        pbar.close()
        logger.info(f"Total reviews scraped: {len(all_reviews)}")
        return all_reviews
    
    def save_to_database(self, reviews_data: List[Dict]) -> int:
        """
        Save scraped reviews to SQLite database
        
        Args:
            reviews_data: List of review dictionaries
            
        Returns:
            Number of reviews saved
        """
        saved_count = 0
        duplicate_count = 0
        
        for review in tqdm(reviews_data, desc="Saving to database"):
            try:
                # Prepare data for insertion
                review_id = review.get('reviewId', '')
                content = review.get('content', '')
                rating = review.get('score', 0)
                
                # Parse the date
                review_date = review.get('at')
                if review_date:
                    date_posted = review_date.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_posted = None
                
                # Additional metadata
                thumbs_up = review.get('thumbsUpCount', 0)
                user_name = review.get('userName', '')
                reply_content = review.get('replyContent', '')
                
                # Check if review already exists
                self.cursor.execute(
                    "SELECT id FROM reviews WHERE review_id = ?",
                    (review_id,)
                )
                
                if self.cursor.fetchone():
                    duplicate_count += 1
                    continue
                
                # Insert into database
                self.cursor.execute("""
                    INSERT INTO reviews (
                        review_id, content, rating, language, 
                        date_posted, date_scraped, processed,
                        thumbs_up, user_name, reply_content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    review_id, content, rating, 'auto',
                    date_posted, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    False, thumbs_up, user_name, reply_content
                ))
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving review: {e}")
                continue
        
        # Commit all changes
        self.conn.commit()
        
        logger.info(f"Saved {saved_count} new reviews to database")
        logger.info(f"Skipped {duplicate_count} duplicate reviews")
        
        return saved_count
    
    def save_to_json(self, reviews_data: List[Dict], filename: str = None):
        """Save reviews to JSON file for backup"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/raw/reviews_{timestamp}.json'
        
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a copy to avoid modifying original data
        json_data = []
        for review in reviews_data:
            review_copy = review.copy()
            # Convert datetime objects to strings
            for key, value in review_copy.items():
                if hasattr(value, 'strftime'):
                    review_copy[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            json_data.append(review_copy)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved reviews to {filepath}")
    
    def get_review_stats(self) -> Dict:
        """Get statistics about scraped reviews"""
        stats = {}
        
        # Total reviews
        self.cursor.execute("SELECT COUNT(*) FROM reviews")
        stats['total_reviews'] = self.cursor.fetchone()[0]
        
        # Average rating
        self.cursor.execute("SELECT AVG(rating) FROM reviews WHERE rating > 0")
        avg_rating = self.cursor.fetchone()[0]
        stats['average_rating'] = round(avg_rating, 2) if avg_rating else 0
        
        # Rating distribution
        self.cursor.execute("""
            SELECT rating, COUNT(*) as count 
            FROM reviews 
            WHERE rating > 0 
            GROUP BY rating 
            ORDER BY rating
        """)
        stats['rating_distribution'] = dict(self.cursor.fetchall())
        
        # Processed vs unprocessed
        self.cursor.execute("SELECT COUNT(*) FROM reviews WHERE processed = 1")
        stats['processed_reviews'] = self.cursor.fetchone()[0]
        stats['unprocessed_reviews'] = stats['total_reviews'] - stats['processed_reviews']
        
        # Date range
        self.cursor.execute("""
            SELECT MIN(date_posted), MAX(date_posted) 
            FROM reviews 
            WHERE date_posted IS NOT NULL
        """)
        date_range = self.cursor.fetchone()
        stats['oldest_review'] = date_range[0]
        stats['newest_review'] = date_range[1]
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")


def main():
    """Main function to run the scraper"""
    logger.info("="*50)
    logger.info("IRCTC Review Scraper Started")
    logger.info("="*50)
    
    # Create scraper instance
    scraper = IRCTCReviewScraper()
    
    try:
        # Get app info
        app_info = scraper.get_app_info()
        
        if not app_info:
            logger.error("Could not fetch app information. Exiting.")
            return
        
        # Scrape reviews
        target_count = 10000  # More reasonable target for testing
        logger.info(f"Target: Scraping {target_count:,} reviews...")
        
        reviews_data = scraper.scrape_reviews(count=target_count, batch_size=200)
        
        if reviews_data:
            # Save to database
            saved_count = scraper.save_to_database(reviews_data)
            
            # Save JSON backup
            scraper.save_to_json(reviews_data)
            
            # Get and display statistics
            stats = scraper.get_review_stats()
            
            logger.info("="*50)
            logger.info("Scraping Complete! Statistics:")
            logger.info(f"Total reviews in database: {stats['total_reviews']:,}")
            logger.info(f"Average rating: {stats['average_rating']}")
            logger.info(f"Rating distribution: {stats['rating_distribution']}")
            logger.info(f"Date range: {stats['oldest_review']} to {stats['newest_review']}")
            logger.info("="*50)
            
            return stats
        else:
            logger.error("No reviews were scraped")
            return None
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
        
    finally:
        scraper.close()


if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n✅ Successfully scraped and saved reviews!")
        print(f"Total reviews in database: {result['total_reviews']:,}")