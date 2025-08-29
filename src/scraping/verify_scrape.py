#!/usr/bin/env python3
"""
Verify the scraped reviews in the database
"""

import sqlite3
from pathlib import Path
import json
from datetime import datetime

def verify_scrape_results():
    """Verify that 100k reviews were scraped correctly"""
    
    db_path = Path('data/reviews.db')
    
    if not db_path.exists():
        print("❌ Database file does not exist!")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Check total count
    cursor.execute("SELECT COUNT(*) FROM reviews")
    total_count = cursor.fetchone()[0]
    print(f"📊 Total reviews in database: {total_count:,}")
    
    # 2. Check for duplicates
    cursor.execute("""
        SELECT COUNT(*) as dup_count 
        FROM (
            SELECT review_id, COUNT(*) as cnt 
            FROM reviews 
            GROUP BY review_id 
            HAVING cnt > 1
        )
    """)
    duplicate_count = cursor.fetchone()[0]
    print(f"🔍 Duplicate reviews found: {duplicate_count}")
    
    # 3. Check content quality
    cursor.execute("SELECT COUNT(*) FROM reviews WHERE content IS NULL OR content = ''")
    empty_content = cursor.fetchone()[0]
    print(f"📝 Reviews with empty content: {empty_content}")
    
    # 4. Rating distribution
    cursor.execute("""
        SELECT rating, COUNT(*) as count 
        FROM reviews 
        WHERE rating IS NOT NULL 
        GROUP BY rating 
        ORDER BY rating
    """)
    rating_dist = cursor.fetchall()
    print("\n⭐ Rating Distribution:")
    for rating, count in rating_dist:
        print(f"  {rating}★: {count:,} reviews ({count/total_count*100:.1f}%)")
    
    # 5. Average rating
    cursor.execute("SELECT AVG(rating) FROM reviews WHERE rating IS NOT NULL")
    avg_rating = cursor.fetchone()[0]
    print(f"\n📈 Average Rating: {avg_rating:.2f}")
    
    # 6. Date range
    cursor.execute("""
        SELECT MIN(date_posted), MAX(date_posted) 
        FROM reviews 
        WHERE date_posted IS NOT NULL
    """)
    date_range = cursor.fetchone()
    print(f"\n📅 Review Date Range:")
    print(f"  Oldest: {date_range[0]}")
    print(f"  Newest: {date_range[1]}")
    
    # 7. Sample reviews
    cursor.execute("""
        SELECT content, rating 
        FROM reviews 
        WHERE content IS NOT NULL 
        LIMIT 5
    """)
    samples = cursor.fetchall()
    print("\n📖 Sample Reviews:")
    for i, (content, rating) in enumerate(samples, 1):
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"  {i}. [{rating}★] {preview}")
    
    # 8. Language diversity check (basic)
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN content LIKE '%है%' OR content LIKE '%हैं%' THEN 1 ELSE 0 END) as hindi_count,
            SUM(CASE WHEN content LIKE '%the%' OR content LIKE '%and%' THEN 1 ELSE 0 END) as english_count
        FROM reviews
    """)
    lang_check = cursor.fetchone()
    print(f"\n🌐 Language Indicators:")
    print(f"  Reviews with Hindi characters: {lang_check[0]:,}")
    print(f"  Reviews with English words: {lang_check[1]:,}")
    
    # 9. Check JSON backup
    raw_dir = Path('data/raw')
    if raw_dir.exists():
        json_files = list(raw_dir.glob('reviews_*.json'))
        print(f"\n💾 JSON Backup Files: {len(json_files)} found")
        for json_file in json_files:
            file_size = json_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {json_file.name} ({file_size:.1f} MB)")
    
    conn.close()
    
    # Validation results
    print("\n" + "="*50)
    print("🎯 VALIDATION RESULTS:")
    
    if total_count >= 100000:
        print(f"✅ Successfully scraped {total_count:,} reviews (target: 100,000)")
        success = True
    elif total_count >= 90000:
        print(f"⚠️  Scraped {total_count:,} reviews (90% of target)")
        success = True
    else:
        print(f"❌ Only scraped {total_count:,} reviews (target: 100,000)")
        success = False
    
    if duplicate_count == 0:
        print("✅ No duplicate reviews found")
    else:
        print(f"⚠️  Found {duplicate_count} duplicate reviews")
    
    if empty_content == 0:
        print("✅ All reviews have content")
    else:
        print(f"⚠️  {empty_content} reviews have empty content")
    
    print("="*50)
    
    return success

if __name__ == "__main__":
    verify_scrape_results()