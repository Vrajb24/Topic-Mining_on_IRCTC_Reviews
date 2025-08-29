import sqlite3
import os
from pathlib import Path

def create_tables():
    db_path = Path("data/reviews.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path, 'r') as f:
        schema = f.read()
    
    cursor.executescript(schema)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created successfully at {db_path}")
    return db_path

def get_connection():
    db_path = Path("data/reviews.db")
    return sqlite3.connect(db_path)

if __name__ == "__main__":
    create_tables()