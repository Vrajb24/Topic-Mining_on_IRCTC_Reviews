#!/bin/bash

echo "🚀 IRCTC Review Analysis - Development Environment Setup"
echo "========================================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}📌 Checking Python version...${NC}"
python3 --version

# Create project structure
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p data/{raw,processed,models,cache}
mkdir -p src/{scraping,preprocessing,modeling,analysis,dashboard,database,api,utils}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p notebooks
mkdir -p logs
mkdir -p config
mkdir -p static/{css,js,images}
mkdir -p templates

# Create virtual environment
echo -e "${BLUE}🐍 Setting up Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}📦 Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Create __init__.py files
echo -e "${BLUE}📄 Creating Python package files...${NC}"
touch src/__init__.py
for dir in scraping preprocessing modeling analysis dashboard database api utils; do
    touch src/$dir/__init__.py
done

# Create configuration file for Chrome in WSL
echo -e "${BLUE}⚙️ Creating Chrome configuration for WSL...${NC}"
cat > config/chrome_config.json << 'EOF'
{
  "chrome_options": {
    "binary_location": "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "arguments": [
      "--headless",
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--window-size=1920,1080",
      "--disable-blink-features=AutomationControlled",
      "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ]
  },
  "driver_options": {
    "use_chromium": false,
    "auto_download": true,
    "version": "latest"
  }
}
EOF

# Create initial database schema
echo -e "${BLUE}🗄️ Creating database schema file...${NC}"
cat > src/database/schema.sql << 'EOF'
-- Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT UNIQUE,
    content TEXT NOT NULL,
    rating INTEGER,
    language TEXT,
    date_posted TIMESTAMP,
    date_scraped TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);

-- Topics table
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_name TEXT NOT NULL,
    topic_words TEXT,
    topic_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Review-Topic mapping
CREATE TABLE IF NOT EXISTS review_topics (
    review_id INTEGER,
    topic_id INTEGER,
    confidence REAL,
    FOREIGN KEY (review_id) REFERENCES reviews (id),
    FOREIGN KEY (topic_id) REFERENCES topics (id)
);

-- Sentiment analysis results
CREATE TABLE IF NOT EXISTS sentiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER,
    sentiment_score REAL,
    sentiment_label TEXT,
    confidence REAL,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (review_id) REFERENCES reviews (id)
);

-- Analytics cache
CREATE TABLE IF NOT EXISTS analytics_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT,
    metric_value TEXT,
    date_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expiry TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_reviews_date ON reviews(date_posted);
CREATE INDEX idx_reviews_processed ON reviews(processed);
CREATE INDEX idx_sentiments_review ON sentiments(review_id);
CREATE INDEX idx_review_topics_review ON review_topics(review_id);
EOF

# Create .gitignore
echo -e "${BLUE}📝 Creating .gitignore...${NC}"
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data (keep structure, ignore contents)
data/raw/*
data/processed/*
data/models/*
data/cache/*
!data/*/.gitkeep

# Database
*.db
*.sqlite
*.sqlite3

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
logs/*.log
*.log

# Cache
cache/
*.cache
.cache/

# OS
.DS_Store
Thumbs.db
desktop.ini

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/

# Distribution
dist/
build/
*.egg-info/
*.egg

# Chrome driver
chromedriver*
*.crdownload

# Model files (large)
*.pkl
*.h5
*.pt
*.pth
*.onnx
*.safetensors

# Temporary
tmp/
temp/
*.tmp
EOF

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/models/.gitkeep
touch data/cache/.gitkeep
touch logs/.gitkeep

# Copy .env.example to .env
echo -e "${BLUE}🔐 Creating environment file...${NC}"
cp .env.example .env

# Create lightweight requirements file
echo -e "${BLUE}📋 Creating lightweight requirements.txt...${NC}"
cat > requirements_light.txt << 'EOF'
# Core dependencies (lightweight version)
# Using SQLite instead of PostgreSQL
# Using free translation services

# Web Scraping
selenium>=4.15.0
beautifulsoup4>=4.12.0
requests>=2.31.0
webdriver-manager>=4.0.0
undetected-chromedriver>=3.5.0

# NLP & Machine Learning (CPU-optimized)
torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu
transformers>=4.35.0
sentence-transformers>=2.2.2
bertopic>=0.16.0
scikit-learn>=1.3.0
hdbscan>=0.8.33
umap-learn>=0.5.4
nltk>=3.8.1
spacy>=3.7.0

# Free translation (instead of Google Cloud API)
googletrans==3.1.0a0
langdetect>=1.0.9
deep-translator>=1.11.4

# Data Processing
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0

# Database (SQLite - built into Python)
sqlalchemy>=2.0.0

# Visualization & Dashboard
streamlit>=1.28.0
plotly>=5.17.0
altair>=5.1.0
wordcloud>=1.9.2
matplotlib>=3.8.0

# Task Scheduling
schedule>=1.2.0
apscheduler>=3.10.0

# Development & Testing
pytest>=7.4.0
python-dotenv>=1.0.0
tqdm>=4.66.0

# Logging
loguru>=0.7.2

# Performance
joblib>=1.3.0
cachetools>=5.3.0
EOF

# Create a simple test script
echo -e "${BLUE}🧪 Creating test script...${NC}"
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify environment setup"""

import sys
print("🔍 Testing Environment Setup...")
print("=" * 50)

# Test Python version
print(f"✅ Python version: {sys.version}")

# Test imports
failed = []
succeeded = []

packages = [
    ("selenium", "Web scraping"),
    ("transformers", "NLP models"),
    ("streamlit", "Dashboard"),
    ("sqlalchemy", "Database"),
    ("googletrans", "Translation"),
]

for package, description in packages:
    try:
        __import__(package)
        succeeded.append(f"✅ {description} ({package})")
    except ImportError:
        failed.append(f"❌ {description} ({package})")

print("\nPackage Status:")
for msg in succeeded:
    print(msg)
for msg in failed:
    print(msg)

# Test Chrome access from WSL
import os
chrome_path = "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
if os.path.exists(chrome_path):
    print(f"\n✅ Chrome found at: {chrome_path}")
else:
    print(f"\n❌ Chrome not found at expected location")

print("\n" + "=" * 50)
if not failed:
    print("🎉 All tests passed! Environment is ready.")
else:
    print(f"⚠️  {len(failed)} packages need to be installed.")
    print("Run: pip install -r requirements_light.txt")
EOF

chmod +x test_setup.py

echo -e "${GREEN}✅ Setup script created successfully!${NC}"
echo ""
echo -e "${YELLOW}📌 Next steps:${NC}"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Install lightweight dependencies:"
echo "   pip install -r requirements_light.txt"
echo ""
echo "3. Test the setup:"
echo "   python test_setup.py"
echo ""
echo "4. Initialize database:"
echo "   python -c \"from src.database import init_db; init_db.create_tables()\""
echo ""
echo -e "${GREEN}Ready to start development!${NC}"