#!/bin/bash

# IRCTC Dashboard - Streamlit Deployment Preparation Script
# This script prepares your project for Streamlit Cloud deployment

echo "🚀 Preparing IRCTC Dashboard for Streamlit Cloud Deployment"
echo "=========================================================="

# Create deployment directory
DEPLOY_DIR="streamlit-deploy"
echo "📁 Creating deployment directory: $DEPLOY_DIR"

# Remove if exists and create fresh
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy essential files
echo "📋 Copying essential files..."
cp streamlit_app.py $DEPLOY_DIR/
cp requirements_deploy.txt $DEPLOY_DIR/requirements.txt

# Create necessary directories
mkdir -p $DEPLOY_DIR/data/models
mkdir -p $DEPLOY_DIR/data/analysis
mkdir -p $DEPLOY_DIR/src/dashboard
mkdir -p $DEPLOY_DIR/.streamlit

# Copy data files (with size check)
echo "📊 Copying data files..."

# Check database size
DB_SIZE=$(du -m data/reviews.db | cut -f1)
if [ $DB_SIZE -gt 100 ]; then
    echo "⚠️  Warning: Database is ${DB_SIZE}MB (>100MB limit)"
    echo "   Creating sampled database..."
    python3 << EOF
import sqlite3
import pandas as pd

# Create smaller sample database
conn_orig = sqlite3.connect('data/reviews.db')
conn_new = sqlite3.connect('$DEPLOY_DIR/data/reviews.db')

# Copy schema
for table in ['reviews', 'processed_reviews', 'review_classifications']:
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10000", conn_orig)
        df.to_sql(table, conn_new, if_exists='replace', index=False)
        print(f"   ✓ Copied {len(df)} rows from {table}")
    except:
        pass

conn_orig.close()
conn_new.close()
EOF
else
    cp data/reviews.db $DEPLOY_DIR/data/
    echo "   ✓ Database copied (${DB_SIZE}MB)"
fi

# Copy model files
if [ -f "data/models/improved_topics.pkl" ]; then
    cp data/models/improved_topics.pkl $DEPLOY_DIR/data/models/
    echo "   ✓ Topic model copied"
fi

if [ -f "data/analysis/root_cause_analysis.pkl" ]; then
    cp data/analysis/root_cause_analysis.pkl $DEPLOY_DIR/data/analysis/
    echo "   ✓ Root cause analysis copied"
fi

# Copy dashboard modules
cp -r src/dashboard/*.py $DEPLOY_DIR/src/dashboard/ 2>/dev/null || true

# Create Streamlit config
echo "⚙️  Creating Streamlit configuration..."
cat > $DEPLOY_DIR/.streamlit/config.toml << 'EOL'
[theme]
primaryColor = "#3b82f6"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1e293b"
textColor = "#ffffff"
font = "sans serif"

[server]
maxUploadSize = 200
maxMessageSize = 200
headless = true
enableCORS = false
EOL

# Create README
echo "📝 Creating README..."
cat > $DEPLOY_DIR/README.md << 'EOL'
# IRCTC Review Analysis Dashboard

Real-time analytics dashboard for IRCTC app reviews.

## Features
- 📊 90,000+ reviews analyzed
- 🏢 Department-wise segregation
- 🔍 Root cause analysis
- 📈 Trend visualization

## Live Demo
Visit: [Your App URL]

## Technology Stack
- Streamlit
- Plotly
- SQLite
- Python

## Data Mining Project 2025
EOL

# Create simplified requirements
echo "📦 Creating requirements file..."
cat > $DEPLOY_DIR/requirements.txt << 'EOL'
streamlit==1.31.0
pandas==2.0.3
numpy==1.24.3
plotly==5.18.0
EOL

# Calculate total size
TOTAL_SIZE=$(du -sh $DEPLOY_DIR | cut -f1)
echo ""
echo "✅ Deployment preparation complete!"
echo "📊 Total size: $TOTAL_SIZE"

# Check if size is acceptable
SIZE_MB=$(du -m $DEPLOY_DIR | cut -f1)
if [ $SIZE_MB -gt 1000 ]; then
    echo "⚠️  Warning: Total size exceeds 1GB limit for free tier"
    echo "   Consider reducing data files or using sampling"
fi

echo ""
echo "📋 Next steps:"
echo "1. cd $DEPLOY_DIR"
echo "2. git init"
echo "3. git add ."
echo "4. git commit -m 'Initial deployment'"
echo "5. Create GitHub repo and push"
echo "6. Deploy on Streamlit Cloud"
echo ""
echo "🎉 Ready for deployment!"