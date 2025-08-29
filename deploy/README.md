# IRCTC Review Analysis - Deployment Guide

## 🚀 Automated Pipeline

### Complete Pipeline
Run the complete automation pipeline that scrapes, analyzes, and updates dashboards:

```bash
# Full pipeline (scrape + analyze + update dashboards)
python run_complete_pipeline.py

# Scrape specific number of reviews
python run_complete_pipeline.py --max-reviews 2000

# Skip scraping, only update models and analysis
python run_complete_pipeline.py --skip-scraping

# Analysis only (no scraping or model training)
python run_complete_pipeline.py --analysis-only
```

### Pipeline Steps
1. **Scrape New Reviews** - Collects latest reviews from IRCTC app
2. **Update Topic Models** - Retrains models with new data
3. **Root Cause Analysis** - Identifies patterns and issues
4. **Generate Reports** - Creates executive and technical reports
5. **Restart Dashboards** - Updates all dashboards with latest data

## 🌐 Hosting Options & Recommendations

### 1. **Streamlit Community Cloud** (Recommended - FREE)
- **Pros**: Free, easy deployment, automatic HTTPS
- **Cons**: Limited to 1GB RAM, public repos only
- **What to upload**: Dashboard + pre-computed models (not raw data)

```
Upload Structure:
├── streamlit_app.py (main app)
├── src/dashboard/ (dashboard files)
├── data/models/ (pre-computed models only - <100MB)
├── requirements_deploy.txt
└── README.md
```

### 2. **Heroku** (Good for small scale)
- **Pros**: Easy deployment, supports databases
- **Cons**: Expensive for large memory needs, limited free tier
- **Cost**: ~$7-25/month for hobby to professional dynos

### 3. **Railway** (Modern alternative)
- **Pros**: Simple deployment, good free tier, auto-scaling
- **Cons**: Newer platform, limited resources on free tier
- **Cost**: Pay-as-you-go starting at $5/month

### 4. **DigitalOcean App Platform** (Best for production)
- **Pros**: Scalable, good performance, reasonable pricing
- **Cons**: No free tier, requires some setup
- **Cost**: $5-12/month for basic apps

### 5. **AWS/GCP/Azure** (Enterprise)
- **Pros**: Unlimited scalability, full control
- **Cons**: Complex setup, expensive
- **Cost**: $10-100+/month depending on usage

## 📦 What to Deploy

### Option A: Dashboard Only (Recommended for hosting)
**Size**: ~50-100MB
**Components**:
- Dashboard files (`src/dashboard/`)
- Pre-computed models (`data/models/*.pkl`)
- Small sample database for demo
- `requirements_deploy.txt`

### Option B: Complete System (For private servers)
**Size**: ~2-5GB
**Components**:
- Full codebase including scraping
- Complete database (90,000+ reviews)
- All models and analysis files
- Full requirements

## 🔧 Deployment Steps for Streamlit Cloud

### Step 1: Prepare Repository
```bash
# Create deployment branch
git checkout -b deploy

# Copy essential files to deploy/
cp -r src/dashboard deploy/
cp -r data/models deploy/data/
cp requirements_deploy.txt deploy/requirements.txt

# Create streamlit config
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > deploy/.streamlit/config.toml
```

### Step 2: Deploy to Streamlit Cloud
1. Push deploy branch to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `deploy/streamlit_app.py` as main file
5. Deploy!

### Step 3: Automated Updates
Set up GitHub Actions for automated deployment:

```yaml
# .github/workflows/deploy.yml
name: Update Dashboard
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Pipeline
        run: python run_complete_pipeline.py --analysis-only
      - name: Commit Updates
        run: |
          git add data/models/ data/reports/
          git commit -m "Automated dashboard update"
          git push
```

## 💾 Data Management for Hosting

### Size Optimization
- **Models**: Compress pickle files (~20-50MB each)
- **Database**: Use SQLite with sample data (~10-50MB)
- **Reports**: Keep only latest reports (~1-5MB)

### Update Strategy
- **Local**: Run complete pipeline with scraping
- **Hosted**: Use pre-computed models, update periodically
- **Hybrid**: Scrape locally, deploy models to cloud

## 🔄 Scheduled Automation

### Cron Job (Linux/Mac)
```bash
# Edit crontab
crontab -e

# Add daily pipeline run at 2 AM
0 2 * * * cd /path/to/project && python run_complete_pipeline.py

# Add weekly full analysis
0 2 * * 0 cd /path/to/project && python run_complete_pipeline.py --max-reviews 5000
```

### Windows Task Scheduler
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (Daily/Weekly)
4. Action: Start Program
5. Program: `python`
6. Arguments: `/path/to/run_complete_pipeline.py`

## 📊 Monitoring & Alerts

### Log Monitoring
```bash
# Monitor pipeline logs
tail -f logs/pipeline.log

# Check dashboard status
curl http://localhost:8502/healthz
```

### Performance Metrics
- **Scraping**: Reviews/minute, success rate
- **Analysis**: Processing time, model accuracy
- **Dashboard**: Response time, user engagement

## 🚀 Recommended Deployment

**For Demo/Portfolio**: Streamlit Community Cloud
**For Production**: DigitalOcean + scheduled local pipeline
**For Enterprise**: AWS/GCP with full automation

This setup gives you both automated data processing and scalable hosting!