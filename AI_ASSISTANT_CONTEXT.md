# 🤖 AI Assistant Context File - IRCTC Review Analysis Project

**Purpose:** This file provides complete context for any AI assistant working on this project in the future.

---

## 🎯 CRITICAL INFORMATION

### User Details:
- **Name:** Vraj B
- **Email:** vrajb24@iitk.ac.in
- **Institution:** IIT Kanpur
- **GitHub Username:** Vrajb24

### Project Status:
- **Completion:** 100% - Fully deployed and operational
- **Last Updated:** August 28, 2025
- **Total Development Time:** ~4 days
- **Current State:** Live and serving users

---

## 🔗 REPOSITORY STRUCTURE (MUST UNDERSTAND)

The project has **TWO SEPARATE REPOSITORIES** for different purposes:

### 1. Portfolio Repository (For Resume/GitHub Profile)
```
Name: Topic-Mining_on_IRCTC_Reviews
URL: https://github.com/Vrajb24/Topic-Mining_on_IRCTC_Reviews
Purpose: Clean, professional code showcase
Branch: main
Status: Public, no external references
Note: This is what recruiters see
```

### 2. Deployment Repository (For Hosting)
```
Name: irctc-dashbaord (note the typo - keep as is!)
URL: https://github.com/Vrajb24/irctc-dashbaord
Purpose: Streamlit Cloud deployment
Branch: main
Live URL: https://irctc-analysis.streamlit.app
Note: Auto-deploys on push
```

**IMPORTANT:** Never mix these repositories. Updates to deployment should go to `irctc-dashbaord`, portfolio updates to `Topic-Mining_on_IRCTC_Reviews`.

---

## 📁 PROJECT STRUCTURE

```
DataMining/                          # Main project directory
├── src/                            # Source code
│   ├── scraping/                   # Web scraping modules
│   │   ├── batch_scraper.py       # Main scraper (10 threads)
│   │   └── irctc_scraper.py       # Core scraping logic
│   │
│   ├── modeling/                   # ML/NLP models
│   │   ├── improved_topic_analysis.py  # Enhanced topic modeling
│   │   └── advanced_models.py         # ML implementations
│   │
│   ├── analysis/                   # Analysis modules
│   │   ├── root_cause_analyzer.py # Root cause analysis
│   │   └── report_generator.py    # Report generation
│   │
│   └── dashboard/                  # Dashboard applications
│       ├── professional_app.py    # Port 8502 (main)
│       ├── segregated_dashboard.py # Port 8505
│       └── root_cause_dashboard.py # Port 8506
│
├── data/                           # Data directory
│   ├── reviews.db                 # SQLite database (34MB)
│   ├── models/                    # Trained models
│   └── analysis/                  # Analysis results
│
├── streamlit-deploy/               # Deployment folder
│   ├── streamlit_app.py          # Unified dashboard
│   └── requirements.txt          # Python 3.13 compatible
│
└── run_complete_pipeline.py       # Main automation script
```

---

## 🐛 COMMON ISSUES & FIXES

### 1. Database Column Names
**Issue:** `no such column: created_at`
**Fix:** Use `date_posted` instead of `created_at`
```python
# Wrong
SELECT * FROM reviews WHERE created_at > '2024-01-01'

# Correct
SELECT * FROM reviews WHERE date_posted > '2024-01-01'
```

### 2. Fillcolor Transparency Error
**Issue:** `Invalid value '#3e7bfa20' for fillcolor`
**Fix:** Convert hex to rgba
```python
# Wrong
fillcolor=f"{color}20"

# Correct
def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'

fillcolor=hex_to_rgba(color, 0.2)
```

### 3. Python 3.13 Compatibility
**Issue:** `No module named 'distutils'`
**Fix:** Update requirements
```txt
# Wrong
numpy==1.24.3

# Correct
numpy>=1.26.0
```

### 4. Topic Model Loading
**Issue:** Topics not loading
**Fix:** Use correct path
```python
# Correct path
with open('data/models/improved_topics.pkl', 'rb') as f:
    topics = pickle.load(f)
```

---

## 📊 KEY STATISTICS

### Data:
- Total Reviews: 90,447
- Database Size: 34MB
- Languages: 12
- Date Range: 2023-2025

### Classification Results:
- App Issues: 36.7% (33,066 reviews)
- Railway Issues: 9.1% (8,191 reviews)
- Unclear: 45.9% (41,384 reviews)
- Mixed: 8.4% (7,577 reviews)

### Performance:
- Scraping Rate: 180 reviews/minute
- Analysis Time: ~5 minutes for full pipeline
- Dashboard Load: <3 seconds
- Memory Usage: ~200MB

---

## 🚀 DEPLOYMENT COMMANDS

### Local Development:
```bash
# Run main dashboard
streamlit run src/dashboard/professional_app.py --server.port 8502

# Run complete pipeline
python run_complete_pipeline.py

# Analysis only (no scraping)
python run_complete_pipeline.py --analysis-only

# Update models only
python run_full_analysis.py
```

### Update Live Dashboard:
```bash
cd streamlit-deploy/
git add .
git commit -m "Update dashboard"
git push origin main
# Wait 2-3 minutes for auto-deploy
```

### Update Portfolio:
```bash
cd /mnt/c/Users/rocke/Desktop/Projects/DataMining
git add .
git commit -m "Update"
git push origin main
```

---

## 🎨 UI/UX SPECIFICATIONS

### Color Scheme (Dark Theme):
```css
Background Main: #0e1117
Background Secondary: #1a1d24
Card Background: #1e293b
Text Primary: #ffffff
Text Secondary: #b8bcc8
Accent Blue: #3b82f6
Success Green: #10b981
Warning Orange: #f59e0b
Danger Red: #ef4444
Border: #334155
```

### Dashboard Ports:
- Professional: 8502
- Segregated: 8505
- Root Cause: 8506

---

## 📝 IMPORTANT NOTES FOR FUTURE DEVELOPMENT

1. **Never delete streamlit-deploy/.git** - It's needed for deployment updates
2. **Keep repositories separate** - Portfolio and deployment serve different purposes
3. **Test locally first** - Before pushing to deployment repo
4. **Database is read-only in deployment** - Updates need local pipeline run
5. **Python 3.13 on Streamlit Cloud** - Ensure compatibility
6. **1GB memory limit on free tier** - Keep models optimized
7. **Public repos only for free hosting** - Don't add sensitive data

---

## 🔄 AUTOMATION PIPELINE

The complete pipeline (`run_complete_pipeline.py`) performs:
1. Scrape new reviews (optional)
2. Update topic models
3. Run root cause analysis
4. Generate reports
5. Restart dashboards

Options:
- `--max-reviews 1000` - Limit new reviews
- `--skip-scraping` - Use existing data
- `--analysis-only` - Skip scraping and modeling

---

## 📈 ROOT CAUSE ANALYSIS METHODS

1. **Temporal Patterns** - Time series analysis
2. **Statistical Analysis** - T-tests, correlations
3. **Five Whys** - Causal chain extraction
4. **Severity Classification** - Critical/High/Medium/Low
5. **ML Clustering** - K-means (10 clusters)
6. **Anomaly Detection** - Z-score based
7. **Pattern Detection** - Regex and NLP

---

## 🎯 KEY FINDINGS

### Top 5 Root Causes:
1. Infrastructure Scaling (356 cases)
2. Payment Gateway Issues (234 cases)
3. Login/Authentication (189 cases)
4. Session Timeouts (167 cases)
5. Server Errors (145 cases)

### Critical Insights:
- Peak issue times: 6-8 AM, 8-10 PM
- Average rating: 1.82/5 (concerning)
- 67% negative reviews (1-2 stars)
- Payment failures affect 23% of users

---

## 📞 CONTACT & SUPPORT

For any questions about this project:
- **Developer:** Vraj B
- **Email:** vrajb24@iitk.ac.in
- **Institution:** IIT Kanpur
- **Project Status:** Completed August 2025

---

## ⚠️ WARNINGS

1. **DO NOT** expose database credentials
2. **DO NOT** commit large files (>100MB)
3. **DO NOT** push to wrong repository
4. **DO NOT** delete production data
5. **DO NOT** modify git history on deployed repos

---

*This context file should be preserved for any future AI assistant or developer working on this project. It contains all critical information needed to understand and continue development.*