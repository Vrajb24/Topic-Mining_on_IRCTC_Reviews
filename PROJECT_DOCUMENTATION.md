# IRCTC Review Analysis System - Project Documentation

## Project Overview
**Objective**: Extract and analyze key topics from IRCTC app reviews to provide actionable insights for the IRCTC team, enabling data-driven improvements to their mobile application.

**Target Audience**: IRCTC Development and Management Team

**Current Status**: ✅ FULLY DEPLOYED - Dashboard operational with 90,447 reviews analyzed

### 🔗 Repository Information
- **Portfolio Repository**: https://github.com/Vrajb24/Topic-Mining_on_IRCTC_Reviews (Clean for resume)
- **Deployment Repository**: https://github.com/Vrajb24/irctc-dashbaord (Streamlit hosting)
- **Live Dashboard**: https://irctc-analysis.streamlit.app
- **Author**: Vraj B (vrajb24@iitk.ac.in)
- **Institution**: IIT Kanpur

## Implementation Summary

### Completed Components
- ✅ **Database**: SQLite with 90,000 IRCTC reviews
- ✅ **Data Collection**: Automated scraping pipeline implemented
- ✅ **Preprocessing**: Text normalization and language detection
- ✅ **Topic Modeling**: LDA analysis with 30 topics identified
- ✅ **Sentiment Analysis**: Rating-based sentiment classification
- ✅ **Dashboard**: Tesla-inspired analytics interface with real data
- ✅ **Search & Filter**: Full-text search and multi-criteria filtering
- ✅ **Visualizations**: Interactive charts with Plotly

### Key Achievements
- Successfully collected and processed 90,000 reviews
- Identified major pain points: payment issues, tatkal booking, server problems
- Built professional dashboard with dark/light mode
- Real-time analytics with sub-second response times

## Core Features

### 1. Data Collection & Processing
- **Initial Dataset**: 100,000 latest reviews from Google Play Store
- **Ongoing Collection**: Scheduled scraping for model retraining
- **Language Support**: Multilingual (Hindi, Hinglish, English, regional languages)

### 2. Analysis Capabilities
- **Topic Mining**: Identify 50-100 unique topics from reviews
- **Sentiment Analysis**: Review-level and topic-level sentiment
- **Root Cause Analysis**: Extract specific reasons for user dissatisfaction
- **Temporal Analysis**: Track topic and sentiment trends over time
- **Live Analysis**: Real-time sentiment updates for new reviews

### 3. Interactive Dashboard
- **Web-based Interface**: Accessible via internet
- **Live Review Input**: Test sentiment changes with custom reviews
- **Visualization**: Interactive charts, word clouds, topic hierarchies
- **Automated Reports**: Weekly/monthly summaries
- **Alert System**: Spike detection for critical issues

## Technical Architecture

### Phase 1: Environment Setup
```
project/
├── data/
│   ├── raw/           # Scraped reviews
│   ├── processed/     # Cleaned data
│   └── models/        # Trained models
├── src/
│   ├── scraping/      # Data collection
│   ├── preprocessing/ # Text cleaning
│   ├── modeling/      # Topic & sentiment models
│   ├── analysis/      # Analytics engine
│   └── dashboard/     # Web interface
├── notebooks/         # Development notebooks
├── tests/            # Unit tests
└── deployment/       # Deployment configs
```

### Phase 2: Technology Stack

#### Core Technologies
- **Python 3.10+**: Main programming language
- **PostgreSQL/SQLite**: Database for reviews and results
- **Redis**: Cache for live predictions
- **Docker**: Containerization

#### Data Collection
- **Selenium**: Web scraping
- **BeautifulSoup4**: HTML parsing
- **Schedule/Celery**: Task scheduling
- **Requests**: API calls

#### NLP & Machine Learning
- **Transformers**: BERT, mBERT, XLM-RoBERTa
- **Sentence-Transformers**: Multilingual sentence embeddings
- **BERTopic**: Topic modeling framework
- **HDBSCAN**: Clustering algorithm
- **UMAP**: Dimensionality reduction
- **Scikit-learn**: ML utilities
- **NLTK/spaCy**: Text preprocessing
- **Google Translate API**: Language normalization

#### Dashboard & Visualization
- **Streamlit**: Main dashboard framework
- **Plotly**: Interactive visualizations
- **Altair**: Statistical graphics
- **WordCloud**: Topic visualization

#### Deployment
- **Streamlit Cloud**: Free hosting (initial)
- **GitHub Actions**: CI/CD
- **Docker**: Container deployment
- **Heroku/Railway**: Alternative free hosting

### Phase 3: Model Pipeline

#### 1. Multilingual Embeddings Comparison
```python
models_to_compare = [
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/distiluse-base-multilingual-cased-v2', 
    'bert-base-multilingual-cased',
    'xlm-roberta-base',
    'google/muril-base-cased',  # Indian languages
    'ai4bharat/indic-bert',     # Indian languages
    # Original models for comparison
    'all-MiniLM-L6-v2',
    'all-mpnet-base-v2'
]
```

#### 2. Clustering Approaches
- HDBSCAN (primary)
- K-Means (comparison)
- DBSCAN (comparison)
- Agglomerative Clustering (hierarchy)

#### 3. Topic Modeling
- BERTopic with Class-based TF-IDF
- LDA (baseline comparison)
- Dynamic topic modeling for temporal analysis

#### 4. Sentiment Analysis
- Fine-tuned mBERT for Indian context
- Aspect-based sentiment extraction
- Confidence scoring

### Phase 4: Implementation Roadmap

#### Week 1-2: Foundation ✅
- [x] Environment setup
- [x] Database schema design
- [x] Basic scraping pipeline
- [x] Data preprocessing module

#### Week 3-4: Model Development ✅
- [x] Embedding comparison framework
- [x] Topic modeling pipeline (LDA implemented)
- [x] Sentiment analysis integration
- [x] Model evaluation metrics

#### Week 5-6: Dashboard Development ✅
- [x] Basic Streamlit interface
- [x] Live review input feature
- [x] Interactive visualizations
- [x] Database integration

#### Week 7-8: Advanced Features 🟡
- [ ] Scheduled retraining
- [ ] Automated reporting
- [ ] Alert system
- [x] Root cause analysis (partial)

#### Week 9-10: Deployment & Testing 🟡
- [ ] Docker containerization
- [ ] Cloud deployment
- [x] Performance optimization
- [x] Documentation & testing (ongoing)

## Environment Setup Instructions

### Prerequisites
```bash
# System requirements
- Python 3.10+
- Git
- Chrome/Chromium (for Selenium)
- 8GB+ RAM recommended
- 10GB+ free disk space
```

### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/irctc-review-analysis.git
cd irctc-review-analysis
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader all
```

4. **Setup Database**
```bash
# PostgreSQL (production)
createdb irctc_reviews

# Or SQLite (development)
python src/database/init_db.py
```

5. **Configure Environment Variables**
```bash
cp .env.example .env
# Edit .env with your configurations
```

6. **Download Chrome Driver**
```bash
python src/scraping/setup_driver.py
```

## Data Collection Strategy

### Initial Scraping (100k reviews)
```python
# Run initial data collection
python src/scraping/initial_scrape.py --limit 100000
```

### Scheduled Scraping
```python
# Setup cron job or use Python scheduler
python src/scraping/scheduled_scrape.py --interval daily
```

### Live Review Integration
- API endpoint for real-time review submission
- WebSocket for live sentiment updates
- Database triggers for model updates

## Model Training & Evaluation

### Training Pipeline
```python
# Preprocess data
python src/preprocessing/clean_data.py

# Compare embeddings
python src/modeling/compare_embeddings.py

# Train topic model
python src/modeling/train_topic_model.py

# Train sentiment model
python src/modeling/train_sentiment.py
```

### Evaluation Metrics
- **Topic Coherence**: C_v score
- **Clustering**: Silhouette score, Davies-Bouldin index
- **Sentiment**: Accuracy, F1-score, confusion matrix
- **Human Evaluation**: Manual topic relevance scoring

## Dashboard Features (Implemented)

### Main Components
1. **Overview Page** ✅
   - Total reviews analyzed: 90,000
   - Overall sentiment distribution (Positive: 35.2%, Negative: 60.3%)
   - Rating distribution visualization
   - Review trends over time

2. **Topic Analysis** ✅
   - 30 topics identified via LDA
   - Interactive word cloud
   - Topic distribution charts
   - Keywords for each topic (payment, booking, tatkal, etc.)

3. **Sentiment Dashboard** ✅
   - Real-time sentiment metrics
   - Sentiment by rating correlation
   - Sample reviews by sentiment
   - Color-coded sentiment indicators

4. **Review Explorer** ✅
   - Full-text search functionality
   - Rating filters (1-5 stars)
   - Sort by date/rating
   - Individual review cards with metadata

5. **Statistics Page** ✅
   - Language distribution (4 languages detected)
   - Rating trends over time
   - Database statistics
   - Performance metrics

## Deployment Instructions

### Local Development
```bash
streamlit run src/dashboard/app.py
```

### Docker Deployment
```bash
docker build -t irctc-analysis .
docker run -p 8501:8501 irctc-analysis
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect repo to Streamlit Cloud
3. Configure secrets in dashboard
4. Deploy application

### Alternative Hosting
- **Heroku**: Use heroku.yml configuration
- **Railway**: One-click deploy from GitHub
- **Render**: Free tier with Docker support

## API Endpoints

```python
POST /api/analyze
- Input: {"review": "text"}
- Output: {"sentiment": 0.8, "topics": [...]}

GET /api/topics
- Output: List of all topics with metadata

GET /api/trends
- Parameters: start_date, end_date
- Output: Temporal analysis data

POST /api/reviews/add
- Input: Review object
- Output: Analysis results
```

## Monitoring & Maintenance

### Performance Metrics
- Response time < 500ms for predictions
- Dashboard load time < 3s
- Model retraining time < 1 hour
- Scraping reliability > 95%

### Logging
- Application logs: `/logs/app.log`
- Scraping logs: `/logs/scraper.log`
- Model training logs: `/logs/training.log`

### Backup Strategy
- Daily database backups
- Model versioning in Git LFS
- Review data archival monthly

## Testing Strategy

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
locust -f tests/load/locustfile.py
```

## Future Enhancements

### Phase 2 Features
- iOS App Store integration
- Competitor analysis
- Predictive modeling
- User segmentation
- Multi-language UI

### Scalability Improvements
- Distributed computing with Dask
- GPU acceleration for embeddings
- Microservices architecture
- Kubernetes orchestration

## Troubleshooting Guide

### Common Issues

1. **Scraping Errors**
   - Check Chrome driver version
   - Verify internet connection
   - Review rate limiting

2. **Model Performance**
   - Increase embedding dimensions
   - Adjust clustering parameters
   - Add more training data

3. **Dashboard Issues**
   - Clear browser cache
   - Check port availability
   - Verify database connection

## Contributing Guidelines

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Write/update tests
5. Submit pull request

## License
MIT License - See LICENSE file

## Contact
For questions or support, please open an issue on GitHub.

## Current Architecture

### Technology Stack in Use
- **Backend**: Python 3.12
- **Database**: SQLite (29MB, 90,000 reviews)
- **Web Framework**: Streamlit
- **ML Libraries**: Scikit-learn, NLTK
- **Visualization**: Plotly, Matplotlib, WordCloud
- **Data Processing**: Pandas, NumPy

### File Structure
```
DataMining/
├── data/
│   ├── reviews.db (90,000 reviews)
│   ├── models/ (LDA models)
│   └── processed/ (cleaned data)
├── src/
│   ├── dashboard/
│   │   ├── analytics_dashboard.py (main dashboard)
│   │   ├── tesla_dashboard.py (Tesla UI)
│   │   └── professional_app.py (original)
│   ├── scraping/ (data collection)
│   ├── preprocessing/ (text processing)
│   └── modeling/ (ML models)
├── notebooks/ (analysis notebooks)
└── reports/ (generated reports)
```

## Next Steps for Data Mining Enhancement

### Priority 1: Advanced NLP Models
- Implement BERT/mBERT for better multilingual support
- Fine-tune models on IRCTC-specific vocabulary
- Add Named Entity Recognition (NER)
- Implement aspect-based sentiment analysis

### Priority 2: Enhanced Topic Modeling
- BERTopic implementation for dynamic topics
- Hierarchical topic clustering
- Topic evolution over time
- Topic correlation analysis

### Priority 3: Deep Analysis Features
- Root cause analysis for negative reviews
- Feature extraction (app features mentioned)
- Issue severity classification
- Resolution recommendation system

---
*Last Updated: August 28, 2025*
*Version: 3.0 - Dashboard Complete, Ready for Advanced Mining*