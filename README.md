# IRCTC Review Analysis System

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://irctc-analysis.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📊 Overview

A comprehensive data mining and analysis system for IRCTC app reviews, featuring advanced NLP techniques, automated data pipelines, and interactive dashboards. This project analyzes 90,000+ user reviews to identify critical issues and provide actionable insights.

## 🎯 Key Features

- **Large-Scale Data Collection**: Automated scraping of 90,000+ reviews
- **Advanced NLP**: Topic modeling with department-wise segregation
- **Root Cause Analysis**: Pattern detection and severity classification
- **Real-time Dashboards**: Interactive visualizations with dark theme
- **Automated Pipeline**: End-to-end automation from data collection to analysis

## 🚀 Live Demo

Check out the live dashboard: [https://irctc-analysis.streamlit.app](https://irctc-analysis.streamlit.app)

## 🛠️ Technical Stack

- **Backend**: Python 3.8+, SQLite
- **NLP**: NLTK, spaCy, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Streamlit
- **Deployment**: Streamlit Cloud, GitHub Actions

## 📁 Project Structure

```
├── src/
│   ├── scraping/          # Web scraping modules
│   ├── modeling/          # NLP and ML models
│   ├── analysis/          # Root cause analysis
│   └── dashboard/         # Interactive dashboards
├── data/
│   ├── models/            # Trained models
│   └── analysis/          # Analysis results
├── run_complete_pipeline.py  # Main automation script
└── requirements.txt       # Dependencies
```

## 📈 Key Results

- **Reviews Analyzed**: 90,447
- **Department Classification**: 36.7% App, 9.1% Railway
- **Critical Issues Identified**: 5 major root causes
- **Anomalies Detected**: 9 significant patterns
- **Processing Time**: ~5 minutes for full analysis

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Vrajb24/irctc-review-analysis.git
cd irctc-review-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis pipeline:
```bash
python run_complete_pipeline.py
```

## 💻 Usage

### Run Complete Pipeline
```bash
# Full pipeline with scraping
python run_complete_pipeline.py

# Analysis only (no scraping)
python run_complete_pipeline.py --analysis-only
```

### Launch Dashboard
```bash
streamlit run src/dashboard/professional_app.py
```

## 📊 Analysis Methods

1. **Topic Modeling**: LDA with custom preprocessing
2. **Sentiment Analysis**: Multi-class classification
3. **Pattern Detection**: Statistical anomaly detection
4. **Root Cause Analysis**: 5-Why methodology
5. **Clustering**: K-means for issue grouping

## 🎯 Key Findings

- **Top Issue**: Payment failures during peak hours (23% of users)
- **Infrastructure**: Scaling issues affecting 356 cases
- **Login Problems**: 35% increase in last 7 days
- **Success Story**: Booking success rate improved by 12%

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Vraj B**
- Email: vrajb24@iitk.ac.in
- Institution: IIT Kanpur
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [@Vrajb24](https://github.com/Vrajb24)

## 🙏 Acknowledgments

- IIT Kanpur for academic support
- IRCTC for providing a platform for analysis
- Open source community for tools and libraries

---
*This project was developed as part of a Data Mining course at IIT Kanpur.*
