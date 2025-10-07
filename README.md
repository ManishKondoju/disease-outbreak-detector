# 🦠 Disease Outbreak Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://manishkondoju-disease-outbreak-detector.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)

## 🔗 Live Demo

**[View Live Dashboard →](https://manishkondoju-disease-outbreak-detector.streamlit.app)**

## Overview

Real-time disease outbreak detection system using ML algorithms to identify anomalies in hospital data across 20+ US cities. Features interactive maps, alerts, and comprehensive analytics.

## 🚀 Features

- **Real-time Detection**: Monitors hospital admissions across multiple cities
- **Multiple Algorithms**: Isolation Forest, Statistical Analysis, DBSCAN Clustering
- **Interactive Visualization**: Geographic outbreak mapping with heatmap overlay
- **Smart Alert System**: Severity-based classification (Critical/High/Medium/Low)
- **Comprehensive Analytics**: Time series, disease distribution, city-level analysis

## 🤖 Detection Methods

- Isolation Forest for multivariate anomaly detection
- Statistical Z-score and EWMA analysis
- DBSCAN for geographic clustering
- Ensemble approach combining all methods

## 📊 Dashboard Features

- Interactive outbreak map with 20+ cities
- Real-time risk assessment gauge
- Time series analysis with severity breakdown
- Disease-severity heatmap
- Downloadable alert reports

## Tech Stack

- Python, Streamlit, Plotly, Folium
- Scikit-learn, SciPy, Pandas, NumPy
- Real-time data streaming capabilities

## Installation
```bash
git clone https://github.com/ManishKondoju/disease-outbreak-detector.git
cd disease-outbreak-detector
pip install -r requirements.txt
streamlit run app/streamlit_app.py
