# Big Data Project - Big Data Analysis on Hotel Reviews

## Overview
A comprehensive big data project that analyzes hotel reviews using Apache Spark (PySpark) and machine learning techniques. The dataset consists of 515,000 hotel reviews from Europe, sourced from Booking.com. The goal is to process, analyze, and extract meaningful insights from the data using machine learning and natural language processing (NLP) techniques.

## Technologies Used
- **Apache Spark (PySpark)**: For handling large-scale data processing.
- **RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)**: For sentiment analysis on hotel reviews.
- **VADER**: To analyze sentiment trends across different seasons.
- **DeepSeek (deepseek-r1:1.5b)**: For generating summaries of hotel reviews.
- **OpenStreetMap API**: For geolocation and address correction.
- **Streamlit**: To build an interactive web-based user interface for data exploration.

## Features
### 1. **Pre-Processing**
- Cleaning and transforming the dataset.
- Handling missing data and applying geolocation corrections.
- Normalizing text for sentiment analysis.

### 2. **Backend Architecture**
The application consists of multiple components:
- `SparkBuilder`: Loads and processes the dataset using PySpark.
- `QueryManager`: Manages all queries related to reviews and sentiments.
- `RoBERTa_Sentiment`: Uses RoBERTa for classifying sentiments into positive, negative, or neutral.
- `DeepSeekSummary`: Summarizes reviews using a large language model.
- `SeasonSentimentAnalysis`: Analyzes sentiment trends across different seasons.
- `utils`: Provides utility functions for geolocation, sentiment visualization, and distance calculations.

### 3. **Sentiment Analysis**
- **RoBERTa Sentiment Model**
  - Processes positive and negative reviews separately.
  - Classifies reviews into sentiment categories (positive, neutral, negative).
  - Computes an overall sentiment score for each hotel.
- **Seasonal Sentiment Analysis**
  - Maps review sentiment trends to different seasons.
  - Uses VADER for quick sentiment scoring.
- **Summary Generation**
  - Aggregates top 50 reviews using DeepSeek.
  - Generates concise hotel summaries.

### 4. **Query & Data Analysis**
- Extracts **common adjectives and adverbs** used in positive and negative reviews.
- Identifies **influential tags** that correlate with review scores.
- Analyzes **review length trends** based on scores.
- Detects **outlier reviews** that differ significantly from the average sentiment.
- Compares **monthly trends of hotel reviews**.
- Finds **top hotels based on user preferences and location**.

## Installation
### Prerequisites
- Python 3.8+
- Apache Spark
- PySpark
- Transformers (Hugging Face)
- NLTK
- Streamlit
- OpenStreetMap API access

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/your-repository.git
cd your-repository

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run frontend/Home.py
```

## Usage
1. **Data Preprocessing:** Run `preprocess.py` to clean and format the dataset.
2. **Spark Backend:** Start the PySpark server for efficient data processing.
3. **Run Sentiment Analysis:** Execute `analyze_sentiment.py` to classify hotel reviews.
4. **Launch Web App:** Use `streamlit run app.py` to explore data through an interactive UI.

## Contributors
- Matteo Greco - [GitHub](https://github.com/matteogreco)
- Vincenzo Presta - [GitHub](https://github.com/vincenzopresta)

