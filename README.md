# Quantifying-Nostalgia
This project quantifies nostalgia in movie reviews by examining linguistic patterns, time gaps &amp; review ratings. It utilize ML models (TF-IDF with Logistic Regression and SVM) to identify nostalgic reviews, finding that explicit nostalgia is strongly encoded in text, though structural metadata alone is insufficient for reliable prediction.
```markdown
# Quantifying Nostalgia in Movie Reviews

## Project Overview
This project aims to analyze and quantify the presence of nostalgia in movie reviews. By leveraging IMDb movie review data and associated metadata, the study explores the relationship between nostalgic language, the time gap since a movie's release, and review ratings. It also investigates the effectiveness of various machine learning models in identifying nostalgic reviews based on both structured metadata and textual content.

## Data Sources
The dataset is comprised of:
- **Movie Metadata**: Information about 1700 movies across various genres, including release year, ratings, number of raters, and number of reviews.
- **Movie Reviews**: Approximately 930,000 raw movie reviews, each linked to a specific movie.
Dataset link "https://ieee-dataport.org/open-access/imdb-movie-reviews-movie-genres-dataset"

## Data Preparation and Feature Engineering
1.  **Data Loading & Merging**: Individual CSV files for movie genres and movie reviews were loaded and merged into a unified DataFrame. A robust merge strategy was implemented using both movie title and year to ensure data integrity and prevent duplication.
2.  **Time Gap Feature**: Calculated as the difference between the review year and the movie's release year. This feature was normalized to a range of 0-1.
3.  **Nostalgia Density**: Identified specific 'nostalgia words' (e.g., "childhood", "used to", "memories") within reviews. The 'nostalgia_count' was then normalized by the 'review_word_count' to create 'nostalgia_density'.
4.  **Sentiment Score**: Utilized the VADER sentiment analysis tool to derive a compound sentiment score for each review.
5.  **Nostalgia Binary Label**: A binary target variable ('nostalgia_binary') was created, indicating whether a review contained any nostalgia-related keywords.

## Exploratory Data Analysis & Statistical Findings
-   Approximately 15.39% of reviews were identified as containing nostalgic language.
-   **Correlation Analysis**: A correlation matrix revealed notable relationships:
    -   `rating_x` (review rating) showed a moderate positive correlation with `sentiment_score` (0.41) and a weaker positive correlation with `time_gap` (0.16) and `nostalgia_density` (0.05).
    -   `nostalgia_density` had a weak positive correlation with `time_gap` (0.10).
-   **Nostalgia vs. Time Gap**: Trends showed that nostalgia density generally increases with a larger time gap since the movie's release, peaking in older movies.
-   **Nostalgia vs. Rating**: Reviews with higher nostalgia density showed slightly higher average ratings. A t-test confirmed a statistically significant difference between ratings of nostalgic and non-nostalgic reviews (p < 0.001). However, Cohen's d effect size (0.16) indicated a small practical effect, suggesting nostalgia plays a modest role in audience evaluation.

## Machine Learning Models
The project explored two main approaches for predicting nostalgic reviews:

### 1. Structured Metadata Models (Predicting `nostalgia_binary` from `time_gap`, `sentiment_score`, `rating_x`, `num_raters`, `num_reviews`)
-   **Logistic Regression** (Accuracy: 0.57, F1-score for nostalgic class: 0.27)
-   **Random Forest** (Accuracy: 0.82, F1-score for nostalgic class: 0.15)
-   **Linear SVM** (Accuracy: 0.56, F1-score for nostalgic class: 0.27)

**Conclusion**: Predictive modeling using only structured metadata yielded poor performance for identifying the minority (nostalgic) class. This suggests that explicit nostalgia is not reliably inferred from temporal and rating-based features alone.

### 2. Text-Based Models (TF-IDF on `review` text, predicting `nostalgia_binary`)
Models were trained on TF-IDF transformed review texts:
-   **Logistic Regression** (Accuracy: 0.96, F1-score for nostalgic class: 0.88)
-   **Naive Bayes** (Accuracy: 0.86, F1-score for nostalgic class: 0.20 - poor recall for positive class)
-   **Linear SVM** (Accuracy: 0.96, F1-score for nostalgic class: 0.89)

**Conclusion**: Text-based models, particularly Logistic Regression and Linear SVM, performed exceptionally well in classifying nostalgic reviews, achieving high precision, recall, and F1-scores. This indicates that explicit nostalgia expression is strongly encoded in the linguistic patterns of the reviews.

### 3. Text-Based Models (TF-IDF on `review_cleaned` text, predicting `nostalgia_binary`)
To further investigate, nostalgia words were removed from the reviews and models were retrained:
-   **Logistic Regression** (Accuracy: 0.70, F1-score for nostalgic class: 0.40)
-   **Naive Bayes** (Accuracy: 0.85, F1-score for nostalgic class: 0.01)
-   **Linear SVM** (Accuracy: 0.70, F1-score for nostalgic class: 0.40)

**Conclusion**: After removing explicit nostalgia terms, classification performance significantly dropped (F1 for nostalgic class around 0.40). This confirms that direct lexical indicators are crucial for detecting nostalgia. However, the moderate performance still achieved suggests that nostalgia might also be partially embedded in broader narrative and affective language patterns beyond just specific keywords.

## Setup and Usage
This project was developed in a Google Colab environment. The data is loaded from Google Drive, unzipped, and processed using standard Python libraries like `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`. The VADER sentiment analyzer was used for sentiment scoring.

To run this notebook:
1.  Ensure you have a Google Colab environment.
2.  Mount your Google Drive to `/content/drive`.
3.  Place `dataset.zip` in `/content/drive/MyDrive/Nostalgia_Dataset/`.
4.  Execute the cells sequentially.
```
