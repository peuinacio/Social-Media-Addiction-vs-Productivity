# Social Media Addiction vs Productivity Analysis

## Project Description
This project performs a comprehensive data science analysis to investigate the relationship between social media addiction and individual productivity. The study covers the entire pipeline, from data cleaning and exploratory analysis to the implementation of machine learning models for regression and classification.

## Technologies and Libraries
The project was developed in Python, utilizing the following libraries:
* **Data Manipulation:** `pandas`, `numpy`, `duckdb`.
* **Visualization:** `matplotlib`, `seaborn`.
* **Statistics & Data Quality:** `statsmodels`, `scipy.stats`, `missdat`.
* **Machine Learning:** `scikit-learn` (Linear Regression, Random Forest, KNNImputer, StandardScaler) and `XGBoost`.

## Dataset Overview
The analysis uses the `social_media_productivity_6000.csv` dataset, containing 6000 records and 9 features, including age, daily screen time, social media hours, study hours, sleep hours, notifications per day, focus score, addiction level, and productivity score.

## Analysis Steps

### 1. Exploratory Data Analysis (EDA)
* Visualized data distributions through histograms to identify normality and outliers.
* Conducted segmented analysis revealing that users with "Very High Screen Time" suffer a 55% drop in productivity scores compared to others.

### 2. Data Treatment
* **Missing Data:** A non-parametric MCAR test (p-value = 0) indicated that data was not missing at random.
* **Imputation:** Missing values were handled using `KNNImputer` (n=5) to maintain data integrity instead of simple deletion.

### 3. Machine Learning: Predicting Productivity (Regression)
* Models tested: Linear Regression, Random Forest, and XGBoost.
* **Top Performer:** XGBoost achieved an R² of ~0.87 and an RMSE of ~9.83.
* **Key Insight:** Study hours were identified as the strongest positive driver for productivity, while social media hours were the primary negative driver.

### 4. Machine Learning: Addiction Level (Classification)
* A classification pipeline was built to predict the user's addiction level.
* **Target Leakage Correction:** Initial models showed near 98% accuracy due to target leakage (addiction level being calculated directly from social media hours).
* **Final Result:** After removing the leaking feature, the models achieved a balanced and realistic accuracy of approximately 67%.

## Key Conclusions
* The analysis confirms that social media addiction is directly responsible for decreased productivity.
* Interestingly, the data suggests that high screen time does not necessarily reduce sleep or study hours in this specific dataset, indicating that it may be displacing other activities like leisure or family time.
