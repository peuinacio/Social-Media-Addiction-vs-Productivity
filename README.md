# 📊 Behavioral Data Science: Predicting Productivity & Social Media Addiction

## 🎯 Project Overview
In the modern digital age, the relationship between digital hygiene (screen time, notifications) and human productivity is highly debated. This project utilizes a [Kaggle dataset](https://www.kaggle.com/datasets/asifxzaman/social-media-addiction-vs-productivity-dataset) to analyze user behavior, test statistical hypotheses regarding digital addiction, and build robust Machine Learning pipelines to predict both numerical productivity scores and categorical addiction levels.

Rather than just training models to maximize accuracy, this project focuses heavily on rigorous statistical validation, extracting feature importances, and identifying common data science pitfalls like target leakage and biased distributions.

## 🧠 Key Analytical Highlights
This project demonstrates several advanced data science practices:
* **Target Leakage Identification & Mitigation:** Detected an unrealistic 100% accuracy in early classification models. Investigated and discovered that the `social_media_hours` feature was directly derived from the target variable (`addiction_level`) in the simulated dataset. The feature was removed to create a genuine, realistic predictive model.
* **Feature Importance Analysis:** Moved beyond black-box modeling by extracting feature importances from tree-based algorithms, mathematically demonstrating which habits (such as study hours and notification volume) have the most significant weight in predicting a user's productivity score.
* **Rigorous Hypothesis Testing:** Instead of blindly applying standard T-Tests, assumptions (Normality via Shapiro-Wilk and Variance Homogeneity via Levene) were rigorously checked. Upon assumption failures, the non-parametric Mann-Whitney U test was applied.
* **Zero-Inflated Distribution Handling:** Identified a massive spike of zeros in the `productivity_score` during Exploratory Data Analysis (EDA), treating it as phantom/inactive users to prevent severe bias in the regression algorithms.
* **Robust Model Validation:** Implemented 5-Fold Cross-Validation combined with a Hold-out test set to ensure model generalizability and prevent overfitting, especially when dealing with imbalanced classes.

## 🛠️ Methodology & Pipeline

### 1. Statistical Inference
* Formulated hypotheses to test the productivity gap between "High" and "Low" addiction users.
* Utilized `scipy.stats` and `statsmodels` to extract p-values and Confidence Intervals.

### 2. Predictive Modeling: Regression (Predicting Productivity)
* **Baseline:** Linear Regression.
* **Advanced Models:** Random Forest Regressor & XGBoost.
* **Evaluation:** Evaluated models using RMSE and R² scores, with XGBoost showing strong capacity to explain the variance in human productivity based strictly on behavioral features.

### 3. Predictive Modeling: Classification (Predicting Addiction Level)
* Addressed ordinal multiclass classification (Low, Medium, High).
* Dealt with Class Imbalance by analyzing macro/weighted F1-Scores.
* **Winner:** Random Forest Classifier achieved a realistic ~69.5% accuracy on unseen data (after removing data leakage), demonstrating a strong capacity to map the behavioral gradient without committing extreme errors (e.g., confusing High with Low).

## 📈 Key Business Insights
1. **Identifying Key Drivers:** Feature importance analysis successfully isolated `study_hours` and `notifications_per_day` as fundamental factors impacting individual focus scores, guiding where behavioral interventions should be focused.
2. **The "Medium" Gravity:** The confusion matrix revealed that borderline user behaviors naturally gravitate towards a "Medium" addiction level classification, representing the statistical norm. The model successfully avoided extreme misclassifications, proving it understood the ordinal nature of the problem.
3. **Behavioral Footprints:** An algorithm can predict with near 70% accuracy whether a user has a high or low digital addiction solely by observing secondary habits (sleep, study time, screen time), without needing to track specific apps.

## 💻 Technologies & Libraries Used
* **Data Manipulation & Math:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Statistical Inference:** `scipy`, `statsmodels`
* **Data Visualization:** `matplotlib`, `seaborn`

## 🚀 Next Steps (Future Roadmap)
* [ ] **Deployment:** Wrap the final XGBoost model into an interactive web application using `Streamlit`.
* [ ] **Generative AI Integration:** Connect the model's predictions to an LLM API to act as a personalized "Productivity Coach," generating custom advice based on the user's specific behavioral bottlenecks.

---
*Created by Pedro Lacerda - Feel free to connect and discuss Data Science!*
