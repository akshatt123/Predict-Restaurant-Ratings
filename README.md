# ***Predict-Restaurant-Ratings***

## Project Overview

Predicting restaurant ratings is essential for businesses to understand customer satisfaction and for users to make informed decisions about dining. This project implements machine learning models to predict restaurant ratings based on various features like price, cuisine, votes, and delivery options. The dataset is preprocessed, explored, and analyzed using advanced regression models with feature engineering and hyperparameter tuning to achieve optimal results.

## Table of Contents

* Project Overview

* Features and Dataset

* Exploratory Data Analysis (EDA)

* Data Processing

* Modeling

 * Linear Regression

 * Elastic Net Regression

 * Decision Tree Regressor

 * Random Forest Regressor

* Results and Evaluation

* Dependencies

* Future Work

## Features and Dataset

**Dataset**

The dataset is provided as Dataset.csv and includes the following key features:

* Numerical Features:

 * Average Cost for Two

 * Price Range

 * Votes

 * Aggregate Rating (Target variable)

* Categorical Features:

 * Has Online Delivery

* Cuisines

* Others (e.g., Restaurant Name, Locality)

**Irrelevant Features Removed:**

* Restaurant ID, Address, Country Code, City, Longitude, Latitude, Rating Color, Rating Text, etc.

## Exploratory Data Analysis (EDA)

**Visualizations**

**1.Target Variable Distribution:**

* A histogram of Aggregate Rating shows the distribution of restaurant ratings.

**2.Correlation Analysis:**

* Heatmap to analyze relationships between numerical features like votes, price, and ratings.

**3.Feature-Specific Insights:**

* Boxplots for the impact of Price Range on ratings.

* Scatterplots for analyzing relationships between cost, votes, and ratings.

**4.Cuisine Analysis:**

Bar chart showcasing the top 10 most common cuisines in the dataset.

## Data Processing

**Handling Missing Values**

**1.Numerical columns:** Missing values are replaced with the column mean.

**2.Categorical columns:** Missing values are replaced with the mode.

**Feature Engineering**

* Removed irrelevant features.

* Used StandardScaler for numerical feature scaling.

* One-hot encoded categorical variables.

* Combined preprocessed numeric and categorical features into the final feature set.

**Dimensionality Reduction**

* Principal Component Analysis (PCA) is applied to reduce the dataset dimensions while retaining 95% of the variance.

## Modeling

**1. Linear Regression**

* Technique: Fit a baseline regression model using PCA-transformed data.

* Evaluation:

 * Mean Squared Error (MSE)

 * R-squared (R²)

 * Cross-validated R² scores

**2. Elastic Net Regression**

* Technique: Combined L1 and L2 regularization for feature selection and performance improvement.

* Hyperparameter Tuning:

 * Grid search over alpha and l1_ratio.

* Evaluation:

 * Best parameters from GridSearchCV

* R² and MSE

**3. Decision Tree Regressor**

* Technique: Non-linear regression using decision trees.

* Hyperparameter Tuning:

 * Grid search over max_depth, min_samples_split, and min_samples_leaf.

* Evaluation:

 * Best parameters and metrics (MSE, R²).

**4. Random Forest Regressor** 

* Technique: Ensemble model using multiple decision trees.

* Hyperparameter Tuning:

 * Grid search over n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features.

* Evaluation:

 * Best parameters and metrics (MSE, R²).

## Results and Evaluation

| Model | Mean Squared Error (MSE) | R-squared (R²)|
| ----- | ------------------------ | ------------- |
| Linear Regression | 1.5124260163596166 | 0.33570441196440604 |
| Elastic Net Regression | 1.6158178993826522 | 0.2902920936176131 |
| Decision Tree | 0.11649167575547087 | 0.9488339228430516 |
| Random Forest | 0.1216047370620497 | 0.9465881375744992 |




* **Observations:**

* The Random Forest model provided the best performance in terms of R² and MSE.

* Elastic Net regression improved over Linear Regression due to regularization.


## Dependencies

* Python 3.8+

* Libraries:

 * pandas

 * numpy

 * seaborn

 * matplotlib

 * scikit-learn



## Future Work

**1.Advanced Models:**

* Explore other regression models like Gradient Boosting, XGBoost, or LightGBM.

**2.Feature Engineering:**

* Use text embeddings for analyzing cuisines.

**3.Hyperparameter Optimization:**

* Employ Bayesian Optimization or Optuna for tuning.

**4.Deployment:**

* Deploy the model using Flask or FastAPI for real-time predictions.

## License

This project is licensed under the MIT License. See LICENSE for more information.



