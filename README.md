# Airbnb-Price-Prediction-Model
Airbnb price prediction model with kaggle dataset 

This project is a regression model developed to predict nightly prices of homes listed on Airbnb in New York City, based on accommodation data. The project aims to help beginner-level users learn essential data science steps such as data cleaning, model training, and visualization.

Project Objective
Work with real-world data

Understand and apply a regression model (Linear Regression)

Implement data visualization and analysis techniques

Learn model evaluation metrics (MSE, R²)

Technologies and Libraries Used
Python (3.9+)

Pandas – data manipulation

NumPy – numerical operations

Matplotlib – visualization

Seaborn – statistical plots

Scikit-learn – machine learning models

Dataset
The dataset used is available on Kaggle - Airbnb NYC 2019 Dataset.

Key Columns:
neighbourhood_group – neighborhood district

room_type – type of room (private room, entire home, etc.)

price – nightly price

number_of_reviews, minimum_nights, etc. – explanatory variables

## Project Steps

1. **Data Loading and Cleaning**
    - Removal of lost and meaningless values
    - Conversion of categorical variables (One-Hot Encoding)
    - Filtering outlier values
2. **Feature Selection**
    - Determination of variables affecting the model
3. **Model Training**
    - Forecasting using Linear Regression model
4. **Evaluation**
    - Mean Squared Error (MSE)
    - R² Score (Coefficient of determination)
5. **Visualization**
    - Price distributions (before / after log)
    - Prices by room type
    - Prices by neighborhood
    - Actual vs predicted values
    - Correlation matrix

## Sample Output
Mean Squared Error: 23927.28
R² Score: 0.168

Actual Price: 58.00 USD, Predict: 91.51 USD
Actual Price: 250.00 USD, Predict: 180.98 USD

## Example Charts

- Actual vs Prediction Price Chart
- Correlation Matrix
- Log(Price) Histogram
- Box Plot by District

## What has been learned

- How to clean real world data?
- How to build a regression model?
- How to analyze model outputs?
- How to interpret the result with visualization?

## Development Ideas

- Comparison with other regression models (Random Forest, XGBoost)
- Feature Importance analysis
- Model optimization (GridSearchCV)
- Interactive forecasting tool with web interface (Flask or Streamlit)
