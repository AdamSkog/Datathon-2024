# Sales Prediction Project

This project implements a sales prediction model to forecast the total sales quantity and selling price for a company. The project uses linear regression models to make predictions based on historical sales data.

## Project Overview

The goal of this project is to build a robust model that can predict future sales metrics for a company. This includes predicting the total sales quantity and the total selling price based on historical data.

## Model Choice

We chose the linear regression model for this project due to its simplicity, interpretability, and efficiency. Linear regression is a foundational algorithm in machine learning that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

### Advantages of Linear Regression:
- **Simplicity**: Easy to implement and understand.
- **Interpretability**: Provides clear insights into the relationships between features.
- **Efficiency**: Computationally less intensive compared to more complex models.

## Project Steps

### 1. Data Preparation

- **Data Loading**: Loaded historical sales data, which includes features such as product quantity and total selling price.
- **Data Aggregation**: Aggregated data to monthly totals to capture trends over time.
- **Handling Missing Values**: Filled missing values using forward fill method to maintain data consistency.

### 2. Feature Engineering

- **Lag Features**: Created lag features to include past values of product quantity and selling price as predictors for future values. This helps capture temporal dependencies.
- **Rolling Window Features**: Added rolling window features to capture the average trends over a specified window (e.g., 3 months).


### 3. Model Training

- **Train-Test Split**: Split the data into training and testing sets to evaluate model performance.
- **Linear Regression Models**: Trained two linear regression models:
  - One for predicting product quantity.
  - One for predicting total selling price.

### 4. Model Evaluation

- **Metrics**: Evaluated the models using the following metrics:
  - **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions.
  - **Mean Squared Error (MSE)**: Measures the average of the squares of the errors.
  - **Root Mean Squared Error (RMSE)**: Square root of the average of the squares of the errors.
  - **R² Score**: Represents the proportion of the variance for a dependent variable that's explained by an independent variable.

### Insights from the Data

1. **Temporal Dependencies**: The lag features and rolling window features significantly improved the model's ability to capture temporal dependencies in the data.
2. **Trend Analysis**: By aggregating the data monthly, we could observe trends and seasonality, which were crucial for making accurate predictions.
3. **Model Performance**: The linear regression models performed well in predicting sales metrics, providing a good balance between simplicity and predictive power.

### Summary of Model Performance

- **Product Quantity Model**:
  - MAE: 3.7216523196548225e-11
  - MSE: 2.1964292241243056e-21
  - RMSE: 4.686607754148309e-11

- **Selling Price Model**:
  - MAE: 8.745701052248478e-11
  - MSE: 1.327882963498726e-20
  - RMSE: 1.1523380421988705e-10

These metrics indicate that the linear regression models are effective in capturing the relationships between historical sales data and future sales metrics.

## Conclusion

The sales prediction project successfully built and evaluated linear regression models to predict future sales quantity and selling price. The models leveraged historical data, lag features, and rolling window features to make accurate predictions. The linear regression approach provided a simple yet powerful method for forecasting sales metrics, making it a valuable tool for business decision-making.
