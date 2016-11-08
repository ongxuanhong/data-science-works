"""
REGRESSION
Case study: Predicting house prices
Models:
    Linear regression
    Regularization: Ridge (L2), Lasso (L1)
Algorithms:
    Gradient descent
    Coordinate descent
Concepts:
    Loss functions, bias-variance tradeoff, cross-validation, sparsity, overfitting, model selection
"""

import os

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def get_home_data():
    """Get home data, from local csv or pandas repo."""
    if os.path.exists("data/home_data.csv"):
        print("-- home_data.csv found locally")
        df = pd.read_csv("data/home_data.csv", index_col=0)

    return df


if __name__ == "__main__":
    df = get_home_data()

    features = list(["bedrooms", "bathrooms", "floors", "waterfront"])
    y = df["price"]
    X = df[features]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    score_trained = linear.score(x_train, y_train)

    predicted = linear.predict(x_test)
    print score_trained
