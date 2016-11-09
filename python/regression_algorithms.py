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
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def get_home_data():
    """Get home data, from local csv or pandas repo."""
    if os.path.exists("data/home_data.csv"):
        print("-- home_data.csv found locally")
        df = pd.read_csv("data/home_data.csv", index_col=0)

    return df


if __name__ == "__main__":
    df = get_home_data()

    # features selection
    features = list(["bedrooms", "bathrooms", "floors", "waterfront"])
    y = df["price"]
    X = df[features]

    # split dataset into training (70%) and testing set (30%)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # training model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # evaluating model
    score_trained = linear.score(x_train, y_train)
    print "Model scored:", score_trained

    # saving model
    joblib.dump(linear, "models/linear_model_v1.pkl")

    # loading model
    clf = joblib.load("models/linear_model_v1.pkl")
    predicted = clf.predict(x_test)
    print "Predicted test:", predicted
