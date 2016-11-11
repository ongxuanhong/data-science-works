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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def get_home_data():
    """Get home data, from local csv."""
    if os.path.exists("data/home_data.csv"):
        print("-- home_data.csv found locally")
        df = pd.read_csv("data/home_data.csv", index_col=0)

    return df


def plotting_features_vs_target(features, x, y):
    # define number of subplot
    num_feature = len(features)
    f, axes = plt.subplots(1, num_feature, sharey=True)

    # plotting
    for i in range(0, num_feature):
        axes[i].scatter(x[features[i]], y)
        axes[i].set_title(features[i])

    plt.show()


if __name__ == "__main__":
    df = get_home_data()

    # features selection
    features = list(["bedrooms", "bathrooms", "grade"])
    print "Features name:", list(df.columns.values)
    print "Selected features:", features
    y = df["price"]
    X = df[features]

    # split data-set into training (70%) and testing set (30%)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # plotting features, target relationships
    plotting_features_vs_target(features, x_train, y_train)

    """
    DEFAULT MODEL
    """
    # training model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # evaluating model
    score_trained = linear.score(x_test, y_test)
    print "Model scored:", score_trained

    """
    LASSO MODEL
    """
    # L1 regularization
    lasso_linear = linear_model.Lasso(alpha=1.0)
    lasso_linear.fit(x_train, y_train)

    # evaluating L1 regularized model
    score_lasso_trained = lasso_linear.score(x_test, y_test)
    print "Lasso model scored:", score_lasso_trained

    """
    RIDGE MODEL
    """
    # L2 regularization
    ridge_linear = Ridge(alpha=1.0)
    ridge_linear.fit(x_train, y_train)

    # evaluating L2 regularized model
    score_ridge_trained = ridge_linear.score(x_test, y_test)
    print "Ridge model scored:", score_ridge_trained

    # saving model
    joblib.dump(linear, "models/linear_model_v1.pkl")

    # loading model
    clf = joblib.load("models/linear_model_v1.pkl")
    predicted = clf.predict(x_test)
    print "Predicted test:", predicted

    """
    POLYNOMIAL REGRESSION
    """
    poly_model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                           ('linear', linear_model.LinearRegression(fit_intercept=False))])
    poly_model = poly_model.fit(x_train, y_train)
    score_poly_trained = poly_model.score(x_test, y_test)
    print "Poly model scored:", score_poly_trained

    poly_model = Pipeline([('poly', PolynomialFeatures(interaction_only=True, degree=2)),
                           ('linear', linear_model.LinearRegression(fit_intercept=False))])
    poly_model = poly_model.fit(x_train, y_train)
    score_poly_trained = poly_model.score(x_test, y_test)
    print "Poly model (interaction only) scored:", score_poly_trained
