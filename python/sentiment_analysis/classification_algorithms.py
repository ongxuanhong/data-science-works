"""
CLASSIFICATION
Case study: Analyzing sentiment
Models:
    Linear classifiers (logistic regression, SVMs, perceptron)
    Kernels
    Decision trees
Algorithms:
    Stochastic gradient descent
    Boosting
Concepts:
    Decision boundaries, MLE, ensemble methods, random forests, CART, online learning
"""
import datetime
import os
import re
import time
from itertools import islice
from operator import itemgetter

import numpy as np
import pandas as pd
from BeautifulSoup import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"


def clean_sentence(sentence):
    # Remove HTML
    review_text = BeautifulSoup(sentence).text

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    return letters_only


def convert_plain_to_csv(plain_name, csv_name):
    t0 = time.time()
    with open(plain_name, "r") as f1, open(csv_name, "w") as f2:
        i = 0
        f2.write("productId,score,summary,text\n")
        while True:
            next_n_lines = list(islice(f1, 9))
            if not next_n_lines:
                break

            # process next_n_lines: get productId,score,summary,text info
            # remove special characters from summary and text
            output_line = ""
            for line in next_n_lines:
                if "product/productId:" in line:
                    output_line += line.split(":")[1].strip() + ","
                elif "review/score:" in line:
                    output_line += line.split(":")[1].strip() + ","
                elif "review/summary:" in line:
                    summary = clean_sentence(line.split(":")[1].strip()) + ","
                    output_line += summary
                elif "review/text:" in line:
                    text = clean_sentence(line.split(":")[1].strip()) + "\n"
                    output_line += text

            f2.write(output_line)

            # print status
            i += 1
            if i % 10000 == 0:
                print "%d reviews converted..." % i

    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))


def get_reviews_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        df = pd.read_csv(file_name)

    return df


def review_to_words(review):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    #
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


def cleaning_data(dataset, file_name):
    t0 = time.time()

    # Get the number of reviews based on the dataframe column size
    num_reviews = dataset["text"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review
    for i in xrange(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if (i + 1) % 10000 == 0:
            print "Review %d of %d\n" % (i + 1, num_reviews)

        # Call our function for each one, and add the result to the list of
        # clean reviews
        productId = str(dataset["productId"][i])
        score = str(dataset["score"][i])
        summary = str(dataset["summary"][i])
        text = review_to_words(str(dataset["text"][i]))

        clean_train_reviews.append(productId + "," + score + "," + summary + "," + text + "\n")

    print "Writing clean train reviews..."
    with open(file_name, "w") as f:
        f.write("productId,score,summary,text\n")
        for review in clean_train_reviews:
            f.write("%s\n" % review)

    print " %s - Write file completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))


def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "Words frequency..."
    for tag, count in zip(vocab, dist):
        print count, tag


if __name__ == "__main__":
    """
    Pre-processing
    """
    # converting plain text for next processing
    convert_plain_to_csv("foods.txt", "foods.csv")

    # Reading the Data
    train = get_reviews_data("foods.csv")
    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["summary"][0], "|", train["text"][0]

    cleaning_data(train, "clean_train_reviews.csv")

    """
    Bag of Words features
    """

    clean_train_reviews = pd.read_csv("clean_train_reviews.csv", nrows=1000)

    # ignore all 3* reviews
    clean_train_reviews = clean_train_reviews[clean_train_reviews["score"] != 3]
    # positive sentiment = 4* or 5* reviews
    clean_train_reviews["sentiment"] = clean_train_reviews["score"] >= 4

    train, test = train_test_split(clean_train_reviews, test_size=0.2)

    print "Creating the bag of words...\n"
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=10)

    train_text = train["text"].values.astype('U')
    test_text = test["text"].values.astype('U')

    # convert data-set to term-document matrix
    X_train = vectorizer.fit_transform(train_text).toarray()
    y_train = train["sentiment"]

    X_test = vectorizer.fit_transform(test_text).toarray()
    y_test = test["sentiment"]

    print_words_frequency(X_train)

    """
    Training
    """

    print "---------------------------"
    print "Training"
    print "---------------------------"

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    # iterate over classifiers
    results = {}

    for name, clf in zip(names, classifiers):
        print "Training " + name + " classifier..."
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results[name] = score

    print "---------------------------"
    print "Evaluation results"
    print "---------------------------"

    # sorting results and print out
    sorted(results.items(), key=itemgetter(1))
    for name in results:
        print name + " accuracy: %0.3f" % results[name]
