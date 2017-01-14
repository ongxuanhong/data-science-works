"""
CLUSTERING & RETRIEVAL
Case study: Finding documents
Models:
    Nearest neighbors
    Clustering, mixtures of Gaussians
    Latent Dirichlet allocation (LDA)
Algorithms:
    KD-trees, locality-sensitive hashing (LSH)
    K-means
    Expectation-maximization (EM)
Concepts:
    Distance metrics, approximation algorithms, hashing, sampling algorithms, scaling up with map-reduce
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors


def load_wiki_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        df = pd.read_csv(file_name)

    return df


def count_words(text):
    wordlist = text.split()
    term_freq = {}
    for w in wordlist:
        if w in term_freq.keys():
            term_freq[w] += 1
        else:
            term_freq[w] = 1
    return term_freq


if __name__ == "__main__":
    # Load wiki data
    people = load_wiki_data("people_wiki.csv")
    people = people.head(36000)
    print people.head()
    print len(people)

    # Explore
    obama = people[people["name"] == "Barack Obama"]
    obama_row_index = obama.index.tolist()[0]
    print "Obama", obama

    taylor = people[people["name"] == "Taylor Swift"]
    print "Taylor Swift", taylor["text"]

    text = obama["text"].tolist()[0]
    print count_words(text)

    # TF-IDF
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(people["text"])
    print "Term frequency", X_train_counts.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    tfidf_matrix = X_train_tfidf.toarray()
    print "TF-IDF", X_train_tfidf.shape

    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X_train_tfidf)
    (distance, found_index) = neigh.kneighbors([tfidf_matrix[obama_row_index]])
    print "Near Obama"
    print people.iloc[found_index.tolist()[0]]
