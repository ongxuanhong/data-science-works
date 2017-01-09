"""
MATRIX FACTORIZATION & DIMENSIONALITY REDUCTION
Case study: Recommending Products
Models:
    Collaborative filtering
    Matrix factorization
    PCA
Algorithms:
    Coordinate descent
    Eigen decomposition
    SVD
Concepts:
    Matrix completion, eigenvalues, random projections, cold-start problem, diversity, scaling up
"""
import os
from math import sqrt

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split


def load_music_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        df = pd.read_csv(file_name)

    return df


def values_to_map_index(values):
    map_index = {}
    idx = 0
    for val in values:
        map_index[val] = idx
        idx += 1

    return map_index


def print_most_popular_songs(song):
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print "Words in vocabulary:", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(song, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "Words frequency..."
    for tag, count in zip(vocab, dist):
        print count, tag


def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


if __name__ == "__main__":

    # Load music data
    song_data = load_music_data("song_data.csv")

    # Reduce complexity by getting first n elements
    n = 10000
    song_data = song_data.head(n)
    user_idx = values_to_map_index(song_data.user_id.unique())
    song_idx = values_to_map_index(song_data.song_id.unique())

    print "-- Explore data"
    print song_data.head()

    print "-- Showing the most popular songs in the dataset"
    unique, counts = np.unique(song_data["song"], return_counts=True)
    popular_songs = dict(zip(unique, counts))
    df_popular_songs = pd.DataFrame(popular_songs.items(), columns=["Song", "Count"])
    df_popular_songs = df_popular_songs.sort_values(by=["Count"], ascending=False)
    print df_popular_songs.head()

    n_users = song_data.user_id.unique().shape[0]
    n_items = song_data.song_id.unique().shape[0]
    print "Number of users = " + str(n_users) + " | Number of songs = " + str(n_items)

    train_data, test_data = train_test_split(song_data, test_size=0.25)
    train_data_matrix = np.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[user_idx[line[1]], song_idx[line[2]]] = line[3]

    test_data_matrix = np.zeros((n_users, n_items))
    for line in test_data.itertuples():
        test_data_matrix[user_idx[line[1]], song_idx[line[2]]] = line[3]

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    user_prediction = predict(train_data_matrix, user_similarity, type='user')

    print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
    print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))

    sparsity = round(1.0 - len(song_data) / float(n_users * n_items), 3)
    print 'The sparsity level is ' + str(sparsity * 100) + '%'

    # get SVD components from train matrix. Choose k.
    u, s, vt = svds(train_data_matrix, k=20)
    s_diag_matrix = np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    print 'User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix))
