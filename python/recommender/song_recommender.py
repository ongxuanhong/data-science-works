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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_music_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        df = pd.read_csv(file_name)

    return df


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


if __name__ == "__main__":
    # Load music data
    song_data = load_music_data("song_data.csv")

    print "-- Explore data"
    print song_data.head()

    print "-- Showing the most popular songs in the dataset"
    unique, counts = np.unique(song_data["song"], return_counts=True)
    popular_songs = dict(zip(unique, counts))
    df_popular_songs = pd.DataFrame(popular_songs.items(), columns=["Song", "Count"])
    df_popular_songs = df_popular_songs.sort_values(by=["Count"], ascending=False)
    print df_popular_songs.head()
    print "-- Number of songs:", len(song_data)

    users = song_data['user_id'].unique()
    print "-- Number of unique users in the dataset:", len(users)

    train, test = train_test_split(song_data, test_size=0.2)
