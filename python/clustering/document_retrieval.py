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
import datetime
import math
import os
import time

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors


def time_diff_str(t1, t2):
    """
    Calculates time durations.
    """
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"


def load_wiki_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        df = pd.read_csv(file_name)

    return df


def freq(word, doc):
    return doc.count(word)


def word_count(doc):
    return len(doc)


def tf(word, doc):
    return (freq(word, doc) / float(word_count(doc)))


def num_docs_containing(word, list_of_docs):
    count = 0
    for document in list_of_docs:
        if freq(word, document) > 0:
            count += 1
    return 1 + count


def idf(word, list_of_docs):
    return math.log(len(list_of_docs) /
                    float(num_docs_containing(word, list_of_docs)))


def tf_idf(word, doc, list_of_docs):
    return (tf(word, doc) * idf(word, list_of_docs))


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


if __name__ == "__main__":
    t_start = time.time()
    print "-- ----------------------------------------------------------------"
    print "-- %s - Start building document retrieval systems" % datetime.datetime.now()
    print "-- ----------------------------------------------------------------"

    n_samples = 2000
    n_features = 1000
    n_topics = 10
    n_top_words = 20

    # Load wiki data
    people = load_wiki_data("people_wiki.csv")
    print people.head()
    print len(people)

    # Explore
    obama = people[people["name"] == "Barack Obama"]
    obama_row_index = obama.index.tolist()[0]
    print "-- Obama:", obama

    taylor = people[people["name"] == "Taylor Swift"]
    taylor_row_index = taylor.index.tolist()[0]
    print "-- Taylor Swift:", taylor

    # Calculate term frequency
    txt_obama = obama["text"].tolist()[0]
    print "-- Obama term frequence"
    for word in txt_obama.split():
        print word, tf(word, txt_obama)

    txt_taylor = taylor["text"].tolist()[0]
    print "-- Taylor Swift term frequence"
    for word in txt_taylor.split():
        print word, tf(word, txt_taylor)

    # Calculate TF-IDF
    print "-- Obama TF-IDF"
    for word in txt_obama.split():
        print word, tf_idf(word, txt_obama, people["text"])

    print "-- Taylor Swift TF-IDF"
    for word in txt_taylor.split():
        print word, tf_idf(word, txt_taylor, people["text"])

    # TF-IDF
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(people["text"])
    print "-- Term frequency matrix:", X_train_counts.shape

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    tfidf_matrix = X_train_tfidf.toarray()
    print "-- TF-IDF matrix:", X_train_tfidf.shape

    # Build nearest matrix
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X_train_tfidf)

    # Looking for some nearest
    (distance, found_index) = neigh.kneighbors([tfidf_matrix[obama_row_index]])
    print "-- Who is closest to Obama?"
    print people.iloc[found_index.tolist()[0]]

    (distance, found_index) = neigh.kneighbors([tfidf_matrix[taylor_row_index]])
    print "-- Who is closest to Taylor Swift?"
    print people.iloc[found_index.tolist()[0]]

    #######
    # NMF #
    #######
    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words='english')
    t0 = time.time()
    tfidf = tfidf_vectorizer.fit_transform(people["text"])
    print("done in %0.3fs." % (time.time() - t0))

    # Fit the NMF model
    print("Fitting the NMF model with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time.time()
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time.time() - t0))

    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    #######
    # LDA #
    #######
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    t0 = time.time()
    tf = tf_vectorizer.fit_transform(people["text"])
    print("done in %0.3fs." % (time.time() - t0))

    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time.time()
    lda.fit(tf)
    print("done in %0.3fs." % (time.time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    print "-- %s * DONE After * %s" % (datetime.datetime.now(), time_diff_str(t_start, time.time()))
