import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def get_tfidf_features_in_doc(doc, n_features):
    """
    Computes the most important n_features out of the document
    :param doc: The input document. Must be a string
    :param n_features: the number of features to extract
    :return: the most important features, ordered by importance
    """

    # Build the vectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True)

    # Split the document in sentences.
    # Each sentence will be a doc

    sentences = nltk.sent_tokenize(doc)
    vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    indices = np.argsort(vectorizer.idf_)[::-1]
    return [features[i] for i in indices[:n_features]]


""" USAGE
doc = 'Word2vec is a group of related models that are used to produce word embeddings. These models are shallow, ' \
      'two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its ' \
      'input a large corpus of text and produces a vector space, typically of several hundred dimensions, ' \
      'with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors are ' \
      'positioned in the vector space such that words that share common contexts in the corpus are located in close ' \
      'proximity to one another in the space.[1] Word2vec was created by a team of researchers led by Tomas Mikolov ' \
      'at Google. The algorithm has been subsequently analysed and explained by other researchers.[2][3] Embedding ' \
      'vectors created using the Word2vec algorithm have many advantages compared to earlier algorithms[1] such as ' \
      'latent semantic analysis. '

most_common = get_tfidf_features_in_doc(doc, 50)

print(most_common)

"""