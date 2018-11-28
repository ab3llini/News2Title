import os
import pickle
import pandas as pd

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
dataset_path = os.path.join(root_path, 'dataset/')
tokenized_path = os.path.join(root_path, 'tokenized/')
embedding_path = os.path.join(root_path, 'embedding/')
tfidf_path = os.path.join(root_path, 'TFIDF/')

files = ['articles1.csv', 'articles2.csv', 'articles3.csv']


def load_tfidf_features(f_name):

    file = os.path.join(tfidf_path, f_name)

    # Read tokenized news and titles
    with open(file, 'rb') as handle:
        features = pickle.load(handle)

    return features


def map_dataset_to_tfidf_features(tfidf_features, limit=10):

    for idx, path in enumerate(os.path.join(dataset_path, file) for file in files):

        # Load frame and drop everything but what we need
        frame = pd.read_csv(path, encoding='utf8').filter(['content'])

        # Fill nan values with nothing but replace NaN which is a float and make the script crash
        frame = frame.fillna('')

        for i, row in frame.iloc[:limit].iterrows():
            print('-' * 200)
            print('=> BEFORE:')
            print(row['content'])
            print('=> AFTER:')

            only_tfidf = ' '.join([w if w in tfidf_features else '' for w in row['content'].split()])

            print(only_tfidf)


tfidf_features = load_tfidf_features('TF_IDF_10000.pkl')
map_dataset_to_tfidf_features(tfidf_features, limit=10)