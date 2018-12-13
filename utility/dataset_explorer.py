import os
import pandas as pd
import pickle

from tqdm import tqdm
import sys

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))

sys.path.append(root_path)

from utility.text import *
from utility.tfidf import *

tqdm.pandas()

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
dataset_path = os.path.join(root_path, 'dataset/')
tokenized_path = os.path.join(root_path, 'tokenized/')
embedding_path = os.path.join(root_path, 'embedding/')
tfidf_path = os.path.join(root_path, 'TFIDF/')

embedding_prefix = 'EMB_'
tokenized_prefix = 'A'


def load_tfidf_features(f_name):
    file = os.path.join(tfidf_path, f_name)

    # Read tokenized news and titles
    with open(file, 'rb') as handle:
        features = pickle.load(handle)

    return features

# ciao

files = ['articles1.csv', 'articles2.csv', 'articles3.csv']

allFiles = [os.path.join(dataset_path, file) for file in files]

list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)

frame = pd.concat(list_, axis = 0, ignore_index = True)
frame = frame.fillna('')

titles = frame['title'].str.split()
articles = frame['content'].str.split()


print('=> Dataset stats')
print(len(titles))
print('Headline length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(titles.tolist()))
print('Article length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(articles.tolist()))

# Preprocessing: remove recurrent headlines (e.g: "- the new york times")
frame['title'] = frame['title'].str.replace(' - The New York Times', '')
frame['title'] = frame['title'].str.replace(' - Breitbart', '')

del titles
del articles

titles = frame['title'].str.split()

print('=> Recurrent')
print(len(titles))
print('Headline length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(titles.tolist()))
del titles


# Remove all non ASCII chars
frame['title'] = frame['title'].replace({r'[^\x00-\x7F]+': ''}, regex=True)
frame['content'] = frame['content'].replace({r'[^\x00-\x7F]+': ''}, regex=True)

titles = frame['title'].str.split()
articles = frame['content'].str.split()

print('=> ASCII Only')
print(len(titles))
print('Headline length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(titles.tolist()))
print('Article length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(articles.tolist()))

del titles
del articles

def len_(x):
    return len(str(x))


frame = frame[frame['title'].str.split().apply(len) >= 5]
frame = frame[frame['content'].str.split().apply(len) >= 10]

titles = frame['title'].str.split()
articles = frame['content'].str.split()

print('=> After removing small news')
print(len(titles))
print('Headline length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(titles.tolist()))
print('Article length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(articles.tolist()))

del titles
del articles

frame['title'] = frame['title'].progress_apply(lambda x: ' '.join(x.split()[:20]))
frame['content'] = frame['content'].progress_apply(lambda x: ' '.join(x.split()[:30]))

titles = frame['title'].str.split()
articles = frame['content'].str.split()

print('=> After truncate')
print(len(titles))
print('Headline length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(titles))
print('Article length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(articles))

del titles
del articles

frame['title'] = frame['title'].str.lower()
frame['content'] = frame['content'].str.lower()

# Tokenize
frame['title'] = frame['title'].progress_apply(lambda row: nltk.word_tokenize(row))
frame['content'] = frame['content'].progress_apply(lambda row: nltk.word_tokenize(row))

tkn_head = frame['title'].tolist()
tkn_desc = frame['content'].tolist()


print('=> After Tokenize')
print('Headline length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(tkn_head))
print('Article length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(tkn_desc))

# Truncate the articles to the first dot
tkn_head = truncate_sentences(tkn_head, 20, stop_words=['.', '!', '?'])
tkn_desc = truncate_sentences(tkn_desc, 30, stop_words=['.', '!', '?'])


print('=> After truncate first paragraph')
print('Headline length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(tkn_head))
print('Article length (words): avg = %.2f, min = %.2f, max = %.2f' % get_text_stats(tkn_desc))

xxx = len(tkn_head)

thrown = 0

for h, d in zip(tkn_head[:], tkn_desc[:]):
    if len(h) < 5 or len(d) < 10:
        tkn_head.remove(h)
        tkn_desc.remove(d)
        thrown += 1

print('=> After tokenization and paragraph truncate', thrown, 'more news have been thrown')
print(xxx - thrown)
print('Headline length (words): avg = %s, min = %s, max = %s' % get_text_stats(tkn_head))
print('Article length (words): avg = %s, min = %s, max = %s' % get_text_stats(tkn_desc))
