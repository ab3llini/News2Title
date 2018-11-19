import pickle
from collections import Counter
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

import matplotlib.pyplot as plt

with open('A1_TKN_500.pkl', 'rb') as handle:
    data = np.array(pickle.load(handle))
    heads, desc = data[:, 0], data[:, 1]

print("Working on ", len(heads), " articles..")


def get_vocab(lst):

    all_tokens = []

    for text in lst:
        for token in text:
            all_tokens.append(token)

    vocab = Counter(all_tokens)
    sorted_voc = list(map(lambda x: x[0], sorted(vocab.items(), key=lambda x: -x[1])))
    return sorted_voc, vocab


sorted_voc, vocab = get_vocab(heads+desc)


def plot_distr(sorted, counter):
    plt.plot([counter[w] for w in sorted]);
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_yscale("log", nonposy='clip')
    plt.title('word distribution in headlines and discription')
    plt.xlabel('rank')
    plt.ylabel('total appearances');
    plt.show()


print("Th 50 most frequent tokens are", sorted_voc[:50])


empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word

def index_mapper(vocab):

    word2idx = dict((word, idx + start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos

    idx2word = dict((idx, word) for word, idx in word2idx.items())

    return word2idx, idx2word

word2idx, idx2word = index_mapper(vocab)

print("loading GloVe embedding")

# Load glove model
glove = pd.read_table('../embedding/glove.6B.100d.txt', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


words_matrix = glove.as_matrix()

def vec(w):
  return glove.loc[w].as_matrix()

def find_closest_word(v):
  diff = words_matrix - v
  delta = np.sum(diff * diff, axis=1)
  i = np.argmin(delta)
  return glove.iloc[i].name


plot_distr(sorted_voc, vocab)

print("Creating GloVe embedding for our voc")
print("In total there are ", len(sorted_voc), " tokens")


word2glove = {}
for w in tqdm(sorted_voc):
    try:
        word2glove[w] = vec(w)
    except Exception as e:
        sorted_voc.remove(w)

print("Found embedding for ", len(sorted_voc), " tokens")


plot_distr(sorted_voc, vocab)
