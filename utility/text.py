from collections import Counter
import numpy as np
import random

def get_vocabulary(list_):

    """
    Computes the vocabulary for the provided list of sentences
    :param list_: a list of sentences (strings)
    :return: a dictionary with key, val = word, count and a sorted list, by count, of all the words
    """

    all_the_words = []

    for text in list_:
        for word in text:
            all_the_words.append(word)

    vocabulary_counter = Counter(all_the_words)
    vocabulary_sorted = list(map(lambda x: x[0], sorted(vocabulary_counter.items(), key=lambda x: -x[1])))
    return vocabulary_sorted, vocabulary_counter


def get_avg_text_len(list_):
    return sum([len(x) for x in list_]) / len(list_)


def get_max_text_len(list_):
    return max([len(x) for x in list_])


def get_min_text_len(list_):
    return min([len(x) for x in list_])


def get_text_stats(list_):
    """
    Returns statistics about the list of sentences in a more efficient way
    w.r.t. calling the three methods above separately
    """

    lens = [len(x) for x in list_]
    l_ = len(list_)

    return sum(lens) / l_, min(lens), max(lens)


def map_sentence_to_glove_index(list_, word2index):
    """
    Will map all the words in the list with their corresponding index in the glove words dict
    :param list_: the list of words
    :param word2index: the dict which contains the mapping between words and glove indices
    :return: the mapped list
    """
    mapped = []

    for word in list_:
        if word in word2index:
            mapped.append(word2index[word])

    return mapped


def map_to_glove_index(sentences, word2index):
    """
    Wrapper around map_sentence_to_glove_index to process multiple headlines/articles
    """
    indexed = []
    for sentence in sentences:
        indexed.append(map_sentence_to_glove_index(sentence, word2index))

    return indexed


def truncate_sentences(sentences, max_len, stop_words=None):
    truncated = []
    for sentence in sentences:

        truncate_idx = max_len

        if stop_words is not None:
            for word in stop_words:
                if word in sentence:
                    truncate_idx = min(sentence.index(word), max_len)
                    break

        truncated.append(sentence[:truncate_idx])

    return truncated


def get_embeddable(words, word2index):
    embeddable = []
    for word in words:
        if word in word2index:
            embeddable.append(word)
    return embeddable


def print_first_n_pairs(a, b, n):
    for (i, (a_, b_)) in enumerate(zip(a[:n], b[:n])):
        print(str(i) + ')\n' + str(a_) + '\n' + str(b_))


def rotate(l, n):
    return l[n:] + l[:n]


def get_reduced_embedding_matrix(vocab, glove_embeddings, word2index, glove_size,truncate_embedding_matrix_to=None):

    new_word2index = {}
    voc_len = len(vocab)
    if truncate_embedding_matrix_to:
        voc_len = truncate_embedding_matrix_to
    new_embedding = np.zeros((voc_len + 4, glove_size))  # +3 to account for start, stop and padding tokens

    # Add start and stop
    new_word2index['start_token'] = voc_len
    new_word2index['stop_token'] = voc_len + 1
    new_word2index['padding_token'] = voc_len + 2
    new_word2index['unknown_token'] = voc_len + 3

    new_embedding[voc_len] = glove_embeddings[voc_len]
    new_embedding[voc_len + 1] = glove_embeddings[voc_len + 1]
    new_embedding[voc_len + 2] = glove_embeddings[voc_len + 2]
    new_embedding[voc_len + 3] = glove_embeddings[voc_len + 3]

    # Modify vocab appending objects
    vocab.append('start_token')
    vocab.append('stop_token')
    vocab.append('padding_token')
    vocab.append('unknown_token')

    for index, word in enumerate(vocab):
        if truncate_embedding_matrix_to:
            # if this option is enabled then only the first n words are mapped to the relative embedding, all the
            # other words are mapped into the unknown token.
            if index > truncate_embedding_matrix_to-1:
                new_word2index[word] = voc_len + 3
            else:
                new_word2index[word] = index
                new_embedding[index] = glove_embeddings[word2index[word]]
        else:
            new_word2index[word] = index
            new_embedding[index] = glove_embeddings[word2index[word]]

    return new_word2index, new_embedding, voc_len, voc_len + 1, voc_len + 2


def add_start_stop_tokens(sentences, start_tkn, stop_tkn, max_len):

    for idx, sentence in enumerate(sentences):

        s_len = len(sentence)

        # First off, add start token
        if s_len == max_len:
            # Throw away last word, we do not have space and insert start token in the first place
            # and end token in the end
            rotate(sentence, 1)
            sentence[0] = start_tkn
            sentence[s_len - 1] = stop_tkn

        elif s_len <= max_len - 2:
            # Enlarge the sentence by concatenating it with start and stop tokens
            sentences[idx] = [start_tkn] + sentence + [stop_tkn]

        else:
            # In this case we have space only for the start token, drop the last word
            sentence[s_len - 1] = stop_tkn
            sentences[idx] = [start_tkn] + sentence



