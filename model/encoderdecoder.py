import numpy as np
import pickle
from embedding.load_glove_embeddings import load_glove_embeddings

from keras.preprocessing.sequence import pad_sequences

from utility.text import *

# Read tokenized news and titles
with open('../preprocessing/A1_TKN_500.pkl', 'rb') as handle:
    data = np.array(pickle.load(handle))
    headlines, articles = data[:, 0], data[:, 1]

# Print how many articles are present in the pickle
# Print even some statistics
print('Loaded %s articles' % len(articles))
print('Headline length (words): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (words): avg = %s, min = %s, max = %s' % get_text_stats(articles))

# Compute the vocabulary for the loaded pickle
# We pass all the possible sentences present in both the headline and the article

# Let's define some control variables, such as the max length of heads and desc
# that we will use in our model
max_headline_len = 25
max_article_len = 50

# Resize the headlines and the articles to the max length
headlines = truncate_sentences(headlines, max_headline_len)
articles = truncate_sentences(articles, max_article_len)

print('\nStats after truncate:')
print('Headline length (words): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (words): avg = %s, min = %s, max = %s' % get_text_stats(articles))

# Compute the vocabulary now, after truncating the lists
# IMPORTANT : The total number of words will still depend on the number of available embedding!
vocabulary_sorted, vocabulary_counter = get_vocabulary(headlines + articles)
print('\nSo far, there are %s different words in headlines and articles (after truncate)' % len(vocabulary_sorted))

# We need to load now our embeddings in order to proceed with further processing
word2index, embedding_matrix = load_glove_embeddings(fp='../embedding/glove.6B.50d.txt', embedding_dim=50)

# We now need to map each word to its corresponding glove embedding index
# IMPORTANT: If a word is not found in glove, IT WILL BE REMOVED! (for the moment..)
headlines = map_to_glove_index(headlines, word2index)
articles = map_to_glove_index(articles, word2index)

print('\nStats after mapping to glove indexes:')
print('Headline length (indices): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (indices): avg = %s, min = %s, max = %s' % get_text_stats(articles))

# Now let's recompute the vocabulary
# In this case of course it will be composed by indices and not words
# Each index though will have a real corresponding glove embedding
glove_vocabulary_sorted, glove_vocabulary_counter = get_vocabulary(headlines + articles)

print('\nSo far, there are %s different glove indices in headlines and articles (after truncate and glove mapping)' % len(glove_vocabulary_sorted))
print('We found embeddings for %.2f%% of words' % (len(glove_vocabulary_sorted)/len(vocabulary_sorted) * 100.0))


# Now we want to pad the headlines and articles to a fixed length
headlines = pad_sequences(headlines, maxlen=max_headline_len, padding='post')
articles = pad_sequences(articles, maxlen=max_article_len, padding='post')

print('\nStats after padding:')
print('Headline length (indices): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (indices): avg = %s, min = %s, max = %s' % get_text_stats(articles))


# Now that we have our vocabulary lock 'n loaded we can proceed
# Now let't translate the latter variables to some others more meaningful for our model
input_characters = glove_vocabulary_sorted
target_characters = glove_vocabulary_sorted

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_len = max_headline_len
max_decoder_seq_len = max_article_len

print('\nNumber of unique words (input/output tokens) :', num_encoder_tokens)


