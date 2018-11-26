import numpy as np
import pickle
from utility.text import *
from utility.model import *
from sklearn.model_selection import train_test_split
import embedding.load_glove_embeddings as emb
from keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------------
# --------------------------------- CONFIGURATION ---------------------------------
# ---------------------------------------------------------------------------------

# test ratio used for the split.
test_ratio = 0.1

# Define which embedding to use
glove_embedding_len = 50

# Let's define some control variables, such as the max length of heads and desc
# that we will use in our model
max_headline_len = 15
max_article_len = 40

# Split data into train and test
# IMPORTANT, chunk size should be COHERENT with the split
# For example : 5000 samples, ratio = 0.1 -> 500 samples will be used for testing.
# We have 4500 samples left for training.
# We cannot set chunk size to 1000 because 4500 is not a multiple of 1000! <------------------IMPORTANT
# The same reasoning goes for the batch size.
# With a chunk of 450 elements, we cannot set a batch size of 100! <------------------IMPORTANT

latent_dim = 256  # Latent dimensionality of the encoding space.

# -------------------------------------------------------------------------------------
# --------------------------------- DATA PROCESSING -----------------------------------
# -------------------------------------------------------------------------------------
tokenized = '../preprocessing/A1_TKN_5000.pkl'
print(tokenized)

# Read tokenized news and titles
with open(tokenized, 'rb') as handle:
    data = np.array(pickle.load(handle))
    headlines, articles = data[:, 0], data[:, 1]

# Print how many articles are present in the pickle
# Print even some statistics
print('Loaded %s articles' % len(articles))
print('Headline length (words): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (words): avg = %s, min = %s, max = %s' % get_text_stats(articles))

print("\nThis is what the first headlines/article pair look like:")
print_first_n_pairs(headlines, articles, 1)

# Resize the headlines and the articles to the max length
headlines = truncate_sentences(headlines, max_headline_len)
articles = truncate_sentences(articles, max_article_len, stop_word='.')

print('\nStats after truncate:')
print('Headline length (words): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (words): avg = %s, min = %s, max = %s' % get_text_stats(articles))


# Compute the vocabulary now, after truncating the lists
# IMPORTANT : The total number of words will still depend on the number of available embedding!
vocabulary_sorted, vocabulary_counter = get_vocabulary(headlines + articles)

print('\nSo far, there are %s different words in headlines and articles (after truncate)' % len(vocabulary_sorted))

print("\nThis is what the first headlines/article pair look like after truncate:")
print_first_n_pairs(headlines, articles, 1)

# We need to load now our embeddings in order to proceed with further processing
word2index, embeddings = emb.load_glove_embeddings(fp='../embedding/glove.6B.' + str(glove_embedding_len) + 'd.txt', embedding_dim=50)

# Save Old Embeddings Length to show shrink ratio
OEL = len(embeddings)

# Find all the words in the truncated sentences for which we have an embedding
embeddable = get_embeddable(vocabulary_sorted, word2index)

# Shrink the embedding matrix as much as possible, by keeping only the embeddings of the words in the vocabulary
# This method will return even start and stop token selected randomly using oov words
word2index, embeddings, START, STOP, PADDING = get_reduced_embedding_matrix(embeddable, embeddings, word2index, glove_embedding_len)
# Save New Embeddings Length to show shrink ratio
NEL = len(embeddings)

# Print shrink ratio
print('Glove embedding matrix shrank by %.2f%%' % ((1 - (NEL / OEL)) * 100))

# We now need to map each word to its corresponding glove embedding index
# IMPORTANT: If a word is not found in glove, IT WILL BE REMOVED! (for the moment..)
headlines = map_to_glove_index(headlines, word2index)
articles = map_to_glove_index(articles, word2index)

print('\nStats after mapping to glove indexes:')
print('Headline length (indices): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (indices): avg = %s, min = %s, max = %s' % get_text_stats(articles))

N_EMBEDDED = len(embeddable)
print('N_embeddins',N_EMBEDDED)

print('\nSo far, there are %s different glove indices in headlines and articles (after truncate and glove mapping)' % N_EMBEDDED)
print('We found embeddings for %.2f%% of words' % (N_EMBEDDED/len(vocabulary_sorted) * 100.0))

print("\nThis is what the first headlines/article pair look like after glove mapping:")
print_first_n_pairs(headlines, articles, 1)


# VERY IMPORTANT
# We want to add first start and stop tokens, and then perform padding!!!
# This is a key part, we the order differs, we will not have what we want
add_start_stop_tokens(headlines, START, STOP, max_headline_len)
add_start_stop_tokens(articles, START, STOP, max_headline_len)

print("\nThis is what the first headlines/article pair look like after adding start and stop tokens:")
print_first_n_pairs(headlines, articles, 1)


# Now we want to pad the headlines and articles to a fixed length
headlines = pad_sequences(headlines, maxlen=max_headline_len, padding='post', value=PADDING)
articles = pad_sequences(articles, maxlen=max_article_len, padding='post', value=PADDING)

print('\nStats after padding:')
print('Headline length (indices): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (indices): avg = %s, min = %s, max = %s' % get_text_stats(articles))

print("\nThis is what the first headlines/article pair look like after padding:")
print_first_n_pairs(headlines, articles, 1)

# Now that we have our vocabulary lock 'n loaded we can proceed
# Now let't translate the latter variables to some others more meaningful for our model
input_words = target_words = embeddable
num_encoder_tokens = num_decoder_tokens = N_EMBEDDED

max_encoder_seq_len = max_article_len
max_decoder_seq_len = max_headline_len

print('\nNumber of unique words (input/output tokens) :', num_encoder_tokens)

# See the difference
print('\nWhole glove embedding matrix that will be used as input weight has %s elements:' % len(embeddings))


print('\nSplitting train and test data + shuffling..')
articles_tr, articles_ts, headlines_tr, headlines_ts = train_test_split(articles, headlines, test_size=test_ratio,
                                                                        shuffle=True)
# Prepare inputs for test data
encoder_input_data_ts, decoder_input_data_ts, decoder_target_data_ts = get_inputs_outputs(
    articles_ts,
    headlines_ts,
    max_decoder_seq_len,
    num_decoder_tokens,
)

# -----------------------------------------------------------------------------------------
# --------------------------------- END DATA PROCESSING -----------------------------------
# -----------------------------------------------------------------------------------------
