import numpy as np
import pickle
from embedding.load_glove_embeddings import load_glove_embeddings
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from utility.text import *
from scipy.ndimage.interpolation import shift


# Define which embedding to use
glove_embedding_len = 50

# Read tokenized news and titles
with open('../preprocessing/A1_TKN_500.pkl', 'rb') as handle:
    data = np.array(pickle.load(handle))
    headlines, articles = data[:, 0], data[:, 1]

# Print how many articles are present in the pickle
# Print even some statistics
print('Loaded %s articles' % len(articles))
print('Headline length (words): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (words): avg = %s, min = %s, max = %s' % get_text_stats(articles))

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

print("\nThis is what the first 2 headlines/article pairs look like:")
print_first_n_pairs(headlines, articles, 2)

# We need to load now our embeddings in order to proceed with further processing
word2index, embeddings = load_glove_embeddings(fp='../embedding/glove.6B.' + str(glove_embedding_len) + 'd.txt', embedding_dim=50)

# Save Old Embeddings Length to show shrink ratio
OEL = len(embeddings)

# Find all the words in the truncated sentences for which we have an embedding
embeddable = get_embeddable(vocabulary_sorted, word2index)

# Shrink the embedding matrix as much as possible, by keeping only the embeddings of the words in the vocabulary
word2index, embeddings = get_reduced_embedding_matrix(embeddable, embeddings, word2index, glove_embedding_len)

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
input_words = glove_vocabulary_sorted
target_words = glove_vocabulary_sorted

num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)

max_encoder_seq_len = max_article_len
max_decoder_seq_len = max_headline_len

print('\nNumber of unique words (input/output tokens) :', num_encoder_tokens)

# See the difference
print('\nWhole glove embedding matrix that will be used as input weight has %s elements:' % len(embeddings))

# Prepare final inputs
encoder_input_data = np.array(articles, dtype='float32')
decoder_input_data = np.array(headlines, dtype='float32')
decoder_target_data = np.zeros(
    (len(headlines), max_decoder_seq_len),
    dtype='float32')

# Prepare target headline for teacher learning
for idx, headline in enumerate(decoder_input_data):
    shifted = np.zeros(shape=max_headline_len)
    for time in range(1, max_headline_len):
        shifted[time] = headline[time - 1]

    decoder_target_data[idx] = shifted

print(str(decoder_input_data[0]))
print(str(decoder_target_data[0]))


# Model definition - KERAS FUNCTIONAL API
# ---------------------------------------
batch_size = 50  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

# Define an input sequence and process it.


deep_inputs = Input(shape=(max_encoder_seq_len,), name='ENCODER INPUT')
embedding = Embedding(
    input_dim=num_encoder_tokens,
    output_dim=glove_embedding_len,
    input_length=max_encoder_seq_len,
    weights=[embeddings],
    trainable=False,
    name='EMBEDDING LAYER'
)(deep_inputs)
encoder = LSTM(latent_dim, return_state=True, name="ENCODER")
encoder_outputs, state_h, state_c = encoder(embedding)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, max_decoder_seq_len), name="DECODER INPUT")

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="DECODER")
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax', name="DECODER OUTPUT")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=[deep_inputs, decoder_inputs], outputs=decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

