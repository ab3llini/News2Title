import pickle
from datetime import time

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from embedding.load_glove_embeddings import load_glove_embeddings
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from sklearn.model_selection import train_test_split
from utility.text import *
from utility.model import *
import random

# ---------------------------------------------------------------------------------
# --------------------------------- CONFIGURATION ---------------------------------
# ---------------------------------------------------------------------------------

# Dataset file name
tokenized = '../preprocessing/A1_TKN_5000.pkl'
print(tokenized)
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
test_ratio = 0.1
chunk_size = 500  # Size of each chunk
batch_size = 50  # Batch size for training on each chunk
tot_epochs = 10  # Number of epochs to train for.
epochs_per_chunk = 2  # Number of epochs to train each chunk on
latent_dim = 256  # Latent dimensionality of the encoding space.

# Output layer config
dense_activation = 'softmax'

# Fit config
optimizer = 'rmsprop'
loss = 'categorical_crossentropy'

# Model save name
model_name = 'n2t_5000'

# Overfitting config
early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0)

# Model checkpoint
checkpoint = ModelCheckpoint(filepath=model_name+'_earlystopped_.h5', monitor='val_loss', save_best_only=True)

# Tensorboard
# histogram_freq=1, write_graph=True
tensorboard = TensorBoard(log_dir="../tensorboard/MODEL")

# Callbacks
callbacks = [early_stopping, checkpoint, tensorboard]

# -------------------------------------------------------------------------------------
# --------------------------------- END CONFIGURATION ---------------------------------
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# --------------------------------- DATA PROCESSING -----------------------------------
# -------------------------------------------------------------------------------------

# Read tokenized news and titles
with open(tokenized, 'rb') as handle:
    data = np.array(pickle.load(handle))
    headlines, articles = data[:, 0], data[:, 1]



# Print how many articles are present in the pickle
# Print even some statistics
print('Loaded %s articles' % len(articles))
print('Headline length (words): avg = %s, min = %s, max = %s' % get_text_stats(headlines))
print('Article length (words): avg = %s, min = %s, max = %s' % get_text_stats(articles))

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

print('\nShuffling data..')
# Shuffle all the articles and save some for test
# TODO: Add shuffle

print('\nSplitting train and test data..')
articles_tr, articles_ts, headlines_tr, headlines_ts = train_test_split(articles, headlines, test_size=test_ratio)

# Prepare inputs for current chunk
encoder_input_data_ts, decoder_input_data_ts, decoder_target_data_ts = get_inputs_outputs(
    articles_ts,
    headlines_ts,
    max_decoder_seq_len,
    num_decoder_tokens,
    max_headline_len
)

print(encoder_input_data_ts, decoder_input_data_ts, decoder_target_data_ts)

# -----------------------------------------------------------------------------------------
# --------------------------------- END DATA PROCESSING -----------------------------------
# -----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# ----------------------------------------- MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------

print('\nBuilding model')


encoder_inputs = Input(shape=(max_encoder_seq_len,), name='ENCODER_INPUT')
encoder_embedding = Embedding(
    input_dim=num_encoder_tokens,
    output_dim=glove_embedding_len,
    input_length=max_encoder_seq_len,
    weights=[embeddings],
    trainable=False,
    name='ENCODER_EMBEDDING'
)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True, name="ENCODER")
encoder_outputs, state_h, state_c = encoder(encoder_embedding)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(max_decoder_seq_len, ), name="DECODER_INPUT")

decoder_embedding = Embedding(
    input_dim=num_decoder_tokens,
    output_dim=glove_embedding_len,
    input_length=max_decoder_seq_len,
    weights=[embeddings],
    trainable=False,
    name='DECODER_EMBEDDING'
)(decoder_inputs)

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder = LSTM(latent_dim, return_sequences=True, return_state=True, name="DECODER")
decoder_outputs, _, _ = decoder(decoder_embedding,
                                initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation=dense_activation, name="DECODER_DENSE")

decoder_time_distributed = TimeDistributed(decoder_dense, name="DECODER_DISTRIBUTED_OUTPUT")

decoder_outputs = decoder_time_distributed(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.summary()

# Run training
model.compile(optimizer=optimizer, loss=loss)

total_chunks = round(len(headlines_tr) / chunk_size)

for epoch in range(tot_epochs):
    print("+++++++++ Starting  epoch %s +++++++++" % (epoch + 1))
    for i in range(total_chunks):
        print("Working on chunk %d/%d" % ((i+1), total_chunks))
        if i != total_chunks - 1:
            X = articles_tr[i * chunk_size: i * chunk_size + chunk_size]
            Y = headlines_tr[i * chunk_size: i * chunk_size + chunk_size]
        else:
            X = articles_tr[i * chunk_size:]
            Y = headlines_tr[i * chunk_size:]

        # Prepare inputs for current chunk
        encoder_input_data, decoder_input_data, decoder_target_data = get_inputs_outputs(
            X,
            Y,
            max_decoder_seq_len,
            num_decoder_tokens,
            max_headline_len
        )

        model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs_per_chunk,
                  validation_data=([encoder_input_data_ts, decoder_input_data_ts], decoder_target_data_ts),
                  callbacks=callbacks
                  )

# Save model
model.save(model_name + '.h5')

# ----------------------------------------------------------------------------------------
# ------------------------------------- END MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------

