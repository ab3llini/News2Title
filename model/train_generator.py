import sys
import os
import time
import keras.backend as K
import numpy as np
import tensorflow as tf

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
tokenized_path = os.path.join(root_path, 'tokenized/')
embedding_path = os.path.join(root_path, 'embedding/')

sys.path.append(root_path)

from model.generator import DataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from model.dataset_manager import DatasetManager
from utility.monitor import *
from model.encoderdecoder import encoder_decoder
from keras import losses

# ---------------------------------------------------------------------------------
# --------------------------------- CONFIGURATION ---------------------------------
# ---------------------------------------------------------------------------------
# Time start, it is used to save the model with a progressive index
ts = str(int(time.time()))

# Define which embedding to use
glove_embedding_len = 300

# Let's define some control variables, such as the max length of heads and desc
# that we will use in our model

# TO MAKE THESE VALUES EFFECTIVE YOU NEED TO RE-TOKENIZE THE DATA USING DatasetManager.tokenize()
# FOLLOWED BY DatasetManager.generate_embeddings()
max_headline_len = 20
max_article_len = 40
min_headline_len = 5
min_article_len = 10

# Split data into train and test
# IMPORTANT, chunk size should be COHERENT with the split
# For example : 5000 samples, ratio = 0.1 -> 500 samples will be used for testing.
# We have 4500 samples left for training.
# We cannot set chunk size to 1000 because 4500 is not a multiple of 1000! <------------------IMPORTANT
# The same reasoning goes for the batch size.
# With a chunk of 450 elements, we cannot set a batch size of 100! <------------------IMPORTANT
test_ratio = 0.1
# chunk_size = 1000  # Size of each chunk
# batch_size = 1000  # Batch size for training on each chunk
tot_epochs = 50  # Number of epochs to train for.
epochs_per_chunk = 1  # Number of epochs to train each chunk on
latent_dim = 512  # Latent dimensionality of the encoding space.

embeddings = DatasetManager.load_embeddings()
word2index = DatasetManager.load_word2index()

index2word = {}

for k, v in word2index.items():
    if v == word2index['unknown_token']:
        if v not in index2word:
            index2word[v] = 'unknown_token'
    else:
        index2word[v] = k


def find_closest_word_index(w):
  diff = embeddings - w
  delta = K.sum(diff * diff, axis=1)
  i = K.argmin(delta)
  return i



def cosine_proximity(y_true, y_pred):

    for et, ep in zip(tf.unstack(y_pred), tf.unstack(y_true)):
        if index2word[find_closest_word_index(ep)] == 'stop_token' and index2word[find_closest_word_index(et)] != 'stop_token':
            return 10000

    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return - K.sum(y_true * y_pred, axis=-1)

tensorboard_log_dir = os.path.join(root_path, 'tensorboard/emb')

# Output layer config
dense_activation = 'linear'

# Fit config
optimizer = 'rmsprop'


loss = losses.cosine_proximity

# Model save name
model_name = 'n2t_full_embedding_' + str(glove_embedding_len) + '_latent_' + str(latent_dim) + '_cosine_patience_15.h5'


# Overfitting config
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0)

# Model checkpoint
# checkpoint = ModelCheckpoint(filepath=model_name+'_earlystopped_.h5', monitor='val_loss', save_best_only=True)

# Tensorboard
# histogram_freq=1, write_graph=True
tensorboard = TensorBoard(log_dir=tensorboard_log_dir, write_graph=True)

# Callbacks
callbacks = [tensorboard, early_stopping]

# -------------------------------------------------------------------------------------
# --------------------------------- END CONFIGURATION ---------------------------------
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# --------------------------------- DATA PROCESSING -----------------------------------
# -------------------------------------------------------------------------------------

mgr = DatasetManager(max_headline_len=max_headline_len, max_article_len=max_article_len,
                     min_headline_len=min_headline_len, min_article_len=min_article_len, verbose=True)

# BEFORE PROCEEDING, YOU MUST HAVE ALREADY TOKENIZED DATASET AND CREATED EMBEDDINGS
# Run these only if you don't have training and testing sets
# THIS IS WORKING FINE:
# IF ANY ERROR WITH TFIDF POPS UP, ROLLBACK HERE

"""
mgr.tokenize(size=1000, only_tfidf=False)
mgr.generate_embeddings(glove_embedding_len=glove_embedding_len)
mgr.generate_emebedded_documents()
"""

# raise Exception('Stop here before training')
print('Before loading test set and allocating first iterator block:', available_ram())

start_time = time.time()

print('Lock \'n loaded.. We are ready to make science, sit tight..')

# ----------------------------------------------------------------------------------------
# ----------------------------------------- MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------

print('\nBuilding model')

model = encoder_decoder(latent_dim=latent_dim, max_encoder_seq_len=max_article_len,
                        max_decoder_seq_len=max_headline_len, num_encoder_tokens=embeddings.shape[0],
                        num_decoder_tokens=embeddings.shape[0], glove_embedding_len=glove_embedding_len,
                        embeddings=embeddings, optimizer=optimizer, dense_activation=dense_activation, loss=loss)

model.summary()

print('*' * 100)
print('*' * 100)
print('We will train for a total of', tot_epochs)
print('Each epoch will run on a number of chunks that depends on how the training iterator has been built')
print(
    'Run tensorboard with="sudo /home/flavio_dipalo/.local/bin/tensorboard --logdir=' + tensorboard_log_dir + ' --port 8080"')
print('If it complains about locales, run "export LC_ALL=C"')
print('Access tensorboard at http://35.225.34.25')
print('*' * 100)
print('*' * 100)
#

data_generator = DataGenerator(max_decoder_seq_len=max_headline_len, decoder_tokens=embeddings.shape[0],
                               test_size=test_ratio, embeddings=embeddings, glove_embedding_len=glove_embedding_len)

# TODO: steps_per_epoch is missing the correct termination for each epoch.
model.fit_generator(generator=data_generator.generate_train(), validation_data=data_generator.generate_test(),
                    validation_steps=data_generator.get_steps_validation(), epochs=tot_epochs, max_queue_size=2,
                    use_multiprocessing=False, verbose=2, steps_per_epoch=data_generator.get_steps_per_epoch(),
                    callbacks=callbacks)
# Save model
print('Saving model...')
model.save(model_name)
