import sys
import os
import time

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
# ciao
tokenized_path = os.path.join(root_path, 'tokenized/')
embedding_path = os.path.join(root_path, 'embedding/')

sys.path.append(root_path)

from model.generator import DataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy
from model.dataset_manager import DatasetManager
from utility.model import *
from utility.monitor import *
from model.encoderdecoder import encoder_decoder

# ---------------------------------------------------------------------------------
# --------------------------------- CONFIGURATION ---------------------------------
# ---------------------------------------------------------------------------------
# Time start, it is used to save the model with a progressive index
ts = str(int(time.time()))

# Define which embedding to use
glove_embedding_len = 50

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
tot_epochs = 500  # Number of epochs to train for.
epochs_per_chunk = 1  # Number of epochs to train each chunk on
latent_dim = 1024  # Latent dimensionality of the encoding space.

tensorboard_log_dir = os.path.join(root_path, 'tensorboard/News2Title')

# Output layer config
dense_activation = 'linear'

# Fit config
optimizer = 'rmsprop'
loss = 'cosine'

# Model save name
model_name = 'n2t_full_embedding'

# Overfitting config
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0)

# Model checkpoint
# checkpoint = ModelCheckpoint(filepath=model_name+'_earlystopped_.h5', monitor='val_loss', save_best_only=True)

# Tensorboard
# histogram_freq=1, write_graph=True
tensorboard = TensorBoard(log_dir=tensorboard_log_dir, write_images=True, write_graph=True, histogram_freq=2)

# Callbacks
callbacks = [early_stopping, tensorboard]

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

mgr.tokenize(size=500, only_tfidf=False)
mgr.generate_embeddings(glove_embedding_len=glove_embedding_len)
mgr.generate_emebedded_documents()


# raise Exception('Stop here before training')

print('Before loading embeddings:', available_ram())
embeddings = DatasetManager.load_embeddings()

print('Before loading test set and allocating first iterator block:', available_ram())

start_time = time.time()

print('Lock \'n loaded.. We are ready to make science, sit tight..')

# ----------------------------------------------------------------------------------------
# ----------------------------------------- MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------

print('\nBuilding model')

model = encoder_decoder(latent_dim=latent_dim, max_encoder_seq_len=max_article_len,
    max_decoder_seq_len=max_headline_len, num_encoder_tokens=embeddings.shape[0],
    num_decoder_tokens=embeddings.shape[0], glove_embedding_len=glove_embedding_len, embeddings=embeddings,
    optimizer=optimizer, dense_activation=dense_activation, loss=loss)

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

data_generator = DataGenerator(max_decoder_seq_len=max_headline_len, decoder_tokens=embeddings.shape[0], test_size=0.20,
                               embeddings=embeddings, glove_embedding_len=glove_embedding_len)

# TODO: steps_per_epoch is missing the correct termination for each epoch.
model.fit_generator(generator=data_generator.generate_train(), validation_data=data_generator.generate_test(),
    validation_steps=data_generator.get_steps_validation(), epochs=tot_epochs, max_queue_size=2,
    use_multiprocessing=False, verbose=2, steps_per_epoch=data_generator.get_steps_per_epoch(), callbacks=callbacks)
# Save model
print('Saving model...')
model.save(
    model_name + ts + '.h5')  # ----------------------------------------------------------------------------------------
# ------------------------------------- END MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------
