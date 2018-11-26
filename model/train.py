import sys
import os
import time

#TODO: remove this part, it is only used for trial without GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
tokenized_path = os.path.join(root_path, 'tokenized/')
embedding_path = os.path.join(root_path, 'embedding/')

sys.path.append(root_path)

from model.dataset_manager import DatasetManager
from utility.model import *
from utility.monitor import *
from model.encoderdecoder import encoder_decoder

# ---------------------------------------------------------------------------------
# --------------------------------- CONFIGURATION ---------------------------------
# ---------------------------------------------------------------------------------

# Define which embedding to use
glove_embedding_len = 50

# Let's define some control variables, such as the max length of heads and desc
# that we will use in our model

# TO MAKE THESE VALUES EFFECTIVE YOU NEED TO RE-TOKENIZE THE DATA USING DatasetManager.tokenize()
# FOLLOWED BY DatasetManager.generate_embeddings()
max_headline_len = 20
max_article_len = 30
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
chunk_size = 1000  # Size of each chunk
batch_size = 1000  # Batch size for training on each chunk
tot_epochs = 50  # Number of epochs to train for.
epochs_per_chunk = 1  # Number of epochs to train each chunk on
latent_dim = 64  # Latent dimensionality of the encoding space.

tensorboard_log_dir = os.path.join(root_path, 'tensorboard/News2Title')

# Output layer config
dense_activation = 'softmax'

# Fit config
optimizer = 'rmsprop'
loss = 'categorical_crossentropy'

# Model save name
model_name = 'n2t_full'

# Overfitting config
early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0)

# Model checkpoint
# checkpoint = ModelCheckpoint(filepath=model_name+'_earlystopped_.h5', monitor='val_loss', save_best_only=True)

# Tensorboard
# histogram_freq=1, write_graph=True
tensorboard = TensorBoard(
    log_dir=tensorboard_log_dir)

# Callbacks
callbacks = [early_stopping, tensorboard]

# -------------------------------------------------------------------------------------
# --------------------------------- END CONFIGURATION ---------------------------------
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# --------------------------------- DATA PROCESSING -----------------------------------
# -------------------------------------------------------------------------------------


mgr = DatasetManager(
    max_headline_len=max_headline_len,
    max_article_len=max_article_len,
    min_headline_len=min_headline_len,
    min_article_len=min_article_len,
    verbose=True
)

# BEFORE PROCEEDING, YOU MUST HAVE ALREADY TOKENIZED DATASET AND CREATED EMBEDDINGS
# Run these only if you don't have training and testing sets
mgr.tokenize(size=3000)
mgr.generate_embeddings(glove_embedding_len=glove_embedding_len)
mgr.generate_emebedded_documents()
mgr.generate_test_set(from_file=os.path.join(root_path, tokenized_path, 'EMB_A0_C1.pkl'), size=500)

print('Before loading embeddings:', available_ram())
embeddings = DatasetManager.load_embeddings()

print('Before loading test set and allocating first iterator block:', available_ram())

start_time = time.time()
training_it, ei_ts, di_ts, dt_ts = mgr.get_train_test(block_size=1000)
print("--- time to load training iterator %s seconds ---" % (time.time() - start_time))

print('Lock \'n loaded.. We are ready to make science, sit tight..')

# ----------------------------------------------------------------------------------------
# ----------------------------------------- MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------

print('\nBuilding model')

model = encoder_decoder(
    latent_dim=latent_dim,
    max_encoder_seq_len=max_article_len,
    max_decoder_seq_len=max_headline_len,
    num_encoder_tokens=embeddings.shape[0],
    num_decoder_tokens=embeddings.shape[0],
    glove_embedding_len=glove_embedding_len,
    embeddings=embeddings,
    optimizer=optimizer,
    loss=loss
)

model.summary()

print('*' * 100)
print('*' * 100)
print('We will train for a total of', tot_epochs)
print('Each epoch will run on a number of chunks that depends on how the training iterator has been built')
print('Run tensorboard with="sudo /home/flavio_dipalo/.local/bin/tensorboard --logdir=' + tensorboard_log_dir + ' --port 8080"')
print('If it complains about locales, run "export LC_ALL=C"')
print('Access tensorboard at http://35.225.34.25')
print('*' * 100)
print('*' * 100)

for epoch in range(tot_epochs):

    for encoder_input_data, decoder_input_data, decoder_target_data in training_it:

        print('=>', available_ram())

        model.fit(x=[encoder_input_data, decoder_input_data], y=decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs_per_chunk,
                  validation_data=([ei_ts, di_ts], dt_ts),
                  callbacks=callbacks,
                  verbose=1
                  )

        # MANDATORY: Delete everything manually when not needed anymore
        # MALLOC Fails otherwise
        del encoder_input_data
        del decoder_input_data
        del decoder_target_data

# Save model
model.save(model_name + '.h5')

# ----------------------------------------------------------------------------------------
# ------------------------------------- END MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------
