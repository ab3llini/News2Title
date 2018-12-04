from keras.models import Model
from keras.layers import Input, Dense, Embedding, TimeDistributed, LSTM
from keras.models import load_model
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split
from model.dataset_manager import DatasetManager, get_inputs_outputs

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
tokenized_path = os.path.join(root_path, 'tokenized/')
embedding_path = os.path.join(root_path, 'embedding/')
sys.path.append(root_path)

max_headline_len = 20
max_article_len = 30
min_headline_len = 5
min_article_len = 10
glove_embedding_len = 50
embeddings = DatasetManager.load_embeddings()
word2index = DatasetManager.load_word2index()

num_encoder_tokens=num_decoder_tokens=embeddings.shape[0]
max_encoder_seq_len = max_article_len
max_decoder_seq_len = max_headline_len
latent_dim = 64  # Latent dimensionality of the encoding space.

from model.generator import DataGenerator
data_generator = DataGenerator(max_decoder_seq_len=max_headline_len, decoder_tokens=embeddings.shape[0],test_size=0.20)

# Restore the model and reconstruct the encoder and decoder.
trained_model = load_model('n2t_full1543297558.h5')
# We reconstruct the model in order to make inference
# Encoder reconstruction


encoder_inputs = trained_model.input[0]
#Input(shape=(max_encoder_seq_len,), name='ENCODER_INPUT')
encoder_embedding = Embedding(
    input_dim=num_encoder_tokens,
    output_dim=glove_embedding_len,
    input_length=max_encoder_seq_len,
    weights=[embeddings],
    trainable=False,
    name='ENCODER_EMBEDDING'
)(encoder_inputs)

encoder = trained_model.layers[4]
#LSTM(latent_dim, return_state=True, name="ENCODER")
encoder_outputs, state_h, state_c = encoder(encoder_embedding)
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder reconstruction
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs = trained_model.input[1]
#Input(shape=(max_decoder_seq_len, ), name="DECODER_INPUT")
decoder_embedding = Embedding(
    input_dim=num_decoder_tokens,
    output_dim=glove_embedding_len,
    input_length=max_decoder_seq_len,
    weights=[embeddings],
    trainable=False,
    name='DECODER_EMBEDDING'
)(decoder_inputs)

decoder_lstm = trained_model.layers[5]
#LSTM(latent_dim, return_sequences=True, return_state=True, name="DECODER")
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)

decoder_dense = Dense(num_decoder_tokens, activation='softmax', name="DECODER_DENSE")
decoder_states = [state_h, state_c]
decoder_time_distributed = trained_model.layers[6]
#TimeDistributed(decoder_dense, name="DECODER_DISTRIBUTED_OUTPUT")
decoder_outputs = decoder_time_distributed(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
encoder_model.summary()
decoder_model.summary()

index2word = {}

for k, v in word2index.items():
    if v == word2index['unknown_token']:
        if v not in index2word:
            index2word[v] = 'unknown_token'
    else:
        index2word[v] = k


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value= encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    decoder_input = np.zeros((1, 20))
    # Populate the first character of target sequence with the start character.
    decoder_input[0, 0] = word2index['start_token']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [decoder_input] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, len(decoded_sentence), :])
        sampled_char = index2word[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'stop_token' or
           len(decoded_sentence) > max_headline_len - 1):
            stop_condition = True

        else:
            decoder_input[0, len(decoded_sentence)] = sampled_token_index

        # Update states
        states_value = [h, c]
    return decoded_sentence

import pickle

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
embedding_prefix = 'EMB_'
tokenized_prefix = 'A'
tokenized_path = os.path.join(root_path, 'tokenized/')
filelist = []

import ntpath
for f in os.listdir(tokenized_path):
    if ntpath.basename(f).startswith(embedding_prefix + tokenized_prefix):
        filelist.append(os.path.join(tokenized_path, f))
train_list,test_list = train_test_split(filelist, test_size=0.33, random_state=42)


def load_tokens(file):
    with open(file, 'rb') as handle:
        data = np.array(pickle.load(handle))
        headlines = list(data[:, 0])
        articles = list(data[:, 1])
        return headlines, articles, data.shape[0]

headline, articles, file_length = load_tokens(test_list[0])
encoder_input_data, decoder_input_data, decoder_target_data = get_inputs_outputs(
    x=articles,
    y=headline,
    max_decoder_seq_len=max_decoder_seq_len,
    num_decoder_tokens=embeddings.shape[0]
)

for element in encoder_input_data:
    print(element)
    # print(decode_sequence(np.array(element).reshape((1,30))))

