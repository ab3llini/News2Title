from keras.engine.saving import load_model
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from model.data_processing import latent_dim, num_encoder_tokens, glove_embedding_len, max_encoder_seq_len, \
    embeddings, num_decoder_tokens, max_decoder_seq_len, word2index, encoder_input_data_ts
from keras.models import load_model
import numpy as np

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

# starting from word2index We build the index2word dictionary that maps the index in the word.
index2word = dict((y,x) for x,y in word2index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    #target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq = np.zeros((1, max_decoder_seq_len))

    # Populate the first character of target sequence with the start character.
    #TODO: spara ad 1 il valore dell'embedding token per start.
    #target_seq[0, 0, word2index['START_tkn']] = 1.
    target_seq[0, 0] = word2index['START_tkn']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        # sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token_index = np.argmax(output_tokens[0,-1, :])
        sampled_char = index2word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'STOP_tkn' or
           len(decoded_sentence) > max_decoder_seq_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        #target_seq = np.zeros((1,1, num_decoder_tokens))
        #target_seq[0, 0, sampled_token_index] = 1.
        target_seq[0,0] = sampled_token_index
        # Update states
        states_value = [h, c]

    return decoded_sentence

for i in range(encoder_input_data_ts.shape[0]):
    seq = encoder_input_data_ts[i,:]
    seq = seq.reshape((1,40))
    print(decode_sequence(seq))

'''
encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
'''
