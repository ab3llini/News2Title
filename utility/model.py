import numpy as np


def get_inputs_outputs(x, y, max_decoder_seq_len, glove_embedding_len, embeddings):

    # Prepare inputs for current chunk
    encoder_input_data = np.array(x)
    # For what concerns the decoder we need to
    decoder_input_data = np.array(y)
    length = len(y)
    decoder_target_data = np.zeros(
        (length, max_decoder_seq_len, glove_embedding_len))

    # Prepare target headline for teacher learning
    for idx, headline in enumerate(y):
        shifted = np.zeros(shape=(max_decoder_seq_len, glove_embedding_len))
        for time in range(max_decoder_seq_len - 1):  # The last Should be stop word
            shifted[time] = embeddings[headline[time + 1]]

        decoder_target_data[idx] = shifted

    return encoder_input_data, decoder_input_data, decoder_target_data

