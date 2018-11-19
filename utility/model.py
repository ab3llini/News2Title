import numpy as np


def get_inputs_outputs(x, y, max_decoder_seq_len, num_decoder_tokens, max_headline_len):

    # Prepare inputs for current chunk
    encoder_input_data = np.array(x)
    decoder_input_data = np.array(y)
    decoder_target_data = np.zeros(
        (len(y), max_decoder_seq_len, num_decoder_tokens))

    # Prepare target headline for teacher learning
    for idx, headline in enumerate(y):
        shifted = np.zeros(shape=(max_headline_len, num_decoder_tokens))
        for time in range(1, max_headline_len):
            shifted[time][headline[time - 1]] = 1.0

        decoder_target_data[idx] = shifted

    return encoder_input_data, decoder_input_data, decoder_target_data

