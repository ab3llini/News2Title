import numpy as np


def get_inputs_outputs(x, y, max_decoder_seq_len, num_decoder_tokens):

    # Prepare inputs for current chunk
    encoder_input_data = np.array(x)

    # For what concerns the decoder we need to
    decoder_input_data = np.array(y)
    decoder_target_data = np.zeros(
        (len(y), max_decoder_seq_len, num_decoder_tokens))

    # Prepare target headline for teacher learning
    for idx, headline in enumerate(y):
        shifted = np.zeros(shape=(max_decoder_seq_len, num_decoder_tokens))
        for time in range(max_decoder_seq_len - 1):  # The last Should be stop word
            shifted[time][headline[time + 1]] = 1.0

        decoder_target_data[idx] = shifted

    return encoder_input_data, decoder_input_data, decoder_target_data

