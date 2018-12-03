from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy


# ----------------------------------------------------------------------------------------
# ----------------------------------------- MODEL ----------------------------------------
# ----------------------------------------------------------------------------------------


def encoder_decoder(
        latent_dim,
        max_encoder_seq_len,
        max_decoder_seq_len,
        num_encoder_tokens,
        num_decoder_tokens,
        glove_embedding_len,
        embeddings,
        optimizer,
        dense_activation,
        loss=categorical_crossentropy

):
    encoder_inputs = Input(
        shape=(max_encoder_seq_len,),
        name='ENCODER_INPUT'
    )
    encoder_embedding = Embedding(
        input_dim=num_encoder_tokens,
        output_dim=glove_embedding_len,
        input_length=max_encoder_seq_len,
        weights=[embeddings],
        trainable=False,
        name='ENCODER_EMBEDDING'
    )(encoder_inputs)

    encoder = LSTM(
        latent_dim,
        return_state=True,
        name="ENCODER"
    )

    encoder_outputs, state_h, state_c = encoder(encoder_embedding)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(
        shape=(max_decoder_seq_len,),
        name="DECODER_INPUT"
    )

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
    decoder = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="DECODER"
    )
    decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(
        glove_embedding_len,
        activation=dense_activation,
        name="DECODER_DENSE"
    )

    decoder_time_distributed = TimeDistributed(
        decoder_dense,
        name="DECODER_DISTRIBUTED_OUTPUT"
    )

    decoder_outputs = decoder_time_distributed(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs
    )

    # Run training
    model.compile(optimizer=optimizer, loss=loss)

    return model
