from keras.models import Model
from keras.layers import Input, Dense, Embedding
from keras.models import load_model
import numpy as np
import os
import sys
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split
import pickle
import ntpath
import nltk

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
embedding_prefix = 'EMB_'
tokenized_prefix = 'A'

sys.path.append(root_path)

from dataset.dataset_manager import DatasetManager
from model.probability import output_generator
from model import config

config = config.probabilistic_cfg

tokenized_path = os.path.join(root_path, config.preprocess_folder)

mgr = DatasetManager(max_headline_len=config.max_headline_len, max_article_len=config.max_article_len,
                     min_headline_len=config.min_headline_len, min_article_len=config.min_article_len, verbose=True,
                     get_in_out=output_generator.get_inputs_outputs)

embeddings = DatasetManager.load_embeddings(embedding_dir=config.embedding_matrix_location)
word2index = DatasetManager.load_word2index(embedding_dir=config.embedding_matrix_location)

from model.probability.generator import DataGenerator

data_generator = DataGenerator(max_decoder_seq_len=config.max_headline_len, decoder_tokens=embeddings.shape[0],
                               test_size=config.test_ratio)

# Restore the model and reconstruct the encoder and decoder.
trained_model = load_model('n2t_tfidf50k_embedding_50_latent_256_patience_5.h5')
# We reconstruct the model in order to make inference
# Encoder reconstruction


encoder_inputs = trained_model.input[0]
# Input(shape=(max_encoder_seq_len,), name='ENCODER_INPUT')
encoder_embedding = Embedding(input_dim=embeddings.shape[0], output_dim=config.glove_embedding_len,
                              input_length=config.max_article_len, weights=[embeddings], trainable=False,
                              name='ENCODER_EMBEDDING')(encoder_inputs)

encoder = trained_model.layers[4]
# LSTM(latent_dim, return_state=True, name="ENCODER")
encoder_outputs, state_h, state_c = encoder(encoder_embedding)
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder reconstruction
decoder_state_input_h = Input(shape=(config.latent_dim,))
decoder_state_input_c = Input(shape=(config.latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs = trained_model.input[1]
# Input(shape=(max_decoder_seq_len, ), name="DECODER_INPUT")
decoder_embedding = Embedding(input_dim=embeddings.shape[0], output_dim=config.glove_embedding_len,
                              input_length=config.max_headline_len, weights=[embeddings], trainable=False,
                              name='DECODER_EMBEDDING')(decoder_inputs)

decoder_lstm = trained_model.layers[5]
# LSTM(latent_dim, return_sequences=True, return_state=True, name="DECODER")
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)

decoder_dense = Dense(config.num_decoder_tokens, activation=config.dense_activation, name="DECODER_DENSE")
decoder_states = [state_h, state_c]
decoder_time_distributed = trained_model.layers[6]
# TimeDistributed(decoder_dense, name="DECODER_DISTRIBUTED_OUTPUT")
decoder_outputs = decoder_time_distributed(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
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
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    decoder_input = np.full((1, config.max_headline_len), fill_value=word2index['padding_token'])
    # Populate the first character of target sequence with the start character.
    decoder_input[0, 0] = word2index['start_token']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    min_before_stop = 0

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([decoder_input] + states_value)

        limit = len(word2index) if len(decoded_sentence) > min_before_stop else len(word2index) - 3

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, len(decoded_sentence), :limit])
        # sampled_token_index = np.argmax(output_tokens[0, len(decoded_sentence), :])

        sampled_char = index2word[sampled_token_index]

        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'stop_token' or len(decoded_sentence) > config.max_headline_len - 1):
            stop_condition = True

        else:
            decoder_input[0, len(decoded_sentence)] = sampled_token_index

        # Update states
        states_value = [h, c]
    return decoded_sentence


filelist = []

for f in os.listdir(tokenized_path):
    if ntpath.basename(f).startswith(embedding_prefix + tokenized_prefix):
        filelist.append(os.path.join(tokenized_path, f))

train_list, test_list = train_test_split(filelist, test_size=config.test_ratio, random_state=42)


def load_tokens(file):
    with open(file, 'rb') as handle:
        data = np.array(pickle.load(handle))
        headlines = list(data[:, 0])
        articles = list(data[:, 1])
        return headlines, articles, data.shape[0]


def map_embeddings_to_clear_sentence(emb_list):
    phrase = []
    for sampled_token_index in emb_list:
        word = index2word[sampled_token_index]
        phrase.append(str(word))
    return phrase


def get_semantic_averaged(emb_list):
    total_elements = 1
    sum_embedding_vector = np.zeros((1, embeddings.shape[1]))

    for word_token in emb_list:
        # print(word_token in index2word.values())
        if word_token not in ['unknown_token', 'padding_token', 'start_token',
                              'stop_token'] and word_token in word2index.keys():
            word_embedding = embeddings[word2index[word_token], :]
            sum_embedding_vector = np.add(sum_embedding_vector, word_embedding)
            total_elements = total_elements + 1
    sum_embedding_vector = sum_embedding_vector[0]
    averaged_embedding = sum_embedding_vector / total_elements
    return averaged_embedding


def embedding_distance(hypothesis, reference):
    hyp_embedd = get_semantic_averaged(hypothesis)
    ref_embedd = get_semantic_averaged(reference)
    if sum(hyp_embedd) == 0 or sum(ref_embedd) == 0:
        return 0
    else:
        distance_score = 1 - cosine(hyp_embedd, ref_embedd)  # it is the cosine similarity
        return distance_score


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def find_common_between(a, b):
    return intersection(a, b)


def find_common_all(a, b, c):
    common = intersection(a, b)
    common = intersection(common, c)

    return common


def highlight(common_h, common_a, common_all, ph, rh, ra, c1, c2, c3):
    ph_ = ph.copy()
    rh_ = rh.copy()
    ra_ = ra.copy()

    for i, w in enumerate(ph_):
        if w in common_h:
            ph_[i] = '<b><font color="' + c1 + '">' + w + '</font></b>\n'
        if w in common_a:
            ph_[i] = '<b><font color="' + c2 + '">' + w + '</font></b>\n'
        if w in common_all:
            ph_[i] = '<b><font color="' + c3 + '">' + w + '</font></b>\n'

    for i, w in enumerate(rh_):
        if w in common_h:
            rh_[i] = '<b><font color="' + c1 + '">' + w + '</font></b>\n'
        if w in common_a:
            rh_[i] = '<b><font color="' + c2 + '">' + w + '</font></b>\n'
        if w in common_all:
            rh_[i] = '<b><font color="' + c3 + '">' + w + '</font></b>\n'

    for i, w in enumerate(ra_):
        if w in common_h:
            ra_[i] = '<b><font color="' + c1 + '">' + w + '</font></b>\n'
        if w in common_a:
            ra_[i] = '<b><font color="' + c2 + '">' + w + '</font></b>\n'
        if w in common_all:
            ra_[i] = '<b><font color="' + c3 + '">' + w + '</font></b>\n'

    return clean(ph_), clean(rh_), clean(ra_)


def difference(list1, list2):
    new_list = []
    for i in list1:
        if i not in list2:
            new_list.append(i)

    for j in list2:
        if j not in list1:
            new_list.append(j)
    return new_list


def create_html_output(ph, rh, ra, bl):
    common_all = find_common_all(ph, rh, ra)
    common_a = difference(find_common_between(ph, rh), common_all)
    common_b = difference(find_common_between(ph, ra), common_all)

    a, b, c = highlight(common_a, common_b, common_all, ph, rh, ra, 'red', 'blue', 'green')

    html = '<b>Predicted</b> : ' + a + '<br>\n'
    html += '<b>Real</b> : ' + b + '<br>\n'
    html += '<b>Article</b> : ' + c + '<br>\n'
    html += '<b>BLEU</b> : ' + str(bl) + '<br>\n'
    html += '<hr><br>\n'

    return html


def clean(s):
    while 'unknown_token' in s:
        s.remove('unknown_token')

    while 'padding_token' in s:
        s.remove('padding_token')

    if 'start_token' in s:
        s.remove('start_token')

    if 'stop_token' in s:
        s.remove('stop_token')

    return ' '.join(s)


list_BLEU = []
list_embedding_score = []

save_name = 'evaluation_unforced.html'

# Clear file
file = open(save_name, 'w+').close()

# Open in append
file = open(save_name, 'a')

best_str = ''
best_bleu = 0

for i in range(10):

    headline, articles, file_length = load_tokens(test_list[i])

    encoder_input_data, decoder_input_data, decoder_target_data = output_generator.get_inputs_outputs(x=articles,
                                                                                                      y=headline,
                                                                                                      max_decoder_seq_len=config.max_headline_len,
                                                                                                      num_decoder_tokens=
                                                                                                      embeddings.shape[
                                                                                                          0])

    for article, headline in zip(encoder_input_data, decoder_input_data):

        real_headline = map_embeddings_to_clear_sentence(headline)
        real_article = map_embeddings_to_clear_sentence(article)

        rh = clean(real_headline)

        predicted_headline = (decode_sequence(np.array(article).reshape((1, config.max_article_len))))

        ph = clean(predicted_headline)
        ra = clean(real_article)

        BLEU_score = nltk.translate.bleu_score.sentence_bleu([predicted_headline], real_headline, weights=[1])

        list_BLEU.append(BLEU_score)
        distance_score = embedding_distance(predicted_headline, real_headline)
        list_embedding_score.append(distance_score)

        s = '*' * 100 + '\nPredicted Headline --> {}\nReal Headline --> {}\nReal Article --> {}\nBLEU-->{}'.format(ph,
                                                                                                                   rh,
                                                                                                                   ra,
                                                                                                                   BLEU_score)

        if BLEU_score > best_bleu:
            best_bleu = BLEU_score
            best_str = s

        print(s)
        file.write(create_html_output(predicted_headline, real_headline, real_article, BLEU_score))

print('final BLEU: {}, final Embedding Evaluation {}: '.format(np.mean(list_BLEU), np.mean(list_embedding_score)))
print('BEST Headline:\n' + best_str)
