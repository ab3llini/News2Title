import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
sys.path.append(root_path)

from model import config
from model.embedding.output_generator import get_inputs_outputs

config = config.embedding_cfg


class DataGenerator():

    def __init__(self, max_decoder_seq_len, decoder_tokens, embeddings, glove_embedding_len, test_size=0.33):
        this_path = os.path.dirname(os.path.realpath(__file__))
        root_path = os.path.abspath(os.path.join(this_path, os.pardir))
        embedding_prefix = 'EMB_'
        tokenized_prefix = 'A'
        tokenized_path = os.path.join(root_path, config.preprocess_folder)
        self.embeddings = embeddings
        self.glove_embedding_len = glove_embedding_len
        filelist = []
        import ntpath

        for f in os.listdir(tokenized_path):
            if ntpath.basename(f).startswith(embedding_prefix + tokenized_prefix):
                filelist.append(os.path.join(tokenized_path, f))
        train_list, test_list = train_test_split(filelist, test_size=test_size, random_state=42)

        self.train_list = train_list
        self.test_list = test_list
        self.max_decoder_seq_len = max_decoder_seq_len
        self.decoder_tokens = decoder_tokens

    def get_steps_per_epoch(self):
        return len(self.train_list) - 2

    def get_steps_validation(self):
        return len(self.test_list) - 2

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.length) / self.batch_size)

    def load_tokens(self, file):
        with open(file, 'rb') as handle:
            data = np.array(pickle.load(handle))
            headlines = list(data[:, 0])
            articles = list(data[:, 1])
            return headlines, articles, data.shape[0]

    def generate_train(self):
        while True:
            for file in tqdm(self.train_list):
                headline, articles, file_length = self.load_tokens(file)
                encoder_input_data, decoder_input_data, decoder_target_data = get_inputs_outputs(x=articles, y=headline,
                                                                                                 max_decoder_seq_len=self.max_decoder_seq_len,
                                                                                                 glove_embedding_len=self.glove_embedding_len,
                                                                                                 embeddings=self.embeddings)
                yield [encoder_input_data, decoder_input_data], decoder_target_data

    def generate_test(self):
        while True:
            for file in tqdm(self.test_list):
                headline, articles, file_length = self.load_tokens(file)
                encoder_input_data, decoder_input_data, decoder_target_data = get_inputs_outputs(x=articles, y=headline,
                                                                                                 max_decoder_seq_len=self.max_decoder_seq_len,
                                                                                                 glove_embedding_len=self.glove_embedding_len,
                                                                                                 embeddings=self.embeddings)
                yield [encoder_input_data, decoder_input_data], decoder_target_data
