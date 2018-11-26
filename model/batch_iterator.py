import numpy as np
import pickle
import os
import sys

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))

sys.path.append(root_path)

this_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(this_path, os.pardir))
dataset_path = os.path.join(root_path, 'dataset/')
tokenized_path = os.path.join(root_path, 'tokenized/')
embedding_path = os.path.join(root_path, 'embedding/')

from utility.monitor import *
from utility.model import *


def shift_list(n, data):
    for _ in range(n):
        data.append(data.pop(0))


class BatchIterator:
    def __init__(self,
                 max_headline_len,
                 num_decoder_tokens,
                 tokenized_paths=None,
                 output_size=5000,
                 verbose=False,
                 ):

        self.max_headline_len = max_headline_len
        self.num_decoder_tokens = num_decoder_tokens
        self.tokenized_paths = tokenized_paths
        self.output_size = output_size
        self.current_path = 0
        self.current_idx = 0
        self.verbose = verbose

        self.paths_left = len(tokenized_paths) - 1

        self.done = False

    @staticmethod
    def load_tokens(file):
        with open(file, 'rb') as handle:
            data = np.array(pickle.load(handle))
            headlines = list(data[:, 0])
            articles = list(data[:, 1])

        return headlines, articles, data.shape[0]

    def get_current_path(self):
        return self.tokenized_paths[0]

    def load_data(self):

        """
        This method loads and returns chunks of data from ,eventually, multiple files
        :return: head and article pairs of output_size
        """

        heads, articles = [], []
        qty = 0

        read_in_prev_file = 0

        print('=> Loading new block.. index is at', self.current_idx)

        while qty < self.output_size:

            current_path = self.get_current_path()

            if self.verbose:
                print('=> Fetching from', current_path)

            heads_, articles_, len_ = self.load_tokens(current_path)
            qty += len_ - self.current_idx  # News not read yet in this file
            heads += heads_[self.current_idx:]
            articles += articles_[self.current_idx:]

            del heads_
            del articles_

            if qty < self.output_size:
                if self.paths_left > 0:
                    self.current_idx = 0  # Will start from the beginning next time
                    shift_list(1, self.tokenized_paths)  # Go to next file next time
                    if self.verbose:
                        t = len(self.tokenized_paths)
                        d = t - self.paths_left
                        print('[*] Need to read another file, news read so far:', qty, '/', self.output_size)
                        print('[*] Tokenized files fetched so far: ', d, '/', t)
                    self.paths_left -= 1
                    read_in_prev_file = qty

                else:
                    self.current_idx = 0  # Will start from the beginning next time
                    shift_list(1, self.tokenized_paths)  # Go to next file next time
                    read_in_prev_file = qty
                    self.done = True
                    if self.verbose:
                        print('[*] This block contains elements already fetched to reach output_size')

            else:
                self.current_idx += (self.output_size - read_in_prev_file)

                if self.verbose:
                    t = len(self.tokenized_paths)
                    d = t - self.paths_left
                    print('[*] Tokenized files fetched so far: ', d, '/', t)

        try:
            encoder_input_data, decoder_input_data, decoder_target_data = get_inputs_outputs(
                x=articles[:self.output_size],
                y=heads[:self.output_size],
                max_decoder_seq_len=self.max_headline_len,
                num_decoder_tokens=self.num_decoder_tokens,
            )

        except MemoryError as me:
            print('\n\nFATAL : Memory alloc failed. Output size might be too large, try decreasing it\n\n')
            raise me

        del heads
        del articles

        return encoder_input_data, decoder_input_data, decoder_target_data

    def __iter__(self):
        return self

    def __next__(self):
        return self.load_data()

class BatchGenerator:
    def __init__(self,
                 max_headline_len,
                 num_decoder_tokens,
                 tokenized_paths=None,
                 output_size=5000,
                 training_batch=500,
                 verbose=False,
                 ):

        self.max_headline_len = max_headline_len
        self.num_decoder_tokens = num_decoder_tokens
        self.tokenized_paths = tokenized_paths
        self.output_size = output_size
        self.current_path = 0
        self.current_idx = 0
        self.verbose = verbose

        self.paths_left = len(tokenized_paths) - 1

        self.done = False

    @staticmethod
    def load_tokens(file):
        with open(file, 'rb') as handle:
            data = np.array(pickle.load(handle))
            headlines = list(data[:, 0])
            articles = list(data[:, 1])

        return headlines, articles, data.shape[0]

    def get_current_path(self):
        return self.tokenized_paths[0]

    def load_data(self):

        """
        This method loads and returns chunks of data from ,eventually, multiple files
        :return: head and article pairs of output_size
        """

        heads, articles = [], []
        qty = 0

        read_in_prev_file = 0

        print('=> Loading new block.. index is at', self.current_idx)

        while qty < self.output_size:

            current_path = self.get_current_path()

            if self.verbose:
                print('=> Fetching from', current_path)

            heads_, articles_, len_ = self.load_tokens(current_path)
            qty += len_ - self.current_idx  # News not read yet in this file
            heads += heads_[self.current_idx:]
            articles += articles_[self.current_idx:]

            del heads_
            del articles_

            if qty < self.output_size:
                if self.paths_left > 0:
                    self.current_idx = 0  # Will start from the beginning next time
                    shift_list(1, self.tokenized_paths)  # Go to next file next time
                    if self.verbose:
                        t = len(self.tokenized_paths)
                        d = t - self.paths_left
                        print('[*] Need to read another file, news read so far:', qty, '/', self.output_size)
                        print('[*] Tokenized files fetched so far: ', d, '/', t)
                    self.paths_left -= 1
                    read_in_prev_file = qty
                '''
                We are excluding the part where everything returns to the beginning. 
                else:
                    self.current_idx = 0  # Will start from the beginning next time
                    shift_list(1, self.tokenized_paths)  # Go to next file next time
                    read_in_prev_file = qty
                    self.done = True
                    if self.verbose:
                        print('[*] This block contains elements already fetched to reach output_size')
                '''
            else:
                self.current_idx += (self.output_size - read_in_prev_file)

                if self.verbose:
                    t = len(self.tokenized_paths)
                    d = t - self.paths_left
                    print('[*] Tokenized files fetched so far: ', d, '/', t)

        try:
            encoder_input_data, decoder_input_data, decoder_target_data = get_inputs_outputs(
                x=articles[:self.output_size],
                y=heads[:self.output_size],
                max_decoder_seq_len=self.max_headline_len,
                num_decoder_tokens=self.num_decoder_tokens,
            )

        except MemoryError as me:
            print('\n\nFATAL : Memory alloc failed. Output size might be too large, try decreasing it\n\n')
            raise me

        del heads
        del articles

        return encoder_input_data, decoder_input_data, decoder_target_data

    def __iter__(self):
        return self

    def __next__(self):
        import tqdm
        while True:
            encoder_input_data, decoder_input_data, decoder_target_data = self.load_data()
            for i in tqdm(range(self.output_size/self.training_batch)):
                slice_enc_input = encoder_input_data[i*self.training_batch:(i+1)*self.training_batch]
                slice_dec_input = decoder_input_data[i*self.training_batch:(i+1)*self.training_batch]
                slice_dec_target = decoder_target_data[i*self.training_batch:(i+1)*self.training_batch]
                yield [slice_enc_input, slice_dec_input], slice_dec_target
