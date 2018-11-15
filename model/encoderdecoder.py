import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from embedding.load_glove_embeddings import load_glove_embeddings

# Define documents
docs = ['Well done!', 'Good work', 'Great effort', 'nice work', 'Excellent!',
        'Weak', 'Poor effort!', 'not good', 'poor work', 'Could have done better.']

# Define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# ---------------------------------------------------------------------------------- TRAIN EMBEDDINGS


own_embedding_vocab_size = 10
encoded_docs_oe = [one_hot(d, own_embedding_vocab_size) for d in docs]
print(encoded_docs_oe)

maxlen = 5
padded_docs_oe = pad_sequences(encoded_docs_oe, maxlen=maxlen, padding='post')
print(padded_docs_oe)

model = Sequential()
model.add(Embedding(input_dim=own_embedding_vocab_size, # 10
                    output_dim=32,
                    input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # Compile the model
print(model.summary())  # Summarize the model
model.fit(padded_docs_oe, labels, epochs=50, verbose=0)  # Fit the model
loss, accuracy = model.evaluate(padded_docs_oe, labels, verbose=0)  # Evaluate the model
print('Accuracy: %0.3f' % accuracy)

# ---------------------------------------------------------------------------------- USE PRE-TRAINED EMBEDDINGS

word2index, embedding_matrix = load_glove_embeddings('../embedding/glove.6B.50d.txt', embedding_dim=50)


def custom_tokenize(docs):
    output_matrix = []
    for d in docs:
        indices = []
        for w in d.split():
            indices.append(word2index[re.sub(r'[^\w\s]', '', w).lower()])
        output_matrix.append(indices)
    return output_matrix


# Encode docs with our special "custom_tokenize" function
encoded_docs_ge = custom_tokenize(docs)
print(encoded_docs_ge)

# Pad documents to a max length of 5 words
maxlen = 5
padded_docs_ge = pad_sequences(encoded_docs_ge, maxlen=maxlen, padding='post')
print(padded_docs_ge)