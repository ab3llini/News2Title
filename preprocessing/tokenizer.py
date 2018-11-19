import pandas as pd
import nltk
import pickle

from keras_preprocessing.text import Tokenizer

SIZE = 5000

data = pd.read_csv('../dataset/articles1.csv').iloc[:SIZE]



# lower all strings
data['title'] = data['title'].str.lower()
data['content'] = data['content'].str.lower()

# Tokenize
tkn_head = data['title'].apply(lambda row: nltk.word_tokenize(row)).tolist()
tkn_desc = data['content'].apply(lambda row: nltk.word_tokenize(row)).tolist()


out = []
for t, d in zip(tkn_head, tkn_desc):
    out.append([t, d])



print(t)

# Save to pickle
with open('A1_TKN_'+SIZE+'.pkl', 'wb') as handle:
   pickle.dump(out, handle)