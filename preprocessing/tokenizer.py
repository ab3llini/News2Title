import pandas as pd
import nltk
import pickle

from keras_preprocessing.text import Tokenizer

SIZE = 5000

data = pd.read_csv('../dataset/articles1.csv', encoding='utf8').iloc[:SIZE]

# lower all strings
data['title'] = data['title'].str.lower()
data['content'] = data['content'].str.lower()

# Preprocessing: revoming recurrent headlines (e.g: "- the new york times")
data['title'] = data['title'].str.replace('- the new york times','')

# Remove all non ASCII chars

data['title'] = data['title'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))
data['content'] = data['content'].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

# Tokenize
tkn_head = data['title'].apply(lambda row: nltk.word_tokenize(row)).tolist()
tkn_desc = data['content'].apply(lambda row: nltk.word_tokenize(row)).tolist()



out = []
for t, d in zip(tkn_head, tkn_desc):
    out.append([t, d])

# Save to pickle
with open('A1_TKN_'+str(SIZE)+'.pkl', 'wb') as handle:
   pickle.dump(out, handle)
