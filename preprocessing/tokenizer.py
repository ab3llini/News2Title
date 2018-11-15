import pandas as pd
import nltk
import pickle

data = pd.read_csv('../dataset/articles1.csv').iloc[:500]

# lower all strings
data['title'] = data['title'].str.lower()
data['content'] = data['content'].str.lower()

# Tokenize
tkn_head = data['title'].apply(lambda row: nltk.word_tokenize(row)).tolist()
tkn_desc = data['content'].apply(lambda row: nltk.word_tokenize(row)).tolist()

print(tkn_head[0])

out = []
for t, d in zip(tkn_head, tkn_desc):
    out.append([t, d])


# Save to pickle
with open('A1_TKN_500.pkl', 'wb') as handle:
    pickle.dump(out, handle)