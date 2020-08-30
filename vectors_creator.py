import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


DIM = 300
N = 6
SPLIT_RATE = 0.8

W2V_PATH = 'wiki.he.vec'
DATA_2_VECTORS_PATH = 'data_to_vectors_conversion_df.pkl'


def get_half_embedding(w2v, sentence):

    emb_vec = np.zeros(DIM)
    words_cnt = 0

    first_sent = list(sentence)

    for word in first_sent:
        if word in w2v:
            original_vector = w2v[word]
            emb_vec += original_vector
            words_cnt += 1

    emb_vec = emb_vec / words_cnt
    return emb_vec


def create_vectors(df, w2v):

    vectors_pkl = pd.DataFrame(index=range(len(df)), columns=range(1, 2 * DIM + 16))

    for i in range(len(df)):
        try:
            sub_df = df.iloc[i]

            vectors_pkl.iloc[i][0:DIM] = get_half_embedding(w2v, sub_df.head(N + 1).head(int(N / 2) + 1))
            vectors_pkl.iloc[i][DIM: 2 * DIM] = get_half_embedding(w2v, sub_df.head(N + 1).tail(int(N / 2)))

            vectors_pkl.iloc[i][2 * DIM + 1] = sub_df["First_Duration"]
            vectors_pkl.iloc[i][2 * DIM + 2] = sub_df["Second_Duration"]
            vectors_pkl.iloc[i][2 * DIM + 3] = sub_df["Third_Duration"]
            vectors_pkl.iloc[i][2 * DIM + 4] = sub_df["Fourth_Duration"]
            vectors_pkl.iloc[i][2 * DIM + 5] = sub_df["Fifth_Duration"]
            vectors_pkl.iloc[i][2 * DIM + 6] = sub_df["Sixth_Duration"]
            vectors_pkl.iloc[i][2 * DIM + 7] = sub_df["First_Normal"]
            vectors_pkl.iloc[i][2 * DIM + 8] = sub_df["Second_Normal"]
            vectors_pkl.iloc[i][2 * DIM + 9] = sub_df["Third_Normal"]
            vectors_pkl.iloc[i][2 * DIM + 10] = sub_df["Fourth_Normal"]
            vectors_pkl.iloc[i][2 * DIM + 11] = sub_df["Fifth_Normal"]
            vectors_pkl.iloc[i][2 * DIM + 12] = sub_df["Sixth_Normal"]
            vectors_pkl.iloc[i][2 * DIM + 13] = sub_df["Middle_Space"]
            vectors_pkl.iloc[i][2 * DIM + 14] = sub_df["Label"]
            vectors_pkl.iloc[i][2 * DIM + 15] = sub_df["ID"]

        except Exception as e:
            print("Error in reading row: " + str(i) + ", " + str(e))
            continue

    return vectors_pkl


data_to_vectors_df = pd.read_pickle(DATA_2_VECTORS_PATH)
word2vec = KeyedVectors.load_word2vec_format(W2V_PATH)

converted_data = create_vectors(data_to_vectors_df, word2vec)
converted_data = converted_data.dropna()
converted_data = converted_data.sample(frac=1)

train_size = int(SPLIT_RATE * len(converted_data))
train_df = converted_data.head(train_size)
test_df = converted_data.tail(len(converted_data) - train_size)

pd.to_pickle(train_df, 'train_vectors.pkl')
pd.to_pickle(test_df, 'test_vectors.pkl')
