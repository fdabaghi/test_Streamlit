
import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import pandas as pd

import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D


df = pd.read_csv("MovieReview.csv")
print(df.head())
print(df.shape)

df = df.drop('sentiment', axis=1)





#nltk.download()
stop_words = stopwords.words('english')

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)

    # remove stopword
    mots = word_tokenize(w.strip())
    mots = [mot for mot in mots if mot not in stop_words]
    return ' '.join(mots).strip()

df.review = df.review.apply(lambda x :preprocess_sentence(x))
print(df.head())

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df.review)

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word
vocab_size = tokenizer.num_words



st.title("Modèle Word2Vec")
# vocab_size = 10000
# embedding_dim = 300
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(vocab_size, activation='softmax'))

#model.load_weights("word2vec.h5")

model=load_model("word2vec.h5")

vectors = model.layers[0].trainable_weights[0].numpy()


# Fonction de calcul de similarité cosinus
def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(dot_product(vec1, vec1) * dot_product(vec2, vec2))

# Fonction pour trouver les mots les plus proches
def find_closest(word_index, vectors, number_closest):
    list1 = []
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist, index])
    return np.asarray(sorted(list1, reverse=True)[:number_closest])

# Fonction pour afficher les mots les plus proches
def print_closest(word_index, vectors, number=10):
    index_closest_words = find_closest(word_index, vectors, number)
    closest_words = []
    for index_word in index_closest_words:
        closest_words.append((idx2word[index_word[1]], index_word[0]))
    return closest_words

# Application Streamlit
st.title("Trouver les mots les plus proches dans le modèle Word2Vec")

# Input de l'utilisateur
input_word = st.text_input("Entrez un mot pour afficher les mots les plus proches:")

# Lorsque l'utilisateur soumet un mot
if input_word:
    if input_word in word2idx:
        # Obtenez l'index du mot choisi
        word_index = word2idx[input_word]
        
        # Trouver les 10 mots les plus proches
        closest_words = print_closest(word_index, vectors, 10)
        
        # Affichez les résultats
        st.write(f"Les 10 mots les plus proches de '{input_word}' sont :")
        for word, score in closest_words:
            st.write(f"- {word} (similarité : {score:.4f})")
    else:
        st.write("Le mot entré n'existe pas dans le vocabulaire.")


  
#Exemple d'utilisation de la fonction print_closest
#print_closest('zombie')
