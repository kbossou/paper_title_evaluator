import spacy
nlp = spacy.load('en_core_web_md') # nlp = spacy.load('en')

import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('english'))

glove_vectors = pickle.load(open("glove_vectors.p", "rb"))

def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    title_tokens = word_tokenize(title)
    title_tokens = [w for w in title_tokens if w not in stop_words]
    
    return title_tokens

title1 = "Inspiring Social Creativity in Children with a Digital Tool"
print('\n', title1)
cleaned_title1 = clean_title(title1)
print(cleaned_title1, '\n')

title2 = "TLyricJam: A system for generating lyrics for live instrumental music"
print(title2)
cleaned_title2 = clean_title(title2)
print(cleaned_title2, '\n')

def get_title_embedding(title):
    tokens = clean_title(title)
    title_embedding = np.zeros(100, dtype="float32")
    num_key_erros = 0
    for w in tokens:
        try:
            title_embedding += glove_vectors[w]
        except KeyError:
            num_key_erros += 1
    title_embedding = title_embedding/(len(tokens)-num_key_erros)
    
    return title_embedding

title1_vec = get_title_embedding(title1)

title2_vec = get_title_embedding(title2)

def cosine_similarity(u, v):
    dot = np.dot(u, v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u**2))
    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v**2))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / np.dot(norm_u, norm_v)
    
    return cosine_similarity


cos_similarity = cosine_similarity(title1_vec, title2_vec)
print(cos_similarity, '\n')