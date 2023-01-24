import pickle

# make glove embedding model
from gensim import downloader

glove_vectors = downloader.load('glove-wiki-gigaword-100')
pickle.dump(glove_vectors, open("glove_vectors.p", "wb"))
