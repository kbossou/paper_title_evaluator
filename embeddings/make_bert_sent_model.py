import pickle

# make bert sentence embedding model
from sentence_transformers import SentenceTransformer

model_sent = SentenceTransformer('all-MiniLM-L6-v2')
pickle.dump(model_sent, open("embeddings/bert_sent_model.p", "wb"))
