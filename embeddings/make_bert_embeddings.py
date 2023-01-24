import pickle
import pandas as pd

from sentence_transformers import SentenceTransformer
model_sent = SentenceTransformer('all-MiniLM-L6-v2')

papers = pd.read_csv('datasets/ai_papers_00-17.csv')

titles_list = papers['paper_title']

titles_embeddings = model_sent.encode(titles_list)
pickle.dump(titles_embeddings, open("embeddings/bert_titles_embeddings_00-17.p", "wb"))

print('\nCompleted successfully\n')