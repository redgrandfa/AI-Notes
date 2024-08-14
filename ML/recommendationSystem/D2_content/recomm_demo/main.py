from typing import Union
from fastapi import FastAPI
from numpy import load
import pandas as pd

df_keys = pd.read_csv('df_keys.csv')
indices = pd.Series(df_keys.index, index = df_keys['title'])
cosine_sim = load('cosine_sim.npy')

app = FastAPI()

def recommend_movie(title, n = 10, cosine_sim = cosine_sim):
    movies = []
    # retrieve matching movie title index
    if title not in indices.index:
        print("Movie not in database.")
        return
    else:
        idx = indices[title]
    
    # cosine similarity scores of movies in descending order
    scores = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    
    # top n most similar movies indexes
    # use 1:n because 0 is the same movie entered
    top_n_idx = list(scores.iloc[1:n].index)
        
    return df_keys['title'].iloc[top_n_idx]

@app.get("/")
def read_root():
  return {"Hello": "World"}

@app.get("/items/{item_str}")
def read_item(item_str: str, q: Union[str, None] = None):
  return {"item_str": recommend_movie(item_str), "q": q}