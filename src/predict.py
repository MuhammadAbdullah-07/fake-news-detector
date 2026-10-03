import os
import pickle
import numpy as np
from gensim.models import Word2Vec 
from src.preprocess import preprocess

# ── Loading model ────────────────────────────────
with open("models/model.pkl","rb") as obj:
    preprocess_obj=pickle.load(obj)

# ── Loading Scaler ────────────────────────────────
with open("models/scaler.pkl","rb") as obj:
    scaler=pickle.load(obj)    

# ── Loading Word2vec ────────────────────────────────    
w2v=Word2Vec.load("models/word2vec.model")

# ── Making predict function ────────────────────────────────

def predict(text):
    cleaning_new_text=preprocess(text)

    ## tokenizarion --> sentences to words
    tokens=cleaning_new_text.split()

    ## Avgwrod2vec
    vectors=[w2v.wv[word] for word in tokens if word in w2v.wv.index_to_key]

    if len(vectors)==0:
        vector=np.zeros(100)
    else:
        vector=np.mean(vectors,axis=0)

    ## — reshape to 2D
    vector=np.array(vector).reshape(1,-1) 

    ## — Scaling
    scaled_text=scaler.transform(vector)


    ## — Preciction
    prediction=preprocess_obj.predict(scaled_text)

    # Step 7 — return result
    if prediction[0]==1:
        return "Real Text"
    else:
        return "Fake Text"


if __name__=="__main__":
    text="BREAKING: imran khan released from jail"

    print(f"Text:{text}")
    print(f"Predicttion:{predict(text)}")       