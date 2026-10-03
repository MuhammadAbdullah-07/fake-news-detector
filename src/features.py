import numpy as np
import os
import pandas as pd
import pickle
from gensim.models import Word2Vec
from tqdm import tqdm 

def get_features():
    df=pd.read_csv("data/preprocessed_data.csv")
    
    # Check nulls first
    print(f"Null values: {df['Cleaned_Content'].isnull().sum()}")  
    df = df.dropna(subset=["Cleaned_Content"])
    df = df.reset_index(drop=True)
    # Check nulls after drop
    print(f"null values:{df['Cleaned_Content'].isnull().sum()}")

    corpus=df["Cleaned_Content"].tolist() # we make list
    ## list needs to be converted into words b/c wrod2vec work on List of Single Single words
    tokenized_corpus=[text.split() for text in corpus]
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10
    )
    print(f"Word2Vec trained!")
    print(f"Vocab size: {len(model.wv)}")

    # AvgWord2Vec — your approach ✅
    def avgword2vec(doc):
        vectors = [model.wv[word] for word in doc if word in model.wv.index_to_key]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    X = np.array([avgword2vec(doc) for doc in tqdm(tokenized_corpus)])
    y = df["label"].tolist()

    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {len(y)}")

    os.makedirs("models",exist_ok=True)
    model.save("models/word2vec.model")
    print("Word2Vec saved → models/word2vec.model")
    return X, y


if __name__=="__main__":
    X, y = get_features()
    print("\nFeatures ready for training!")