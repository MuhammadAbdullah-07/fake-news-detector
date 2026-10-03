import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

def preprocess(text):
    ##cleaning
    text=re.sub(r'http\S+|www\S+', '', text) ## remove http tags
    text=re.sub(r'[^a-zA-Z]',' ',text.lower()) ## lowercase and punctuations
    text=text.strip()

    tokens=word_tokenize(text)

    ##stopwords
    token=[lemmatizer.lemmatize(word) 
           for word in tokens
           if word not in stop_words]

    return " ".join(token)

def build_corpus(df):
    corpus=[]
    for i in range(0,len(df)):
        cleaned=preprocess(df["Content"][i])
        corpus.append(cleaned)

    return corpus

if __name__=="__main__":
    df=pd.read_csv("data/merged_data.csv")

    corpus=build_corpus(df=df)

    ## cleaned content

    df["Cleaned_Content"]=corpus

    df=df.drop(columns=["Content"])

    ## save back the cleaned data into new csv

    df.to_csv("data/preprocessed_data.csv",index=False)
    print("New csv saved!")

    
