import pandas as pd
import os

def merge_datasets():
    true_df=pd.read_csv('News_dataset/True.csv')
    Fake_df=pd.read_csv('News_dataset/Fake.csv')

    print(f"True news shape: {true_df.shape}")
    print(f"Fake news shape: {Fake_df.shape}")


    true_df["label"]=0
    Fake_df["label"]=1

    ## merging 2 dataset
    df=pd.concat([true_df,Fake_df], ignore_index=True)


    ## merging 2 columns title and text
    df["Content"]=df["title"]+" "+df["text"]
    df=df[["Content","label"]]
    print(df.tail())

    ## merged dataset have first all 0s and then all 1s, we have to shuffle
    df=df.sample(frac=1,random_state=42).reset_index(drop=True)
    print(df.head())

    ## making seperate folder to save merge data
    os.makedirs("data",exist_ok=True)
    df.to_csv("data/merged_data.csv",index=False)
    return df







if __name__ == "__main__":
    merge_datasets()
