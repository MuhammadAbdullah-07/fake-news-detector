import pickle
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from src.features import get_features

def train():
    X,y=get_features()
    # Splitting features (X) and target labels (y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2,       # 20% of data goes to the test set, 80% to training
        random_state=42,     # Controls the shuffling for reproducible results
        stratify=y           # Ensures class proportions are identical in both sets
    )
    print(X_train.shape)
    print(X_test.shape)

    # ── Standard Scaling ──────────────────────────────────────
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)


    # ── Pipelines ──────────────────────────────────────
    pipeline={
        "logisticRegression":Pipeline([
            ("model",LogisticRegression())
            ]),
        "RandomForest":Pipeline([
            ("model",RandomForestClassifier())
            ]),
        "GaussianNb":Pipeline([
            ("model",GaussianNB())
            ])}
    
    # ── Parameters ─────────────────────────────────────
    params={
           "logisticRegression":{
               "model__C": [0.1, 1, 10],
               "model__solver": ["lbfgs", "liblinear"],
               "model__max_iter": [100, 200]
           },
           "RandomForest":{
               "model__n_estimators": [100, 200],
               "model__max_depth": [10, 20, None]
               },
               "GaussianNb":{
                   "model__var_smoothing": [1e-9, 1e-8, 1e-7]
               }
               }
    
# ── Storing back best data ───────────────────────────────────
    results={}
    for name in pipeline:
        grid=GridSearchCV(
            pipeline[name],
            params[name],
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        y_pred = grid.best_estimator_.predict(X_test)

    # ── Calculating accuracy & classification report ───────────────────
        acc_score=accuracy_score(y_test,y_pred)
        Classification_Report=classification_report(y_test,y_pred)

        results[name]={
            "best_params":grid.best_params_,
            "pipeline":grid.best_estimator_,
            "accuracy":acc_score,
            "best_score":grid.best_score_,
            "classification_Report":Classification_Report
        }
        print(f"{name} → Accuracy: {acc_score:.4f}")
        print(f"{name} → classification_Report: {Classification_Report}")

    # ── Best Model ─────────────────────────────────────
    best_name = max(results, key=lambda x: results[x]["accuracy"])
    best_pipeline = results[best_name]["pipeline"]

    print(f"\nBest Model: {best_name}")
    print(f"Accuracy:   {results[best_name]['accuracy']:.4f}")

    # ── Save ───────────────────────────────────────────

    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nModel saved  → models/model.pkl")
    print("Scaler saved → models/scaler.pkl")        




if __name__=="__main__":
    train()
