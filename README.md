# 📰 Fake News Detector

A machine learning web application that detects fake news articles (On American Dataset) using NLP and Word2Vec embeddings, achieving **97.46% accuracy** on 44,889 news articles.

---

## 🚀 Live Demo
> Coming soon — deploying on AWS EC2

---

## 📌 Project Overview

This project builds an end-to-end fake news detection pipeline:
- Text preprocessing using NLTK
- Feature extraction using Word2Vec + AvgWord2Vec
- Model training with GridSearchCV
- Flask web application for real-time prediction

**Current Version:** Trained on US political news dataset (2016-2018)
> 🔄 **In Progress:** Adding Pakistani news dataset support for better coverage

## 🔮 Future Improvements

- [ ] Add Pakistani news dataset (Dawn, Geo, ARY News)
- [ ] Implement LSTM/BiLSTM for better context understanding  
- [ ] Train on WELFake dataset for more diverse coverage
- [ ] Add confidence percentage to predictions
- [ ] Support Urdu language news detection