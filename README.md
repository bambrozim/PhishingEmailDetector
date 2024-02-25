# Phishing Email Detector

This project uses Natural Language Processing techniques to classify whether is a phishing email or not based on the content of the email.
It was trained on a dataset available at Kaggle ([Phishing Email Detection](https://www.kaggle.com/datasets/subhajournal/phishingemails)).

The content of the jupyter notebooks is as follows:
- EDA and Feature Engineering: contains the exploratory analysis and feature engineering techniques used to enhance the dataset.
- Training: contains the model training, it outputs a trained model into a pickle file "phishing_detector.pkl".
- Predicting: contains code used to be able to predict any new email to the model and get their classification.

The model presents an ROC AUC score of 99% and a F1 score of 97%.

This project is for research purposes, of course there is much to adapt into a real-life running model.