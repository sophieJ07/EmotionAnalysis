import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.svm import LinearSVC
import numpy as np
from preprocess import preprocess_text

# Load dataset
data = pd.read_csv('data/tweets.csv', header=0, nrows=300000)

emotion_labels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

# Preprocess 
data['processed_text'] = data.iloc[:, 0].apply(preprocess_text)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['processed_text'])
y = data.iloc[:, 1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train 
model = LinearSVC(multi_class='crammer_singer', random_state=42)
model.fit(X_train, y_train)
