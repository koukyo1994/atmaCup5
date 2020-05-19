import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


class Tfidf:
    def __init__(self, **kwargs):
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.feature_names = []

    def fit_transform(self, X: pd.Series):
        vectorized = self.vectorizer.fit_transform(X)
        self.feature_names = self.vectorizer.get_feature_names()
        return vectorized

    def transform(self, X: pd.Series):
        return self.vectorizer.transform(X)
