import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


class Tfidf:
    def __init__(self, **kwargs):
        self.vectorizers = {}
        self.feature_names = {}
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame):
        columns = X.columns
        vectors = {}
        for column in columns:
            self.vectorizers[column] = TfidfVectorizer(**self.kwargs)
            vectorized = self.vectorizers[column].fit_transform(X[column])
            self.feature_names[column] = self.vectorizers[
                column].get_feature_names()

            vectors[column] = vectorized
        return vectors

    def transform(self, X: pd.DataFrame):
        vectors = {}
        for column, vectorizer in self.vectorizers.items():
            vectorized = vectorizer.transform(X[column])
            vectors[column] = vectorized
        return vectors
