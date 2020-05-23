import pandas as pd

from category_encoders.target_encoder import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class Tfidf:
    def __init__(self, **kwargs):
        self.vectorizers = {}
        self.feature_names = {}

        for key, value in kwargs.items():
            if isinstance(value, str):
                if value.startswith("(") and value.endswith(")"):
                    kwargs[key] = eval(value)
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


class TargetEncoding:
    def __init__(self, **kwargs):
        self.encoders = {}
        self.kwargs = kwargs

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        columns = X.columns
        encoded = pd.DataFrame(index=X.index, columns=X.columns)
        for column in columns:
            self.encoders[column] = TargetEncoder(**self.kwargs)
            encoded[column] = self.encoders[column].fit_transform(X[column], y)
        return encoded

    def transform(self, X: pd.DataFrame):
        encoded = pd.DataFrame(index=X.index, columns=X.columns)
        for column, encoder in self.encoders.items():
            encoded[column] = encoder.transform(X[column])
        return encoded


class ApplyNothing:
    def __init__(self, **kwargs):
        pass

    def fit_transform(self, X: pd.DataFrame):
        return self.transform(X)

    def transform(self, X: pd.DataFrame):
        return X
