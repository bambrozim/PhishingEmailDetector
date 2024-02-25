import pandas as pd
import numpy as np

# nlp
import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# machine learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD

class PreProcessEmail(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp_ = spacy.load('en_core_web_sm')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cleaned_emails = X['Email Text'].str.replace('\n', ' ').str.replace(r'\s\\', '')
        nlp_docs = [self.nlp_(doc) for doc in cleaned_emails]
        return nlp_docs

class GetTokenStatistics(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tokens_statistics = [
            (
                len(doc), # how many tokens
                len([token for token in doc if token.is_alpha]), # how many words
                len([token for token in doc if (token.is_alpha) and (len(wn.synsets(token.text)) > 0)]), # how many recognizable words
                len([token for token in doc if token.is_punct]), # how many special chars tokens
                len([token for token in doc if token.like_email]), # how many email tokens
                len([token for token in doc if token.like_url]) # how many url tokens
            ) for doc in X
        ]
        
        tokens_stats_df = pd.DataFrame(tokens_statistics, columns=['tokens_len', 'tokens_words', 'tokens_real_words', 'tokens_special', 'tokens_email', 'tokens_url'])
        
        # augmenting base dataframe
        new_X = pd.DataFrame()
        new_X['tokens_len'] = tokens_stats_df['tokens_len'].values
        new_X['tokens_words_perc'] = np.where(tokens_stats_df['tokens_len'] > 0, (tokens_stats_df['tokens_words'] / tokens_stats_df['tokens_len']), 0)
        new_X['tokens_real_words_perc'] = (tokens_stats_df['tokens_real_words'] / tokens_stats_df['tokens_words']).values
        new_X['tokens_special_perc'] = (tokens_stats_df['tokens_special'] / tokens_stats_df['tokens_len']).values
        new_X['has_email'] = (tokens_stats_df['tokens_email'] > 0).values
        new_X['has_url'] = (tokens_stats_df['tokens_url'] > 0).values

        self.new_X_ = new_X.fillna(0)

        return new_X

    def get_feature_names_out(self, X, y=None):
        return self.new_X_.columns

class NMFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = Pipeline([('tf-idf', TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')), ('nmf', NMF(n_components=6, random_state=42))])

    def _preprocess_email(self, doc):
        return " ".join([token.lemma_ for token in doc if (not token.is_stop) and (token.is_alpha) and (len(token) > 2)])
    
    def fit(self, X, y=None):

        processed_emails = [self._preprocess_email(doc) for doc in X]
        
        self.pipeline.fit(processed_emails)
        return self

    def transform(self, X, y=None):
        processed_emails = [self._preprocess_email(doc) for doc in X]
        
        nmf_topics = self.pipeline.transform(processed_emails)
        
        new_X = pd.DataFrame()
        new_X['email_text'] = [doc.text for doc in X]
        new_X['topic'] = [doc.argmax() for doc in nmf_topics]

        self.new_X_ = new_X

        return new_X

    def get_feature_names_out(self, X, y=None):
        return self.new_X_.columns

class DataFrameFeatureUnion(TransformerMixin, BaseEstimator):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (_, transformer) in self.transformer_list:
            transformer.fit(X, y)
        return self

    def transform(self, X):
        X_transformed = pd.concat(
            [transformer.transform(X) for _, transformer in self.transformer_list],
            axis=1
        )
        return X_transformed

    def get_feature_names_out(self):
        feature_names = []
        for name, transformer in self.transformer_list:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out())
            else:
                feature_names.extend(transformer.columns)
        return feature_names