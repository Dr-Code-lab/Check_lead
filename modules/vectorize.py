import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def vectorize(df_for_vectorizing):
    """Function make vectors from text"""
    corpus = df_for_vectorizing['processed'].apply(lambda row: ' '.join(row))
    index_s = df_for_vectorizing['message']

    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(corpus).toarray()
    bag_of_words = pd.DataFrame(data=bow_matrix,
                                index=index_s,
                                columns=bow_vectorizer.get_feature_names_out())
    return bag_of_words

