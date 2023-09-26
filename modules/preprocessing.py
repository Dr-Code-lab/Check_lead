import string
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from emot import core
import contractions


def emoticon_transform(text, emot_core):
    """Function replace emoticons with text"""
    emoticons = emot_core.emoticons(text).get('value')
    means = emot_core.emoticons(text).get('mean')
    if emoticons:
        for i, emoticon in enumerate(emoticons):
            text = text.replace(emoticon, ' ' + means[i] + ' ')
    return text


def drop_stopwords(text, lst_stopwords):
    """Function is cleaning Dataframe from english stopwords"""
    tokens_lst = []
    for token in text.split():
        if token.lower() not in lst_stopwords:
            tokens_lst.append(token)

    return ' '.join(tokens_lst)


def drop_hashtags_and_mentions(text):
    """Function is cleaning Dataframe from hashtags and mentions"""
    clean_text = re.sub("@[A-Za-z0-9_]+", "", text)  # replace mention with empty string
    clean_text = re.sub("#[a-zA-Z0-9_]+", "", clean_text)  # replace hashtag with empty string
    return clean_text


def drop_url(text):
    """Function is cleaning Dataframe from URLs"""
    return re.sub('http\S+', '', text)


def drop_ticks_and_nextone(text):
    """Function is cleaning Dataframe from ticks
    and the next character (do it when contractions has fixed only)"""
    return re.sub(r"\'\w+", '', text)


def drop_numbers(text):
    """Function is cleaning Dataframe from numbers"""
    return re.sub(r"\d+", ' ', text)


def drop_punctuations(text):
    """Function is cleaning Dataframe from punctuations"""
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)


def drop_over_spaces(text):
    text = ' '.join(text.split())
    return re.sub(r"\s{2,}", ' ', text)


def cleaning_data(df_data, msg_column):
    """Function fix contractions"""
    df_data[msg_column] = df_data[msg_column].apply(lambda x: contractions.fix(x))
    emot_core = core.emot()
    df_data[msg_column] = df_data[msg_column].apply(lambda x: emoticon_transform(x, emot_core))
    ru_stopwords = stopwords.words('russian')
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_stopwords(x, ru_stopwords))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_hashtags_and_mentions(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_url(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_ticks_and_nextone(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_numbers(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_punctuations(x))
    """Function is converting Dataframe to lower case"""
    df_data[msg_column] = df_data[msg_column].apply(lambda x: x.lower())
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_over_spaces(x))
    """Function is makeing Dataframe with unique rows"""
    df_data.drop_duplicates(subset=msg_column)
    return df_data


def tokenization(message):
    """Just tokenize text - split text to text units"""
    return word_tokenize(message)


def vectorize(df_for_vectorizing, corpus_column, index_column):
    """Function make vectors from text"""
    corpus = df_for_vectorizing[corpus_column].apply(lambda row: ' '.join(row))
    index_s = df_for_vectorizing[index_column]

    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(corpus).toarray()
    bag_of_words = pd.DataFrame(data=bow_matrix,
                                index=index_s,
                                columns=bow_vectorizer.get_feature_names_out())
    return bag_of_words
