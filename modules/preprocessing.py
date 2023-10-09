import string
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import contractions


def spell_checker(message):
    pass

def drop_stopwords(text, lst_stopwords):
    """Function is cleaning Dataframe from english stopwords"""
    tokens_lst = []
    for token in text.split():
        if token.lower() not in lst_stopwords:
            tokens_lst.append(token)

    return ' '.join(tokens_lst)


def drop_hashtags_and_mentions(text):
    """Function is cleaning Dataframe from hashtags and mentions"""
    clean_text = re.sub("@[А-Яа-яA-Za-z0-9_]+", "", text)  # replace mention with empty string
    clean_text = re.sub("#[А-Яа-яa-zA-Z0-9_]+", "", clean_text)  # replace hashtag with empty string
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
    text = text.replace('”', ' ').replace('“', ' ').replace('«', ' ').replace('»', ' ')
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)


def drop_over_spaces(text):
    text = ' '.join(text.split())
    return re.sub(r"\s{2,}", ' ', text)


def get_stopwords():
    my_file = open("data/stopwords-ru.txt", "r")
    data = my_file.read()
    stopwords = data.split("\n")
    return stopwords


def cleaning_data(df_data, msg_column):
    """Function fix contractions"""
    df_data[msg_column] = df_data[msg_column].apply(lambda x: contractions.fix(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_hashtags_and_mentions(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_url(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_punctuations(x))
    ru_stopwords = stopwords.words('russian') #get_stopwords()
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_stopwords(x, ru_stopwords))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_ticks_and_nextone(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_numbers(x))
    """Function is converting Dataframe to lower case"""
    df_data[msg_column] = df_data[msg_column].apply(lambda x: x.lower())
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_over_spaces(x))
    """Function is makeing Dataframe with unique rows"""
    df_data.drop_duplicates(subset=msg_column)
    return df_data



def tokenization(message):
    """Just tokenize text - split text to text units"""
    return word_tokenize(message)


def stemming(message_tokens):
    stemmer = PorterStemmer()
    stem_lst = list()
    for word in message_tokens:
        stem = stemmer.stem(word)
        stem_lst.append(stem)
    return stem_lst


def lemmatization(message_tokens):
    lemmatizer = WordNetLemmatizer()
    lemm_lst = list()
    for word in message_tokens:
        lemm = lemmatizer.lemmatize(word)
        lemm_lst.append(lemm)
    return lemm_lst


def vectorize_bow(df_for_vectorizing, corpus_column, index_column):
    """Word count: Bag-of-Words — это статистический анализ,
    анализирующий количественное вхождение слов в документах."""

    corpus = df_for_vectorizing[corpus_column].apply(lambda row: ' '.join(row))
    index_s = df_for_vectorizing[index_column]

    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(corpus).toarray()
    bag_of_words = pd.DataFrame(data=bow_matrix,
                                index=index_s,
                                columns=bow_vectorizer.get_feature_names_out())
    return bag_of_words


def vectorize_tf_idf(df_for_vectorizing, corpus_column, index_column):
    """TF-IDF
    (от англ. TF — term frequency, IDF — inverse document frequency) — статистическая мера,
    используемая для оценки важности слова в контексте документа, являющегося частью коллекции документов или корпуса.
    Вес некоторого слова пропорционален частоте употребления этого слова в документе
    и обратно пропорционален частоте употребления слова во всех документах коллекции."""

    corpus = df_for_vectorizing[corpus_column].apply(lambda row: ' '.join(row))
    index_s = df_for_vectorizing[index_column]

    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_matrix = tf_idf_vectorizer.fit_transform(corpus).todense()
    tf_idf = pd.DataFrame(data=tf_idf_matrix,
                          index = index_s,
                          columns = tf_idf_vectorizer.get_feature_names_out())

    return tf_idf


def doc2vec_model(df_data):
    from gensim.test.utils import common_texts
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]

    """First training Doc2Vec model"""
    d2v_model = Doc2Vec(documents, window=2, min_count=1, workers=4, epochs=12)
    return d2v_model


def doc2vec_vectorize(target_message_tokens, message_list, model):
    return model.infer_vector(target_message_tokens), [model.infer_vector(msg) for msg in message_list]
#%%
