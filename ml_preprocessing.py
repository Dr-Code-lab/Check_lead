import string
import re
from nltk.corpus import stopwords
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


def drop_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U00000000-\U00000009"
                               u"\U0000000B-\U0000001F"
                               u"\U00000080-\U00000400"
                               u"\U00000402-\U0000040F"
                               u"\U00000450-\U00000450"
                               u"\U00000452-\U0010FFFF"
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1F2-\U0001F1F4"  # Macau flag
                               u"\U0001F1E6-\U0001F1FF"  # flags
                               u"\U0001F600-\U0001F64F"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F1F2"
                               u"\U0001F1F4"
                               u"\U0001F620"
                               u"\u200d"
                               u"\u2640-\u2642"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


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
    text = text.replace('”', ' ').replace('“', ' ').replace('«', ' ').replace('»', ' ').replace('\n', ' ')
    return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)


def drop_over_spaces(text):
    text = ' '.join(text.split())
    return re.sub(r"\s{2,}", ' ', text)


def get_stopwords():
    my_file = open("data/stopwords-ru.txt", "r")
    data = my_file.read()
    stop_words = []
    stop_words = data.split("\n")
    return stop_words


def cleaning_data(data, msg_column):
    df_data = data.dropna(subset=[msg_column]).copy()
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_emoji(x))
    """Function fix contractions"""
    df_data[msg_column] = df_data[msg_column].apply(lambda x: contractions.fix(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_hashtags_and_mentions(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_url(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_ticks_and_nextone(x))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_punctuations(x))
    ru_stopwords_nltk = stopwords.words('russian')
    ru_stopwords_my = get_stopwords()
    ru_stopwords = ru_stopwords_my + ru_stopwords_nltk
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_stopwords(x, ru_stopwords))
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_numbers(x))
    """Function is converting Dataframe to lower case"""
    df_data[msg_column] = df_data[msg_column].apply(lambda x: x.lower())
    df_data[msg_column] = df_data[msg_column].apply(lambda x: drop_over_spaces(x))
    """Function is makeing Dataframe with unique rows"""
    # df_data.drop_duplicates(subset=msg_column)
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


def navecizing(token_list, navec):
    max_len = 125
    unk = navec['<unk>']
    text_embeddings = []
    for token in token_list:
        embedding = navec.get(token, unk)
        text_embeddings.append(embedding)
    total_len = len(text_embeddings)
    if total_len > max_len:
        text_embeddings = text_embeddings[:max_len]
    else:
        text_embeddings.extend([navec['<pad>']] * (max_len - total_len))
    return text_embeddings
