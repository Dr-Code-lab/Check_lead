import numpy as np
import pandas as pd
from navec import Navec
import joblib
from ml_preprocessing import cleaning_data, tokenization, stemming, lemmatization, navecizing


def get_subcategory_setup(project_id=None, subcategory_id=None):
    subcategory = pd.read_csv('data/subcats.csv')                                   #TODO Set source
    if project_id:
        subcategory = subcategory[subcategory['project_id'] == project_id]
    if subcategory_id:
        subcategory = subcategory[subcategory['id'] == subcategory_id]
    subcategory['stop_words'] = subcategory['stop_words'].apply(lambda words: np.array(words[1:-1].split(', ')))
    subcategory['tags'] = subcategory['tags'].apply(lambda words: np.array(words[1:-1].split(', ')))
    if subcategory_id:
        subcategory_stopwords = np.array(subcategory['stop_words'].values[0])
        subcategory_tags = np.array(subcategory['tags'].values[0])
    else:
        subcategory_stopwords = np.concatenate(np.array(subcategory['stop_words'].drop_duplicates()))
        subcategory_tags = np.concatenate(np.array(subcategory['tags'].drop_duplicates()))
    subcategory_stopwords = [text.replace("'", "") for text in subcategory_stopwords]
    subcategory_tags = [text.replace("'", "") for text in subcategory_tags]
    subcategory_stopwords = list(filter(len, subcategory_stopwords))
    subcategory_tags = list(filter(len, subcategory_tags))
    return subcategory_stopwords, subcategory_tags


def preprocessing(df_data):
    df_data_preprocessed = cleaning_data(df_data, 'message')
    df_data_preprocessed['preprocessed'] = (df_data_preprocessed['message']
                                            .apply(lambda message: tokenization(message))
                                            .apply(lambda msg_tokens: stemming(msg_tokens))
                                            .apply(lambda msg_tokens: lemmatization(msg_tokens)))
    return df_data_preprocessed


def stopwords_check(target_message, stopwords_arr, tags_arr):
    filtred_lst = []
    stoplist = []
    flag = 0
    # target = target_message.lower()
    # # for tag in tags_arr:
    # #     if tag in f"""{target}""":
    # for stopword in stopwords_arr:
    #     if stopword in f"""{target}""":
    #         stoplist.append(target_message)
    #         flag = 1
    #         break
    if flag == 0:
        filtred_lst.append(target_message)
    return filtred_lst

navec = Navec.load("data/navec_hudlit_v1_12B_500K_300d_100q.tar")
classificator = joblib.load('data/svg976.pkl')
classificator_LR = joblib.load('data/lr979.pkl')
classificator_P = joblib.load('data/p981.pkl')


def main(data_target_message, project_id=None, subcategory_id=None):
    """Function check if target message is lead"""
 
    stop_words, tags = get_subcategory_setup(project_id, subcategory_id)
    targets_msg_list = stopwords_check(data_target_message, stop_words, tags)

    if len(targets_msg_list):
        df_target = pd.DataFrame.from_dict(
            {'target': [tar for tar in targets_msg_list], 'message': [tar for tar in targets_msg_list]})
        df_target_message_normal = preprocessing(df_target)
        df_target_message_normal['embeddings'] = df_target_message_normal.apply(
            lambda row: navecizing(row['preprocessed'], navec), axis=1)
        target_embeddings = [emb for emb in df_target_message_normal['embeddings']]

        for index, msg in enumerate(targets_msg_list):
            embeddings = target_embeddings[index]
            target = np.array(embeddings).reshape(-1)
            prediction = (classificator.predict([target]) ==
                          classificator_LR.predict([target]) ==
                          classificator_P.predict([target]))
            if not prediction:
                mark = 0
            elif prediction:
                mark = 1
            else:
                def next_validation():
                    return str((classificator.predict([target]) + 
                                classificator_LR.predict([target]) + 
                                classificator_P.predict([target])
                                )[0]) + f"/3 SCORE  {data_target_message}"
                mark = next_validation()
        return mark                                 # just for single message !!!!
    else:
        return f"STOPPED {data_target_message}"
