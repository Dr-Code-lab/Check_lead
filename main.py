import numpy as np
import pandas as pd
from check_lead_by_cosine_similarity.modules import preprocessing as pre
from check_lead_by_cosine_similarity.modules import cosine_similarity as cs


def concatinate_data(df1, df2):
    return pd.concat([df1, df2])


def parse_syntethic_data():
    synthetic_data = pd.read_excel('data/message_synthetic.xlsx')
    synthetic_data['id'] = 0
    leads = synthetic_data[synthetic_data['mark'] == 1][['id', 'message']]
    not_leads = synthetic_data[synthetic_data['mark'] == 0][['id', 'message']]
    return leads, not_leads

def get_subcategory_setup(project_id, subcategory_id = None):
    subcategory = pd.read_csv('data/freelancer_sub.csv')
    # subcategory = subcategory[subcategory['project_id'] == project_id]
    if subcategory_id:
        subcategory = subcategory[subcategory['id'] == subcategory_id]
    subcategory['stop_words'] = subcategory['stop_words'].apply(lambda words: np.array(words[1:-1].split(',')))
    subcategory['tags'] = subcategory['tags'].apply(lambda words: np.array(words[1:-1].split(',')))
    if subcategory_id:
        subcategory_stopwords = np.array(subcategory['stop_words'].values[0])
        subcategory_tags = np.array(subcategory['tags'].values[0])
    else:
        subcategory_stopwords = np.array(np.concatenate(subcategory['stop_words']))
        subcategory_tags = np.array(np.concatenate(subcategory['tags']))
    subcategory_stopwords = [text.replace("'", "") for text in subcategory_stopwords]
    subcategory_tags = [text.replace("'", "") for text in subcategory_tags]
    return subcategory_stopwords, subcategory_tags


def data_prepare(target_message_lst, df_data):
    """Function for read files and prepare dataframe"""
    df_target = None
    for target_message in target_message_lst:
        if df_target is None:
            df_target = pd.DataFrame.from_dict({'id': [-1], 'message': [f"""{target_message}"""],})
        else:
            df_target = concatinate_data(df_target, pd.DataFrame.from_dict({'id': [-1], 'message': [f"""{target_message}"""],}))

    # df_leads = df_data.query("nontarget_count < target_count")
    df_nonleads = df_data.query("nontarget_count > 0 and target_count == 0")
    df_trueleads = df_data.query("target_count > 0 and nontarget_count == 0")
    # df_nontargeted = df_data.query("target_count == nontarget_count")

    leads = concatinate_data(df_target, df_trueleads[['id', 'message']])
    not_leads = concatinate_data(df_target, df_nonleads[['id', 'message']])

    # synthetic_leads, synthetic_not_leads = parse_syntethic_data()
    # leads = concatinate_data(leads, synthetic_leads)
    # not_leads = concatinate_data(not_leads, synthetic_not_leads)

    return leads[['id', 'message']], not_leads[['id', 'message']]


def preprocessing(df_data):
    df_data_pured = pre.cleaning_data(df_data, 'message')
    df_data_pured['preprocessed'] = (df_data_pured['message']
                                     .apply(lambda message: pre.tokenization(message))
                                     .apply(lambda msg_tokens: pre.stemming(msg_tokens))
                                     .apply(lambda msg_tokens: pre.lemmatization(msg_tokens)))
    df_vectors = pre.vectorize_tf_idf(df_data_pured, 'preprocessed', 'message')   # here we make vectors from tokenized text
    return df_vectors


def stopwords_check(target_message_lst, stopwords_arr, tags_arr):
    # Later should return dataframe with stopped messages
    filter_lst = []
    for target_message in target_message_lst:
        flag = 1
        for tag in tags_arr:
            if tag in f"""{target_message}""":
                for stopword in stopwords_arr:
                    if stopword in f"""{target_message}""":
                        print(target_message, '\nstopword: ', stopword, 'STOP, it is not lead') # Delete before release
                        break
                    else:
                        flag = 0
        if flag == 0:
            filter_lst.append(target_message)
    return filter_lst


def check_result(leads, not_leads):     # Test
    if leads > not_leads:
        return "It is LEAD"
    else:
        return "NO, it's NOT LEAD"


def main(project_id, df_target_message, subcategory_id = None):
    """Function check if target message is lead"""
    df_data = pd.read_csv('data/freelancer.csv')
    target_message_lst = np.array(df_target_message['message'])

    stop_words, tags = get_subcategory_setup(project_id, subcategory_id)

    target_message_lst = stopwords_check(target_message_lst, stop_words, tags)

    df_leads, df_nonleads = data_prepare(target_message_lst, df_data)

    df_dict = {'df_leads': df_leads, 'df_nonleads': df_nonleads}
    result = None
    INPUT_SIZE = len(target_message_lst)

    for _, key in enumerate(df_dict):
        df_vectors = preprocessing(df_dict.get(key))
        target_msg_cleaned = df_vectors.index[0:INPUT_SIZE]
        df_similar_avg = cs.cos_similarity(df_vectors, target_msg_cleaned, key)
        if result is None:
            result = df_similar_avg
        else:
            if result.shape[0] >= df_similar_avg.shape[0]:
                merge_side = 'left'
            else:
                merge_side = 'right'
            result = result.merge(df_similar_avg, how=merge_side, on='message').fillna(0)
    result['delta'] = 0
    result['lead_mark'] = None
    result['delta'] = result.apply(lambda x: x['df_leads_sim_avg'] - x['df_nonleads_sim_avg'], axis=1)
    result['lead_mark'] = result.apply(lambda x: "LEAD" if x['delta'] > 0 else "NONLEAD", axis=1)
    return result


if __name__ == '__main__':
    df = pd.read_csv('data/freelancer.csv')
    df_nontargeted = df.query("target_count == nontarget_count")


msgs = df_nontargeted[::25].head(6)
print(main(4, msgs))
#%%

#%%
