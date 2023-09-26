import numpy as np
import pandas as pd
from check_lead_by_cosine_similarity.modules import preprocessing as pre
from check_lead_by_cosine_similarity.modules import cosine_similarity as cs


def get_data(target_message):
    """Function for read files and prepare dataframe for processing"""
    leads = pd.read_csv('data/leads.csv')
    not_leads = pd.read_csv('data/non_leads.csv')
    df_target = pd.DataFrame(data={'id': [-1], 'message': [target_message]})
    leads = pd.concat([df_target, leads])
    not_leads = pd.concat([df_target, not_leads[['id', 'message']]])
    leads['preprocessed'] = None
    not_leads['preprocessed'] = None

    subcategory = pd.read_csv('data/subcategories.csv')
    subcategory['stop_words'] = subcategory['stop_words'].apply(lambda words: np.array(words[1:-1].split(',')))
    subcategory_stopwords = list(set(np.concatenate(subcategory['stop_words'])))#.to_numpy()
    return leads[['id', 'message']], not_leads[['id', 'message']], subcategory_stopwords

def preprocessing(df_data):
    df_data_pured = pre.cleaning_data(df_data, 'message')
    df_data_pured['preprocessed'] = df_data_pured['message'].apply(lambda x: pre.tokenization(x))
    df_vectors = pre.vectorize(df_data_pured, 'preprocessed', 'message')   # here we make vectors from tokenized text
    target_msg_cleaned = df_data_pured['message'].iloc[0]
    return df_vectors, target_msg_cleaned


def check_result(leads, not_leads):     # Test
    if leads > not_leads:
        return "It is LEAD"
    else:
        return "NO, it's NOT LEAD"


def main(target_message):
    """Function check if target message is lead"""
    leads_cos_sim_avg = 0
    not_leads_cos_sim_avg = 0
    df_leads, df_not_leads, stop_words = get_data(target_message)

    df_dict = {'df_leads': df_leads, 'df_not_leads': df_not_leads}
    for _, key in enumerate(df_dict):
        df_vectors, target_msg_cleaned = preprocessing(df_dict.get(key))
        df_top_similar = cs.cos_similarity(df_vectors, target_msg_cleaned)
        if key == 'df_leads':
            leads_cos_sim_avg = df_top_similar.mean(numeric_only=True)['cosine_similarity']
        else:
            not_leads_cos_sim_avg = df_top_similar.mean(numeric_only=True)['cosine_similarity']
    return check_result(leads_cos_sim_avg, not_leads_cos_sim_avg)


if __name__ == '__main__':
    print(main('добрый день нужен срочно морковка рублей пишите'))
#%%

#%%
