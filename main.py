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


def parse_data(project_id, target_message, subcategory_id = None):
    """Function for read files and prepare dataframe"""
    df_target = pd.DataFrame(data={'id': [-1], 'message': [target_message]})
    leads = pd.read_csv('data/leads.csv')
    not_leads = pd.read_csv('data/non_leads.csv')
    leads = concatinate_data(df_target, leads[['id', 'message']])
    not_leads = concatinate_data(df_target, not_leads[['id', 'message']])

    synthetic_leads, synthetic_not_leads = parse_syntethic_data()

    leads = concatinate_data(leads, synthetic_leads)
    not_leads = concatinate_data(not_leads, synthetic_not_leads)

    subcategory = pd.read_csv('data/subcategories.csv')
    subcategory = subcategory[subcategory['project_id'] == project_id]
    if subcategory_id:
        subcategory = subcategory[subcategory['id'] == subcategory_id]
    subcategory['stop_words'] = subcategory['stop_words'].apply(lambda words: np.array(words[1:-1].split(',')))
    if subcategory_id:
        subcategory_stopwords = list(subcategory['stop_words'].values[0])
    else:
        subcategory_stopwords = list(set(np.concatenate(subcategory['stop_words'])))

    return leads[['id', 'message']], not_leads[['id', 'message']], subcategory_stopwords


def preprocessing(df_data):
    df_data_pured = pre.cleaning_data(df_data, 'message')
    df_data_pured['preprocessed'] = (df_data_pured['message']
                                     .apply(lambda msg: pre.stemming(msg))
                                     .apply(lambda msg: pre.lemmatization(msg))
                                     .apply(lambda msg: pre.tokenization(msg)))
    df_vectors = pre.vectorize_tf_idf(df_data_pured, 'preprocessed', 'message')   # here we make vectors from tokenized text
    target_msg_cleaned = df_data_pured['message'].iloc[0]
    return df_vectors, target_msg_cleaned


def stopwords_check(target_message, stopwords_list):
    for stopword in stopwords_list:
        if stopword in target_message:
            print('stopword: ',stopword)
            return 1


def check_result(leads, not_leads):
    if leads > not_leads:
        return "It is LEAD"
    else:
        return "NO, it's NOT LEAD"


def main(project_id, target_message, subcategory_id = None):
    """Function check if target message is lead"""
    leads_cos_sim_avg = 0
    not_leads_cos_sim_avg = 0
    df_leads, df_not_leads, stop_words = parse_data(project_id, target_message, subcategory_id)

    if stopwords_check(target_message, stop_words):
        return "STOP, it's NOT LEAD"

    df_dict = {'df_leads': df_leads, 'df_not_leads': df_not_leads}
    for _, key in enumerate(df_dict):
        df_vectors, target_msg_cleaned = preprocessing(df_dict.get(key))
        df_top_similar = cs.cos_similarity(df_vectors, target_msg_cleaned)
        if key == 'df_leads':
            leads_cos_sim_avg = df_top_similar.mean(numeric_only=True)['cosine_similarity']
        else:
            not_leads_cos_sim_avg = df_top_similar.mean(numeric_only=True)['cosine_similarity']
    print("lead avg:", leads_cos_sim_avg, "\nnot_lead avg:", not_leads_cos_sim_avg)     # удалить перед релизом
    return check_result(leads_cos_sim_avg, not_leads_cos_sim_avg)


if __name__ == '__main__':
    msgs = [
        'хочу арендовать автомобиль',       # 1
        'нужен трансфер в аэропорт москва',     # 1
        'добрый день подскажите нужен срочно хуй ',     # 0
        'Планирую поездку в Японию, ищу компаньона!',       # 0  -  требует доработки
        'Как мне найти людей для совместного путешествия?',     # 0
        'Какие города считаются самыми красивыми в мире?',      # 0
        'Привет! Мне нужно арендовать жилье на неделю отпуска на берегу моря. '
        'Можете мне помочь найти подходящее жилье?',       # 1
        'Ищу яхту для аренды на время отпуска. Можете помочь?',     # 1
        'Привет! Организуй, пожалуйста, тур по Южной Америке для меня.',    # 0
        'Ищу поставщика услуг по прокату автомобилей.',     # 0
        'Ищу бизнес-джет для полета из Нью-Йорка в Лос-Анджелес.',      # 1
        'Добрый день! Ищу хороший салон красоты, чтобы сделать свадебную прическу. Можете посоветовать?',       # 0
        'Планирую путешествие по Италии и ищу жилье для аренды на две недели в центре Рима. Жилье должно быть с '
        'четырьмя спальными комнатами, двумя ванными и террасой.'        # 1

    ]
for msg in msgs:
    print(msg, '\n', main(4, msg), '\n')
#%%

#%%
