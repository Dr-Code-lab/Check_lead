import pandas as pd
from Bot.modules import cleaning_data as cln, vectorize as vec, cosine_similarity as cs, tokenize as tok


def get_data(target_message):
    """Function for read files and prepare dataframe for processing"""
    leads = pd.read_csv('./data/leads.csv')
    not_leads = pd.read_csv('./data/non_leads.csv')
    df_target = pd.DataFrame(data={'id': [-1], 'message': [target_message]})
    leads = pd.concat([df_target, leads])
    not_leads = pd.concat([df_target, not_leads[['id', 'message']]])
    leads['processed'] = None
    not_leads['processed'] = None
    return leads[['id', 'message']], not_leads[['id', 'message']]


def get_cosine_similarity(df_data, target_message):
    df_data_pured = cln.cleaning_data(df_data)
    df_data_pured['processed'] = df_data_pured['message'].apply(lambda x: tok.tokenization(x))
    df_vectors = vec.vectorize(df_data_pured)   # here we make vectors from tokenized text
    return cs.cos_similarity(df_vectors, target_message)


def check_result(leads, not_leads):
    if leads > not_leads:
        print("It is LEAD")
    else:
        print("NO, it's NOT LEAD")


def main(target_message):
    """Function check if target message is lead"""
    try:
        leads_cos_sim_avg = 0
        not_leads_cos_sim_avg = 0
        df_leads, df_not_leads = get_data(target_message)

        df_dict = {'df_leads': df_leads, 'df_not_leads': df_not_leads}
        for _, key in enumerate(df_dict):
            df_top_similar = get_cosine_similarity(df_dict.get(key), target_message)
            if key == 'df_leads':
                leads_cos_sim_avg = df_top_similar.mean(numeric_only=True)['cosine_similarity']
            else:
                not_leads_cos_sim_avg = df_top_similar.mean(numeric_only=True)['cosine_similarity']
        check_result(leads_cos_sim_avg, not_leads_cos_sim_avg)
    except:
        Exception()


if __name__ == '__main__':
    main('добрый день нужен срочно морковка оплата рублей пишите лс подробностями')
#%%
