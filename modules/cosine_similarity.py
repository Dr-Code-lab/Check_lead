import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_cos_sim_matrix(df_vectors):
    """Making matrix of cosine_similarity for dataset's vectors"""
    cos_sim = cosine_similarity(df_vectors.values)  # here we make calculation
    df_cos_sim_matrix = pd.DataFrame(cos_sim,
                                     columns=df_vectors.index.values,
                                     index=df_vectors.index)  # here we make matrix
    return df_cos_sim_matrix


def new_top_candidate(message, rate):
    """Function make new row for appending to top dataframe"""
    rate_data = {'message': [message], 'cosine_similarity': [rate]}
    df_candidate = pd.DataFrame(rate_data)
    return df_candidate


def calc_average(target, cos_sim_top, name):
    return pd.DataFrame.from_dict({'message' : target,
                                   f"""{name}_sim_avg""" : [cos_sim_top.mean(numeric_only=True)['cosine_similarity']]})


def cos_similarity(df_data, target_msg_lst, name):
    """Find most similar messages one for each
    return TOP"""
    total_top = None
    for target in target_msg_lst:
        df_similarity = get_cos_sim_matrix(df_data)
        cos_sim_top = pd.DataFrame(columns=['message', 'cosine_similarity'])
        for Y_message, row in df_similarity.iterrows():
            if Y_message == target:
                for X_message, rate in row.items():
                    if Y_message != X_message and 0 < rate < 1:  # rules for filtering
                        if cos_sim_top.shape[0] < 3:
                            df_top_candidate = new_top_candidate(X_message, rate)
                            if not cos_sim_top.shape[0]:
                                cos_sim_top = df_top_candidate
                            else:
                                cos_sim_top = (pd.concat([cos_sim_top, df_top_candidate])
                                               .sort_values(by='cosine_similarity', ascending=False))
                            del df_top_candidate
                        else:
                            min_rate = cos_sim_top.cosine_similarity.min()
                            if min_rate < rate:
                                cos_sim_top = cos_sim_top.head(2)  # get just top-1
                                df_top_candidate = new_top_candidate(X_message, rate)  # make row for adding
                                cos_sim_top = (pd.concat([cos_sim_top, df_top_candidate])
                                               .sort_values(by='cosine_similarity',
                                                            ascending=False)  # sort descending by one column
                                               .drop_duplicates())  # drop duplicates by one column
                                del df_top_candidate
        if cos_sim_top.shape[0]:
            result = calc_average(target, cos_sim_top, name)
            if total_top is None:
                total_top = result
            else:
                total_top = pd.concat([total_top, result])
    return total_top
