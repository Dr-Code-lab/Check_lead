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
    """function make new row for appending to top dataframe"""
    rate_data = {'message': [message], 'cosine_similarity': [rate]}
    df_candidate = pd.DataFrame(rate_data)
    return df_candidate


def cos_similarity(df_data, target_message):
    """Find 10 most similar messages one for each
        return TOP 10"""

    df_similarity = get_cos_sim_matrix(df_data)

    cos_sim_top = pd.DataFrame(columns=['message', 'cosine_similarity'])
    for Y_message, row in df_similarity.iterrows():
        if Y_message == target_message:
            for X_message, rate in row.items():
                if Y_message != X_message and rate > 0 and rate < 1:  # rules for filtering
                    if cos_sim_top.shape[0] < 10:
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
                            cos_sim_top = cos_sim_top.head(9)  # get just top-9
                            df_top_candidate = new_top_candidate(X_message, rate)  # make 10th for adding
                            cos_sim_top = (pd.concat([cos_sim_top, df_top_candidate])
                                           .sort_values(by='cosine_similarity',
                                                        ascending=False)  # sort descending by one column
                                           .drop_duplicates())    # drop duplicates by one column
                            del df_top_candidate
    return cos_sim_top

