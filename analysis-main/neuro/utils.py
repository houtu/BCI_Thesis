import os
import pandas as pd


def get_data(method="SparseFBCSP+LDA"):
    """

    """
    list_t_mi = ['[4.0, 6.0]', '[6.0, 8.0]']
    df_all = []
    for group in ["G1", "G2", "G3"]:

        # read df
        df = pd.read_csv(f"data/df_benchmark_v2_{group}.csv")
        
        # crop time seg, method
        df = df.loc[df["method"] == method] \
                .loc[df["t_mi"].isin(list_t_mi)]
        
        # assign group
        df["group"] = group
        df["runs"] = ["run"+str(int(j)) for j in df["runs"]]

        # append
        df_all.append(df.iloc[:,1:])
    
    df_all = pd.concat(df_all, ignore_index=True)

    return df_all