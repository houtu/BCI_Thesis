"""
python -m benchmark.ml.run
"""
import mne
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# dataset
from flex2023 import Flex2023_moabb
from formulate import Formulate
# ml
from pipeline import Pipeline_ML


LIST_FILTER_BANK = ([8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32])


def ml_flex(
        #list_subjects = [15,16,18,20,22,24], #G3
        #list_subjects = [25,26,27,29], # G2
        #list_subjects = [10,11,12,13], #G1
        list_subjects = [15,16,18,20,25,26,27,29,10,11,12,13],
        list_run = [1,2],
        list_t_mi = [(0,2),(1,3),(2,4)],
        list_method = ["SparseFBCSP+LDA",],
        list_model_name = ["MI_2class_hand"],
        #channels = ("C3", "Cz", "C4"),
        channels = ("C3", "Cz", "C4","FC1","FC2","FC5","FC6"),
        # channels = ('Fz', 'F3', 'F4',
        #             'Fp1','Fp2','F7','F8'),
        # channels = ('Cz', 'Fz', 'Fp1', 'F7', 'F3', 
        #             'FC1', 'C3', 'FC5', 'FT9', 'T7', 
        #             'CP5', 'CP1', 'P3', 'P7', 'PO9', 
        #             'O1', 'Pz', 'Oz', 'O2', 'PO10', 
        #             'P8', 'P4', 'CP2', 'CP6', 'T8', 
        #             'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'),
        save_path = "/Users/hutu41./IU/Thesis/Result/channels_Central.csv"
    ):
    """
    
    """

    ## Dataset
    dataset = Flex2023_moabb()
    fs = 128

    ## log
    df = pd.DataFrame()
    j = 0

    for subject in list_subjects:
        for run in list_run:
            if subject < 12:
                print(subject)
                run_to_split = run
            else:
                run_to_split = None
            dataset.runs = run

            for t_mi in list_t_mi:
                for method in list_method:
                    for model_name in list_model_name:

                        # data
                        if "FBCSP" in method:
                            bandpass = LIST_FILTER_BANK
                        elif "alphaCSP" in method:
                            bandpass = [[8,13]]
                        else:
                            bandpass = [[8,30]]
                        
                        f = Formulate(dataset, fs, subject, 
                                    bandpass=bandpass,
                                    channels=channels,
                                    t_rest=(-4,-2),
                                    t_mi=t_mi,
                                    run_to_split=run_to_split,
                                    )
                        x, y = f.form(model_name)
                        _,count = np.unique(y, return_counts=True)

                        ##
                        p = Pipeline_ML(method=method)
                        _df = p.run(x, y)
                    

                        # save
                        df.loc[j, "subject"] = subject
                        df.loc[j, "runs"] = run
                        df.loc[j, "trials/class"] = f"{count.mean():.1f}"
                        df.loc[j, "t_mi"] = str([j+4 for j in t_mi]) # the event start at 4s
                        df.loc[j, "method"] = method
                        df.loc[j, "model_name"] = model_name
                        df.loc[j, "channels"] = str(channels)
                        df.loc[j, "score"] = _df['score'].mean()
                        j += 1

                        print(df)
                        print(df['score'].mean())
                        if save_path is not None:
                            df.to_csv(save_path)




# def plot(path_df):
#     """
#     catplot visualization
#     """

#     df = pd.read_csv(path_df)
#     g = sns.catplot(
#         data=df,
#         x="t_mi",
#         y="score",
#         hue="method",
#         row="model_name",
#         # row="channels",
#         kind="bar",
#         # palette="viridis", 
#         height=5, aspect=3,
#     )
#     # iterate through axes
#     for ax in g.axes.ravel():
#         # add annotations
#         for c in ax.containers:
#             labels = [f"{v.get_height():.2f}" for v in c]
#             ax.bar_label(c, labels=labels, label_type='edge')
#         ax.margins(y=0.2)
#         plt.setp(ax.get_xticklabels(), visible=True)

#     plt.ylim((0,1))
#     plt.show()



#############
if __name__ == "__main__":
    start_time = time.time()
    ml_flex()  # Call your function here
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to run ml_flex(): {elapsed_time} seconds")
    
