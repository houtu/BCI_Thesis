import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statannot
from .utils import get_data



class STAT:
    def __init__(self, ):
        pass

    
    #---------------------------------------#
    def survey(self, **kwargs):
        """
        Args:
            path_survey (str): path to csv file of questionaries
            path_save (str): path to save figures
        """

        ## CONFIG
        df = pd.read_csv("data/survey_bci2.csv")
        subject_group = {
            "G1": [10, 11, 12, 13], # image
            "G2": [25, 26, 27, 29], # arrow (split run)
            "G3": [15, 16, 18, 20], # arrow + feedback (split run)
            # "G1": [10, 11, 12, 13], # image
            # "G2+G3": [25, 26, 27, 29, 15, 16, 18, 20], # arrow (split run)
        }
        list_metric = df.columns[2:-1]

        ## process df
        df1 = pd.DataFrame()
        j = 0
        for group in subject_group.keys():
            for subject in subject_group[group]:
                for run in [1,2]:
                    for metric in list_metric:
                        if metric == "Interest": 
                            continue

                        df1.loc[j, "group"] = group
                        df1.loc[j, "subject"] = subject
                        df1.loc[j, "run"] = run
                        df1.loc[j, "metric"] = metric

                        tmp = df.loc[df["ID"] == f"F{subject}"] \
                                .loc[df["Run"] == run]
                        df1.loc[j, "score"] = tmp[metric].values
                        j+=1
        print(df1)
        print(np.unique(df1["metric"]))

        ## PLOT BAR 
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12,5))
        sns.barplot(
            data=df1, 
            ax=axes, 
            x="metric", y="score", hue="group",
            width=0.4,
        )
        axes.set_ylim((0,5))
        axes.set_xlabel("")
        axes.tick_params(axis='both', labelsize=12)
        
        ## STAT ANNOTATION
        # box_pairs = []
        # for metric in list_metric:
        #     if metric == "Interest": 
        #         continue
        #     tmp = [
        #         ((metric, "G1"), (metric, "G2")),
        #         ((metric, "G2"), (metric, "G3")),
        #         ((metric, "G3"), (metric, "G1")),
        #     ]
        #     box_pairs.extend(tmp)

        # stat_test = "Mann-Whitney"
        # statannot.add_stat_annotation(
        #     data=df1, 
        #     ax=axes, 
        #     x="metric", y="score", hue="group",
        #     box_pairs=box_pairs,
        #     test=stat_test,
        #     text_format="star", # simple, 
        #     loc="inside",
        #     verbose=2
        # )

        ## SAVE
        plt.legend(loc='upper right')
        plt.tight_layout()
        if kwargs["path_save"] is not None:
            plt.savefig(kwargs["path_save"])
        plt.close()
        plt.show()


    #---------------------------------------#
    def within_group(self, **kwargs):
        """
        Compare match-run within each group, e.g.,
        2sample x 4subject => 8 sample / group
            Fx_run1_4-6_m1  Fx_run2_4-6_m1
            Fx_run1_6-8_m1  Fx_run2_6-8_m1
            ...
        """
        df = get_data(kwargs["method"])


        ## stats
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
        list_model_name = ["MI_2class_hand", "MI_2class_foot"]
        list_group = ["G1", "G2", "G3"]
        list_palette = ["Reds", "Blues"]
        
        for i, model_name in enumerate(list_model_name):
            for j, group in enumerate(list_group):
            
                ## plot-##
                data = df.loc[df["model_name"]==model_name] \
                            .loc[df["group"] == group]
                ax = axes[i,j]

                box = sns.boxplot(
                    ax=ax, 
                    data=data,
                    x="runs", y="score",
                    width=0.4,
                    showmeans=True,
                    boxprops=dict(linestyle='-', linewidth=2),
                    medianprops=dict(linestyle='-', linewidth=2),
                    whiskerprops=dict(linestyle='--', linewidth=2),
                    capprops=dict(linestyle='-', linewidth=2),
                    meanprops=dict(markerfacecolor='white',markeredgecolor='black', markersize=8),
                    palette=list_palette[i],
                )
                ax.set_ylim((0,1))
                ax.grid(False)
                fontsize = 18
                ax.tick_params(axis='both', labelsize=fontsize)
                ax.set_xlabel("", fontsize=fontsize)
                if (i==0 and j==0) or (i==1 and j==0):
                    ax.set_ylabel("(mean ROC)", fontsize=fontsize)
                ax.set_title(group, fontsize=fontsize, fontweight='bold')

                ## statannot ##
                if (i==0 and j==2):
                    box_pairs=[("run1", "run2")]
                    stat_test = "Wilcoxon"
                    statannot.add_stat_annotation(
                        ax=ax,
                        data=data,
                        x="runs", y="score",
                        box_pairs=box_pairs,
                        test=stat_test,
                        loc="inside",
                        text_format="star", 
                        comparisons_correction=None,
                        verbose=2
                    )

                if (i==0 and j==2) or (i==1 and j==2):
                    ax.legend(handles=[
                        mpatches.Patch(color=sns.color_palette(list_palette[i])[2], 
                                    label=f'[{model_name}] run1'),
                        mpatches.Patch(color=sns.color_palette(list_palette[i])[4], 
                                    label=f'[{model_name}] run2'),
                                        ],
                        loc='lower right', fontsize=12
                    )
        # plt.legend(loc='upper right', fontsize=14)        
        plt.tight_layout()
        if kwargs["path_save"] is not None:
            plt.savefig(kwargs["path_save"])
        plt.close()
        plt.show()


    #---------------------------------------#
    def between_group(self, **kwargs):
        """
        Compare each group using all data
        """
        df = get_data(kwargs["method"])

        ## plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,7))
        list_model_name = ["MI_2class_hand", "MI_2class_foot"]
        
        for i, model_name in enumerate(list_model_name):

            data = df.loc[df["model_name"]==model_name]
            ax = axes[i]
            box = sns.boxplot(
                ax=ax, 
                data=data,
                x="group", y="score", 
                width=0.4,
                showmeans=True,
                meanprops=dict(markerfacecolor='white',markeredgecolor='black', markersize=8),
                boxprops=dict(linestyle='-', linewidth=2),
                medianprops=dict(linestyle='-', linewidth=2),
                whiskerprops=dict(linestyle='--', linewidth=2),
                capprops=dict(linestyle='-', linewidth=2),
            )
            ax.set_ylim((0,1))
            ax.grid(False)

            ## ---------statannot------------##
            if i==0:
                box_pairs=[("G1", "G2"), ("G1", "G3")]
                stat_test = "Mann-Whitney"
                statannot.add_stat_annotation(
                    ax=ax,
                    data=data,
                    x="group", y="score",
                    box_pairs=box_pairs,
                    test=stat_test,
                    loc="inside",
                    text_format="star", 
                    # text_format="full", # simple, 
                    comparisons_correction=None,
                    verbose=2
                )

            fontsize = 18
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_ylabel("(mean ROC)", fontsize=fontsize)
            ax.set_xlabel("", fontsize=fontsize)
            ax.legend(loc='upper right', fontsize=14)
            ax.set_title(model_name, fontsize=fontsize, fontweight=fontsize)
            ax.set_yticklabels([str(i/10) for i in range(0,11,2)], fontsize=fontsize)
            
        plt.tight_layout()
        if kwargs["path_save"] is not None:
            plt.savefig(kwargs["path_save"])
        plt.close()
        plt.show()





    #---------------------------------------#
    def get(self, mode:str, config:dict):
        caller = getattr(self, mode)
        return caller(**config)