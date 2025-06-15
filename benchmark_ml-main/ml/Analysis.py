#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:55:15 2024

@author: hutu41.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statannot import add_stat_annotation

# Load the CSV file
csv_files = ['/Users/hutu41./IU/Thesis/Result/channels_Frontal.csv','/Users/hutu41./IU/Thesis/Result/channels_Central.csv','/Users/hutu41./IU/Thesis/Result/channels_Cz.csv']
# Read each CSV file into a DataFrame and store in a list
df_list = [pd.read_csv(file) for file in csv_files]
# Concatenate all DataFrames in the list into a single DataFrame
df = pd.concat(df_list, ignore_index=True)

def perform_analysis(group_by_column):
    # Group data by the specified column and calculate the ANOVA test
    grouped = df.groupby(group_by_column)
    scores_by_group = [group['score'].values for name, group in grouped]

    # Perform ANOVA test
    f_value, p_value = f_oneway(*scores_by_group)
    print(f"ANOVA results for {group_by_column}: F-value = {f_value}, P-value = {p_value}")

    # Visualization with significance annotation
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x=group_by_column, y='score', data=df)
    plt.title(f'Performance Scores by {group_by_column} ')
    plt.xlabel(group_by_column)
    plt.ylabel('Score')

    # Add statistical annotation if p-value is significant
    if p_value < 0.05:
        pairs = [(a, b) for a in df[group_by_column].unique() for b in df[group_by_column].unique() if a < b]
        add_stat_annotation(ax, data=df, x=group_by_column, y='score',
                            box_pairs=pairs,
                            test='t-test_ind', text_format='star', loc='inside', verbose=2)

    plt.show()

    # Interpretation and explanation
    if p_value < 0.05:
        print(f"The ANOVA test shows significant differences between groups for {group_by_column}, meaning that at least one group's mean score is significantly different from the others.")
    else:
        print(f"The ANOVA test does not show significant differences between groups for {group_by_column}, meaning that the differences in mean scores are not statistically significant.")

# Choose the column to group by ('t_mi' or 'method')
group_by_column = 'channels'  # Change this to 'method' if needed
perform_analysis(group_by_column)

