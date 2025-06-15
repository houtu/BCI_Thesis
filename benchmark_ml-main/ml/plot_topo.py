#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:13:43 2024

@author: hutu41.
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

dataset = Flex2023_moabb()

def ml_flex(
        #list_groups = [[10,11,12,13],[25,26,27,29],[15,16,18,20]],
        list_subjects = [15,16,18,20,22], #G3
        #list_subjects = [25,26,27,29], # G2
        #list_subjects = [10,11,12,13], #G1
        #list_subjects = [15,16,15],
        list_run = [1,2],
        list_t_mi = [(0,2)],
        list_method = ["alphaCSP",],
        list_model_name = ["MI_all"],
        #channels = ("C3", "Cz", "C4"),
        #channels = ("C3", "Cz", "C4","FC1","FC2","FC5","FC6"),
        # channels = ('Fz', 'F3', 'F4',
        #             'Fp1','Fp2','F7','F8'),
        channels = ('Cz', 'Fz', 'Fp1', 'F7', 'F3', 
                    'FC1', 'C3', 'FC5', 'FT9', 'T7', 
                    'CP5', 'CP1', 'P3', 'P7', 'PO9', 
                    'O1', 'Pz', 'Oz', 'O2', 'PO10', 
                    'P8', 'P4', 'CP2', 'CP6', 'T8', 
                    'FT10', 'FC6', 'C4', 'FC2', 'F4', 'F8', 'Fp2'),
        save_path = None
    ):
    ## Dataset
    dataset = Flex2023_moabb()
    fs = 128
    ## log
    df = pd.DataFrame()
    j = 0
    print(j)
    
    all_x = []
    all_y = []
    
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
                        print(x.shape, y.shape)
                        # Append the data to the lists
                        all_x.append(x)
                        all_y.append(y)
    
    all_x = np.concatenate(all_x, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_y = all_y.astype(np.int32)
    
    print(all_x.shape)
    print(all_y.shape)
    print(all_y)
                        
    # Create an info object
    info = mne.create_info(
        ch_names=list(channels),
        sfreq=fs,  # Sampling frequency
        ch_types='eeg'
    )
    
    
    # Set the standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    
    # Separate the data by label
    unique_labels = np.unique(all_y)
    evoked_dict = {}
    
    for label in unique_labels:
        label_indices = np.where(all_y == label)[0]
        x_label = all_x[label_indices]
        
        # Create MNE Epochs
        events = np.column_stack((np.arange(len(label_indices)), np.zeros(len(label_indices), int), all_y[label_indices]))
        epochs = mne.EpochsArray(x_label, info, events)
        
        # Compute the average (Evoked)
        evoked_dict[label] = epochs.average()
    
    # Select a specific time point (e.g., 2 seconds)
    time_point = 1.0  # 2 seconds
    time_index = int(time_point * fs)
    
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(15, 5))
    
    for ax, (label, evoked) in zip(axes, evoked_dict.items()):
        im, _ = mne.viz.plot_topomap(evoked.data[:, time_index], evoked.info, axes=ax, cmap=None, show=False)
        ax.set_title(f'Label {label}\nTime: {time_point:.2f}s')
    
    
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
   
    #plt.tight_layout()
    plt.show()
                   

    if save_path is not None:
        df.to_csv(save_path)
                            
if __name__ == "__main__":
    ml_flex()  # Call your function here
    
