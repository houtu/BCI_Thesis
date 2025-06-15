"""
Motor Imagery dataset conducted at International University (VNU-HCM)

=================
Author: Cuong Pham
cuongquocpham151@gmail.com

"""
import os
import numpy as np
import pandas as pd
import mne
from moabb.datasets.base import BaseDataset


#=========================#
## CONFIG
ROOT = "/home/pham/bci/DATASET/FLEX"
SESSION = "A"
LIST_SUBJECTS =  list(range(10, 50))
ALL_EVENTS_IDS = dict(right_hand=1, left_hand=2, right_foot=3, left_foot=4)
EEG_CH_NAMES = [
    'Cz', 'Fz', 'Fp1', 'F7', 'F3', 
    'FC1', 'C3', 'FC5', 'FT9', 'T7', 
    'CP5', 'CP1', 'P3', 'P7', 'PO9', 
    'O1', 'Pz', 'Oz', 'O2', 'PO10', 
    'P8', 'P4', 'CP2', 'CP6', 'T8', 
    'FT10', 'FC6', 'C4', 'FC2', 'F4', 
    'F8', 'Fp2'
]
FS = 128



#=========================#
class Flex2023_moabb(BaseDataset):
    """Motor Imagery moabb dataset
    adapt to BrainConnects project (compare 3 groups)"""

    def __init__(self):
        super().__init__(
            subjects=LIST_SUBJECTS,
            sessions_per_subject=1,
            events= ALL_EVENTS_IDS,
            code="Flex2023",
            interval=[4, 8], # events at 4s
            paradigm="imagery",
            doi="",
        )
        self.runs = -1
    
    def _flow(self, raw0, stim):

        ## get eeg (32,N)
        data = raw0.get_data(picks=EEG_CH_NAMES)

        # stack eeg (32,N) with stim (1,N) => (32, N)
        data = np.vstack([data, stim.reshape(1,-1)])

        ch_types = ["eeg"]*32 + ["stim"]
        ch_names = EEG_CH_NAMES + ["Stim"]
        info = mne.create_info(ch_names=ch_names, 
                                ch_types=ch_types, 
                                sfreq=FS)
        raw = mne.io.RawArray(data=data, info=info, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        # raw.set_eeg_reference(ref_channels="average")
        
        return raw


    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""

        # path
        list_edf = self.data_path(subject)
        if subject in [10,11] or self.runs == -1:  
            list_edf_select = list_edf
        else:
            list_edf_select = [p for p in list_edf if f"run{self.runs}" in p]

        # concat runs 
        list_raw = []
        for _edf in list_edf_select:
            raw0 = mne.io.read_raw_edf(_edf, preload=False)
            stim = get_stim_data(raw0, subject)
            raw_run = self._flow(raw0, stim)
            list_raw.append(raw_run)
        raw = mne.concatenate_raws(list_raw)

        return {"0": {"0": raw}}


    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        list_edf = []
        for root, dirs, files in os.walk(ROOT):
            for file in files:
                if file.endswith(".edf") and \
                    (f"F{subject}_{SESSION}" in file) and \
                    (".md" not in file):
                        list_edf.append(os.path.join(root, file))
        return list_edf



#=========================#
def get_stim_data(edf_raw, subject):

    #--------------#
    if subject == 10:
        for root, dirs, files in os.walk(ROOT):
            for file in files:
                if file.endswith(".csv") and ("intervalMarker" in file) and ("F10_A" in file):
                        path_csv = os.path.join(root, file)
        return fix_stim(edf_raw, path_csv=path_csv)

    #--------------#
    elif subject == 11:
        return fix_stim(edf_raw, path_csv=None)

    #--------------#
    else:
        return edf_raw.get_data(picks=["MarkerValueInt"], units='uV')[0]
    

#=========================#
def fix_stim(edf_raw, path_csv=None):
    """ fix stim for old procedure (F10, F11)"""

    ## get markers (biocalib+kines+ mi). This is the event of trial.
    markerIndex = edf_raw.get_data(picks=["MarkerIndex"], units='uV')[0]
    markers = np.where(markerIndex != 0)[0] # (320,)

    ## stim fix (because we assign value0)
    markerValueInt = edf_raw.get_data(picks=["MarkerValueInt"], units='uV')[0]
    stim = np.zeros_like(markerValueInt)

    ## Use edited.csv (for F10 only because a few trials get wrong markerValueInt)
    if path_csv is not None:
        df = pd.read_csv(path_csv)

    for i, value in enumerate(markers):
        # get the base to be offset
        if path_csv is not None:
            base = df.loc[i, "marker_value"]
        else:
            base = markerValueInt[value]

        # 120 biocalib, offset = 20
        if 0 <= i < 120:
            offset = 20
            stim[value] = base + offset
        
        # 20 kines, offset = 10
        elif 120 <= i < 140:
            offset = 10
            stim[value] = base + offset
        
        # 140 MI, 1,2,3,4
        else:
            if base == 0:
                stim[value] = 1
            else:
                stim[value] = base + 1

    # ## unique check
    # a,b = np.unique(stim, return_counts=True)
    # print([(i,v) for i,v in zip(a,b)])

    # ## plot check
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3, 1, sharex=True) 
    # ax[0].plot(markerIndex)
    # ax[0].set_title("markerIndex")
    # ax[1].plot(markerValueInt, label="markerValueInt")
    # ax[1].set_title("markerValueInt")
    # ax[2].plot(stim, label="markerValueInt_fixed")
    # ax[2].set_title("markerValueInt_fixed")
    # plt.show()

    return stim

