
"""
Data Formulation for FLEX DATASET

======================
Authors: Cuong Pham
cuongquocpham151@gmail.com

"""
import numpy as np
from moabb.paradigms import MotorImagery, FilterBankMotorImagery
from sklearn.preprocessing import LabelEncoder


################################
class Formulate():
    def __init__(
        self, 
        dataset = None, 
        fs = 128, 
        subject = 1, 
        bandpass = [[8,13]],
        channels = ("C3", "Cz", "C4"),
        t_rest = (-2,0),
        t_mi = (0,2),
        run_to_split = None,
        ):
        """
        Usage:
            dataset = Flex2023_moabb()
            dataset.runs = 1
            f = Formulate(dataset, fs=128, subject=1, 
                        bandpass = [[8,13]],
                        channels = ("C3", "Cz", "C4"),
                        t_rest = (-4,-2),
                        t_mi = (0,2),
                        run_to_split=None,
                        )
            x, y = f.form(model_name="MI_2class_hand")
        """
        self.dataset = dataset
        self.fs = fs
        self.subject = subject
        self.bandpass = bandpass
        self.channels = channels
        self.t_rest = t_rest
        self.t_mi = t_mi
        self.run_to_split = run_to_split

        self.event_ids_all = dataset.event_id
        self.event_ids_hand = dict(right_hand=1, left_hand=2)
        self.event_ids_foot = dict(right_foot=3, left_foot=4)


    #-----------------------------------#
    def _extract_split_run(self, event_ids, interval):
        """ 
        Function to split run for data that has run combined 
        Usage for F10, F11 only 
        """

        x, y = self._extract("xy", self.event_ids_all, interval)

        if self.subject == 11: # 60trial x 3run
            split = [range(0, 60), range(60, 120), range(120, 180)]
        elif self.subject == 10: # 72trial x3run
            split = [range(0, 72), range(72, 144), range(144, 216)]

        # split for each run
        idx_r = self.run_to_split - 1
        x = x[split[idx_r]]
        y = y[split[idx_r]]

        # split for tasks
        idx_t = [i for i,v in enumerate(y) \
            if v in list(event_ids.keys())]
        x = x[idx_t]
        y = y[idx_t]

        return x, y


    #-----------------------------------#
    def _extract(self, returns:str, event_ids:dict, interval:tuple):
        """
        Get data/epochs
        """
        if self.bandpass is None:
            paradigm = MotorImagery(
                    events = list(event_ids.keys()),
                    n_classes = len(event_ids.keys()),
                    fmin = 0, 
                    fmax = self.fs/2-0.001, 
                    tmin = interval[0], 
                    tmax = interval[1], 
                    channels=self.channels,
                    resample=128,
                    )

        elif len(self.bandpass) == 1:
            paradigm = MotorImagery(
                    events = list(event_ids.keys()),
                    n_classes = len(event_ids.keys()),
                    fmin = self.bandpass[0][0], 
                    fmax = self.bandpass[0][1],
                    tmin = interval[0], 
                    tmax = interval[1], 
                    channels=self.channels,
                    resample=128,
                    )
        
        elif len(self.bandpass) > 1:
            paradigm = FilterBankMotorImagery(
                    filters=self.bandpass,
                    events = list(event_ids.keys()),
                    n_classes = len(event_ids.keys()),
                    tmin = interval[0],
                    tmax = interval[1],
                    channels=self.channels,
                    resample=128,
                    )

        if returns == "epochs":
            # do not use epochs.event in this case
            epochs,_,_ = paradigm.get_data(dataset=self.dataset,
                    subjects=[self.subject], return_epochs=True)
            return epochs

        elif returns == "xy":
            x,y,_ = paradigm.get_data(dataset=self.dataset,
                    subjects=[self.subject])
            return x, y

        
    #-----------------------------------#
    def form_binary(self)->None:
        """ get data for Rest/NoRest classifier"""

        x_rest, _ = self._extract("xy", self.event_ids_all, self.t_rest)
        x_mi, _ = self._extract("xy", self.event_ids_all, self.t_mi)

        y_rest = np.zeros(x_rest.shape[0])
        y_mi = np.ones(x_mi.shape[0])

        x = np.concatenate((x_rest, x_mi))
        y = np.concatenate((y_rest, y_mi))

        return x, y

    #-----------------------------------#
    def form_mi_all(self)->None:
        """ get data for MI model (4class), checked with moabb evaluation """

        if (self.bandpass is not None) and (len(self.bandpass) > 1): 
            x, y = self._extract("xy", self.event_ids_all, self.t_mi)
            le = LabelEncoder()
            y = le.fit_transform(y)

        else:
            epochs = self._extract("epochs", self.event_ids_all, self.t_mi)
            x = epochs.get_data(units='uV')
            y = epochs.events[:,-1]
        return x, y


    #-----------------------------------#
    def form_mi_2class_hand(self)->None:
        """ get data for MI model (LH/RH) """

        if self.run_to_split is not None:
            x, y = self._extract_split_run(self.event_ids_hand, self.t_mi)
        else:
            x, y = self._extract("xy", self.event_ids_hand, self.t_mi)
        
        le = LabelEncoder()
        y = le.fit_transform(y)
        return x, y


    def form_mi_2class_foot(self)->None:
        """ get data for MI model (LF/RF) """

        if self.run_to_split is not None:
            x, y = self._extract_split_run(self.event_ids_foot, self.t_mi)
        else:
            x, y = self._extract("xy", self.event_ids_foot, self.t_mi)

        le = LabelEncoder()
        y = le.fit_transform(y)
        return x, y

    #-----------------------------------#
    def form_mi_2class_hand_foot(self)->None:
        """ get data for MI model (hand/foot) """
        
        if self.run_to_split is not None:
            x1, _ = self._extract_split_run(self.event_ids_hand, self.t_mi)
            x2, _ = self._extract_split_run(self.event_ids_foot, self.t_mi)
        else:
            x1,_ = self._extract("xy", self.event_ids_hand, self.t_mi)
            x2,_ = self._extract("xy", self.event_ids_foot, self.t_mi)

        y1 = np.zeros(x1.shape[0],)
        y2 = np.ones(x2.shape[0],)
        
        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        le = LabelEncoder()
        y = le.fit_transform(y)
        return x, y


    #-----------------------------------#
    def form_global(self)->None:
        """ get data for MI model (8 class) """

        x1, y_global = self._extract("xy", self.event_ids_all, (0, 2))
        y1 = [i[:-2] if "_r" in i else i for i in y_global]
        
        x2,_ = self._extract("xy", self.event_ids_all, (2.5, 4))
        y2 = ["rest" if "_r" in i else "no_rest" for i in y_global]

        # # debug
        # for i in range(len(y_global)):
        # 	print(f"index {i} | global: {y_global[i]}, local1: {y1[i]}, local2: {y2[i]}")

        le1 = LabelEncoder(); y1 = le1.fit_transform(y1)
        le2 = LabelEncoder(); y2 = le2.fit_transform(y2)
        le = LabelEncoder(); y_global = le.fit_transform(y_global)

        # print(x1.shape)
        # print(y1.shape)
        a,b = np.unique(y1, return_counts=True)
        print(f"y1: {y1.shape} | unique: {[(i,v) for (i,v) in zip(a,b)]}")

        # print(x2.shape)
        # print(y2.shape)
        a,b = np.unique(y2, return_counts=True)
        print(f"y2: {y2.shape} | unique: {[(i,v) for (i,v) in zip(a,b)]}")
        
        a,b = np.unique(y_global, return_counts=True)
        print(f"y_global: {y_global.shape} | unique: {[(i,v) for (i,v) in zip(a,b)]}")

        data = (x1,y1,x2,y2,y_global, le1, le2, le)
        return data



    #-----------------------------------#
    def form(self, model_name:str) -> None:
        """ caller """

        if model_name == "MI_2class_hand":
            x, y = self.form_mi_2class_hand()
        elif model_name == "MI_2class_foot":
            x, y = self.form_mi_2class_foot()
        elif model_name == "MI_2class_hand_foot":
            x, y = self.form_mi_2class_hand_foot()
        elif model_name == "MI_all":
            x, y = self.form_mi_all()
        elif model_name == "Rest/NoRest":
            x, y = self.form_binary()

        print(f"\n({model_name}) | x: {x.shape}, y: {y.shape}")
        a,b = np.unique(y, return_counts=True)
        print(f"({model_name}) | unique: {[(i,v) for (i,v) in zip(a,b)]}")

        return x, y

	
