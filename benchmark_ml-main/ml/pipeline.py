
"""
Machine Learning Pipeline
"""

import numpy as np
import pandas as pd
from moabb.pipelines.utils import FilterBank
from mne.decoding import CSP, UnsupervisedSpatialFilter

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import Lasso
import mrmr
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
# from mvlearn.embed import CCA, MCCA, KMCCA



SEEDS = 42
KFOLD = 10

#=================================#
class METHODS:
    pipeline = {

        "alphaCSP+LDA": Pipeline(steps=[
                ('pca', UnsupervisedSpatialFilter(PCA(), average=False)),
                ('csp', CSP(n_components=8)),
                # ('csp', CSP(n_components=8, reg='ledoit_wolf', log=True)),
                # ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ('lda', LDA())]
            ),
        "alphaCSP+SVM": Pipeline(steps=[
                ('pca', UnsupervisedSpatialFilter(PCA(), average=False)),
                ('csp', CSP(n_components=8)),
                # ('csp', CSP(n_components=8, reg='ledoit_wolf', log=True)),
                # ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ('svm', SVC(kernel='linear', random_state=42, probability=True))]
            ),
        "alphaCSP+LR": Pipeline(steps=[
                ('pca', UnsupervisedSpatialFilter(PCA(), average=False)),
                ('csp', CSP(n_components=8)),
                # ('csp', CSP(n_components=8, reg='ledoit_wolf', log=True)),
                # ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ('TS', TSclassifier())]
            ),
        "alphaCSP+CCA+LDA": None,
        "FBCSP+LDA": None,
        "mrmrFBCSP+LDA": None,
        "SparseFBCSP+LDA": None,

        # #--------#
        "Cov+Tangent+SVM": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('tangent_Space', TangentSpace(metric="riemann")),
                ('svm', SVC(kernel='linear', random_state=42, probability=True))]
            ),

        "Cov+Tangent+LDA": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('tangent_Space', TangentSpace(metric="riemann")),
                ('lda', LDA())],
            ),

        "Cov+Tangent+LR": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('TS', TSclassifier())]
            ),

        "Cov+MDM": Pipeline(steps=[
                ('covariances', Covariances("oas")),
                ('mdm', MDM(metric=dict(mean='riemann', distance='riemann')))]
            ),

    }



#####################################
class Pipeline_ML():
    """
    Run pipeline for (x,y), either one of 3 models 
        MI_2class / MI_all / Rest_NoRest
    """
    def __init__(self, method="CSP+LDA"):
        self.method = method

    def _pipe(self, x_train, y_train, x_test, y_test):
        """ 
        fit for 1 fold. return:
            pipe_ml: trained model on train-set
            r: metrics on test set (binary-> auc_roc | multiclass -> accuracy)
        """
        #------------#
        if "FBCSP" in self.method:
            fbcsp = make_pipeline(FilterBank(CSP(n_components=4)))
            fbcsp.fit(x_train, y_train)
            ft_train = fbcsp.transform(x_train)
            ft_test = fbcsp.transform(x_test) 

            if "Sparse" in self.method:
                # Apply sparse feature selection using Lasso
                clf = Lasso(alpha=0.05,
                            random_state=SEEDS,
                            selection="random")

                clf.fit(ft_train, y_train)
                idx = [i for i,v in enumerate(clf.coef_) if v!=0]
                ft_train_select = ft_train[:, idx]
                ft_test_select = ft_test[:, idx]
            
            elif "mrmr" in self.method:
                selected = mrmr.mrmr_classif(X=pd.DataFrame(ft_train), 
                                            y=pd.Series(y_train), 
                                            K=int(0.5*ft_train.shape[1]))
                ft_train_select = ft_train[:, selected]
                ft_test_select = ft_test[:, selected]
            
            else:
                ft_train_select = ft_train
                ft_test_select = ft_test

            ## train
            pipe_ml = Pipeline(steps=[('classifier', LDA())])
            pipe_ml.fit(ft_train_select, y_train)
            ## test
            if len(np.unique(y_train))==2: # binary
                r = metrics.roc_auc_score(y_test, 
                            pipe_ml.predict_proba(ft_test_select)[:, 1])
            else: # multiclass
                r = metrics.accuracy_score(y_test, 
                            pipe_ml.predict(ft_test_select))
        
        #------------#
        else:
            # ## csp
            # pipe = Pipeline(steps=[('csp', CSP(n_components=8))])
            # pipe.fit(x_train, y_train)
            # _ft_train = pipe.transform(x_train)
            # _ft_test = pipe.transform(x_test)
            # print(_ft_train.shape)
            # print(y_train.shape)

            # from sklearn.preprocessing import OneHotEncoder
            # encoder = OneHotEncoder(sparse_output=False)
            # y_encoded = encoder.fit_transform(y_train.reshape(-1, 1))

            # ## cca
            # cca = CCA(n_components=2)
            # cca.fit([_ft_train, y_encoded])
            # ft_train = cca.transform(_ft_train)
            # ft_test = cca.transform(_ft_test)
            # print(ft_train.shape)
            # ## train
            # pipe_ml = Pipeline(steps=[('classifier', LDA())])
            # pipe_ml.fit(ft_train, y_train)
            # ## test
            # if len(np.unique(y_train))==2: # binary
            #     r = metrics.roc_auc_score(y_test, 
            #                 pipe_ml.predict_proba(ft_test)[:, 1])
            # else: # multiclass
            #     r = metrics.accuracy_score(y_test, 
            #                 pipe_ml.predict(ft_test))

            ## train
            pipe_ml = METHODS.pipeline[self.method]
            pipe_ml.fit(x_train, y_train)
            ## test
            if len(np.unique(y_train))==2: # binary
                r = metrics.roc_auc_score(y_test, 
                            pipe_ml.predict_proba(x_test)[:, 1])
            else: # multiclass
                r = metrics.accuracy_score(y_test, 
                            pipe_ml.predict(x_test))

        return pipe_ml, r




    def run(self, x, y):
        """ kfold cross cv"""

        df = pd.DataFrame()
        skf = StratifiedKFold(n_splits=KFOLD,
                            shuffle=True,
                            random_state=SEEDS)

        for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
            pipe, r = self._pipe(x[train_index],
                                    y[train_index],
                                    x[test_index],
                                    y[test_index],
                                )
            df.loc[f"fold-{fold}", "score"] = r
        
        return df

        

