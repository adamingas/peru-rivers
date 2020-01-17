'''
The default grids used to replicate the experiments of the paper.
Each grid is the space of hyperparamters over which the cross validation method sarches over to find the most optimal
set

'''
import numpy as np
from hyperopt import  hp


default_grids = {
    "RandomForest_cv" : {
            'max_depth': (list(range(3, 7)) + [None]),
            'n_estimators': [100, 300, 500, 1000],
            "min_samples_split": [2, 3, 4],
            "class_weight": ["balanced"],
            "bootstrap": [False]
        },
    "LogisticRegression_cv":{
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 20, 50] + list(np.arange(2, 20, 1))

                ,'fit_intercept': [True]
            },
    "SVM_cv":{"kernel":["poly"],"gamma":["auto","scale",0.1,1,10],"class_weight":["balanced",None],
                    'C':[0.1, 1, 0.5], "degree":[1,2],"coef0" :[0,0.1,1,0.001]},
    "SOM_cv":{
                "x": [8, 10],
                'sigma': [1, 2, 0.5],
                'learning_rate': [0.1],
                "neighborhood_function": ['gaussian', 'mexican_hat', 'bubble', 'triangle']
            },
    "KNN_cv":{
                "n_neighbors": range(1, 30),
                # ,"braycurtis"],
                "weights": ["uniform", "distance"],
                "p": range(1, 6)
            },
    "XGB_cv":{
                'max_depth': range(1, 3),
                'n_estimators': [500, 600],
                "min_child_weight": range(1, 3)
                # "reg_alpha":[0,0.3,0.7,1],
                # "reg_lambda":[0,0.3,0.7,1]

            },
    "RandomForest_rcv" : {
            'max_depth': (list(range(3, 7)) + [None]),
            'n_estimators': [100, 300, 500, 1000],
            "min_samples_split": [2, 3, 4],
            "class_weight": ["balanced"],
            "bootstrap": [False]
        },
    "LogisticRegression_rcv":{
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 20, 50] + list(np.arange(2, 20, 1))

                ,'fit_intercept': (True)
            },
    "SVM_rcv":{"kernel":["poly"],"gamma":["auto","scale",0.1,1,10],"class_weight":["balanced",None],
                    'C':[0.1, 1, 0.5], "degree":[1,2],"coef0" :[0,0.1,1,0.001]},

}
# adding grids for naive bayes classifiers
default_grids.update(dict.fromkeys(["MultinomialNB_cv","MultinomialNB_rcv","BernoulliNB_cv","BernoulliNB_rcv",
                                    "ComplementNB_cv","ComplementNB_rcv"],{
            "alpha":np.linspace(1e-10,1,num=50),
            "fit_prior":[True,False]}))

# An example of a Bayesian grid
space = {
            'RFR__max_depth': hp.choice('RFR__max_depth', list(range(3, 7)) + [None]),
            'RFR__n_estimators': hp.choice("RFR__n_estimators", [100, 300, 500, 1000]),
            "RFR__min_samples_split": hp.choice("RFR__min_samples_split", [2, 3, 4]),
            "RFR__class_weight": hp.choice("RFR__class_weight", ["balanced"]),
            "RFR__bootstrap": hp.choice("RFR__bootstrap", [False])
            # "resampler__k_neighbors": hp.choice("resampler__k_neighbors", list(np.arange(1, 9))),
            # "resampler__m_neighbors": hp.choice("resampler__m_neighbors", list(np.arange(8, 13))),
            #
            # "resampler__kind": hp.choice("resampler__kind", ["borderline-1", "borderline-2"])
        }
