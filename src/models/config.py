import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import  hp
from sklearn.ensemble import RandomForestClassifier

# Data set requirements
# Two distinct sets, one with features and the other with the target variables.
# Have to have the same number of samples. Name of samples must be on the 0th column
# Target column has to be named "target" or the name should be given
# Group column should be named "group" or be given. Otherwise "target" column is used with StratifiedKFold


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

"""
A sample dictionary used to run the water as labels experiments. We are using the dissimilarity case"""
water_dictionary = { # [M] Mandatory, [O] Optional
    "Data": [{"features":"riverdf", "target":"wwfdf","target_col": "Water","train_test_group":"Area_group"}], # [M] What set of features and target data sets to use
    # [O] The target_col is the column where the labels can be found, default "target"
    # [O] Train_test_group is the column used to split the set to train and
    # test, default "group" if it exists, if not then "target" column
    "train_test_split_method": [{"name":"GroupKFold","n_splits":7}], # [O] How to generate train-test splits, default "StratifiedKFold
    "models": [
        {
            "estimators":[ {"name":"MultinomialNB"},{"name": "RandomForest"}],
            "css": [ "CSSLOG","CSS",None],
            "scaler": [None],
            "resampler":[None],
            "validation":[{"name":"GroupKFold","group_col":"Area_group","n_splits":6}]
        },
        {
            "estimators": [{"name":"SVM","kernel":"poly","degree":1}],
            "css": ["CSSLOG", "CSS", None],

            "scaler": [{"name":"StandardScaler","with_mean":False}],
            "resampler":[None],
            "validation":[{"name":"GroupKFold","group_col":"Area_group","n_splits":6}]
        }
    ]
}

river_loc = { # [M] Mandatory, [O] Optional
    "Data": [{"features":"riverdf", "target":"wwfdf","target_col": "River_loc"},
    {"features":"fulldf", "target":"wwfdf","target_col": "River_loc"}], # [M] What set of features and target data sets to use
    # [O] The target_col is the column where the labels can be found, default "target"
    # [O] Train_test_group is the column used to split the set to train and
    # test, default "group" if it exists, if not then "target" column
    "train_test_split_method": [{"name":"StratifiedKFold","n_splits":7}], # [O] How to generate train-test splits, default "StratifiedKFold
    "models": [
        {
            "estimators":[ {"name":"MultinomialNB"},{"name": "RandomForest","random_state":11235}],
            "css": [ "CSSLOG","CSS",None],
            "scaler": [None],
            "resampler":[None,{"name":"RandomOverSampler","random_state":11235}],
            "validation":[{"name":"StratifiedKFold","n_splits":6}]
        },
        {
            "estimators": [{"name":"SVM","kernel":"poly","degree":1}],
            "css": ["CSSLOG", "CSS", None],

            "scaler": [{"name":"StandardScaler","with_mean":False}],
            "resampler":[None,{"name":"RandomOverSampler","random_state":11235}],
            "validation":[{"name":"StratifiedKFold","n_splits":6}]
        }
    ]
}

river_size = { # [M] Mandatory, [O] Optional
    "Data": [{"features":"riverdf", "target":"wwfdf","target_col": "River_size"},
    {"features":"fulldf", "target":"wwfdf","target_col": "River_size"}], # [M] What set of features and target data sets to use
    # [O] The target_col is the column where the labels can be found, default "target"
    # [O] Train_test_group is the column used to split the set to train and
    # test, default "group" if it exists, if not then "target" column
    "train_test_split_method": [{"name":"StratifiedKFold","n_splits":7}], # [O] How to generate train-test splits, default "StratifiedKFold
    "models": [
        {
            "estimators":[ {"name":"MultinomialNB"},{"name": "RandomForest","random_state":11235}],
            "css": [ "CSSLOG","CSS",None],
            "scaler": [None],
            "resampler":[None,{"name":"RandomOverSampler","random_state":11235}],
            "validation":[{"name":"StratifiedKFold","n_splits":6}]
        },
        {
            "estimators": [{"name":"SVM","kernel":"poly","degree":1}],
            "css": ["CSSLOG", "CSS", None],
            "scaler": [{"name":"StandardScaler","with_mean":False}],
            "resampler":[None, {"name":"RandomOverSampler","random_state":11235}],
            "validation":[{"name":"StratifiedKFold","n_splits":6}]
        }
    ]
}
"""
The program will loop through all dictionaries in the hypothesis list and execute them
"""
hypothesis = [water_dictionary, river_size, river_loc]
experiment_dictionary = [{ # [M] Mandatory, [O] Optional
    "Data": [{"features":"riverdf", "target":"wwfdf","target_col": "Trip","train_test_group":"Trip"}], # [M] What set of features and target data sets to use
    # [O] The target_col is the column where the labels can be found, default "target"
    # [O] Train_test_group is the column used to split the set to train and
    # test, default "group" if it exists, if not then "target" column
    "train_test_split_method": [{"name":"StratifiedKFold","n_splits":7}], # [O] How to generate train-test splits, default "StratifiedKFold
    "number_of_folds":[7], # [O] Number of train-test folds
    "models": [{# Inside here is information used for the training step of the procedure
        "estimators": [{"name":"RandomForest","cv":"random","n_iter":100}],#, {"name":"LogisticRegression","penalty":"l2","fit_intercept":True},
                       #{"name":"SVM","kernel":"poly","degree":1}],# [M] Models to use for classification. Custom models
        # can also be passed and custom grid spaces.
        "resampler": [None],# [O] If not given, None is used
        "css": [ "CSSLOG"], # [O] If not given, None is used
        # [O] The hyperparameter search  If not given, Grid is used possible options are {Grid, Bayes, Random}
        "scaler": [None], # [O] Scaler to use on data, default None
        "validation":[{"name":"GroupKFold","group_col":"Water"}],
        "draws":[100]
        # [O] How to split train set to validation folds and which column of the meta data to use. Default value is
        # the train_test splitting method and the train_test column.
    },
        {
            "estimators":[{"name":"Pipeline","method":Pipeline([("resampler",RandomOverSampler()),
                                                                ("RFR",RandomForestClassifier())]),"grid":space,"n_iter":2,"cv":"bayes"}]
        }]
}]
complicated_dictionary = [{ # [M] Mandatory, [O] Optional
    "Data": [{"features":"riverdf", "target":"wwfdf","target_col": "Water","train_test_group":"Area_group"},
             {"features":"riverdf", "target":"wwfdf","target_col": "Water","train_test_group":"ID_nosamples"}], # [M] What set of features and target data sets to use
    # [O] The target_col is the column where the labels can be found, default "target"
    # [O] Train_test_group is the column used to split the set to train and
    # test, default "group" if it exists, if not then "target" column
    "train_test_split_method": ["GroupKFold"], # [O] How to generate train-test splits, default "StratifiedKFold
    "number_of_folds":[7], # [O] Number of train-test folds
    "models": [{# Inside here is information used for the training step of the procedure
        "estimators": [{"name":"RandomForest","cv":"grid"}, {"name":"LogisticRegression","penalty":"l2","fit_intercept":True},
                       {"name":"SVM","kernel":"poly","degree":1}],# [M] Models to use for classification. Custom models
        # can also be passed and custom grid spaces.
        "resampler": [{"name": "RandomOverSampler","katialo":34, "random_state": 11235}],# [O] If not given, None is used
        "css": [None, "CSS", "CSSLOG"], # [O] If not given, None is used
       # [O] The hyperparameter search  If not given, Grid is used possible options are {Grid, Bayes, Random}
        "scaler": [None, {"name": "StandardScaler", "with_mean": False}], # [O] Scaler to use on data, default None
        #"Validation_split_method":["GroupKFold"], # [O] How to split train set to validation folds. default is
        # Train_test_split_method
        #"Validation_group":["ID_nosamples"], # [O] The column of the grouping variable used to split the train set to
        # validation folds, default is Train_test_group
        "validation":[{"method":"GroupKFold","group_col":"Water"}]
    },
        {
            "estimators":[{"name":"Pipeline","method":Pipeline([("resampler",RandomOverSampler()),
                                                                ("RFR",RandomForestClassifier())]),"grid":space}]
        }]
}]

# Experiments are created by taking all possible combinations of options in each experiment dictionary and creating
# an experiment object out of each combination.
# TODO: describe which fields/keys are used for the product
# DEFAULT GRIDS

