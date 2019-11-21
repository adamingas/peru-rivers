import methods as mth
from cssnormaliser import CSSNormaliser
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from imblearn import FunctionSampler
from sklearn.exceptions import NotFittedError
# import scipy.stats as st
#
# from sklearn.svm import SVC
# from sklearn.base import BaseEstimator
# import pymc3 as pm
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb
# from sklearn.neighbors import KNeighborsClassifier
# import sklearn.metrics as metrics
# import copy
# import prettyconfusion as prt
# from sklearn.pipeline import Pipeline

# self organising maps
# from minisom import MiniSom
# import SimpSOM as sps
# import re
# import csv
# from sklearn.naive_bayes import MultinomialNB,ComplementNB,GaussianNB,BernoulliNB
# from scipy.spatial.distance import cdist

# Sampling methods
# from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN,BorderlineSMOTE,SVMSMOTE
# from imblearn.pipeline import Pipeline
# Bayesian optimisation
# from hyperopt import Trials,STATUS_OK,fmin,tpe,hp
# from hyperopt import base
# base.have_bson = False





class Experiment():
    def __init__(self, meta_data:pd.DataFrame, estimator,grid:dict = None, draws=100,  train_test_column="group",
                 target_column="target", names = (None, None), train_test_split_method=StratifiedKFold(n_splits=7),
                 css_normalisation=CSSNormaliser(identity=True), validation_method_group: tuple = (None, None),
                 scaler=FunctionTransformer(validate=False), resampler=FunctionSampler(),
                 cv_suffix: str = "_cv", **kwargs):
        """
        Each instance of this class is a classification experiment
        which stores all the configurations and the results.
        This object can be used to run a hypothesis by supplying it with X,y
        TODO: Fit such that model is usable for prediction, check if model was fitted before prediction
        :param meta_data:
            The meta data set that contains the target, train_test_group and validation_group columns
        :param train_test_split_method: StratifiedKFold or GroupKFold
        :param models: (estimator object,grid of hyperparameters)
        :param css_normalisation: CSSNormalisation object
        :param validation_method_group: (StratifiedKFold or GroupKFold,column_name_for_grouping)
        :param scaler: sklearn.preprocessing method
        :param resampler: imblearn.over_sampler method
        :param cv_method: str
            Choose from {"_cv","_bcv","_rcv"}
        :param names: (str,str)
            Names of feature and meta data set
        :param kwargs:
        """
        self.names= names
        self.target_column = mth.checking_columns(dataframe=meta_data, column=target_column, x = target_column)
        try:
            # Get group column
            self.train_test_split_column = mth.checking_columns(meta_data, train_test_column, x=train_test_column)
        except KeyError:
            # If it doesnt exist assign target column as group column as well
            self.train_test_split_column = self.target_column
        self.css = css_normalisation
        self.train_test_split_method = train_test_split_method
        if not validation_method_group[0]:
            self.validation_method = train_test_split_method
        else:
            self.validation_method = validation_method_group[0]

        self.validation_group = mth.checking_columns(dataframe=meta_data, column=validation_method_group[1],
                                                     x= validation_method_group[1], handle=lambda x: self.train_test_split_column)

        self.scaler = scaler
        self.resampler = resampler
        self.cv = cv_suffix
        self.grid = grid
        self.estimator = estimator
        # In the case the user interacts directly with this class but they don't want to choose a grid themselves
        # or want the default
        if self.grid is None or type(self.grid) == str and self.grid.lower() == "default":
            self.default_grid()
        self.kwargs = kwargs

    def default_grid(self):
        estimator_name = mth.classifier_to_string.get(type(self.estimator))
        if type(estimator_name) == str:
            self.grid = mth.default_grids.get(estimator_name+self.cv)
        elif estimator_name is None:
            raise Exception("These are the classifiers we have default grids for. If your classifier is not included in these"
                  " then pass a grid to the Experiment instance. {}".format(mth.classifier_to_string.keys()))

    def run(self, X:pd.DataFrame,y:pd.DataFrame):
        """
        Runs the experiment on the data set using the parameters used to initialise the object.
        The results are stored as attributes of the object.
        :return:
        """

        # Getting features, target and group from data_tuple. Also store name to report it at the end
        features, target, train_test_group,validation_group = X, y.loc[:,self.target_column], \
                                                              y.loc[:,self.train_test_split_column],y.loc[:,self.validation_group]

        # Split data set into train-test using the supplied KFold metho and grouping variable
        # TODO: Catch errors that arise from missspecification of groups and n_splits, do it in or loop in exp creation
        train_test_sets = self.train_test_split_method.split(X=features, y=train_test_group, groups=train_test_group)

        # Getting index of samples to be used to determined missclassified samples.
        yindex = target.index

        # Creating variables to store results of experiment
        # best_parameters is a list of the best hyperparameters for each test set
        # y_pred are the predictions of the classifier
        # coefficients is a list of coefficients for features if it is supported by the estimator. If not it is a list
        # of None
        # false_samples is a list of the names of falsely classified samples, for easy retrieval (although the same
        # information is contained in the y_pred dataframe)
        best_parameters = []
        self.y_pred = pd.DataFrame(index=yindex, data={"predictions": np.zeros(yindex.shape)})
        coefficients = []
        false_samples = []

        # TODO: Calculate confusion from y_pred and target
        confusion = np.zeros((np.unique(target).shape[0], np.unique(target).shape[0]))

        # Timing the procedure
        start = timer()

        for i, index in enumerate(train_test_sets):
            train_index, test_index = index
            xtrain, xtest = features.iloc[train_index], features.iloc[test_index]
            ytrain, ytest = target.iloc[train_index], target.iloc[test_index]

            # Perform grid CV using Kfolds as folds.
            # set_parameters are best hyperparameters of this set
            # set_coef are feature coefficients for this set
            set_parameters, set_coef = self.fit(x_train=  xtrain,meta_train= ytrain,validation_group=validation_group.iloc[train_index])
            # Predict class of test set
            set_predictions = self.predict(xtest)
            # Update dataframe of predictions
            self.y_pred.iloc[test_index, 0] = set_predictions

            conf_matrix = metrics.confusion_matrix(ytest, set_predictions)

            best_parameters.append(set_parameters)
            # scoring_results.append(metrics.accuracy_score(ytest, predicted))
            coefficients.append(set_coef)
            if conf_matrix.shape == (1, 1):
                if all(ytest == 0) or all(ytest == "Black"):
                    conf_matrix = np.array([[conf_matrix.item(), 0], [0, 0]])
                else:
                    conf_matrix = np.array([[0, 0], [0, conf_matrix.item()]])
            print(conf_matrix)
            confusion += conf_matrix

            # Checking which samples where wrongly predicted
            false_samples += yindex[test_index[ytest != set_predictions]].tolist()
        # TODO: Compare confusion at the end with confusion
        self.confusion_at_the_end = metrics.confusion_matrix(target,self.y_pred)
        end = timer()
        time = end -start
        dictr = {"y_pred": self.y_pred, "best_parameters": best_parameters, "coefficients": coefficients, "time": time,
                 "false_samples": false_samples,"confusion":confusion,"confusion_at_the_end":self.confusion_at_the_end}
        return dictr

    def fit(self, x_train: pd.DataFrame, meta_train:pd.DataFrame,validation_group = None):
        """
        Trains the model by cross validation using the parameters of the object.
        For this method, the cv generator is taken to be the validation_method. train_test_method which was used
        un the run() method to split the set into train and test is not used in this case.
        :param x_train: samples with features
        :param meta_train: meta data with target and validation columns
        :param validation_group: an array with the validation group values.
            if none is given then the meta_train is indexed to get the target and validation columns. If an array is given
            then the meta_train is taken to be just the target variable that will be used to fit the model
        :return:
        A trained model that can be used to predict labels of a new data set
        """
        np.random.seed(11235)

        if validation_group is None:
            target, set_for_kfold = meta_train.loc[:, self.target_column], meta_train.loc[:,self.validation_group]
        else:
            target = meta_train
            set_for_kfold = validation_group
        xtrain, ytrain = self.resampler.fit_resample(X=x_train, y=target)
        if type(self.resampler) != type(FunctionSampler()):
            # we change set_for_kfold to ytrain if we resampled ytrain since it no longer has the same shape as
            # the validation_group variable. This means that resampling can't work when the foldgenerator used to split data
            # into validation folds is set to GrxoupKFold and the validation_group is not equal to the targets. This is
            # because we don't know how to resample another variable other than the target. eg if target is water colour
            # and validation_group is the area number, we wouldn't be able to resample the water colour AND find what
            # area number the new samples would take/
            set_for_kfold = ytrain
        # Scaling the train set
        xtrain = self.scaler.fit_transform(xtrain)

        # Splitting the train set into validation folds that will be used in training
        validation_sets = self.validation_method.split(xtrain, y=set_for_kfold, groups=set_for_kfold)

        # Perform grid CV using Kfolds as folds.
        self.model = mth.CV_models(grid=self.grid, estimator=self.estimator, parameter_search=self.cv,**self.kwargs)
        # The estimator object is an sklearn classifier
        set_parameters, estimator, set_coef = self.model.fit(features=xtrain, target=ytrain, ksets=validation_sets)

        return set_parameters,set_coef

    def predict(self,xtest):

        # The test sample has to be transformed to the same scale as the train set
        try:
            xtest = self.scaler.transform(xtest)
            return self.model.predict(xtest)
        except AttributeError as e:
            print("You have to fit the model first before predicting.\n",
                  "To do this first run Experiment(settings).fit(train_features,targets)\n",
                  "and then Experiment.predict(test_features)")
            raise e

    def __str__(self):
        return(str(self.__dict__))


if __name__ == "__main__":
    path_to_data = "../../data/processed/"
    riverdf = pd.read_csv(path_to_data + "riverdf", index_col=0)
    wwfdf = pd.read_csv(filepath_or_buffer=path_to_data + "wwfdf", encoding="ISO-8859-1", index_col="ID")

    rfr_grid = {
        'max_depth': (list(range(3, 7)) + [None]),
        # 'n_estimators': [100, 300, 500, 1000],
        # "min_samples_split": [2, 3, 4],
        # "class_weight": ["balanced"],
        "bootstrap": [False]
    }
    testExperiment = Experiment(model_name="RFR", css=CSSNormaliser(log=True), fold_generator=StratifiedKFold,
                                x=riverdf, meta_data=wwfdf.Water, scaler=None, resampler=None,
                                cv=mth.rfr_cv, grid=rfr_grid, penalty="l1")

    testExperiment.run()
