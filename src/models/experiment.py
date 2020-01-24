"""
.. module:: experiment
   :synopsis: Contains the class used to create the Experiment class

"""
import src.models.methods as mth
from src.models.cssnormaliser import CSSNormaliser
import pandas as pd
import numpy as np
from timeit import default_timer as timer
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from imblearn import FunctionSampler
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
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
    """This class is used for experiment creation.

    """
    def __init__(self, meta_data:pd.DataFrame, estimator,grid:dict = None,estimator_name=None,  train_test_column="group",
                 target_column="target", names = (None, None), train_test_split_method=StratifiedKFold(n_splits=7),
                 css_normalisation=CSSNormaliser(identity=True), validation_method_group: tuple = (None, None),
                 scaler=FunctionTransformer(validate=False), resampler=FunctionSampler(),
                 cv_suffix: str = "_cv",features = None, **kwargs):
        """
        Each instance of this class is a classification experiment which stores all the configurations and the results.
        This object is used to run the experiment procedure, using all the attributes passed or implied by its creation.
        The only parameters necessary for the creation of an experiment are the meta-data Dataframe (properly formated)
        and an estimator (from the default ones).

        .. note::

            A properly formatted meta-data dataframe has the values of the response (target) variable under the column
            `target`, and the values of the variable used to split the dataset into train and test sets (and the
            train into validation sets) under the column `group`. For the purposes of the paper, the meta-data dataframe
            is the wwfdf csv found in `data/processed`. The target values can be either under the column `Water` or
            `River_loc`.

        A minimal example initialising an Experiment object is given below

        example::

            from sklearn.ensemble import RandomForestClassifier
            import pandas as pd
            # project_dir is the path to the project directory
            wwfdf = pd.read_csv(project_dir+"data/processed/wwfdf",index_col = 0)
            # creating the object
            experiment_object = Experiment(meta_data = wwfdf, estimator = RandomForestClassifier,
                                           target_column = "Water",train_test_column = "Area_group")

        :param meta_data: The meta-data dataframe that contains the target, train_test_group and validation_group columns.
            The default names for the columns are "target" for the target variable, and "group" for the train_test_group
            and validation_group
        :param target_column: The column in the meta_data dataframe where the values of the target variable are found.
            default is `target`
        :param train_test_column: The column in the meta_data dataframe where the values of the variable used to split
            the samples into train.and test sets are found. default is `group`
        :param train_test_split_method: StratifiedKFold or GroupKFold
        :param estimator: estimator object,
        :param grid: grid of hyperparameters
        :param estimator_name: name of estimator
        :param css_normalisation: CSSNormalisation object
        :param validation_method_group: A tuple which specifies the method used to split the train set into folds,
                                        and which column of the meta_data is going to be used as the group variable
                                        (StratifiedKFold or GroupKFold,column_name_for_grouping)
        :param scaler: sklearn.preprocessing method
        :param resampler: imblearn.over_sampler method
        :param cv_method: str
            Choose from {"_cv","_bcv","_rcv"}
        :param names: (str,str)
            Names of feature and meta data set
        :param features: pandas DataFrame
            The features to be used for running the experiment or for fitting. If not given then when calling then run
            method of the object, they have to be passed together with meta_data
        :param kwargs:
        """
        self.names= names
        self.target_column = mth.checking_columns(dataframe=meta_data, column=target_column, x = target_column)
        self.y_true = meta_data.loc[:, self.target_column]

        # Get group column if it is present in the meta_data frame. If not, the target column is used
        self.train_test_split_column = mth.checking_columns(meta_data, train_test_column, x=train_test_column,
                                                            handle=lambda x: self.target_column)

        self.css = css_normalisation
        self.train_test_split_method = train_test_split_method
        # IF validation_method_group is None then the validation method is the same as the train-test splitting method
        if not validation_method_group[0]:
            self.validation_method = train_test_split_method
        # Otherwise the validation metho is the first element of the tuple validation_method_group
        else:
            self.validation_method = validation_method_group[0]

        # The validation group column is set to the train-test split column if not specified in the validation_method_group
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
        if estimator_name is  None:
            self.estimator_name = type(self.estimator).__name__
        else:
            self.estimator_name = estimator_name
        self.meta_data = meta_data
        self.features = features
        self.kwargs = kwargs


    def default_grid(self):
        """If a hyperparameter grid is not specified when the experiment is created then this method is used to
        find the appropriate grid. If the same experiment is used but with a different (deafult) classifier then use
        this method to update the grid.

        """
        estimator_name = mth.classifier_to_string.get(type(self.estimator))
        if type(estimator_name) == str:
            self.grid = mth.default_grids.get(estimator_name+self.cv)
        elif estimator_name is None:
            raise Exception("These are the classifiers we have default grids for. If your classifier is not included in these"
                  " then pass a grid to the Experiment instance. {}".format(mth.classifier_to_string.keys()))

    def run(self, X:pd.DataFrame = None,y:pd.DataFrame = None):
        """
        Runs the experiment on the data set using the attributes used to create the object.
        The results are stored as attributes of the object.

        :param X: The features used to train the classifiers. If None, then the features given in the construction of
            the objection will be used
        :param y: The meta-data dataframe used to initiate the experiment object. It's a redundant variable that's only
            left in the package in case the user decides to delete the meta-data attribute before saving the object into
            pickle format in order to conserve space.

        :return: A dictionary with the following keys
            "y_pred": predictions made by classifier
            "best_parameters": best hyper parameters
            "coefficients": coefficients of features if present
            "time": time it takes for the procedure to complete
            "false_samples": which samples where wrongly classified
            "confusion":confusion matrix of
        """
        if X is None:
            if self.features is None:
                raise AttributeError("No features given. Either pass a feature Dataframe to the run method's X variable "
                                     "or to the features object attribute.")
            else:
                X = self.features
        if y is None:
            # We dont check wether this attribute is None since it is requiried in the construction of the object
            y = self.meta_data

        # Getting features, target and group from data_tuple. Also store name to report it at the end
        features, target, train_test_group,validation_group = X, y.loc[:,self.target_column], \
                                                              y.loc[:,self.train_test_split_column],y.loc[:,self.validation_group]

        # Split data set into train-test using the supplied KFold metho and grouping variable
        # TODO: Catch errors that arise from missspecification of groups and n_splits, do it in or loop in exp creation
        train_test_sets = self.train_test_split_method.split(X=features, y=train_test_group, groups=train_test_group)

        # Getting index of samples to be used to determined missclassified samples.
        yindex = target.index

        # Creating variables to store results of experiment
        # self.best_parameters is a list of the best hyperparameters for each test set
        # y_pred are the predictions of the classifier
        # coefficients is a list of coefficients for features if it is supported by the estimator. If not it is a list
        # of None
        # false_samples is a list of the names of falsely classified samples, for easy retrieval (although the same
        # information is contained in the y_pred dataframe)
        self.best_parameters = []
        self.y_pred = pd.DataFrame(index=yindex, data={"predictions": np.zeros(yindex.shape)})
        self.coefficients = []
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
            set_parameters, set_coef = self.fit(x_train=  xtrain,meta_train= ytrain,
                                                validation_group=validation_group.iloc[train_index])
            # Predict class of test set
            set_predictions = self.predict(xtest)

            # Update dataframe of predictions
            self.y_pred.iloc[test_index, 0] = set_predictions

            conf_matrix = metrics.confusion_matrix(ytest, set_predictions)

            self.best_parameters.append(set_parameters)
            # scoring_results.append(metrics.accuracy_score(ytest, predicted))
            self.coefficients.append(set_coef)
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
        self.confusion = metrics.confusion_matrix(target,self.y_pred)
        end = timer()
        self.time = end -start
        dictr = {"y_pred": self.y_pred, "best_parameters": self.best_parameters, "coefficients": self.coefficients, "time": self.time,
                 "false_samples": false_samples,"confusion":self.confusion}
        self.accuracy = metrics.accuracy_score(self.y_true,self.y_pred)
        return dictr

    def fit(self, x_train: pd.DataFrame, meta_train:pd.DataFrame,validation_group = None):
        """
        Trains the model by cross validation using the parameters of the object.
        For this method, the cv generator is taken to be the validation_method. train_test_method which was used
        in the run() method to split the set into train and test is not used in this case.

        :param x_train: samples with features
        :param meta_train: meta data with target and validation columns
        :param validation_group: an array with the validation group values.
            if none is given then the meta_train is indexed to get the target and validation columns. If an array is given
            then the meta_train is taken to be just the target variable that will be used to fit the model
        :return: best set of hyperparameters from crossvalidation and the feature coefficients
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
        xtrain = self.css.fit_transform(xtrain)
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
            xtest = self.css.transform(xtest)
            xtest = self.scaler.transform(xtest)
            return self.model.predict(xtest)
        except AttributeError as e:
            print("You have to fit the model first before predicting.\n",
                  "To do this first run Experiment(settings).fit(train_features,targets)\n",
                  "and then Experiment.predict(test_features)")
            raise e

    def return_dictionary(self):
        """
        returns some parameters of the instance in a dictionary
        :return:
        """
        parameters = self.__dict__
        parameters_to_print = ["estimator_name", "css", "scaler", "resampler", "names", "target_column",
                               "cv", "train_test_split_method", "train_test_split_column", "validation_method",
                               "validation_group", "estimator"]
        self.parameters_dictionary = {i: parameters[i] for i in parameters_to_print}
        return self.parameters_dictionary

    def __str__(self):
        return(str(self.return_dictionary()))



