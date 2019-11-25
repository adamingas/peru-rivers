import numpy as np
import pandas as pd
from config import default_grids
# import pymc3 as pm
# import theano.tensor as tt
# from theano import shared
from sklearn.model_selection import  cross_val_score, GridSearchCV, StratifiedKFold, GroupKFold, \
    RandomizedSearchCV
from timeit import default_timer as timer
import re
from itertools import product
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.preprocessing as prepro
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.naive_bayes import MultinomialNB,ComplementNB,GaussianNB,BernoulliNB

from hyperopt import Trials,STATUS_OK,fmin,tpe,hp
from hyperopt import base
base.have_bson = False

# from sksom import SKSOM
# import xgboost as xgb
model_choices = {"RandomForest": RandomForestClassifier, "LogisticRegression": LogisticRegression, "SVM": SVC}

classifier_to_string = {
    type(RandomForestClassifier()):"RandomForest",
    type(LogisticRegression()):"LogisticRegression",
    type(SVC()):"SVM",
    type(KNeighborsClassifier()):"KNN",
    type(BernoulliNB()):"BernoulliNB",
    type(MultinomialNB()): "MultinomialNB",
    type(ComplementNB()): "ComplementNB"
}
def string_to_key(string: str):
    """
    Processes strings to be read by dictionaries linking to methods
    :param string:
    :return:
    """
    try:
        # Removing all characters not in alphabet and making the string lower case
        return re.compile("[^A-Za-z]").sub("", string).lower()
    except TypeError:
        # If string is actually None we return None
        return None


def check_if_it_can_fit(object):
    if hasattr(object, "fit") and hasattr(object, "predict") and hasattr(object, "get_params") and hasattr(object,
                                                                                                           "set_params"):
        return object
    else:
        raise Exception("Pass an estimator that has methods fit predict set_params get_params")


def select_estimator(estimator_dict):
    """

    :param estimator_dict: dict
        Dictionary containing name of estimator together with kwargs
    :return: Estimator
    """
    if type(estimator_dict) is dict:
        try:
            # Using the string to obtain the appropriate function from the object
            estimator_name = estimator_dict.pop("name")
        except KeyError as e:
            print("Specify a name for your estimator or pass a string as an estimator. e.g. {\"name\":\"RandomForest\","
                  "\"kwargs_of_RandomForest\":values} or \"RandomForest\"")
            raise e
        try:
            estimator = model_choices[estimator_name]()
        except KeyError:
            print(
                "{} is not one of the default estimators. Choose from {}".format(estimator_name, model_choices.keys()))
            print("Using Custom estimator if specified using the \"method\" key")
            try:
                estimator = estimator_dict.pop("method")
            except KeyError as e:
                print("Supply a valid sklearn estimator using the \"method\" key")
                raise e
            check_if_it_can_fit(estimator)
        if estimator_dict.get("cv"):
            cv_suffix = cv_choices.get(string_to_key(estimator_dict.pop("cv")))
        else:
            cv_suffix = "_cv"
        if estimator_dict.get("grid") and type(estimator_dict.get("grid")) is dict:
            print("custom grid for {}".format(estimator_name))
            grid = estimator_dict.pop("grid")
        else:
            try:
                grid = default_grids[estimator_name + cv_suffix]
            except KeyError as e:
                print("No {} grid found".format(estimator_name + cv_suffix))
                raise e
        # Getting all legal parameters
        param_dict = estimator.get_params()
        # intersection of legal parameters and kwargs of object. We do this to avoid errors made by users in specifying
        # method arguments
        intersection = {i: estimator_dict[i] for i in estimator_dict if i in param_dict}
        return {"estimator": estimator.set_params(**intersection), "grid": grid, "cv_suffix": cv_suffix,
                "estimator_name":estimator_name,**estimator_dict}

    elif type(estimator_dict) is str:
        estimator_name = estimator_dict
        try:
            # Using the string to obtain the appropriate function from the object
            estimator = model_choices[estimator_name]()
            grid = default_grids[estimator_name + "_cv"]
            # Getting all legal parameters

            return {"estimator": estimator, "grid": grid, "cv_suffix": "_cv","estimator_name":estimator_name}
        except (AttributeError, KeyError) as e:
            print("{} is not a proper estimator. Choose from {}".format(estimator_name, model_choices.keys()))
            raise e


def indexing_columns(name, dataframe, column):
    # subsets column from dataframe trying to catch exceptions
    try:
        values = dataframe.loc[:, column]
    except KeyError as e:
        print("No {} column found in {}".format(column, name))
        raise e
    return values


def read_datasets(data_string):
    """
    Reads in either a dictionary or a tuple and tries to open up the datasets.
    The dictionary has the form {"features":features_data_set,"target":target_data_set,
    optional"target_col":column_of_target_variable default is "target", optional"train_test_group":column_of_grouping_variable
    default is "group" if it doesn't exist it is set to target variable}
    The tuple has the name of the features dataset as first element and the name of the target as second.
    default columns are used for target and train_test_group

    :param data_string: tuple or dict
    :return:
    (features_name,target_name,target_col,train_test_group_col), features_set,target_set,target,group
    """
    if type(data_string) is dict:
        features_file = data_string["features"]
        target_file = data_string["target"]
        if data_string.get("target_col"):
            target_col = data_string.get("target_col")
        else:
            target_col = "target"
        if data_string.get("train_test_group"):
            train_test_col = data_string.get("train_test_group")
        else:
            train_test_col = "group"
    elif type(data_string) is tuple:
        features_file = data_string[0]
        target_file = data_string[1]
        target_col = "target"
        train_test_col = "group"

    else:
        raise Exception(
            "Data has to be expressed in either a tuple (features,target) or dictionary {\"features\":\"your_features\"," +
            "\"target\":\"your_target\"")
    # opening data
    data_directory = "../../data/processed/"
    try:
        X = pd.read_csv(data_directory + features_file, index_col=0)
        y = pd.read_csv(data_directory + target_file, index_col=0, encoding="ISO-8859-1")
    except FileNotFoundError:
        print("Files not in data/preprocessed, searching for them in the application's directory. You should run the" +
              " program from its directory: python program.py instead of python /somewhere/else/program.py")
        X = pd.read_csv(features_file, index_col=0)
        y = pd.read_csv(target_file, index_col=0, encoding="ISO-8859-1")
    except pd.errors.ParserError as e:
        print("Pandas seams to be unable to read this file. Make sure it's a csv")
        raise e
    except UnicodeDecodeError as e:
        print("The encoding of either the features or the targets is not encoded using UTF-8 or ISO-8859-1")
        raise e
    # Check to see if columns exist and return them
    target_col = checking_columns(y, target_col, x=target_col)

    # Get group column
    train_test_col = checking_columns(y, train_test_col, x=train_test_col, handle=lambda x: target_col)

    return features_file, target_file, X, y, target_col, train_test_col


def my_product(inp, just_values=True):
    """"
    Product of values in dictionary
    :param inp: dict

    :returns (product):
    """
    if just_values:
        return list(product(*inp.values()))
    else:
        return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def catch_key_errors(function, handle=lambda x: x, *args, **kwargs):
    """
    A shortcut for try: except KeyError to clean up code.
    :param function:
    :param handle:
    :return:
    """
    try:
        return (function(*args, **kwargs))
    except KeyError as e:
        return (handle)


def catch(object, handle, method_dict, **kwargs):
    """

    :param object: Any class whose method we wish to extract
    :param handle: If the specific method is not found in object, this method is returned
    :param method_dict: A dictionary of options for the method. It has to have the name of the method in the key "name"
    :return:
    method from method_dict["name"] or handle() if that is not present
    """
    if type(method_dict) is dict:
        try:
            # Using the string to obtain the appropriate function from the object
            method = getattr(object, method_dict.pop("name"))
            # Getting all legal parameters
            param_dict = dir(method())
            # intersection of legal parameters and kwargs of object. We do this to avoid errors made by users in specifying
            # method arguments
            intersection = {i: method_dict[i] for i in {**method_dict, **kwargs} if i in param_dict}
            return method(**intersection)
        except (AttributeError, TypeError):
            return handle
    elif type(method_dict) is str or type(None):
        try:
            # Using the string to obtain the appropriate function from the object
            method = getattr(object, method_dict)
            # Getting all legal parameters
            param_dict = dir(method())

            # intersection of legal parameters and kwargs of object. We do this to avoid errors made by users in specifying
            # method arguments
            intersection = {i: kwargs[i] for i in kwargs if i in param_dict}
            return method(**intersection)
        except (AttributeError, TypeError):
            return handle


def runningsplittest(foldgenerator, model_cv, model_name, features, targets, grid, number_of_splits=7,
                     number_of_folds=6, **kwargs):
    """
    Input
    foldgenerator: Stratified or Group KFold
    model_cv:      The model CV (cross validation) to use, eg log_cv/ rfr_cv/rfr_bcv
                   All the options are found in this methods file
    model_name:    Name of model to be used
    features:      The input data
    targets:       What's being predicted
    grid:          The grid of hyperparameters to search over

    Optional Inputs:
    number_of_splits: In how many folds the dataset will be split for testing purposes. The splitting is done using the
                      foldgenerator.
                      default = 7
    number_of_folds:The number of folds the training set is split to perform cross validation.
                    deffault = 6
    train_test_group: grouping variable to split dataset to train and test sets, and loop over them
                      default = targets
    validation_group: grouping variable to split train into validation sets for cross validation
                      default = train_test_group
    **kwargs: any other arguments that will be passed to the cross validation and testing procedures
        cvgenerator2: Fold generator for splitting into validation sets
                      default = foldgenerator
    :returns:
    Dictionary:
    {
    "y_pred": ndarray of shape (len(targets)) that contains estimator predictions,
    "best_parameters": list of length = number_of_splits that contains the best hyperparameters found in CV for each
                        fold,
    "coefficients": list of length = number_of_splits containing ndarrays of shape (features.shape[1]) containing the
                    coefficients of each feature give by the estimator. Some estimators don't use any coefficients in
                    such a way and will give an empty list instead, eg KNN.
    "time": the time in seconds it takes for the procedure to finish
    "false_samples": list of falsely predicted samples
    }
    """

    # Setting group variable used for the stratified or group split of
    # train-test split and for validation fold splits. If not set as kwarg
    # the response variable is used, This amounts to the usual stratified
    # sampling
    try:
        train_test_group = kwargs["train_test_group"]
    except KeyError:
        print("Using class labels as groupping variables. To choose a different groupping ",
              " set pass a list of the variables to groupby. Pass the list to ygrouplist")
        train_test_group = targets
    try:
        validation_group = kwargs["validation_group"]
    except KeyError:
        print("Using the same groupping svariable in the validation split")
        validation_group = train_test_group

    try:
        estimator_random_state = kwargs["random_state"]
    except (KeyError, AttributeError):
        estimator_random_state = 11235
    # Running the experiment
    y_pred, best_parameters, coefficients, time, false_samples = splithypothesis(cvgenerator=foldgenerator,
                                                                                 xset=features, model_cv=model_cv,
                                                                                 number_of_splits=number_of_splits,
                                                                                 number_of_folds=number_of_folds,
                                                                                 yset=targets, ygroup=train_test_group,
                                                                                 ygroup2=validation_group,
                                                                                 random_state=estimator_random_state,
                                                                                 grid=grid, **kwargs)

    dictr = {"y_pred": y_pred, "best_parameters": best_parameters, "coefficients": coefficients, "time": time,
             "false_samples": false_samples}
    return dictr


def splithypothesis(cvgenerator, xset, model_cv, number_of_splits, number_of_folds, yset, ygroup, ygroup2, grid,
                    **kwargs):
    """
    Tests the splits hypothesis on the LogisticRegression classifier.
    Input:
    cvgenerator: A cross-validator that can split data into folds
    xset: The features that will be used on the classifier
    model_cv: A function that will perform the cross-validation and select the hyperparameters
              of the classifier based on its performance on the cvgenerators folds
              examples are log_cv and rfr_cv
    number_of_splits: Number of ways to split the data into train-test
    number_of_folds: Number of ways to split the train set into validation sets
    yset: Labels of data, must match length of xset
    Output:
    scoring_results: A list of the accuracy of the classifier on each test fold
    best_parameters: A list of parameters used by the classifier to calculate the score for each test fold
    coefficients: A list of arrays with the iportance of the features
    """
    np.random.seed(11235)
    traintestsplit = cvgenerator(n_splits=number_of_splits)

    # Storing Index
    yindex = yset.index
    Ksets = traintestsplit.split(xset, y=ygroup, groups=ygroup)
    # scoring_results is a list with the scores of the estimator on each test set in
    # the traintestsplit folds.
    best_parameters = []
    y_pred = pd.DataFrame(index=yindex, data={"predictions": np.zeros(yindex.shape)})
    coefficients = []
    false_samples = []
    confusion = np.zeros((np.unique(yset).shape[0], np.unique(yset).shape[0]))
    # Vlidation cross vlidator
    try:
        cvgenerator2 = kwargs["cvgenerator2"]
    except KeyError:
        print("Same CV generator as train-test split. To change it set cvgenerator2 to StratifiedKFold or GroupKFold")
        cvgenerator2 = cvgenerator
    try:
        scaler = kwargs["scaler"]
    except (KeyError, AttributeError):
        print("The identity scaler is used. To change it pass a preprocessing class ",
              "with the variable scaler")
        scaler = prepro.FunctionTransformer(validate=False)

    # Timing the procedure
    start = timer()
    for i, index in enumerate(Ksets):
        train_index, test_index = index
        xtrain, xtest = xset.iloc[train_index], xset.iloc[test_index]
        ytrain, ytest = yset.iloc[train_index], yset.iloc[test_index]
        set_for_kfold = ygroup2.loc[xtrain.index]
        print(i)
        np.random.seed(11235)

        try:
            xtrain, ytrain = kwargs["resampler"].fit_resample(X=xtrain, y=ytrain)
            # we change set_for_kfold to ytrain because we resampled ytrain and it no longer has the same shape as
            # the ygroup2 variable. This means that resampling can't work when the foldgenerator used to split data
            # into train and test is set to GroupKFold and the train_test_group is not equal to the targets. This is
            # because we don't know how to resample another variable other than the target. eg if target is water colour
            # and train_test_group is the area number, we wouldn't be able to resample the water colour AND find what
            # area number the new samples would take/
            set_for_kfold = ytrain
        except (KeyError, AttributeError):
            print("No resampling is performed on the data. To change it pass a resampling method",
                  "from the imbalanced learn library")

        # Scaling the train set
        xtrain = scaler.transform(xtrain)

        CVfolds = cvgenerator2(n_splits=number_of_folds)
        Kfolds = CVfolds.split(xtrain, y=set_for_kfold, groups=set_for_kfold)
        #         for ind1,ind2 in Kfolds:
        #             print(set_for_kfold.iloc[ind2])

        # Perform grid CV using Kfolds as folds.
        set_parameters, CVgrid, set_coef = model_cv(X=xtrain, y=ytrain, trainfolds=Kfolds, grid=grid, **kwargs)
        # Transform our test data using the same scaler as the train
        xtest = scaler.transform(xtest)
        set_predictions = CVgrid.predict(xtest)

        y_pred.iloc[test_index, 0] = set_predictions

        # parameters are the best parameters of the model and CVgrid is the output
        # of the GridSearchCV method
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
    end = timer()
    time = start - end
    return y_pred, best_parameters, coefficients, time, false_samples


def fitmodel(xtrain, ytrain, scaler, cvgenerator2, number_of_folds, set_for_kfold, model_cv, grid, **kwargs):
    # Scaling the train set
    xtrain = scaler.transform(xtrain)

    CVfolds = cvgenerator2(n_splits=number_of_folds)
    Kfolds = CVfolds.split(xtrain, y=set_for_kfold, groups=set_for_kfold)
    #         for ind1,ind2 in Kfolds:
    #             print(set_for_kfold.iloc[ind2])

    # Perform grid CV using Kfolds as folds.
    set_parameters, CVgrid, set_coef = model_cv(X=xtrain, y=ytrain, trainfolds=Kfolds, grid=grid, **kwargs)


def checking_columns(dataframe, column, function=lambda x: x, handle=lambda x: x, *args, **kwargs):
    """
    Checks to see if dataframe has a column and returns a dunction
    :param dataframe:
    :param column:
    :param function:
    :param handle:
    :param args:
    :param kwargs:
    :return:
    """
    # subsets column from dataframe trying to catch exceptions
    try:
        dataframe.loc[:, column]
        return function(*args, **kwargs)
    except KeyError as e:
        print("No {} column found in {}".format(column, dataframe.columns))
        return handle(e)


class BayesSearchCV:
    """
    Bayesian search on hyper parameters.

    BayesSearchCV implements a "fit"  method.
    It also implements "predict" if it is implemented in the
    estimator used.

    The parameters of the estimator used to apply these methods are optimized
    by cross-validated search over parameter settings.

    In contrast to GridSearchCV, not all parameter values are tried out, but
    rather a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is
    given by n_iter.

    The parameters have to be presented in distributions or lists using the hp class of
    the hyperopt library.
    """
    def __init__(self, estimator, grid, cv=None, scoring="accuracy", verbose=1,
                 n_jobs=-1, refit="accuracy", return_train_score=False, iid=True,n_iter=100, **kwargs):
        """

        :param estimator: estimator object.
        A object of that type is instantiated for each grid point.
        This is assumed to implement the scikit-learn estimator interface.

        :param grid: dict
        Dictionary with parameters names (string) as keys and distributions
        of parameters to try. Distributions must provide a `'hp'`
        distribution ( from hyperopt.hp).

        :param cv:int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        :param scoring:string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        :param verbose: int
        Controls the verbosity: the higher, the more messages.

        :param n_jobs:n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

        :param n_iter:int, default=100
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

        :param kwargs:
        """

        self.verbose = verbose
        self.scoring = scoring
        try:
            self.cv = list(cv)
        except TypeError:
            if cv is None:
                self.cv = 3
            else:
                self.cv = cv
        self.n_jobs = n_jobs

        self.estimator = estimator
        self.grid = grid
        self.kwargs = kwargs
        self.n_iter = n_iter
    def fit(self,features,target):

        bayes_trials = Trials()
        # Define the search space


        # Optimize
        self.best_params_ = fmin(fn=lambda x: self.objective_function(x, X=features, y=target), space=self.grid, algo=tpe.suggest,
                                 max_evals=self.n_iter, trials=bayes_trials, return_argmin=False,verbose = self.verbose)
        model = self.estimator.set_params(**self.best_params_)
        self.best_estimator_ = model.fit(features, target)

        return self

    def objective_function(self,hyperparameters,X,y):
        model = self.estimator
        model.set_params(**hyperparameters)
        score = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs)
        best_score = score.mean()
        loss = 1 - best_score
        return {"loss": loss, "params": hyperparameters, "status": STATUS_OK}

    def predict(self,features):
        return self.best_estimator_.predict(features)

    def set_params(self,**kwargs):
        self.__dict__.update(kwargs)
        try:
            self.cv = list(self.cv)
        except TypeError:
            if self.cv is None:
                self.cv = 3
            else:
                self.cv = self.cv
        return self


class CV_models():
    search_methods = {
        "_cv": GridSearchCV,
        "_rcv": RandomizedSearchCV,
        "_bcv":BayesSearchCV

    }

    def __init__(self, grid, estimator, parameter_search, **kwargs):
        """
        Parameters
        -----------
        :param grid: dict
            Grid of parameter space to search for the best set of hyperparameters
        :param estimator: sklearn estimator
        :param parameter_search: str
            The method used to search over parameter space. Options are "grid", "random" and "bayes"
            Default is Grid, but Randomized and bayesian will be developed as well.
        :param validation_kfold_method: sklearn.model_selection KFold method
            StratifiedKFold or GroupKFold methods
        :param validation_group:
        :param number_of_kfolds: int
            Number of times to split the data
        :param kwargs: optional kwargs
        """
        self.kwargs = kwargs
        self.grid = grid
        self.estimator = estimator
        self.parameter_search = parameter_search

    def fit(self, features, target, ksets=None):
        """
        Fit the estimator to features and targets, using Cross validation. The Cross validation procedure uses the
        validation_group as the groupping/strata.

        :param ksets: * None, to use the default 3-fold cross validation,
                      * integer, to specify the number of folds in a (Stratified)KFold,
                      * CV splitter,
                      * An iterable yielding (train, test) splits as arrays of indices.
        :param features: ndarray or pandas dataframe
            Features set, or X
        :param target: 1darray or pandas series
            The class of each sample, or y
        :param validation_group: 1darray or pandas series
            Group/Strata for GroupKFold/StratifiedKFold
        :returns: best_parameters, model, coefficients
        :list: Set of best hyperparameters
        model: fitted model on features and target
        coefficients: Cofficients or feature importances for each feature. Can be NoneType if the estimator doesnt
            support it
        """

        # TODO: Incorporate bayes as well
        # ksets = self.cvgenerator(n_splits=self.number_of_kfolds).split(X=features, y=validation_group, groups= validation_group)
        search_instance = self.search_methods[self.parameter_search](self.estimator, self.grid)
        # intersection of parameters of split method and kwargs
        optional_in_search_method = set(dir(search_instance)) & set(self.kwargs.keys())
        # storing appropriate kwargs for this split method, eg n_iter for RandomSearchCV
        optional_kwargs = {i: self.kwargs[i] for i in optional_in_search_method}
        self.search_results = search_instance.set_params(cv=ksets, scoring="accuracy", verbose=1,
                                                         n_jobs=-1, refit="accuracy", return_train_score=False,
                                                         iid=True, **optional_kwargs).fit(features, target)
        best_parameters = self.search_results.best_params_
        # Check to see if estimator has attribute feature importances (if it uses trees) or coefficients
        # (logistic regression)
        try:
            intersection = set(dir(self.search_results.best_estimator_)) & {"feature_importances_", "coef_"}
            coefficients = getattr(self.search_results.best_estimator_, *intersection)
        except TypeError:
            coefficients = None

        return best_parameters, self.search_results, coefficients

    def predict(self, features):
        """
        The trained model is used to predict the class of features

        :param features: array-like
            has to have the same number of features as train-features set
        :return: predictions
        """
        return self.search_results.predict(features)

    def rfr_cv(X, y, trainfolds, grid, **kwargs):
        """
        Function performs Grid search Cross validation for random forrests with trainfolds as folds generator.
        Inputs:
        X: Train set
        y: Test set
        trainfolds: CV folds generator
        Returns:
        best_params: Best parameters of the CV procedure
        grid_result: The output of the GridSearchCV method when fitted to
                     X and y
                     :param X:
                     :param y:
                     :param trainfolds:
                     :return:
        """
        # Perform Grid-Search
        gsc = GridSearchCV(
            estimator=RandomForestClassifier(random_state=kwargs["random_state"]),
            param_grid=grid,
            cv=trainfolds, scoring=["accuracy"], verbose=1,
            n_jobs=-1, refit="accuracy", return_train_score=False)
        #  Grid result is the output of the gridsearchcv
        # best_params are the parameters of the highest scoring algorithm
        # coefficients are the weights of the fetures which signify which one is important
        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_
        coefficients = grid_result.best_estimator_.feature_importances_

        return (best_params, grid_result, coefficients)

    def log_cv(*args, X, y, trainfolds):
        """
        Function performs Grid search Cross validation with trainfolds as folds generator.
        Inputs:
        X: Train set
        y: Test set
        trainfolds: CV folds generator
        Returns:
        best_params: Best parameters of the CV procedure
        grid_result: The output of the GridSearchCV method when fitted to
                     X and y
        """
        try:
            penalty = args[0]["penalty"]
        except KeyError:
            penalty = "l1"
            print("You haven't specified a penalty, we will be using l1. Pass your desired penalty",
                  " to the variable penalty")
        if trainfolds == None:
            foldscv = StratifiedKFold(n_splits=5, random_state=11235)
            trainfolds = foldscv.split(X, wwfdf.Area_group.loc[X.index])
        gsc = GridSearchCV(
            estimator=LogisticRegression(penalty=penalty, solver='liblinear', max_iter=1000, random_state=11235,
                                         fit_intercept=True),
            param_grid={
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 20, 50] + list(np.arange(2, 20, 1))

                # ,'fit_intercept': (True)
            },
            cv=trainfolds, scoring=["accuracy"], verbose=1, n_jobs=-1, refit="accuracy",
            return_train_score=False
        )

        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_
        coefficients = grid_result.best_estimator_.coef_
        # rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)
        # Perform K-Fold CV
        # scores = cross_val_score(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')

        return best_params, grid_result, coefficients

    def som_cv(*args, X, y, trainfolds):
        if trainfolds == None:
            foldscv = StratifiedKFold(n_splits=5, random_state=11235)
            trainfolds = foldscv.split(X, wwfdf.Area_group.loc[X.index])
        # Perform Grid-Search
        gsc = GridSearchCV(
            estimator=SKSOM(),
            param_grid={
                "x": [8, 10],
                'sigma': [1, 2, 0.5],
                'learning_rate': [0.1],
                "neighborhood_function": ['gaussian', 'mexican_hat', 'bubble', 'triangle']
            },
            cv=trainfolds, scoring=["accuracy"], verbose=1,
            n_jobs=-1, refit="accuracy", return_train_score=False)
        #  Grid result is the output of the gridsearchcv
        # best_params are the parameters of the highest scoring algorithm
        # coefficients are the weights of the fetures which signify which one is important
        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_

        return (best_params, grid_result, [])

    def knn_cv(*args, X, y, trainfolds):
        """
        Function performs Grid search Cross validation for random forrests with trainfolds as folds generator.
        Inputs:
        X: Train set
        y: Test set
        trainfolds: CV folds generator
        Returns:
        best_params: Best parameters of the CV procedure
        grid_result: The output of the GridSearchCV method when fitted to
                     X and y
        """
        print(args)
        try:
            knnmetric = args[0]["metric"]
        except KeyError:
            knnmetric = "minkowski"
        if trainfolds == None:
            foldscv = StratifiedKFold(n_splits=5, random_state=11235)
            trainfolds = foldscv.split(X, wwfdf.Area_group.loc[X.index])
        # Perform Grid-Search
        gsc = GridSearchCV(
            estimator=KNeighborsClassifier(metric=knnmetric),
            param_grid={
                "n_neighbors": range(1, 30),
                # ,"braycurtis"],
                "weights": ["uniform", "distance"],
                "p": range(1, 6)
            },
            cv=trainfolds, scoring=["accuracy"], verbose=1,
            n_jobs=-1, refit="accuracy", return_train_score=False)
        #  Grid result is the output of the gridsearchcv
        # best_params are the parameters of the highest scoring algorithm
        # coefficients are the weights of the fetures which signify which one is important
        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_

        return (best_params, grid_result, None)

    def xgb_cv(X, y, trainfolds):
        """
        Function performs Grid search Cross validation for random forrests with trainfolds as folds generator.
        Inputs:
        X: Train set
        y: Test set
        trainfolds: CV folds generator
        Returns:
        best_params: Best parameters of the CV procedure
        grid_result: The output of the GridSearchCV method when fitted to
                     X and y
        """
        if trainfolds == None:
            foldscv = StratifiedKFold(n_splits=5, random_state=11235)
            trainfolds = foldscv.split(X, wwfdf.Area_group.loc[X.index])
        # Perform Grid-Search
        gsc = GridSearchCV(
            estimator=xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.2),
            param_grid={
                'max_depth': range(1, 3),
                'n_estimators': [500, 600],
                "min_child_weight": range(1, 3)
                # "reg_alpha":[0,0.3,0.7,1],
                # "reg_lambda":[0,0.3,0.7,1]

            },
            cv=trainfolds, scoring=["accuracy"], verbose=1,
            n_jobs=-1, refit="accuracy", return_train_score=False)
        #  Grid result is the output of the gridsearchcv
        # best_params are the parameters of the highest scoring algorithm
        # coefficients are the weights of the fetures which signify which one is important
        grid_result = gsc.fit(X, y)
        best_params = grid_result.best_params_
        coefficients = grid_result.best_estimator_.feature_importances_

        return (best_params, grid_result, coefficients)




cv_choices = {"grid": "_cv", "bayes": "_bcv", "random": "_rcv"}
