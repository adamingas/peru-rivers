import pandas as pd
from src.models.cssnormaliser import CSSNormaliser
from src.models.default_grids import default_grids
# import pymc3 as pm
# import theano.tensor as tt
# from theano import shared
from sklearn.model_selection import  cross_val_score, GridSearchCV, StratifiedKFold, GroupKFold, \
    RandomizedSearchCV
import re
from itertools import product
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,ComplementNB,GaussianNB,BernoulliNB
import os
from hyperopt import Trials,STATUS_OK,fmin,tpe,hp
from hyperopt import base
base.have_bson = False
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Get current directory of file
dirname = os.path.dirname(__file__)
project_dir = os.path.join(dirname,"..","..")
# from sksom import SKSOM
# import xgboost as xgb
model_choices = {"RandomForest": RandomForestClassifier, "LogisticRegression": LogisticRegression, "SVM": SVC,
                 "BernoulliNB":BernoulliNB,"MultinomialNB":MultinomialNB,"ComplementNB":ComplementNB,
                 "KNN":KNeighborsClassifier}
cv_choices = {"grid": "_cv", "bayes": "_bcv", "random": "_rcv"}
css_choices = {"csslog": CSSNormaliser(log = True),"css": CSSNormaliser(), None: CSSNormaliser(identity=True)}
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

    :param estimator_dict: dict or string
        Dictionary containing name of estimator together with kwargs. A string can also be used, and the default
        estimator will be used.
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
        target_file = data_string["meta"]
        if data_string.get("target_col"):
            target_col = data_string.get("target_col")
        else:
            target_col = "target"
        if data_string.get("train_test_col"):
            train_test_col = data_string.get("train_test_col")
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
    data_directory = os.path.join(project_dir,"data/processed/")
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

        # Choosing the search method from Grid, Random and Bayes
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
        except (TypeError,AttributeError):
            coefficients = None
        # cv attribute of the GridSearchCV/RandomSearchCV/BayesSearchCV class is a generator which can't be pickled
        # TODO: Change cv to list before passing it to searchCV classes.
        self.search_results.cv = None
        return best_parameters, self.search_results, coefficients

    def predict(self, features):
        """
        The trained model is used to predict the class of features

        :param features: array-like
            has to have the same number of features as train-features set
        :return: predictions
        """
        return self.search_results.predict(features)


