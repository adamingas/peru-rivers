from methods import string_to_key, select_estimator, read_datasets, cv_choices
from  sklearn import model_selection, preprocessing
import imblearn
from cssnormaliser import CSSNormaliser
from config import hypothesis
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import pandas as pd
from experiment import Experiment
import methods as mth
from itertools import product
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""
A script that converts a configuration file (dictionary) into experiments
"""




# def get_data(data_string):
#     """
#     Reads in either a dictionary or a tuple and tries to open up the datasets.
#     The dictionary has the form {"features":features_data_set,"target":target_data_set,
#     optional"target_col":column_of_target_variable default is "target", optional"train_test_group":column_of_grouping_variable
#     default is "group" if it doesn't exist it is set to target variable}
#     The tuple has the name of the features dataset as first element and the name of the target as second.
#     default columns are used for target and train_test_group
#
#     :param data_string: tuple or dict
#     :return:
#     (features_name,target_name,target_col,train_test_group_col), features_set,target_set,target,group
#     """
#     if type(data_string) is dict:
#         features =data_string["features"]
#         target =data_string["target"]
#         if data_string.get("target_col"):
#             target_col = data_string.get("target_col")
#         else:
#             target_col = "target"
#         if data_string.get("train_test_group"):
#             train_test = data_string.get("train_test_group")
#         else:
#             train_test = "group"
#     elif type(data_string) is tuple:
#         features = data_string[0]
#         target = data_string[1]
#         target_col = "target"
#         train_test = "group"
#
#     else:
#         raise Exception("Data has to be expressed in either a tuple (features,target) or dictionary {\"features\":\"your_features\","+
#               "\"target\":\"your_target\"")
#     # opening data
#     data_directory = "../../data/processed/"
#     try:
#         X = pd.read_csv(data_directory + features, index_col=0)
#         y = pd.read_csv(data_directory + target, index_col=0, encoding="ISO-8859-1")
#     except FileNotFoundError:
#         print("Files not in data/preprocessed, searching for them in the application's directory. You should run the"+
#               " program from its directory: python program.py instead of python /somewhere/else/program.py")
#         X = pd.read_csv(features, index_col=0)
#         y = pd.read_csv(target, index_col=0, encoding="ISO-8859-1")
#     except pd.errors.ParserError as e:
#         print("Pandas seams to be unable to read this file. Make sure it's a csv")
#         raise e
#     except UnicodeDecodeError as e:
#         print("The encoding of either the features or the targets is not encoded using UTF-8 or ISO-8859-1")
#         raise e
#     y_target = indexing_columns(target, y, target_col)
#     try:
#         # Get group column
#         y_group = indexing_columns(target, y, train_test)
#     except KeyError:
#         # If it doesnt exist assign target column as group column as well
#         y_group = y_target
#         train_test = target_col
#     return (features,target,target_col,train_test),X,y,y_target,y_group


more_complex_dict = {
    "Data": [("otu1", "meta1")],
    "Kfold-traintest": ["StratifiedKFold"],
    "models": [{
        "model_name": ["RFR"],
        "resampler": [{"name": "RandomOverSampler", "with_mean": False, "random_state": 11235}],
        "css": [None, "CSSLOG"],
        "cv": ["Grid", "Bayes"]
    },
        {
            "model_name": ["LOG"],
            "resampler": [{"name": "SMOTE", "k_neighbours": 4, "random_state": 11235}],
            "css": [None, "CSSLOG"],
            "cv": ["Grid"]
        }]
}
literal_dict ={}
css_choices = {"csslog": CSSNormaliser(log = True),"css": CSSNormaliser(), None: CSSNormaliser(identity=True)}
data_directory ="../../data/processed/"

def convert_string_dictionary(list_of_hypothesis:list):
    """
    Parses the list of dictionaries that make up the configuration to objects and files used by experiments.
    It outputs a list of dictionaries used for the creation of experiment instances

    :param list_of_hypothesis:
    :return:
    list of dictionaries of all combinations
    """
    list_of_experiment_configurations = []

    # Looping through all dictionaries in the list
    for hypothesis in list_of_hypothesis:
        # Each hypothesis is made up of data, including target and group column, and how train and test are created
        data_tuples =[read_datasets(i) for i in hypothesis["Data"]]
        data_dict ={
        "data_tuples" : [{"names":(i[0],i[1]),"X":i[2],"meta_data":i[3],"target_column":i[4],"train_test_column":i[5]}  for i in data_tuples],
        "train_test_split_method" : [mth.catch(model_selection,StratifiedKFold(), k,
                                    ) for k in hypothesis["train_test_split_method"]]}
        # After parsing in all data, a list of all combinations is created between
        list_data_dict = list(mth.my_product(data_dict,just_values=False))
        list_of_settings= []

        for i in hypothesis["models"]:
            # Looping through all settings for the particular data
            literal_dict = {}
            # model_dict =[k["name"].lower for k in i.pop("model_name")]
            literal_dict["estimators"] = [(select_estimator(k)) for k in i.pop("estimators")]
            try:
                literal_dict["resampler"] = [mth.catch(imblearn.over_sampling,imblearn.FunctionSampler(), k) for k in i.pop("resampler")]
            except KeyError:
                # If none is present or if resampler was not even mentioned
                # print("no resampler chosen")
                literal_dict["resampler"] =[imblearn.FunctionSampler()]
            try:
                literal_dict["scaler"] = [mth.catch(preprocessing, preprocessing.FunctionTransformer(validate=False),  k) for k in i.pop("scaler")]
            except (KeyError):
                # If none is present or if scaler was not even mentioned
                literal_dict["scaler"] = [preprocessing.FunctionTransformer(validate= False)]
            try:
                literal_dict["css_normalisation"] = [css_choices.get(string_to_key(k)) for k in i.pop("css")]
            except KeyError:
                literal_dict["css_normalisation"] = [CSSNormaliser(identity=True)]
            try:
                literal_dict["cv_method"] = [cv_choices.get(string_to_key(k)) for k in i.pop("cv")]
            except KeyError:
                print("The default \"grid\" search method has been selected")
                literal_dict["cv_method"] =["_cv"]

            try:
                validation_method_group =[]
                for k in i.pop("validation"):
                    validation_method_group.append( (mth.catch(model_selection,None, k),
                                                      k.get("group_col")))
            except KeyError:
                validation_method_group = [(None,None)] # The default method is the same as the train-test split
                #validation_group =None # The default group is the train-test group [k[3] for k in data_tuples]
            literal_dict["validation_method_group"] = validation_method_group
            try:
                literal_dict["draws"] = i.pop("draws")
            except KeyError:
                literal_dict["draws"] = [100]
            kwargs = i
            # a list of all possible combination of settings
            combination_of_choices =list(mth.my_product(literal_dict, just_values=False))
            for number,element in enumerate(combination_of_choices):
                combination_of_choices[number] = {**element.pop("estimators"),**element,**kwargs}
            # appending the list of configurations to a global list
            list_of_settings += combination_of_choices
            # adding back to all elements kwargs and also taking the estimators dictionary out so that it is more accesible
        # combinations of data and model choices
        list_of_experiment_configurations += [{**value[0],**value[1]} for value in product(list_data_dict,list_of_settings)]

    return list_of_experiment_configurations

def create_and_run_exp(list_of_hypothesis:list):
    list_of_experiments = []
    experiment_results = {}
    for settings in convert_string_dictionary(list_of_hypothesis):
        data_dictionary = settings.pop("data_tuples")
        try: X = data_dictionary.pop("X")
        except KeyError: pass
        exp_instance = Experiment(**{**data_dictionary,**settings})
        list_of_experiments.append(exp_instance)
        try:
            print(exp_instance)

            exp_result = exp_instance.run(X,data_dictionary["meta_data"])
            list_of_experiments.append(exp_instance)
            # adding to results the parameters of the experiment to make it easier to store them
            exp_result = {**exp_instance.return_dictionary(),**exp_result,"accuracy":exp_instance.accuracy}
            # experiment_results.setdefault("classifier",[]).append(exp_instance.estimator_name)
            # experiment_results.setdefault("CSS", []).append(str(exp_instance.css))
            # experiment_results.setdefault("scaler", []).append(str(exp_instance.scaler))
            # experiment_results.setdefault("resampler", []).append(str(exp_instance.resampler))
            # experiment_results.setdefault("data_names", []).append(exp_instance.names)
            # experiment_results.setdefault("target_column", []).append(str(exp_instance.target_column))
            # experiment_results.setdefault("accuracy", []).append(exp_instance.accuracy)
            for j in exp_result.keys():
                experiment_results.setdefault(j,[]).append(exp_result[j])
        except ValueError as vale:
            # This error arises when the GroupKFold method fails to split the data because the number of distinct groups in
            # validation variable is less than the number of splits
            print(vale)
            # print(exp_instance.validation_method," can't split ", exp_instance.validation_group, " because the number of "
            #         "splits is more than the number of factors in the grouping variable")
            continue
    return experiment_results,list_of_experiments

experiment_results,__ = create_and_run_exp(hypothesis)
resultsdf = pd.DataFrame(experiment_results)
resultsdf.to_pickle("../../results/supervised/results")

# TODO: Check after objection creation if we can check for duplicates using fancy __eq__ and __repl__
