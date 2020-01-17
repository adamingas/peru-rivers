"""
A custom configuration file used to run custom experiments.
A dictionary outlining the attributes of a number of experiments is created, and then the product of the choices  is
used to initialise a number of experiments succintly.

We provide an example dictionary which can be used as a template for creating experimetns.
The water_dissimilarity dictionaryis used to run the water as labels experiments in the dissimilarity setting
(GroupKFold).

All values to keys have to be in a list.

The product of all choices is used to create experiments. In this case the experiments created are:
Features Meta target train_test FoldMethod splits Estimator    css  scaler resampler validationKFold splits cv_search 
---------------------------------------------------------------------------------------------------------------------
riverdf  wwfdf Water Area_group GroupKFold 7      RandomForest None None   None      GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      RandomForest None None   CSS       GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      RandomForest None None   CSSLOG    GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      MultinomialNBNone None   None      GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      MultinomialNBNone None   CSS       GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      MultinomialNBNone None   CSSLOG    GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      SVM          None None   None      GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      SVM          None None   CSS       GroupKfold      6      grid
riverdf  wwfdf Water Area_group GroupKFold 7      SVM          None None   CSSLOG    GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      RandomForest None None   None      GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      RandomForest None None   CSS       GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      RandomForest None None   CSSLOG    GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      MultinomialNBNone None   None      GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      MultinomialNBNone None   CSS       GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      MultinomialNBNone None   CSSLOG    GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      SVM          None None   None      GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      SVM          None None   CSS       GroupKfold      6      grid
fulldf   wwfdf Water Area_group GroupKFold 7      SVM          None None   CSSLOG    GroupKfold      6      grid


Specifying all these would be tiresome and unnecesary since they share so many attributes. Thus this system of products 
was devised. In particular the product happens like this:
The "Data" and "train_test_split_method" values are multiplied. For the water dissimilarity dictionary, there are two 
elements for data and one for train_test_split_method, thus there are 2 combinations.
All values of the "models" keys are multiplied; in the first element of the model list there are 6 combinations
(RandomForest and MultinomialNB), and in the second 3 (SVM). 
Thus in total there are 9 model combinations and 2 data. 2X9 = 18 total experiments.

All dictionaries have to be passed in the configuration_list which is looped by config_to_experiment.py to create and run
the experiments
"""

water_dissimilarity_dictionary = { # [M] Mandatory, [O] Optional
    "Data": [{"features":"riverdf", "meta":"wwfdf","target_col": "Water","train_test_col":"Area_group"},
             {"features":"fulldf", "meta":"wwfdf","target_col": "Water","train_test_col":"Area_group"}],
    # [M] features: The name of the features data set. The program will search the folder data/processed
    # [M] meta: The name of the meta data set. The program will search the folder data/processed
    # [O] target_col: is the column where the labels can be found, default value is  "target"
    # [O] train_test_col: is the column used to split the set to train and
    # test, default "group" if it exists, if not then "target" column
    "train_test_split_method": [{"name":"GroupKFold","n_splits":7}],
    # [O] How to generate train-test splits, can either be passed as a string or a dictionary. The dictionary will pass
    #     all keys associated with the method as arguments.
    #     name: The method used to split, most commonly used are StratifiedKFold and GroupKFold. All possible choices
    #     can be found in sklearn.model_selection
    #     https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #     Common arguments:
    #      n_splits: Number of folds/splits to create
    #      random_state: The seed used by random procedures (e.g. StratifiedKFold)
    "models": # [M] This key ise used to set up the experiments by choosing what classifiers to use. what normalisation
              # and transformation techniques and how to split the training set so as to perform cross validation
        [
        {
            "estimators":[ {"name":"MultinomialNB"},{"name": "RandomForest"}],
            # estimators: A list of strings/dictionaries specifiying what estimators to use and with what arguments (with
            #             the dictionary).
            #     name: The name of the estimator used. The default estimators are {"RandomForest","LogisticRegression"
            #     , "SVM","BernoulliNB","MultinomialNB","ComplementNB","KNN" (for K nearest Neighbours)}, and are
            #     implemented in the sklearn library.
            #     Optional Arguments
            #     cv: Cross validation search method to use. How to search over hyperparameters. The choices currently
            #     implementes are random and grid. default is grid
            #     n_iter: If random cv is chosen, n_iter controls the number of
            #     grid: A dictionary of hyperparameters over which the cv method will search to find the best set
            #     Estimator Arguments: Any other arguments that the estimators use. Search for their sklearn
            #     implementation to find out what keys can be used.
            #    Custom estimators can also be constructed and used by the program. To do so the additional method and
            #    grid key has to be used.
            #      method: The estimator object that is compatible with sklearn. It must have the attributes fit,
            #      predict, set_params, get_params.
            #      grid: A dictionary specifying the space over which the cross validation procedure will search.
            #      Example grids are given above in the default_grids variable which is a dictionary of grids for
            #      various classifiers
            #      Example Dictionary
            #      {"name":"Pipeline","method":Pipeline([("resampler",RandomOverSampler()),
            #       ("RFR",RandomForestClassifier())]),"grid":space,"n_iter":2,"cv":"bayes"}
            #
            #
            "css": [ "CSSLOG","CSS",None],
            # [O] Controls the use of CSS normalisation and log transformation. The program iteratires over all choices
            #     passed here. The choices are
            #     None: Nothing is used
            #     CSS: CSS normalisation is used.
            #     CSSLOG: CSS normalisation and log transformation is used
            "scaler": [None],
            # [O] Controls the use of scalers, implemented in the sklearn.preprocessing package. The preferred scaler
            #     can be passed as a string or a dictionary. The form of the dictionary is
            #     {"name":"any_appropriate_scaler>", "optional_arguments_for_the_scaler":values}.
            #     An example of a Standard scaler is
            #     {"name":"StandardScaler","with_mean":False}
            "resampler":[None],
            # [O] Controls the use of resamplers, implemented in the imblearn.over_sampling package. The preferred
            #     resampler can be passed as a string or dictionary.> The form of the dictionary is
            #     {"name":"<any_appropriate_resampler_in_imblearn>", "optional_arguments_of_the_resampler":values}
            #     Examples of OverSamplers include RandomOverSampling, SMOTE, SMOTE1, SMOTE2, ADASYN
            "validation":[{"name":"GroupKFold","group_col":"Area_group","n_splits":6}]
            # [O] The validation method and group column to use for cross validation.
            #     name: The name of the splitting method, any object in sklearn.model_selection package can be used,
            #     default StratiedKFold
            #     n_splits: Number of folds to create, default 3
            #     group_col: The column in the meta_data to use for splitting the folds

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
"""
The program will loop through all dictionaries in the configuration list and execute them
"""
configuration_list = [water_dissimilarity_dictionary]



# Experiments are created by taking all possible combinations of options in each experiment dictionary and creating
# an experiment object out of each combination.
# TODO: describe which fields/keys are used for the product
# DEFAULT GRIDS

