# Run this file interactively. It will set up the right environment t be able to open 
# the picke files
# To run interactively execute
# python -i opening_results.py
import sys
import os
from src.models import cssnormaliser
import numpy as np
import pandas as pd
from src.visualization.barplot import *
from src.visualization.prettyconfusion import *
from src.visualization.classification_on_map import *


def results_to_csv(dataframe,fname,columns = None):
    """
    Saves dataframe to csv with certain columns.
    The columns to be saved are:
    columns_to_save = ["estimator_name","target_column","css","resampler","scaler",'names',
      'cv', 'train_test_split_method',
     'train_test_split_column', 'validation_method', 'validation_group','confusion', 'confusion_at_the_end', 'accuracy']
    :param dataframe: Dataframe of results
    :param fname: Directory and name of file to be saved.

    Optional
    --------
    :param columns: List of alternative columns to save
    :return:
    """
    columns_to_save = ["estimator_name","target_column","css","resampler","scaler",'names',
      'cv', 'train_test_split_method',
     'train_test_split_column', 'validation_method', 'validation_group','confusion', 'confusion_at_the_end', 'accuracy']
    if columns is not None:
        columns_to_save = columns
    css_dict = dict(zip(dataframe.css.unique(), ["CSSLOG", "CSS", "NoCSS"]))
    dataframe["css"] =dataframe["css"].apply(lambda x: css_dict[x])
    dataframe.loc[:,columns_to_save].to_csv(fname)

def replicate_classification_maps(results):
    """
    Replicates the maps made for classification when running the default set of experiments. Do not use
    when not replicating the paper's results.
    :param results: The results pandas dataframe
    :return:
    """
    # Creating dictionaries that map the values found in the dataframe which are hard to read
    # to values which are much easier.
    # eg CSSNormaliser(log = False,identity = True) which means that no css normalisation is applied
    # is converted into NoCSS. The same happens for resampling and scaling
    css_dict = dict(zip(results.css.unique(), ["CSSLOG", "CSS", "NoCSS"]))
    resample_dict = dict(zip(results.resampler.apply(str).unique(), ["NoRsmpl", "RndmOvrSmpl"]))
    scaler_dict = dict(zip(results.scaler.apply(str).unique(), ["NoScl", "StndSdclMF"]))
    for i in results.index:
        # for each row/experiment
        row = results.loc[i]
        resample_str = resample_dict[str(row.resampler)]
        scaler_str = scaler_dict[str(row.scaler)]
        css_str = css_dict[row.css]
        filename = "map" + row.estimator_name + row.target_column + resample_str + scaler_str + css_str
        # we plot the classification results on a map of data
        plot_classification_on_map(row.y_pred, label=True,
                title=row.estimator_name + " " + resample_str + scaler_str + css_str,
                                   fname=project_dir + "reports/figures/" + filename + ".png",
                                   show = False, figsize = (5,5), accuracy = row.accuracy)
        plt.close()

def replicate_confusion_matrix(results):
    """
        Replicates the confusion matrices made for classification when running the default set of experiments. Do not use
        when not replicating the paper's results.
        :param results: The results pandas dataframe
        :return:
    """
    css_dict = dict(zip(results.css.unique(), ["CSSLOG", "CSS", "NoCSS"]))
    resample_dict = dict(zip(results.resampler.apply(str).unique(), ["NoRsmpl", "RndmOvrSmpl"]))
    scaler_dict = dict(zip(results.scaler.apply(str).unique(), ["NoScl", "StndSdclMF"]))
    for i in results.index:
        row = results.loc[i]
        resample_str = resample_dict[str(row.resampler)]
        scaler_str = scaler_dict[str(row.scaler)]
        css_str = css_dict[row.css]
        filename = row.estimator_name + row.target_column + resample_str + scaler_str + css_str

        confusion_matrix_visualisation(row.confusion, label=True,
                                       title=row.estimator_name + " " + resample_str + scaler_str + css_str, fname=
                                        project_dir + "reports/figures/" + filename + ".png", show = False, figsize = (5, 5))
        plt.close()



project_dir =os.path.join(os.path.dirname(sys.argv[0]),"../../")
sys.modules["cssnormaliser"] = cssnormaliser
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("This file sets up the environment such that the pickled result file can be ",
        "opened. It should be run interactively with python -i opening_results.py",
        "\n or ipython -i opening_results.py ")
print("The project directory is stored in variable project_dir")
print("Already imported pandas as pd, numpy as np, matplotlib as plt, seaborn as sns.")
print("To open pickle file run pd.read_pickle(\"name_of_file.pickl\")")
print("The following functions can be used to produce certain visualisations of the results. ")
print("To get a better description of how these functions work you can type help(<function>)"
      " where <function> is one of the following.\n")
print("plot_classification_on_map(labels,title,show,accuracy,fname):\n Plots the results of the classification"
      " on a map of easting and northing coordinates.\n")
print("confusion_matrix_visualisation(confusion,title,show,label,custom_labels,figsize,fname):\n"
      "Plots the confusion matrix of a numpy confusion matrix. The label variable can be set to true to"
      " use default labels for replicating the paper's results")
print("barplot_sort(list_of_coef,number_of_bars,df_sourced,title,fname,figszie,show):"
      "\nPlots bars of the <number_of_bars> most important features. To be used with Random Forest"
      "results. Pass it the coefficient column of the results (a list of np array)")
print("Other Functions\n"
      "create_coef_df: Creates a wide dataframe from list of arrays found in the coefficients column\n"
      "replicate_confusion_matrix(results): pass it the default results dataframe and to replicate the"
      " confusion matrix figures found in the paper\n"
      "replicate_classification_maps(results): pass it the default results dataframe to replicate the "
      "maps for classification\n")
