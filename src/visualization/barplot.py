import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
#
# results = pd.read_pickle("../results/supervised/results.pickl")
# waterdf = results[results.target_column == "Water"]
# otu = pd.read_csv("../data/processed/riverdf",index_col = 0)
# fuldf = pd.read_csv("../data/processed/fulldf",index_col = 0)
file_dir = os.path.dirname(os.path.realpath(__file__))
taxa_dir = os.path.join(file_dir,"../..","data/processed/taxadf")
taxadf = pd.read_csv(taxa_dir,index_col = 0)

def create_coef_df(list_of_coef,df_sourced):
    """
    Converts list of arrays into a wide format dataframe, with otu taxonomy as the column names
    if it exists
    :param list_of_coef: list of arrays of feature importances
    :param df_sourced: The dataframe that was used to train the classifier
    """
    otu_only_numbers = df_sourced.columns.to_series().apply(lambda x :x[3:])# taxadf.index.to_series().apply(lambda x :x[3:])
    filtered_order = taxadf.Order.apply(lambda x:x[0:5])
    filtered_family = taxadf.fillna("").Family.apply(lambda x:x[0:5])
    column_names =(otu_only_numbers +" "+ filtered_order.loc[df_sourced.columns].fillna("")+" "+
                   filtered_family.loc[df_sourced.columns].fillna(""))
    coef_df = pd.DataFrame(list_of_coef, columns=column_names)
    return coef_df

def barplot_sort(coef_df,number_of_bars,title = "Feature Importance",fname = None,figsize= (9,9),plot = True):
    """
    Sorts coefficient dataframe and plots the most importantd features as bars..

    :param coef_df: A dataframe of size  number of train-test splits x otus/features.
    :param number_of_bars: Number of bars to draw
    :param title: Title of graph, default is "Feature Importance"
    :param plot: Whether or not to show the plot
    """
    fig,ax = plt.subplots(figsize= figsize)
    mean_coef = coef_df.mean()
    sort_indx = np.argsort(mean_coef)
    sorted_coef_df = coef_df.iloc[:,sort_indx]
    colour_list = colour_generator_from_series(sorted_coef_df.columns.to_series().apply(lambda x: x.split(" ")[1]))
    # sns.barplot(data=sorted_coef_df.iloc[:,-number_of_bars:])
    ax.set_title(title)

    plt.bar(x = sorted_coef_df.columns[-number_of_bars:],height=sorted_coef_df.mean().iloc[-number_of_bars:],
            yerr = sorted_coef_df.var().apply(np.sqrt).iloc[-number_of_bars:],color = colour_list[-number_of_bars:],
            axes = ax)
    ax.set_ylabel("Feature Importance")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    fig.set_tight_layout(True)
    if fname:
        fig.savefig(fname,dpi = 300)
    if plot:
        plt.show()

def colour_generator_from_series(series):
    colord = len(set(series))
    uniq = list(set(series))

    cmap = plt.get_cmap('tab20')
    colours = [cmap(i) for i in np.linspace(0, 1, colord)]

    dict_of_colours = dict(zip(uniq,colours))
    return [dict_of_colours[i] for i in series]


# for i in waterdf[waterdf.estimator_name == "RandomForest"].coefficients:
#     np.array(i)
#
# waterdf[waterdf.estimator_name == "RandomForest"].coefficients
# waterdf[waterdf.estimator_name == "RandomForest"].coefficients[0]
# waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]
# np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3])
# mean_array = np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]).mean(axis=0)
# var_array = np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]).var(axis=0)
# np.argsort(mean_array)
# mean_array[np.argsort(mean_array)]
# waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]
# sort_ind = np.argsort(mean_array)
#
# coef_array = waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]
# np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3])
# coef_array =waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]
#
#
# pd.DataFrame(columns =otu.columns).append(pd.DataFrame(coef_array,columns = otu.columns),ignore_index = True)
# coef_df =pd.DataFrame(columns =otu.columns).append(pd.DataFrame(coef_array,columns = otu.columns),ignore_index = True)
# sns.barplot(data = coef_df)
# mean_array
# mean_array = np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]).mean(axis = 0)
# sort_ind =np.argsort(mean_array)
# coef_df =pd.DataFrame(columns =otu.columns[sort_ind]).append(pd.DataFrame(coef_array[sort_ind],columns = otu.columns[sort_ind]),ignore_index = True)
# sort_ind
# otu.columns[sort_ind]
# coef_array[sort_ind]
# coef_array
# coef_df =pd.DataFrame(columns =otu.columns[sort_ind]).append(pd.DataFrame(coef_array,columns = otu.columns)[sort_ind],ignore_index = True)
# pd.DataFrame(coef_array,columns = otu.columns)[sort_ind]
# pd.DataFrame(coef_array,columns = otu.columns)
# pd.DataFrame(coef_array,columns = otu.columns).iloc[:,sort_ind]
# coef_df =pd.DataFrame(columns =otu.columns[sort_ind]).append(pd.DataFrame(coef_array,columns = otu.columns).iloc[:,sort_ind],ignore_index = True)
# sns.barplot(data = coef_df.iloc[:.-20:-1])
# sns.barplot(data = coef_df.iloc[:,-20:])
# get_ipython().run_line_magic('save', '1-56')
