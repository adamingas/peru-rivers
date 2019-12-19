import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

file_dir = os.path.dirname(os.path.realpath(__file__))
wwfdf_dir = os.path.join(file_dir,"../..","data/processed/wwfdf")
wwfdf = pd.read_csv(wwfdf_dir,index_col = 0)


def edge_colors():
    edgecolors = np.array([[1,0,0]]*164,dtype = float)
    edgecolors[wwfdf.Trip ==2] = [0,1,0]
    edgecolors[wwfdf.Trip ==3] =[0,0,1]
    return edgecolors

def add_jitter(arr):
    np.random.seed(1125)
    stdev = .02*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot_classification_on_map(labels, title="Classification",show = True,accuracy = None,**kwargs):
    fig,ax = plt.subplots(figsize=(7, 7))
    X, Y = (add_jitter(wwfdf.Easting), add_jitter(wwfdf.Northing))
    edge_array = (edge_colors())
    if type(labels) is (pd.DataFrame):
        labels = labels["predictions"]
    palette = sns.color_palette("Set2", np.unique(labels).shape[0])
    sns.scatterplot(x=X, y=Y, hue=labels, palette=palette,
                    style=wwfdf.Water, s=100 + 20 * wwfdf.Trip, edgecolor=(0, 0, 0), linewidth=0.5,
                    alpha=0.8,ax=ax)
    if accuracy:
        ax.text(x = 0.2,y = 0.05, s = "accuracy: "+str(100*round(accuracy,5))+"%",transform =ax.transAxes)
    ax.set_title(title)
    if kwargs.get("fname"):
        fig.savefig(fname=kwargs["fname"], dpi=300)
    if show:
        plt.show()