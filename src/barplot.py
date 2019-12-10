# coding: utf-8
project_dir
get_ipython().run_line_magic('cd', 'project_dir')
os.chdir(project_dir)
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', 'peru-rivers/results/supervised/')
results= pd.read_pickle("results.pickl")
results.colnames
results.columns
results[results.target_column == "Water"]
waterdf =results[results.target_column == "Water"]
waterdf
waterdf[waterdf.estimator_name = "RandomForest"]
waterdf[waterdf.estimator_name == "RandomForest"]
import matplolib.pyplot as plt
import matplotlib.pyplot as plt
for i in waterdf[waterdf.estimator_name == "RandomForest"].coefficients:
    np.array(i)
    
waterdf[waterdf.estimator_name == "RandomForest"].coefficients
waterdf[waterdf.estimator_name == "RandomForest"].coefficients[0]
waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]
np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3])
mean_array = np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]).mean(axis = 0)
mean_array
var_array = np.array(waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]).var(axis = 0)
var_array
np.argsort(mean_array)
mean_array[np.argsort(mean_array)]
mean_array[np.argsort(mean_array)]
help
help(np.argsort)
waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]
sort_ind = np.argsort(mean_array)
import seaborn as sns
coef_array =waterdf[waterdf.estimator_name == "RandomForest"].coefficients[3]
