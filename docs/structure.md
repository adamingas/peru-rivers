# Structure of program
## Aims
* easily replicate the main findings of the project. This should be the first concern of the design
* replicate additional models or supporting info of project. This should be able to be paired with replication of main findings 
* select custom variables for classification. This shouldn't be an absolutely flexible option. The user can use the method directly if specific combinations of variables are needed.

## Interaction of user with interface
* if not options are given then the main findings of the project should be replicated.
* if custom option is given then either the user goes through command prompted question to select options or selects them through configuration file 

A python program is used to generate the configuration file which stores all settings needed to run the procedure.
```
configuration:
	python configuration.py
```

The file is read by another program that will run
```
scheme_dictionary  = {"location":{"foldgenerator":StratifiedKfold, "y":[wwfdf.location_labels]}}
for i in model:
	for scheme_key in scheme_dictionary:
		runninghypothesis(i,foldgenerator,y,X,indexnames,scaler,resampler, ygrouplist,css,**kwargs)
pandas dataframe:
model    
```
REsponse user gives for each otu and meta
{"model":{
	"name": ["RFR","LOGL1","KNN"]
	"css":[None,"css","csslog"],
	"scheme":["water dis","water sim","location"],
	"scaler":["None","StandardScaler"],
	"resampler":["RandomOverSampler","SMOTE"],
	"CV":["Grid","Bayes"]
}}

Config file example setup
every value has to be in a list because we are going to do a cartesian products on all elements of the lists os as to get all possible combinations without repeating ourselves
{
	"Data":[("otu1","meta1")],
	"Kfold-traintest":["StratifiedKFold"]
	"models":[{
		"model_name":["RFR"],
		"resampler":[{"name":"RandomOverSampler","with_mean":False,"random_state":11235}],
		"css":[None,"CSSLOG"],
		"cv":["Grid","Bayes"]
	},
	{
		"model_name":["LOG"],
		"resampler":[{"name":"SMOTE","k_neighbours":4,"random_state":11235}],
		"css":[None,"CSSLOG"],
		"cv":["Grid"]
	}]
}

The configuration produced by the interface has the format:
{
	"Data":[("otu1","meta1")],

	"Kfold-traintest":["StratifiedKFold"]
	"models":[{
		"model_name":["RFR","LOG","KNN"],
		"resampler":[{"name":"RandomOverSampler","with_mean":False,"random_state":11235}],
		"css":[None,"CSS","CSSLOG"],
		"cv":["Grid","Bayes"],
		"scaler":[None,{"name":"StandardScaler","with_mean":False}]
	}]
}
This dictionary is then read, flatten into lists and with some conditionals we remove some choices. eg 
```python
for i in list_of_products:
	if i["model_name"] == "RFR":
		i["scaler"] =None
	if "NB" in i["model_name"] :
		i["cv"] = "Grid"
	#add other conditionals
# then take only unique elements so as not to repeat ourselves
unique_list= unique(list_of_products)
# create experiment objects out of list
list_experiments =[]
for i in unique_list:
	list_experiments.append(Experiment(**i))
# save list of experiments to be used by 
```	
## Requirements 

How do we split up code? 
It has to be split up based on experiments
How is each experiment defined? Is each combination of each option an experiment?
It is, but there is some convenience in groupping together by groups. We have been using models to group responses together but this might be limited. A user might want to group by rivers and see which model performs better, or by accuracy and see which combination performs better. Therefore the framework should be able to accomodate any groupping

Each experiment therefore is an object which has inherent values its settings and configurations and the results of the experiment.

The responses of user should create experiments.

Some combinations might not be possible or sensible, thus, to save omputational time, conditionals should be put in place to catch and eliminate bad combinations. Such combinations include rfr with a scaler, or knn and random search.

User should be able to customise and run their own experiments, either through the code or through a config file. If it is a config file there must be some grouping variable otherwise the experiments can't be easily implemented (eg they have to specify all combinations by hand). 

if it is through code there should be an interface to a function that can run multiple experiments, and easy enough to edit parameters of cv. We could put all grids in a file that the user can edit, and allow the user to specify hidden kwargs like number of folds, ygrouplist1/2 so that there is no interaction with the code.

## Code tree/structure
e = program that is executed 
c = configurable file that user should be able to access
m = method or program not meant to be accesible to user
User 
|
|e-command line program that creates experiments "interface.py" using datasets provided by user
  |c-Configuration file created by user responses 
    |
    |m-Experiment objects for each possible combination "experiment.py"
    |r-Runs experiments and stores results in each experiment "run.py"
    |m-saves results in a dataset and in disk in a user
    |accessible format "save.py"
|
|
|c-Configuration file that user can change to create experiments
  |
  |m-Experiment objects for each possible combination
|
|c-Grid values
  |
  |m-Experiment objects


interface: otu, meta -> config file, grid
experiment: config file,grid -> experiments (list of objects)
run: experiments -> database (of results)

multidataset: [otu1,otu2],[meta1,meta2] - run interface multiple ti
interface: otu,meta - checks if otu and meta have same number of samples - check columns of meta if column named target and group are present - ask for stratified or groupkfold if target and group are different -> config,grid
	


### Functions



```python

```

Variables that have to be chosen:
* model: [RFR,LOG,KNN,SVM,MNB,BNB,CNB]
* dataset: [riverdf,riverdf100s,fulldf,fulldf100s]
* cssnormalisation and log transformation [CSS,CSSLOG]
* Scheme of prediction or target variable: [water dissimilarity/similarity,rivers_labels,location_labels]
* scaler: custom
* resampler: custom
* cross validation procedure: [Grid,Random,Bayes]

