Readme
======

Machine Learning analysis performed on eDNA samples sourced from Norther
Peruvian Rivers.

Project Organization
--------------------

::

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- The documentation of the project using Sphinx (website). 
    │                         Still a  work in progress. 
    │
    ├── experiments        <- Saved objects of the Experiments class saved after running the config-to-experiment.py 
    │                         script.Each object includes the trained model and the data used to train it,as well as all
    │                         the parameters used to set up the experiment
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is 
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `as-unsupervised-ml`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── results            <- The results from running the program saved in pickle format.
    │   ├── supervised     <- The results from classification. To interact with the files the 
    │   │                      opening-results.py script should be run interactively.
    │   └── unsupervised   <- The clustering results saved in csv file format.
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate data
    │   │   └── make_dataset.R
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to configure and run the experiments.
    │   │   │                 
    │   │   ├── config-to-experiment.py <- Reads the config file, generates experiment objects and runs them
    │   │   ├── deafault_config.py      <- Default configuration file for the creation of experiments
    │   │   ├── custom_config.py        <- Custom configuration file for the creation of experiments
    │   │   ├── default_grids.py        <- Default hyperparameter grids that are searched over during CrossValidation 
    │   │   ├── cssnormaliser.py        <- Python wrapper for CSS normalisation
    │   │   ├── experiment.py           <- Experiment Class
    │   │   ├── methods.py              <- Methods used by program to train and test classifiers   
    │   │   ├── package_installer.R     <- R function used to install R libraries
    │   │   ├── r_install.py            <- Checks and installs necessary R libraries
    │   │   └── unsupervised_methods.py <- Script that reproduces unsupervised section of paper 
    │   │
    │   └── visualization  <- Scripts to create results oriented visualizations
    │       │── barplot.py              <- Functions to generate feature importance bar plots
    │       ├── classification_on_map.py<- Functions to plot classification results using location axes 
    │       └── prettyconfusion.py      <- Functions to draw confusion matrix
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

This project contains the package and set up requirements for
reproducing the machine learning results of the NatureMetrics peru
paper. The Makefile should be enough for reproducing the results. To
create custom experiments and run them the ``config.py`` and
``config-to-experiment.py`` files should be used.

Using the Makefile
------------------

To use the Makefile the user needs the GNU make program, available for
linux, macos and windows. To view all possible commands of the Makefile
start a shell in the projects directory and execute the command
``make``. This will list all the available options. The programming
languages used in the package are python and R, so make sure they are
installed. Then run

::

    make create-environment

To create a new conda or virtual environment. Then, activate the new environment using

::

    source activate peru-rivers

To install the necessary packages run

::

    make requirements

This will install in the ``peru-rivers`` conda environment all the
necessary python and R packages. If the metagenomeSeq package is not
installed in the default R location then the BiocManager will be
installed in order to download and install metagenomeSeq, The R
libraries will be stored in a folder named R in the root directory of
the project.

To reproduce the results choose the replicate option

::

    make replicate

The results will be saved in the ``/results/supervised/`` folder as a
pickle file and ``/results/unsupervised/`` as a csv file. To explore the
supervised results navigate to the folder in a shell and run

::

    ipython -i opening_results.py


which will open up an interactive python session preloaded with all
the necessary functions and classes needed to produce the paper plots.
The unsupervised csv file's rows are the samples of the peru rivers.
The columns indicate which clustering algorithm, data set and
transformation techniques were used to obtain the results. The number
assigned to each sample is it's group membership in the clustering
result.

Custom configuration
--------------------

The package can also be used to run experiments not design for the
purpose of the paper. To do so, the user will need to modify files in
the ``src/models/`` directory. The script that creates experiments from
the configuration files and also runs them is the
``config_to_experiment.py``. Running it using
``python config_to_experiment.py`` uses the default configuration
(replicating the paper's supervised results) and saves the results in
the ``results/supervised/`` directory. The experiments are saved in the
``experiments/`` directory.

To run custom experiments, the user has to edit the
``src/models/custom_config.py`` file or create a new one that follows
its structure. Instructions on how the configuration file works can be
found as comments inside the file itself.

The purpose of the configuration file is the mass creation of
experiments which have some attributes common between them (let's say
the use of CSS normalisation and the datasets) but differ in another
(which classifier is to be used). All possible combinations of settings
are put in a dictionary and the product of all possible choices is used
as attributes to create experiments.


To use the custom configuration file use the ``--config`` argument
when running the script, and pass it the name of the configuration file.
If the current directory is src/models/ and the chosen configuration
file is custom\_config.py then run:

::

    python config_to_experiment.py --config custom_config


To append a custom string at the start of the results filename use the
``--name`` argument and pass the desired string after it.
Running

::

    python config_to_experiment.py --config custom_config --name 1000rivers

will produce a results file with the name "1000riversresults"

--------------

.. raw:: html

   <p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>



