Makefile Commands
=================

The Makefile contains the central entry points for common tasks related to this project.
It is recommended for users to run all the commands in the `Setting up Environment`_ section even if they choose not to replicate the results of the paper. This is to ensure that all necessary libraries required for the proper functioning of the package are installed.

Setting up Environment
^^^^^^^^^^^^^^^^^^^^^^

* **`make test-environment`** runs `test-environment.py` which tests whether python 3 is installed. 
* **`make create_environment`** creates a python interpreter environment with either conda or virtualenv. 
* **`make requirements`** runs **`make test-environment`**. Then installs all python and R requirements.
* **`make data`** Converts raw data into usable otu-matrices and properly formatted metadata.

Replicating paper
^^^^^^^^^^^^^^^^^

* **`make supervised`** runs **`make data`** and **`make requirements`**. Then replicates supervised results of paper. Results are stored in `results/supervised/` as a pickled dataframe and in `experiments/` as pickled experiment objects. 
* **`make unsupervised`** runs **`make data`** and **`make requirements`**. Then replicates unsupervised results of paper. Results are stored in `results/unsupervised/` in a .csv file.
* **`make replicate`** runs **`make unsupervised`** and then **`make supervised`**. 

