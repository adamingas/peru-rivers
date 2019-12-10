# Run this file interactively. It will set up the right environment t be able to open 
# the picke files
# To run interactively execute
# python -i opening_results.py
import sys
import os
from src.models import cssnormaliser
import numpy as np
import pandas as pd

project_dir =os.path.join(os.path.dirname(sys.argv[0]),"../../")
sys.modules["cssnormaliser"] = cssnormaliser
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("This file sets up the environment such that the pickled result file can be ",
        "opened. It should be run interactively with python -i opening_results.py",
        "\n or ipython -i opening_results.py ")
print("The project directory is stored in variable project_dir")
print("Already imported pandas and numpy.")
print("To open pickle file run pd.read_pickle(\"name_of_file.pickl\")")
