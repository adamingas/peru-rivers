import os
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../../R')
libpaths =robjects.r[".libPaths"]
print(libpaths())
paths = robjects.StrVector((*libpaths(),filename))
libpaths(paths)
print(libpaths())
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')

def importr_tryhard(packname, contriburl = None):
    try:
        rpack = importr(packname)
    except:
        utils.install_packages(packname,filename)
        rpack = importr(packname)
    return rpack
#Check if we can bypass all these installations, either by using default R installation
# or by installing only metagenomeSeq
script_dir = os.path.dirname(__file__)
abs_fil_path = os.path.join(script_dir, "package_installer.R")
r = robjects.r
#r.source(abs_fil_path)
#r.pkgTest("BiocManager")
importr_tryhard("stringi")
BiocManager = importr_tryhard("BiocManager")
try:
    msq = importr_tryhard("metagenomeSeq") 
except:
    BiocManager.install("metagenomeSeq",lib = "./R")
    msq = importr_tryhard("metagenomeSeq") 

