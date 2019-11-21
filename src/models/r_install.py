from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects

rpy2.robjects.numpy2ri.activate()
utils = importr('utils')

# def importr_tryhard(packname, contriburl):
#     try:
#         rpack = importr(packname)
#     except RRuntimeError:
#         utils.install_packages(packname,contriburl)
#         rpack = importr(packname)
#     return rpack
# Check if we can bypass all these installations, either by using default R installation
# or by installing only metagenomeSeq
r = robjects.r
r.source("package_installer.R")
r.pkgTest("BiocManager")
BiocManager = importr("BiocManager")
BiocManager.install("metagenomeSeq")
msq = importr("metagenomeSeq") 
