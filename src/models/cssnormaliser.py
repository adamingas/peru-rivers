from rpy2.rinterface_lib.embedded import RRuntimeError
import numpy as np
from sklearn.base import TransformerMixin
from rpy2.robjects.packages import importr


msq = importr("metagenomeSeq")
# BiocManager = importr_tryhard("BiocManager","http://cran.us.r-project.org")
# msq =importr_tryhard("metagenomeSeq","https://cran.ma.imperial.ac.uk/")


class CSSNormaliser(TransformerMixin):
    def __init__(self,log=False,identity=False):
        """
        :param log: boolean
            whether to apply a log transfosmation to the data
            after normalisation
        :param identity: If true then CSSNormaliser does nothing
        """
        self.log = log
        self.identity = identity
        pass

    def fit(self, X, y=None):
        MRTrain =msq.newMRexperiment(X.T)
        self.p = msq.cumNormStat(MRTrain)
        return(self)
    def transform(self, X):
        if self.identity:
            return X
        MRTest = msq.newMRexperiment(X.T)
        Normalisation =msq.cumNorm(MRTest, p = self.p)
        Rmatrix=msq.MRcounts(Normalisation,norm = True,log = self.log)
        # converting r matrix to python
        Normalised_numpy = np.array(Rmatrix).T
        return(Normalised_numpy)

    def __hash__(self):
        return hash((self.log,self.identity))

    def __str__(self):
        return "CSSNormaliser(log = {},identity = {})".format(self.log,self.identity)
