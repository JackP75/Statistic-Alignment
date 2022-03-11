"""correlation alignment

author: Jack Poole, University of Sheffield 

"""

import numpy as np


class CORAL:
    
    "Before applying CORAL the data should be standardised to align the mean and std (A-stand or NCA)"
    def fit(self, Xs, Xt,ys=None,yt=None):
        
        "Finds the transformation matrix for CORAL"
        d=Xs.shape[1]
        Cs=np.cov(Xs.T)+np.eye(d,d)
        Ct=np.cov(Xt.T)+np.eye(d,d)
        Cs_rt=np.linalg.cholesky(Cs)
        Ct_rt=np.linalg.cholesky(Ct)
        self.A=np.linalg.inv(Cs_rt).dot(Ct_rt)
        return self.A
    
    def fit_normal(self,Xs,Xt,ys,yt):
        "Finds the transformation matrix for NCORAL"
        d=Xs.shape[1]
        Xs_n=Xs[np.where(ys == 0)[0],:]
        Xt_n=Xt[np.where(yt == 0)[0],:]
        
        Cs=np.cov(Xs_n.T)+np.eye(d,d)
        Ct=np.cov(Xt_n.T)+np.eye(d,d)
        Cs_rt=np.linalg.cholesky(Cs)
        Ct_rt=np.linalg.cholesky(Ct)
        self.A=np.linalg.inv(Cs_rt).dot(Ct_rt)
        return self.A
    
    def transform(self,X):
        return X.dot(self.A)
    
    