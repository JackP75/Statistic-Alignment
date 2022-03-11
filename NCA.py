"""normal condition alignment

author: Jack Poole, University of Sheffield 

"""
import numpy as np 

def NCA(Xs,Xt,ys,yt,Xtest=None,ytest=None):
    """inputs:
    Xs- source data (ns,d)
    Xt -target data (nt,d)
    ys -normal condition labels [1,0] (ns,1) (or labels in common between source and target)
    yt -normal condition labels (nt,1)
    Note: if more than one class is used the labels should be balanced"""
    
    #1) normalise the source domain
    mu_s=np.mean(Xs,axis=0) 
    lmda_s=np.std(Xs,axis=0)
    Xs=(Xs-mu_s)/lmda_s 
    
    #2) estiamte mu and std of the normal condition (class 0)
    Xs_n=Xs[np.where(ys == 0)[0],:]
    mu_sn=np.mean(Xs_n,axis=0) 
    lmda_sn=np.std(Xs_n,axis=0)
        
    Xt_n=Xt[np.where(yt == 0)[0],:]
    mu_tn=np.mean(Xt_n,axis=0)
    lmda_tn=np.std(Xt_n,axis=0)
    
    #3) standardise and align the target
    Xt=(Xt-mu_tn)*(lmda_sn/lmda_tn)+mu_sn
    
    if Xtest is not None:   
            Xtest=(Xtest-mu_tn)*(lmda_sn/lmda_tn)+mu_sn
            return Xs,Xt,Xtest
        
    return Xs, Xt
