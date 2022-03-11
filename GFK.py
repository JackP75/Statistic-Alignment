"""geodesic flow kernel

This algorithm is an adaptation of: https://www.idiap.ch/software/bob/docs/bob/bob.learn.linear/stable/_modules/bob/learn/linear/GFK.html#GFKMachine

author: Jack Poole, University of Sheffield 

"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

from utils import plot_transfer


class GFK():
    def __init__(self):
        self.pca_s=PCA()
        self.pca_t=PCA()
        self.G=None
        self.X=None   
        self.classifier=None
        self.binary=True

        
    def fit(self,X_s,X_t,d=1):
        
        #Get basis of subspaces for the source Ps and target Pt, and the orthoganel compliment Rs
        D=X_s.shape[1]
        self.pca_s.fit(X_s)
        self.pca_t.fit(X_t)
        self.X=np.vstack((X_s,X_t))
        Ps=self.pca_s.components_[:d,:].T
        Rs=self.pca_s.components_[d:,:].T
        Pt=self.pca_t.components_[:d,:].T

        #compute SVD and get U1,U2 and principle angles
        U1,Gamma,V1=np.linalg.svd(Ps.T.dot(Pt))
        U2,Epsil,V2=np.linalg.svd(Rs.T.dot(Pt))
        theta=np.arccos(Gamma)
        #construct GFK
      
        epsilon=1e-12 #The angle could be 0 so needed for computation
        L1=np.diag(0.5*(1+np.sin(2*theta)/np.maximum(epsilon,2*theta)))
        L2=np.diag(0.5*((np.cos(2*theta)-1)/np.maximum(epsilon,2*theta)))
        L3=np.diag(0.5*(1-np.sin(2*theta)/np.maximum(epsilon,2*theta)))

        delta1_1 = np.hstack((U1, np.zeros(shape=(d, D - d))))
        delta1_2 = np.hstack((np.zeros(shape=(D - d, d)), U2))
        delta1 = np.vstack((delta1_1, delta1_2))
        
        delta2_1 = np.hstack((L1, L2, np.zeros(shape=(d, D - 2 * d))))
        delta2_2 = np.hstack((L2, L3, np.zeros(shape=(d, D - 2 * d))))
        delta2_3 = np.zeros(shape=(D - 2 * d, D))
        delta2 = np.vstack((delta2_1, delta2_2, delta2_3))

        delta3_1 = np.hstack((U1, np.zeros(shape=(d, D - d))))
        delta3_2 = np.hstack((np.zeros(shape=(D - d, d)), U2))
        delta3 = np.vstack((delta3_1, delta3_2)).T
        delta = np.dot(np.dot(delta1, delta2), delta3)
        
        PR=np.hstack((Ps,Rs))
        self.G = np.dot(np.dot(PR, delta), PR.T)
        
    def transform(self,X_test):
        return X_test.dot(self.G).dot(self.X.T)
    
    def train(self,X_s,y_s,classifier=KNeighborsClassifier()):
        '''Train a classifier using fitted data'''
        Z_s=self.transform(X_s)
        if np.unique(y_s).shape[0]>2:
            #Allows any classifier to be used
            self.binary=False
            self.classifier = OneVsRestClassifier(classifier).fit(Z_s, np.ravel(y_s)) 
        else:
            self.classifier = classifier.fit(Z_s,np.ravel(y_s))  
            
    def predict(self,X_t,y_t=None):
        '''Predict classes in the target domain'''
        
        Z_t=self.transform(X_t)
        pred=self.classifier.predict(Z_t)
        if y_t is not None and not self.binary:
            acc=self.classifier.score(Z_t,np.ravel(y_t)) 
            f1=f1_score(np.ravel(y_t), pred,average='macro')
            return pred, acc,f1
    
        elif y_t is not None:
            acc=self.classifier.score(Z_t,np.ravel(y_t)) 
            f1=f1_score(np.ravel(y_t), pred,average='macro')
            return pred, acc,f1
        
        else:
            return pred
 