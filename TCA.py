"""transfer component analysis 

author: Jack Poole, University of Sheffield 

"""


import numpy as np 
from scipy.spatial.distance import cdist
import scipy.linalg
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.metrics import f1_score




class TCA():
    def __init__(self,params={'classifier':KNeighborsClassifier(n_neighbors=1),'mu':0.1,'k':2,'kernel':'rbf'}):
        self.params=params
        self.W=None
        self.X=None
        self.classifier=self.params['classifier']
        self.binary=True
        
    def get_kernel(self,X,Y=None):
           """Calculate kernel matrix between X and Y."""
           kernel=self.params['kernel']
           #If Y is None calculate the kernel with itself
           if Y is None:
               Y=X
               
           if kernel=='linear':
               return np.dot(X,Y.T) 
           elif kernel=='poly': #params:[gamma,b,M]
               (self.params['gamma']*np.dot(X,Y.T)+self.params['b'])**self.params['M']
           elif kernel=='rbf': #params:[BW]
               D=np.square(pairwise_distances(X,Y))
               median_heuristic=1/np.median(D)
               return  np.exp(-median_heuristic*cdist(X, Y, metric='euclidean'))
           else:
               raise ValueError('Invalid kernel')

    def get_L(self,n_s,n_t):
            """Creates the MMD matrix"""
            n_s_ones=1.0/n_s*np.ones((n_s,1))
            n_t_ones=-1.0/n_t*np.ones((n_t,1))
            n_stack=np.vstack((n_s_ones,n_t_ones))
            L=np.dot(n_stack,n_stack.T)
            return L 

    def get_H(self,n):
            """Creates the centering matrix"""
            return np.eye(n)-1./n * np.ones((n,n))

    def fit(self,X_s,X_t):
        
        mu,k=self.params['mu'],self.params['k']
        n_s=X_s.shape[0]
        n_t=X_t.shape[0]
        self.X=np.vstack((X_s, X_t)) 
        n,m=self.X.shape
        
        if k>m:
            raise ValueError('Requested too many dimensions!')
        
        L=self.get_L(n_s,n_t)
        H=self.get_H(n)
        
        K=self.get_kernel(self.X,Y=None) 
        
        if np.linalg.det(K)==0:
                count=0
                while np.linalg.det(K)==0 and count<10:
                    K+=np.eye(K.shape[0])*1*10**-6
                    count+=1
                    
        mini=np.dot(np.dot(K.T,L),K)+mu*np.eye(n) #minimise MMD distance      
        st=np.dot(np.dot(K.T,H),K)              #subject to variance =1
        
        eigval,eigvec=scipy.linalg.eig(mini,st)
        index=np.argsort(np.absolute(eigval))                
        
        self.W=np.real(eigvec[:,index][:,:k]) #Full rank nxn transformation matrix
        
        Z=np.dot(K,self.W)
        Z_s=Z[:n_s,:] 
        Z_t=Z[n_s:,:]
        return Z_s,Z_t

    def transform(self,X_test):
                    
        K=self.get_kernel(X_test,self.X) 
        Z=np.dot(K,self.W)
            
        return Z
    
    def train(self,X_s,y_s,classifier=KNeighborsClassifier()):
        '''Train a classifier using fitted data'''
        Z_s=self.transform(X_s)
        
        if np.unique(y_s).shape[0]>2:
            #Allows any classifier to be used
            self.binary=False
            self.classifier = OneVsRestClassifier(classifier).fit(Z_s, np.ravel(y_s))
        else:
            self.classifier = classifier.fit(Z_s,np.ravel(y_s))  
            
    
    def predict(self,X_test,y_t=None):
        '''Predict classes in the target domain'''
        Z=self.transform(X_test)
        pred=self.classifier.predict(Z)
        
        if y_t is not None and not self.binary:
            acc=self.classifier.score(Z,np.ravel(y_t)) 
            f1=f1_score(np.ravel(y_t), pred,average='macro')
            return pred, acc,f1
        
        elif y_t is not None:
            acc=self.classifier.score(Z,np.ravel(y_t)) 
            f1=f1_score(np.ravel(y_t), pred)
            return pred, acc,f1
        
        else:
            return pred
