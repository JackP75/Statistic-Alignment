"""joint distribution adatpation

author: Jack Poole, University of Sheffield 

"""


import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import scipy.linalg
from TCA import TCA

class JDA(TCA):
    def __init__(self,params={'classifier':KNeighborsClassifier(),'kernel':'rbf','lmda':1,'k':2,'max_iter':5}):
        super().__init__(params)   

    def get_M(self,n_s,n_t,class_mask=1):
        
        if type(class_mask)!=int:
            nsc=np.where(class_mask[:n_s]!=0)[0].shape[0]
            ntc=np.where(class_mask[n_s:]!=0)[0].shape[0]
        else:
            nsc=n_s
            ntc=n_t
        
        n_s_ones=1.0/nsc*np.ones((n_s,1))
        try:
            n_t_ones=-1.0/ntc*np.ones((n_t,1))
        except:
            n_t_ones=np.zeros((n_t,1))
        n_stack=np.vstack((n_s_ones,n_t_ones))
        n_stack=class_mask*n_stack #class_mask acts as an on/off mask
        M=np.dot(n_stack,n_stack.T)
        return M

    def get_M_sum(self,M_dict):
        M=np.zeros(M_dict['M_0'].shape)
        for label,Mn in M_dict.items(): 
            M+=Mn
        return M

    def fit(self,X_s,X_t,y_s):
        
        n_s=X_s.shape[0]       #source examples
        n_t=X_t.shape[0]       #target examples
        self.X=np.vstack((X_s,X_t))
       
        classes=np.unique(y_s) #array of class labels
        if classes.shape[0]>2:
            self.binary=False
        kernel,lmda,k,max_iter=self.params['kernel'],self.params['lmda'],self.params['k'],self.params['max_iter']
        if kernel is not None:
            #print('Using kernalisation')
            X=self.get_kernel(self.X)
        
        n,m=self.X.shape       #total examples, dimensions
        I=np.eye(n)            #Identity mxm
                
        M_dict={}
        M_dict['M_0']=self.get_M(n_s,n_t) #MMD matrix for marginal distributions
        H=self.get_H(n)     #Centering matrix
        
            
        #initialise 
        count=0
        while count<max_iter:#error is decreasing, labels are changing
        
            count+=1
            M=self.get_M_sum(M_dict)

            st=np.dot(np.dot(X.T,M),X) + lmda*I
            obj=np.dot(np.dot(X.T,H),X)

            eigval,eigvec=scipy.linalg.eig(st,obj)
            index=np.argsort(np.absolute(eigval))                
            self.W=np.real(eigvec[:,index][:,:k]) #Full rank nxn transformation matrix
            
            Z=np.dot(X,self.W)
            Z_s=Z[:n_s,:] 
            Z_t=Z[n_s:,:]
       
            self.classifier=OneVsRestClassifier(self.classifier).fit(Z_s,y_s.reshape((n_s,)))
            y_t=self.classifier.predict(Z_t)
        
            y=np.vstack((y_s.reshape(-1,1),y_t.reshape(-1,1)))
            for c in classes: 
                index=np.zeros((n,1))
                index[np.argwhere(y == c)]=1
                M_dict['M_'+str(c+1)]=self.get_M(n_s,n_t,index)

        return Z_s,Z_t
    

