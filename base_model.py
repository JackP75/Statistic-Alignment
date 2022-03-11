"""
Base class for DNN domain adaptation 

@author: Jack Poole, University of Sheffield
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from utils import plot_transfer,Proxy_A
import math
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import tensorflow.keras as keras



optimiser=tf.keras.optimizers.Adam(lr=1e-4,
    beta_1=0.9,
    beta_2=0.999)


class base_model(tf.keras.models.Model):
    def __init__(self,params={'feature_nodes':10,
                              'num_feat_layers':2,
                              'num_class_layers':2,
                              'input_dim':3,
                              'output_size':3 ,
                              'drop_rate':0.25,
                              'reg':0.0001,
                              'entropy':1e-6,
                              'BN':True,
                              'lr':1e-3,
                              'kernel':'linear',
                              'adapt_extractor':False,
                              'type':'dense',
                              'filters':10,
                              'kernel_size':(3,1),
                              'stride':1}):
        super().__init__()   
        
        self.params=params
        
        #hyperparameters
        self.feature_nodes=params['feature_nodes']
        self.output_size=params['output_size']
        self.drop_rate=params['drop_rate']
        self.reg=params['reg']
        self.BN=params['BN']
        
        #extractor type
        self.type=params['type']

        #feature extractor
        self.feature_extractor=[]
        
        #FOR SPECIFYING INPUT SIZE
        # #layer 1
        # self.feature_extractor.append(layers.Dense(self.feature_nodes, activation=None, 
        #                            kernel_initializer=tf.keras.initializers.he_normal(),
        #                            kernel_regularizer=keras.regularizers.l2(self.reg))),
        #input_shape=(,params['input_dim'])
        # self.feature_extractor.append(layers.Dropout(self.drop_rate))
        # if params['BN']:
        #     self.feature_extractor.append(layers.BatchNormalization())
        # self.feature_extractor.append(layers.LeakyReLU())
        
        if self.type=='conv2d':
            self.feature_extractor.append(layers.Conv1D(filters=params['filters'], 
                                          kernel_size=params['kernel_size'], 
                                          strides=params['stride'], padding='valid',
                                          activation=None, kernel_initializer=tf.keras.initializers.he_normal(),
                                          input_shape=(None,params['input_dim'][0],params['input_dim'][1])))
            
            if params['BN']:
                      self.feature_extractor.append(layers.BatchNormalization()) 
                      self.feature_extractor.append(layers.MaxPool1D(pool_size= 2))  
        for i in range(params['num_feat_layers']):
            if self.type=='dense':
                self.feature_extractor.append(layers.Dense(self.feature_nodes, activation=None, 
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=keras.regularizers.l2(self.reg)))
                if params['BN']:
                  self.feature_extractor.append(layers.BatchNormalization())  
                self.feature_extractor.append(layers.Dropout(self.drop_rate))
            elif self.type=='conv2d':
                self.feature_extractor.append(layers.Conv1D(filters=params['filters'], 
                                                            kernel_size=params['kernel_size'], strides=params['stride'], 
                                                            padding='valid',activation=None, 
                                                            kernel_initializer=tf.keras.initializers.he_normal()))
                if params['BN']:
                  self.feature_extractor.append(layers.BatchNormalization()) 
                  self.feature_extractor.append(layers.MaxPool1D(pool_size= 2))  
            else:
                raise NotImplementedError('Layer type not defined')
               
            self.feature_extractor.append(layers.LeakyReLU())
            
        if self.type=='conv2d':
            self.feature_extractor.append(layers.Flatten())
            self.feature_extractor.append(layers.Dense(self.feature_nodes, activation=None, 
                                       kernel_initializer=tf.keras.initializers.he_normal(),
                                       kernel_regularizer=keras.regularizers.l2(self.reg)))
            
            
        #classifier
        self.classifier=[]
        for i in range(params['num_class_layers']-1):
            self.classifier.append(layers.Dense(self.feature_nodes, activation=None, 
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=keras.regularizers.l2(self.reg)))
            if self.BN:
                self.classifier.append(layers.BatchNormalization())
            self.classifier.append(layers.Dropout(self.drop_rate))
            self.classifier.append(layers.LeakyReLU())
            
        #classifier layer, outputs logits
        self.class_out=layers.Dense(self.output_size, activation=None, 
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=keras.regularizers.l2(self.reg))
        
        
        #performance metrics
        self.f1=[]
        self.class_loss=[]
        self.domain_loss=[]

    def get_feature(self,x_in,training=False):
        x=x_in
        activations=[]
        for layer in self.feature_extractor:
            x=layer(x,training=training)
            if 're' in layer.name:
                activations.append(x)
        
        if self.adapt_extractor:
            return activations
        else:
            return [x]

    def classify(self,feature,training=False):
        x=feature
        activations=[]
        for layer in self.classifier:
            x=layer(x,training=training)
            if 're' in layer.name:
                activations.append(x)
        Clogit=self.class_out(x)
        return activations+[Clogit]
    
    def call(self, x_in, train=False):
       
        #feature extractor
        feat_activations=self.get_feature(x_in,training=train)   

        #classifier
        class_activations=self.classify(feat_activations[-1],training=train)
        activations=feat_activations+class_activations
        
        return activations #returns logits
    
    def get_performance(self,x,y):
        #labels assumed to be in categorical 0,1,2 etc
        logit_C=self.call(x)[-1]
        y_pred=np.argmax(logit_C,axis=1)
        f1=f1_score(y, y_pred,average='macro')
        acc=accuracy_score(y,y_pred)
        
        return f1,acc
    


    def plot_loss(self):
        #plot loss
        epochs=len(self.class_loss)
        n=range(1,math.floor(epochs)+1,1)
        plt.plot( n,self.domain_loss, label = "domain loss") 
        plt.plot( n,self.class_loss, label = "class loss") 
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend() 
        plt.show()
        
        #plot F1
        plt.plot( n,self.f1, label = "F1 per 10 epochs") 
        plt.xlabel('epoch')
        plt.ylabel('x_test F1')
        plt.legend() 
        plt.show()
    
    def plot_feature(self,test_data,colour='domain',name=None):
        Xt=test_data[0]
        yt=test_data[1]
        Xs=test_data[2]
        ys=test_data[3]
        nt,_=Xt.shape
        Zs=self.get_feature(Xs)[-1].numpy()
        Zt=self.get_feature(Xt)[-1].numpy()
        Z=np.vstack((Zs,Zt))
        scale=StandardScaler()
        Z=scale.fit(Z).transform(Z)
        pca=PCA()
        pca.fit(Z)
        Zs   =pca.transform(Zs)
        Zt   =pca.transform(Zt)
        cum_var=np.cumsum(pca.explained_variance_ratio_)
        print('Variance in the first 4 PCs: '+str(cum_var[0:4]))
        sub=np.random.randint(0,Zt.shape[0],(min(nt,300),))
        plot_transfer(Zs[sub,:],Zt[sub,:],ys[sub],yt[sub],colour='domain',name=name,xlabel='PC 1', ylabel='PC 2')
        plt.show()

        
    def get_A_distance(self,test_data,K='linear'):
        Xt=test_data[0]
        Xs=test_data[2]
        yt=test_data[1]
        ys=test_data[3]
        Yt=np.unique(yt).shape[0]
        Ys=np.unique(ys).shape[0]
        Zs=self.get_feature(Xs)[-1].numpy()
        Zt=self.get_feature(Xt)[-1].numpy()
        return Proxy_A(Zs,Zt,kernel=K,Ys=Ys,Yt=Yt)
    
    def transductive_PC_knn(self,test_data):
    #train a knn on a the first two PCs of the feature
        Xt=test_data[0]
        yt=test_data[1]
        Xs=test_data[2]
        ys=test_data[3]
        Zs=self.get_feature(Xs)[-1].numpy()#.numpy()
        Zt=self.get_feature(Xt)[-1].numpy()#.numpy()
        Z=np.vstack((Zs,Zt))
        scale=StandardScaler()
        Z=scale.fit(Z).transform(Z)
        pca=PCA()
        pca.fit(Z)
        Zs   =pca.transform(Zs)
        Zt   =pca.transform(Zt)
        knn=KNeighborsClassifier()
        knn.fit(Zs,ys)
        yt_pred=knn.predict(Zt)
        acc=accuracy_score(yt,yt_pred)
        f1=f1_score(yt, yt_pred,average='micro')
        return yt_pred,acc,f1
    
    def get_summary(self):
        self.built=True
        self.summary()
    
