# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:42:26 2021

@author: mep20jap
"""
import tensorflow as tf
import numpy as np
from  tensorflow.keras import layers
import math,random,os,imageio
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
import scipy.io as spio

plt.style.use('seaborn-whitegrid')

#gradient reverse layer
@tf.custom_gradient
def grad_reverse(x,lmda=1.0):
    y = tf.identity(x)
    def custom_grad(dy):
        return lmda*-dy,None      #None is for grad regarding lmda
    return y, custom_grad

class GradReverse(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x,lmda=1.0):
        return grad_reverse(x,lmda)
    
    def get_config(self):
        config={}#not sure if this will work
        
def soft_loss(yPred,yTrue):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yPred, labels=yTrue))

def sig_loss(yPred,yTrue):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=yPred, labels=yTrue))

#lmda scheduler
def get_lambda(progress):
    return 2. / (1+np.exp(-10.*progress)) - 1. #higher value?

#lr schedule
def lr_schedule(progress, initial_lr, alpha=10.0, beta=0.75):
    return initial_lr / pow((1 + alpha * progress), beta)

#convert labels to 0,1,2 etc
def encode(y):
    "Change to binary encoding"
    enc=LabelEncoder()
    label_encoder=enc.fit(y)
    y_enc=label_encoder.transform(y)
    return y_enc

#shuffles data
def randomise(x,y):
    index=np.random.permutation(x.shape[0])
    x=x[index,:]
    y=y[index]
    return x,y

def TL_stand(Xs,Xt,ys,yt,Xtest=None,ytest=None,partial=False):
     #inputs:
    #Xs- source data (ns,d)
    #Xt -target data (nt,d)
    #ys -normal condition labels [1,0] (ns,1) (or labels in common between source and target)
    #yt -normal condition labels (nt,1)
    #Note: if more than one class is used the labels should be balanced
    
    #1) normalise the source domain
    mu_s=np.mean(Xs,axis=0) #scaling by either makes sense? multi-source needs to be one, target most sensible
    lmda_s=np.std(Xs,axis=0)
    Xs=(Xs-mu_s)/lmda_s #how can variance be used???
    
    #2) match std of known classes
    Xs_n=Xs[np.where(ys == 0)[0],:]
    mu_sn=np.mean(Xs_n,axis=0) #difference in the means for normal conidition
    mu_t=np.mean(Xt,axis=0)

    if partial:
        #3) adapt the target variance of only the normal condition
        #for this to be adapted to operation a novelty dector could specify whether to do this
        Xt_n=Xt[np.where(yt == 0)[0],:]
        lmda_sn=np.std(Xs_n,axis=0)
        lmda_tn=np.std(Xt_n,axis=0)
        Xt=(Xt-mu_t)*lmda_sn/lmda_tn
        Xt_n=Xt[np.where(yt == 0)[0],:]
        mu_tn=np.mean(Xt_n,axis=0)
        #4) bring the known target mean to the corresponding source mean
        Xt=(Xt-mu_tn+mu_sn)     
        if Xtest is not None:   
            Xtest=(Xtest-mu_t)*lmda_sn/lmda_tn
            Xtest=(Xtest-mu_tn+mu_sn)    

            return Xs,Xt,Xtest
    else:
        lmda_t=np.std(Xt,axis=0)
        Xt=(Xt-mu_t)/lmda_t
        Xt_n=Xt[np.where(yt == 0)[0],:]
        mu_tn=np.mean(Xt_n,axis=0)
        Xt=(Xt-mu_tn+mu_sn)     
        if Xtest is not None:   
            Xtest=(Xtest-mu_t)/lmda_t
            Xtest=(Xtest-mu_tn+mu_sn)    
            return Xs,Xt,Xtest
    
    return Xs,Xt

def plot_data(X_s,X_t,y_s,y_t):
    #creates a pairplot of two domains (3 features)
    y=np.vstack((y_s.reshape(-1,1),y_t.reshape(-1,1)))
    x=np.vstack((X_s[:,:3],X_t[:,:3]))
    print(x.shape,y.shape)
    ind=np.vstack((np.zeros(y_s.reshape(-1,1).shape),np.ones(y_t.reshape(-1,1).shape))).reshape(-1,1)

    frame=DataFrame(np.hstack((ind.reshape(-1,1),x)),columns=['y','1','2','3'])
    sns.pairplot(data=frame,hue='y')
    plt.show()
    
def plot_transfer(Xs,Xt,ys,yt,colour='Domain',xlabel='Axis 1', ylabel='Axis 2',name=None):
    X=np.vstack((Xs,Xt))
    ind=np.vstack((np.zeros((Xs.shape[0],1)),np.ones((Xt.shape[0],1))))
    y=np.vstack((ys.reshape(-1,1),yt.reshape(-1,1)))
    inp=np.hstack((ind,np.hstack((y,X))))
    cols=['Domain','Class']
    for i in range(X.shape[1]):
        cols.append('Component '+str(i+1))
    dater=DataFrame(data=inp,columns=cols)
    

    # if colour=='Domain':
    g=sns.JointGrid()
    sns.scatterplot(data=dater, x="Component 1", y="Component 2", hue="Domain", style="Class", ax=g.ax_joint,s=200,legend=True)
    sns.kdeplot(data=dater, x="Component 1", hue="Domain",fill=True,ax=g.ax_marg_x,legend=False)
    sns.kdeplot(data=dater, y="Component 2", hue="Domain",fill=True,ax=g.ax_marg_y,legend=False)
    # else:
    #     g=sns.JointGrid()
    #     sns.scatterplot(data=dater, x="Component 1", y="Component 2", hue="Class", style="Domain", ax=g.ax_joint,palette=sns.color_palette()[:np.unique(y).shape[0]],legend=False)
    #     sns.kdeplot(data=dater, x="Component 1", hue="domain",fill=True,ax=g.ax_marg_x,legend=False,s=200)
    #     sns.kdeplot(data=dater, y="Component 2", hue="domain",fill=True,ax=g.ax_marg_y,legend=False)
    g.ax_joint.set_xlabel(xlabel,fontsize=22)
    g.ax_joint.set_ylabel(ylabel,fontsize=22)
    if name is not None:
        print('Saving figure...')
        plt.savefig('fig/'+name+'.png')

    plt.show()

def drop_class(x,y,clss=2):
    
    ind=np.where(y == clss)[0]
    xnew=np.delete(x,ind,axis=0)
    ynew=np.delete(y,ind,axis=0)
    return xnew,ynew


from sklearn.svm import SVC

def Proxy_A(Xs,Xt,kernel='linear',Ys=1,Yt=1):
    ns=Xs.shape[0]
    nt=Xt.shape[0]
    Xs=Xs[:nt,:]
    y=np.vstack((np.zeros((nt,1)),np.ones((nt,1))))
    X=np.vstack((Xs,Xt))
    svm=SVC(kernel=kernel,probability=True)
    svm.fit(X,y)
    pred=svm.predict(X)
    error=1-svm.score(X,y)#1/(ns+nt)*np.sum(np.abs(pred.reshape(-1,1)-y))
    N=(Ys+Yt)/Yt
    A_dist=2*(1-N*error)
    return A_dist