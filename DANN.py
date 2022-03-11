"""domain adversarial neural network

author: Jack Poole, University of Sheffield 

"""




import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from utils import GradReverse, TL_stand,soft_loss,sig_loss,get_lambda,plot_transfer,randomise
import time,tensorflow.keras as keras
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from base_model import base_model
#define training step
optimiser=tf.keras.optimizers.Adam(lr=1e-4,
    beta_1=0.9,
    beta_2=0.999)


class DANN_model(base_model):
    def __init__(self,params={'feature_nodes':10, 
                                'num_feat_layers':2,
                                'num_class_layers':2,
                                'disc_nodes':20,
                                'num_disc_layers':2,
                                'input_dim':(1000,3),
                                'output_size':4,
                                'drop_rate':0.25,
                                'reg':0.0001,
                                'entropy':0,
                                'BN':True,
                                'lr':1e-4,
                                'type':'dense',
                                'filters':10,
                                'kernel_size':5,
                                'stride':1}):
        super().__init__(params)    
        
        #hyperparameters
        self.adapt_extractor=False # the lower layers of extractor are not adapted in DANN
        self.disc_nodes=params['disc_nodes']
        self.reg=params['reg']
        self.BN=params['BN']
        self.entropy=params['entropy']
        self.lr=params['lr']

        
        #domain discriminator
        self.reverse=GradReverse()
        self.discriminator=[]
        for i in range(params['num_disc_layers']-1):
            self.discriminator.append(layers.Dense(self.disc_nodes, activation=None, 
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=keras.regularizers.l2(self.reg)))
            if self.BN:
                self.discriminator.append(layers.BatchNormalization())
            self.discriminator.append(layers.Dropout(self.drop_rate))
            self.discriminator.append(layers.LeakyReLU())
        self.disc_out=layers.Dense(2,activation=None)
        
        #performance metrics
        self.disc_f1=[]
        self.class_loss_valid=[]
        self.disc_loss_valid=[]
        

    
    def call(self, x_in, lmda=1.0, train=False):
       
        #feature extractor
        feat_activations=self.get_feature(x_in,training=train)   

        #classifier
        class_activations=self.classify(feat_activations[-1],training=train)
    
        #domain discriminator
        d=self.reverse(feat_activations[-1],lmda)
        for layer in self.discriminator:
            d=layer(d)
        Dlogit=self.disc_out(d)
        
        return (Dlogit,class_activations[-1]) #returns logits of discriminator and classifier
    
    def predict(self,x,y):
        logit_C=self.call(x)[-1]
        return np.argmax(logit_C,axis=1)
    
    def get_performance(self,x,y):
        #labels assumed to be in categorical 0,1,2 etc
        y_pred=self.predict(x,y)
        acc=accuracy_score(y,y_pred)
        f1=f1_score(y, y_pred,average='macro')
        
        return f1,acc
    
    def get_disc_loss(self,ds,dt):
   
        yd=tf.concat([ds[0],dt[0]],0)
        domain= np.vstack([np.tile([1., 0.], [ds[0].shape[0], 1]),
                           np.tile([0., 1.], [dt[0].shape[0], 1])]).astype('float32')
        disc_loss =sig_loss(yd,domain)
        return disc_loss
    
    def get_disc_f1(self,test_data):
        Xt=test_data[0]
        Xs=test_data[2]
        ds=self.call(Xs)
        dt=self.call(Xt)
        yd=tf.concat([ds[0],dt[0]],0)
        domain= np.vstack([np.tile([1], [ds[0].shape[0], 1]),
                               np.tile([0], [dt[0].shape[0], 1])]).astype('float32')
        y_pred=np.argmax(yd,axis=1)
        f1=f1_score(domain, y_pred,average='micro')
        #acc=accuracy_score(domain,y_pred)
        return f1
    
    def get_valid_loss(self,test_data):
        Xt=test_data[0]
        Xs=test_data[2]
        C=self.params['output_size']
        ys=tf.one_hot(test_data[3],C)
        source_activation=self.call(Xs)
        target_activation=self.call(Xt)
        class_loss=soft_loss(source_activation[-1],ys)
        disc_loss=self.get_disc_loss(source_activation,target_activation)
        return class_loss,disc_loss
   
   # @tf.autograph.experimental.do_not_convert
    @tf.function 
    def train_step(self,Xs,ys,Xt,lmda=1.0):

        with tf.GradientTape() as tape:
            
            ds =self.call(Xs,lmda,train=True)
            dt =self.call(Xt,lmda,train=True)
            class_loss=soft_loss(ds[-1],ys)
            
            pt=tf.nn.softmax(dt[-1],axis=1)
            h=tf.multiply(pt,tf.math.log(pt))            
            target_entropy=-tf.reduce_mean(h,axis=0)
            disc_loss =self.get_disc_loss(ds,dt)
            loss=class_loss+disc_loss+tf.add_n(self.losses)+self.entropy*target_entropy
            
        #get gradients 
        gradients= tape.gradient(loss, self.trainable_variables)
   
        #update
        optimiser.apply_gradients(zip(gradients, self.trainable_variables)) 
        
        return class_loss,disc_loss
        
    def train(self,dataset,test_data,epochs=10,pretrain=False):
        """Inputs: dataset - tf dataset with  the source features, source labels and target features Xs,ys,Xt
                   test_data - list of (Xt_test, yt_test, Xs, ys)
                   epochs - number of training epochs
                   pretrain - if true the source data is used to train the feature extractor and classifier"""
        tf.keras.backend.set_value(optimiser.lr, self.lr)

        for epoch in range(epochs):
            #init cumulative loss for each epoch
            cum_class_loss=0
            cum_disc_loss =0
            
            start= time.time()

            #set hyperparameters for epoch
            progress=epoch/epochs
            if pretrain:
                lmda=0.0
            else:
                lmda=(get_lambda(progress)).astype('float32')
  
            #train one batch
            for Xs,ys,Xt in dataset:
            
                class_loss,disc_loss=self.train_step(Xs,ys,Xt,lmda)
                cum_class_loss+=class_loss
                cum_disc_loss+=disc_loss
            
            
            if epoch%100==0:
                #print metrics
                print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                f1,acc=self.get_performance(test_data[0],test_data[1])
                f1s,accs=self.get_performance(test_data[2],test_data[3])
                print('F1: '+str(f1)+' for source: '+str(f1s)+' lmda: '+ str(lmda))
            
            #update metric lists
            self.class_loss.append(cum_class_loss)
            self.domain_loss.append(cum_disc_loss)
            self.f1.append(f1)
            
            self.class_loss_valid.append(self.get_valid_loss(test_data)[0])   
            self.disc_loss_valid.append(self.get_valid_loss(test_data)[1])
            self.disc_f1.append(self.get_disc_f1(test_data))

    
    def get_summary(self):
        self.built=True
        self.summary()
    

