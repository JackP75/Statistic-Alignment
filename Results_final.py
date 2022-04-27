"""Run results for numerical population

author: Jack Poole, University of Sheffield 

"""

import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import tensorflow as tf
from matplotlib.transforms import Bbox

from TCA import TCA
from GFK import GFK
from BDA import BDA
from CORAL import CORAL
from select_DANN import evaluate_DANN
from utils import Proxy_A,randomise
from NCA import NCA

# np.random.seed(9)
# tf.random.set_seed(9)

def TL_with_standardisation(Xs,Xt,x_test):
    "Uses standardisation to align the means (mu=0) and standard deviations (std=1) of two datasets Xs and Xt"
    stand=StandardScaler()
    Xs=stand.fit_transform(Xs)
    Xt=stand.fit_transform(Xt)
    x_test=stand.transform(x_test)
    return Xs, Xt, x_test

def pairplt(X_s,X_t,y_s,y_t,title):
    "creates a pairplot for 3dim features, which is a plot of all the possible combinations of features"
    
    ns,nt=y_s.shape[0],y_t.shape[0]
    
    X=np.vstack((X_s,X_t))
    y=np.vstack((y_s.reshape(-1,1),y_t.reshape(-1,1)))
    
    ind=np.vstack((np.zeros((ns,1)),np.ones((nt,1)))) #labels for the domains, source=0, target=1
    
    #put the data into dataframes (this is only required for seaborn)
    frame=DataFrame(np.hstack((ind,X)),columns=['Domain','$\omega_{d,1}$','$\omega_{d,2}$','$\omega_{d,3}$'])
    frame2=DataFrame(y,columns=['Class']) 
    
    #g is a grid of subplots
    g = sns.PairGrid(frame,hue='Domain')
    
    #estimates the distributions of the data using kernel density estimation (dont worry about the maths)
    g.map_diag(sns.kdeplot)
    g.map_lower(sns.kdeplot)
    
    g.map_upper(sns.scatterplot,style=frame2['Class'],s=100)
    g.add_legend(title="",adjust_subtitles=False)
    g.fig.suptitle(title, y=1.08)
    plt.show()

def knn_f1(Xs,ys,xtest,ytest):
    knn=KNeighborsClassifier(n_neighbors=1).fit(Xs,ys)
    pred=knn.predict(xtest)

    return f1_score(np.ravel(ytest), pred, average='macro')
          
def DA_methods(X_s,X_t,ys,yt,x_test,y_test,reps=1):
    "For a given source and target dataset get the macro-F1 scores and PADs for the GFK, TCA, BDA and the DANN"
    n_t=X_t.shape[0]

    gfk=GFK()
    gfk.fit(X_s,X_t,1)
    gfkA=Proxy_A(X_s,X_t,kernel='rbf')
    gfk.train(X_s,ys,KNeighborsClassifier())
    _,_,gfkF1=gfk.predict(x_test,y_test)
    
    tca=TCA()
    Xs,Xt=tca.fit(X_s,X_t)
    tca.train(X_s,ys) 
    tcaA=Proxy_A(Xs,Xt,kernel='rbf')
    preds,acc,tcaF1=tca.predict(x_test,y_test)
    print(preds,acc)
     
    bda=BDA()
    Xs,Xt=bda.fit(X_s,X_t,ys)
    bdaA=Proxy_A(Xs,Xt,kernel='rbf')
    bdaF1=bda.predict(x_test,y_test)[2]
    
    #set up the data for TF
    X_s,ys=randomise(X_s, ys)#data is shuffled so each batch contains a mix of classes
    X_t=np.vstack((X_t,X_t)) #data is fed into the DANN in pairs and X_t is much smaller than X_s
    yt=np.vstack((yt.reshape(-1,1),yt.reshape(-1,1)))
    X_t,yt=randomise(X_t, yt)

    u=np.unique(y_s).shape[0]
    test_data=(x_test,y_test,X_s,ys)
    ys=tf.one_hot(ys, u)
    yt=tf.one_hot(yt, u)
    dataset = tf.data.Dataset.from_tensor_slices((X_s[:n_t*2,:],ys[:n_t*2,:],X_t)).shuffle(200).batch(32).prefetch(tf.data.AUTOTUNE)
    
    params={'feature_nodes':10, 
            'num_feat_layers':2,
            'num_class_layers':2,
            'disc_nodes':20,
            'num_disc_layers':2,
            'input_dim':None,
            'output_size':u,
            'drop_rate':0.00,
            'reg':0.0001,
            'entropy':0,
            'BN':True,
            'lr':1e-4,
            'type':'dense'}
    dannF1,DANN_best_loss, DANN_list_f1,dannA= evaluate_DANN(params,dataset,test_data,
                                                                    epochs=500,repeats=reps,method='adversarial',
                                                                    name='DANN',pretrain=False)
    dann_mean = np.mean(DANN_list_f1)
    dann_std = np.std(DANN_list_f1)
    
    return [gfkF1,tcaF1,bdaF1,dannF1],[gfkA,tcaA,bdaA,dannA],[dann_mean,dann_std,max(DANN_list_f1),min(DANN_list_f1)]

def DA_results(X_s,ys,X_t,yt,x_test,y_test,name,reps=10):
    "Returns F1 scores and PADs for a the N-stand,A-stand,CORAL, NCA, NCORAL, GFK, TCA, BDA and the DANN"
    

    #A-stand
    Xs,Xt,xtest=TL_with_standardisation(X_s,X_t,x_test)
    standA=Proxy_A(Xs,Xt,kernel='rbf')
    standF1=knn_f1(Xs,ys,xtest,y_test)
    
    #CORAL
    coral=CORAL()
    coral.fit(Xs,Xt)
    Xs=coral.transform(Xs)
    coralA=Proxy_A(Xs,Xt,kernel='rbf')
    coralF1=knn_f1(Xs,ys,xtest,y_test)
    
    #NCA
    Xs,Xt,xtest=NCA(X_s,X_t,ys,yt,x_test,y_test)
    alignA=Proxy_A(Xs,Xt,kernel='rbf')
    alignF1=knn_f1(Xs,ys,xtest,y_test)   
    
    #NCORAL
    coral=CORAL()
    coral.fit(Xs,Xt)
    Xs=coral.transform(Xs)
    ncoralA=Proxy_A(Xs,Xt,kernel='rbf')
    ncoralF1=knn_f1(Xs,ys,xtest,y_test)
    
    #N-stand
    stand=StandardScaler().fit(np.vstack((X_s,X_t)))
    Xs=stand.transform(X_s)
    Xt=stand.transform(X_t)
    xtest=stand.transform(x_test)
    rawA =Proxy_A(Xs,Xt,kernel='rbf')
    rawF1=knn_f1(Xs,ys,xtest,y_test)
    
    #[GFK, TCA, BDA, DANN]
    f1s, pads, dann_stats=DA_methods(Xs,Xt,ys,yt,xtest,y_test,reps)
    
    f1S,padS=[rawF1,standF1,coralF1,alignF1,ncoralF1]+f1s,[rawA,standA,coralA,alignA,ncoralA]+pads
    
    return f1S,padS, dann_stats

def pre_DA_results(X_s,ys,X_t,yt,x_test,y_test,name,reps=100):
    
    stand=StandardScaler().fit(np.vstack((X_s,X_t)))
    Xs=stand.transform(X_s)
    Xt=stand.transform(X_t)
    xtest=stand.transform(x_test)
    rawA =Proxy_A(Xs,Xt,kernel='rbf')
    rawF1=knn_f1(Xs,ys,xtest,y_test)
    #pairplt(Xs,Xt,y_s,y_t,name+'stand')
    f1s,pads, dannstats1=DA_methods(Xs,Xt,ys,yt,xtest,y_test, reps)
    f1_col1,A_col1=[rawF1]+f1s,[rawA]+pads
    
   
    #Astand
    Xs,Xt,xtest=TL_with_standardisation(X_s,X_t,x_test)
    standA=Proxy_A(Xs,Xt,kernel='rbf')
    standF1=knn_f1(Xs,ys,xtest,y_test)
    f1s,pads, dannstats2=DA_methods(Xs,Xt,ys,yt,xtest,y_test, reps)
    f1_col2,A_col2=[standF1]+f1s,[standA]+pads
   
    #coral
    Xs,Xt,xtest=TL_with_standardisation(X_s,X_t,x_test)
    coral=CORAL()
    coral.fit(Xs,Xt)
    Xs=coral.transform(Xs)
    coralA=Proxy_A(Xs,Xt,kernel='rbf')
    coralF1=knn_f1(Xs,ys,xtest,y_test)
    f1s,pads, dannstats3=DA_methods(Xs,Xt,ys,yt,xtest,y_test, reps)
    f1_col3,A_col3=[coralF1]+f1s,[coralA]+pads
    
    #nca
    Xs,Xt,xtest=NCA(X_s,X_t,ys,yt,x_test,y_test)
    alignA=Proxy_A(Xs,Xt,kernel='rbf')
    alignF1=knn_f1(Xs,ys,xtest,y_test)   
   # pairplt(Xs,Xt,y_s,y_t,name+'align')
    f1s,pads, dannstats4=DA_methods(Xs,Xt,ys,yt,xtest,y_test, reps)
    f1_col4,A_col4=[alignF1]+f1s,[alignA]+pads

     
    #ncoral
    Xs,Xt,xtest=NCA(X_s,X_t,ys,yt,x_test,y_test)
    coral=CORAL()
    coral.fit_normal(Xs,Xt,y_s,y_t)
    Xs=coral.transform(Xs)
    coralA=Proxy_A(Xs,Xt,kernel='rbf')
    coralF1=knn_f1(Xs,ys,xtest,y_test)
   # pairplt(Xs,Xt,y_s,y_t,name+'coral')
    f1s,pads, dannstats5=DA_methods(Xs,Xt,ys,yt,xtest,y_test, reps)
    f1_col5,A_col5=[coralF1]+f1s,[coralA]+pads 
       
    f1_array=np.array((f1_col1,f1_col2,f1_col3,f1_col4,f1_col5))
    A_array=np.array((A_col1,A_col2,A_col3,A_col4,A_col5))
    stats_array = np.array((dannstats1, dannstats2, dannstats3, dannstats4, dannstats5))
    
    return f1_array,A_array, stats_array
 #%% 3-storey population
methods=['N-Stand','A-Stand','CORAL','NCA','NCORAL','GFK','TCA','BDA','DANN'] 

name='3_to_3.p'
X_s,y_s,X_t,y_t,X_test,y_test=np.load('.\\data\\3_to_3.p', allow_pickle=True)
pairplt(X_s,X_t,y_s,y_t,name)
f1_list,pad_list,stats_3_to_3=DA_results(X_s,y_s,X_t,y_t,X_test,y_test,name,reps=100)

#put data in frame
df1=DataFrame(data=np.array((f1_list,pad_list)),columns=methods,index=['F1 Score','PAD'])


 #%% 3-storey population w/ class imbalance
 
name='3_to_3_imbalance.p'
X_s,y_s,X_t,y_t,X_test,y_test=np.load(".\\data\\"+name, allow_pickle=True)
pairplt(X_s,X_t,y_s,y_t,name)
f1_list,pad_list,stats_3_to_3imb=DA_results(X_s,y_s,X_t,y_t,X_test,y_test,name,reps=100)

#put data in frame
df2=DataFrame(data=np.array((f1_list,pad_list)),columns=methods,index=['F1 Score','PAD'])


#%% 3- to 7-storey population 

name='3_to_7.p'
X_s,y_s,X_t,y_t,X_test,y_test=np.load(".\\data\\"+name, allow_pickle=True)
pairplt(X_s,X_t,y_s,y_t,name)
f1_array,pad_array, stats_array_3_to_7=pre_DA_results(X_s,y_s,X_t,y_t,X_test,y_test,name,reps=100)

#put data in frame
df3=DataFrame(data=f1_array.T,columns=['N-Stand','A-Stand','CORAL','NCA','NCORAL'], index=['only stat align','GFK','TCA','BDA','DANN'])

df4=DataFrame(data=pad_array.T,columns=['N-Stand','A-Stand','CORAL','NCA','NCA+CORAL'], index=['only stat align','GFK','TCA','BDA','DANN'])


#%% plot results 
#bar chats showing f1 and PAD for part 1

def plot_bar(df,name,stats,methods):
    
    leg=df.columns
    X = np.arange(len(leg))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    methods=methods[:-1]
    
    ax.bar(X[:-1] , df.iloc[[0]][methods].squeeze().to_numpy().T, width = 0.9,alpha=0.9)
    ax.bar(X[-1] , stats[0], yerr = stats[1], width = 0.9,alpha=0.9, color='tab:blue')

    ax.set_xticks(X-0.2)
    ax.set_xticklabels(leg)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation=45)
    h1,l1=ax.get_legend_handles_labels()
    ax.legend(h1,l1, bbox_to_anchor=(1.4, 1.0),fontsize=14)
    ax.set_xlabel('DA Method',fontsize=18)
    ax.set_ylabel('F1 score',fontsize=18)
    #plt.savefig('.\\results\\'+name+'bar(paper).png',bbox_inches=Bbox([[-1, -1], [9, 5]]))
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_axes([0,0,1,1])
    ax2.bar(X,df.iloc[[1]].squeeze().to_numpy().T,color='orange', width = 0.9,alpha=0.9)
    ax2.set_xticks(X-0.2)
    ax2.set_xticklabels(leg)
    plt.setp(ax2.get_xticklabels(), fontsize=14, rotation=45)
    h2,l2=ax2.get_legend_handles_labels()
    ax2.legend(h2,l2, bbox_to_anchor=(1.4, 1.0),fontsize=14)
    ax2.set_xlabel('DA Method',fontsize=18)
    ax2.set_ylabel('PAD',fontsize=18) 
    #plt.savefig('.\\results\\'+name+'bar(paper).png',bbox_inches=Bbox([[-1, -1], [9, 5]]))
    plt.show()

def plot_bar2(df, name, y_label, stats):
    leg=df.columns
    methods=df.index
    methods=['KNN']+list(methods[1:])
    X = np.arange(len(leg))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    #ax.set_ylim(bottom=min(F1t_test)-0.05)
    
    #second axis
    off=0
    delta=0.9/len(methods)
    count=0
    
    for method in methods:
        if method == 'DANN':
            ax.bar(X + off, stats[:,0], yerr = stats[:,1], width = delta, label=method, alpha=0.9)
        else:
            ax.bar(X + off, df.iloc[[count]].squeeze().to_numpy(), width = delta, label=method, alpha=0.9)
        off+=delta
        count+=1 
    
    ax.set_xticks(X+0.3)
    ax.set_xticklabels(leg)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation=45)
    h1,l1=ax.get_legend_handles_labels()
    ax.legend(h1,l1, bbox_to_anchor=(1.4, 1.0),fontsize=14)
    ax.set_xlabel('DA Method',fontsize=18)
    ax.set_ylabel(y_label,fontsize=18)
    #plt.savefig('.\\results\\'+name+'bar(paper).png',bbox_inches=Bbox([[-1, -1], [9, 5]]))
    plt.show()
    
plot_bar(df1,'3_to_3',stats_3_to_3,methods)
plot_bar(df2,'3_to_3_im',stats_3_to_3imb,methods)
plot_bar2(df3,'3_to_7','F1 Score', stats_array_3_to_7) #[['N-Stand','A-Stand','CORAL','NCA','NCA+CORAL']]
plot_bar2(df4,'3_to_7_pad','PAD')
