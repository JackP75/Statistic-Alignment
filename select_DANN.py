"""Select the top performing DANN in an unsupervised manner

author: Jack Poole, University of Sheffield 

"""

from DANN import DANN_model
import numpy as np
import os

#Find the best result for a number of repeats
def evaluate_once(model, dataset, test_data, best_loss=1000, model_f1=[], pretrain=False, name='DANN', epochs=500, method = 'adversarial'):   
    "Train a model and return the best loss and append its F1 score"
    if pretrain==True:
        model.train(dataset,test_data,epochs,pretrain=True)
        
    model.train(dataset,test_data,epochs)
    class_loss,domain_loss=model.get_valid_loss(test_data)
    loss=class_loss.numpy()-domain_loss.numpy()

    model_f1.append(model.f1[-1])
        
    if  loss<best_loss:
        best_loss=loss
        try:
            model.save_weights('.\\'+name+'\\'+str(best_loss)+'.h5')
        except:
            os.mkdir('.\\'+name)
            model.save_weights('.\\'+name+'\\'+str(best_loss)+'.h5')
            
    return best_loss,model_f1

def evaluate_DANN(params,dataset, test_data, epochs, repeats, method='adversarial', name='DANN', pretrain='False'):
    "Train a DANN for n repeats and select the model with the lowest overall loss"
    best_loss=1000
    model_f1=[]    
    
    for n in range(repeats):
        model=DANN_model(params)
        best_loss,model_f1=evaluate_once(model,dataset,test_data,best_loss,model_f1,pretrain,name,epochs,method )

    model.built=True
    if best_loss!=1000:
        model.load_weights('.\\'+name+'\\'+str(best_loss)+'.h5')
        f1,acc=model.get_performance(test_data[0],test_data[1])
        PAD=model.get_A_distance(test_data,K='rbf')
        print('Best F1 on test set: '+str(f1))
        print('Average F1 scores: '+str(np.mean(model_f1)))
    else:
        print('No runs converged sucessfully!')
        best_loss='Fail'
        f1=0
        PAD=2
    return f1,best_loss, model_f1,PAD

