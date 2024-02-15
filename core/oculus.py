"""
Core Component:  oculus caster
    naming from:

    The Oculus 
    Swirling Crystal
    http://classic.battle.net/diablo2exp/items/normal/usorceress.shtml

    The forseer, the prophet.

    Classes: 
    -----------
        Oculus: core class, model trainer, predictor

    Functions:
    -----------
"""

import numpy as np
import pandas as pd
import json

# LASSO
from sklearn.linear_model import LassoCV, Lasso

# SVM
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from imblearn.ensemble import BalancedRandomForestClassifier

# CNN
import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F

# grid search
from sklearn.model_selection import GridSearchCV as gscv

# calculate metrics
import sklearn.metrics as skm

import joblib as jl
import lib.utils as utils

print_prefix='core.oculus>>'



class Oculus:
    '''
    oculus caster: core class, model trainer, predictor
    '''
    def __init__(self, etl_mgr):

        self.Xnames=etl_mgr.Xnames 
        self.Xfile=etl_mgr.feature_lib_file.split('.')[0]
        self.infer_file=etl_mgr.infer_file.split('.')[0]
        self.model_name=etl_mgr.model_name
        self.iteration_times=etl_mgr.iteration_times
    
    def train(self, cfg, X_train, y_train):
        ''' train the predictor for tasks '''
        
        utils.write_log(print_prefix+'training...')
        utils.write_log(print_prefix+self.model_name+' training...')
        
        self.cv=int(cfg['CORE']['cv'])
        self.score_method=cfg['CORE']['gs_score']
        if self.model_name=='lasso':

            # below for lassocv
            lassocv_model=LassoCV(
                    max_iter=self.iteration_times,cv=self.cv).fit(
                            X_train,y_train)
            magic_alpha = lassocv_model.alpha_
            
            utils.write_log(print_prefix+'lasso best alpha: %7.5f' % magic_alpha)
            # above for lassocv

            # training
            lasso_model=Lasso( alpha=magic_alpha)
            lasso_model.fit(X_train, y_train)

            self.built_model=lasso_model
        
        if self.model_name=='svm':
            svm_C=float(cfg['CORE']['svm_C'])
            svm_epsilon=float(cfg['CORE']['svm_epsilon'])

            svm_model=make_pipeline(StandardScaler(), 
                    SVR(C=svm_C, epsilon=svm_epsilon,max_iter=self.iteration_times))
            svm_model.fit(X_train, y_train)
            self.built_model=svm_model


        if self.model_name in ['random_forestC', 'random_forestR']:
            
            rf_trees=utils.get_para_int_list(cfg['CORE']['rf_max_trees'])
            rf_depth=utils.get_para_int_list(cfg['CORE']['rf_max_depth'])
            njobs=int(cfg['CORE']['ntasks'])
            
            param_grid = [
                    {'n_estimators': rf_trees, 'max_depth': rf_depth}]
                

        if self.model_name=='random_forestC':
            if cfg['CORE']['label_standardize']=='False':
                #rf_model = RandomForestClassifier(
                #       n_estimators=rf_trees, max_depth=rf_depth, random_state=0, n_jobs=njobs) 
                utils.write_log(print_prefix+'gridsearch with scoring method: %s' % self.score_method)
                #rf_model = RandomForestClassifier()
                rf_model = BalancedRandomForestClassifier(random_state=42)
                grid_search = gscv(
                        rf_model, param_grid, cv=self.cv, 
                        verbose=1, scoring=self.score_method,n_jobs=njobs)
            else:
                utils.throw_error(print_prefix+' cannot use random forest classifier on standardized classification tags!')
            
            grid_search.fit(X_train, y_train)
            utils.write_log(print_prefix+'gridsearch best paras:')
            print(grid_search.best_params_)

            best_rf_model=grid_search.best_estimator_
            self.built_model=best_rf_model

        if self.model_name=='random_forestR':
            rf_model = RandomForestRegressor(n_estimators=rf_trees, max_depth=rf_depth, random_state=0, n_jobs=njobs) 
            rf_model.fit(X_train, y_train)
            self.built_model=rf_model

        if self.model_name=='cnn':
            BATCH_SIZE=int(cfg['CORE']['cnn_batch_size'])
            EPOCH=int(cfg['CORE']['cnn_epoch'])
            n_hidden=int(cfg['CORE']['cnn_n_hidden'])


            x_cnn = Variable(torch.tensor(X_train.astype(np.single)))
            y_cnn = Variable(torch.tensor(y_train[:, np.newaxis].astype(np.single)))
           
            xsize=X_train.shape

            nn_model = Net(n_feature=xsize[1], n_hidden=n_hidden) 
            
            optimizer = torch.optim.Adam(nn_model.parameters(), lr=float(cfg['CORE']['cnn_lr']))
            loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss 
            
            torch_dataset = Data.TensorDataset(x_cnn, y_cnn)

            loader = Data.DataLoader(
                dataset=torch_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True, num_workers=int(cfg['CORE']['ntasks']),)


            # start training
            for epoch in range(EPOCH):
                utils.write_log(print_prefix+'training epoch:'+str(epoch+1))
                for step, (batch_x, batch_y) in enumerate(loader): # for each training step
                    
                    b_x = Variable(batch_x)
                    b_y = Variable(batch_y)

                    prediction = nn_model(b_x)     # input x and predict based on x
                    loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
                    optimizer.zero_grad()   # clear gradients for next train
                    loss.backward()         # backpropagation, compute gradients
                    optimizer.step()        # apply gradients

                utils.write_log(print_prefix+'training epoch:'+str(epoch+1)+' done. Loss = %.4f' % loss.data.numpy())
            
            self.built_model=nn_model

        if not(hasattr(self, 'built_model')):
            utils.throw_error(print_prefix+'[CORE][model_name]='+self.model_name+' not exist!')

    def evaluate(
            self, etl_mgr, X_train, X_test, y_train, y_test, lb):
        ''' print evaluation matrix '''
        
        model = self.built_model
        model_name=self.model_name

        # evaluation dict
        edic={'model':model_name, 'yname':lb}
        
        # ---------- Special for NN -----------
        if model_name == 'cnn':
            x_train_torch = Variable(torch.tensor(X_train.astype(np.single)))
            y_train_torch = Variable(torch.tensor(y_train[:, np.newaxis].astype(np.single)))
            
            x_test_torch = Variable(torch.tensor(X_test.astype(np.single)))
            y_test_torch = Variable(torch.tensor(y_test[:, np.newaxis].astype(np.single)))

            y_train_p_torch = model(x_train_torch)
            y_test_p_torch = model(x_test_torch)
            
            y_train_predict=y_train_p_torch.detach().numpy()
            y_test_predict=y_test_p_torch.detach().numpy()


        # ---------- Common for sklearn -----------
        # predict by the built model
        if model_name in ['lasso', 'random_forestR', 'random_forestC', 'svm']:
            y_train_predict = model.predict(X_train)
            y_test_predict = model.predict(X_test)
           
        if model_name == 'lasso':
            w=model.coef_
            b=model.intercept_
            features=np.where(w!=0)[0]
            
            edic.update({
                'w':w[w!=0].tolist(),
                'w_idx':features.tolist(),
                'w_name':[self.Xnames[itm] for itm in features],
                'b':b,
            })
            
        if model_name in ['lasso', 'random_forestR', 'cnn', 'svm']:
            edic.update({
                    'exp_var_train':skm.explained_variance_score(y_train, y_train_predict),
                    'exp_var_test':skm.explained_variance_score(y_test, y_test_predict),
                    'RMSE_train':skm.mean_squared_error(y_train, y_train_predict, squared=False),
                    'RMSE_test':skm.mean_squared_error(y_test, y_test_predict, squared=False),
                    })
        else:
            edic.update({
                    'acc_score_train':skm.accuracy_score(y_train, y_train_predict),
                    'acc_score_test':skm.accuracy_score(y_test, y_test_predict),
                    #'balanced_acc_score_train':skm.balanced_accuracy_score(y_train, y_train_predict),
                    #'balanced_acc_score_test':skm.balanced_accuracy_score(y_test, y_test_predict),
                    'recall_train':skm.recall_score(y_train, y_train_predict),
                    'recall_test':skm.recall_score(y_test, y_test_predict),
                    'precision_train':skm.precision_score(y_train, y_train_predict),
                    'precision_test':skm.precision_score(y_test, y_test_predict),
                    'f1_train':skm.f1_score(y_train, y_train_predict),
                    'f1_test':skm.f1_score(y_test, y_test_predict),
                    })

        if etl_mgr.std_label:
            edic.update({'ymean':etl_mgr.amean[lb],'ystd':etl_mgr.astd[lb]})
        
        self.edic=edic
        
        if model_name == 'random_forestC':
            importance=model.feature_importances_
            indices=np.argsort(importance)[::-1]
            for idx in range(len(self.Xnames)):
                #print('%s--%f' % (self.Xnames[indices[idx]], importance[indices[idx]]))
                edic.update({
                    self.Xnames[indices[idx]]:importance[indices[idx]]
                    })

       
        utils.write_log(print_prefix+model_name+' model evaluation dict:')  
        print(edic)

    def archive(self,lbname):
        '''archive the model and evaluation dict'''
        
        # archive evaluation dict
        with open(
                './db/'+self.Xfile+'.'+lbname+'.'+self.model_name+'_edic.json', 'w') as f:
            json.dump(self.edic,f)
       
        # archive cnn using torch
        if self.model_name == 'cnn':
            torch.save(
                    self.built_model,
                    './db/'+self.Xfile+'.'+lbname+'.'+self.model_name+'.torch')
        else:
            # archive model by joblib
            jl.dump(
                    self.built_model,
                    './db/'+self.Xfile+'.'+lbname+'.'+self.model_name+'.jl')

    def load(self, lbname):
        # load model
        if self.model_name == 'cnn':
            self.built_model=torch.load(
                    './db/'+self.infer_file+'.'+lbname+'.'+self.model_name+'.torch')
            self.built_model.eval()
        else:
            self.built_model=jl.load(
                    './db/'+self.infer_file+'.'+lbname+'.'+self.model_name+'.jl')
        
        # load evalution dict
        with open(
                './db/'+self.infer_file+'.'+lbname+'.'+self.model_name+'_edic.json', 'r') as json_file:
            self.edic = json.load(json_file)

    def cast(self, etl_mgr,  lbname):
        ''' cast predictor for inference tasks '''
        utils.write_log(print_prefix+self.model_name+' cast, with model edic:')
        print(self.edic)

        X=etl_mgr.X
        date_range=etl_mgr.Xdate
        
        model=self.built_model
        
        if self.model_name == 'cnn':
            x_torch = Variable(torch.tensor(X.astype(np.single)))
            y_torch = model(x_torch)
            y_predict=y_torch.detach().numpy()
        else:
            y_predict =model.predict(X)
        
        if 'ymean' in self.edic.keys():
            y_mean=self.edic['ymean']
            y_std=self.edic['ystd']
            y_predict =y_std*y_predict+y_mean
        
        if self.model_name == 'random_forestC':
            y_pred_proba=model.predict_proba(X)
            y_predict=y_predict[:, np.newaxis]
            y_predict=np.append(y_predict,y_pred_proba,axis=1)
            result=pd.DataFrame(y_predict,index=date_range,columns=[lbname,'prob0', 'prob1'])
        else:
            result=pd.DataFrame(y_predict,index=date_range,columns=[lbname])

        result.to_csv('./output/'+self.infer_file+'.'+lbname+'.'+self.model_name+'.predict.csv')
'''
nn_model = torch.nn.Sequential(
    torch.nn.Linear(93, 10),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(10, 5),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(5, 1),
)

            nn_model = torch.nn.Sequential(
                    torch.nn.Linear(93, 5),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(5, 1),
                )
'''


# define the network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, 1)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x



