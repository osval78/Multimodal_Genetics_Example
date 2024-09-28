#Python 3.10 and tensorflow 2.10, Win
import numpy as np
import pandas as pd
import tensorflow as tf
#from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from statsmodels.graphics.api import abline_plot
from sklearn.preprocessing import StandardScaler
# #https://github.com/fmfn/BayesianOptimization
# #!pip install bayesian-optimization
from bayes_opt import BayesianOptimization
import pyreadr, sys, datetime, time, os, random# random.sample
import pickle
from patsy import dmatrix
os.getcwd()

#Models
Dir_Progs = '.'
sys.path.append(Dir_Progs)
from MPGIs_ResNet_MPIs_Vals_V2_PC15Dic23 import *
def MR_f(Params,dat_Dict,verbose=0):    
    nHLB2, UnitsB2 = Params['nHLUnitsB2_ls']#UnitsB2 list
    Units_ls, nHL_ls, temp = [], [], 0
    for i in range(len(dat_Dict['x_ls'])):
        nHLi, Unitsi = Params['nHLs'][temp], Params['Units'][temp]
        Units_ls.append([int(Unitsi/(2**l)) for l in range(nHLi)])
        nHL_ls.append(nHLi)
        temp = temp + 1
    UnitsB2 = [int(UnitsB2/(2**i)) for i in range(nHLB2)]
    nHLUnitsB2_ls = [nHLB2,UnitsB2]
    Params['nHLs'], Params['Units'] = nHL_ls, Units_ls
    Params['nHLUnitsB2_ls'] = nHLUnitsB2_ls
    Ms = MPGIs_f(Params,dat_Dict,verbose=verbose)
    return Ms
#Re-definiendo model MR_f
def M_f(Hypers_a,Hypers_Fixed,dat_Dict,verbose=0):
    Params = {**Hypers_a,**Hypers_Fixed}# changes made in dict2
    Ms = MR_f(Params,dat_Dict,verbose)
    return Ms
M_f
"""
This function evaluates the performance of a deep learning model on validation data using a given set of hyperparameter values (Hypers_a).
The evaluation process is carried out through IK-fold cross-validation, where "Use" subsets are used as validation sets.
"""
def IKFCV_f(Hypers_a,Hypers_Fixed,dat_Dict,KI=5,Use=1):
    y_tr = dat_Dict['y']
    IFold = SIFold(y_tr,K=KI,nq=10,random_state=1)#IKFCV #len(IFold)
    MSEP_v = []#np.repeat(0.01,K)
    #mu, sigma =  np.mean(y_tr), np.std(y_tr)
    #y_tr = (y_tr-mu)/sigma
    for k in range(1,Use+1):
        Pos_ival = list(np.where(IFold==k)[0])#IFold[tst_index] = k
        Hypers_Fixed['Pos_ival'] = Pos_ival
        Ms = M_f(Hypers_a,Hypers_Fixed,dat_Dict,verbose=0)
        MSEP_v.append(Ms['MSEP_Val'])#Cor_v[k-1] = Cor
        #print('Fold=', k,'MSEP=',MSEP,'Cor=',Cor)
    Outs = dict()
    Outs['MSEP'] = MSEP_v#Outs['IFold'] = IFold
    return Outs
# Setting fixed hyperparameters for the model
""" 
Fixed hyperparameters: Epochs, Batch_size, Early Stopping (ES)
Monitor='loss' indicates that the penalized loss function will be monitored. The early 
stopping criteria will depend on this metric and will stop the optimization process involved
in the model's training when the default early stopping criteria of TensorFlow occur.
ValS='Val_dat' indicates that the validation data will be provided by an inner 10-fold cross-validation formed only from the training data.
Number of iterations for Bayesian optimization: Iters
"""
Epochs, Batch_size, ES, Iters = 48, 32, True, 50
Monitor, ValS = 'loss', 'Val_dat'#0.2
Hypers_Fixed =  dict()
Hypers_Fixed['AF'], Hypers_Fixed['typeRN'] =  'relu', 'DB1'# # AF: activation function, and 'DB1': dense block ResNet
Hypers_Fixed['Batch_size'], Hypers_Fixed['Epochs'] = Batch_size, Epochs
Hypers_Fixed['ValS'], Hypers_Fixed['Shuf'] = ValS, True# ValS: validation set, Shuf: shuffle data
Hypers_Fixed['ES'], Hypers_Fixed['Monitor'] =  ES, Monitor
Hypers_Fixed['Plot'],Hypers_Fixed['SaveM']=  False, False# No plot and do not save the trained DL model
Hypers_Fixed['steps_per_epoch']  = None# Number of steps per epoch, None indicates use all baches (number of samples/ Batch_size)
Traits = ["Heading","Germination","Yield","Height","Maturity"] # List of traits that will be used to create the file names of the
# corresponding datasets to be read for performance evaluation prediction.
#Envs_df = pd.DataFrame()
for t in range(1,2):# Change 2 to 6 to evaluate the performance of the best-calibrated model in all traits
    Dir_dat = ["dat-Trait-",str(Traits[t-1]),'.RData']# Define the name of dataset to read
    Dir_dat = ''.join(Dir_dat)
    OD = pyreadr.read_r(Dir_dat)# Read the dataset specified in Dir_dat. This dataset needs to be saved with the .RData extension in R and must contain 
    # the objects 'Pheno', 'G', and 'Fold', which are the phenotypic data, the genomic relationship matrix, and a column vector indicating 
    # to which fold each observation in the dataset belongs, respectively.
    print(OD.keys())
    dat_F =  OD['Pheno']#Phenotypic information (GID)
    type(dat_F)
    dat_F.head()
    list(dat_F.columns)
    Envs =  np.unique(dat_F['Env'])
    #Envs_df = pd.concat([Envs_df,pd.DataFrame({'Trait':Traits[t-1],'Env':Envs})])
    #Envs_df.to_csv('Traits-Envs.csv')
    # Response phenotypic value in OD['Pheno']
    y = dat_F[Traits[t-1]]
    Pos_NA = np.where(y.isna())
    y =  y.to_numpy(copy=True)
    y = y.astype(float)
    y = np.reshape(y,(len(y),1))

    #Matrix desing of environments
    XE =  pd.get_dummies(dat_F['Env'],columns=['Env'],drop_first=False)
    XE.head()
    Cols_XE =  XE.columns
    XE = XE.to_numpy()
    # Computing XL as ZL multiplied by the Cholesky decomposition of G
    G = OD['G']#Genomic-relationship matrix
    #GIDs = dat_F['GID'].unique()
    GIDs_G = G.columns.to_numpy()
    GIDs_G = ['GID'+i for i in GIDs_G ]
    G = pd.DataFrame(G)
    ZL = pd.get_dummies(dat_F['GID'],columns=['GID'])# Matriz desing of 
    L = np.linalg.cholesky(G)
    ZL_a = np.matmul(ZL,L)
    XL = ZL_a.to_numpy()
    ZL_a = None
    # Saving the fold value corresponding to each observation, indicating to which group each sample belongs in index order
    Fold = OD['Fold']
    Fold = Fold.to_numpy()
    Fold = np.reshape(Fold,(len(y)))
    # K = 10# 10KFCV
    # KF = KFold(n_splits = K, random_state=10, shuffle=True)
    # Pos_tst = [test_index for train_index, test_index in KF.split(y)]
    # Fold = np.zeros(len(y))
    # for k in range(K):
      # Fold[Pos_tst[k]] = k + 1
    # np.sum(Fold==1)
    
    Dir_Outs = 'Outs'#''.join(['BO-SmallExample',Hypers_Fixed['typeRN'],'-Epochs-',str(Epochs),'-Iters',str(Iters)])#BO-G'
    if not os.path.exists(Dir_Outs):
        os.mkdir(Dir_Outs)
    Model = 'MP2Is_ResNet-Env+GID'
    Monitor = 'loss'
    Model = ['-Model-',Model]
    Model = "".join(Model)
    #Envs =  np.unique(dat_F['Env'])
    #Pos_Envs = Match_f(Envs,Cols_XE)
    #===============================================================================
    # Start the 5-fold cross-validation performance evaluation
    #===============================================================================
    Tab =  pd.DataFrame()
    for p in range(5):
        Time = time.time()
        Ind_tr = np.ones(np.shape(y)[0])
        random.seed(p)
        Ind_tr = Fold!=(p+1)#np.where
        y_tr = y[Ind_tr==True,:]
        mu, sigma =  np.mean(y_tr), np.std(y_tr)
        y_tr =  (y_tr-mu)/sigma
        xE_tr = XE[Ind_tr==True,:]# Extracting the information of XE corresponding to the training data
        xL_tr = XL[Ind_tr==True,:]# Extracting the information of XL corresponding to the training data
        dat_Dict = dict()# A dictionary where the training data is saved and passed to the IKFCV_f function
        # to obtain and average the performance on the 5-fold validation set of the model,
        # which is the objective function to minimize in the Bayesian optimization algorithm invoked below.          
        x_ls = [xE_tr,xL_tr]# Training data in a list containing the inputs from two modalities (XE and XL) to feed the model
        pdims = [np.shape(x_ls[i])[1] for i in range(len(x_ls))]
        dat_Dict['y'], dat_Dict['x_ls'], dat_Dict['pdims'] = y_tr, x_ls, pdims
        x_ls = None
    # =============================================================================
        KI = 10  # Inner folds: number of subsets into which the training data is divided in an approximately balanced way,
        # where one subset is used for validation. The average of the MSE across all "Use" validation sets is computed.
        Use = 2# Inner folds used
        Hypers_Fixed['SaveM'] =  False
        def f_O(nHL1,Units1,nHL2,Units2,nHLB2,UnitsB2,l,DO,llr,lwd,Pat):
            Ind = np.all(2**(nHL1)<=Units1 and 2**(nHL2)<=Units2  and 2**(nHLB2)<=UnitsB2)
            if Ind:#IKFCV_f(Hypers_a,Hypers_Fixed,dat_Dict,KI=5):
                nHLs, Units = [nHL1,nHL2], [Units1,Units2]
                nHLs = [int(nHLs[i]) for i in range(len(nHLs))]
                Units = [int(Units[i]) for i in range(len(nHLs))]
                nHLUnitsB2_ls = [int(nHLB2),int(UnitsB2)]
                Hypers_a = {'nHLs':nHLs,'Units':Units,'nHLUnitsB2_ls':nHLUnitsB2_ls}
                Hypers_a = {**Hypers_a,**{'l':l,'DO':DO,'lr':np.exp(llr)}}
                Hypers_a = {**Hypers_a,**{'wd':np.exp(lwd),'Pat':int(Pat)}}
                MSEP_f = IKFCV_f(Hypers_a,Hypers_Fixed,dat_Dict,KI=KI,Use=Use)
                MSEP_Val = -sigma**2*np.mean(MSEP_f['MSEP'])
                return MSEP_Val
            else:
                return -1e5
        # Reduced bounds in the search domain space. Change to those used in the paper if needed.
        Bounds = {'nHL1':[1,2],'nHL2':[1,2], #'nHL3':(1,6),
                  'Units1':[4,8],'Units2':[32,128],#'Units3':(1,1024),
                  'nHLB2':[1,2], 'UnitsB2':[32,128], #B2
                  'l':(1e-8,1e-2),'DO':(1e-4,0.5),#'Batch_size':(64,512),
                  'Pat':(1,5),
                  'llr': (np.log(1e-8), np.log(1e-2)), 'lwd': (np.log(4e-5), np.log(4e-1))}
        BO = BayesianOptimization(f=f_O,pbounds=Bounds,random_state=1,verbose=1)
        BO.maximize(init_points=5, n_iter=Iters)
        print(BO.max)
        # "Optimal" hyper-parameters found after 150 iterations of the Bayesian optimization algorithm
        Pars_O = BO.max['params']
        nHLs, Units = [], []
        for i in range(1,len(dat_Dict['x_ls'])+1):
            nHLs.append(int(Pars_O[''.join(['nHL',str(i)])]))
            Units.append(int(Pars_O[''.join(['Units',str(i)])]))
        nHLUnitsB2_ls = [int(Pars_O['nHLB2']), int(Pars_O['UnitsB2'])]
        Hypers_aO = {'nHLs':nHLs,'Units':Units,'nHLUnitsB2_ls':nHLUnitsB2_ls}
        Hypers_aO['lr'], Hypers_aO['wd'] = np.exp(Pars_O['llr']), np.exp(Pars_O['lwd'])
        Hypers_aO['Pat'] =  int(Pars_O['Pat'])
        Hypers_aO = {**Hypers_aO,**{'l':Pars_O['l'],'DO':Pars_O['DO']}}
        Hypers_aO
        # Fitting the model with the "optimal" hyperparameters found using Bayesian optimization:
        # the model is trained "Use" times, with (IK-1) out of the IK validation subsets used for training the model
        # and the remaining subset used for validation. After training, the model is tested on the test set each time,
        # and the predictions from all "Use" training iterations are averaged to obtain a final prediction for the test set.
        IFold = SIFold(y_tr,K=KI,nq=10,random_state=2)
        len(IFold)
        xE_tst = XE[Ind_tr==False,:]
        xL_tst = XL[Ind_tr==False,:]
        xls_tst = [xE_tst,xL_tst]
        y_tst = y[Ind_tr==False,:]
        y_tst =  y_tst[:,0]
        yp_mean = np.repeat(0,len(y_tst))
        Epochs_df = pd.DataFrame()
        Hypers_Fixed['SaveM'] =  True
        for k in range(1,Use+1):
            Pos_ival = list(np.where(IFold==k)[0])#Validation data$# type(Pos_ival)
            Hypers_Fixed['Pos_ival'] = Pos_ival
            Ms = M_f(Hypers_aO,Hypers_Fixed,dat_Dict,verbose=0)
            MO = Ms['Model']
            yp_tst = MO.predict(xls_tst,verbose=0)#,batch_size=int(Batch_size))
            yp_mean = yp_mean + yp_tst[:,0]
            Epochs_df =  pd.concat([Epochs_df, pd.DataFrame([{'IPT':k,'Epochs_O':Ms['Epochs']}])])
            print('k=',k)
        yp_mean = yp_mean/Use
        yp_tst = yp_mean
        Epochs_O, MSEP_Val = None, -BO.max['target']# Ms['Epochs'] sigma**2*Ms['MSEP_Val']        
        yp_tst = mu+sigma*yp_tst# Final prediction of the test set
        #Computing the evaluation metrics
        MSEP, Cor = np.mean((y_tst-yp_tst)**2), pearsonr(y_tst, yp_tst)[0]
        NRMSEP = np.sqrt(MSEP)/np.mean(y_tst)
        Tb_t = {'PT':p+1,'MSEP':MSEP,'MSEP_Val':MSEP_Val,'Cor':Cor,'NRMSEP':NRMSEP,'Epochs':Epochs_O,**Pars_O}
        Tb_t = pd.DataFrame([Tb_t])
        Tab =  pd.concat([Tab,Tb_t])
        Tab
        df_Preds = pd.DataFrame({'PT':p+1,'y':y_tst,'yp':yp_tst,'MSEP':MSEP,'Cor':Cor,'NRMSEP':NRMSEP})
        Dir_Preds = [Dir_Outs,'/Preds-','Trait-',str(Traits[t-1]),Model,'-PT-',str(p+1),'.csv']
        Dir_Preds = "".join(Dir_Preds)
        Time = time.time() - Time
        df_Preds.to_csv(Dir_Preds)
        print('p=',p,'Cor=',Cor,'MSEP=',MSEP,'MSEP_Val=',MSEP_Val)

    print(Tab)
    Dir_Tab = [Dir_Outs,'/Tab-','Trait-',str(Traits[t-1]),Model,'.csv']
    Dir_Tab = "".join(Dir_Tab)
    Tab.to_csv(Dir_Tab)



