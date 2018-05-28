'''
--------------------------------------------------------------------
 To Calculate feature contributions at observation level using XGBoost
--------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
import time
import xgboost as xg
from sklearn import metrics


#------------------------------------------------------
# Wrap configurations and helper functions in a class
#------------------------------------------------------
class Contrib:
def __init__(self):
self.dataDir = 'data/'
self.header_csv = self.dataDir+'train_sample_header.csv'
self.train_csv = self.dataDir+'train_sample.csv'
self.test_csv = self.dataDir+'oot_sample.csv'
self.test_header_csv = self.dataDir+'oot_sample_header.csv'
self.labels = ['ind']
self.keys = ['id','as_of_dt']
self.num_round = 100 # Number of trees (iterations) for XGBoost
self.n_top = 100 # Number of observations selected for interpretation
self.modelPath = "xgb.pk"
self.features = []

def readData(self,dataPath,headerPath):
# Read data from csv file, and return a dataframe
df = pd.read_csv(dataPath,header=None)
df.columns = open(headerPath,'r').readline().rstrip().split(',')
df['id'] = df['id'].astype(str)
return df

def readTrainData(self):
df = self.readData(self.train_csv,self.header_csv)
return df

def readTestData(self):
df = self.readData(self.test_csv,self.test_header_csv)
return df

def dfToDmatrix(self,df):
# Convert a dataframe to DMatrix
labels_keys = self.labels + self.keys
self.features = [c for c in df.columns if c not in labels_keys]
X = df[self.features].values
Y = df[self.labels[0]].values if self.labels[0] in df.columns.values else []
dmatrix = xg.DMatrix(X, Y)
return dmatrix, self.features

def getModel(self, trainFlag, dtrain):
if trainFlag:
# Train model
print '\n2. Training model...'
param = {'max_depth':5, 'eta':0.05, 'silent':0, 'objective':'binary:logistic','lambda':1.0, 'alpha':0, 'scale_pos_weight':1, 'subsample':1, 'max_delta_step':1}
bst = xg.train(param, dtrain, self.num_round)
preds_train = bst.predict(dtrain,pred_contribs=False)
print 'GINI (train)', 2*metrics.roc_auc_score(trainDf['ind'].values, preds_train) - 1
bst.save_model('xgb.model')
else:
# Load model
print '2. Loading previously trained model'
bst = xg.Booster(model_file='xgb.model')
preds_train = bst.predict(dtrain,pred_contribs=False)
print 'GINI (train)', 2*metrics.roc_auc_score(trainDf['ind'].values, preds_train) - 1
return bst

def selectTop(self,df):
# Sort dataframe by score from high to low, and select top scorers
df = df.sort_values(by=['score'],ascending=False)
df = df.reset_index(drop=True)
df_new = df.iloc[:self.n_top,:-1]
return df_new, df

def explain(self,bst,df_oot_contrib,doot_contrib,df_oot,iids):
# Obtain feature contributions in XGBoost
preds_oot = bst.predict(doot_contrib,pred_contribs=True) # feature contributions, ndarry of size [n_sample, (n_features + 1)]

# Join feature contributions with actual values
df_preds_oot = pd.DataFrame(preds_oot[:,:-1])
df_preds_oot.columns = self.features #doot_columns
df_preds_oot['id'] = df_oot['id']
df_preds_oot['ind'] = df_oot['ind']
dict_preds_oot = df_preds_oot.to_dict(orient="index") # customer, feature contributions

df_oot_contrib.set_index("id", drop=True, inplace=True)
dict_df_oot = df_oot_contrib.to_dict(orient="index") # customer, feature values

# Output feature contributions
for iid in iids:
dict_sample = sorted( ((v,k) for k,v in dict_preds_oot[iid].iteritems() if (k != 'id' and k != 'score')), reverse=True)
cust = dict_preds_oot[iid]['id']
print 'id: ', cust
print 'score: ',df_oot['score'].values[1]
print '\nContribution,Feature,Value:'
for (v,k) in dict_sample:
if k != 'ind':
print ','.join((str(v),k,str(dict_df_oot[cust][k])))


#---------------
# Main program
#---------------
if __name__ == '__main__':
print 'Starting program...'
startTime = time.time()
#---------------------------
# 0. Initiate configuration
#---------------------------
trainFlag = False # To train the model or not. If True, train a new model; if False, load a previously-trained model
model = Contrib()

#------------------------
# 1. Load training data
#------------------------
print '1. Loading training data...'
trainDf = model.readTrainData()
dtrain, dtrain_columns = model.dfToDmatrix(trainDf)

#------------------------
# 2. Train or load model
#------------------------
print '\n2. Training/Loading model...'
startTrainTime = time.time()
bst = model.getModel(trainFlag,dtrain)
endTrainTime = time.time()

#------------------------
# 3. Load testing data
#------------------------
print '3. Loading testing data...'
df_oot = model.readTestData()
doot, doot_columns = model.dfToDmatrix(df_oot)
preds_test = bst.predict(doot,pred_contribs=False) # probability scores
df_oot['score'] = pd.DataFrame(preds_test)
print 'GINI (out-of-time test)', 2*metrics.roc_auc_score(df_oot['ind'].values, preds_test) - 1

#---------------------------------------------------
# 4. Calculate feature contributions for test data
#---------------------------------------------------
print '\n4. Interpreting predictions...'
# Select top scores for explanation
df_oot_contrib, df_oot = model.selectTop(df_oot)
doot_contrib, doot_contrib_columns = model.dfToDmatrix(df_oot_contrib)
iids = [1]
#iids = range(1)
startExpTime = time.time() 
model.explain(bst, df_oot_contrib, doot_contrib, df_oot, iids)

endExpTime = time.time()
endTime = time.time()

#----------------------
# 5. Output run time
#---------------------- 
print '\n---------------------------------------'
print 'Number of instance for explanation: ', len(iids)
print 'Explanation run time: ',endExpTime - startExpTime, 's'
print 'Total run time: ',endTime - startTime, 's'
print '---------------------------------------'