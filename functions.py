'''
------------------------------------------Importing the Libraries--------------------------------------------
'''
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import lightgbm as lgb
import numpy as np
import math
from sklearn.metrics import roc_curve, auc, accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
import pandas as pd
from catboost import CatBoostClassifier
from tqdm import tqdm
from bayesian_optimization import BayesianOptimization
'''
-----------------------------------------Function Definations--------------------------------------------
'''
RANDOM_STATE = 22

def get_experience_category(exp):
    if exp < 5:
        return "Junior"
    elif exp >=5 and exp < 12:
        return "Mid"
    else:
        return "High"

def drop_cols_with_unique_values(df):
    #check for columns having only a single unique values 
    col_with_single_unique_value = []
    for col in df.columns:
        if len(df[col].unique()) == 1 :
            col_with_single_unique_value.append(col)

    df.drop(col_with_single_unique_value,axis = 1 , inplace= True)
    return df

def scaling(data,colname):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data[[colname]])
    

def one_hot(data,colname):
    one_hot_encoded_data = pd.get_dummies(data[[colname]],prefix = colname,drop_first= True)
    data.drop([colname],axis = 1,inplace = True)
    data = pd.concat([data,one_hot_encoded_data],axis = 1)
    return data

def auc_score(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_result = auc(fpr, tpr)
    print(auc_result)  
    return auc_result

def predict_model_bank(model_bank , test_features,test_labels):
    return [model.predict_proba(test_features)[:,1] for model in model_bank]
    
def evaluate(model, test_features, test_labels, save_metrics = False, eval_param = 'auc' , model_type = 'non-xgb'):
    if model_type == 'xgb':
        pred_Y = model.predict_proba(test_features)[:,1]
    else:
        pred_Y = model.predict(test_features)
    if eval_param == 'auc':
        fpr, tpr, thresholds = roc_curve(test_labels, pred_Y)
        report = pd.DataFrame({"fpr": fpr , "tpr" : tpr , "thresholds" : thresholds})
        auc_result = auc(fpr, tpr)
        print(auc_result)
        if save_metrics == True: report.to_csv("metrics_report1.csv")

    elif eval_param == 'f1_score':
        precision, recall, threshold = precision_recall_curve(test_labels, pred_Y)
        report = pd.DataFrame({"precision": precision[:-1] , "recall" : recall[:-1] , "threshold" : threshold})
        report['f1_score'] = 2 * report['precision'] * report['recall'] / (report['precision'] + report['recall'])
        print('Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['f1_score']))
        print('Threshold with Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['threshold']))
        if save_metrics == True: report.to_csv("metrics_report1.csv")
        return report.ix[report['f1_score'].idxmax()]['threshold']
        
    
def training_lgbm_model(train_X,train_Y,seed = 37):
    params = {
    'learning_rate': 0.03, 
    'boosting': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc', 
    'is_training_metric': False, 
    'scale_pos_weight': 0.5,
    'max_depth': 7  ,  
    'min_child_samples': 50,  
    'max_bin': 50,  
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,
    'seed': seed
    }
    X_train, X_valid, y_train, y_valid = train_test_split(train_X,
                                                    train_Y, 
                                                    shuffle=True,
                                                    random_state=seed,
                                                    train_size=0.8,
                                                    stratify=train_Y)

    fit_model = lgb.train( params, train_set= lgb.Dataset(X_train, label=y_train), num_boost_round= 1000,
            valid_sets = lgb.Dataset(X_valid, label=y_valid), verbose_eval = 20, early_stopping_rounds = 100)
    return fit_model



def training_xgb_model2(train_X,train_Y,seed=27):
    xgb_model = xgb.XGBClassifier()
    parameters = {'nthread':[6], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.05,0.01], #so called `eta` value
                  'max_depth': [5,10],
                  'min_child_weight': [5,10],
                  'silent': [1],
                  #'subsample': [0.8,0.2],
                  #'colsample_bytree': [0.7,0.1],
                  'n_estimators': [50,500], #number of trees, change it to 1000 for better results
                  'missing':[-999],
                  'seed': [1337]}
    
    
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                       cv=StratifiedKFold(n_splits=5, shuffle=True), 
                       scoring='roc_auc',
                       verbose=2, refit=True)
    
    clf.fit(train_X, train_Y)
    
    #trust your CV!
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))   
    return clf    
        
def training_xgb_model(train_X,train_Y,seed=27):
    
    xgb2 = XGBClassifier(
     learning_rate =0.05,
     n_estimators=1500,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.6,
     colsample_bytree=0.6,
     objective= 'binary:logistic',
     nthread=8,
     n_jobs= 8,
     reg_lambda = 0.01,
     #scale_pos_weight=2,
     seed=seed)
    model = modelfit(xgb2, train_X, train_Y)
    return model

def modelfit(alg, train_X, train_Y,useTrainCV=True, cv_folds=5, early_stopping_rounds=30):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_X.values, label=train_Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=True, shuffle= True, stratified=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(train_X, train_Y,eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(train_X)
    dtrain_predprob = alg.predict_proba(train_X)[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_Y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_Y, dtrain_predprob))
                    
    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')    
    return alg

def train_cat_boost(X_train, y_train,cat_features):
    
    params = {'depth':[2, 3, 4],
              'loss_function': ['Logloss', 'CrossEntropy'],
              'l2_leaf_reg':[0.01,0.05]
    }
    
    #param = catboost_GridSearchCV(X_train, y_train, params, cat_features)
    param = {
        "loss_function" : 'Logloss',
        "depth" : 4,
        "l2_leaf_reg" :0.05
    }
    clf = CatBoostClassifier(iterations=1500,
                            loss_function = param['loss_function'],
                            depth=param['depth'],
                            l2_leaf_reg = param['l2_leaf_reg'],
                            eval_metric = 'F1',
                            leaf_estimation_iterations = 10,
                            use_best_model=True,
                            thread_count=5  
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                        y_train, 
                                                        shuffle=True,
                                                        random_state=RANDOM_STATE,
                                                        train_size=0.8,
                                                        stratify=y_train
    )
    clf.fit(X_train, 
            y_train,
            cat_features=cat_features,
            logging_level='Silent',
            eval_set=(X_valid, y_valid)
    )
    return clf
def cross_val(X, y, param, cat_features, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    #print('missing value in y_train : {}'.format(sum(y.isna())))
    acc = []
    predict = None
    
    for tr_ind, val_ind in skf.split(X, y):
        X_train = X[tr_ind]
        y_train = y[tr_ind]
        
        X_valid = X[val_ind]
        y_valid = y[val_ind]
        #print('missing value in y_valid : {}'.format(sum(y_valid.isna())))
        clf = CatBoostClassifier(iterations=500,
                                loss_function = param['loss_function'],
                                depth=param['depth'],
                                l2_leaf_reg = param['l2_leaf_reg'],
                                eval_metric = 'Logloss',
                                leaf_estimation_iterations = 10,
                                use_best_model=True,
                                logging_level='Silent',
                                thread_count=5,
                                n_estimators=500
        )
        
        clf.fit(X_train, 
                y_train,
                cat_features=cat_features,
                eval_set=(X_valid, y_valid)
        )
        
        y_pred = clf.predict(X_valid)
        accuracy = auc_score(y_valid, y_pred)
        acc.append(accuracy)
    return sum(acc)/n_splits
    
def catboost_GridSearchCV(X, y, params, cat_features, n_splits=5):
    ps = {'acc':0,
          'param': []
    }
    
    predict=None
    
    for prms in tqdm(list(ParameterGrid(params)), ascii=True, desc='Params Tuning:'):
                          
        acc = cross_val(X, y, prms, cat_features, n_splits=5)

        if acc>ps['acc']:
            ps['acc'] = acc
            ps['param'] = prms
    print('Acc: '+str(ps['acc']))
    print('Params: '+str(ps['param']))
    
    return ps['param']    


def submission_ensemble(model_bank, X_test_cat,X_test_xgb, Y_test , test_xgb , test_cat, test_bkp):
    save_metrics = True
    predict1 = model_bank[0].predict_proba(X_test_xgb)[:,1]
    predict2 = model_bank[1].predict_proba(X_test_cat)[:,1]
    pred_Y = (predict1 + predict2)/2
    precision, recall, threshold = precision_recall_curve(Y_test, pred_Y)
    report = pd.DataFrame({"precision": precision[:-1] , "recall" : recall[:-1] , "threshold" : threshold})
    report['f1_score'] = 2 * report['precision'] * report['recall'] / (report['precision'] + report['recall'])
    print('Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['f1_score']))
    print('Threshold with Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['threshold']))
    if save_metrics == True: report.to_csv("metrics_report2.csv")
    
    cutoff =  report.ix[report['f1_score'].idxmax()]['threshold']
    test_xgb = test_xgb.drop(['is_promoted'],axis = 1,inplace = False)
    test_cat = test_cat.drop(['is_promoted'],axis = 1,inplace = False)
    predict1 = model_bank[0].predict_proba(test_xgb)[:,1]
    predict2 = model_bank[1].predict_proba(test_cat)[:,1]
    pred_test = (predict1 + predict2)/2
    submission = pd.DataFrame({'employee_id': test_bkp.employee_id, 'is_promoted': pred_test})
    submission['is_promoted'] = [1 if score > cutoff else 0 for score in submission['is_promoted']]
    submission.to_csv('submission4.csv',index=False)
    print(submission.head(5))



iter_no = 5
gp_params = {'alpha': 1e-5}
cv_splits = 8



def training_xgb_model3(X_train, y_train,seed = 37):
    def treesCV(eta, gamma,max_depth,min_child_weight,subsample,colsample_bytree,n_estimators):
        #function for cross validation gradient boosted trees
        return cross_val_score(xgb.XGBRegressor(objective='binary:logistic',
                                                    tree_method = 'hist',
                                                    learning_rate=max(eta,0),
                                                    gamma=max(gamma,0),
                                                    max_depth=int(max_depth),
                                                    min_child_weight=int(min_child_weight),
                                                    silent=True,
                                                    subsample=max(min(subsample,1),0.0001),
                                                    colsample_bytree=max(min(colsample_bytree,1),0.0001),
                                                    n_estimators=int(n_estimators),
                                                    seed=42,nthread=-1), X=X_train, y=y_train,  cv=cv_splits, n_jobs=-1).mean()

        #Bayesian Hyper parameter optimization of gradient boosted trees
    treesBO = BayesianOptimization(treesCV,{'eta':(0.001,0.4),
                                            'gamma':(8,12),
                                            'max_depth':(400,700),
                                            'min_child_weight':(0.1,1),
                                            'subsample':(0.3,0.6),
                                            'colsample_bytree':(0.6,1),
                                            'n_estimators':(600,800)})
    treesBO.maximize(n_iter=iter_no, **gp_params)
    tree_best = treesBO.res['max']

    #train tree with best paras
    trees_model = xgb.XGBRegressor(objective='binary:logistic',
    								tree_method = 'hist',
                                    seed=42,
                                    learning_rate=max(tree_best['max_params']['eta'],0),
                                    gamma=max(tree_best['max_params']['gamma'],0),
                                    max_depth=int(tree_best['max_params']['max_depth']),
                                    min_child_weight=int(tree_best['max_params']['min_child_weight']),
                                    silent=True,
                                    subsample=max(min(tree_best['max_params']['subsample'],1),0.0001),
                                    colsample_bytree=max(min(tree_best['max_params']['colsample_bytree'],1),0.0001),
                                    n_estimators=int(tree_best['max_params']['n_estimators']),nthread=-1)
    trees_model.fit(X_train, y_train)
    return trees_model

def lgbm_fit_predict(data, y, test):
    dtrain = lgb.Dataset(data=data, label=y, free_raw_data=False)
    dtrain.construct()

    oof_preds = np.zeros(data.shape[0])
    sub_preds = np.zeros(test.shape[0])

    lgb_params = {
        "objective" : "binary",
        "metric" : "binary_logloss",
#         "metric" : "auc",

#         'max_depth': 3,
        "num_leaves": 10,
        "min_data_in_leaf": 10,
        "learning_rate": 0.01,

        "feature_fraction": 0.3,
        "feature_fraction_seed": 10,

        "bagging_fraction": 0.8,
        "bagging_freq" : 10,
        "bagging_seed" : 42, #2018

        "verbosity" : 1,
#         'lambda_l1' : 10,
#         'lambda_l2' : 10,
        'max_bin' : 50
    }

    folds = StratifiedKFold(n_splits=7, shuffle=True, random_state=2)

    counter = 1
    for trn_idx, val_idx in folds.split(data,y):
        print('----------------------------')
        print('Fold: %d' % counter)

        trn_d = dtrain.subset(trn_idx)
        val_d = dtrain.subset(val_idx)

        clf = lgb.train(
            params=lgb_params,
            train_set=trn_d,
            valid_sets=[trn_d, val_d],
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=50
        )

        oof_preds[val_idx] = clf.predict(dtrain.data[val_idx, :])
        sub_preds += clf.predict(test) / folds.n_splits
        
        counter += 1

    precision, recall, threshold = precision_recall_curve(y, oof_preds)
    report = pd.DataFrame({"precision": precision[:-1] , "recall" : recall[:-1] , "threshold" : threshold})
    report['f1_score'] = 2 * report['precision'] * report['recall'] / (report['precision'] + report['recall'])
    print('Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['f1_score']))
    print('Threshold with Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['threshold']))
    cutoff = report.ix[report['f1_score'].idxmax()]['threshold']
    return oof_preds, sub_preds, cutoff


def catboost_fit_predict(data, y, cat_features, test):
    oof_preds = np.zeros(data.shape[0])
    sub_preds = np.zeros(test.shape[0])

    param = {
        "loss_function" : 'Logloss',
        "learning_rate" : 0.08,
        "border_count" : 40,
        "depth" : 6,
        "l2_leaf_reg" :5,
        "leaf_estimation_iterations" : 20
    }
    clf = CatBoostClassifier(iterations=1500,
                            loss_function = param['loss_function'],
                            learning_rate= param['learning_rate'],
                            depth=param['depth'],
                            border_count= param['border_count'],
                            l2_leaf_reg = param['l2_leaf_reg'],
                            eval_metric = 'F1',
                            leaf_estimation_iterations = param['leaf_estimation_iterations'],
                            use_best_model=True,
                            thread_count=5  
    )
    
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    counter = 1
    for trn_idx, val_idx in folds.split(data,y):
        print('----------------------------')
        print('Fold: %d' % counter)

        trn_d = data[trn_idx, :]
        val_d = data[val_idx, :]
        trn_y = y[trn_idx]
        val_y = y[val_idx]

        clf.fit(trn_d, 
            trn_y,
            cat_features=cat_features,
            #logging_level='Silent',
            eval_set=(val_d, val_y),
            early_stopping_rounds=100
        )    

        oof_preds[val_idx] = clf.predict_proba(data[val_idx, :])[:,1]
        sub_preds += clf.predict_proba(test)[:,1] / folds.n_splits
        
        counter += 1

    precision, recall, threshold = precision_recall_curve(y, oof_preds)
    report = pd.DataFrame({"precision": precision[:-1] , "recall" : recall[:-1] , "threshold" : threshold})
    report['f1_score'] = 2 * report['precision'] * report['recall'] / (report['precision'] + report['recall'])
    print('Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['f1_score']))
    print('Threshold with Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['threshold']))
    cutoff = report.ix[report['f1_score'].idxmax()]['threshold']
    return oof_preds, sub_preds, cutoff


def fit_predict_xgboost(data, y, test,feature_names):

    oof_preds = np.zeros(data.shape[0])
    dtest = xgb.DMatrix(test, feature_names=feature_names)
    sub_preds = np.zeros(test.shape[0])
    
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'max_depth': 3,
        'eta': 0.01,
        'eval_metric': 'logloss',
        'subsample': 0.8,
        'colsample_bytree': 0.3,
    #     'min_child_weight': 5,
    #     'lambda': 10,
    #     'alpha': 10,
        'nthread': 4
    #     'silent': 1,
    }

    folds = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)

    counter = 1
    for trn_idx, val_idx in folds.split(data,y):
        print('----------------------------')
        print('Fold: %d' % counter)

        dtrain_sample = xgb.DMatrix(data[trn_idx], label=y[trn_idx], feature_names=feature_names)
        dvalid_sample = xgb.DMatrix(data[val_idx], label=y[val_idx], feature_names=feature_names)
        assert(dtrain_sample.num_col() == dvalid_sample.num_col())
        assert(dtrain_sample.num_col() == len(feature_names))
        assert(dtrain_sample.num_col() == data.shape[1])
        assert(dvalid_sample.num_col() == data.shape[1])

        watchlist  = [(dtrain_sample, 'train'), (dvalid_sample, 'valid')]
        num_round = 4000
        xgb_model = xgb.train(params, dtrain_sample, num_round, watchlist,
                              verbose_eval=50,
                              early_stopping_rounds=100)

        oof_preds[val_idx] = xgb_model.predict(dvalid_sample)
        sub_preds += xgb_model.predict(dtest) / folds.n_splits

        counter += 1

    precision, recall, threshold = precision_recall_curve(y, oof_preds)
    report = pd.DataFrame({"precision": precision[:-1] , "recall" : recall[:-1] , "threshold" : threshold})
    report['f1_score'] = 2 * report['precision'] * report['recall'] / (report['precision'] + report['recall'])
    print('Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['f1_score']))
    print('Threshold with Maximum f1 score is {}'.format(report.ix[report['f1_score'].idxmax()]['threshold']))
    cutoff = report.ix[report['f1_score'].idxmax()]['threshold']
    return oof_preds, sub_preds, cutoff