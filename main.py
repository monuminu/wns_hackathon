'''
------------------------------------------Importing the Libraries--------------------------------------------
'''
print('Importing the Libraries Started')
import numpy as np
import pandas as pd
import swifter
import os
from sklearn.model_selection import train_test_split
import math
from imblearn.over_sampling import SMOTE 
import pickle
from functions import *

def main():

    '''
    -----------------------------------------Setting the Home Directory--------------------------------------------
    '''
    print('Setting the Home Directory Started')
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    '''
    ------------------------------------------Importing the Dataseet--------------------------------------------
    '''
    print('Importing the Dataseet Started')
    train = pd.read_csv('train.csv')
    test  = pd.read_csv('test.csv')
    test_bkp = test
    data  = pd.concat([train,test],axis = 0, sort= False).reset_index()
    pd.options.display.float_format = '{:.4f}'.format 

    '''
    ----------------------------------------Checking for data details--------------------------------------------
    '''
    print('Checking for data details Started')
    print("Train shape is ", train.shape)
    print("TEst shape is ", test.shape)
    print('Balance of Class is ', sum(train.is_promoted)/train.shape[0] )
    #print(data.isnull().sum())  #To Check missing values per column
    #print(data.describe())      #To Check the Descriptive Statistics of the Data
    #print(data.dtypes)          #To Check the Data type of each column

    '''
    ----------------------------------------Missing values Treatment--------------------------------------------
    '''
    print('Missing values Treatment Started')
    data['previous_year_rating'] = data['previous_year_rating'].fillna(3) #default rating is 3
    data['education'] = data['education'].fillna("Bachelor's")

    '''
    ----------------------------------------Feature Engineering--------------------------------------------
    '''
    print('Feature Engineering Started')
    data = drop_cols_with_unique_values(data)
    col_to_scale = ['no_of_trainings','age','previous_year_rating','length_of_service',
                    'avg_training_score']

    for col in col_to_scale:
        data[col] = scaling(data,col)
    
    data_bkp = data.copy()

    one_hot_check = True
    if one_hot_check:
        data = one_hot(data , 'department')   
        data = one_hot(data , 'region')   
        data = one_hot(data , 'education')   
        data = one_hot(data , 'gender')   
        data = one_hot(data , 'recruitment_channel')   
    
    data.drop(['employee_id','index'],axis = 1 , inplace= True)  #Dropping in Age In Days Since we are going with Age in Years

    '''
    -------------------------------------Build Machine Learning Model--------------------------------------------
    '''
    print('Splitting Train and Test')
    #print(data.head(5))
    #Spliting the train , test after the preprocessing the result [test date will go for Final evaluation in Hackathon]
    test = data[data.is_promoted.isnull()].reset_index()
    train = data[~data.is_promoted.isnull()].reset_index()
    X_train, X_test, Y_train, Y_test = train_test_split(train.loc[:,train.columns!='is_promoted']
                                                        ,train.is_promoted, test_size=0.2, random_state=35)

    
    X_test_xgb = X_test
    X_test_xgb.drop(['index'],axis = 1 , inplace= True)
    Y_test_xgb = Y_test
    test_xgb = test 
    test.drop(['index'],axis = 1 , inplace= True)
    data = data_bkp.copy()
    data.drop(['employee_id','index'],axis = 1 , inplace= True)  #Dropping in Age In Days Since we are going with Age in Years

    '''
    -------------------------------------Build Machine Learning Model--------------------------------------------
    '''
    print('Splitting Train and Test')
    #print(data.head(5))
    #Spliting the train , test after the preprocessing the result [test date will go for Final evaluation in Hackathon]
    test = data[data.is_promoted.isnull()].reset_index(drop = True)
    train = data[~data.is_promoted.isnull()].reset_index(drop = True)
    X_train, X_test, Y_train, Y_test = train_test_split(train.loc[:,train.columns!='is_promoted']
                                                        ,train.is_promoted, test_size=0.2, random_state=35)

    X_test_cat = X_test
    Y_test_cat = Y_test
    test_cat = test
    print(sum(Y_test_cat != Y_test_xgb))    


    seed = 27
    model_file = "xgboost_wns_.pkl"
    if os.path.exists(model_file):
        print("Model already exists")
        model1 = pickle.load(open(model_file, 'rb'))
    else:
        print('Model Training Started')
        model1 = training_xgb_model(X_train,Y_train,seed)
        pickle.dump(model1, open(model_file, 'wb'))
    
    cat_features = np.where(X_train[X_train.columns].dtypes == np.object)[0]
    print(cat_features)
    model_file = "catboost_wns_.pkl"
    if os.path.exists(model_file):
        print("Model already exists")
        model2 = pickle.load(open(model_file, 'rb'))
    else:
        print('Model Training Started')
        model2 = train_cat_boost(X_train.values,Y_train.values,cat_features)
        pickle.dump(model2, open(model_file, 'wb'))

    cutoff1 = evaluate(model1, X_test_xgb, Y_test, save_metrics=False, eval_param="f1_score", model_type='xgb')
    cutoff2 = evaluate(model2, X_test_cat, Y_test, save_metrics=False, eval_param="f1_score", model_type='xgb')
    submission_ensemble([model1, model2], X_test_cat,X_test_xgb, Y_test , test_xgb , test_cat, test_bkp)
    #cutoff = evaluate(model1, X_test, Y_test, save_metrics=False, eval_param="f1_score", model_type='xgb')

    submission_check = False
    if submission_check == True:
        print('Predicting Test Data and Submission')
        #Submit the prediction of Original test data in the hackathon website 
        test = test.drop(['is_promoted','index'],axis = 1,inplace = False)
        pred_test = model1.predict_proba(test)[:,1]
        submission = pd.DataFrame({'employee_id': test_bkp.employee_id, 'is_promoted': pred_test})
        submission['is_promoted'] = [1 if score > cutoff else 0 for score in submission['is_promoted']]
        submission.to_csv('submission.csv',index=False)
        print(submission.head(5))   

if __name__ == '__main__':
    main()  
