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
from scipy.sparse import csr_matrix
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
    #data['exp_category'] = data['length_of_service'].swifter.apply(get_experience_category)
    col_to_scale = ['no_of_trainings','age','previous_year_rating','length_of_service',
                    'avg_training_score']

    for col in col_to_scale:
        data[col] = scaling(data,col)

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
    test = data[data.is_promoted.isnull()].reset_index(drop = True)
    train = data[~data.is_promoted.isnull()].reset_index(drop = True)
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(train.loc[:,train.columns!='is_promoted']
                                                        ,train.is_promoted, test_size=0.2, random_state=35)
    
    '''
    X_train = train.loc[:,train.columns!='is_promoted']
    Y_train = train.is_promoted
    cat_features = np.where(X_train[X_train.columns].dtypes == np.object)[0]
    seed = 27
    test = test.drop(['is_promoted'],axis = 1,inplace = False)
    pred_train1, pred_test1, cutoff1 = lgbm_fit_predict(X_train.values, Y_train, test)
    pred_train2, pred_test2, cutoff2 = catboost_fit_predict(X_train.values, Y_train, cat_features, test)
    pred_train3, pred_test3, cutoff3 = fit_predict_xgboost(X_train.values, Y_train, test, X_train.columns)

    pred_train = (pred_train1 + pred_train2 + pred_train3)/3
    pred_test = (pred_test1 + pred_test2 + pred_test3)/3
    cutoff = (cutoff1 + cutoff2 + cutoff3)/3
    print(cutoff)
    submission_check = True
    if submission_check == True:
        print('Predicting Test Data and Submission')
        submission = pd.DataFrame({'employee_id': test_bkp.employee_id, 'is_promoted': pred_test})
        submission.to_csv('submission_prev.csv',index=False)
        submission['is_promoted'] = [1 if score > cutoff else 0 for score in submission['is_promoted']]
        submission.to_csv('submission1.csv',index=False)
        print(submission.head(5))   

if __name__ == '__main__':
    main()  