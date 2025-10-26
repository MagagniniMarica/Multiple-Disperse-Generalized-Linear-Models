# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
path = '../Datasets/'
Boston Dataset loading and preprocessing
"""

import pandas as pd
from sklearn import preprocessing


# This function returns a dict, for each feature is associated 
# the type (categorical (also binary), integer or numerical (continuaos))
def feature_type(dataset):
        features=list(dataset.columns)
        feat_type =[]
        for x in dataset.dtypes:
            if x.name == 'category':
                feat_type.append('Categorical')
            elif x.name == 'int64':
                feat_type.append('Integer')
            else:
                feat_type.append('Numerical')                
        features_type = dict(zip(features, feat_type))
        return features_type



    
# This function returs:
    # boston: the features DataDrame
    # target: the target Series
    # fetaures: the name of features Index
    # fetaures_type: the type of features Dict
def data(path, task):
    
    full_path =path+'HousingData.csv'
    data = pd.read_csv(full_path)
    
    
    
    #imputation
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')  # 'mean', 'median', 'most_frequent'
    boston = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    target = boston['MEDV']
    boston.drop('MEDV', axis=1, inplace=True)
    
    
    #Categorical variables
    boston['CHAS']=boston['CHAS'].astype('category') 

    
    #Target

    target = pd.Series(target)
    if task == 'classification' :
        target= target.apply(lambda x: 0 if x<=22 else 1) # for classification task
    
 
    features=boston.columns              # names of columns vector (Index)
    features_type=feature_type(boston)   # type of each feature vector (Dict)


    #Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    boston_scaled =pd.DataFrame( min_max_scaler.fit_transform(boston),  columns= features)

    
    
    
    return boston_scaled, target, features, features_type
