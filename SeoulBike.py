# -*- coding: utf-8 -*-
"""

@author: Marica Magagnini

Seoul Bike rent per hour
path = '../Datasets/'
Poission Regression dataset
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# This function returns a dict, for each feature is associated 
# the type (categorical (also binary), integer or numerical (continuaos))
def feature_type(dataset):
        features=list(dataset.columns)
        feat_type =[]
        for x in dataset.dtypes:
            if x.name == 'category' or x.name == 'object':
                feat_type.append('Categorical')
            else:
                feat_type.append('Numerical')                
        features_type = dict(zip(features, feat_type))
        return features_type
    
    
class Encoder_(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            unique_vals = X[col].nunique()
            if unique_vals == 2:
                # LabelEncoder x binary
                le = LabelEncoder()
                le.fit(X[col])
                self.encoders[col] = ('label', le)
            else:
                # OneHotEncoder x multicategorical
                ohe = OneHotEncoder(sparse_output=False) 
                ohe.fit(X[[col]])
                self.encoders[col] = ('onehot', ohe)
        return self

    def transform(self, X):
        transformed_parts = []
        for col in X.columns:
            method, encoder = self.encoders[col]
            if method == 'label':
                transformed = encoder.transform(X[col])
                transformed_parts.append(pd.Series(transformed, name=col))
            else:  # onehot
                transformed = encoder.transform(X[[col]])
                cols = encoder.get_feature_names_out([col])
                transformed_parts.append(pd.DataFrame(transformed, columns=cols, index=X.index))
        return pd.concat(transformed_parts, axis=1)


def data(path, task):
    
    
    full_path =path+'SeoulBikeData.csv'
    raw_data = pd.read_csv(full_path, encoding='latin1')
    
    target = raw_data['Rented Bike Count']
    
    bs1 = raw_data.drop(['Date','Rented Bike Count'], axis = 1)
    bs1_features = bs1.columns
    bs1_features_type=feature_type(bs1)   # type of each feature vector (Dict)
    bs1_num_f = [fn for fn in bs1_features if bs1_features_type[fn] == 'Numerical']
    bs1_cat_f = [fn for fn in bs1_features if bs1_features_type[fn] == 'Categorical']
    
    
    # Categorical Variables
    
    encoder = Encoder_()
    bs2 = encoder.fit_transform(bs1[bs1_cat_f])
    for f in bs2.columns:
        bs2[f]=bs2[f].astype('category') 
    
    bs3 = pd.concat([bs1[bs1_num_f],bs2], axis=1)
    
    
    features=bs3.columns              # names of columns vector (Index)
    features_type=feature_type(bs3)   # type of each feature vector (Dict)
    

    #Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    BikeSeoul_scaled =pd.DataFrame( min_max_scaler.fit_transform(bs3),  columns= features)
  
    
   
    return BikeSeoul_scaled, target, features, features_type
