#import statements
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

from scipy.stats import skew



#Ordinal text to numeric helper functions

def condToNum(df, col):
    df = df.fillna({col: "NA"})
    condition_nums = {col:  {"NA":0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}}
    
    df.replace(condition_nums, inplace=True)
    return df

def finToNum(df, col):
    df = df.fillna({col: "NA"})
    fin_nums = {col:  {"NA":0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}}
    
    df.replace(fin_nums, inplace=True)
    return df

def funcToNum(df, col):
    df = df.fillna({col: "NA"})
    func_nums = {col:  {"Sal":0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7, "NA":7}}
    
    df.replace(func_nums, inplace=True)
    return df

def shapeToNum(df, col):
    df = df.fillna({col: "NA"})
    shape_nums = {col:  {"IR3":0, "IR2": 1, "IR1": 2, "Reg": 3, "NA": 3}}
    
    df.replace(shape_nums, inplace=True)
    return df

def slopeToNum(df, col):
    df = df.fillna({col: "NA"})
    slope_nums = {col:  {"Gtl":0, "Mod": 1, "Sev": 2, "NA": 0}}
    
    df.replace(slope_nums, inplace=True)
    return df

def garFinToNum(df, col):
    df = df.fillna({col: "NA"})
    slope_nums = {col:  {"NA":0, "Unf": 1, "RFn": 2, "Fin": 3}}
    
    df.replace(slope_nums, inplace=True)
    return df

def exposToNum(df, col):
    df = df.fillna({col: "None"})
    expos_nums = {col:  {"None":0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}}
    
    df.replace(expos_nums, inplace=True)
    return df

#Use DataFrame for preprocessing of text ordinal features

def text_preprocessing(df):
    qual_ordin_cols= ['HeatingQC', 'FireplaceQu', 'PoolQC', 'GarageQual'
                      , 'GarageCond', 'ExterQual', 'BsmtQual', 'BsmtCond', 'KitchenQual']
    for col in qual_ordin_cols:
        df = condToNum(df,col)
    
    df = shapeToNum(df,'LotShape')
    df = slopeToNum(df,'LandSlope')
    df = garFinToNum(df,'GarageFinish')
    df = finToNum(df,'BsmtFinType1')
    df = finToNum(df,'BsmtFinType2')
    df = funcToNum(df, 'Functional')
    df = exposToNum(df,'BsmtExposure')
    df = df.fillna({'Fence': 'None'})
    df = df.fillna({'Alley': 'None'})
    df = df.fillna({'GarageType': 'None'})
    df = df.fillna({'MasVnrType': 'None'})
    
    return df

def remove_outliers(df):
    #Remove obvious outliers
    
    liv_cut = df[(df['GrLivArea']>4000)& (df['SalePrice']<300000)]
    
    lot_cut = df[((df['LotArea']>30000) & (df['SalePrice']<500000)) | 
                   ((df['LotFrontage']>300) & (df['SalePrice']<500000))]
    bsmt_cut = df[(df['TotalBsmtSF']>0)&(df['TotalBsmtSF']>5000)& (df['SalePrice']<200000)]
    
    indicies = liv_cut.index.values.tolist()
    
    for ind in lot_cut.index.values.tolist():
        if ind not in indicies:
            indicies.append(ind)
    
    for ind in bsmt_cut.index.values.tolist():
        if ind not in indicies:
            indicies.append(ind)
            
    df.drop(indicies, inplace=True)
    
    return df


def add_special_dummies(df, a):
    #Add dummies from continuous variables to better handle zeros
    
    df['has_YearRemod'] = np.where(
                        (df['YearRemodAdd']!=df['YearBuilt'])& (
                         df['YearRemodAdd']>1950), 1,0)
    
    df['has_allUtil'] = np.where(df['Utilities']=='AllPub',1,0)
    
    df['has_normCond1'] = np.where(df['Condition1']=='Norm',1,0)
    df['has_normCond2'] = np.where(df['Condition2']=='Norm',1,0)
    
    df['roof_isCompShg'] = np.where(df['RoofMatl']=='CompShg',1,0) 
    
    df['has_MasVnr'] = np.where(df['MasVnrArea']>0, 1,0)
    df['has_Basement'] = np.where(df['TotalBsmtSF']>0, 1,0)
    df['has_LowQualSF'] = np.where(df['LowQualFinSF']>0, 1,0)
    df['has_EncPor'] = np.where(df['EnclosedPorch']>0, 1,0)
    df['has_3SsnPor'] = np.where(df['3SsnPorch']>0, 1,0)
    df['has_WoodDeck'] = np.where(df['WoodDeckSF']>0, 1,0)
    df['has_OpenPorch'] = np.where(df['OpenPorchSF']>0, 1,0)
    df['has_ScnPorch'] = np.where(df['ScreenPorch']>0, 1,0)
    df['has_Pool'] = np.where(df['PoolArea']>0, 1,0)
    
    
    cols_added = ['has_YearRemod', 'has_allUtil', 'has_normCond1', 'has_normCond2'
                  ,'roof_isCompShg', 'has_MasVnr', 'has_Basement', 'has_LowQualSF'
                  , 'has_EncPor', 'has_3SsnPor', 'has_WoodDeck', 'has_OpenPorch'
                  , 'has_ScnPorch', 'has_Pool']
    
    for col in cols_added:
        a.append(col)
    
    return df, a

def add_attributes(df, n, a, combine_garage=False):
    
    #add additional attributes
    #Total bathrooms, combine garage features, collapse rare features into 'other' 
    
    df['Bathrooms_tot'] = df['BsmtFullBath']+0.5*df['BsmtHalfBath']+df['FullBath']+0.5*df['HalfBath']
    df['Bathrooms_tot'] = np.where(df['Bathrooms_tot'] > 3.5, 4, df['Bathrooms_tot'])
    
    if combine_garage:
        df['GarageComb'] = df['GarageArea']*df['GarageCars']
        n.append('GarageComb')
        
    else:
        df['has_Garage'] = np.where(df['GarageArea']>0, 1,0)
        a.append('has_Garage')
        n.append('GarageArea')
        
    #collapse some of the rare features into 'Other'    
    ext1 = df.groupby(['Exterior1st'])['Id'].count()
    ext1 = ext1[ext1 <5]
    
    for ext in ext1.index:
        df['Exterior1st'] = np.where(df['Exterior1st']==ext, 'Other', df['Exterior1st'])
        
    ext2 = df.groupby(['Exterior2nd'])['Id'].count()
    ext2 = ext2[ext2 <5]    
    
    for ext in ext2.index:
        df['Exterior2nd'] = np.where(df['Exterior2nd']==ext, 'Other', df['Exterior2nd'])
        
    sale = df.groupby(['SaleType'])['Id'].count()
    sale = sale[sale < 8]
    
    for s in sale.index:
        df['SaleType'] = np.where(df['SaleType']==s, 'Other', df['SaleType'])
        
    n.append('Bathrooms_tot')
    
    
    return df, n, a

def preprocess_features(df, numer, addedc, combine_garage=False):
    #Combine preprocessing steps
    
    df_temp = text_preprocessing(df)
    df_temp, addedc = add_special_dummies(df_temp, addedc)
    df_temp, numer, addedc = add_attributes(df_temp, numer, addedc, combine_garage=combine_garage)
    
    return df_temp, numer, addedc

def process_features(df, dftest, numeric, cate, added,drop=True, add_poly=False):
    

    add_poly=add_poly

    #Numerical columns
    imputer_med = Imputer(strategy='median')
    scaler = StandardScaler()

    df_tr = df[numeric+cate+added].copy()
    dftest_tr = dftest[numeric+cate+added].copy()
    #reset index
    df_tr.reset_index(drop=True, inplace=True)
    dftest_tr.reset_index(drop=True, inplace=True)

    #Numerical columns
    
    df_tr[numeric] = imputer_med.fit_transform(df_tr[numeric])
    
    dftest_tr[numeric] = imputer_med.transform(dftest_tr[numeric])
    
    for d in [df_tr, dftest_tr]:
        if add_poly:
            new_cols=[]
            for c in numeric:
                d[c+'^2']=d[c]*d[c]
                d[c+'^3']=d[c]*d[c]*d[c]
                new_cols.append(c+'^2')
                new_cols.append(c+'^3')
        
            all_num = numeric+new_cols
    
        else:
            all_num = numeric 
            
    skewness = df_tr[all_num].apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]
    print(str(skewness.shape[0]) + " skewed numerical features to log transform")
    skewed_features = skewness.index
    df_tr[skewed_features] = np.log1p(df_tr[skewed_features])
    dftest_tr[skewed_features] = np.log1p(dftest_tr[skewed_features])
        
    #Scale data
    df_tr[all_num] = scaler.fit_transform(df_tr[all_num])  
    dftest_tr[all_num] = scaler.transform(dftest_tr[all_num])
    
    #Categorical Columns    
    final_df = pd.get_dummies(df_tr, columns = cate, drop_first=drop)
    final_df_test = pd.get_dummies(dftest_tr, columns = cate, drop_first=drop)
    
    
    return final_df, final_df_test
