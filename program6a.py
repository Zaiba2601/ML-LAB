import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
# housing.hist()
# plt.show()
# Data Cleaning

# missing values 
    # get rid of corresponding districts
    # get rid of whole attribute 
    # set value to some value (0 or mean or median )
print(housing.describe())
# drop(), dropna(), fillna()
print("#######################################################################################################")
print(housing.dropna(subset=["total_bedrooms"]).describe())
print(housing.drop("total_bedrooms",axis=1))
median = housing["total_bedrooms"].median()
print(median)
housing["total_bedrooms"].fillna(median,inplace=True)
print(housing["total_bedrooms"])
print(housing.info())
# df.method({col: value}, inplace=True)
# print(housing.method({total_bedrooms:median} ,inplace=True))
# df[col] = df[col].method(value)
# housing["total_bedrooms"]=housing["total_bedrooms"].method(median)
# print(housing["total_bedrooms"].method(median))
# print(housing.info())

# Simple Imputer Class


imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
print("##########################################################################################################")
print(imputer.fit(housing_num))
print("##########################################################################################################")
print(imputer.statistics_)
print("##########################################################################################################")
print(housing_num.median().values)

X = imputer.transform(housing_num)
print("##########################################################################################################")
print(X)
print(type(X))
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
print(housing_tr.info())
print(housing_tr)


# Handling text and Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print("##########################################################################################################")
print(housing_cat_encoded[:10])
print("##########################################################################################################")
print("Categories of the encoders are:",ordinal_encoder.categories_)

# converting to 0 or 1 

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1Hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1Hot)
print("##########################################################################################################")
print(housing_cat_1Hot.toarray())
print(cat_encoder.categories_)




