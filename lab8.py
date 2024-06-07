from sklearn.linear_model import LinearRegression
import os 
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error

file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
median = housing["total_bedrooms"].median()
print(median)
housing["total_bedrooms"].fillna(median,inplace=True)
housing_num = housing.drop("ocean_proximity", axis=1)

housing_num['income_cat'] = pd.cut(x=housing_num['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
print(housing_num.head())
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X=housing_num, y=housing_num['income_cat']):
    strat_train_set = housing_num.loc[train_index]
    strat_test_set = housing_num.loc[test_index]

print(strat_test_set.head())
print(strat_train_set.head())

strat_train_set_labels = strat_train_set["median_house_value"].copy()
strat_train_set = strat_train_set.drop("median_house_value", axis=1)

strat_test_set_labels = strat_test_set["median_house_value"].copy()
strat_test_set = strat_test_set.drop("median_house_value", axis=1)

print("\n \n ######################################### \n")
lin_reg = LinearRegression()
lin_reg.fit(X=strat_train_set, y=strat_train_set_labels)

print("\n model coeff :",lin_reg.coef_)
print("\n model Intercept ",lin_reg.intercept_)
print("\n score for trainin data (R^2)",lin_reg.score(strat_train_set,strat_train_set_labels))
print("\n score (R^2) for testing data ",lin_reg.score(strat_test_set,strat_test_set_labels))

housing_prediction  = lin_reg.predict(strat_test_set)
lin_mse = mean_squared_error(strat_test_set_labels,housing_prediction)
lin_rmse = np.sqrt(lin_mse)
print("\n RMSE for linear regression",lin_rmse)
print("mean absolute error",mean_absolute_error(strat_test_set_labels,housing_prediction))  
print("\n ##################################### \n \n ")




from sklearn.linear_model import Ridge

ridgemodel = Ridge(alpha=10).fit(strat_train_set,strat_train_set_labels)
print("Ridge model coeff :",ridgemodel.coef_)
print("Ridge model Intercept ",ridgemodel.intercept_)
print("Ridge score for trainin data (R^2)",ridgemodel.score(strat_train_set,strat_train_set_labels))
print("Ridge score (R^2) for testing data ",ridgemodel.score(strat_test_set,strat_test_set_labels))

housing_prediction  = ridgemodel.predict(strat_test_set)
lin_mse = mean_squared_error(strat_test_set_labels,housing_prediction)
lin_rmse = np.sqrt(lin_mse)
print("RMSE for linear regression",lin_rmse)
print("mean absolute error",mean_absolute_error(strat_test_set_labels,housing_prediction))  
print("\n ##################################### \n \n")



from sklearn.linear_model import Lasso

lassomodel = Lasso(alpha=10,max_iter=10000).fit(strat_train_set,strat_train_set_labels)
print("Lasso model coeff :",lassomodel.coef_)
print("Lasso model Intercept ",lassomodel.intercept_)
print("Lasso score for trainin data (R^2)",lassomodel.score(strat_train_set,strat_train_set_labels))
print("Lasso score (R^2) for testing data ",lassomodel.score(strat_test_set,strat_test_set_labels))

housing_prediction  = lassomodel.predict(strat_test_set)
lin_mse = mean_squared_error(strat_test_set_labels,housing_prediction)
lin_rmse = np.sqrt(lin_mse)
print("RMSE for linear regression",lin_rmse)
print("mean absolute error",mean_absolute_error(strat_test_set_labels,housing_prediction))  
print(" number of features used are :",np.sum(lassomodel.coef_ !=0))
print("\n ##################################### \n \n")

