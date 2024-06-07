import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
file_path = os.path.join("datasets","housing","housing.csv")

housing = pd.read_csv(file_path)
housing["median_income"].hist()
housing.hist()
plt.show()

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.,4.5,6,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()
plt.show()

split= StratifiedShuffleSplit(n_splits=1, test_size = 0.2 , random_state = 21)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("\n\n Percentage of income category in complete housing dataset ranging from 1-5 ")
print(housing["income_cat"].value_counts()/len(housing))

print("\n\n Percentage of income category in stratified sampling train dataset ranging from 1-5 ")
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))

print("\n\n Percentage of income category in stratified sampling test dataset ranging from 1-5 ")
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

housing.plot(kind="scatter",x="longitude",y="latitude",title="scatter plot with all visible datapoints")
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1,title="scatter plot with transparency")
plt.show()

housing.plot(kind="scatter",x="longitude",y="latitude",alpha = 0.4 , s=housing["population"]/100 ,label="population" ,figsize =(10,7,) , c="median_house_value" , cmap=plt.get_cmap("jet"), colorbar= True )
plt.show()

pd.plotting.scatter_matrix(housing)
plt.show()

housing.plot(kind="scatter",x="median_income", y="median_house_value" , alpha=0.1)
plt.show()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_household"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr(method="pearson" , numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Bar chart of mean median_house_value by income_cat
#Extra added to based on the requirements of question
income_cat_means = housing.groupby("income_cat")["median_house_value"].mean()
income_cat_means.plot(kind="bar", title="Mean Median House Value by Income Category", ylabel="Mean Median House Value", xlabel="Income Category")
plt.show()
