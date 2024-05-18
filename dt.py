import os
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay


file_path = os.path.join("datasets","newdata","newdata.csv")

df = pd.read_csv(file_path)
# nd = np.array(df)
X = df.iloc[:, :3].values
y = df.iloc[:, -1].values

print(X)
print(y)

X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0)

clf = DecisionTreeClassifier(max_leaf_nodes=10,random_state=0)
clf.fit(X_train,y_train)
tree.plot_tree(clf)
plt.show()

y_pred = clf.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()