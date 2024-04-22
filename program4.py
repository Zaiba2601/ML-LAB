import os
import pandas as pd 

filexl=os.path.join("datasets","housing","housing.xlsx")
data = pd.read_excel(filexl) 
print("EXCEL LOAD AND PRINT")
print(data)

file_path=os.path.join("datasets","housing","housing.csv")
df = pd.read_csv(file_path) 
print("CSV LOAD AND PRINT")
print(df)

print("About excel file")
print("Using info() to print details about columns along with datatypes",data.info())
print("Using describe() to print summary of a dataframe",data.describe())
print("Using head() to print starting 5 rows by default",data.head())
print("Using tail() to print last 5 rows by default",data.tail())
print("Using value_counts() to count categorical values of a column",data["ocean_proximity"].value_counts())
print("EXCEL INFO END ")

print("About CSV file")
print("Using info() to print details about columns along with datatypes",df.info())
print("Using describe() to print summary of a dataframe",df.describe())
print("Using head() with - index ",df.head(-3))
print("Using tail() with - index ",df.tail(-3))
print("Using value_counts() to count categorical values of a column",df["ocean_proximity"].value_counts())
print("CSV INFO END ")




