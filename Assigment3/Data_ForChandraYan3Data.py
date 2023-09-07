import pandas as pd


readDatafromCSV = pd.read_csv(
    r"C:\Users\talkt\Desktop\python_detail\AIWithPython\TestData\propulsion_module.csv"
)
print(readDatafromCSV.info())

print(readDatafromCSV.head())

# check if data has some null values
print("Checking if csv have null values", readDatafromCSV.isnull().sum())

print("Describe the Dataset", readDatafromCSV.describe())
