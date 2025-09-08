import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import missingno as msn
import matplotlib.pyplot as plt

np.random.seed(66)

df = pd.read_csv('https://raw.githubusercontent.com/yijiazho/dataset/main/Real_Estate.csv')
print(df.info())

property_type_mapping = {
    'Commercial': 1,
    'Residential': 2,
    'Vacant Land': 3,
    'Apartments': 4,
    'Industrial': 5,
    'Public Utility': 6,
    'Condo': 7,
    'Two Family': 8,
    'Three Family': 9,
    'Single Family': 10,
    np.nan: 0  # Using np.nan to handle NaN values directly
}
df['Property Type'] = df['Property Type'].map(property_type_mapping)

residential_type_mapping = {
    'Single Family': 1,
    'Two Family': 2,
    'Three Family': 3,
    'Four Family': 4,
    'Condo': 5,
    np.nan: 0  # Using np.nan to handle NaN values directly
}
df['Residential Type'] = df['Residential Type'].map(residential_type_mapping)
print(df.select_dtypes('object').columns)
print(df.select_dtypes('number').columns)
print('--------------------------------------------')

# Too many nulls and objects, select some columns, especially numbers for training

# Select relevant columns
columns_to_use = ['Serial Number', 'List Year', 'Date Recorded', 'Assessed Value', 'Sale Amount', 'Sales Ratio', 'Property Type']
X = df[columns_to_use].copy()

# Convert date columns using .loc to avoid SettingWithCopyWarning
X.loc[:, 'Date Recorded'] = pd.to_datetime(X['Date Recorded'])
X.loc[:, 'Date Recorded'] = X['Date Recorded'].map(pd.Timestamp.toordinal)

X = X.dropna()
y = df['Residential Type']
y = y.dropna()

# Ensure indices match between X and y after dropping rows
X, y = X.align(y, join='inner', axis=0)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f"Train data size is {X_train.shape}")

clf = DecisionTreeClassifier()
clf
print(X_train.info())


## Model Hyperparameter Fine Tuning
print('--------------------------------------------')
param_grid = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 10, 20],
              'max_depth': [5, 10, 20, 25, 30],
              'min_samples_leaf': [1, 5, 10],
              'max_leaf_nodes': [2, 5, 10, 20]}
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy',verbose=3)
grid.fit(X_train, y_train)
print('Fit down')

## Output Best Model Hyperparameters

print(grid.best_score_)
for hps, values in grid.best_params_.items():
  print(f"{hps}: {values}")
  
# Visualization

plt.figure(figsize=(16, 8))
df['Property Type'].value_counts(normalize=True).plot.bar(rot=45)
plt.xlabel("Property Type")
plt.ylabel("Distribution")