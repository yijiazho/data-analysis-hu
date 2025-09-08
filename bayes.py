import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import StandardScaler


path = 'data/Real_Estate.csv'
df = pd.read_csv(path)


# Step 2: Sample 20% of the dataset
df_sampled = df.sample(frac=0.2, random_state=42) 
df_sampled.info()

numeric_column = df_sampled[['Assessed Value']]
scaler = StandardScaler()
X = standardized_column = scaler.fit_transform(numeric_column)

pd.DataFrame(X).applymap(lambda x: abs(x))

print(df_sampled.isna().sum().sum())
print(df_sampled.duplicated().sum())
X = df_sampled.iloc[:, :-1].select_dtypes('number')
y = df_sampled.iloc[:, -1]


bnb = BernoulliNB()
bnb_pred = bnb.fit(X, y).predict(X)
metrics.accuracy_score(y, bnb_pred)
