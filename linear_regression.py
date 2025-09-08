## Import Python Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

housing = fetch_california_housing()

housing_data = pd.DataFrame(housing.data, columns = housing.feature_names)
housing_data['Price'] = housing.target
X = housing_data.drop('Price', axis=1)
y = housing_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(housing_data.info())


# df = pd.read_csv('star_dataset.csv')
# print(df.info())
# selected_columns = ['Distance (ly)', 'Luminosity (L/Lo)', 'Radius (R/Ro)']

# X = df[selected_columns]
# y = df['Temperature (K)']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lr_pipe = LinearRegression()
lr_pipe.fit(X_train, y_train)
y_pred = lr_pipe.predict(X_test)
print(f"RMSE: {round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 3)}")


plt.figure(figsize=(16, 8))
plt.scatter(y_test, y_pred, s=10, alpha=0.3)
plt.plot([0, 4], [0, 4], '--k')
plt.axis('tight')
plt.tight_layout()
plt.xlabel('True price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.show()

for idx, col_name in enumerate(X_train.columns):
    print(f"The coefficient for {col_name} is {lr_pipe.coef_[idx]}")
  
plt.scatter(lr_pipe.predict(X_train), lr_pipe.predict(X_train)-y_train, c='b', s=10, alpha=0.3)
plt.hlines(y=0, xmin=0, xmax=10)
plt.xlabel("Fitted")
plt.ylabel("Residuals")
plt.title("Residuals vs. fitted")
plt.show()


df2 = housing_data.copy()
print(df2.info())
df2['Price'] = pd.qcut(df2['Price'], 2, labels=["low", "high"])
X2 = df2.drop('Price', axis=1)
y2 = df2['Price']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=16)
df2.Price.value_counts()

print('--------------------------------')
logr_pipe = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))
logr_pipe.fit(X_train2, y_train2)
y_pred2 = logr_pipe.predict(X_test2)
print(y_pred2)
pd.Series(y_pred2).value_counts()
res = pd.DataFrame(pd.Series(y_pred2).value_counts(), columns=['count']).assign(pct=lambda x: x['count'] / x['count'].sum())
print(res.info())

from sklearn.metrics import roc_curve, roc_auc_score

# Predict probabilities for the positive class
y_pred_prob = logr_pipe.predict_proba(X_test2)[:, 1]

# Compute ROC curve and ROC area for each class
fpr, tpr, thresholds = roc_curve(y_test2, y_pred_prob, pos_label="high")
roc_auc = roc_auc_score(y_test2, y_pred_prob)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()