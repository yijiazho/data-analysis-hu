import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, ShuffleSplit, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc



np.random.seed(42)
# url = 'https://raw.githubusercontent.com/yhat/demo-churn-pred/master/model/churn.csv'
url = 'churn.csv'
churn = pd.read_csv(url)

print(churn.info())

churn['Int\'l Plan'] = churn['Int\'l Plan'].map(dict(yes=1, no=0))
churn['VMail Plan'] = churn['VMail Plan'].map(dict(yes=1, no=0))

print(churn.info())

num_vars = churn.select_dtypes(['int64', 'float64']).columns
# all columns except State and Churn?
print(num_vars)
X = churn[num_vars]
y = churn['Churn?'].map({'True.': 1, 'False.': 0})

print('-------------------------------------------')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_test.shape)
print(y_train.shape)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)


## Deep Learning Model Architecture
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(6, activation='relu', input_shape=(18, ))) # 114 parameters: (18+1)*6=114
model.add(Dense(6, activation='relu')) # (6+1)*6 = 42 parameters
model.add(Dense(1, activation='sigmoid')) # 7 parameters
print(model.output_shape)
print(model.summary())

print(model.get_config())
print(model.get_weights())
## Model Specification and Training

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=18, batch_size=5) # Consider using validation set

print(model.summary())

# Predict probabilities
y_pred_prob = model.predict(X_test)

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()