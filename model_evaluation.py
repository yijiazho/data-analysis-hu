import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, ShuffleSplit, LeaveOneOut
from sklearn import metrics
from matplotlib import pyplot as plt




def main():
    np.random.seed(66)
    df = pd.read_csv('https://raw.githubusercontent.com/yijiazho/dataset/main/Real_Estate.csv')
    print(df.info())
    df_cleaned = df.dropna(subset=['Residential Type'])
    df_cleaned = df.dropna(subset=['Property Type'])
    df_cleaned.info()
    print('-----------------------------------------------')
    
    accepted_values = ['Commercial', 'Residential']
    msk = df_cleaned['Property Type'].isin(accepted_values)
    df_filtered = df_cleaned[msk]   
    
    
    # Use a 10% subset of the filtered data for processing
    subset_size = int(len(df_filtered) * 0.1)
    df_subset = df_filtered.sample(subset_size, random_state=66)
    
    
    columns_to_use = ['List Year', 'Assessed Value', 'Sale Amount', 'Sales Ratio']
    X = df_filtered[columns_to_use].copy()
    y = df_filtered['Property Type'].map({'Commercial': 1, 'Residential': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)    
    print('-----------------------------------------------')

    plt.hist(y_pred, bins=2, edgecolor='black')
    plt.xticks(ticks=[0, 1], labels=['Residential (0)', 'Commercial (1)'])
    plt.title('Distribution of Predicted Property Types')
    plt.xlabel('Predicted Property Type')
    plt.ylabel('Number of Predictions')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()
    print('-----------------------------------------------')

    print(f"Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100, 2)}%")

    df_confusion = pd.crosstab(y_test, y_pred)
    df_confusion.index = [['Real', 'Real'], ['Stay', 'Leave']]
    df_confusion.columns = [['Predict'] * 2, ['Stay', 'Leave']]
    print(df_confusion)
    print('-----------------------------------------------')

    print(metrics.classification_report(y_test, y_pred))
    
    param_grid = {'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 10, 20, 30],
                'max_depth': [4, 5, 6, 10, 15, 20],
                'min_samples_leaf': [ 1, 5, 10],
                'max_leaf_nodes': [2, 5, 10, 20]}
    grid = GridSearchCV(clf, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(grid)       
    print('-----------------------------------------------')

    bstrap = ShuffleSplit(n_splits=10, test_size=0.3, random_state=16)
    print(bstrap)
    grid_bstrap = GridSearchCV(clf, param_grid, cv=bstrap)
    grid_bstrap.fit(X_train, y_train)
    print(grid_bstrap)
    print('-----------------------------------------------')
    ## Leave One Out

    loocv = LeaveOneOut()
    lv_score = cross_val_score(clf, X, y, cv=loocv)
    print(f"Leave One Out accuracy is {round(lv_score.mean(), 2)}")
    
    y_pred_prob = grid.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"AUC = {auc}")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    import seaborn as sns
    sns.lineplot(fpr, tpr)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')

if __name__ == "__main__":
    main()