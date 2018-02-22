from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def bootstrapping():
    for i in range(18):
        bootstrap_size = 0.8 * (len(cancer_df))
        resample(X_train, n_samples=bootstrap_size)


cancer_df = pd.read_csv('Cancer_small.csv')

# create a list of feature names from Start to End feature *exclusive
feature_cols = cancer_df.columns[0:-1]

# create a DataFrame from the selected features
X = cancer_df[feature_cols]

# create a DataFrame using the last column
y = cancer_df[cancer_df.columns[-1]]

# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# print(X_train.shape)
# print(y_train.shape)

# print(X_train)
# print('\n')
# print(y_train)

# print(X_test.shape)
# print(y_test.shape)

# print(X_test)
# print('\n')
# print(y_test)

my_DecisionTree = DecisionTreeClassifier(random_state=2)
my_DecisionTree.fit(X_train, y_train)

y_predict = my_DecisionTree.predict(X_test)

# print(y_predict)

score = accuracy_score(y_test, y_predict)

# print(score)

results = pd.DataFrame()

results['actual'] = y_test
results['prediction'] = y_predict

# print(results)

bootstrapping()
