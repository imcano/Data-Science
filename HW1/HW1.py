from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import numpy as np
import pandas as pd


def grunt_work():

    cancer_df = pd.read_csv('Cancer_small.csv')

    # create a list of feature names from Start to End feature *exclusive
    feature_cols = cancer_df.columns[0:-1]

    # create a DataFrame from the selected features
    X = cancer_df[feature_cols]

    # create a DataFrame using the last column
    y = cancer_df[cancer_df.columns[-1]]

    # create testing and training DataFrames
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    my_DecisionTree = DecisionTreeClassifier(random_state=2)
    my_DecisionTree.fit(X_train, y_train)

    y_predict = my_DecisionTree.predict(X_test)

    dt_score = accuracy_score(y_test, y_predict)

    results = pd.DataFrame()

    results['actual'] = y_test
    results['prediction'] = y_predict

    print(dt_score)

    b_results = []

    for i in range(19):
        bootstrap_size = 0.8 * (X_train.shape[0])
        B, c = resample(X_train, y_train, n_samples=int(bootstrap_size), random_state=i, replace=True)

        Base_DecisionTree = DecisionTreeClassifier(random_state=2)
        Base_DecisionTree.fit(B, c)

        c_predict = Base_DecisionTree.predict(X_test)

        b_results.append(c_predict)

    col = len(b_results[0])
    row = len(b_results)

    voting = []

    for i in range(col):
        vote_col = 0
        for j in range(row):
            if b_results[j][i] == 0:
                vote_col -= 1
            else:
                vote_col += 1
        if vote_col < 0:
            vote_col = 0
        elif vote_col > 0:
            vote_col = 1

        voting.append(vote_col)

    b_score = accuracy_score(y_test, voting)

    bagging_results = pd.DataFrame()
    bagging_results['actual'] = y_test
    bagging_results['prediction'] = voting

    print(b_score)

    my_AdaBoost = AdaBoostClassifier(n_estimators=19, random_state=2)
    my_AdaBoost.fit(X_train, y_train)

    ab_predict = my_AdaBoost.predict(X_test)

    ab_score = accuracy_score(y_test, ab_predict)

    ab_results = pd.DataFrame()
    ab_results['actual'] = y_test
    ab_results['prediction'] = ab_predict

    print(ab_score)

    my_RandomForest = RandomForestClassifier(n_estimators=19, bootstrap=True, random_state=2)
    my_RandomForest.fit(X_train, y_train)

    rf_predict = my_RandomForest.predict(X_test)

    rf_score = accuracy_score(y_test, rf_predict)

    rf_results = pd.DataFrame()
    rf_results['actual'] = y_test
    rf_results['prediction'] = rf_predict

    print(rf_score)

grunt_work()
