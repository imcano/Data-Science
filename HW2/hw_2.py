import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier


# create a DataFrame from label.csv
digits_df = pd.read_csv('label.csv')

# create a list of feature names from Start to End *exclusive
feature_cols = digits_df.columns[0:-1]

# create a DataFrame from the selected features
filename = digits_df[feature_cols]

# create a list from 0 to 64
columns = [i for i in range(64)]

# instantiate the feature matrix
X = pd.DataFrame(columns=columns)

# constructing the feature matrix
for i in range(len(filename)):
     X.loc[i] = np.array(mpimg.imread("Digit/" + str(i) + ".jpg")).flatten()

# prints the feature matrix (1797x64)
print(X)

y = digits_df[digits_df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

my_ANN = MLPClassifier(hidden_layer_sizes=(80,), activation= 'logistic',
                       solver='adam', alpha=1e-5, random_state=1,
                       learning_rate_init = 0.002)

my_ANN.fit(X_train, y_train)

y_predict = my_ANN.predict(X_test)

score_ann = accuracy_score(y_test, y_predict)
print(score_ann)

cm_ANN = confusion_matrix(y_test, y_predict)
print(cm_ANN)

# define a range for the 'number of neurons'
neuron_number = [(i,) for i in range(50,200)]

# create a dictionary for grid parameter:
param_grid = dict(hidden_layer_sizes = neuron_number)

# instantiate the model:
my_ANN2 = MLPClassifier(activation='logistic', solver='adam',
                        alpha=1e-5, random_state=1,
                        learning_rate_init = 0.002)

grid = GridSearchCV(my_ANN, param_grid, cv=10, scoring='accuracy')

grid.fit(X, y)

print(grid.best_score_)
print(grid.best_params_)


