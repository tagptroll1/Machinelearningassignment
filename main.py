import sklearn as skl
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

n_level = {
    "L":0,
    "M":1,
    "H":2
}

species = {
    "brownii":0,
    "pringlei":1,
    "trinervia":2,
    "ramosissima":3,
    "robusta":4,
    "bidentis":5
}

def get_binary_n(n_input):
    byte_array = [0, 0, 0]
    byte_array[n_input] = 1
    return byte_array

def get_binary_species(n_species):
    byte_array = [0, 0, 0, 0, 0, 0]
    byte_array[n_species] = 1
    return byte_array

with open("Flaveria.csv", "r") as csv:
    #contains dataset with n_level, species and plant_weight
    csv_lines = csv.readlines() 

array = []
for line in csv_lines:
    line = line.replace("\n","")
    array.append(line.split(","))

nparray = np.array(array)



X_binary = []
X = list()
y = []
for entry in nparray[1:]:
    n_level_entry = n_level[entry[0]]
    species_entry = species[entry[1]]
    _L, _M, _H = get_binary_n(n_level_entry)
    _A, _B, _C, _D, _E, _F = get_binary_species(species_entry)

    weight = float(entry[2])
    X.append([n_level_entry, species_entry])
    X_binary.append([_L, _M, _H, _A, _B, _C, _D, _E, _F])
    y.append(weight)


X_binary = np.array(X_binary)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y, test_size=0.3)

Xx_train, Xx_test, yy_train, yy_test = train_test_split(X, y, test_size=0.3)
#print("X_train",X_train, "y_train", y_train)
#print("X_test",X_test, "y_test", y_test)


## Testing
tree_param = {
    "criterion": "mse", 
    "splitter": "best", 
    "max_depth": None, 
    "min_samples_split": 3, 
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0, 
    "max_features": None, 
    "random_state": None, 
    "max_leaf_nodes": None, 
    "min_impurity_decrease": 0.0, 
    "min_impurity_split": None, 
    "presort": False
}
tree1 = tree.DecisionTreeRegressor(**tree_param)
tree1 = tree1.fit(X_train, y_train)
tree_score = tree1.score(X_test, y_test)
print(f"Tree score: {tree_score:.2f}")

tree2 = tree.DecisionTreeRegressor(**tree_param)
tree2 = tree2.fit(Xx_train, yy_train)
tree_score2 = tree2.score(Xx_test, yy_test)
print(f"Tree2 score: {tree_score2:.2f}")


linearRegression_param = {
    "fit_intercept": True,
    "normalize": False,
    "copy_X": True,
    "n_jobs": 1
}
lr = linear_model.LinearRegression(**linearRegression_param)
lr = lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print(f"Score: {score:.2f}")

knn_param = {
    "n_neighbors": 5, 
    "weights": "uniform", 
    "algorithm": "auto", 
    "leaf_size": 30, 
    "p": 2, 
    "metric": "minkowski", 
    "metric_params": None, 
    "n_jobs": 1
}
knn = KNeighborsRegressor(**knn_param)
knn = knn.fit(X_train, y_train)
score_knn = knn.score(X_test, y_test)
print("knn score:", f"{score_knn:.2f}")

ridge_param = {
    "alpha": 1.0, 
    "fit_intercept": True, 
    "normalize": False, 
    "copy_X": True, 
    "max_iter": None, 
    "tol": 0.001, 
    "solver": "auto", 
    "random_state": None
}
rid = Ridge(**ridge_param)
rid = rid.fit(X_train, y_train)
#score_rid = rid.score(X_test, y_test)
#print("ridge score:", f"{score_rid:.2f}")

"""
lgr = LogisticRegression()

lgr.fit(X_train, y_train)
score_lgr = lgr.score(X_test, y_test)
print("lgr score:", f"{score_lgr:.2f}")
prediction_lgr = np.array([[n_level["M"], species["trinervia"]]])
print("lgr prediction:", lgr.predict(prediction_lgr))
"""
