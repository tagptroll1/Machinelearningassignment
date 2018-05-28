import sklearn as skl
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
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

X = []
y = []

for entry in nparray[1:]:
    n_level_entry = n_level[entry[0]]
    species_entry = species[entry[1]]
    _L, _M, _H = get_binary_n(n_level_entry)
    _A, _B, _C, _D, _E, _F = get_binary_species(species_entry)

    weight = float(entry[2])
    X.append([_L, _M, _H, _A, _B, _C, _D, _E, _F])
    y.append(weight)

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print("X_train",X_train, "y_train", y_train)
#print("X_test",X_test, "y_test", y_test)
## Testing
linearRegression_param = {
    "fit_intercept": True,
    "normalize": False,
    "copy_X": True,
    "n_jobs": 1
}
lr = linear_model.LinearRegression(**linearRegression_param)
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print("Score",f"{score:.2f}")

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
knn.fit(X_train, y_train)
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
rid.fit(X_train, y_train)
score_rid = rid.score(X_test, y_test)
print("ridge score:", f"{score_rid:.2f}")

"""
lgr = LogisticRegression()

lgr.fit(X_train, y_train)
score_lgr = lgr.score(X_test, y_test)
print("lgr score:", f"{score_lgr:.2f}")
prediction_lgr = np.array([[n_level["M"], species["trinervia"]]])
print("lgr prediction:", lgr.predict(prediction_lgr))
"""
