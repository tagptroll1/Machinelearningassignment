import itertools as it

import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt


data = pandas.read_csv("Flaveria.csv", delimiter=",")

n_level = data["N level"]
species = data["species"]
weight  = data["Plant Weight(g)"]

data = pandas.concat([data.drop("N level", axis=1), 
        pandas.get_dummies(data["N level"])], axis=1)

data = pandas.concat([data.drop("species", axis=1),
        pandas.get_dummies(data["species"])], axis=1)

x = data.iloc[:,1:].values
y = data.iloc[:,:1].values

models = {
    linear_model.LinearRegression(): {
        "fit_intercept": True,
        "normalize": False,
        "copy_X": True,
        "n_jobs": 1
    },
    Ridge():{
        "alpha": 1,
        "fit_intercept": True,
        "normalize": False,
        "copy_X": True,
        "max_iter": None,
        "tol": 0.001,
        "solver": "auto",  # ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        "random_state": 2
    },
    KNeighborsRegressor():{
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "brute",  # ["auto", "ball_tree", "kd_tree", "brute"]
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": 1
    }
}

all_scores = dict()
for model, params in models.items():
    all_scores[model.__class__.__name__] = dict()
    all_scores[model.__class__.__name__]["tests"] = dict()
    all_scores[model.__class__.__name__]["tests"]["score"] = list()
    all_scores[model.__class__.__name__]["tests"]["random_state"] = list()
    model_max = 0
    model_min = 0
    model_avg = 0

    regr = model.set_params(**params)
    for i in range(1,100):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=i)

        regr.fit(X_train, y_train)
        all_scores[model.__class__.__name__]["tests"]["score"].append(
            regr.score(X_test, y_test))
        all_scores[model.__class__.__name__]["tests"]["random_state"].append(i)

    model_scores = all_scores[model.__class__.__name__]["tests"]["score"]
    model_max = float(max(model_scores))
    all_scores[model.__class__.__name__]["tests"]["max"] = model_max

    model_min = float(min(model_scores))
    all_scores[model.__class__.__name__]["tests"]["min"] = model_min

    model_avg = sum(model_scores)/len(model_scores)
    all_scores[model.__class__.__name__]["tests"]["avg"] = model_avg

for model, stats in all_scores.items():
    print(f"\n{model}:\n")
    print(f'Max:  {stats["tests"]["max"]:.2f}')
    print(f'Min:  {stats["tests"]["min"]:.2f}')
    print(f'Avg:  {stats["tests"]["avg"]:.2f}\n')
    for score, random in zip(stats["tests"]["score"], stats["tests"]["random_state"]):
        print(f"- {score:>5.2f},  random: {random}")
#print(f"""
#Max score {max(scores):.2f}
#Min score {min(scores):.2f}
#Average score {sum(scores)/len(scores):.2f}""")









"""
X = nparray
y = weights

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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
ridge = Ridge(**ridge_param)
ridge = ridge.fit(X_train, y_train)
#score_rid = ridge.score(X_test, y_test)
#print("ridge score:", f"{score_rid:.2f}")

"""
