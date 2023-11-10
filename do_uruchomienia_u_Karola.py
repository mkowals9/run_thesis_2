import joblib
import json
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

def run_thesis():
    # WCZYTYWANIE DANYCH WEJŚCIOWYCH - decybelowo
    # Reading JSON from a file
    with open('decybelowo_jsons/data_intend_decybelowo_z_e.json', 'r') as json_file:
        data_intend = json.load(json_file)

    # Reading JSON from a file
    with open('decybelowo_jsons/data_phase_decybelowo_z_e.json', 'r') as json_file:
        data_phase = json.load(json_file)

    # Reading JSON from a file
    with open('decybelowo_jsons/data_period_decybelowo_z_e.json', 'r') as json_file:
        data_period = json.load(json_file)

    # Reading JSON from a file
    with open('decybelowo_jsons/data_length_decybelowo_z_e.json', 'r') as json_file:
        data_length = json.load(json_file)

    # Reading JSON from a file
    with open('decybelowo_jsons/combined_data_decybele_final.json', 'r') as json_file:
        data_new = json.load(json_file)

    # mieszanie listy
    data = data_intend + data_phase + data_period + data_length + data_new
    random.shuffle(data)

    #podział na dane wejściowe i wyjściowe

    X_array = []
    y_array = []

    for single_simulation in data:
        temp_y = [single_simulation['intend'], single_simulation['length'], single_simulation['phase'], single_simulation['period']]
        y_array.append(temp_y)
        X_array.append(single_simulation['results'])

    #podział na dane treningowe i testowe

    X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(X_array, y_array, test_size=0.25, random_state=1010801)

    #zmiana list na numpy arrays

    X_train = np.array(X_train_array)
    X_test = np.array(X_test_array)
    y_train = np.array(y_train_array)
    y_test = np.array(y_test_array)

    #skalowanie

    sc = StandardScaler()
    sc_2 = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    sc_2.fit(y_train)
    y_train = sc_2.transform(y_train)
    y_test = sc_2.transform(y_test)

    # model z XGBOOST i GridSearch

    param_grid = {
        "estimator__device": ["cuda"],
        "estimator__tree_method": ['hist'],
        "estimator__gamma": [0.1, 0.5, 1, 1.5, 2, 0.05],
        "estimator__learning_rate": [0.05, 0.1, 0.2, 0.25, 0.01, 0.02, 0.03],
        "estimator__subsample": [0.1, 0.5, 0.7, 1],
        "estimator__n_estimators": [800, 1000],
        "estimator__max_depth": [10, 20, 50]
    }
    #to turn off logs, remove verbose
    grid_search = GridSearchCV(MultiOutputRegressor(XGBRegressor()), param_grid=param_grid, cv=3, verbose=1)
    grid_search.fit(X_train, y_train)
    print("BEST PARAMS ", grid_search.best_params_, " najlepszy wynik walidacji krzyzowej ", grid_search.best_score_)

    y_pred = grid_search.best_estimator_.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2) Score:", r2)

    my_model = grid_search.best_estimator_

    # Save the model to a file

    joblib.dump(grid_search.best_estimator_, 'xgboost_decybele_grid_search_1.pkl')