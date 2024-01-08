from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor


from xgboost import XGBRegressor 
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from pickle import dump

def createLinearRegression(X_train, y_train, X_test, y_test):
    model = LinearRegression().fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/LinearRegression.pkl', 'wb'))


def createDecisionTree(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor().fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/DecisionTree.pkl', 'wb'))

def createRandomForest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor().fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/RandomForest.pkl', 'wb'))

def createXGBregressor(X_train, y_train, X_test, y_test):
    model = XGBRegressor().fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/XGBRegressor.pkl', 'wb'))

def createLogisticRegression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(class_weight= 'balanced').fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_acc = accuracy_score(y_train, y_fit)
    test_acc = accuracy_score(y_test, y_pred)

    y_pred_proba = pd.Series(model.predict_proba(X_test)[:,1], index = X_test.index)

    print("Train score:", train_acc)
    print("Test score:", test_acc)
    print(y_pred_proba)

    dump(model, open('pkl_models/LogisticRegression.pkl', 'wb'))


def createAndSaveScaler(data):
    lat_scaler = StandardScaler().fit(data[['LAT_WGS84']])
    long_scaler = StandardScaler().fit(data[['LONG_WGS84']])

    data['LAT_WGS84'] = lat_scaler.transform(data[['LAT_WGS84']])
    data['LONG_WGS84'] = long_scaler.transform(data[['LONG_WGS84']])

    dump(lat_scaler, open('pkl_models/lat_scaler.pkl', 'wb'))
    dump(long_scaler, open('pkl_models/long_scaler.pkl', 'wb'))
    return data

def loadScaler(lat_scaler_path, long_scaler_path):
    return pickle.load(open(lat_scaler_path, 'rb')), pickle.load(open(long_scaler_path, 'rb'))

def loadModel(model_path):
    return pickle.load(open('pkl_models/' + model_path, 'rb'))