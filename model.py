from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, recall_score, precision_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

from xgboost import XGBRegressor 
import pandas as pd
import matplotlib.pyplot as plt

import pickle
from pickle import dump

def createDeterministicProcessIndex(data, fourier_freq = 'H', fourier_order = 2):

    '''
    Taking too much memory. Unsuitable
    '''
    fourier = CalendarFourier(fourier_freq, fourier_order)

    dp = DeterministicProcess(
        index = data.index,
        constant = True,
        order = fourier_order,
        seasonal= True,
        additional_terms=[fourier]
    )

    X = dp.in_sample()
    modified_data = data.join(X, how = 'left')
    return modified_data

def createLinearRegression(X_train, y_train, X_test, y_test):
    # model = LinearRegression(random_state= 0).fit(X_train, y_train)
    model = LinearRegression().fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/LinearRegression.pkl', 'wb'))


def createDecisionTreeRegressor(X_train, y_train, X_test, y_test):
    model = DecisionTreeRegressor(random_state= 0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/DecisionTreeRegressor.pkl', 'wb'))

def createRandomForestRegressor(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(random_state= 0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/RandomForestRegression.pkl', 'wb'))

def createElasticNet(X_train, y_train, X_test, y_test):
    model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/ElasticNet.pkl', 'wb'))

def createXGBregressor(X_train, y_train, X_test, y_test):
    model = XGBRegressor(random_state= 0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/XGBRegressor.pkl', 'wb'))

def createLogisticRegression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(class_weight= 'balanced', random_state= 0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_acc = accuracy_score(y_train, y_fit)
    test_acc = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)

    y_pred_proba = pd.Series(model.predict_proba(X_test)[:,1], index = X_test.index)

    print("Train score:", train_acc)
    print("Test score:", test_acc)
    print("Recall Test score:", test_recall)
    print("Precision Test score:", test_precision)
    print(y_pred_proba)

    dump(model, open('pkl_models/LogisticRegression.pkl', 'wb'))

def createDecisionTreeClassifier(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(class_weight= 'balanced',random_state= 0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_acc = accuracy_score(y_train, y_fit)
    test_acc = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)

    y_pred_proba = pd.Series(model.predict_proba(X_test)[:,1], index = X_test.index)

    print("Train score:", train_acc)
    print("Test score:", test_acc)
    print("Recall Test score:", test_recall)
    print("Precision Test score:", test_precision)
    print(y_pred_proba)

    # dump(model, open('pkl_models/SVM.pkl', 'wb'))
    dump(model, open('pkl_models/DecisionTreeClassifier.pkl', 'wb'))

def createRandomForestClassifier(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(class_weight= 'balanced',random_state= 0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_acc = accuracy_score(y_train, y_fit)
    test_acc = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)

    y_pred_proba = pd.Series(model.predict_proba(X_test)[:,1], index = X_test.index)

    print("Train score:", train_acc)
    print("Test score:", test_acc)
    print("Recall Test score:", test_recall)
    print("Precision Test score:", test_precision)
    print(y_pred_proba)

    dump(model, open('pkl_models/RandomForestClassifier.pkl', 'wb'))

def createSVC(X_train, y_train, X_test, y_test):
    model = SVC(kernel = 'rbf', class_weight = 'balanced',random_state= 0).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_acc = accuracy_score(y_train, y_fit)
    test_acc = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)

    y_pred_proba = pd.Series(model.predict_proba(X_test)[:,1], index = X_test.index)

    print("Train score:", train_acc)
    print("Test score:", test_acc)
    print("Recall Test score:", test_recall)
    print("Precision Test score:", test_precision)
    print(y_pred_proba)

    dump(model, open('pkl_models/SVC.pkl', 'wb'))

def createSVR(X_train, y_train, X_test, y_test):
    model = SVR(kernel='rbf', C=0.5, cache_size=8000).fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    train_mse = mean_squared_error(y_train, y_fit)
    test_mse = mean_squared_error(y_test, y_pred)

    print("Train score:", train_mse)
    print("Test score:", test_mse)

    dump(model, open('pkl_models/SVR.pkl', 'wb'))

def createKNN(X_train, y_train, X_test, y_test):
    param_nn = {'n_neighbors': [1, 3, 5, 7, 9]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_nn, cv=5)
    grid_search.fit(X_train, y_train)

    best_param_nn = grid_search.best_params_['n_neighbors']
    final_knn_model = KNeighborsClassifier(n_neighbors=best_param_nn)
    final_knn_model.fit(X_train, y_train)
    
    y_fit = pd.Series(final_knn_model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(final_knn_model.predict(X_test), index = X_test.index)

    train_acc = accuracy_score(y_train, y_fit)
    test_acc = accuracy_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)

    y_pred_proba = pd.Series(final_knn_model.predict_proba(X_test)[:,1], index = X_test.index)

    print("Train score:", train_acc)
    print("Test score:", test_acc)
    print("Recall Test score:", test_recall)
    print("Precision Test score:", test_precision)
    print(y_pred_proba)

    dump(final_knn_model, open('pkl_models/KNN.pkl', 'wb'))
    
def createAndSaveScaler(data):
    lat_scaler = StandardScaler().fit(data[['LAT_WGS84']])
    long_scaler = StandardScaler().fit(data[['LONG_WGS84']])
    crime_count_scaler = StandardScaler().fit(data[['crime_count']])

    data['LAT_WGS84'] = lat_scaler.transform(data[['LAT_WGS84']])
    data['LONG_WGS84'] = long_scaler.transform(data[['LONG_WGS84']])
    data['crime_count'] = crime_count_scaler.transform(data[['crime_count']])

    dump(lat_scaler, open('pkl_models/lat_scaler.pkl', 'wb'))
    dump(long_scaler, open('pkl_models/long_scaler.pkl', 'wb'))
    dump(crime_count_scaler, open('pkl_models/crime_count_scaler.pkl', 'wb'))
    return data

def loadScaler(lat_scaler_path, long_scaler_path, crime_count_scaler_path):
    return pickle.load(open(lat_scaler_path, 'rb')), pickle.load(open(long_scaler_path, 'rb')), pickle.load(open(crime_count_scaler_path, 'rb'))

def loadModel(model_path):
    return pickle.load(open('pkl_models/' + model_path, 'rb'))