from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

from pickle import dump

def createLinearRegression(data):
    X_train, y_train, X_test, y_test = data[data.index <= '2023-06-30'].drop('crime_count', axis = 1), data[data.index <= '2023-06-30']['crime_count'], data[data.index > '2023-06-30'].drop('crime_count', axis = 1), data[data.index > '2023-06-30']['crime_count']
    model = LinearRegression().fit(X_train, y_train)

    y_fit = pd.Series(model.predict(X_train), index = X_train.index)
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    ax = y_fit.plot()
    ax = y_pred.plot(ax = ax)
    plt.show()


def createAndSaveScaler(data):
    lat_scaler = StandardScaler().fit(data[['LAT_WGS84']])
    long_scaler = StandardScaler().fit(data[['LONG_WGS84']])

    data['LAT_WGS84'] = lat_scaler.transform(data[['LAT_WGS84']])
    data['LONG_WGS84'] = long_scaler.transform(data[['LONG_WGS84']])

    dump(lat_scaler, open('lat_scaler.pkl', 'wb'))
    dump(long_scaler, open('long_scaler.pkl', 'wb'))
    return data