import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

from pickle import dump
import sys

from model import createLinearRegression, createAndSaveScaler, loadScaler, loadModel, createXGBregressor, createLogisticRegression,\
createDeterministicProcessIndex, createDecisionTreeClassifier, createDecisionTreeRegressor,\
createKNN, createElasticNet, createRandomForestRegressor, createRandomForestClassifier

def clean_csv_data(filename):
   '''
   (str) -> None
   Clean inputted csv files by dropping unnecessary columns and converting dtypes to date. Save cleaned file to csv. 
   '''
   df = pd.read_csv(filename)

   drop_cols = ['X', 'Y', 'OBJECTID', 'REPORT_DATE', \
      'REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY', 'REPORT_DOY', 'REPORT_DOW',\
      'REPORT_HOUR', 'OCC_YEAR', 'OCC_MONTH', 'OCC_DAY', 'OCC_DOY', 'OCC_DOW',\
   'DIVISION', 'UCR_CODE',\
      'UCR_EXT', 'HOOD_158',\
      'HOOD_140', 'NEIGHBOURHOOD_140']

   df.drop(columns = drop_cols, inplace = True)
   df = df[(df.LONG_WGS84 != 0) & (df.LAT_WGS84 != 0)]
   date = pd.to_datetime(df['OCC_DATE']).dt.date
   df['OCC_DATE'] = date
   df.to_csv('data_clean.csv', index = False)

def prepare_data(df):
   '''
   (pandas.DataFrame) -> (pandas.DataFrame)
   Return prepared data that is ready to be processed. 
   '''
   unique_lat_long = get_lat_long_combo(df[['LAT_WGS84', 'LONG_WGS84']])
   df['date'] = pd.to_datetime(df['OCC_DATE'].astype(str) + ' ' + df['OCC_HOUR'].astype(str) + ':00')
   dateRange = pd.date_range(df.date.min(), df.date.max(), freq = 'H').to_frame(False, 'date')
   # df = df[['date', 'PREMISES_TYPE', 'MCI_CATEGORY', 'LAT_WGS84', 'LONG_WGS84']]
   df = df[['date', 'LAT_WGS84', 'LONG_WGS84', 'MCI_CATEGORY']]
   df = df.groupby(['date', 'LAT_WGS84', 'LONG_WGS84'], as_index = False).count()
   df = df.rename(columns = {'MCI_CATEGORY' : 'crime_count'})

   data_prepared = pd.DataFrame()

   c = 0
   for lat, long in unique_lat_long:
      temp_df = df[(df.LAT_WGS84 == lat) & (df.LONG_WGS84 == long)]
      temp_df = dateRange.set_index('date').join(temp_df.set_index('date'), how = 'left')
      temp_df['LAT_WGS84'] = lat
      temp_df['LONG_WGS84'] = long
      temp_df.fillna(0, inplace= True)

      if len(data_prepared) == 0:
         data_prepared = temp_df
      else:
         data_prepared = pd.concat([data_prepared, temp_df], axis = 0)

      if c == 500:        #To reduce number of data received 
         break
      c+= 1
   return data_prepared

   
def get_lat_long_combo(df):
   '''
   (pandas.DataFrame) -> set(tuple(float, float))
   Return a set of unique Latitude and Longitude from DataFrame
   '''
   lat_long_set = set()
   for row in df.itertuples():
      lat_long_set.add((row[1], row[2]))
   return lat_long_set

def get_all_models_prediction(models, X):
   '''
   (List(str)), (pandas.DataFrame) -> (pandas.DataFrame)
   Using the provided list of model file names, load the model and predict using the provided X values. 
   Return X with predictions of each model
   '''

   modified_df = X.copy()

   for m in models:
      model = loadModel(m)

      y_pred = pd.Series(model.predict(X), index = X.index)

      modified_df[m] = y_pred
   
   return modified_df

def train_pipeline(data, train_test_date_split = '2023-06-30', regressor_models = ['LinearRegression.pkl', 'DecisionTree.pkl', 'ElasticNet.pkl', 'RandomForest.pkl', 'XGBRegressor.pkl'], classifier_models = ['RandomForestClassifier.pkl', 'LogisticRegression.pkl']):
   '''
   (pandas.DataFrame), (String), list(String), list(String) -> (None)

   Pipeline to train models. 
   Note: regressor_models and classifier_models parameters are not implemented yet. 
   '''
   
   data = createAndSaveScaler(data)
   lat_scaler, long_scaler, crime_count_scaler = loadScaler('pkl_models/lat_scaler.pkl', 'pkl_models/long_scaler.pkl', 'pkl_models/crime_count_scaler.pkl')
   data['LAT_WGS84'] = lat_scaler.transform(data[['LAT_WGS84']])
   data['LONG_WGS84'] = long_scaler.transform(data[['LONG_WGS84']])
   data['crime_count'] = crime_count_scaler.transform(data[['crime_count']])

   data_train = data[data.index <= train_test_date_split]              
   data_test = data[data.index > train_test_date_split]

   cluster = KMeans(n_init = 'auto', n_clusters = 3).fit(data_train.drop('crime_count', axis = 1))
   dump(cluster, open('pkl_models/cluster.pkl', 'wb'))
   data_train['cluster'] = cluster.predict(data_train.drop('crime_count', axis = 1))
   data_test['cluster'] = cluster.predict(data_test.drop('crime_count', axis = 1))

   X_train, y_train, X_test, y_test = data_train.drop('crime_count', axis = 1), data_train['crime_count'], data_test.drop('crime_count', axis = 1), data_test['crime_count']

   print('\nLinReg')
   createLinearRegression(X_train, y_train, X_test, y_test)

   print('\nCART')
   createDecisionTreeRegressor(X_train, y_train, X_test, y_test)

   print('\nElasticNet')
   createElasticNet(X_train, y_train, X_test, y_test)

   print('\nRandomForest')
   createRandomForestRegressor(X_train, y_train, X_test, y_test)
   
   print('\nXGB')
   createXGBregressor(X_train, y_train, X_test, y_test)
   
   modified_df_train = get_all_models_prediction(regressor_models, X_train)
   modified_df_test = get_all_models_prediction(regressor_models, X_test)

   X_train_modified, X_test_modified = modified_df_train, modified_df_test

   crime_scaler = MinMaxScaler((0,1)).fit(y_train.to_frame())
   dump(crime_scaler, open('pkl_models/crime_minmax_scaler.pkl', 'wb'))
   y_train_scaled = pd.Series(crime_scaler.transform(y_train.to_frame())[:,0], index = y_train.index)
   y_test_scaled = pd.Series(crime_scaler.transform(y_test.to_frame())[:,0], index = y_test.index)

   y_train_scaled = y_train_scaled.apply(lambda x: 1 if x > 0 else 0)
   y_test_scaled = y_test_scaled.apply(lambda x: 1 if x > 0 else 0)

   print('\nLogistic Regression')
   createLogisticRegression(X_train_modified, y_train_scaled, X_test_modified, y_test_scaled )
   print('\nTree Classifier')
   createDecisionTreeClassifier(X_train_modified, y_train_scaled, X_test_modified, y_test_scaled)
   print('\nRF Classifier')
   createRandomForestClassifier(X_train_modified, y_train_scaled, X_test_modified, y_test_scaled)


def run_pipeline(data, regressor_models = ['LinearRegression.pkl', 'DecisionTree.pkl', 'ElasticNet.pkl', 'RandomForest.pkl', 'XGBRegressor.pkl'], classifier_models = ['RandomForestClassifier.pkl', 'LogisticRegression.pkl']):
   lat_scaler, long_scaler, crime_count_scaler = loadScaler('pkl_models/lat_scaler.pkl', 'pkl_models/long_scaler.pkl', 'pkl_models/crime_count_scaler.pkl')
   data['LAT_WGS84'] = lat_scaler.transform(data[['LAT_WGS84']])
   data['LONG_WGS84'] = long_scaler.transform(data[['LONG_WGS84']])
   data['crime_count'] = crime_count_scaler.transform(data[['crime_count']])

   cluster = loadModel('cluster.pkl')
   data['cluster'] = cluster.predict(data.drop('crime_count', axis = 1))

   X, y = data.drop('crime_count', axis = 1), data['crime_count']
   modified_X = get_all_models_prediction(regressor_models, X)
   
   crime_scaler = loadModel('crime_minmax_scaler.pkl')
   y_scaled = pd.Series(crime_scaler.transform(y.to_frame())[:,0], index = y.index)

   y_scaled = y_scaled.apply(lambda x: 1 if x >= 0.01 else 0)
   modified_data = get_all_models_prediction(classifier_models, modified_X)
   return modified_data[classifier_models], y_scaled

if __name__ == '__main__':
   #------------------IF Running for the first time-----------------------
   # clean_csv_data('Major_Crime_Indicators_Open_Data.csv')
   # df = pd.read_csv('data_clean.csv')
   # df = df[df.OCC_DATE >= '2022-01-01']

   # df = prepare_data(df)
   # df.to_csv('processed_data.csv')
   #====================================================================

   data = pd.read_csv('processed_data.csv', index_col= 'date', parse_dates= True)         #lat long example: 43.6384649321311  -79.4378661170172
   data = data.to_period('H')
   train_pipeline(data)
   print(run_pipeline(data))