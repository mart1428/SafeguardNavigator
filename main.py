import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from pickle import dump

from model import createLinearRegression, createAndSaveScaler, loadScaler, createDecisionTree, loadModel, createRandomForest, createXGBregressor

def clean_csv_data(filename):
   '''
   (str) -> None
   Save cleaned file to csv. 
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
   (List(str)), (pandas.DataFrame)
   '''

   modified_df = X.copy()

   for m in models:
      model = loadModel(m)

      y_pred = pd.Series(model.predict(X), index = X.index)

      modified_df[m] = y_pred
   
   return modified_df

if __name__ == '__main__':
   # clean_csv_data('Major_Crime_Indicators_Open_Data.csv')
   # df = pd.read_csv('data_clean.csv')
   # df = df[df.OCC_DATE >= '2022-01-01']

   # df = prepare_data(df)
   # df.to_csv('processed_data.csv')

   data = pd.read_csv('processed_data.csv', index_col= 'date', parse_dates= True)
   data = data.to_period('H')

   # data = createAndSaveScaler(data)
   lat_scaler, long_scaler = loadScaler('pkl_models/lat_scaler.pkl', 'pkl_models/long_scaler.pkl')
   data['LAT_WGS84'] = lat_scaler.transform(data[['LAT_WGS84']])
   data['LONG_WGS84'] = long_scaler.transform(data[['LONG_WGS84']])

   data_train = data[data.index <= '2023-06-30']
   data_test = data[data.index > '2023-06-30']

   # cluster = KMeans().fit(data_train.drop('crime_count', axis = 1))
   # dump(cluster, open('cluster.pkl', 'wb'))
   cluster = loadModel('cluster.pkl')

   data_train['cluster'] = cluster.predict(data_train.drop('crime_count', axis = 1))
   data_test['cluster'] = cluster.predict(data_test.drop('crime_count', axis = 1))


   X_train, y_train, X_test, y_test = data_train.drop('crime_count', axis = 1), data_train['crime_count'], data_test.drop('crime_count', axis = 1), data_test['crime_count']

   models = ['LinearRegression.pkl', 'DecisionTree.pkl', 'RandomForest.pkl', 'XGBRegressor.pkl']
   # print('LinReg')
   # createLinearRegression(X_train, y_train, X_test, y_test)

   # print('CART')
   # createDecisionTree(X_train, y_train, X_test, y_test)

   # print('RandomForest')
   # createRandomForest(X_train, y_train, X_test, y_test)
   
   # print('XGB')
   # createXGBregressor(X_train, y_train, X_test, y_test)

   print(get_all_models_prediction(models, X_train))
