import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model import createLinearRegression, createAndSaveScaler
from pickle import dump

def clean_csv_data(filename):
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
   return a set of unique Latitude and Longitude from DataFrame
   '''
   lat_long_set = set()
   for row in df.itertuples():
      lat_long_set.add((row[1], row[2]))
   return lat_long_set


if __name__ == '__main__':
   # clean_csv_data('Major_Crime_Indicators_Open_Data.csv')
   # df = pd.read_csv('data_clean.csv')
   # df = df[df.OCC_DATE >= '2022-01-01']

   # df = prepare_data(df)
   # df.to_csv('processed_data.csv')

   data = pd.read_csv('processed_data.csv', index_col= 'date', parse_dates= True)
   data = data.to_period('H')

   data = createAndSaveScaler(data)
   createLinearRegression(data)
