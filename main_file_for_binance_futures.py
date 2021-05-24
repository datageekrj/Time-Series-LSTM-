# Instructions Regarding this piece of code
# Import of required libraries.
# If you get any error, then that means
# these libraries have not been installed
# on your system. To install these, you can copy the following commands
# one by one, and paste it on your terminal. 
# pip install python-binance
# pip install pandas
# pip install tensorflow
# pip install sklearn
# pip install numpy
# pip install mysql-connector
# All the other libraries comes pre built with python.

from binance.client import Client
import datetime
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn import preprocessing
import random
import mysql.connector
from collections import deque
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use("ggplot")
warnings.filterwarnings("ignore")

### API
binance_api_key = ''     # Input your own binance API key here
binance_api_secret = ''  # Input your own binance secret key here
binance_client = Client(binance_api_key, binance_api_secret)

def get_time(thirteen_digit):
  your_dt = datetime.datetime.fromtimestamp(int(thirteen_digit)/1000)  # using the local timezone
  your_dt = your_dt.strftime("%Y-%m-%d %H:%M:%S")  # 2018-04-07 20:48:08, YMMV
  return your_dt
def get_timestamp(time):
  time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
  return int(datetime.datetime.timestamp(time) * 1000)

def make_data(range_of_data, symbol, interval):
  data_from_server = binance_client.get_historical_klines(symbol, interval, range_of_data)
  openTime = [get_time(data[0]) for data in data_from_server]
  open_val = [data[1] for data in data_from_server]
  closeTime = [get_time(data[6]) for data in data_from_server]
  close_val = [data[4] for data in data_from_server]
  high = [data[2] for data in data_from_server]
  low = [data[3] for data in data_from_server]
  volume = [data[5] for data in data_from_server]
  numTrades = [data[8] for data in data_from_server]
  quoteAssetVolume = [data[7] for data in data_from_server]
  takerBuyBaseAssetVolume = [data[9] for data in data_from_server]
  takerBuyQuoteAssetVolume = [data[10] for data in data_from_server]

  names = ["open time", "open", "close time", "close", "high", "low", "volume", "num of trades", "quote asset volume",
          "taker buy base asset volumne", "taker buy quote asset volume"]
  datas = [openTime, open_val, closeTime, close_val, high, low,volume, numTrades, quoteAssetVolume,takerBuyBaseAssetVolume,
          takerBuyQuoteAssetVolume]
  required_data = pd.DataFrame({ names[i]:datas[i] for i in range(len(datas))})
  required_data["open"] = required_data["open"].apply(lambda x: float(x))
  required_data["close"] = required_data["close"].apply(lambda x: float(x))
  required_data["high"] = required_data["high"].apply(lambda x: float(x))
  required_data["low"] = required_data["low"].apply(lambda x: float(x))
  required_data["volume"] = required_data["volume"].apply(lambda x: float(x))
  required_data["close time"]  = pd.to_datetime(required_data["open time"])
  required_data.set_index("close time", inplace=True)
  return required_data

binance_instruments = ["BTC", "ETH", "BCH", "XRP",
                        "EOS", "LTC", "TRX", "ETC",
                        "LINK", "XLM", "ADA", "XMR",
                        "DASH", "ZEC", "XTZ", "BNB",
                        "ATOM", "ONT", "IOTA", "BAT",
                        "VET", "NEO", "QTUM", "IOST"]

# Step. 1 Getting the past 7 days of data for a corresponding instrument.
# Step. 2 Ploting the full data.
# Step. 3 Scaling the data.
# Step. 4 After scalling, getting the train-test split.
# Step. 5 Make training 60 time stamps data.
# Step. 6 Make testing 60 time stamps data.
# Step. 7 Build a LSTM model with the training data with 60 time stamps.
# Step. 8 getting the predictions from the model on the test data. (Forward Testing) 
# Step. 9 Inverse Transform the predictions to bring them in same scale.
# Step. 10 Plot actual, predicted and all the data together on one plot
# Step. 11 Finally, plot on latest (last) 60 timestamps, to get the LONG or SHORT trade for today
# Step.12 Save the nature of trade in a empty list. 
# Step. 13 Update the database

def get_the_data(symbol, interval_in_minutes = "1m",how_long = "7 day ago"):
  return make_data(how_long , symbol, interval_in_minutes)

def scale_the_data(data):
  '''data with only close column'''
  data = data[["close"]]
  dates = data.index

  scalar = MinMaxScaler()
  transformed_close = scalar.fit_transform(data)

  return [transformed_close, scalar, dates]

def make_split(close, dates):
  training_data_len = int(len(close) * 0.8)
  training_data = close[:training_data_len]
  training_dates  = dates[:training_data_len]
  testing_data = close[training_data_len:]
  testing_dates = dates[training_data_len:]
  return training_data,training_dates,testing_data,testing_dates

def get_60_time_stamps_data(data, dates):
  X = []
  y = []
  required_dates = []
  for i in range(60,len(data)):
    X.append(data[i-60:i])
    y.append(data[i])
    required_dates.append(dates[i])
  X = np.array(X)
  y = np.array(y)
  X = X.reshape((X.shape[0],X.shape[1],1))
  return X, y, required_dates

def make_a_model(training_input):
  model = tf.keras.Sequential()
  model.add(layers.LSTM(400, return_sequences=True, input_shape = (training_input.shape[1],1)))
  model.add(layers.LSTM(300 , return_sequences=False))
  model.add(layers.Dense(200))
  model.add(layers.Dense(1))
  model.compile(optimizer="adam", loss = "mean_squared_error")
  return model

def fit_the_model(input, output, model):
  '''input should be 3 dimensional for number of
     rows, number of timestamps and no of cols'''
  model.fit(input, output, epochs = 1, batch_size = 100)
  return model

def predict_from_model(model, test, scalar):
  predictions = model.predict(test)
  transformed_predictions = scalar.inverse_transform(predictions)
  return transformed_predictions

def final_forward_test_plot(train_dates, train_data, test_dates, test_data, predictions, symbol):
  plt.figure(figsize = (20,8))
  plt.title("Actual vs Predicted " + symbol + " prices..")
  plt.plot(train_dates, train_data, label = "Data Neural Network Trained on..")
  plt.plot(test_dates,predictions, label = "Predictions", color = "green")
  plt.plot(test_dates,test_data, label = "Actual Values")
  plt.legend(loc = "best")
  plt.xlabel("Time")
  plt.ylabel("Closing Price")
  plt.show()

def go_short_or_long(model, data, scalar,the_latest_price):
  '''This data should be the transformed one, scaled one'''
  latest_60_timestamps = data[-60:].reshape((1,60,1))
  predictions = model.predict(latest_60_timestamps)
  transformed_predictions = scalar.inverse_transform(predictions)[0]
  if transformed_predictions > the_latest_price:
    return ["LONG",transformed_predictions, the_latest_price]
  else:
    return ["SHORT",transformed_predictions, the_latest_price]

def final_run(symbol):
  print("Getting the data...wait", "Getting the data for..", symbol)
  data = get_the_data(symbol, "1m", "7 day ago")
  plt.figure(figsize = (20,8))
  plt.plot(data["close"])
  plt.title("Close price for the last 7 days....")
  close,scalar,dates = scale_the_data(data)
  training_data,training_dates,testing_data,testing_dates = make_split(close, dates)

  train_x, train_y, train_dates = get_60_time_stamps_data(training_data, training_dates)

  test_x, test_y, test_dates = get_60_time_stamps_data(testing_data, testing_dates)

  model = make_a_model(train_x)
  print("Training the Neural network on the data..", symbol, " price data...")
  model = fit_the_model(train_x, train_y, model)

  predictions = predict_from_model(model, test_x, scalar)

  train_y = scalar.inverse_transform(train_y)
  test_y = scalar.inverse_transform(test_y)
  final_forward_test_plot(train_dates, train_y, test_dates, test_y, predictions, symbol)
  final_result_trade = go_short_or_long(model,close , scalar, test_y[-1])

  entrytime = data.iloc[-2:-1,:].index[0]
  if entrytime.minute + 1 > 59:
    exittime  = entrytime.replace(minute = 1, hour = entrytime.hour + 1)
  else:
    exittime  = entrytime.replace(minute = entrytime.minute + 1)
  return (symbol,final_result_trade[0],str(final_result_trade[2][0]) ,str(entrytime),str(final_result_trade[1][0]), str(exittime))

def get_db_connection():
  db = mysql.connector.connect(host = "95.216.224.96",
                            user = "xchangev2",
                             password = "Deneme321!",
                             database = "xchangev2")
  
  cursor = db.cursor()
  return [cursor,db]

def update_all_data():
  for symbol in binance_instruments:
    print("Getting things done for symbol: ", symbol)
    current_data = final_run(symbol + "USDT")
    cursor,db = get_db_connection()
    sql_query = 'INSERT INTO binance_data (Instrument, Type, EntryPrice, Entrytime, ExitPrice, Exittime) VALUES (%s,%s,%s,%s,%s,%s)'
    cursor.execute(sql_query, current_data)
    db.commit()
    cursor.close()
    db.close()
    print("Done saving the data for symbol:  ", symbol)
    print("Closing the database connection..")
  print("Done Update of Data....")

update_all_data()
