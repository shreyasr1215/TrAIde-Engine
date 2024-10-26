import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def fetch_stock_data(ticker):
    data = yf.download(tickers=ticker, period='1d', interval='5m')
    return data

def train_lstm(data):
    features = ['Open', 'High', 'Low', 'Volume']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    X = []
    y = []

    for i in range(5, len(scaled_data)):
        X.append(scaled_data[i-5:i])
        y.append(data['Adj Close'].values[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=5, batch_size=8, verbose=1)

    return model, scaler

def predict_and_mark(data, model, scaler):
    features = ['Open', 'High', 'Low', 'Volume']
    markers = []

    # Scale the data using the existing scaler
    scaled_data = scaler.transform(data[features])

    # Loop over the data in 5-minute intervals but only store markers every hour
    factor = []
    
    for i in range(5, len(scaled_data)):
        input_data = scaled_data[i - 5:i]
        input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))

        predicted_price = model.predict(input_data)[0, 0]
        actual_price = data['Adj Close'].values[i]
        factor.append(actual_price/predicted_price)
    factor = sum(factor)/len(factor)
    for i in range(5, len(scaled_data)):
        if data.index[i].minute == 0 or data.index[i].minute == 30:  # Check if it's the start of an hour
            input_data = scaled_data[i - 5:i]
            input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))

            predicted_price = model.predict(input_data)[0, 0]
            actual_price = data['Adj Close'].values[i]
            print(actual_price/predicted_price)

            # Generate buy/sell marker
            if predicted_price*factor > actual_price:
                markers.append(('buy', data.index[i], actual_price))  # Buy signal
            else:
                markers.append(('sell', data.index[i], actual_price))  # Sell signal

    print("Markers:", markers)
    return markers