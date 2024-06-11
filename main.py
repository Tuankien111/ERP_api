from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Preprocessing data lib
import requests
from bs4 import BeautifulSoup
from io import StringIO
import datetime

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Declare variable
scaler = MinMaxScaler(feature_range=(0, 1))

# PAGES=============================================
@app.get("/")
async def root():
    return { "message": "OK"}

@app.get("/data")
def handleProcessingData(currency: str):    
    df = getData(currency)
    # Plot chart
    plot_path = "./chart.png"
    plot_and_save(df["Close"], plot_path, currency)
    return df

def plot_and_save(data, plot_path, currency):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Close")
    plt.title(f'Close price of {currency} from 01-01-2020 ')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

@app.get("/chart")
def get_plot_image():
    plot_path = "./plotUSDVND.png"
    return StreamingResponse(open(plot_path, "rb"), media_type="image/png")

@app.get("/predict")
def make_predict(model: str, days: int, currency: str):
    data = getData(currency)

    if data is None:
        return "No data"
    
    if model == "Linear Regression":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        # linear = LinearRegression()
        # linear.fit(data_X_scaled, data_y_scaled)
        linear = pickle.load(open('./save_model/linear_model.h5','rb'))
        return predictRateInTheFuture("Linear Regression", linear, data, future_days=days)

    elif model == "Decision Tree Regression":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        decisionTree = DecisionTreeRegressor()
        decisionTree.fit(data_X_scaled, data_y_scaled)
        return predictRateInTheFuture("Decision Tree Regression", decisionTree, data, future_days=days)
    
    elif model == "Random Forest Regression":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        randomForest =RandomForestRegressor()
        randomForest.fit(data_X_scaled, data_y_scaled)
        return predictRateInTheFuture("Random Forest Regression", randomForest, data, future_days=days)
    
    elif model == "Stacking Regression":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        # Define base models
        base_models = [
            ('linear', LinearRegression()),
            ('decision_tree', DecisionTreeRegressor()),
            ('random_forest', RandomForestRegressor())
        ]
        # Define meta model
        meta_model = LinearRegression()
        # Initialize StackingRegressor
        stacking_reg = StackingRegressor(estimators=base_models, final_estimator=meta_model)
        stacking_reg.fit(data_X_scaled, data_y_scaled)
        return predictRateInTheFuture("Stacking Regression", stacking_reg, data, future_days=days)
    
    elif model == "LSTM":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        X_real, y_real = createSequentialData(data_y_scaled)
        model_name = "LSTM"
        lstm = Sequential()
        lstm.add(LSTM(50, return_sequences=True, input_shape=(X_real.shape[1], 1)))
        lstm.add(LSTM(50))
        lstm.add(Dense(1))
        lstm.compile(optimizer='adam', loss='mean_squared_error')
        lstm.fit(X_real, y_real, epochs=25, batch_size=64)
        # lstm = pickle.load(open('./save_model/lstm_model.h5','rb'))
        return predictDLInTheFuture(model_name, lstm, data, future_days=days)
    return "Model Not found"

# Function definition ===============================
def predictRateInTheFuture(model_name, model,data, lookback = 30, future_days = 1):
    data_X_scaled, data_y_scaled = preprocess_data(data)
    last_input_data = data_X_scaled[-lookback:]
    now = datetime.datetime(2024, 5, 31)
    print(data_X_scaled.shape)
    predicted_values = []
    forecasting_dates = []

    for day in range(future_days):
        next_date = now + datetime.timedelta(days=day+1)
        forecasting_dates.append(next_date.strftime('%Y-%m-%d'))

        prediction = model.predict(last_input_data)
        prediction = scaler.inverse_transform(prediction.reshape(-1,1))
        # print(prediction)
        predicted_values.append(round(float(prediction.flatten()[0]),2))
        last_input_data = np.roll(last_input_data, -1, axis=0)
    predictions = pd.DataFrame(list(zip(forecasting_dates, predicted_values)), columns=['Date','Predicted'])
    print("Predict exchange rate with",model_name, f'in {future_days} days')
    print(predictions)
    return predictions

def predictDLInTheFuture(model_name, model, data, future_days = 1):
    data_X_scaled, data_y_scaled = preprocess_data(data)
    X_real, y_real = createSequentialData(data_y_scaled)

    last_input_data = X_real[-1:]
    now = datetime.datetime(2024, 5, 31)
    # Arrays to store predicted values and dates
    predicted_values = []
    forecasting_dates = []

    for day in range(future_days):
        next_date = now + datetime.timedelta(days=day+1)
        forecasting_dates.append(next_date.strftime('%Y-%m-%d'))

        predicted_price_scaled = model.predict(last_input_data)

        predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))
        predicted_values.append(round(float(predicted_price[0, 0]),2))
        last_input_data = np.roll(last_input_data, -1, axis=0)
        predicted_price_scaled = predicted_price_scaled[:, :, np.newaxis]
        last_input_data = np.concatenate([last_input_data[:, 1:, :], predicted_price_scaled], axis=1)

    predictions = pd.DataFrame(list(zip(forecasting_dates, predicted_values)), columns=['Date','Predict Close'])

    print("Predict exchange rate with",model_name, f'in {future_days} days')
    print(predictions)
    return predictions

def preprocess_data(data):
    features = ['Open', 'High', 'Low']
    target = ['Close']

    data_X_scaled = scaler.fit_transform(data[features])
    data_y_scaled = scaler.fit_transform(data[target])

    return data_X_scaled, data_y_scaled

def createSequentialData(data, window_size=14):
    X,y = [],[]
    for i in range(len(data)-window_size):
        X.append(data[i:(i+window_size)])
        y.append(data[(i+window_size)])
    return np.array(X),np.array(y)

def getData(currency):
    df = pd.read_csv('data/USDVND_data.csv')
    df = df.drop(['Adj Close'], axis=1)
    df.dropna(inplace=True)
    return df
