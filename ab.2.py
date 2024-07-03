import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the prediction model
model = load_model(r'C:\Users\me\Prediction.keras')

st.header('Stock Market Predictor')

# Input stock symbol
stock = st.text_input('Enter Stock Symbol', 'AAPL')
start = '2016-01-01'
end = '2024-07-01'

# Download stock data
data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Split data into training and test sets
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

# Prepare test data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'black', label='MA50')
plt.plot(data.Close, 'grey', label='Close')
plt.legend()
plt.show()
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'black', label='MA50')
plt.plot(ma_100_days, 'grey', label='MA100')
plt.plot(data.Close, 'orange', label='Close')
plt.legend()
plt.show()
st.pyplot(fig2)

# Plot Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'black', label='MA100')
plt.plot(ma_200_days, 'grey', label='MA200')
plt.plot(data.Close, 'orange', label='Close')
plt.legend()
plt.show()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Predict
predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(16,8))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

# Calculate Supertrend
def calculate_supertrend(data, period, multiplier):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr = pd.DataFrame(index=data.index)
    tr['tr0'] = abs(high - low)
    tr['tr1'] = abs(high - close.shift(1))
    tr['tr2'] = abs(low - close.shift(1))
    tr['TR'] = tr[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr['TR'].rolling(period).mean()
    
    hl2 = (high + low) / 2
    upperband = hl2 - (multiplier * atr)
    lowerband = hl2 + (multiplier * atr)
    
    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()
    
    supertrend = [True] * len(data)
    
    for i in range(1, len(data.index)):
        if close.iloc[i] > final_upperband.iloc[i-1]:
            supertrend[i] = True
        elif close.iloc[i] < final_lowerband.iloc[i-1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i-1]
            if supertrend[i] and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            if not supertrend[i] and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]
                
        if supertrend[i]:
            final_upperband.iloc[i] = np.nan
        else:
            final_lowerband.iloc[i] = np.nan
            
    return pd.Series(supertrend, index=data.index), final_upperband, final_lowerband

# Calculate Supertrend with given parameters
period = 10
multiplier = 3.0
supertrend, final_upperband, final_lowerband = calculate_supertrend(data, period, multiplier)

# Add Supertrend to data
data['Supertrend'] = supertrend
data['Final Upperband'] = final_upperband
data['Final Lowerband'] = final_lowerband

# Plot Supertrend Indicator
st.subheader('Supertrend Indicator')
fig5 = plt.figure(figsize=(16,8))
plt.plot(data['Close'], 'yellow', label='Close Price')
plt.plot(data['Final Upperband'], 'r', label='Final Upperband')
plt.plot(data['Final Lowerband'], 'g', label='Final Lowerband')
plt.fill_between(data.index, data['Final Upperband'], data['Final Lowerband'], where=data['Supertrend'], color='green', alpha=0.1, label='Uptrend')
plt.fill_between(data.index, data['Final Upperband'], data['Final Lowerband'], where=~data['Supertrend'], color='red', alpha=0.1, label='Downtrend')
plt.legend()
plt.show()
st.pyplot(fig5)
