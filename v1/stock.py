import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv('data.csv')

# select the columns needed for the model
data = df[['Close']].values

# normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# set the number of time steps
timesteps = 60

# create the input and output arrays
X, Y = [], []
for i in range(timesteps, len(data)):
    X.append(data[i-timesteps:i, 0])
    Y.append(data[i, 0])
X, Y = np.array(X), np.array(Y)

# reshape the input array to a 3D array
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# create the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# set the number of epochs
epochs = 100

# train the model
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=32, validation_data=(X_test, Y_test))

# make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# plot the results
plt.plot(Y_test, color='blue')
plt.plot(predictions, color='orange')
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['Actual', 'Predicted'])
plt.show()

# save the refined data to a csv file
refined_data = pd.DataFrame(scaler.inverse_transform(data), columns=['Close'])
refined_data.to_csv('data-refined.csv', index=False)
