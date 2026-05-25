# Ex.No: 6               HOLT WINTERS METHOD
### Date: 25.05.2026



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# -------------------- LOAD DATA -------------------- #
data = pd.read_csv('/content/sales.csv')

# -------------------- DISPLAY DATA -------------------- #
print("Columns in Dataset:\n")
print(data.columns)

print("\nFirst 5 Rows:\n")
print(data.head())

# -------------------- SELECT COLUMN -------------------- #
# Change this column if needed
column = 'revenue'

# -------------------- DATA CLEANING -------------------- #

# Convert date column to datetime
data['order_date'] = pd.to_datetime(data['order_date'])

# Convert selected column to numeric
data[column] = pd.to_numeric(data[column], errors='coerce')

# Remove missing values
data = data.dropna(subset=[column])

# -------------------- SET DATE INDEX -------------------- #
data.set_index('order_date', inplace=True)

# -------------------- MONTHLY RESAMPLING -------------------- #
data_monthly = data[column].resample('MS').sum()

print("\nMonthly Data:\n")
print(data_monthly.head())

# -------------------- PLOT ORIGINAL DATA -------------------- #
plt.figure(figsize=(12,5))

plt.plot(data_monthly)

plt.title('Monthly Revenue Data')
plt.xlabel('Date')
plt.ylabel('Revenue')

plt.grid(True)

plt.show()

# -------------------- SCALE DATA -------------------- #
scaler = MinMaxScaler()

scaled_values = scaler.fit_transform(
    data_monthly.values.reshape(-1,1)
).flatten()

scaled_data = pd.Series(
    scaled_values,
    index=data_monthly.index
)

# Make all values positive
scaled_data = scaled_data + 1

# -------------------- PLOT SCALED DATA -------------------- #
plt.figure(figsize=(12,5))

plt.plot(scaled_data)

plt.title('Scaled Revenue Data')

plt.grid(True)

plt.show()

# -------------------- SEASONAL DECOMPOSITION -------------------- #
# Use smaller period to avoid errors
decomposition = seasonal_decompose(
    scaled_data,
    model='additive',
    period=4
)

decomposition.plot()

plt.show()

# -------------------- TRAIN TEST SPLIT -------------------- #
train_size = int(len(scaled_data) * 0.8)

train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

print("\nTrain Size:", len(train_data))
print("Test Size:", len(test_data))

# -------------------- HOLT-WINTERS MODEL -------------------- #
# Smaller seasonal period avoids initialization errors

seasonal_period = 4

model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=seasonal_period
).fit()

# -------------------- FORECAST TEST DATA -------------------- #
test_predictions = model.forecast(
    steps=len(test_data)
)

# -------------------- VISUAL EVALUATION -------------------- #
plt.figure(figsize=(12,5))

plt.plot(train_data, label='Train Data')

plt.plot(test_data, label='Test Data')

plt.plot(test_predictions, label='Predictions')

plt.legend()

plt.title('Holt-Winters Prediction')

plt.xlabel('Date')
plt.ylabel('Scaled Revenue')

plt.grid(True)

plt.show()

# -------------------- RMSE -------------------- #
rmse = np.sqrt(
    mean_squared_error(test_data, test_predictions)
)

print("\nRMSE Value:", rmse)

# -------------------- FINAL MODEL -------------------- #
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='add',
    seasonal_periods=seasonal_period
).fit()

# -------------------- FUTURE FORECAST -------------------- #
future_steps = 12

future_predictions = final_model.forecast(
    steps=future_steps
)

print("\nFuture Forecast Values:\n")
print(future_predictions)

# -------------------- FORECAST PLOT -------------------- #
plt.figure(figsize=(12,5))

plt.plot(scaled_data, label='Original Data')

plt.plot(future_predictions, label='Future Forecast')

plt.legend()

plt.title('Future Forecast using Holt-Winters')

plt.xlabel('Date')
plt.ylabel('Scaled Revenue')

plt.grid(True)

plt.show()
```
### OUTPUT:
<img width="545" height="387" alt="image" src="https://github.com/user-attachments/assets/b8564b53-5d63-4ab4-a839-18984e5345b8" />

<img width="780" height="319" alt="image" src="https://github.com/user-attachments/assets/5a32a962-2dae-4e42-b7c3-e03e8406bc72" />

<img width="680" height="308" alt="image" src="https://github.com/user-attachments/assets/b3cf4567-542f-43b6-a8aa-d08f4507e5b1" />

<img width="484" height="359" alt="image" src="https://github.com/user-attachments/assets/413c9eaa-3673-434a-861f-dbe48b6b75ef" />


TEST_PREDICTION
<img width="752" height="323" alt="image" src="https://github.com/user-attachments/assets/eb7c5e11-2eaa-4f01-999e-e62ded674992" />



FINAL_PREDICTION
<img width="685" height="310" alt="image" src="https://github.com/user-attachments/assets/226c5dca-c45c-4e0a-9eaa-53a296619514" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
