import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import sklearn.model_selection 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# Data Cleaning
df = pd.read_csv('../acm/max_planck_weather_ts.csv')

print(df.head())
print(df.info())
print(df.shape)
print(df.describe())

print("\nMissing valyes by col umn:\n")
print(df.isna().any().to_frame(name="missing?"))

print(df.tail)

#checking for dupes and dropping them
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"\nRemoved {before - after} duplicate rows.")

#setting the target variable
target = df[['T (degC)']]
print("\nTarget variable (T in °C):")
print(target.head())

#extracting the target variables: temp and humidity
target_variables = df[['T (degC)', 'rh (%)']]

print("\nTarget variables (Temperature and Humidity):")
print(target_variables.head())

# Clean column names for better readability
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
    .str.replace('%', 'percent', regex=False)
)

df['date_time'] = pd.to_datetime(df['date_time'], format='%d.%m.%Y %H:%M:%S')

df.dropna(inplace=True)

date_col = df['date_time']
numeric_df = df.drop(columns=['date_time'])

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_df)
normalized_df = pd.DataFrame(normalized_data, columns=numeric_df.columns)

normalized_df['date_time'] = date_col

normalized_df.set_index('date_time', inplace=True)

target_variables = normalized_df[['t_degc', 'rh_percent']]
print("\nNormalized target variables (Temperature & Humidity):")
print(target_variables.head(10))

#reset index to access "date_time" again
yearly_df = normalized_df.reset_index()
yearly_df['year'] = yearly_df['date_time'].dt.year

yearly_avg = yearly_df.groupby('year')[['t_degc', 'rh_percent']].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_avg.index, yearly_avg['t_degc'], marker='o', label='Avg Temp (°C)', color='orange')
plt.plot(yearly_avg.index, yearly_avg['rh_percent'], marker='o', label='Avg RH (%)', color='blue')

plt.title('Yearly Average Temperature and Humidity')
plt.xlabel('Year')
plt.ylabel('Normalized Average Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#do a boxplot of the distribution of temp for celc
plt.figure(figsize=(8, 4))
sns.boxplot(x=normalized_df['t_degc'])
plt.title('Box Plot of Temperature (°C)')
plt.xlabel('t_degc')
plt.grid(True)
plt.tight_layout()
plt.show()

# Most temperatures fall between ~0.4 and ~0.7 (normalized scale).
# There are some cold outliers below 0.2 and hot outliers above 0.9.
# The data is fairly symmetric, with a centered median around 0.6.
# This means temperatures are mostly stable, with rare extreme values.
# Outliers could be unusual weather events or sensor errors worth checking.

#normalization
plt.figure(figsize=(8, 4))
sns.histplot(data=normalized_df, x='t_degc', kde=True, bins=50)
plt.title("Distribution of Normalized Temperature")
plt.xlabel("t_degc")
plt.tight_layout()
plt.show()

# Boxplot for normalized humidity
plt.figure(figsize=(8, 4))
sns.boxplot(x=normalized_df['rh_percent'])
plt.title("Boxplot of Normalized Humidity")
plt.xlabel("rh_percent")
plt.tight_layout()
plt.show()

#histogram Results:
# Most temperatures fall between ~0.4 and ~0.7 (normalized).
# The shape is symmetric and bell-shaped, peaking near 0.55–0.6.
# Very few temperature values exist below 0.2 or above 0.9.
# This shows temperature data is stable with few extremes.
# Matches the skewness value (~0), indicating no major skew.


#Second Box plot results:
# Most humidity values fall between ~0.6 and ~0.9 (normalized).
# The median is around 0.75, showing high humidity is common.
# There are many outliers below 0.2, indicating rare low-humidity events.
# The data is slightly left-skewed, with more extreme low values.
# Overall, humidity tends to stay high, with few dry conditions.

# Skewness of temperature and humidity
print("Skewness of Temp:", normalized_df['t_degc'].skew())
print("Skewness of Humidity:", normalized_df['rh_percent'].skew())


# Skewness of Temp: -0.0186 → nearly symmetric
# Skewness of Humidity: -0.6725 → moderate left skew
# The temperature distribution is well balanced
# Humidity has more extreme low values than high ones
# Both metrics align well with their visualizations


# Using iterrows()
for index, row in normalized_df.iterrows():
    print(f"Row {index}: Temp = {row['t_degc']}, Humidity = {row['rh_percent']}")
    break  # Only show the first row

# Using itertuples() (faster than iterrows)
for row in normalized_df.itertuples(index=True):
    print(f"Temp = {row.t_degc}, Humidity = {row.rh_percent}")
    break  # Only show the first row


#Time Series
# Plot 1: Original time series
plt.figure(figsize=(14, 6))
plt.plot(target_variables)
plt.title('Original Time Series: Temperature and Humidity')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.legend(target_variables.columns)
plt.tight_layout()
plt.show()

# Resampling to daily, weekly, and monthly averages
target_daily = target_variables.resample('D').mean()
target_weekly = target_variables.resample('W').mean()
target_monthly = target_variables.resample('ME').mean()  # Month end

# Plot 2: Daily resampled
plt.figure(figsize=(14, 6))
plt.plot(target_daily)
plt.title('Daily Resampled Time Series')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.legend(target_variables.columns)
plt.tight_layout()
plt.show()

# Plot 3: Weekly resampled
plt.figure(figsize=(14, 6))
plt.plot(target_weekly)
plt.title('Weekly Resampled Time Series')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.legend(target_variables.columns)
plt.tight_layout()
plt.show()

# Plot 4: Monthly resampled
plt.figure(figsize=(14, 6))
plt.plot(target_monthly)
plt.title('Monthly Resampled Time Series')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.grid(True)
plt.legend(target_variables.columns)
plt.tight_layout()
plt.show()


#boxplot and histogram 
# Boxplot of temperature (normalized version from earlier)
plt.figure(figsize=(8, 4))
sns.boxplot(x=normalized_df['t_degc'])
plt.title("Boxplot of Temperature")
plt.xlabel("t_degc")
plt.tight_layout()
plt.show()

# Histogram to inspect the distribution
plt.figure(figsize=(8, 4))
normalized_df[['t_degc']].hist(bins=30)
plt.suptitle("Histogram of Normalized Temperature")
plt.tight_layout()
plt.show()

# StandardScaler (if needed)
scaler = StandardScaler()
normalized_df['temperature_scaled'] = scaler.fit_transform(normalized_df[['t_degc']])

# Optionally preview the scaled column
print("\nFirst 5 standardized temperature values:")
print(normalized_df[['temperature_scaled']].head())


print(df.columns.tolist())

#linear regression model
FEATURES = ['t_degc', 'wv_m/s', 'tdew_degc']
TARGET = 'rh_percent'
# -------------------------
# Linear Regression Model
# -------------------------

# Make sure the features exist
available_cols = df.columns.tolist()
missing_cols = [col for col in FEATURES + [TARGET] if col not in available_cols]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# Prepare input and target arrays
X = df[FEATURES].values
y = df[[TARGET]].values

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== Linear Regression Results =====")
print(f"MAE:   {mae:.4f}")
print(f"MSE:   {mse:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²:    {r2:.4f}")

# Optional: plot predictions vs actual
plt.figure(figsize=(12, 5))
plt.plot(y_test[:100], label="Actual", alpha=0.7)
plt.plot(y_pred[:100], label="Predicted", alpha=0.7)
plt.title("Linear Regression Prediction (First 100 Points)")
plt.xlabel("Sample Index")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(df.columns.tolist())


#random forres regressor model upgrade

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.ravel())  # Flatten target array

# Predict on test set
y_rf_pred = rf_model.predict(X_test)

# Evaluate
mae_rf = mean_absolute_error(y_test, y_rf_pred)
mse_rf = mean_squared_error(y_test, y_rf_pred)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_rf_pred)

print("\n===== Random Forest Regressor Results =====")
print(f"MAE:   {mae_rf:.4f}")
print(f"MSE:   {mse_rf:.4f}")
print(f"RMSE:  {rmse_rf:.4f}")
print(f"R²:    {r2_rf:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(12, 5))
plt.plot(y_test[:100], label="Actual", alpha=0.7)
plt.plot(y_rf_pred[:100], label="RF Predicted", alpha=0.7)
plt.title("Random Forest Prediction (First 100 Points)")
plt.xlabel("Sample Index")
plt.ylabel("Humidity (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(X_train.shape)
print(X_test.shape)
print(y_pred[:5])
print(y_test[:5])

##model evaluation
scores = cross_val_score(rf_model, X, y.ravel(), cv=5, scoring='r2')
print("Cross-validated R² scores:", scores)
print("Mean R²:", scores.mean())

import joblib

# Save your trained model
joblib.dump(rf_model, 'rf_model.pkl')

# (Optional) Save your scaler too if you're using one
# joblib.dump(scaler, 'scaler.pkl')
