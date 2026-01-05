import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import  PoissonRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.api import OLS, add_constant
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from sklearn.impute import SimpleImputer

# Load datasets
TTC = pd.read_csv('ttc-bus-delay-data-2022.csv')
weather = pd.read_csv('Toronto_Climate_2022.csv')

# Data Cleaning and Preprocessing
# TTC Data Cleaning
TTC = TTC.drop_duplicates(subset=['Date', 'Route', 'Time'], keep='first')
TTC = TTC.dropna(subset=['Min Gap', 'Route', 'Direction', 'Incident'])
TTC['Date'] = pd.to_datetime(TTC['Date'])
TTC['Time'] = pd.to_datetime(TTC['Time'], format='%H:%M', errors='coerce')
new_TTC = TTC[['Date', 'Min Delay', 'Time']]

# Weather Dataset Cleaning
new_weather = weather[['Date/Time', 'Total Rain (mm)', 'Snow on Grnd (cm)']]
new_weather['Date/Time'] = pd.to_datetime(new_weather['Date/Time'])
new_weather['Date'] = new_weather['Date/Time'].dt.date
new_weather['Date'] = pd.to_datetime(new_weather['Date'])
new_weather = new_weather.drop(columns=['Date/Time'])

# Impute missing values in weather data
imputer = SimpleImputer(strategy='mean')
new_weather[['Total Rain (mm)', 'Snow on Grnd (cm)']] = imputer.fit_transform(new_weather[['Total Rain (mm)', 'Snow on Grnd (cm)']])

# Merging data
merge_data = pd.merge(new_TTC, new_weather, on='Date', how='inner')

# Feature Engineering
merge_data['Month'] = merge_data['Date'].dt.month
merge_data['Hour of Day'] = merge_data['Time'].dt.hour
merge_data['Season'] = pd.cut(merge_data['Month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])

# Time of Day Categorization
def time_of_day(hour):
    if 6 <= hour < 9:
        return 'Morning Rush Hour'
    elif 9 <= hour < 15:
        return 'Daytime'
    elif 15 <= hour < 18:
        return 'Afternoon Rush Hour'
    elif 18 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
merge_data['Time of Day'] = merge_data['Hour of Day'].apply(time_of_day)

# Visualizations
# Binned Data for Snow and Rain
merge_data['Snow Category'] = pd.cut(merge_data['Snow on Grnd (cm)'], bins=[0, 0.5, 2, 5, 10, 20], labels=['Light', 'Moderate', 'Heavy', 'Very Heavy', 'Extreme'])
merge_data['Rain Category'] = pd.cut(merge_data['Total Rain (mm)'], bins=[0, 2, 5, 10, 20], labels=['Light', 'Moderate', 'Heavy', 'Extreme'])

# Boxplot for Snow Category and Min Delay
plt.figure(figsize=(10, 6))
sns.boxplot(x=merge_data['Snow Category'], y=merge_data['Min Delay'])
plt.title("Bus Delays by Snow Category")
plt.xlabel("Snow Category")
plt.ylabel("Min Delay (minutes)")
plt.show()

# Boxplot for Rain Category and Min Delay
plt.figure(figsize=(10, 6))
sns.boxplot(x=merge_data['Rain Category'], y=merge_data['Min Delay'])
plt.title("Bus Delays by Rain Category")
plt.xlabel("Rain Category")
plt.ylabel("Min Delay (minutes)")
plt.show()

# Group by Month and visualize average delay
month_delay = merge_data.groupby('Month')['Min Delay'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=month_delay['Month'], y=month_delay['Min Delay'])
plt.title("Average Bus Delays by Month")
plt.xlabel("Month")
plt.ylabel("Average Delay (minutes)")
plt.xticks([1, 2, 3, 4, 5, 6], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
plt.show()

# Boxplot by Time of Day and Snow/Rain
plt.figure(figsize=(10, 6))
sns.boxplot(x=merge_data['Time of Day'], y=merge_data['Min Delay'], hue=merge_data['Snow on Grnd (cm)'] > 0, palette={True: 'blue', False: 'gray'})
plt.legend(title='Snow', labels=['No Snow', 'Snow'])
plt.title("Bus Delays by Time of Day and Snow")
plt.xlabel("Time of Day")
plt.ylabel("Min Delay (minutes)")
plt.xticks(rotation=45)
plt.show()

# Correlation Matrix (Fix: Filter numeric columns)
numeric_columns = merge_data[['Min Delay', 'Total Rain (mm)', 'Snow on Grnd (cm)']]

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Weather Features and Bus Delays")
plt.show()

# Random Forest Model

# Define feature columns
feature_cols = ['Total Rain (mm)', 'Snow on Grnd (cm)', 'Month', 'Hour of Day']

# Define input (X) and target variable (y)
X = merge_data[feature_cols]
y = merge_data['Min Delay']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cross_val = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Print evaluation results
print(f'Random Forest Performance:')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-Squared: {r2:.2f}')
print(f'Cross-Validation Score: {-cross_val.mean()}')

importances = rf_model.feature_importances_
features = X.columns
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.show()

# Poisson Regression Model
feature_cols = ['Total Rain (mm)', 'Snow on Grnd (cm)', 'Month']
X = merge_data[feature_cols]
y = merge_data['Min Delay']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poisson_model = PoissonRegressor()
poisson_model.fit(X_train, y_train)
y_pred_poisson = poisson_model.predict(X_test)

# Evaluation of Poisson Regression
aberror_poisson = mean_absolute_error(y_test, y_pred_poisson)
mse_poisson = mean_squared_error(y_test, y_pred_poisson)
r2_poisson = r2_score(y_test, y_pred_poisson)
poisson_crossvalidation_scores = cross_val_score(poisson_model, X, y, cv=5, scoring='neg_mean_squared_error')

print(f'Poisson Regression Performance:')
print(f'Mean Absolute Error: {aberror_poisson:.2f}')
print(f'Mean Squared Error: {mse_poisson:.2f}')
print(f'R-Squared: {r2_poisson:.2f}')
print(f'Cross-Validation Score: {poisson_crossvalidation_scores.mean()}')

# Neural Network Model
mlp_model = MLPRegressor()
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)

# Evaluation of Neural Network
aberror_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
neuralnetwork_cv_scores = cross_val_score(mlp_model, X, y, cv=5, scoring='neg_mean_squared_error')

print(f'Neural Network Performance:')
print(f'Mean Absolute Error: {aberror_mlp:.2f}')
print(f'Mean Squared Error: {mse_mlp:.2f}')
print(f'R-Squared: {r2_mlp:.2f}')
print(f'Cross-Validation Score: {neuralnetwork_cv_scores.mean()}')

# Scatterplots for Actual vs Predicted Delays (Random Forest, Poisson Regression, and Neural Network Models)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Delay')
plt.ylabel('Predicted Delay')
plt.title('Actual vs Predicted Bus Delays using Random Forest')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poisson, alpha=0.7)
plt.xlabel('Actual Delay')
plt.ylabel('Predicted Delay')
plt.title('Actual vs Predicted Bus Delays using Poisson Regression')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_mlp, alpha=0.7)
plt.xlabel('Actual Delay')
plt.ylabel('Predicted Delay')
plt.title('Actual vs Predicted Bus Delays using Neural Network')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

# Linear Regression Model
X_regression = add_constant(X)
linear_model = OLS(y, X_regression).fit()
print(linear_model.summary())

# Time Series Analysis (SARIMA)
# Check stationarity
monthly_delays = merge_data.groupby(merge_data['Date'].dt.to_period('M'))['Min Delay'].mean()
monthly_delays.index = monthly_delays.index.to_timestamp()
result = adfuller(monthly_delays)

if result[1] > 0.05:
    print("Time series is NOT stationary. Differencing will be applied.")
    monthly_delays_diff = monthly_delays.diff().dropna()
else:
    print("Time series is stationary.")
    monthly_delays_diff = monthly_delays

# Auto ARIMA model
auto_model = pm.auto_arima(monthly_delays, seasonal=False, trace=True, stepwise=True)
order = auto_model.order
sarima_model = SARIMAX(monthly_delays, order=order, enforce_stationarity=False, enforce_invertibility=False)
sarima_results = sarima_model.fit()

# Forecast next 6 months
forecast_steps = 6
predict = sarima_results.get_forecast(steps=forecast_steps)
predict_mean = predict.predicted_mean
predict_confidence_interval = predict.conf_int()

# Plot forecast results
plt.figure(figsize=(10, 6))
plt.plot(monthly_delays, label='Actual Delays')
plt.plot(predict_mean, label='Forecast', linestyle='dashed', color='red')
plt.fill_between(predict_confidence_interval.index,
                 predict_confidence_interval.iloc[:, 0],
                 predict_confidence_interval.iloc[:, 1], color='red', alpha=0.3)
plt.title('SARIMA Forecast for Future Bus Delays')
plt.xlabel('Date')
plt.ylabel('Avg Delay (Minutes)')
plt.legend()
plt.grid(True)
plt.show()

# Residual analysis
residuals = sarima_results.resid
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals', color='orange')
plt.title('Residuals from SARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Plot residuals distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()