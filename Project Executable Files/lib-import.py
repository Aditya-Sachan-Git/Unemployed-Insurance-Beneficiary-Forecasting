#importing required libraries
import pandas as pd
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error,r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import prophet
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import AutoReg


#Reading data 
import pandas as pd # Redundant import for troubleshooting
df = pd.read_csv(r'C:\Users\Sri charan\Downloads\unemployment-insurance-beneficiaries-and-benefit-amounts-paid-beginning-2001-1 (1).csv')
df.head()
df.info()
df.shape
print(df.isna().sum())
print(df.Region.value_counts())
print(df.duplicated().sum())
df.columns = df.columns.str.strip()
df['Beneficiaries_diff']=df['Beneficiaries'].diff() #creation of Beneficiaries_diff colounm

#Visualization of Data
#univariate analysis
fig = px.line(df, x='Year', y='Benefit Amounts (Dollars)')
fig.show()
fig = px.line(df, x='Year', y='Beneficiaries')
fig.show()

#Bivariate analysis
df1=df.query("County in ['Hamilton','Kings']")
fig = px.line(df1, x='Year', y='Beneficiaries', color='County')
fig.update_traces(textposition="bottom right")
fig.show()
fig=px.bar(df,x='Region',y='Beneficiaries',color='Region',text_auto=True)
fig.show()

plt.figure(figsize=(20,5))
sns.barplot(x=df.Region,y=df.Beneficiaries)
plt.show()

#Multivariate Analysis
for i in df.columns:
  if(df[i].dtype)=='int64':
    boxplot=sns.boxplot(x=df[i])
    plt.title(i)
    plt.show()
#Decriptive Analysis
df.describe()

#Training and Testing 

df.dropna(inplace=True)
df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
df.set_index('ds', inplace=True)
if df.index.duplicated().sum() > 0:
    print("Duplicates found. Fixing...")
    df = df[~df.index.duplicated(keep='first')]

df = df.asfreq('MS')
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]


#Checking and converstion of Data (Stainonary to non-stationary)
#Augmented Dickey-Fuller test(used to convert non-stationary data to
#stationary data)

adf=adfuller(df['Beneficiaries'],autolag='AIC')
print("P-Value",adf[1])

adf=adfuller(train['Beneficiaries_diff'],autolag='AIC')
print("P-Value",adf[1])

#ACF and PACF(to check how our data is correlated with ACF &PACF)
plot_acf(train['Beneficiaries'], lags=30, title='Original ACF')
plot_pacf(train['Beneficiaries'], lags=30, title='Original PACF')
plt.show()
#differenced ACF and PCAF
plot_acf(train['Beneficiaries_diff'], lags=30, title='Differenced ACF')
plot_pacf(train['Beneficiaries_diff'], lags=30, title='Differenced PACF')
plt.show()
#Augmented Dickey-Fuller test(used to convert non-stationary data to
#stationary data)

adf=adfuller(df['Beneficiaries'],autolag='AIC')
print("P-Value",adf[1])

adf=adfuller(train['Beneficiaries_diff'],autolag='AIC')
print("P-Value",adf[1])

#ACF and PACF(to check how our data is correlated with ACF &PACF)
plot_acf(train['Beneficiaries'], lags=30, title='Original ACF')
plot_pacf(train['Beneficiaries'], lags=30, title='Original PACF')
plt.show()
#differenced ACF and PCAF
plot_acf(train['Beneficiaries_diff'].dropna(), lags=30, title='Differenced ACF')
plot_pacf(train['Beneficiaries_diff'].dropna(), lags=30, title='Differenced PACF')
plt.show()


#smoothing out our data(visual representation)
plt.plot(train['Beneficiaries'])
plt.plot(train['Beneficiaries_diff'])
plt.show()

#Model Building 
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train['Beneficiaries_diff'].dropna(), order=(5,0,0)) # Pass the series with the correct index
model_arima=model.fit()
model_arima.summary()

#SARIMA
model=SARIMAX(train['Beneficiaries_diff'].dropna(),order=(5,0,0),seasonal_order=(0,1,2,3))
model_sarima=model.fit()
model_sarima.summary()


#Auto Regression
model_ar=AutoReg(train['Beneficiaries_diff'].dropna(), lags=10).fit()
model_ar.summary()

#Prophet
model =VAR(train[['Beneficiaries_diff','Benefit Amounts (Dollars)']].dropna()) # Add dropna()
model_AR = model.fit(maxlags=10)
model_AR.summary()

#PERFORMNCE TESTING OF THE MODEL

#Arima
prediction_arima =model_arima.predict(start=len(train),
                                    end=len(train)+len(test)-1,
                                    type='levels')
prediction_arima
#metrics for checking the model
print(mean_squared_error(test['Beneficiaries_diff'],prediction_arima))
print(mean_absolute_error(test['Beneficiaries_diff'],prediction_arima))
print(r2_score(test['Beneficiaries_diff'],prediction_arima))

#SARIMA
prediction_sarima=model_sarima.predict(start=len(train),
                               end=len(train)+ len(test)-1,
                               type='levels')
#metric check for SARIMA
print(mean_squared_error(test['Beneficiaries_diff'],prediction_sarima))
print(mean_absolute_error(test['Beneficiaries_diff'],prediction_sarima))


#building  AR model
predictions_ar=model_ar.predict(start=len(train),end=len(train)+len(test)-1)
#building VAR model
prediction_var = model_AR.forecast(train[['Beneficiaries_diff','Benefit Amounts (Dollars)']].values,steps=5)

import prophet

# Prepare the data for Prophet
# Prophet requires the dataframe to have columns named 'ds' and 'y'
prophet_df = df[['Year', 'Month', 'Beneficiaries']].copy()
prophet_df['ds'] = pd.to_datetime(prophet_df['Year'].astype(str) + '-' + prophet_df['Month'].astype(str))
prophet_df = prophet_df[['ds', 'Beneficiaries']].rename(columns={'Beneficiaries': 'y'})

# Instantiate and fit the Prophet model
model_prophet = prophet.Prophet()
model_prophet.fit(prophet_df)

# Make future predictions
# Create a dataframe with future dates for prediction
future = model_prophet.make_future_dataframe(periods=len(test))
# Predict future values
forecast = model_prophet.predict(future)
# Display the predictions
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
model_prophet.plot(forecast)
