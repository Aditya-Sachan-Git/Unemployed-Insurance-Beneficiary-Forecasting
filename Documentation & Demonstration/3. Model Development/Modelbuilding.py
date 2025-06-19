#ModelBuilding

#Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
adf=adfuller(df['beneficiaries'],autolag='AIC')
print("P-Value",adf[1])
adf=adfuller(train['beneficiaries_diff'],autolag='AIC')
print("P-Value",adf[1])

#ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plot_acf(train['beneficiaries'], lags=30, title='Original ACF')
plot_pacf(train['beneficiaries'], lags=30, title='Original PACF')
plt.show()
plot_acf(train['beneficiaries_diff'], lags=30, title='Differenced ACF')
plot_pacf(train['beneficiaries_diff'], lags=30, title='Differenced PACF')
plt.show()

#ARIMA
from pmdarima import auto_arima
stepwise=auto_arima(df['beneficiaries_diff'], trace=True, supress_warnings=True)
model = ARIMA(train['beneficiaries_diff'], order=(5,0,0))
model_arima=model.fit()
model_arima.summary()

#SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
model=SARIMAX(train['beneficiaries_diff'],order=(5,0,0),seasonal_order=(0,1,2,3))
model_sarima=model.fit()
model_sarima.summary()

#Auto Regression
from statsmodels.tsa.ar_model import Autoreg
model_ar=Autoreg(train['beneficiaries_diff'], lags=10).fit()
model.ar.summary()

#Prophet
from statsmodels.tsa.api import VAR
model_AR = model.fit(maxlags=10)


