#Perfromace Testing
#Arima
prediction_arima =model_arima.predict(start=len(train),  #check this before implementation
                                    end=len(train)+len(test)-1,
                                    type='levels')
prediction_arima

#metrics for checking the model
mean_squared_error(test['beneficiaries_diff'],prediction_arima)
mean_absolute_error(test['beneficiaries_diff'],prediction_arima)
r2_score(test['beneficiaries_diff'],prediction_arima)

#SARIMA
prediction_sarima=model_sarima(start=len(train),
                               end=(train)+len(test)-1,
                               type='levels')
#metric check for SARIMA
mean_squared_error(test['Beneficiaries_diff'],prediction_sarima)
mean_absolute_error(test['Beneficiaries_diff'],prediction_sarima)

#building  AR model
predictions_ar=model_ar.predict(start=len(train),end=len(train)+len(test)-1)
#building VAR model
prediction_var = model_AR.forecast(train[['Beneficiaries_diff','Beneficiaries_amount_dollars']].values,steps=5)

#predicting prophet model
future = model_AR.make_future_dataframe(preiods=len(test))
forecast = model_AR.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].head()
model_AR.plot(forecast)#visualization of our forecast

#checking the preformance of our model
actual_values=test['y'].values
predicted_values = forecast[-len(test):]['yhat'].values

#applying metrics 
mae=mean_absolute_error(actual_values,predicted_values)
mse=mean_squared_error(actual_values,predicted_values)
rmse=np.sqrt(mse)
r2=r2_score(actual_values,predicted_values)

print(mae,mse,rmse,r2)
