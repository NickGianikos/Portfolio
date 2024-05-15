#!/usr/bin/env python
# coding: utf-8

# ## This script is indeded for BSYS 4005 Module 4
# The purpose is to evaluate different approaches to ARIMA forcasting and looking at how to test and select parameters
# It roughly follows the instructions here:
# https://medium.com/@raj.saha3382/forecasting-of-stock-market-using-arima-in-python-cd4fe76fc58a

# In[1]:


#for this to work you will need to install any required libraries first
 #pmdarima # for the auto arima model
 #pandas_datareader # for reading data from the web

#on windows from a command prompt (start run cmd) paste and run this line
    #pip install ####LIBRARYNAME####
#or in anaconda make sure the library is included in the envorinments page (search installed/uninstalled) and check and apply and then wait many minutes...
    

#import all libraries used below
import pandas as pd # for data analytics
import pandas_datareader.data as pdr # for reading web data
import datetime as dt #for managing dates
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for statistics visuals
from statsmodels.tsa.stattools import adfuller as adf # for the adf stationary check
from statsmodels.tsa.seasonal import seasonal_decompose as sd # for a seasonal decomposition
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima as aa #for selecting the model options
import numpy as np # for math
from sklearn.metrics import mean_squared_error as mse #for RMSE calculation
from sklearn.metrics import mean_absolute_error as mae #for MAE calculation

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print('Done')


# # Download the data and save to a CSV to avoid too many web data reads
# load the data from the web reader and then store into a local csv file

# In[2]:


#this seems to have been depreciated
#df = web.get_data_yahoo(['NVDA'], start=datetime.datetime(2022, 1, 1), end=datetime.datetime(2023, 12, 2))['Close']
#https://pandas-datareader.readthedocs.io/en/latest/remote_data.html

#the stooq data reader is working as of Jan 2024 - use whatever stock ticker you want
df = pdr.DataReader('NVDA', 'stooq')
print(df.head())

#write to local csv
df.to_csv("M4_stock_history.csv")

#load up to the end of 2023
stock = pd.read_csv("M4_stock_history.csv")
print(stock.head())

#replace the index with the date and delete the redundant date
stock.index = pd.to_datetime(stock['Date'], format='%Y-%m-%d')
print(stock.head())

#leave just the closing price
del stock['Date']
del stock['Open']
del stock['High']
del stock['Low']
del stock['Volume']

#reserve 2024 data for final check and only include up to the end of 2023 for training/testing
stock_2025 = stock[stock.index >= pd.to_datetime("2025-01-01", format='%Y-%m-%d')]
stock = stock[stock.index < pd.to_datetime("2025-01-01", format='%Y-%m-%d')]

#convert the index to show daily data
#stock.index = pd.DatetimeIndex(stock.index).to_freq('D')
stock = stock.asfreq('D')
stock_2025 = stock_2025.asfreq('D')
stock = stock.fillna(method='ffill')
stock_2025 = stock_2025.fillna(method='ffill')
print(stock.head())
print(stock.index)


# # Visualize the Stock data and check for stationary assumptions
# load the data from the web reader and then store into a local csv file

# In[3]:


#plot some stuff
sns.set()

#4. Checking whether the data is stationary or not:  Before checking that, we need to know what stationary data means…right? The data is stationary if they do not have trend or any seasonal effects. And if the data is non-stationary, then we have to convert it to stationary data before fitting into the ARIMA model. To check whether the data is stationary, we will use Augmented Dicky Fuller(ADF) test.

#define a function for the ADF test
def test_adf(timeseries):
  moving_avg=timeseries.rolling(12).mean()
  moving_std=timeseries.rolling(12).std()
  plt.figure(figsize=(20,10))
  plt.plot(timeseries, color='blue', label='Data')
  plt.plot(moving_avg, color='red', label='Rolling Mean')
  plt.plot(moving_std, color='black', label='Rolling Standard')
  plt.legend(loc='best')
  plt.title('Rolling Mean and Standard Deviation')
  plt.show(block=False)
  print("Results of ADF test")
  adft=adf(timeseries,autolag='AIC')
  output=pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of Observations used']) 
  for key,value in adft[4].items():
    output['Critical Value (%s)'%key]=value
  print(output)

test_adf(stock)

#if the test statistic is greater than the critical values, so p>0.05, we fail to reject the null hypothesis meaning that the time series data is non-stationary
#in this case the data is stationary so we do not need to do a log conversion or anything like that


# In[4]:


#we should also check the seasinality components to see if this will help in the modelling
#see what happens when you change the seasonality from 7 to 14 to 28
result=sd(stock,model='multiplicative',period=28)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16,9)


# # split the data into multiple training and testing sets
# choice of split is up to you so we will try a few and compare results
# plot the two to see the difference and do different models as a 'cross validation' approach

# In[5]:


train_2021 = stock[stock.index < pd.to_datetime("2021-01-01", format='%Y-%m-%d')]
test_2021 = stock[stock.index >= pd.to_datetime("2021-01-01", format='%Y-%m-%d')]
pd.DatetimeIndex(train_2021.index).to_period('D')
pd.DatetimeIndex(test_2021.index).to_period('D')

train_2022 = stock[stock.index < pd.to_datetime("2022-01-01", format='%Y-%m-%d')]
test_2022 = stock[stock.index >= pd.to_datetime("2022-01-01", format='%Y-%m-%d')]
pd.DatetimeIndex(train_2022.index).to_period('D')
pd.DatetimeIndex(test_2022.index).to_period('D')

train_2023 = stock[stock.index < pd.to_datetime("2023-01-01", format='%Y-%m-%d')]
test_2023 = stock[stock.index >= pd.to_datetime("2023-01-01", format='%Y-%m-%d')]
pd.DatetimeIndex(train_2023.index).to_period('D')
pd.DatetimeIndex(test_2023.index).to_period('D')

#test
train_2024 = stock[stock.index < pd.to_datetime("2024-01-01", format='%Y-%m-%d')]
test_2024 = stock[stock.index >= pd.to_datetime("2024-01-01", format='%Y-%m-%d')]
pd.DatetimeIndex(train_2024.index).to_period('D')
pd.DatetimeIndex(test_2024.index).to_period('D')

train_2025 = stock[stock.index < pd.to_datetime("2025-01-01", format='%Y-%m-%d')]
test_2025 = stock[stock.index >= pd.to_datetime("2025-01-01", format='%Y-%m-%d')]
pd.DatetimeIndex(train_2025.index).to_period('D')
pd.DatetimeIndex(test_2025.index).to_period('D')


print(train_2021.head())
print(train_2021.index)
print(test_2021.head())
print(test_2021.index)
print(train_2022.head())
print(test_2022.head())
print(test_2022.index)
print(train_2023.head())
print(test_2023.head())
print(test_2023.index)

print(train_2024.head())
print(test_2024.head())
print(test_2024.index)

print(train_2025.head())
print(test_2025.head())
print(test_2025.index)


# In[6]:


#2021 split plot
plt.plot(train_2021.index, train_2021['Close'], color = "black")
plt.plot(test_2021.index, test_2021['Close'], color = "red")
plt.ylabel('Stock Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Stock Data based on 2021")
plt.show()

#2022 split plot
plt.plot(train_2022.index, train_2022['Close'], color = "black")
plt.plot(test_2022.index, test_2022['Close'], color = "red")
plt.ylabel('Stock Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Stock Data based on 2022")
plt.show()

#2023
plt.plot(train_2023.index, train_2023['Close'], color = "black")
plt.plot(test_2023.index, test_2023['Close'], color = "red")
plt.ylabel('Stock Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Stock Data based on 2023")
plt.show()


# # Autoregressive Moving Average (ARIMA) Model comparisons
# Try different techniques and see what options result in better models

# In[7]:


#use the auto arima to compare which parameters have the best score... is it the same for both 2021 and 2022?
model_ARIMA_options_2021 = aa(train_2021
 ,start_p=0
 ,start_1=0
 ,test='adf'
 ,max_p=7
 ,max_q=7
 ,m=1
 ,d=None
 ,seasonal=False
 ,start_P=0
 ,D=0
 ,trace=True
 ,error_action='ignore'
 ,suppress_warning=True
 ,stepwise=True)

#use the auto arima to compare which parameters have the best score... is it the same for both 2021 and 2022?
model_ARIMA_options_2022 = aa(train_2022
 ,start_p=0
 ,start_1=0
 ,test='adf'
 ,max_p=7
 ,max_q=7
 ,m=1
 ,d=None
 ,seasonal=False
 ,start_P=0
 ,D=0
 ,trace=True
 ,error_action='ignore'
 ,suppress_warning=True
 ,stepwise=True)

model_ARIMA_options_2023 = aa(train_2023
 ,start_p=0
 ,start_1=0
 ,test='adf'
 ,max_p=7
 ,max_q=7
 ,m=1
 ,d=None
 ,seasonal=False
 ,start_P=0
 ,D=0
 ,trace=True
 ,error_action='ignore'
 ,suppress_warning=True
 ,stepwise=True)


model_ARIMA_options_2024 = aa(train_2024
 ,start_p=0
 ,start_1=0
 ,test='adf'
 ,max_p=7
 ,max_q=7
 ,m=1
 ,d=None
 ,seasonal=False
 ,start_P=0
 ,D=0
 ,trace=True
 ,error_action='ignore'
 ,suppress_warning=True
 ,stepwise=True)


# # 2021 options
# Try different techniques and see what options result in better models

# In[8]:


#now do the actual model fit and forecast
model_ARIMA_options_2021.fit(train_2021)
#model_ARIMA_options_2022.fit(train_2022)


forecast_2021_autoarima = model_ARIMA_options_2021.predict(n_periods=len(test_2021))
forecast_2021_autoarima = pd.DataFrame(forecast_2021_autoarima,index = test_2021.index,columns=['Prediction'])

#also compare some other options
model_2021_011=ARIMA(train_2021, order = (0,1,1))
model_2021_011_fit = model_2021_011.fit()
forecast_2021_011 = model_2021_011_fit.get_forecast(len(test_2021.index))
forecast_2021_011_df = forecast_2021_011.conf_int(alpha = 0.05) 
forecast_2021_011_df["Predictions"] = model_2021_011_fit.predict(start = forecast_2021_011_df.index[0], end = forecast_2021_011_df.index[-1])
forecast_2021_011_df.index = test_2021.index
forecast_2021_011_out = forecast_2021_011_df["Predictions"]
print(forecast_2021_011_out.head())

model_2021_112=ARIMA(train_2021, order = (1,1,2))
model_2021_112_fit = model_2021_112.fit()
forecast_2021_112 = model_2021_112_fit.get_forecast(len(test_2021.index))
forecast_2021_112_df = forecast_2021_112.conf_int(alpha = 0.05) 
forecast_2021_112_df["Predictions"] = model_2021_112_fit.predict(start = forecast_2021_112_df.index[0], end = forecast_2021_112_df.index[-1])
forecast_2021_112_df.index = test_2021.index
forecast_2021_112_out = forecast_2021_112_df["Predictions"]
print(forecast_2021_112_out.head())

model_2021_013=ARIMA(train_2021, order = (0,1,3))
model_2021_013_fit = model_2021_013.fit()
forecast_2021_013 = model_2021_013_fit.get_forecast(len(test_2021.index))
forecast_2021_013_df = forecast_2021_013.conf_int(alpha = 0.05) 
forecast_2021_013_df["Predictions"] = model_2021_013_fit.predict(start = forecast_2021_013_df.index[0], end = forecast_2021_013_df.index[-1])
forecast_2021_013_df.index = test_2021.index
forecast_2021_013_out = forecast_2021_013_df["Predictions"]
print(forecast_2021_013_out.head())

#plot the predictions for validation set
plt.plot(train_2021, label='Train')
plt.plot(test_2021, label='Validation')
plt.plot(forecast_2021_autoarima, label='Auto Arima Prediction')
plt.plot(forecast_2021_011_out, label='Arima (0,1,1) Prediction')
plt.plot(forecast_2021_112_out, label='Arima (1,1,2) Prediction')
plt.plot(forecast_2021_013_out, label='Arima (0,1,3) Prediction')
plt.legend()
plt.show()




# # 2022 options
# Try different techniques and see what options result in better models

# In[9]:


#now do the actual model fit and forecast
model_ARIMA_options_2022.fit(train_2022)

forecast_2022_autoarima = model_ARIMA_options_2022.predict(n_periods=len(test_2022))
forecast_2022_autoarima = pd.DataFrame(forecast_2022_autoarima,index = test_2022.index,columns=['Prediction'])

#also compare some other options
model_2022_011=ARIMA(train_2022, order = (0,1,1))
model_2022_011_fit = model_2022_011.fit()
forecast_2022_011 = model_2022_011_fit.get_forecast(len(test_2022.index))
forecast_2022_011_df = forecast_2022_011.conf_int(alpha = 0.05) 
forecast_2022_011_df["Predictions"] = model_2022_011_fit.predict(start = forecast_2022_011_df.index[0], end = forecast_2022_011_df.index[-1])
forecast_2022_011_df.index = test_2022.index
forecast_2022_011_out = forecast_2022_011_df["Predictions"]
print(forecast_2022_011_out.head())

model_2022_112=ARIMA(train_2022, order = (1,1,2))
model_2022_112_fit = model_2022_112.fit()
forecast_2022_112 = model_2022_112_fit.get_forecast(len(test_2022.index))
forecast_2022_112_df = forecast_2022_112.conf_int(alpha = 0.05) 
forecast_2022_112_df["Predictions"] = model_2022_112_fit.predict(start = forecast_2022_112_df.index[0], end = forecast_2022_112_df.index[-1])
forecast_2022_112_df.index = test_2022.index
forecast_2022_112_out = forecast_2022_112_df["Predictions"]
print(forecast_2022_112_out.head())

#plot the predictions for validation set
plt.plot(train_2022, label='Train')
plt.plot(test_2022, label='Validation')
plt.plot(forecast_2022_autoarima, label='Auto Arima Prediction')
plt.plot(forecast_2022_011_out, label='Arima (0,1,1) Prediction')
plt.plot(forecast_2022_112_out, label='Arima (1,1,2) Prediction')
plt.legend()
plt.show()


# # Which model is better? - compare RMSE

# In[10]:


#plot the predictions for validation set FOR 2021
plt.plot(train_2021, label='2021 Train')
plt.plot(test_2021, label='2021 Validation')
plt.plot(forecast_2021_autoarima, label='2021 Auto Arima Prediction')
plt.plot(forecast_2021_011_out, label='Arima (0,1,1) Prediction')
plt.plot(forecast_2021_112_out, label='Arima (1,1,2) Prediction')
plt.legend()
plt.show()

rmse_2021_011 = np.sqrt(mse(test_2021.values, forecast_2021_011_out))
print("ARIMA(0,1,1) RMSE 2021 : ",rmse_2021_011)
rmse_2021_112 = np.sqrt(mse(test_2021.values, forecast_2021_112_out))
print("ARIMA(1,1,2) RMSE 2021: ",rmse_2021_112)
rmse_2021_AutoArima = np.sqrt(mse(test_2021.values, forecast_2021_autoarima))
print("AutoArima RMSE 2021 : ",rmse_2021_AutoArima)

print()

mae_2021_011 = mae(test_2021.values, forecast_2021_011_out)
print("ARIMA(0,1,1) MAE 2021 : ",mae_2021_011)
mae_2021_112 = mae(test_2021.values, forecast_2021_112_out)
print("ARIMA(1,1,2) MAE 2021: ",mae_2021_112)
mae_2021_AutoArima = mae(test_2021.values, forecast_2021_autoarima)
print("AutoArima MAE 2021 : ",mae_2021_AutoArima)


# In[11]:


#plot the predictions for validation set FOR 2022
plt.plot(train_2022, label='2022 Train')
plt.plot(test_2022, label='2022 Validation')
plt.plot(forecast_2022_autoarima, label='2022 Auto Arima Prediction')
plt.plot(forecast_2022_011_out, label='Arima (0,1,1) Prediction')
plt.plot(forecast_2022_112_out, label='Arima (1,1,2) Prediction')
plt.legend()
plt.show()

rmse_2022_011 = np.sqrt(mse(test_2022.values, forecast_2022_011_out))
print("ARIMA(0,1,1) RMSE 2022 : ",rmse_2022_011)
rmse_2022_112 = np.sqrt(mse(test_2022.values, forecast_2022_112_out))
print("ARIMA(1,1,2) RMSE 2022: ",rmse_2022_112)
rmse_2022_AutoArima = np.sqrt(mse(test_2022.values, forecast_2022_autoarima))
print("AutoArima RMSE 2022: ",rmse_2022_AutoArima)

print()

mae_2022_011 = mae(test_2022.values, forecast_2022_011_out)
print("ARIMA(0,1,1) MAE 2022 : ",mae_2022_011)
mae_2022_112 = mae(test_2022.values, forecast_2022_112_out)
print("ARIMA(1,1,2) MAE 2022: ",mae_2022_112)
mae_2022_AutoArima = mae(test_2022.values, forecast_2022_autoarima)
print("AutoArima MAE 2022 : ",mae_2022_AutoArima)


# # Finally - use the withheld validation data set for 2024
# # what parameters will you use?
# who can get the lowest RMSE???

# In[12]:


#this is left as an individual assignment
#you are going to submit the code in this snippet and your output as a mini quiz on the hub
#I want to see your RMSE calculation on the 2024 data and describe the model you chose.


# In[13]:


#2023 Options
#now do the actual model fit and forecast
model_ARIMA_options_2023.fit(train_2023)

forecast_2023_autoarima = model_ARIMA_options_2023.predict(n_periods=len(test_2023))
forecast_2023_autoarima = pd.DataFrame(forecast_2023_autoarima,index = test_2023.index,columns=['Prediction'])

#also compare some other options
model_2023_011=ARIMA(train_2023, order = (0,1,1))
model_2023_011_fit = model_2023_011.fit()
forecast_2023_011 = model_2023_011_fit.get_forecast(len(test_2023.index))
forecast_2023_011_df = forecast_2023_011.conf_int(alpha = 0.05) 
forecast_2023_011_df["Predictions"] = model_2023_011_fit.predict(start = forecast_2023_011_df.index[0], end = forecast_2023_011_df.index[-1])
forecast_2023_011_df.index = test_2023.index
forecast_2023_011_out = forecast_2023_011_df["Predictions"]
print(forecast_2023_011_out.head())

model_2023_112=ARIMA(train_2023, order = (1,1,2))
model_2023_112_fit = model_2023_112.fit()
forecast_2023_112 = model_2023_112_fit.get_forecast(len(test_2023.index))
forecast_2023_112_df = forecast_2023_112.conf_int(alpha = 0.05) 
forecast_2023_112_df["Predictions"] = model_2023_112_fit.predict(start = forecast_2023_112_df.index[0], end = forecast_2023_112_df.index[-1])
forecast_2023_112_df.index = test_2023.index
forecast_2023_112_out = forecast_2023_112_df["Predictions"]
print(forecast_2023_112_out.head())

#added for the new trained data
model_2023_013=ARIMA(train_2023, order = (0,1,3))
model_2023_013_fit = model_2023_013.fit()
forecast_2023_013 = model_2023_013_fit.get_forecast(len(test_2023.index))
forecast_2023_013_df = forecast_2023_013.conf_int(alpha = 0.05) 
forecast_2023_013_df["Predictions"] = model_2023_013_fit.predict(start = forecast_2023_013_df.index[0], end = forecast_2023_013_df.index[-1])
forecast_2023_013_df.index = test_2023.index
forecast_2023_013_out = forecast_2023_013_df["Predictions"]
print(forecast_2023_013_out.head())


#plot the predictions for validation set
plt.plot(train_2023, label='Train')
plt.plot(test_2023, label='Validation')
plt.plot(forecast_2023_autoarima, label='Auto Arima Prediction')
plt.plot(forecast_2023_011_out, label='Arima (0,1,1) Prediction')
plt.plot(forecast_2023_112_out, label='Arima (1,1,2) Prediction')
plt.plot(forecast_2023_013_out, label='Arima (0,1,3) Prediction')
plt.legend()
plt.show()


# In[14]:


#now do the actual model fit and forecast
model_ARIMA_options_2024.fit(train_2024)

forecast_2024_autoarima = model_ARIMA_options_2024.predict(n_periods=len(test_2024))
forecast_2024_autoarima = pd.DataFrame(forecast_2024_autoarima,index = test_2024.index,columns=['Prediction'])

#also compare some other options
model_2024_011=ARIMA(train_2024, order = (0,1,1))
model_2024_011_fit = model_2024_011.fit()
forecast_2024_011 = model_2024_011_fit.get_forecast(len(test_2024.index))
forecast_2024_011_df = forecast_2024_011.conf_int(alpha = 0.05) 
forecast_2024_011_df["Predictions"] = model_2024_011_fit.predict(start = forecast_2024_011_df.index[0], end = forecast_2024_011_df.index[-1])
forecast_2024_011_df.index = test_2024.index
forecast_2024_011_out = forecast_2024_011_df["Predictions"]
print(forecast_2024_011_out.head())

model_2024_112=ARIMA(train_2024, order = (1,1,2))
model_2024_112_fit = model_2024_112.fit()
forecast_2024_112 = model_2024_112_fit.get_forecast(len(test_2024.index))
forecast_2024_112_df = forecast_2024_112.conf_int(alpha = 0.05) 
forecast_2024_112_df["Predictions"] = model_2024_112_fit.predict(start = forecast_2024_112_df.index[0], end = forecast_2024_112_df.index[-1])
forecast_2024_112_df.index = test_2024.index
forecast_2024_112_out = forecast_2024_112_df["Predictions"]
print(forecast_2024_112_out.head())

#added for the new trained data
model_2024_013=ARIMA(train_2024, order = (0,1,3))
model_2024_013_fit = model_2024_013.fit()
forecast_2024_013 = model_2024_013_fit.get_forecast(len(test_2024.index))
forecast_2024_013_df = forecast_2024_013.conf_int(alpha = 0.05) 
forecast_2024_013_df["Predictions"] = model_2024_013_fit.predict(start = forecast_2024_013_df.index[0], end = forecast_2024_013_df.index[-1])
forecast_2024_013_df.index = test_2024.index
forecast_2024_013_out = forecast_2024_013_df["Predictions"]
print(forecast_2024_013_out.head())


#plot the predictions for validation set
plt.plot(train_2024, label='Train')
plt.plot(test_2024, label='Validation')
plt.plot(forecast_2024_autoarima, label='Auto Arima Prediction')
plt.plot(forecast_2024_011_out, label='Arima (0,1,1) Prediction')
plt.plot(forecast_2024_112_out, label='Arima (1,1,2) Prediction')
plt.plot(forecast_2024_013_out, label='Arima (0,1,3) Prediction')
plt.legend()
plt.show()

