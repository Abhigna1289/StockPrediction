# StockPrediction
Stock Price Prediction of steel industry using data from Thomson Reuters in Python

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import eikon as ek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#We installed the necessary packaged based on the code requirement.

ek.set_app_key('9070c68586e24e0096fcc6ba8172be383ce24c40')
df = ek.get_timeseries(["SAIL.NS"], start_date = "1998-10-10", end_date = "2019-11-11", interval = "daily")
print(df)

#For getting the timeseries data of SAIL company from Thomson Reuters using an APP key.

pf = df[['CLOSE']]
print(pf)

#Out of the CLOSE, HIGH, LOW etc will only consider the CLOSE price.

gdp = ek.get_timeseries(["INGDPY=ECI"], start_date = "2006-09-25", end_date = "2019-11-11", interval = "yearly")
print(gdp)
pf['gdp']=gdp
pf['gdp'].interpolate(method="time",inplace=True, limit_direction="both")


#We took the gdp data as yearly as we have the annual values

inf = ek.get_timeseries(["aINCCPIYF"], start_date = "2006-09-25", end_date = "2019-11-11", interval = "monthly")
print(inf)
pf['inf']=inf
pf['inf'].interpolate(method="time",inplace=True, limit_direction="both")

#We took the inflation data as monthly as we have the monthly values

cpi = ek.get_timeseries(["USCPNY=ECI"], start_date = "2001-10-10", end_date = "2019-02-05", interval = "monthly")
print(cpi)
pf['cpi']=cpi
pf['cpi'].interpolate(method="time",inplace=True, limit_direction="both")

#We took the gdp data as monthly as we have the monthly values

unemp = pd.read_csv('E:/Python/unem.csv', parse_dates=["DATE"],index_col="DATE")
print(unemp)
pf['unemp']=unemp
pf['unemp'].interpolate(method="time",inplace=True, limit_direction="both")

#For the inavailability of Unemployment data in Thomson Reuters we took the data from statista website and used the csv file. 

#For the above macro indicators we only took the timeframe based on the availability of data.

print(pf)
pf.corr()

#To find out the correlation based on which we can understand whether the taken economic indicators are significant or not

plt.figure(figsize=(15,10))
sns.heatmap(pf.corr(),annot=True)

#Heatmap for Correlation

forecast_out = int(input("enter no.of days"))
pf['Prediction'] = pf[['CLOSE']].shift(-forecast_out)

x = np.array(pf.drop(['Prediction','CLOSE'],1))
x = x[:-forecast_out]
print(x)
y = np.array(pf['CLOSE'])
y = y[:-forecast_out]
print(y)

#Here we took x as independent variables and y as dependent variable.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#We split the data into Test and Train with 80 20 ratio.

lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence) 

x_forecast = np.array(pf.drop(['Prediction','CLOSE'],1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)



#We Predict the 30 day close price of the taken company using linear regression

svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

#We predict the 30 day close price of the company using Support Vector Machine

plt.figure(figsize=(18,6))
plt.plot(y_test, color='red')
plt.plot(lr_prediction, color='blue')
plt.legend()
plt.show()

plt.plot(lr_prediction)
plt.plot(svm_prediction)

