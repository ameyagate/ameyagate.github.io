
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cars_data= pd.read_csv('cars_sampled.csv')
cars=cars_data.copy()

print(cars.info())
summary= cars.describe()

col=['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen'] #dropping unwanted col
cars=cars.drop(columns=col, axis=1)

cars.drop_duplicates(keep='first', inplace=True) #removing duplicate row data

cars.isnull().sum()

#Identify working ranges columnwise

#variable yearOfRegistration
yearwise_counts=cars['yearOfRegistration'].value_counts().sort_index()
sns.regplot(x='yearOfRegistration', y='price', fit_reg=False, data=cars)
#working range 1950 to 2018

#variable price
price_counts=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
#working range 100 to 150000


#variable powerPS
power_count= cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS', y='price', fit_reg=False, data=cars)
sum(cars['powerPS'] >500)
sum(cars['powerPS'] <10)
#working range 10 and 500

#-------------
#Working range of data
#-------------

#keeping only working range and dropping rest records (outliers)
cars= cars[
    (cars.yearOfRegistration <=2018)
    & (cars.yearOfRegistration >=1950)
    & (cars.price >= 100)
    & (cars.price <= 150000)
    & (cars.powerPS >=10)
    & (cars.powerPS <= 500)
    ]

#combining year of reg and month of reg

cars['monthOfRegistration']/=12

cars['Age']= (2018- cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']= round(cars['Age'],2)
cars['Age'].describe()

cars=cars.drop(columns=['yearOfRegistration', 'monthOfRegistration'], axis=1)


#visualizing parameters

#age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#age vs price
sns.regplot(x='Age', y='price', fit_reg=False, data=cars)

#powerPS vs price
sns.regplot(x='powerPS', y='price', fit_reg=False, data=cars)

#checking and removing insignificant categorical variables

#Variable seller
cars['seller'].value_counts() #gives count
pd.crosstab(cars['seller'], columns='count', normalize=True) #gives percent
sns.countplot(x='seller', data=cars)
#very few cars have 'commercial' seller, hence insignificant

#Variable offerType
cars['offerType'].value_counts() #gives count
pd.crosstab(cars['offerType'], columns='count', normalize=True) #gives percent
sns.countplot(x='offerType', data=cars)
#All cars have same offer, hence insignificant

#Variable abtest
cars['abtest'].value_counts() #gives count
pd.crosstab(cars['abtest'], columns='count', normalize=True) #gives percent
sns.countplot(x='abtest', data=cars)
sns.boxplot(x='abtest', y='price', data=cars)
#Price is not affected by this variable, hence insignificant

#Variable vehicleType
cars['vehicleType'].value_counts() #gives count
pd.crosstab(cars['vehicleType'], columns='count', normalize=True) #gives percent
sns.countplot(x='vehicleType', data=cars)
sns.boxplot(x='vehicleType', y='price', data=cars)
#affects price 

#Variable gearbox
cars['gearbox'].value_counts() #gives count
pd.crosstab(cars['gearbox'], columns='count', normalize=True) #gives percent
sns.countplot(x='gearbox', data=cars)
sns.boxplot(x='gearbox', y='price', data=cars)
#affects price

#removing insignificant variables
col2=['seller','offerType', 'abtest']
cars=cars.drop(columns=col2, axis=1)

#Correlation with price
cars_select1=cars.select_dtypes(exclude=[object])
correlation= cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs()

#removing missing values
cars_omit= cars.dropna(axis=0)
cars_omit= pd.get_dummies(cars_omit, drop_first= True)

#importing necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#-----------
#model building with omitted data
#-----------

#separating input and output features
x1= cars_omit.drop(['price'], axis=1, inplace=False)
y1= cars_omit['price'] 

#plotting price and best checking distribution curve
prices= pd.DataFrame({'before':y1, 'after':np.log(y1)})
prices.hist()
#logarithmic values has natural bell shpaed curve, hence transforming price as log values
y1= np.log(y1)

#splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1,y1,test_size=0.3, random_state=3)

#finding mean for test data value
base_pred= np.mean(y_test)

base_pred= np.repeat(base_pred, len(y_test)) #repeating same value till length of test data

#finding RSME
base_root_mean_squared_error= np.sqrt(mean_squared_error(y_test, base_pred))

#----------
#Linear regression with omitted data
#----------
lgr= LinearRegression(fit_intercept=True)

#model
model_lin1= lgr.fit(X_train, y_train)

#predicting model on test set
cars_predictions_lin1= lgr.predict(X_test)

#computing MSE & RMSE
lin_mse1= mean_squared_error(y_test, cars_predictions_lin1)
lin_mse1= np.sqrt(lin_mse1)

#R squared value - checks how good is the model able to explain the variablity
r2_lin_test1= model_lin1.score(X_test, y_test)
r2_lin_train1= model_lin1.score(X_train, y_train)
print(r2_lin_test1, r2_lin_train1) #should be same or close

#regression diagnostics- residual plot analysis
residuals1= y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1, y=residuals1, fit_reg=False) #values should be closer to 0
residuals1.describe() #mean is near 0, shows good indication that predicted and actual values are close

input=
output=model_lin1.predict(input)

#---------
# Random Forest with omitted data
#---------

#model parameters
rf= RandomForestRegressor(n_estimators=100, max_depth=100, 
                          min_samples_split=10, min_samples_leaf=4,
                          random_state=1)

#model
model_rf1= rf.fit(X_train, y_train)

#predicting model on test set
cars_prediction_rf1 = rf.predict(X_test)

#computing MSE & RMSE
rf_mse1= mean_squared_error(y_test, cars_predictions_rf1)
rf_rmse1= np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
r2_rf_test1= model_rf1.score(X_test, y_test)
r2_rf_train1= model_rf1.score(X_train, y_train)
print(r2_rf_test1, r2_rf_train1)


#-----------
# Model building with imputed data
#-----------

cars_imputed= cars.apply(lambda x:x.fillna(x.median())
                         if x.dtype== 'float' else
                         x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()









