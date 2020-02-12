# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:49:09 2020

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\Dell\Desktop\New folder\NewYork Taxi TripDuration\nyctrain\train.csv")
test = pd.read_csv(r"C:\Users\Dell\Desktop\New folder\NewYork Taxi TripDuration\nyctest\test.csv")
df.columns
df.info()
test.info()
# There are no missing values in training and testing set
df.isna().sum()
test.isna().sum()

df.describe()


#           OUTLIERS TREATMENT

# The passenger count varies from 1 to 9 in which most people are numbered to be 1 or2
#Some Outliers in - Trip duration(varies from 1 second to 538 hrs)
# TRIP DURATION OUTLIERS
plt.subplots(figsize=(15,5))
plt.title("Outliers visualization")
df.boxplot();
print( df['trip_duration'].nlargest(10))
df = df[(df.trip_duration < 6000)]
df = df[(df.trip_duration > 0)]

#pickup/dropoff_latitude,pickup/dropoff_longitude OUTLIERS
dropoff_longitude = list(df.dropoff_longitude)
dropoff_latitude = list(df.dropoff_latitude)
plt.subplots(figsize=(12,3))
plt.plot(dropoff_longitude, dropoff_latitude, '.', alpha = 1, markersize = 10)
plt.xlabel('dropoff_longitude')
plt.ylabel('dropoff_latitude')
plt.show()  

pickup_longitude = list(df.pickup_longitude)
pickup_latitude = list(df.pickup_latitude)
plt.subplots(figsize=(12,4))
plt.plot(pickup_longitude, pickup_latitude, '.', alpha = 1, markersize = 10)
plt.xlabel('pickup_longitude')
plt.ylabel('pickup_latitude')
plt.show()

#Remove position outliers
df = df[(df.pickup_longitude > -100)]
df = df[(df.pickup_latitude < 50)]
df = df[(df.dropoff_longitude < -70) & (df.dropoff_longitude > -80)]
df = df[(df.dropoff_latitude < 50)]
 
# FEATURE ENGINEERING

# 1.pickup_datetime and dropoff_datetime are non num.columns.we convert to datetime format

df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

df.drop(['dropoff_datetime'], axis=1, inplace=True) #as we don't have this feature in the testset

# 2.Date features creations and deletions
df['month'] = df.pickup_datetime.dt.month
df['week'] = df.pickup_datetime.dt.week
df['weekday'] = df.pickup_datetime.dt.weekday
df['hour'] = df.pickup_datetime.dt.hour
df['minute'] = df.pickup_datetime.dt.minute
df['minute_oftheday'] = df['hour'] * 60 + df['minute']
df.drop(['minute'], axis=1, inplace=True)

test['month'] = test.pickup_datetime.dt.month
test['week'] = test.pickup_datetime.dt.week
test['weekday'] = test.pickup_datetime.dt.weekday
test['hour'] = test.pickup_datetime.dt.hour
test['minute'] = test.pickup_datetime.dt.minute
test['minute_oftheday'] = test['hour'] * 60 + test['minute']
test.drop(['minute'], axis=1, inplace=True)

df.drop(['pickup_datetime'], axis=1, inplace=True)

df.columns
test.columns

test.drop(['pickup_datetime'], axis=1, inplace=True)

# 3.Dealing with Categorical Features
df = pd.concat([df, pd.get_dummies(df['store_and_fwd_flag'])], axis=1)
test = pd.concat([test, pd.get_dummies(test['store_and_fwd_flag'])], axis=1)

df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
test.drop(['store_and_fwd_flag'], axis=1, inplace=True)


df = pd.concat([df, pd.get_dummies(df['vendor_id'])], axis=1)
test = pd.concat([test, pd.get_dummies(test['vendor_id'])], axis=1)

df.drop(['vendor_id'], axis=1, inplace=True)
test.drop(['vendor_id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)


#4.Target(Trip duration)
plt.subplots(figsize=(12,3))
plt.hist(df['trip_duration'].values, bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()
 # Right Skewed perform log transformation
plt.subplots(figsize=(12,3))
df['trip_duration'] = np.log(df['trip_duration'].values)
plt.hist(df['trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()

#######################################
y = df["trip_duration"]
df.drop(["trip_duration"], axis=1, inplace=True)
df.drop(['id'], axis=1, inplace=True)
X = df

X.shape, y.shape

 ########        MODELS       ############

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, y_train.shape, X_test.shape, y_test.shape


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
print(gb.score(X_train, y_train), gb.score(X_test, y_test))
print(np.sqrt(MSE(y_test, gb.predict(X_test))))

test.columns
"""GBR_Output = gb.predict(test)
test_for_id = pd.read_csv(r"C:\Users\Dell\Desktop\New folder\NewYork Taxi TripDuration\nyctest\test.csv")
Id = test_for_id.id

output = pd.DataFrame({'Id': Id, 'trip_duration': GBR_Output})


output.to_csv('GBR.csv', index=False)
"""




