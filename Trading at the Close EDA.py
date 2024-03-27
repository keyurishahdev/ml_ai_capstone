#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis for Trading at Close
# 
# 
# #### - The d is available at kaggle for competitione https://www.kaggle.com/competitions/optiver-trading-at-the-close/dataam.
# 
# 
# # WhTrading at CloseenaStock exchanges are fast-paced, high-stakes environments where every second counts. The intensity escalates as the trading day approaches its end, peaking in the critical final ten minutes. These moments, often characterised by heightened volatility and rapid price fluctuations, play a pivotal role in shaping the global economic narrative for the day.
# 
# Each trading day on the Nasdaq Stock Exchange concludes with the Nasdaq Closing Cross auction. This process establishes the official closing prices for securities listed on the exchange. These closing prices serve as key indicators for investors, analysts and other market participants in evaluating the performance of individual securities and the market as a whole.
# 
# Within this complex financial landscape operates Optiver, a leading global electronic market maker. Fueled by technological innovation, Optiver trades a vast array of financial instruments, such as derivatives, cash equities, ETFs, bonds, and foreign currencies, offering competitive, two-sided prices for thousands of these instruments on major exchanges worldwide.
# 
# In the last ten minutes of the Nasdaq exchange trading session, market makers like Optiver merge traditional order book data with auction book data. This ability to consolidate information from both sources is critical for providing the best prices to all market participants.future health.

# # Data Definitions 
# 
# Files
# [train/test].csv The auction data. The test data will be delivered by the API.
# 
# stock_id - A unique identifier for the stock. Not all stock IDs exist in every time bucket.
# date_id - A unique identifier for the date. Date IDs are sequential & consistent across all stocks.
# imbalance_size - The amount unmatched at the current reference price (in USD).
# imbalance_buy_sell_flag - An indicator reflecting the direction of auction imbalance.
# buy-side imbalance; 1
# sell-side imbalance; -1
# no imbalance; 0
# reference_price - The price at which paired shares are maximized, the imbalance is minimized and the distance from the bid-ask midpoint is minimized, in that order. Can also be thought of as being equal to the near price bounded between the best bid and ask price.
# matched_size - The amount that can be matched at the current reference price (in USD).
# far_price - The crossing price that will maximize the number of shares matched based on auction interest only. This calculation excludes continuous market orders.
# near_price - The crossing price that will maximize the number of shares matched based auction and continuous market orders.
# [bid/ask]_price - Price of the most competitive buy/sell level in the non-auction book.
# [bid/ask]_size - The dollar notional amount on the most competitive buy/sell level in the non-auction book.
# wap - The weighted average price in the non-auction book.
# BidPrice∗AskSize+AskPrice∗BidSizeBidSize+AskSize
# seconds_in_bucket - The number of seconds elapsed since the beginning of the day's closing auction, always starting from 0.
# target - The 60 second future move in the wap of the stock, less the 60 second future move of the synthetic index. Only provided for the train set.
# The synthetic index is a custom weighted index of Nasdaq-listed stocks constructed by Optiver for this competition.
# The unit of the target is basis points, which is a common unit of measurement in financial markets. A 1 basis point price move is equivalent to a 0.01% price move.
# Where t is the time at the current observation, we can define the target:
# Target=(StockWAPt+60StockWAPt−IndexWAPt+60IndexWAPt)∗10000
# All size related columns are in USD terms.
# 
# All price related columns are converted to a price move relative to the stock wap (weighted average price) at the beginning of the auction period.
# 
# sample_submission A valid sample submission, delivered by the API. See this notebook for a very simple example of how to use the sample submission.
# 
# revealed_targets When the first time_id for each date (i.e. when seconds_in_bucket equals zero) the API will serve a dataframe providing the true target values for the entire previous date. All other rows contain null values for the columns of interest.
# 
# public_timeseries_testing_util.py An optional file intended to make it easier to run custom offline API tests. See the script's docstring for details. You will need to edit this file before using it.
# 
# example_test_files/ Data intended to illustrate how the API functions. Includes the same files and columns delivered by the API. The first three date ids are repeats of the last three date ids in the train set, to enable an illustration of how the API functions.
# 
# optiver2023/ Files that enable the API. Expect the API to deliver all rows in under five minutes and to reserve less than 0.5 GB of memory. The first three date ids delivered by the API are repeats of the last three date ids in the train set, to better illustrate how the API functions. You must make predictions for those dates in order to advance the API but those predictions are not scored.

# # Standard Package imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode
import seaborn as sns
import datetime as dt
from statsmodels.graphics.tsaplots import plot_acf
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import category_encoders as ce


# In[3]:


tac_raw = pd.read_csv('TAC_Train.csv', sep = ',')
tac_test = pd.read_csv('TAC_Test.csv', sep = ',')


# In[4]:


tac_raw.head()


# In[18]:


tac_raw.info()


# In[20]:


tac_raw.nunique()


# In[22]:


for col in (tac_raw.columns):
    print(tac_raw[col].value_counts(), '\n')
    print(f"""total: {tac_raw[col].count()}  Null: {tac_raw[col].isnull().sum()}""", '\n')


# In[24]:


#Imbalance Buy Sell Distribution

plt.figure(figsize=(10,5))
sns.histplot(x=tac_raw['imbalance_buy_sell_flag'],color='Lime',label='Imbalance Buy Sell')
plt.axvline(x=tac_raw['imbalance_buy_sell_flag'].mean(),color='k',linestyle ="--",label='Mean Age: {}'.format(round(tac_raw['imbalance_buy_sell_flag'].mean(),2)))
plt.legend()

plt.title('Distribution of Imbalance Buy Sell')
plt.show()


# In[26]:


COLS = list(tac_raw.drop(['stock_id', 'date_id', 'seconds_in_bucket', 'target', 'row_id', 'imbalance_buy_sell_flag', 'time_id'], axis=1).columns)
fig, axes = plt.subplots(10, 1, sharex=True, figsize=(35,60))
for col, ax in zip(COLS, axes.ravel()):
    sns.lineplot(data=tac_raw, x='time_id', y=tac_raw[col], hue='stock_id', ax=ax)
    ax.legend(loc='best')       


# In[30]:


plt.figure(figsize=(35, 10))
sns.histplot(data=tac_raw, x='target', kde=True) 

# Based on below histplot we can see target mostly converges around closing 


# In[44]:


analysis_df = list(tac_raw.drop(['stock_id', 'date_id', 'seconds_in_bucket', 'target', 'row_id', 'imbalance_buy_sell_flag', 'time_id', 'near_price', 'far_price'], axis=1).columns)
fig, axes = plt.subplots(10, 1, sharex=True, figsize=(35,60))
for col, ax in zip(analysis_df, axes.ravel()):
    stock = tac_raw[tac_raw['stock_id']==1]
    plot_acf(stock[col][:500000], lags=50, ax=ax)
    ax.set_title(str(col))

#Below plots gives us trend of various attributes 


# In[42]:


fig, ax = plt.subplots()
ax.scatter(tac_raw['bid_price'],tac_raw['bid_size'], marker='*',
          s=60, c=tac_raw['bid_price'], edgecolors='k',alpha=0.5)
ax.set_xlabel('bid_price')
ax.set_ylabel('bid_size');

## Based on below graph we can conclude that bid size increases mostly as demand when bid price is 1


# In[48]:


# Checking correlation matrix to see which parameters most affect target
corrmat= tac_raw.corr()
plt.figure(figsize=(15,15))  

cmap = sns.diverging_palette(250, 10, s=80, l=55, n=9, as_cmap=True)

sns.heatmap(corrmat,annot=True, cmap=cmap, center=0)

# Based on below we can say ['bid_price', 'ask_price', 'ask_size', 'wap'] has most greatest impact on target


# # Data cleaning 

# In[5]:


tac_clean = tac_raw.dropna()


# # Feature Extraction 

# In[34]:


from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

# Split the dataset into features (X) and target variable (y)
# select only columns that are important 
X = tac_raw[['stock_id', 'date_id','time_id','seconds_in_bucket','bid_price', 'ask_price', 'ask_size', 'wap']]
y = tac_raw['target']

# Drop rows with missing values
X.dropna(inplace=True)
y = y[X.index]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


X.head()


# In[35]:


display('check the shape of train sets', X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[29]:


# Create an imputer instance
imputer = SimpleImputer(strategy='mean')

# Fit and transform the data
X = imputer.fit_transform(X)
# Create the HistGradientBoostingRegressor model
rf_model = HistGradientBoostingRegressor()

# Fit the model to the data
rf_model.fit(X_train, y_train)

# Define the machine learning model you want to use (e.g., Random Forest)
model = RandomForestRegressor()

# Define the hyperparameters to tune and their respective ranges
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Perform cross-validation with hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Retrieve the best model and its corresponding hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the performance of the best model using cross-validation
#cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5)

# Print the best hyperparameters and the mean cross-validation score
print("Best Models: ", best_model)
print("Best Hyperparameters: ", best_params)
print("Mean Cross-Validation Score: ", cross_val_scores.mean())


# In[ ]:


# Model Selection: Since you are dealing with time series data, time series models such as ARIMA, SARIMA, or Prophet 
# can be suitable options. 
# Additionally, decision tree-based models like Random Forest and K-Nearest Neighbors (KNN) can also be considered. 
# Evaluate the performance of these models and choose the one that provides the best results for your specific problem.


# In[23]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
#from fbprophet import Prophet
warnings.filterwarnings("ignore")
# Time Series Model (ARIMA or Prophet)
arima_model = ARIMA(y_train, order=(1, 0, 1))
arima_model.fit()

# Decision Tree Model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

# Random Forest Model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# KNN Model
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# Evaluate the models (if applicable)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)

rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)

# Assess the performance of the models and select the best algorithm
model_scores = {'Decision Tree': dt_mse, 'Random Forest': rf_mse, 'KNN': knn_mse}
best_model = min(model_scores, key=model_scores.get)

# Deploy the recommended model (Random Forest for price forecasting)
if best_model == 'Random Forest':
    deployed_model = rf_model
elif best_model == 'Decision Tree':
    deployed_model = dt_model
elif best_model == 'KNN':
    deployed_model = knn_model
    
print("Mean Squared Error (MSE):")
print("Decision Tree:", dt_mse)
print("Random Forest:", rf_mse)
print("KNN:", knn_mse)
print("")

print("Best Model:", best_model)
print("Deployed Model:", deployed_model)


# In[ ]:


# We will also do a KNN Model 


# In[30]:


from sklearn.metrics import mean_squared_error

best_mse = float('inf')
best_n_neighbors = None

for n_neighbors in range(1, 11):
    knn_model = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, knn_predictions)
    if mse < best_mse:
        best_mse = mse
        best_n_neighbors = n_neighbors

print("Best n_neighbors:", best_n_neighbors)


# In[31]:


knn_model = KNeighborsRegressor(n_neighbors=best_n_neighbors)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
mse = mean_squared_error(y_test, knn_predictions)
print("Optimized KNN MSE:", mse)

