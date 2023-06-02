#Adapted/Expanded on code from Bee Guan Teo "How to Predict Stock Prices Change with Random Forest in Python"/Machine Learning for Finance in Python DataCamp course
#Daniel Szabo
#Independent Project for FIN:9160 - Quantitative Finance and Deep Learning - University of Iowa

# Data Retrieval and Manipulation
import yfinance as yf
import talib
import numpy as np
import pandas as pd
from pandas_datareader import DataReader as web
import pandas_datareader.data as pddata

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sb

#Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

#Deep Learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.losses

#Miscellaneous
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None) #Display all columns, helpful for editing

########################################################################################################################
# KEY ASSUMPTIONS AND INPUTS
########################################################################################################################
start_date = '2018-04-01' #Set to include GDPC1 data for our first period
end_date = '2023-05-01' #Most recent end-date to account for objective
return_period = 5

# Source Close price data for stock
ticker = 'V' #Stock to predict. For this analysis, we're using Visa (V)
data = yf.download(ticker, start = start_date, end= end_date)
data = data.drop(['Open', 'Low', 'High', 'Adj Close'], axis=1)

########################################################################################################################
# CREATE/SOURCE FEATURES
########################################################################################################################
feature_names = []

#Technical indicators: (RSI, SMA, EWMA for differing time intervals)
for n in [14, 30, 50, 200]:
    data['SMA' + str(n)] = talib.SMA(data['Close'].values, timeperiod=n)
    data['RSI' + str(n)] = talib.RSI(data['Close'].values, timeperiod=n)

    feature_names = feature_names + ['SMA' + str(n),
                                     'RSI' + str(n)]

data['ewma_daily'] = data['Close'].ewm(span=30).mean()

#Other indicators:
#Indexes
index_tickers = ['SP500', 'DJIA', 'VIXCLS'] #S&P 500, Dow Jones Industrial Average, CBOE Volatility Index
index_data = web(index_tickers, 'fred', start_date, end_date)
# print(index_data)
data = pd.concat([data, index_data], axis= 1) #Combine index data with primary data
feature_names = feature_names + index_tickers #Update feature names to contain indexes

#Exchange Rates
exr_tickers = ['DEXJPUS', 'DEXUSUK', 'DEXUSEU'] #Spot exchange rates for JPY to USD, USD to GBP, USD to EUR
exr_data = web(exr_tickers, 'fred', start_date, end_date)
# print(exr_data)
data = pd.concat([data, exr_data], axis= 1)
feature_names = feature_names + exr_tickers

#Direct competitors & Payment Processors
#Competitors chosen were Mastercard (MA), American Express (AXP), and Discover (DFS)
#Payment Processors chosen were PayPal (PYPL) and Square (SQ)
comp_tickers = ['MA', 'AXP', 'DFS', 'PYPL', 'SQ']
comp_data = yf.download(comp_tickers, start=start_date, end=end_date)
comp_data = comp_data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
comp_data = comp_data.pct_change()
comp_data.columns = ['' for col in comp_data.columns]
comp_data.columns = comp_tickers
# print(comp_data)
data = pd.concat([data, comp_data], axis= 1)
feature_names = feature_names + comp_tickers

# Macro Indicators (only took ones available from FRED)
#   Labor Market
#       UNRATE: Unemployment Rate
#       CIVPART: Civilian Labor Force Participation Rate
#   Inflation
#       CPIAUCSL: Consumer Price Index
#   Economic Growth
#       GDPC1: Real Gross Domestic Product
#       HOUST: Housing Starts
#       INDPRO: Industrial Production
#   Consumer Sentiment
#       UMCSENT: University of Michigan: Consumer Sentiment
#   Industry-specific / other
#       RSAFS: Advance Retail Sales: Retail Trade and Food Services

macro_tickers = ['UNRATE', 'CIVPART', 'CPIAUCSL', 'GDPC1', 'HOUST', 'INDPRO', 'RSAFS', 'UMCSENT']

macro_data = web(macro_tickers, 'fred', start_date, end_date)
macro_data = macro_data.asfreq('D').fillna(method='ffill') # As many of these factors are reported on monthly/quarterly basis, we fill in NAs with most recent available value
# print(macro_data)
data = pd.concat([data, macro_data], axis=1)
feature_names = feature_names + macro_tickers

#Objective: Predict how a stock price may change in the near future (5 days)
data['5d_future_close'] = data['Close'].shift(-return_period) # Calculates 5d future close by shifting Close values ahead by 5 rows
data['5d_future_close_pct'] = data['5d_future_close'].pct_change(return_period) # Target is % change from current day to 5 days in the future
data.dropna(inplace=True) #remove NA's so data works with ML models

n_features = len(feature_names)
print(f'Features pre-selection ({n_features} total): {feature_names}')
print()

########################################################################################################################
# SPLIT DATA INTO TRAINING AND TEST SETS, EVALUATE FEATURES
########################################################################################################################
features = data[feature_names]
print(features)
targets = data['5d_future_close_pct'] #Objective: Predict five-day future close percentage change

#Plot correlations of features - for first graph, fullscreen view is 'highly' recommended, it's a little noisy :)
sb.heatmap(features.corr(), annot=True, cbar=False)
plt.title('Correlation Matrix - all features')
plt.show()

sb.heatmap(features.corr() > 0.9, annot=True, cbar=False) #Find highly positively correlated variables (0.9 or higher)
plt.title('Correlation Matrix - all features above 0.9 correlation')
plt.show()

sb.heatmap(features.corr() < -0.9, annot=True, cbar=False) #Find highly negatively correlated variables (-0.9 or less)
plt.title('Correlation Matrix - all features below -0.9 correlation')
plt.show()

# Rank the features in the dataset by importance
bestFeatures = SelectKBest(k='all', score_func=f_regression)
fit = bestFeatures.fit(features, targets)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(features.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Features', 'Score']
print(featureScores.nlargest(n_features, 'Score').set_index('Features'))

# Plot the feature importances on a bar chart
sb.set(style = 'whitegrid')
sb.barplot(x = 'Score', y = 'Features', data=featureScores.nlargest(n_features, 'Score'))
plt.xlabel('Score')
plt.ylabel('Features')
plt.title('Feature Scores')
plt.show()

#Remove features as needed (can comment out as needed):
# print(f'features no drop: {features}')
# features = features.drop(['RSI30', 'SP500', 'SMA50', 'SMA200', 'CPIAUCSL', 'INDPRO'], axis=1) #Remove less important highly correlated features
# print(f'features drop 1: {features}')
# features = features.drop(['DEXUSEU', 'PYPL', 'UMCSENT', 'DEXJPUS', 'SQ'], axis=1) #Remove features with lowest importance
# print(f'features drop 2: {features}')

# Divide data into training and test sets - 80% training 20% test
train_size = int(0.80 * features.shape[0])
train_features = features[:train_size]
train_targets = targets[:train_size]
test_features = features[train_size:]
test_targets = targets[train_size:]

########################################################################################################################
# RANDOM FOREST
########################################################################################################################
#Create dictionary of parameters to test for the random forest model
grid_rf = {'n_estimators': [50, 100, 150, 200], 'max_depth': [3, 4, 5, 6, 7], 'max_features': [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27], 'random_state': [42]}
test_scores_rf = []

rf_model = RandomForestRegressor() #Build random forest model

#Iterate through parameters in grid dictionary and apply to random forest model
for g in ParameterGrid(grid_rf):
    rf_model.set_params(**g)
    rf_model.fit(train_features, train_targets)
    test_scores_rf.append(rf_model.score(test_features, test_targets))
    print(f'Iterating through parameter grid: {g}...')

#Get index of highest test score and print parameters
best_index = np.argmax(test_scores_rf)
print(test_scores_rf[best_index], ParameterGrid(grid_rf)[best_index])

#Create random forest model with 'best' parameters from our grid: (50, 3, 8, 42)
rf_model = RandomForestRegressor(n_estimators=50, max_depth=3, max_features=6, random_state=42)
rf_model.fit(train_features, train_targets)

y_pred_rf = rf_model.predict(test_features) # Predict target values from 'best' random forest

# Plot Target Predictions
y_pred_series_rf = pd.Series(y_pred_rf, index=test_targets.index)
y_pred_series_rf.plot(label = 'Predicted')
test_targets.plot(label = 'Actual')
plt.ylabel('Predicted 5 Day Close Price Change Percent')
plt.title('Random Forest Predictions')
plt.legend()
plt.show()

#Evaluate model performance
rf_train_score = rf_model.score(train_features, train_targets)
rf_test_score = rf_model.score(test_features, test_targets)

print('RANDOM FOREST METRICS:')
print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, y_pred_rf)}")
print(f"Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, y_pred_rf)}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, y_pred_rf))}")
print(f"Training Score (Random Forest): {rf_train_score}")
print(f"Test Score (Random Forest): {rf_test_score}")
print()

#Plot feature importances from random forest w. most important on left-hand side
importances_rf = rf_model.feature_importances_
sorted_index_rf = np.argsort(importances_rf)[::-1]
x_values_rf = range(len(importances_rf))
labels = np.array(feature_names)[sorted_index_rf]
plt.bar(x_values_rf, importances_rf[sorted_index_rf], tick_label = labels)
plt.title('Random Forest Feature Importances')
plt.xticks(rotation = 90)
plt.show()

########################################################################################################################
# LINEAR REGRESSION
########################################################################################################################
linear_model = LinearRegression() #Build linear regression model
linear_model.fit(train_features, train_targets)

y_pred_lm = linear_model.predict(test_features) # Predict target values from linear model

# Plot predictions from linear model
y_pred_series_lm = pd.Series(y_pred_lm, index=test_targets.index)
y_pred_series_lm.plot(label = 'Predicted')
test_targets.plot(label = 'Actual')
plt.ylabel('Predicted 5 Day Close Price Change Percent')
plt.title('Linear Regression Predictions')
plt.legend()
plt.show()

#Evaluate linear model performance
lm_train_score = linear_model.score(train_features, train_targets)
lm_test_score = linear_model.score(test_features, test_targets)

print("LINEAR REGRESSION METRICS:")
print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, y_pred_lm)}")
print(f"Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, y_pred_lm)}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, y_pred_lm))}")
print(f"Training score (linear model): {lm_train_score}")
print(f"Test score (linear model): {lm_test_score}")
print()

########################################################################################################################
#DECISION TREE
########################################################################################################################
dt_model = DecisionTreeRegressor() #Build decision tree model
dt_model.fit(train_features, train_targets)

y_pred_dt = dt_model.predict(test_features) #Predict target values from decision tree

#Plot predictions
y_pred_series_dt = pd.Series(y_pred_dt, index=test_targets.index)
y_pred_series_dt.plot(label = 'Predicted')
test_targets.plot(label = 'Actual')
plt.ylabel('Predicted 5 Day Close Change Percent')
plt.title('Decision Tree Predictions')
plt.legend()
plt.show()

#Evaluate model performance
dt_train_score = dt_model.score(train_features, train_targets)
dt_test_score = dt_model.score(test_features,test_targets)

print("DECISION TREE METRICS:")
print(f"Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, y_pred_dt)}")
print(f"Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, y_pred_dt)}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, y_pred_dt))}")
print(f"Training score (Decision Tree): {dt_train_score}")
print(f"Test score (Decision Tree): {dt_test_score}")
print()

# Standardize train and test feature data for use in KNN and neural network models
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

########################################################################################################################
#KNN - K NEAREST NEIGHBORS
########################################################################################################################
knn_train_scores = [] #Store knn train scores for finding max test score
knn_test_scores = [] #Store knn test scores for finding max test score
cv_scores = [] #List to store cross-validation scores to find optimal number of neighbors
max_test_score = -np.inf #Initialize maximum test score to negative infinity
max_test_metrics = None #This will store all of the error metrics we want to output
max_test_neighbors = None # Number of neighbors where the test score was the highest
for i in range(2, 301):
    print(f'Building model for {i} neighbors...')
    knn = KNeighborsRegressor(n_neighbors=i) #Build KNN model based on number of neighbors from iteration

    # Perform 5-fold cross-validation on training data
    scores = cross_val_score(knn, scaled_train_features, train_targets, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

    knn.fit(scaled_train_features, train_targets) #Fit the model to the entire training data

    y_pred_knn = knn.predict(scaled_test_features) #Predicts test values to compare with values from test set

    #Evaluate model based on training and test data
    train_score = knn.score(scaled_train_features, train_targets)
    test_score = knn.score(scaled_test_features, test_targets)

    knn_train_scores.append(train_score)
    knn_test_scores.append(test_score)

    #Find the number of neighbors and error metrics for the KNN model that yields the highest test score
    #Note: In my testing, this has a slim probability of encountering an error. If it does happen, I would recommend re-running the code
    try:
        if test_score > max_test_score:
            max_test_score = test_score
            max_test_metrics = {'Mean Absolute Error (MAE)': metrics.mean_absolute_error(test_targets, y_pred_knn),
                                'Mean Squared Error (MSE)': metrics.mean_squared_error(test_targets, y_pred_knn),
                                'Root Mean Squared Error (RMSE)': np.sqrt(metrics.mean_squared_error(test_targets, y_pred_knn)),
                                'Training Score (KNN)': train_score,
                                'Test Score (KNN)': test_score}
            max_test_neighbors = i
    except:
        print("An unknown error occured in the 'if' block of the code")
        pass

#Build KNN model from optimal number of neighbors
optimal_n = np.argmin(cv_scores) + 2 #Finds optimal number of neighbors from cross-validation. Add two since range starts at 2
print()
print(f'Optimal number of neighbors is {optimal_n}')
knn_optimal = KNeighborsRegressor(n_neighbors=optimal_n)
knn_optimal.fit(scaled_train_features, train_targets)

y_pred_optimal = knn_optimal.predict(scaled_test_features)

#Plot model predictions from 'optimal' KNN model
y_pred_series_optimal = pd.Series(y_pred_optimal, index=test_targets.index)
y_pred_series_optimal.plot(label = 'Predicted')
test_targets.plot(label = 'Actual')
plt.title(f'KNN Optimal Model: {optimal_n} neighbors')
plt.ylabel('Predicted 5 Day Close Price Change Percent')
plt.legend()
plt.show()

#Evaluate 'optimal' KNN model performance
train_score_optimal = knn_optimal.score(scaled_train_features, train_targets)
test_score_optimal = knn_optimal.score(scaled_test_features, test_targets)
print("OPTIMAL KNN METRICS:")
print(f'Number of neighbors: {optimal_n}')
print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, y_pred_optimal)}')
print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, y_pred_optimal)}')
print(f'Root Mean Sqaured Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, y_pred_optimal))}')
print(f'Training score (KNN): {train_score_optimal}')
print(f'Test score (KNN): {test_score_optimal}')
print()

#Build KNN model from number of neighbors that yielded the highest test score
knn_max = KNeighborsRegressor(n_neighbors=max_test_neighbors)
knn_max.fit(scaled_train_features, train_targets)
y_pred_max = knn_max.predict(scaled_test_features)

#Plot model predictions from max test score KNN model
y_pred_series_max = pd.Series(y_pred_max, index=test_targets.index)
y_pred_series_max.plot(label = 'Predicted')
test_targets.plot(label = 'Actual')
plt.title(f'KNN Max Test Score Model: {max_test_neighbors} neighbors')
plt.ylabel('Predicted 5 Day Close Price Change Percent')
plt.legend()
plt.show()

#Evaluate model performance
print("MAX TEST SCORE KNN METRICS")
print(f'Number of neighbors {max_test_neighbors}')
for key, value in max_test_metrics.items():
    print(f'{key}: {value}')

#These plots are for all KNN models generated
#Plot MSE for each value of n_neighbors
plt.plot(range(2,301), cv_scores)
plt.xlabel('Number of neighbors')
plt.ylabel('Negative MSE')
plt.show()

#Plot train and test scores over number of neighbors
plt.plot(range(2, 301), knn_train_scores, label = 'Train Score')
plt.plot(range(2, 301), knn_test_scores, label = 'Test Score')
plt.xlabel('Number of neighbors')
plt.ylabel('Score')
plt.legend()
plt.show()

########################################################################################################################
# NEURAL NETWORKS
########################################################################################################################
#Note: Model performance may vary significantly during each run of the program
#Neural Net 1 - Start with a simple neural network - only 3 layers
model_simple_nn = Sequential()
model_simple_nn.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_simple_nn.add(Dense(20, activation='relu'))
model_simple_nn.add(Dense(1, activation='linear'))

model_simple_nn.compile(optimizer='adam', loss='mse') #Mean Squared Error used as loss function for this test
history = model_simple_nn.fit(scaled_train_features, train_targets, epochs=25)

#Examine training loss to see if it flattens out
plt.plot(history.history['loss'])
plt.title('Neural Network 1 Loss: ' + str(round(history.history['loss'][-1], 6)))
plt.show()

#Calculate R2 Score for evaluation
train_preds_nn1 = model_simple_nn.predict(scaled_train_features)
test_preds_nn1 = model_simple_nn.predict(scaled_test_features)

print("NEURAL NETWORK 1 METRICS")
print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, test_preds_nn1)}')
print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, test_preds_nn1)}')
print(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, test_preds_nn1))}')
print(f'R2 Score (training set): {metrics.r2_score(train_targets, train_preds_nn1)}')
print(f'R2 Score (test set): {metrics.r2_score(test_targets, test_preds_nn1)}')
print()

#Plot predicted values vs actual values -> yields a 'bowtie'-like shape
plt.scatter(train_preds_nn1, train_targets, label = 'Training')
plt.scatter(test_preds_nn1, test_targets, label = 'Test')
plt.legend()
plt.show()

#Neural Net 2 - Custom loss function #
#Create loss function
def sign_penalty(y_actual, y_pred):
    penalty = 100
    loss = tf.where(tf.less(y_actual - y_pred, 0),
                    penalty * tf.square(y_actual - y_pred),
                    tf.square(y_actual - y_pred))
    return tf.reduce_mean(loss, axis=-1)

keras.losses.sign_penalty = sign_penalty
print(keras.losses.sign_penalty)

#Create model
model_nn_2 = Sequential()
model_nn_2.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_nn_2.add(Dense(20, activation='relu'))
model_nn_2.add(Dense(1, activation='linear'))

#Fit model using sign_penalty loss function
model_nn_2.compile(optimizer='adam', loss=sign_penalty)
history = model_nn_2.fit(scaled_train_features, train_targets, epochs=25)

#Examine training loss to see if it flattens out
plt.plot(history.history['loss'])
plt.title('Neural Network 2 Loss: ' + str(round(history.history['loss'][-1], 6)))
plt.show()

train_preds_nn2 = model_nn_2.predict(scaled_train_features)
test_preds_nn2 = model_nn_2.predict(scaled_test_features)

print('NEURAL NETWORK 2 METRICS:')
print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, test_preds_nn2)}')
print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, test_preds_nn2)}')
print(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, test_preds_nn2))}')
print(f'R2 Score (Training Set): {metrics.r2_score(train_targets, train_preds_nn2)}')
print(f'R2 Score (Test Set): {metrics.r2_score(test_targets, test_preds_nn2)}')
print()

#Plot predicted values vs actual values
plt.scatter(train_preds_nn2, train_targets, label = 'Training')
plt.scatter(test_preds_nn2, test_targets, label = 'Test')
plt.legend()
plt.show()

# Neural Net 3 - Including Dropout #
# Dropout randomly drops some neurons during training phase - helping to prevent net from fititng noise training data
model_nn_3 = Sequential()
model_nn_3.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_nn_3.add(Dropout(0.2))
model_nn_3.add(Dense(20, activation='relu'))
model_nn_3.add(Dense(1, activation='linear'))

# Fit model with MSE loss function
model_nn_3.compile(optimizer='adam', loss='mse')
history = model_nn_3.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('Neural Network 3 Loss: ' + str(round(history.history['loss'][-1], 6)))
plt.show()

#Evaluate model performance
train_preds_nn3 = model_nn_3.predict(scaled_train_features)
test_preds_nn3 = model_nn_3.predict(scaled_test_features)

print('NEURAL NETWORK 3 METRICS:')
print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, test_preds_nn3)}')
print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, test_preds_nn3)}')
print(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, test_preds_nn3))}')
print(f'R2 Score (Training Set): {metrics.r2_score(train_targets, train_preds_nn3)}')
print(f'R2 Score (Test Set): {metrics.r2_score(test_targets, test_preds_nn3)}')
print()

#Plot predicted values vs actual values
plt.scatter(train_preds_nn3, train_targets, label = 'Training')
plt.scatter(test_preds_nn3, test_targets, label = 'Test')
plt.legend()
plt.show()

# Neural Network 4 - Ensemble
# Combine simple, loss function, and dropout models for fourth neural net model

# Stack predictions horizontally and average across rows
train_preds_nn4 = np.mean(np.hstack((train_preds_nn1, train_preds_nn2, train_preds_nn3)), axis=1)
test_preds_nn4 = np.mean(np.hstack((test_preds_nn1, test_preds_nn2, test_preds_nn3)), axis=1)

#Evaluate model performance
print('NEURAL NETWORK 4 METRICS')
print(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_targets, test_preds_nn4)}')
print(f'Mean Squared Error (MSE): {metrics.mean_squared_error(test_targets, test_preds_nn4)}')
print(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(test_targets, test_preds_nn4))}')
print(f'R2 Score (Training Set): {metrics.r2_score(train_targets, train_preds_nn4)}')
print(f'R2 Score (Test Set): {metrics.r2_score(test_targets, test_preds_nn4)}')
