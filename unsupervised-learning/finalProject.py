import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 


# Chess Board files 
cb_files = {'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4,
 'e' : 5, 'f': 6, 'g': 7, 'h': 8}
# Dataset columns
dsColumns = ['wking_file', 'wking_rank',
     'wrook_file', 'wrook_rank','bking_file', 'bking_rank', 'result']

# Store the rsme values from different regressors
combined_rsm = [0., 0., 0., 0., 0.]
dfRegressorResults = pd.DataFrame({
    "Regressor": ['Polynomial', 'K-Neibhour', 'Decision Tree', 'Random Forest',
    'Support Vector'],
    "Avg. RSME": combined_rsm
})

# Function to compute area
def computeArea() -> pd.Series:
    ''' 
    Function returns a Series of area values
    '''
    # Check if pieces form a valid triangle
    isTriangle = (dfKRK['dist_w_rook_w_king']
     + dfKRK['dist_w_king_b_king']) > dfKRK['dist_b_king_w_rook']

    # Half perimeter
    halfPerimeter = (dfKRK['dist_w_rook_w_king'] +\
                        dfKRK['dist_w_king_b_king'] +\
                            dfKRK['dist_b_king_w_rook'])/2
    # Area Calculation                   
    return np.where(isTriangle.values == True, 
         np.sqrt(halfPerimeter * (
        halfPerimeter - dfKRK['dist_w_rook_w_king']) * (
        halfPerimeter - dfKRK['dist_w_king_b_king']) * (
        halfPerimeter - dfKRK['dist_b_king_w_rook'])
        ), -1.0)

def getRowIndex(row, moveCount) -> int:
    '''
        Return the index of the area matching closest to the given area
    '''
    dfKRK_moveCount = dfKRK[dfKRK['result'] == moveCount] 
    x = np.argmin(np.abs(dfKRK_moveCount['Area'] - row['Area']))
    return x

#%% Loading the data set

dfKRK = pd.read_csv(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data',
    names=dsColumns)

# As we will be using only 5 results (zero, one, two, three, four) out of the 
# complete dataset, hence, removing all other rows
dfKRK = dfKRK[dfKRK['result'].isin(['zero', 'one', 'two', 'three', 'four'])]

#%% Data Pre-processing for prediction
'''
The goal of this step is to prepare the dataset for predictions. Steps involved
here are -
1. Files will be converted to int values to form a XY coordinate system.
2. Ecludian Distances between (White King, White Rook), (White King, Black King)
    and (White Rook, Black King) will be computed and stored as columns.
3. Compute area of the triangles formed by the 3 pieces. If triangle cannot be 
    formed, the data is removed from the dataset
4. Add columns to be used for predictions
'''

# 1. Files converted to int's
dfKRK['wking_x'] = dfKRK['wking_file'].map(cb_files)
dfKRK['wrook_x'] = dfKRK['wrook_file'].map(cb_files)
dfKRK['bking_x'] = dfKRK['bking_file'].map(cb_files)

# 2. Computing Eculidian distances between pieces 
# White Rook and White King
dfKRK['dist_w_rook_w_king'] = np.sqrt( ((dfKRK['wking_x'] - dfKRK['wrook_x'])**2) +\
    ((dfKRK['wking_rank'] - dfKRK['wrook_rank'])**2))

# Black King and White King
dfKRK['dist_w_king_b_king'] = np.sqrt( ((dfKRK['bking_x'] - dfKRK['wking_x'])**2) +\
    ((dfKRK['bking_rank'] - dfKRK['wking_rank'])**2))

# Black King and White Rook
dfKRK['dist_b_king_w_rook'] = np.sqrt( ((dfKRK['bking_x'] - dfKRK['wrook_x'])**2) +\
    ((dfKRK['bking_rank'] - dfKRK['wrook_rank'])**2))

# 3. Compute area covered by the triangles formed by the 3 pieces
dfKRK['Area'] = computeArea()
dfKRK = dfKRK[dfKRK['Area'] != -1.0]
dfKRK = dfKRK.reset_index().drop(['index'], axis = 1)

# 4. Add column for True_Value, True_Position, Pred_Value, Pred_Positon
dfKRK['True_Value'] = ''
dfKRK['True_Position'] = ''


#%% Compute True Values for prediction (An intutive heuristic)
'''
Heuristic - For a given area which needs X moves to win (= A(X)), 
            find the area within the data, with wins in X -1 moves (= A(X-1)),
            such that, abs(A(X) - A(X-1)) is minimum.
'''

# Find min index
areaIdx = np.where(dfKRK.columns.values == 'Area')[0][0]
resultList = dfKRK['result'].unique().tolist()

for result in resultList:
    if result == 'zero' or result == 'one':
        continue
    else:
        tempDf = dfKRK[dfKRK['result'] == result]
        prevResult = resultList[resultList.index(result) - 1]
        idx_result = tempDf.apply(
            lambda row: getRowIndex(row, prevResult), axis = 1)
        dfKRK.loc[dfKRK['result'] == result, ['True_Value']] =\
            dfKRK.iloc[idx_result, areaIdx].values
        dfKRK.loc[dfKRK['result'] == result, ['True_Position']] = idx_result

# %% Scatter Plot and Pearson co-orelation
X = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['Area'].values.reshape(-1,1)
Y = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['True_Value'].values.reshape(-1,1)

fig, ax = plt.subplots(1, figsize = (7,5))

ax.scatter(X, Y)
plt.xlabel('Area (covered by pieces in current position)')
plt.ylabel('Area (covered by pieces after one move)')
plt.title('Coorelation between data points')
plt.tight_layout()
plt.show()

print('Pearson''s coorelation cooficient = {coff}'.format(
    coff = round(pearsonr(dfKRK[dfKRK['result'].isin(
        ['two', 'three', 'four'])]['Area'].values,
        dfKRK[dfKRK['result'].isin(
            ['two', 'three', 'four'])]['True_Value'].values)[0],4)
))

#%% Using Linear models for prediction
degreeList = [ i for i in range(3, 8)]
rmse = list()
r2 = list()

X = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['Area'].values.reshape(-1,1)
Y = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['True_Value'].values.reshape(-1,1)

# Find the degree which gives the min. error
for degree in degreeList:
    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.5)
    Y_train = Y_train[:, 0].astype('float64')
    # Compute weights     
    weights = np.polyfit(X_train[:, 0], Y_train, degree)
    model = np.poly1d(weights)
    # Predict
    predicted = model(X_test)
    # Compute Error
    rmse.append(np.sqrt(mean_squared_error(Y_test, predicted)))

# Plot the errors by degrees
fig, ax = plt.subplots(1, figsize = (7,5))

plt.plot(degreeList, rmse)
plt.xlabel('degree')
plt.ylabel('RMSE')
plt.title('Polynomial Regressor')
plt.tight_layout()
plt.show()

# Add mean rsme to the final list
combined_rsm[0] = round(sum(rmse)/len(rmse), 4)


# %% Using K-Neighbor Regressor to predict
kList = [ i for i in range(1, 10)]
rmse = list()

# Find the best K which gives the min. error
for k in kList:
    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.5)
    # Fit
    knn_Regressor= KNeighborsRegressor(n_neighbors = k)
    knn_Regressor.fit(X_train, Y_train)
    # Predict
    predicted = knn_Regressor.predict(X_test)
    # Compute error
    rmse.append(np.sqrt(mean_squared_error(Y_test, predicted)))

# Plot the errors by degrees
fig, ax = plt.subplots(1, figsize = (7,5))

plt.plot(kList, rmse)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('K- nearest neighbor')
plt.tight_layout()
plt.show()

# Add mean rsme to the final list
combined_rsm[1] = round(sum(rmse)/len(rmse), 4)

#%% Decision Tree Regressor

X = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['Area'].values.reshape(-1,1)
Y = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['True_Value'].values.reshape(-1,1)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.5)
# Fit
dt_Regressor= DecisionTreeRegressor(criterion='friedman_mse')
dt_Regressor.fit(X_train, Y_train)
# Predict
predicted = dt_Regressor.predict(X_test)
# Compute Error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))

# Add mean rsme to the final list
combined_rsm[2] = round(rmse, 4)

#%% Random Forest Regression

trees = [ i for i in range(1, 10)]
rmse = list()

X = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['Area'].values.reshape(-1,1)
Y = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['True_Value'].values.reshape(-1,1)

# Find the best number of estimators to give the min error
for treeCount in trees:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.5)
    rf_Regressor= RandomForestRegressor(n_estimators = treeCount)
    rf_Regressor.fit(X_train, Y_train)
    predicted = rf_Regressor.predict(X_test)
    rmse.append(np.sqrt(mean_squared_error(Y_test, predicted)))

fig, ax = plt.subplots(1, figsize = (7,5))

plt.plot(trees, rmse)
plt.xlabel('Number of trees')
plt.ylabel('RMSE')
plt.title('Random Forest Regression')
plt.tight_layout()
plt.show()

# Add mean rsme to the final list
combined_rsm[3] = round(sum(rmse)/len(rmse), 4)

# %% Support Vector Regression (SVR)
X = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['Area'].values.reshape(-1,1)
Y = dfKRK[dfKRK['result'].isin(['two', 'three', 'four'])]['True_Value'].values.reshape(-1,1)
rsme = []

# Setup regressors
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
svrs = [svr_rbf, svr_lin, svr_poly]

# Loop through the 3 types of regressors
for svr in svrs:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.5)
    predicted = svr.fit(X_train, Y_train).predict(X_test)
    rmse.append(np.sqrt(mean_squared_error(Y_test, predicted)))

combined_rsm[4] = round(sum(rmse)/len(rmse),4)


dfRegressorResults['Avg. RSME'] = combined_rsm
# %%
