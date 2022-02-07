import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.svm as svm


# Load data
df = pd.read_csv('housepricedata.csv')
dataset = df.values

# Pre processing
min_max_scaler = preprocessing.MinMaxScaler()
X = dataset[:,0:10]
Y = dataset[:,10]

# Split between train-test
X_scale = min_max_scaler.fit_transform(X)
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale,
 Y, test_size=0.3)


# Classify above/below median score using 
# Polnomial Kernel
svm_classifier = svm.SVC(kernel='poly', degree= 3)
svm_classifier.fit(X_train, Y_train)
predicted = svm_classifier.predict(X_val_and_test)

# Accuracy of model
svm_classifier.score(X_val_and_test, Y_val_and_test)


